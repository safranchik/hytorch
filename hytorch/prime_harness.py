"""Prime Agent runtime with node-local persistent native state."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ._environment import command_environment
from ._native_view import native_node_view
from .harness import Result, Session, Usage

_DEFAULT_PROVIDER = "openai-codex"
_DEFAULT_MODEL = "gpt-5.6-terra"
_TRANSIENT_PROFILE_ENTRIES = {
    "auth.json",
    "daemon-update-restart.json",
    "daemon-update-restarts",
    "daemon-workers",
    "kernel-venv",
}
_auth_lock = threading.Lock()


class PrimeRuntime:
    """Run one persistent Prime Agent session through its JSON interface."""

    def __init__(
        self,
        provider: str | None = None,
        model: str = _DEFAULT_MODEL,
        binary: str = "prime-agent",
        *,
        harness_name: str = "prime-agent",
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.provider = provider or _DEFAULT_PROVIDER
        self._provider_explicit = bool(provider)
        self.model = model or _DEFAULT_MODEL
        self.binary = binary or "prime-agent"
        self.harness_name = harness_name
        self._runner = runner
        self._usage = Usage()
        self._usage_lock = threading.Lock()

    def usage(self) -> Usage:
        with self._usage_lock:
            return self._usage

    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        _, session_dir = _prime_state_paths(directory)
        session_file = _single_session_file(session_dir)
        return self._invoke(
            directory,
            prompt,
            mtype,
            session_dir=session_dir,
            session_file=session_file,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )

    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        if session.harness != self.harness_name:
            raise ValueError(
                "hytorch Prime Agent harness cannot resume "
                f"a {session.harness!r} session"
            )
        if not os.path.isfile(session.storage):
            raise RuntimeError(
                "hytorch Prime Agent harness: saved session "
                f"{session.id!r} is unavailable"
            )
        result = self._invoke(
            directory,
            prompt,
            mtype,
            session_dir=os.path.dirname(session.storage),
            session_file=session.storage,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )
        if result.session.id != session.id:
            raise RuntimeError(
                "hytorch Prime Agent harness: resumed session "
                f"{result.session.id!r}, expected {session.id!r}"
            )
        return result

    def close(self, session: Session) -> None:
        if session.harness != self.harness_name:
            raise ValueError(
                "hytorch Prime Agent harness cannot close "
                f"a {session.harness!r} session"
            )
        # Prime's headless worker has exited. Its session and artifacts remain
        # in the supplied workspace.

    def _invoke(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        session_dir: str,
        session_file: str | None,
        temperature: float | None,
        max_tokens: int | None,
        read_only: tuple[str, ...],
    ) -> Result:
        _validate_sampling(temperature, max_tokens)
        _validate_read_only(directory, read_only)
        profile_dir, _ = _prime_state_paths(directory)
        environment = command_environment()
        provider = self._provider_for(environment)

        with native_node_view(directory) as execution_root:
            with _staged_prime_profile(profile_dir) as staged_profile:
                profile_link = os.path.join(execution_root, "prime-profile")
                os.symlink(staged_profile, profile_link, target_is_directory=True)
                stable_session_dir = os.path.join(
                    execution_root, "workspace", ".prime", "sessions"
                )
                environment.update(
                    {
                        "PI_SKIP_VERSION_CHECK": "1",
                        "PRIME_AGENT_CODING_AGENT_DIR": profile_link,
                        "PRIME_AGENT_SESSION_DIR": stable_session_dir,
                        "HOME": os.path.join(
                            execution_root, "workspace", ".prime", "home"
                        ),
                    }
                )
                os.makedirs(
                    os.path.join(os.path.dirname(profile_dir), "home"), exist_ok=True
                )
                args = [
                    self.binary,
                    "--mode",
                    "json",
                    "--cwd",
                    execution_root,
                    "--session-dir",
                    stable_session_dir,
                    "--provider",
                    provider,
                    "--model",
                    mtype or self.model,
                ]
                if session_file is not None:
                    args.extend(
                        [
                            "--resume",
                            os.path.join(
                                stable_session_dir, os.path.basename(session_file)
                            ),
                        ]
                    )
                args.extend(["--", prompt])
                try:
                    try:
                        runner = self._runner or subprocess.run
                        completed = runner(
                            args,
                            cwd=execution_root,
                            capture_output=True,
                            text=True,
                            check=False,
                            env=environment,
                        )
                    except FileNotFoundError as exc:
                        raise RuntimeError(
                            "hytorch Prime Agent harness: executable "
                            f"{self.binary!r} is unavailable"
                        ) from exc
                finally:
                    if os.path.lexists(profile_link):
                        os.remove(profile_link)

        if completed.returncode != 0:
            raise RuntimeError(
                "hytorch Prime Agent harness: exit "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
        text, session_id, usage = _parse_json_events(completed.stdout)
        resolved_file = session_file or os.path.join(session_dir, f"{session_id}.jsonl")
        if not os.path.isfile(resolved_file):
            matches = [
                path
                for path in (
                    os.path.join(session_dir, name) for name in os.listdir(session_dir)
                )
                if path.endswith(".jsonl") and os.path.isfile(path)
            ]
            if len(matches) == 1:
                resolved_file = matches[0]
            else:
                raise RuntimeError(
                    "hytorch Prime Agent harness: persisted session file is unavailable"
                )
        with self._usage_lock:
            self._usage = Usage(
                input_tokens=self._usage.input_tokens + usage.input_tokens,
                output_tokens=self._usage.output_tokens + usage.output_tokens,
                cache_read_tokens=(
                    self._usage.cache_read_tokens + usage.cache_read_tokens
                ),
                cache_write_tokens=(
                    self._usage.cache_write_tokens + usage.cache_write_tokens
                ),
            )
        return Result(
            text=text,
            session=Session(
                harness=self.harness_name,
                id=session_id,
                storage=os.path.realpath(resolved_file),
            ),
        )

    def _provider_for(self, environment: dict[str, str]) -> str:
        if not self._provider_explicit and "OPENAI_API_KEY" in environment:
            return "openai"
        return self.provider


def _prime_state_paths(directory: str) -> tuple[str, str]:
    root = os.path.realpath(directory)
    workspace = os.path.realpath(os.path.join(root, "workspace"))
    if os.path.commonpath((root, workspace)) != root or not os.path.isdir(workspace):
        raise RuntimeError("hytorch Prime Agent harness: node workspace is unavailable")
    state = os.path.join(workspace, ".prime")
    profile = os.path.join(state, "agent")
    sessions = os.path.join(state, "sessions")
    os.makedirs(profile, exist_ok=True)
    os.makedirs(sessions, exist_ok=True)
    return profile, sessions


def _single_session_file(session_dir: str) -> str | None:
    sessions = sorted(
        os.path.join(session_dir, name)
        for name in os.listdir(session_dir)
        if name.endswith(".jsonl") and os.path.isfile(os.path.join(session_dir, name))
    )
    if len(sessions) > 1:
        raise RuntimeError(
            "hytorch Prime Agent harness: node state contains more than one root session"
        )
    return sessions[0] if sessions else None


@contextmanager
def _staged_prime_profile(persistent: str) -> Iterator[str]:
    persistent = os.path.realpath(persistent)
    os.makedirs(persistent, exist_ok=True)
    for name in _TRANSIENT_PROFILE_ENTRIES:
        path = os.path.join(persistent, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.remove(path)

    operator = os.path.realpath(
        os.path.expanduser(
            os.environ.get("PRIME_AGENT_CODING_AGENT_DIR", "~/.prime/agent")
        )
    )
    os.makedirs(operator, exist_ok=True)
    parent = tempfile.mkdtemp(prefix="hytorch-prime-agent-")
    staged = os.path.join(parent, "agent")
    shutil.copytree(persistent, staged)
    operator_auth = os.path.join(operator, "auth.json")
    with _auth_lock:
        if os.path.isfile(operator_auth):
            shutil.copy2(operator_auth, os.path.join(staged, "auth.json"))
    try:
        yield staged
    finally:
        try:
            with _auth_lock:
                _promote_newer_auth(os.path.join(staged, "auth.json"), operator)
            _replace_profile(staged, persistent)
        finally:
            shutil.rmtree(parent, ignore_errors=True)


def _replace_profile(candidate: str, destination: str) -> None:
    parent = os.path.dirname(destination)
    staged = tempfile.mkdtemp(prefix=".hytorch-prime-profile-", dir=parent)
    try:
        for name in os.listdir(candidate):
            if name in _TRANSIENT_PROFILE_ENTRIES:
                continue
            source = os.path.join(candidate, name)
            target = os.path.join(staged, name)
            if os.path.isdir(source) and not os.path.islink(source):
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target, follow_symlinks=False)
        shutil.rmtree(destination)
        os.replace(staged, destination)
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def _promote_newer_auth(candidate: str, destination_dir: str) -> None:
    if not os.path.isfile(candidate):
        return
    destination = os.path.join(destination_dir, "auth.json")
    try:
        candidate_data = json.loads(Path(candidate).read_text(encoding="utf-8"))
        current_data = (
            json.loads(Path(destination).read_text(encoding="utf-8"))
            if os.path.isfile(destination)
            else {}
        )
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    for provider, value in candidate_data.items():
        if not isinstance(value, dict):
            continue
        current = current_data.get(provider)
        candidate_expiry = value.get("expires", 0)
        current_expiry = current.get("expires", 0) if isinstance(current, dict) else 0
        if provider not in current_data or candidate_expiry > current_expiry:
            current_data[provider] = value
            changed = True
    if changed:
        os.makedirs(destination_dir, exist_ok=True)
        Path(destination).write_text(
            json.dumps(current_data, indent=2) + "\n", encoding="utf-8"
        )


def _parse_json_events(output: str) -> tuple[str, str, Usage]:
    session_id = ""
    final_text = ""
    usage = Usage()
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"hytorch Prime Agent harness: invalid JSON event: {line}"
            ) from exc
        if event.get("type") == "session" and isinstance(event.get("id"), str):
            session_id = event["id"]
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                final_text = text
        raw = message.get("usage", {})
        if isinstance(raw, dict):
            usage = Usage(
                input_tokens=usage.input_tokens + int(raw.get("input", 0)),
                output_tokens=usage.output_tokens + int(raw.get("output", 0)),
                cache_read_tokens=(
                    usage.cache_read_tokens + int(raw.get("cacheRead", 0))
                ),
                cache_write_tokens=(
                    usage.cache_write_tokens + int(raw.get("cacheWrite", 0))
                ),
            )
    if not session_id or not final_text:
        raise RuntimeError("hytorch Prime Agent harness: incomplete JSON event stream")
    return final_text.strip(), session_id, usage


def _validate_read_only(directory: str, read_only: tuple[str, ...]) -> None:
    root = os.path.realpath(directory)
    for path in read_only:
        if os.path.commonpath((root, os.path.realpath(path))) != root:
            raise ValueError(
                "hytorch Prime Agent harness: read-only path escaped the node root"
            )


def _validate_sampling(temperature: float | None, max_tokens: int | None) -> None:
    if temperature is not None and (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise ValueError(
            "hytorch Prime Agent harness: temperature must be non-negative or None"
        )
    if max_tokens is not None and (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError(
            "hytorch Prime Agent harness: max_tokens must be positive or None"
        )


__all__ = ["PrimeRuntime"]
