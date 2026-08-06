"""Local Hermes Agent CLI harness with an isolated native profile."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
from collections.abc import Callable, Mapping

from ._environment import command_environment
from ._native_view import native_node_view
from .harness import Harness, Result, Session

Runner = Callable[..., subprocess.CompletedProcess[str]]
_SESSION_ID = re.compile(
    r"^[ \t]*session_id:[ \t]*(\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
_SECRET_FILES = (".env", "auth.json", ".anthropic_oauth.json")


class HermesHarness(Harness):
    """Run one persistent Hermes session from a node's private workspace.

    ``HERMES_HOME`` is the complete opaque agent profile. Optional credential
    files are copied into that home only while Hermes runs. They are copied
    back to their external sidecars after token refresh and then removed from
    the profile before HyTorch can promote it.
    """

    name = "hermes"

    def __init__(
        self,
        name: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        binary: str = "hermes",
        environment: Mapping[str, str] | None = None,
        credential_files: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        super().__init__(name)
        self.provider = provider
        self.model = model
        self.binary = binary
        self._environment = dict(environment or {})
        self._credential_files = _validate_credential_files(credential_files or {})
        self._runner = runner

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
        """Continue the profile's latest session, or create its first one."""
        return self._invoke(
            directory,
            prompt,
            mtype,
            session_id=None,
            temperature=temperature,
            max_tokens=max_tokens,
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
        """Resume Hermes and return its current compression-lineage tip."""
        self._validate_session(session)
        return self._invoke(
            directory,
            prompt,
            mtype,
            session_id=session.id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def close(self, session: Session) -> None:
        """Detach from a session without deleting its persistent profile."""
        self._validate_session(session)

    def _invoke(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        session_id: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Result:
        del temperature  # Hermes has no stable per-turn temperature flag.
        _, _, profile = _node_paths(directory)
        with native_node_view(directory) as execution_root:
            return self._invoke_at(
                execution_root, profile, prompt, mtype, session_id, max_tokens
            )

    def _invoke_at(
        self,
        execution_root: str,
        profile: str,
        prompt: str,
        mtype: str | None,
        session_id: str | None,
        max_tokens: int | None,
    ) -> Result:
        stable_profile = os.path.join(execution_root, "workspace")
        environment = self._profile_environment(
            execution_root, profile, stable_profile, max_tokens
        )
        session_id = session_id or _saved_session(profile)
        command = [
            self.binary,
            "chat",
            "-Q",
            "-q",
            prompt,
            "--yolo",
            "--no-restore-cwd",
        ]
        if session_id is None and _has_hermes_session(profile):
            command.append("--continue")
        elif session_id is not None:
            command.extend(("--resume", session_id))
        if self.provider:
            command.extend(("--provider", self.provider))
        model = mtype or self.model
        if model:
            command.extend(("--model", model))

        self._stage_credentials(profile)
        try:
            try:
                completed = self._runner(
                    command,
                    cwd=execution_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"hytorch Hermes harness: executable {self.binary!r} is unavailable"
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"hytorch Hermes harness: exit {completed.returncode}: {detail}"
                )
            resolved_id = _parse_hermes_session_id(completed.stderr)
            _save_session(profile, resolved_id)
            text = completed.stdout.strip()
            if not text:
                raise RuntimeError("hytorch Hermes harness: output has no final text")
            return Result(
                text=text,
                session=Session(
                    harness=self.name,
                    id=resolved_id,
                    storage=profile,
                ),
            )
        finally:
            self._unstage_credentials(profile)

    def _profile_environment(
        self,
        execution_root: str,
        profile: str,
        stable_profile: str,
        max_tokens: int | None,
    ) -> dict[str, str]:
        home = os.path.join(profile, "home")
        os.makedirs(home, exist_ok=True)
        environment = command_environment(values=self._environment)
        environment.update(
            {
                "HOME": os.path.join(stable_profile, "home"),
                "HERMES_HOME": stable_profile,
                "TERMINAL_CWD": execution_root,
            }
        )
        if max_tokens is not None:
            environment["HERMES_MAX_TOKENS"] = str(max_tokens)
        return environment

    def _stage_credentials(self, profile: str) -> None:
        for name in _SECRET_FILES:
            destination = os.path.join(profile, name)
            if not os.path.lexists(destination):
                continue
            if name in self._credential_files:
                message = "credential overlay target already exists"
            else:
                message = "credentials must not be stored in agent state"
            raise RuntimeError(f"hytorch Hermes harness: {message}")
        for name in _SECRET_FILES:
            destination = os.path.join(profile, name)
            source = self._credential_files.get(name)
            if source is None:
                continue
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)

    def _unstage_credentials(self, profile: str) -> None:
        for name in _SECRET_FILES:
            candidate = os.path.join(profile, name)
            sidecar = self._credential_files.get(name)
            if sidecar is not None and os.path.isfile(candidate):
                temporary = sidecar + ".hytorch.tmp"
                shutil.copy2(candidate, temporary)
                os.chmod(temporary, 0o600)
                os.replace(temporary, sidecar)
            if os.path.lexists(candidate):
                os.remove(candidate)

    def _validate_session(self, session: Session) -> None:
        if session.harness != self.name:
            raise ValueError(
                f"hytorch Hermes harness cannot use a {session.harness!r} session"
            )
        if not session.id.strip():
            raise ValueError("hytorch Hermes harness: session id is empty")


def _validate_credential_files(values: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in values.items():
        if name not in _SECRET_FILES:
            raise ValueError(
                f"hytorch Hermes harness: unsupported credential file {name!r}"
            )
        resolved = os.path.realpath(os.path.expanduser(path))
        if not os.path.isfile(resolved):
            raise ValueError(
                f"hytorch Hermes harness: credential file is unavailable: {resolved}"
            )
        result[name] = resolved
    return result


def _node_paths(directory: str) -> tuple[str, str, str]:
    root = os.path.realpath(directory)
    statespace = os.path.join(root, "statespace")
    profile = os.path.join(root, "workspace")
    if not os.path.isdir(statespace):
        raise RuntimeError("hytorch harness: node statespace directory is unavailable")
    if not os.path.isdir(profile):
        raise RuntimeError("hytorch harness: node workspace directory is unavailable")
    return root, statespace, profile


def _has_hermes_session(profile: str) -> bool:
    database = os.path.join(profile, "state.db")
    if not os.path.isfile(database):
        return False
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE source = 'cli' LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise RuntimeError(
            "hytorch Hermes harness: native session database is unreadable"
        ) from exc
    return row is not None


def _parse_hermes_session_id(stderr: str) -> str:
    matches = _SESSION_ID.findall(stderr)
    if not matches:
        raise RuntimeError("hytorch Hermes harness: output has no session id")
    return matches[-1]


def _session_pointer(profile: str) -> str:
    return os.path.join(profile, ".hytorch", "hermes-session")


def _saved_session(profile: str) -> str | None:
    path = _session_pointer(profile)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as file:
        value = file.read().strip()
    return value or None


def _save_session(profile: str, session_id: str) -> None:
    path = _session_pointer(profile)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(session_id + "\n")


__all__ = ["HermesHarness"]
