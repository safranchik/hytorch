"""Local Codex CLI harness with node-local, persistent native state."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path

from ._environment import command_environment
from ._native_view import native_node_view
from .harness import Harness, Result, Session, Usage

_DEFAULT_MODEL = "gpt-5.6-terra"


class CodexHarness(Harness):
    """Run one persistent Codex CLI session from each node workspace.

    ``workspace/.hytorch/codex`` is the node's opaque ``CODEX_HOME``. HyTorch can
    version the complete workspace while authentication remains an external
    runtime input.
    """

    name = "codex"

    def __init__(
        self,
        name: str | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        binary: str = "codex",
        environment: Mapping[str, str] | None = None,
        auth_file: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.binary = binary
        self.environment = dict(environment or {})
        self.auth_file = auth_file
        self._runner = runner
        self._usage = Usage()
        self._usage_lock = threading.Lock()

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
        """Start a node session, or continue the session in its workspace state."""
        self._validate_sampling(temperature, max_tokens)
        home, workspace = self._paths(directory)
        with native_node_view(directory) as project:
            command = self._command(Path(project), mtype)
            if _has_codex_session(home):
                command.extend(["resume", "--last", "--all", "-"])
            else:
                command.append("-")
            return self._invoke(command, home, workspace, Path(project), prompt)

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
        """Append one turn to the node's current Codex session."""
        self._validate_session(session)
        self._validate_sampling(temperature, max_tokens)
        home, workspace = self._paths(directory)
        with native_node_view(directory) as project:
            command = self._command(Path(project), mtype)
            command.extend(["resume", session.id, "-"])
            return self._invoke(command, home, workspace, Path(project), prompt)

    def close(self, session: Session) -> None:
        """Release a completed invocation without deleting native state."""
        self._validate_session(session)

    def usage(self) -> Usage:
        """Return aggregate token use for this harness instance."""
        with self._usage_lock:
            return self._usage

    def _paths(self, directory: str) -> tuple[Path, Path]:
        workspace = Path(directory).resolve() / "workspace"
        if not workspace.is_dir():
            raise RuntimeError("hytorch Codex harness: node workspace is unavailable")
        # Keep user-level Codex state separate from project-level .codex files.
        home = workspace / ".hytorch" / "codex"
        home.mkdir(parents=True, exist_ok=True)
        return home, workspace

    def _command(self, workspace: Path, mtype: str | None) -> list[str]:
        return [
            self.binary,
            "exec",
            "--json",
            "--model",
            mtype or self.model,
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
            "--cd",
            str(workspace),
        ]

    def _invoke(
        self,
        command: list[str],
        home: Path,
        workspace: Path,
        project: Path,
        prompt: str,
    ) -> Result:
        environment = command_environment(values=self.environment)
        stable_workspace = project / "workspace"
        environment["CODEX_HOME"] = str(stable_workspace / ".hytorch" / "codex")
        tool_home = workspace / ".hytorch" / "home"
        tool_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(stable_workspace / ".hytorch" / "home")
        with _credential_overlay(home / "auth.json", self.auth_file):
            try:
                completed = self._runner(
                    command,
                    cwd=project,
                    env=environment,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"hytorch Codex harness: executable {self.binary!r} is unavailable"
                ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"hytorch Codex harness: CLI exited with {completed.returncode}: {detail}"
            )
        text, session_id, usage = _parse_codex_output(completed.stdout)
        self._add_usage(usage)
        return Result(
            text=text,
            session=Session(self.name, session_id, str(home)),
        )

    def _add_usage(self, usage: Usage) -> None:
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

    def _validate_session(self, session: Session) -> None:
        if session.harness != self.name:
            raise ValueError(
                f"hytorch Codex harness cannot resume a {session.harness!r} session"
            )

    @staticmethod
    def _validate_sampling(temperature: float | None, max_tokens: int | None) -> None:
        # Codex CLI has no stable flags for these generic optimizer controls.
        del temperature, max_tokens


def _has_codex_session(home: Path) -> bool:
    sessions = home / "sessions"
    return sessions.is_dir() and any(sessions.rglob("*.jsonl"))


def _parse_codex_output(output: str) -> tuple[str, str, Usage]:
    session_id = ""
    text = ""
    usage = Usage()
    try:
        events = [json.loads(line) for line in output.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise RuntimeError("hytorch Codex harness: invalid JSONL output") from exc
    for event in events:
        if event.get("type") == "thread.started":
            session_id = str(event.get("thread_id", ""))
        item = event.get("item", {})
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "agent_message"
        ):
            text = str(item.get("text", "")).strip()
        if event.get("type") == "turn.completed":
            raw = event.get("usage", {})
            usage = Usage(
                input_tokens=int(raw.get("input_tokens", 0)),
                output_tokens=int(raw.get("output_tokens", 0)),
                cache_read_tokens=int(raw.get("cached_input_tokens", 0)),
            )
    if not session_id or not text:
        raise RuntimeError("hytorch Codex harness: incomplete CLI result")
    return text, session_id, usage


@contextmanager
def _credential_overlay(target: Path, source: str | None):
    if target.exists():
        raise RuntimeError(
            "hytorch Codex harness: credentials must not be stored in agent state"
        )
    if source is not None:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise RuntimeError("hytorch Codex harness: auth file is unavailable")
        shutil.copy2(source_path, target)
    try:
        yield
    finally:
        # Also remove a credential that the CLI created from runtime auth.
        target.unlink(missing_ok=True)


__all__ = ["CodexHarness"]
