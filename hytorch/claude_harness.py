"""Local Claude Code harness with node-local, persistent native state."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path

from ._environment import command_environment
from ._native_view import native_node_view
from .harness import Harness, Result, Session, Usage

_DEFAULT_MODEL = "sonnet"
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class ClaudeCodeHarness(Harness):
    """Run one persistent Claude Code session from each node workspace."""

    name = "claude-code"

    def __init__(
        self,
        name: str | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        binary: str = "claude",
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
        command = self._command(mtype)
        prior = _latest_claude_session(home)
        if prior is not None:
            command.extend(["--resume", prior])
        with native_node_view(directory) as project:
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
        """Append one turn to the node's current Claude Code session."""
        self._validate_session(session)
        self._validate_sampling(temperature, max_tokens)
        home, workspace = self._paths(directory)
        command = self._command(mtype)
        command.extend(["--resume", session.id])
        with native_node_view(directory) as project:
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
            raise RuntimeError(
                "hytorch Claude Code harness: node workspace is unavailable"
            )
        # Keep user-level Claude state separate from project-level .claude files.
        home = workspace / ".hytorch" / "claude"
        home.mkdir(parents=True, exist_ok=True)
        return home, workspace

    def _command(self, mtype: str | None) -> list[str]:
        return [
            self.binary,
            "--print",
            "--output-format",
            "json",
            "--model",
            mtype or self.model,
            "--permission-mode",
            "bypassPermissions",
            "--dangerously-skip-permissions",
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
        environment["CLAUDE_CONFIG_DIR"] = str(stable_workspace / ".hytorch" / "claude")
        tool_home = workspace / ".hytorch" / "home"
        tool_home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(stable_workspace / ".hytorch" / "home")
        with _credential_overlay(home / ".credentials.json", self.auth_file):
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
                    "hytorch Claude Code harness: executable "
                    f"{self.binary!r} is unavailable"
                ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "hytorch Claude Code harness: CLI exited with "
                f"{completed.returncode}: {detail}"
            )
        text, session_id, usage = _parse_claude_output(completed.stdout)
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
                "hytorch Claude Code harness cannot resume a "
                f"{session.harness!r} session"
            )

    @staticmethod
    def _validate_sampling(temperature: float | None, max_tokens: int | None) -> None:
        # Claude Code has no exact per-turn equivalents for these controls.
        del temperature, max_tokens


def _latest_claude_session(home: Path) -> str | None:
    projects = home / "projects"
    if not projects.is_dir():
        return None
    candidates = [
        path
        for path in projects.rglob("*.jsonl")
        if _UUID.fullmatch(path.stem) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
    ).stem


def _parse_claude_output(output: str) -> tuple[str, str, Usage]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("hytorch Claude Code harness: invalid JSON output") from exc
    text = str(payload.get("result", "")).strip()
    session_id = str(payload.get("session_id", ""))
    raw = payload.get("usage", {})
    usage = Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0)),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens", 0)),
    )
    if payload.get("is_error") or not text or not session_id:
        raise RuntimeError("hytorch Claude Code harness: incomplete CLI result")
    return text, session_id, usage


@contextmanager
def _credential_overlay(target: Path, source: str | None):
    if target.exists():
        raise RuntimeError(
            "hytorch Claude Code harness: credentials must not be stored in agent state"
        )
    if source is not None:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise RuntimeError("hytorch Claude Code harness: auth file is unavailable")
        shutil.copy2(source_path, target)
    try:
        yield
    finally:
        # Also remove a credential that the CLI created from runtime auth.
        target.unlink(missing_ok=True)


__all__ = ["ClaudeCodeHarness"]
