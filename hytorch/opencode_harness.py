"""Local OpenCode CLI harness with an isolated native agent profile."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Mapping
from typing import Any

from ._environment import command_environment
from ._native_view import native_node_view
from .harness import Harness, Result, Session

Runner = Callable[..., subprocess.CompletedProcess[str]]


class OpenCodeHarness(Harness):
    """Run one persistent OpenCode session from a node's private workspace.

    OpenCode stores its native session, instructions, skills, configuration,
    and other local state below XDG directories. This harness places all of
    those directories below ``workspace/``. Provider credentials stay in the
    process environment and are not part of that profile.
    """

    name = "opencode"

    def __init__(
        self,
        name: str | None = None,
        *,
        model: str | None = None,
        binary: str = "opencode",
        environment: Mapping[str, str] | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.binary = binary
        self._environment = dict(environment or {})
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
        """Execute another turn in an exact OpenCode session."""
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
        del temperature  # OpenCode has no stable per-turn temperature flag.
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
        environment = self._profile_environment(
            profile, os.path.join(execution_root, "workspace"), max_tokens
        )
        session_id = session_id or _saved_session(profile)
        model = mtype or self.model
        command = [
            self.binary,
            "run",
            "--format",
            "json",
            "--dir",
            execution_root,
            "--auto",
        ]
        if session_id is not None:
            command.extend(("--session", session_id))
        if model:
            command.extend(("--model", model))

        auth_file = os.path.join(profile, "data", "opencode", "auth.json")
        if os.path.lexists(auth_file):
            raise RuntimeError(
                "hytorch OpenCode harness: credentials must not be stored "
                "in agent state"
            )
        try:
            try:
                completed = self._runner(
                    command,
                    cwd=execution_root,
                    env=environment,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "hytorch OpenCode harness: executable "
                    f"{self.binary!r} is unavailable"
                ) from exc
        finally:
            # Also remove credentials created from runtime authentication.
            if os.path.lexists(auth_file):
                os.remove(auth_file)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"hytorch OpenCode harness: exit {completed.returncode}: {detail}"
            )
        text, resolved_id = _parse_opencode_output(completed.stdout)
        _save_session(profile, resolved_id)
        return Result(
            text=text,
            session=Session(
                harness=self.name,
                id=resolved_id,
                storage=profile,
            ),
        )

    def _profile_environment(
        self, profile: str, stable_profile: str, max_tokens: int | None
    ) -> dict[str, str]:
        actual_locations = {
            "HOME": os.path.join(profile, "home"),
            "XDG_DATA_HOME": os.path.join(profile, "data"),
            "XDG_CONFIG_HOME": os.path.join(profile, "config"),
            "XDG_STATE_HOME": os.path.join(profile, "state"),
            "XDG_CACHE_HOME": os.path.join(profile, "cache"),
        }
        for path in actual_locations.values():
            os.makedirs(path, exist_ok=True)
        locations = {
            name: os.path.join(stable_profile, os.path.basename(path))
            for name, path in actual_locations.items()
        }
        environment = command_environment(values=self._environment)
        environment.update(locations)
        environment.update(
            {
                "OPENCODE_AUTO_SHARE": "false",
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_CLAUDE_CODE": "true",
            }
        )
        if max_tokens is not None:
            environment["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max_tokens)
        return environment

    def _validate_session(self, session: Session) -> None:
        if session.harness != self.name:
            raise ValueError(
                f"hytorch OpenCode harness cannot use a {session.harness!r} session"
            )
        if not session.id.strip():
            raise ValueError("hytorch OpenCode harness: session id is empty")


def _node_paths(directory: str) -> tuple[str, str, str]:
    root = os.path.realpath(directory)
    statespace = os.path.join(root, "statespace")
    profile = os.path.join(root, "workspace")
    if not os.path.isdir(statespace):
        raise RuntimeError("hytorch harness: node statespace directory is unavailable")
    if not os.path.isdir(profile):
        raise RuntimeError("hytorch harness: node workspace directory is unavailable")
    return root, statespace, profile


def _parse_opencode_output(output: str) -> tuple[str, str]:
    session_id = ""
    final_text = ""
    for number, source in enumerate(output.splitlines(), 1):
        if not source.strip():
            continue
        try:
            event: Any = json.loads(source)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"hytorch OpenCode harness: invalid JSON event on line {number}"
            ) from exc
        if not isinstance(event, dict):
            continue
        current_id = event.get("sessionID")
        if isinstance(current_id, str) and current_id:
            session_id = current_id
        if event.get("type") != "text":
            continue
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            final_text = part["text"].strip()
    if not session_id:
        raise RuntimeError("hytorch OpenCode harness: output has no session id")
    if not final_text:
        raise RuntimeError("hytorch OpenCode harness: output has no final text")
    return final_text, session_id


def _session_pointer(profile: str) -> str:
    return os.path.join(profile, "state", "hytorch", "opencode-session")


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


__all__ = ["OpenCodeHarness"]
