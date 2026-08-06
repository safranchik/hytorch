"""Agent-container environment loading outside model and Git state."""

from __future__ import annotations

import ast
import contextlib
import os
import re
import subprocess
import tempfile
import threading
import warnings
from collections.abc import Iterator, Mapping

PROJECT_ENV = ".hytorch.env"
GLOBAL_ENV = os.path.join("hytorch", "secrets.env")
EXPLICIT_ENV = "HYTORCH_ENV_FILE"
KNOWN_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "NOUS_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
)
RUNTIME_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_warned_tracked: set[str] = set()
_warning_lock = threading.Lock()


def project_root(start: str | None = None) -> str:
    """Find the nearest Git or Python project root from the working directory."""
    current = os.path.realpath(start or os.getcwd())
    while True:
        if os.path.exists(os.path.join(current, ".git")) or os.path.isfile(
            os.path.join(current, "pyproject.toml")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.realpath(start or os.getcwd())
        current = parent


def environment_files(start: str | None = None) -> tuple[str, ...]:
    """Return existing environment files in increasing precedence order."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    global_file = os.path.join(config_home, GLOBAL_ENV)
    project_file = os.path.join(project_root(start), PROJECT_ENV)
    result = [path for path in (global_file, project_file) if os.path.isfile(path)]

    explicit = os.environ.get(EXPLICIT_ENV)
    if explicit:
        explicit_path = os.path.realpath(os.path.expanduser(explicit))
        if not os.path.isfile(explicit_path):
            raise RuntimeError(
                f"hytorch: {EXPLICIT_ENV} does not name a file: {explicit_path}"
            )
        result = [path for path in result if os.path.realpath(path) != explicit_path]
        result.append(explicit_path)
    return tuple(result)


def agent_environment(start: str | None = None) -> dict[str, str]:
    """Merge declared agent values, then apply exported-value overrides."""
    values: dict[str, str] = {}
    root = project_root(start)
    project_file = os.path.join(root, PROJECT_ENV)
    for path in environment_files(start):
        values.update(_read_env(path))
        _warn_if_tracked(path, root, project_file)
    for name in tuple(values) + KNOWN_PROVIDER_KEYS:
        if name in os.environ:
            values[name] = os.environ[name]
    return values


def command_environment(
    start: str | None = None, *, values: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build a minimal process environment plus declared agent values."""
    environment = {
        name: os.environ[name] for name in RUNTIME_ENV_KEYS if name in os.environ
    }
    environment.update(agent_environment(start))
    if values is not None:
        environment.update(values)
    return environment


@contextlib.contextmanager
def docker_environment_file(
    start: str | None = None, *, values: Mapping[str, str] | None = None
) -> Iterator[str | None]:
    """Yield a temporary mode-0600 Docker env file and remove it afterward."""
    values = dict(values) if values is not None else agent_environment(start)
    if not values:
        yield None
        return
    descriptor, path = tempfile.mkstemp(prefix="hytorch-env-", suffix=".list")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            for name in sorted(values):
                value = values[name]
                if "\n" in value or "\r" in value or "\x00" in value:
                    raise ValueError(
                        f"hytorch: environment value {name!r} cannot contain a newline or NUL"
                    )
                file.write(f"{name}={value}\n")
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _read_env(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as file:
        for number, source in enumerate(file, 1):
            line = source.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, raw = line.partition("=")
            name = name.strip()
            if not separator or not _NAME.fullmatch(name):
                raise ValueError(f"hytorch: invalid {path}:{number} environment entry")
            values[name] = _parse_value(raw.strip(), path, number)
    return values


def _parse_value(raw: str, path: str, number: int) -> str:
    if not raw:
        return ""
    if raw[0] in {"'", '"'}:
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"hytorch: invalid quoted value at {path}:{number}"
            ) from exc
        if not isinstance(value, str):
            raise ValueError(f"hytorch: invalid quoted value at {path}:{number}")
        return value
    return raw.split(" #", 1)[0].rstrip()


def _warn_if_tracked(path: str, root: str, project_file: str) -> None:
    if os.path.realpath(path) != os.path.realpath(project_file):
        return
    resolved = os.path.realpath(path)
    with _warning_lock:
        if resolved in _warned_tracked:
            return
        tracked = (
            subprocess.run(
                ["git", "-C", root, "ls-files", "--error-unmatch", "--", PROJECT_ENV],
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        )
        _warned_tracked.add(resolved)
    if tracked:
        warnings.warn(
            f"hytorch: {PROJECT_ENV} is tracked by Git; add it to .gitignore",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = [
    "EXPLICIT_ENV",
    "GLOBAL_ENV",
    "KNOWN_PROVIDER_KEYS",
    "PROJECT_ENV",
    "agent_environment",
    "command_environment",
    "docker_environment_file",
    "environment_files",
    "project_root",
]
