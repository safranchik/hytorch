import os
import stat
import subprocess

import pytest

from hytorch._environment import agent_environment, docker_environment_file


def _project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    return root


def test_agent_environment_uses_dedicated_files_and_precedence(tmp_path, monkeypatch):
    root = _project(tmp_path)
    nested = root / "src"
    nested.mkdir()
    config = tmp_path / "config"
    global_dir = config / "hytorch"
    global_dir.mkdir(parents=True)
    (global_dir / "secrets.env").write_text("GLOBAL_ONLY=global\nSHARED=global\n")
    (root / ".env").write_text("UNRELATED=do-not-forward\n")
    (root / ".hytorch.env").write_text(
        "PROJECT_ONLY=project\nSHARED=project\nEXPORTED=from-file\n"
    )
    explicit = tmp_path / "team.env"
    explicit.write_text("EXPLICIT_ONLY=explicit\nSHARED=explicit\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("HYTORCH_ENV_FILE", str(explicit))
    monkeypatch.setenv("EXPORTED", "from-shell")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("UNRELATED_SHELL", "do-not-forward")

    values = agent_environment(str(nested))

    assert values == {
        "EXPLICIT_ONLY": "explicit",
        "EXPORTED": "from-shell",
        "GLOBAL_ONLY": "global",
        "OPENAI_API_KEY": "provider-key",
        "PROJECT_ONLY": "project",
        "SHARED": "explicit",
    }
    assert "UNRELATED" not in values
    assert "UNRELATED_SHELL" not in values


def test_docker_environment_file_is_private_and_temporary(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (root / ".hytorch.env").write_text(
        'PLAIN=value\nQUOTED="value with spaces"\nCOMMENTED=value # note\n'
    )

    with docker_environment_file() as path:
        assert path is not None
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        contents = open(path, encoding="utf-8").read()
        assert contents == ("COMMENTED=value\nPLAIN=value\nQUOTED=value with spaces\n")
    assert not os.path.exists(path)


def test_missing_explicit_environment_file_fails(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setenv("HYTORCH_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(RuntimeError, match="HYTORCH_ENV_FILE does not name a file"):
        agent_environment(str(root))


def test_invalid_environment_entry_fails(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.delenv("HYTORCH_ENV_FILE", raising=False)
    (root / ".hytorch.env").write_text("NOT VALID\n")
    with pytest.raises(ValueError, match="invalid .* environment entry"):
        agent_environment(str(root))


def test_tracked_project_environment_warns(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.delenv("HYTORCH_ENV_FILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (root / ".hytorch.env").write_text("TOKEN=secret\n")
    subprocess.run(["git", "init", "-b", "main", root], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", root, "add", "-f", ".hytorch.env"],
        check=True,
        capture_output=True,
    )

    with pytest.warns(RuntimeWarning, match="tracked by Git"):
        assert agent_environment(str(root))["TOKEN"] == "secret"
