import json
import os
import sqlite3
import subprocess

import pytest

from hytorch.harness import Result, Session
from hytorch.hermes_harness import HermesHarness
from hytorch.opencode_harness import OpenCodeHarness


def _node(tmp_path):
    root = tmp_path / "node"
    (root / "statespace").mkdir(parents=True)
    (root / "workspace").mkdir()
    return root


def _opencode_output(session_id="ses_one", text="done"):
    return "\n".join(
        (
            json.dumps({"type": "step_start", "sessionID": session_id}),
            json.dumps(
                {
                    "type": "text",
                    "sessionID": session_id,
                    "part": {"text": text},
                }
            ),
        )
    )


def test_opencode_start_creates_then_continues_native_profile(tmp_path):
    root = _node(tmp_path)
    calls = []

    def run(command, **kwargs):
        assert os.path.realpath(os.path.join(kwargs["cwd"], "statespace")) == str(
            (root / "statespace").resolve()
        )
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, _opencode_output(), "")

    harness = OpenCodeHarness(
        model="openai/gpt-5",
        environment={"OPENCODE_AUTH_CONTENT": '{"openai":{"type":"api","key":"x"}}'},
        runner=run,
    )
    result = harness.start(str(root), "solve it", None, max_tokens=321)
    continued = harness.start(str(root), "next epoch", None)

    command, kwargs = calls[0]
    stable = command[5]
    assert command == [
        "opencode",
        "run",
        "--format",
        "json",
        "--dir",
        stable,
        "--auto",
        "--model",
        "openai/gpt-5",
    ]
    assert kwargs["input"] == "solve it"
    assert kwargs["cwd"] == stable
    assert kwargs["env"]["HOME"] == os.path.join(stable, "workspace", "home")
    assert kwargs["env"]["XDG_DATA_HOME"] == os.path.join(stable, "workspace", "data")
    assert kwargs["env"]["OPENCODE_AUTH_CONTENT"].startswith("{")
    assert kwargs["env"]["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] == "321"
    assert result.text == "done"
    assert calls[1][0][calls[1][0].index("--session") + 1] == "ses_one"
    assert continued.session.id == "ses_one"
    assert result.session == Session("opencode", "ses_one", str(root / "workspace"))


def test_opencode_resume_uses_exact_session_and_returns_current_handle(tmp_path):
    root = _node(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, _opencode_output("ses_one", "learned"), ""
        )

    harness = OpenCodeHarness(runner=run)
    result = harness.resume(
        Session("opencode", "ses_one", str(root / "workspace")),
        str(root),
        "update yourself",
        "anthropic/claude-sonnet-4",
    )

    assert calls[0][-4:] == [
        "--session",
        "ses_one",
        "--model",
        "anthropic/claude-sonnet-4",
    ]
    assert result.text == "learned"
    assert result.session.id == "ses_one"


def test_opencode_close_preserves_profile(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    marker = profile / "memory"
    marker.write_text("keep")

    OpenCodeHarness().close(Session("opencode", "one", str(profile)))

    assert marker.read_text() == "keep"


def test_opencode_rejects_invalid_event_output(tmp_path):
    root = _node(tmp_path)
    harness = OpenCodeHarness(
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "not-json", ""
        )
    )

    with pytest.raises(RuntimeError, match="invalid JSON event"):
        harness.start(str(root), "task", None)


def test_opencode_rejects_credentials_in_persistent_profile(tmp_path):
    root = _node(tmp_path)
    auth = root / "workspace" / "data" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True)
    auth.write_text("{}")
    harness = OpenCodeHarness(
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, _opencode_output(), ""
        )
    )

    with pytest.raises(RuntimeError, match="credentials must not be stored"):
        harness.start(str(root), "task", None)

    assert auth.exists()


def test_opencode_removes_runtime_auth_artifact(tmp_path):
    root = _node(tmp_path)

    def run(command, **kwargs):
        auth = root / "workspace" / "data" / "opencode" / "auth.json"
        auth.parent.mkdir(parents=True, exist_ok=True)
        auth.write_text('{"secret":"runtime"}')
        return subprocess.CompletedProcess(command, 0, _opencode_output(), "")

    OpenCodeHarness(runner=run).start(str(root), "task", None)

    assert not (root / "workspace" / "data" / "opencode" / "auth.json").exists()


def test_hermes_starts_first_profile_session_and_parses_id(tmp_path):
    root = _node(tmp_path)
    calls = []

    def run(command, **kwargs):
        assert os.path.realpath(os.path.join(kwargs["cwd"], "statespace")) == str(
            (root / "statespace").resolve()
        )
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, "finished\n", "session_id: h_1\n"
        )

    harness = HermesHarness(
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        environment={"OPENROUTER_API_KEY": "secret"},
        runner=run,
    )
    result = harness.start(str(root), "solve it", None, max_tokens=654)

    command, kwargs = calls[0]
    assert command == [
        "hermes",
        "chat",
        "-Q",
        "-q",
        "solve it",
        "--yolo",
        "--no-restore-cwd",
        "--provider",
        "openrouter",
        "--model",
        "anthropic/claude-sonnet-4",
    ]
    stable = kwargs["cwd"]
    assert kwargs["env"]["HERMES_HOME"] == os.path.join(stable, "workspace")
    assert kwargs["env"]["HOME"] == os.path.join(stable, "workspace", "home")
    assert kwargs["env"]["TERMINAL_CWD"] == stable
    assert kwargs["env"]["HERMES_MAX_TOKENS"] == "654"
    assert result == Result(
        "finished", Session("hermes", "h_1", str(root / "workspace"))
    )


def test_hermes_start_continues_existing_native_session(tmp_path):
    root = _node(tmp_path)
    database = root / "workspace" / "state.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE sessions (source TEXT)")
    connection.execute("INSERT INTO sessions VALUES ('cli')")
    connection.commit()
    connection.close()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "continued", "session_id: h_2")

    HermesHarness(runner=run).start(str(root), "next epoch", None)

    assert "--continue" in calls[0]


def test_hermes_start_resumes_saved_compaction_tip_across_epochs(tmp_path):
    root = _node(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, "continued", "session_id: compacted-tip"
        )

    harness = HermesHarness(runner=run)
    harness.start(str(root), "epoch one", None)
    harness.start(str(root), "epoch two", None)

    assert "--resume" not in calls[0]
    assert calls[1][calls[1].index("--resume") + 1] == "compacted-tip"


def test_hermes_resume_tracks_compaction_tip(tmp_path):
    root = _node(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            "updated\n",
            "warning\nsession_id: old\nsession_id: compacted_tip\n",
        )

    result = HermesHarness(runner=run).resume(
        Session("hermes", "old", str(root / "workspace")),
        str(root),
        "learn",
        None,
    )

    assert calls[0][-2:] == ["--resume", "old"]
    assert result.session.id == "compacted_tip"


def test_hermes_credential_sidecar_never_remains_in_profile(tmp_path):
    root = _node(tmp_path)
    sidecar = tmp_path / "auth.json"
    sidecar.write_text('{"token":"old"}')
    observed = []

    def run(command, **kwargs):
        staged = root / "workspace" / "auth.json"
        observed.append((staged.exists(), oct(staged.stat().st_mode & 0o777)))
        staged.write_text('{"token":"refreshed"}')
        return subprocess.CompletedProcess(command, 0, "done", "session_id: h_2\n")

    harness = HermesHarness(credential_files={"auth.json": str(sidecar)}, runner=run)
    harness.start(str(root), "task", None)

    assert observed == [(True, "0o600")]
    assert sidecar.read_text() == '{"token":"refreshed"}'
    assert not (root / "workspace" / "auth.json").exists()


def test_hermes_rejects_credentials_in_persistent_profile(tmp_path):
    root = _node(tmp_path)
    (root / "workspace" / ".env").write_text("TOKEN=secret")
    harness = HermesHarness(
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "done", "session_id: h_3\n"
        )
    )

    with pytest.raises(RuntimeError, match="credentials must not be stored"):
        harness.start(str(root), "task", None)

    assert (root / "workspace" / ".env").exists()


def test_hermes_close_preserves_native_profile(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    database = profile / "state.db"
    database.write_bytes(b"native state")

    HermesHarness().close(Session("hermes", "one", str(profile)))

    assert database.read_bytes() == b"native state"


def test_harnesses_reject_foreign_sessions():
    with pytest.raises(ValueError, match="cannot use"):
        OpenCodeHarness().close(Session("hermes", "one", "/profile"))
    with pytest.raises(ValueError, match="cannot use"):
        HermesHarness().close(Session("opencode", "one", "/profile"))
