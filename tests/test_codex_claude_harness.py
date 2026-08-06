import json
import os
import subprocess
from pathlib import Path

import pytest

from hytorch.claude_harness import ClaudeCodeHarness
from hytorch.codex_harness import CodexHarness
from hytorch.harness import Result, Session, Usage


def _node(tmp_path: Path) -> Path:
    root = tmp_path / "node"
    (root / "workspace").mkdir(parents=True)
    (root / "statespace").mkdir()
    return root


def _codex_output(session_id: str, text: str = "done") -> str:
    events = [
        {"type": "thread.started", "thread_id": session_id},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": text},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 4,
                "output_tokens": 3,
            },
        },
    ]
    return "\n".join(json.dumps(event) for event in events)


def _claude_output(session_id: str, text: str = "done") -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": False,
            "result": text,
            "session_id": session_id,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 5,
                "cache_read_input_tokens": 6,
                "cache_creation_input_tokens": 2,
            },
        }
    )


def test_codex_start_uses_isolated_native_home_and_machine_output(tmp_path):
    root = _node(tmp_path)
    calls = []

    def runner(args, **kwargs):
        assert os.path.realpath(Path(kwargs["cwd"]) / "statespace") == str(
            (root / "statespace").resolve()
        )
        assert os.path.realpath(Path(kwargs["cwd"]) / "workspace") == str(
            (root / "workspace").resolve()
        )
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, _codex_output("codex-one"), "")

    harness = CodexHarness(runner=runner, environment={"CODEX_API_KEY": "secret"})
    result = harness.start(str(root), "work", "gpt-test")

    assert isinstance(result, Result)
    assert result.text == "done"
    assert result.session == Session(
        "codex", "codex-one", str(root / "workspace" / ".hytorch" / "codex")
    )
    args, kwargs = calls[0]
    stable = Path(args[9])
    assert args == [
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-test",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "--cd",
        str(stable),
        "-",
    ]
    assert kwargs["cwd"] == stable
    assert kwargs["input"] == "work"
    assert kwargs["env"]["CODEX_HOME"] == str(
        stable / "workspace" / ".hytorch" / "codex"
    )
    assert harness.usage() == Usage(10, 3, 4, 0)


def test_codex_start_continues_native_candidate_session(tmp_path):
    root = _node(tmp_path)
    session_file = root / "workspace" / ".hytorch" / "codex" / "sessions" / "run.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.write_text("native session\n")
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, _codex_output("codex-old"), "")

    harness = CodexHarness(runner=runner)
    result = harness.start(str(root), "next epoch", None)

    assert result.session.id == "codex-old"
    assert calls[0][-4:] == ["resume", "--last", "--all", "-"]


def test_codex_resume_returns_result_and_close_preserves_state(tmp_path):
    root = _node(tmp_path)
    marker = root / "workspace" / ".hytorch" / "codex" / "memory.db"
    marker.parent.mkdir(parents=True)
    marker.write_text("memory")
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, _codex_output("codex-one"), "")

    harness = CodexHarness(runner=runner)
    session = Session("codex", "codex-one", str(marker.parent))
    result = harness.resume(session, str(root), "feedback", None)
    harness.close(result.session)

    assert isinstance(result, Result)
    assert calls[0][-3:] == ["resume", "codex-one", "-"]
    assert marker.read_text() == "memory"


def test_codex_auth_overlay_is_not_persistent(tmp_path):
    root = _node(tmp_path)
    source = tmp_path / "auth.json"
    source.write_text('{"token":"secret"}')
    observed = []

    def runner(args, **kwargs):
        target = root / "workspace" / ".hytorch" / "codex" / "auth.json"
        observed.append(target.read_text())
        return subprocess.CompletedProcess(args, 0, _codex_output("codex-one"), "")

    harness = CodexHarness(auth_file=str(source), runner=runner)
    harness.start(str(root), "work", None)

    assert observed == ['{"token":"secret"}']
    assert not (root / "workspace" / ".hytorch" / "codex" / "auth.json").exists()


def test_claude_start_uses_isolated_native_home_and_machine_output(tmp_path):
    root = _node(tmp_path)
    calls = []

    def runner(args, **kwargs):
        assert os.path.realpath(Path(kwargs["cwd"]) / "statespace") == str(
            (root / "statespace").resolve()
        )
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args, 0, _claude_output("00000000-0000-4000-8000-000000000001"), ""
        )

    harness = ClaudeCodeHarness(
        runner=runner, environment={"ANTHROPIC_API_KEY": "secret"}
    )
    result = harness.start(str(root), "work", "claude-test")

    assert isinstance(result, Result)
    assert result.text == "done"
    assert result.session == Session(
        "claude-code",
        "00000000-0000-4000-8000-000000000001",
        str(root / "workspace" / ".hytorch" / "claude"),
    )
    args, kwargs = calls[0]
    assert args == [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        "claude-test",
        "--permission-mode",
        "bypassPermissions",
        "--dangerously-skip-permissions",
    ]
    stable = Path(kwargs["cwd"])
    assert stable.parent.name == "hytorch-native"
    assert kwargs["input"] == "work"
    assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(
        stable / "workspace" / ".hytorch" / "claude"
    )
    assert harness.usage() == Usage(11, 5, 6, 2)


def test_claude_start_breaks_equal_session_mtimes_deterministically(tmp_path):
    root = _node(tmp_path)
    old_id = "00000000-0000-4000-8000-000000000001"
    new_id = "00000000-0000-4000-8000-000000000002"
    project = root / "workspace" / ".hytorch" / "claude" / "projects" / "node"
    project.mkdir(parents=True)
    old = project / f"{old_id}.jsonl"
    new = project / f"{new_id}.jsonl"
    old.write_text("old")
    new.write_text("new")
    old.touch()
    new.touch()
    timestamp = max(old.stat().st_mtime_ns, new.stat().st_mtime_ns)
    os.utime(old, ns=(timestamp, timestamp))
    os.utime(new, ns=(timestamp, timestamp))
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, _claude_output(new_id), "")

    harness = ClaudeCodeHarness(runner=runner)
    result = harness.start(str(root), "next epoch", None)

    assert result.session.id == new_id
    assert calls[0][-2:] == ["--resume", new_id]


def test_claude_resume_and_auth_overlay_preserve_native_state(tmp_path):
    root = _node(tmp_path)
    session_id = "00000000-0000-4000-8000-000000000001"
    source = tmp_path / "credentials.json"
    source.write_text('{"oauth":"secret"}')
    marker = root / "workspace" / ".hytorch" / "claude" / "memory.md"
    marker.parent.mkdir(parents=True)
    marker.write_text("memory")
    observed = []

    def runner(args, **kwargs):
        target = marker.parent / ".credentials.json"
        observed.append((args, target.read_text()))
        return subprocess.CompletedProcess(args, 0, _claude_output(session_id), "")

    harness = ClaudeCodeHarness(auth_file=str(source), runner=runner)
    result = harness.resume(
        Session("claude-code", session_id, str(marker.parent)),
        str(root),
        "feedback",
        None,
    )
    harness.close(result.session)

    assert isinstance(result, Result)
    assert observed[0][0][-2:] == ["--resume", session_id]
    assert observed[0][1] == '{"oauth":"secret"}'
    assert not (marker.parent / ".credentials.json").exists()
    assert marker.read_text() == "memory"


@pytest.mark.parametrize(
    ("harness", "output"),
    [
        (CodexHarness(), _codex_output("codex-one")),
        (
            ClaudeCodeHarness(),
            _claude_output("00000000-0000-4000-8000-000000000001"),
        ),
    ],
)
def test_local_cli_harnesses_tolerate_generic_sampling(harness, output, tmp_path):
    root = _node(tmp_path)
    harness._runner = lambda args, **kwargs: subprocess.CompletedProcess(
        args, 0, output, ""
    )

    result = harness.start(str(root), "work", None, temperature=0.4, max_tokens=512)

    assert result.text == "done"
