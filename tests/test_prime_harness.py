import json
import subprocess
from pathlib import Path

import pytest

import hytorch
import hytorch.prime_harness as prime_module
from hytorch.prime_harness import PrimeRuntime, _staged_prime_profile


def _json_run(session_id: str, text: str) -> str:
    return "\n".join(
        [
            json.dumps({"type": "session", "id": session_id}),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                        "usage": {
                            "input": 11,
                            "output": 7,
                            "cacheRead": 3,
                            "cacheWrite": 2,
                        },
                    },
                }
            ),
        ]
    )


def test_prime_start_persists_and_continues_complete_native_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(prime_module, "command_environment", lambda: {})
    root = tmp_path / "node"
    (root / "workspace").mkdir(parents=True)
    (root / "statespace").mkdir()
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs["env"]))
        sessions = root / "workspace" / ".prime" / "sessions"
        session_file = sessions / "prime-session.jsonl"
        session_file.write_text('{"type":"session"}\n', encoding="utf-8")
        artifacts = root / "workspace" / ".prime" / "session-artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "kernel-state.dill").write_text("kernel", encoding="utf-8")
        profile = Path(kwargs["env"]["PRIME_AGENT_CODING_AGENT_DIR"])
        (profile / "memory.md").write_text("native memory", encoding="utf-8")
        return subprocess.CompletedProcess(
            args, 0, _json_run("prime-session", "finished"), ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    harness = PrimeRuntime(binary="prime-agent-test")

    first = harness.start(str(root), "forward one", None)
    second = harness.start(str(root), "forward two", None)

    first_args, first_env = calls[0]
    second_args, _ = calls[1]
    assert first_args[:3] == ["prime-agent-test", "--mode", "json"]
    assert "--resume" not in first_args
    resumed = second_args[second_args.index("--resume") + 1]
    assert Path(resumed).name == Path(first.session.storage).name
    assert (
        first_args[first_args.index("--cwd") + 1]
        == second_args[second_args.index("--cwd") + 1]
    )
    assert first_env["PRIME_AGENT_SESSION_DIR"].endswith("/workspace/.prime/sessions")
    assert second.session.id == first.session.id
    assert (root / "workspace" / ".prime" / "agent" / "memory.md").read_text(
        encoding="utf-8"
    ) == "native memory"
    assert (
        root / "workspace" / ".prime" / "session-artifacts" / "kernel-state.dill"
    ).is_file()
    assert harness.usage() == hytorch.harness.Usage(11 * 2, 7 * 2, 3 * 2, 2 * 2)


def test_prime_resume_returns_result_and_close_keeps_native_state(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(prime_module, "command_environment", lambda: {})
    root = tmp_path / "node"
    (root / "statespace").mkdir(parents=True)
    session_dir = root / "workspace" / ".prime" / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "same.jsonl"
    session_file.write_text("session\n", encoding="utf-8")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, _json_run("same", '{"feedback": []}'), ""
        ),
    )
    harness = PrimeRuntime()
    session = hytorch.harness.Session("prime-agent", "same", str(session_file))

    result = harness.resume(session, str(root), "backward", None)
    harness.close(result.session)

    assert isinstance(result, hytorch.harness.Result)
    assert result.text == '{"feedback": []}'
    assert session_file.is_file()


def test_prime_profile_excludes_auth_and_process_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    operator = tmp_path / "home" / ".prime" / "agent"
    operator.mkdir(parents=True)
    (operator / "auth.json").write_text(
        '{"openai-codex":{"expires":1,"access":"secret"}}', encoding="utf-8"
    )
    persistent = tmp_path / "candidate" / "agent"
    (persistent / "daemon-workers").mkdir(parents=True)
    (persistent / "daemon-workers" / "worker.json").write_text(
        "secret token", encoding="utf-8"
    )
    (persistent / "auth.json").write_text("leaked", encoding="utf-8")

    with _staged_prime_profile(str(persistent)) as staged:
        assert Path(staged, "auth.json").is_file()
        Path(staged, "harness").mkdir()
        Path(staged, "harness", "harness_state.json").write_text(
            '{"memory":"lesson"}', encoding="utf-8"
        )
        Path(staged, "auth.json").write_text(
            '{"openai-codex":{"expires":2,"access":"refreshed"}}',
            encoding="utf-8",
        )

    assert not (persistent / "auth.json").exists()
    assert not (persistent / "daemon-workers").exists()
    assert (persistent / "harness" / "harness_state.json").is_file()
    assert "refreshed" in (operator / "auth.json").read_text(encoding="utf-8")


def test_prime_rejects_wrong_session_and_invalid_json(tmp_path, monkeypatch):
    root = tmp_path / "node"
    (root / "statespace").mkdir(parents=True)
    session_dir = root / "workspace" / ".prime" / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "same.jsonl"
    session_file.write_text("session\n", encoding="utf-8")
    harness = PrimeRuntime()

    with pytest.raises(ValueError, match="cannot resume"):
        harness.resume(
            hytorch.harness.Session("pi", "same", str(session_file)),
            str(root),
            "prompt",
            None,
        )

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(prime_module, "command_environment", lambda: {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "not-json", ""),
    )
    with pytest.raises(RuntimeError, match="invalid JSON event"):
        harness.start(str(root), "prompt", None)


def test_prime_provider_selection_uses_api_key_only_when_implicit():
    assert PrimeRuntime()._provider_for({"OPENAI_API_KEY": "secret"}) == "openai"
    assert (
        PrimeRuntime(provider="openai-codex")._provider_for(
            {"OPENAI_API_KEY": "secret"}
        )
        == "openai-codex"
    )
