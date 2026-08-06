"""Integration test that spawns a real Pi (@earendil-works/pi-coding-agent)
session for one node execution, via the private Pi runtime script. Gated
behind HYTORCH_PI_TEST=1 so `pytest` never depends on network/npm/Pi
credential availability:

    HYTORCH_PI_TEST=1 python3 -m pytest tests/test_pi.py -v -s

Requires either a Pi "ChatGPT Plus/Pro (Codex)" login (`pi` then `/login`,
provider "openai-codex") or an OPENAI_API_KEY in the environment (set
HYTORCH_PI_PROVIDER=openai in that case).
"""

import os
import subprocess
from pathlib import Path

import pytest

import hytorch
from hytorch.pi_harness import PiRuntime, _staged_agent_directory


def test_pi_harness_defaults_to_terra():
    harness = PiRuntime()
    assert harness.provider == "openai-codex"
    assert harness.model == "gpt-5.6-terra"
    assert harness.docker is True
    assert harness._provider_for({"OPENAI_API_KEY": "secret"}) == "openai"


def test_explicit_codex_provider_ignores_api_key_for_provider_selection():
    harness = PiRuntime(provider="openai-codex")
    assert harness._provider_for({"OPENAI_API_KEY": "secret"}) == "openai-codex"


def test_named_pi_harness_can_place_a_complete_model():
    harness = hytorch.harness.PiHarness("remote-pi")
    assert harness.name == "remote-pi"
    assert harness._runtime.harness_name == "remote-pi"


def test_pi_harness_builds_isolated_docker_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    harness = PiRuntime()
    harness._docker_image = "hytorch-pi-runtime:test"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("test")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    learned_workspace = workspace / "workspace"
    learned_workspace.mkdir()
    environment = tmp_path / "agent.env"
    environment.write_text("OPENAI_API_KEY=secret\n")

    args = harness._docker_run_args(
        str(workspace),
        str(prompt),
        "gpt-5.6-terra",
        session_dir=str(session_dir),
        temperature=0.4,
        max_tokens=512,
        read_only=(str(learned_workspace),),
        environment_path=str(environment),
    )

    assert args[:4] == ["docker", "run", "--rm", "--init"]
    assert "hytorch-pi-runtime:test" in args
    assert "/workspace" in " ".join(args)
    assert any("dst=/workspace/workspace,readonly" in arg for arg in args)
    assert args[args.index("--env-file") + 1] == str(environment)
    assert args[-8:] == [
        "--model",
        "gpt-5.6-terra",
        "--session-dir",
        "/run/hytorch/session",
        "--temperature",
        "0.4",
        "--max-tokens",
        "512",
    ]


def test_pi_uses_standard_docker_context_and_external_image(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-gpu")
    monkeypatch.setenv("HYTORCH_PI_IMAGE", "registry.example.com/team/pi:latest")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "sha256:resolved\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    harness = PiRuntime()
    harness._ensure_docker_image(str(tmp_path), "ignored")

    assert harness._docker_image == "sha256:resolved"
    assert calls[0][0] == [
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        "registry.example.com/team/pi:latest",
    ]
    assert "--context" not in calls[0][0]
    assert calls[0][1].get("env") is None
    assert os.environ["DOCKER_CONTEXT"] == "remote-gpu"


def test_missing_external_pi_image_does_not_build(monkeypatch, tmp_path):
    monkeypatch.setenv("HYTORCH_PI_IMAGE", "missing:image")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "missing"),
    )

    with pytest.raises(RuntimeError, match="active Docker context"):
        PiRuntime()._ensure_docker_image(str(tmp_path), "ignored")


def test_remote_context_uses_volume_transport(monkeypatch, tmp_path):
    harness = PiRuntime()
    harness._docker_image = "sha256:runtime"
    root = tmp_path / "node"
    statespace = root / "statespace"
    workspace = root / "workspace"
    session = tmp_path / "session"
    for path in (statespace, workspace, session):
        path.mkdir(parents=True)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")
    environment = tmp_path / "agent.env"
    environment.write_text("OPENAI_API_KEY=secret\n")
    uploads = []
    downloads = []
    calls = []
    monkeypatch.setattr(
        harness,
        "_upload_volume",
        lambda source, volume: uploads.append((source, volume)),
    )
    monkeypatch.setattr(
        harness,
        "_download_volume",
        lambda volume, destination: downloads.append((volume, destination)),
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:4] == ["docker", "run", "--rm", "--init"]:
            return subprocess.CompletedProcess(
                args,
                0,
                '{"text":"done","session_id":"one","session_file":"one.jsonl"}\n',
                "",
            )
        return subprocess.CompletedProcess(args, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = harness._run_remote_docker(
        str(root),
        str(prompt),
        "gpt-5.6-terra",
        session_dir=str(session),
        session_file=None,
        temperature=None,
        max_tokens=None,
        read_only=(str(statespace),),
        environment_path=str(environment),
        provider="openai",
        agent_dir=str(tmp_path / "agent"),
    )

    run = next(
        args for args in calls if args[:4] == ["docker", "run", "--rm", "--init"]
    )
    assert result.returncode == 0
    assert len(uploads) == 5
    assert not any(destination == str(statespace) for _, destination in downloads)
    assert any(destination == str(workspace) for _, destination in downloads)
    assert any(destination == str(session) for _, destination in downloads)
    assert any(destination == str(tmp_path / "agent") for _, destination in downloads)
    assert any("dst=/workspace/statespace,readonly" in arg for arg in run)
    assert any("dst=/workspace/workspace" in arg for arg in run)
    assert run[run.index("--env-file") + 1] == str(environment)
    assert len([args for args in calls if args[1:3] == ["volume", "rm"]]) == 5


def test_docker_context_detection_recognizes_remote_endpoint(monkeypatch):
    harness = PiRuntime()
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-gpu")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "ssh://gpu.example.com\n", ""
        ),
    )

    assert harness._uses_remote_docker()


def test_pi_start_continues_node_session_across_epochs(tmp_path, monkeypatch):
    root = tmp_path / "node"
    (root / "workspace").mkdir(parents=True)
    calls = []
    harness = PiRuntime(docker=False)

    def fake_invoke(directory, prompt, mtype, **kwargs):
        calls.append(kwargs["session_file"])
        session_file = kwargs["session_file"]
        if session_file is None:
            session_file = os.path.join(kwargs["session_dir"], "native.jsonl")
            with open(session_file, "w", encoding="utf-8") as file:
                file.write('{"type":"session","id":"native"}\n')
        return prompt, "native", session_file, hytorch.harness.Usage()

    monkeypatch.setattr(harness, "_invoke", fake_invoke)

    first = harness.start(str(root), "epoch one", None)
    second = harness.start(str(root), "epoch two", None)

    assert calls == [None, first.session.storage]
    assert second.session.id == first.session.id
    assert second.session.storage.startswith(str(root / "workspace"))


def test_pi_resume_returns_result_and_close_preserves_session(tmp_path, monkeypatch):
    root = tmp_path / "node"
    session_dir = root / "workspace" / ".pi" / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "native.jsonl"
    session_file.write_text("session\n", encoding="utf-8")
    harness = PiRuntime(docker=False)
    session = hytorch.harness.Session("pi", "native", str(session_file))
    monkeypatch.setattr(
        harness,
        "_invoke",
        lambda *args, **kwargs: (
            "updated",
            "native",
            str(session_file),
            hytorch.harness.Usage(),
        ),
    )

    result = harness.resume(session, str(root), "feedback", None)
    harness.close(result.session)

    assert isinstance(result, hytorch.harness.Result)
    assert result.text == "updated"
    assert session_file.is_file()


def test_pi_profile_keeps_native_files_but_never_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    operator = tmp_path / "home" / ".pi" / "agent"
    operator.mkdir(parents=True)
    (operator / "auth.json").write_text(
        '{"openai-codex":{"expires":1,"access":"secret"}}', encoding="utf-8"
    )
    persistent = tmp_path / "candidate" / "agent"
    persistent.mkdir(parents=True)
    (persistent / "auth.json").write_text("leaked", encoding="utf-8")
    (persistent / "settings.json").write_text("{}", encoding="utf-8")

    with _staged_agent_directory(str(persistent)) as staged:
        assert Path(staged, "auth.json").is_file()
        Path(staged, "memory.md").write_text("learned", encoding="utf-8")
        Path(staged, "auth.json").write_text(
            '{"openai-codex":{"expires":2,"access":"refreshed"}}',
            encoding="utf-8",
        )

    assert (persistent / "memory.md").read_text(encoding="utf-8") == "learned"
    assert not (persistent / "auth.json").exists()
    assert "refreshed" in (operator / "auth.json").read_text(encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("HYTORCH_PI_TEST") != "1",
    reason="set HYTORCH_PI_TEST=1 to run this test (spawns a real Pi/OpenAI session)",
)
def test_pi_harness_runs_real_node(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")
    root = hytorch.Space("**/*", repo=new_repo, commit=base)

    instruction = "Create hello.txt containing exactly: hello from hytorch"

    harness = hytorch.harness.PiHarness(
        provider=os.environ.get("HYTORCH_PI_PROVIDER", "openai-codex"),
        model=os.environ.get("HYTORCH_PI_MODEL", "gpt-5.6-terra"),
    )

    class OneNode(hytorch.mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = hytorch.mn.Linear(1, 1, bias=instruction)

        def forward(self, state):
            return self.layer(state)

    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters(), temp=0.2, max_tokens=2_000)
    out = model(root)

    assert len(out) == 1
    assert out[0].summary, "expected a non-empty handoff summary from pi"
    print("pi handoff summary:", out[0].summary)

    with open(os.path.join(out[0].dir, "hello.txt")) as f:
        content = f.read()
    print("hello.txt content:", content)
    assert content == "hello from hytorch"

    before = model.layer.weight[0].revision
    hytorch.Loss(
        out[0],
        feedback=(
            "The output is correct. Add a short reusable verification checklist "
            "to workspace/AGENTS.md. You must edit the workspace before replying."
        ),
    ).backward()
    optimizer.step()

    assert model.layer.weight[0].revision != before
    assert out[0].feed_fn.released


@pytest.mark.skipif(
    os.environ.get("HYTORCH_PI_TEST") != "1",
    reason="set HYTORCH_PI_TEST=1 to run this test (spawns real Pi/OpenAI sessions)",
)
def test_pi_harness_runs_real_multilayer_training_step(new_repo):
    root = hytorch.space(new_repo.root)
    harness = hytorch.harness.PiHarness(
        provider=os.environ.get("HYTORCH_PI_PROVIDER", "openai-codex"),
        model=os.environ.get("HYTORCH_PI_MODEL", "gpt-5.6-terra"),
    )

    class ResearchAndSynthesize(hytorch.mn.Module):
        def __init__(self):
            super().__init__()
            self.research = hytorch.mn.Linear(
                1,
                2,
                bias="Research independently and preserve concrete evidence.",
            )
            self.synthesize = hytorch.mn.Linear(
                2,
                1,
                bias="Combine every research input and verify the requested result.",
            )

        def forward(self, state):
            branches = self.research(
                state,
                task=(
                    "Your node prompt identifies you as research[0] or research[1]. "
                    "Read base.txt. Create research-<your agent index>.md. Its first "
                    "line must be `research branch <your agent index>`. Add one short "
                    "observation. Do not change existing files."
                ),
            )
            return self.synthesize(
                branches,
                task=(
                    "Merge both research inputs. Confirm that research-0.md and "
                    "research-1.md are present. Create answer.txt containing exactly "
                    "`combined by hytorch` with no trailing newline."
                ),
            )[0]

    model = ResearchAndSynthesize().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters(), temp=0.2, max_tokens=2_500)
    views = [
        model.research.weight[0],
        model.research.weight[1],
        model.synthesize.weight[0],
    ]
    before = [view.text() for view in views]

    output = model(root)

    with open(os.path.join(output.dir, "answer.txt"), encoding="utf-8") as f:
        assert f.read() == "combined by hytorch"
    for index in range(2):
        path = os.path.join(output.dir, f"research-{index}.md")
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            assert f.readline().strip() == f"research branch {index}"

    hytorch.Loss(
        output,
        feedback=(
            "The result is correct. Improve your persistent workspace by adding a "
            "reusable verification checklist to AGENTS.md and committing it. Your "
            "JSON feedback array must contain this exact direction once for each "
            "input: `Add a reusable verification checklist to AGENTS.md and commit "
            "the workspace change before replying.` Every agent must edit its "
            "workspace."
        ),
    ).backward()
    optimizer.step()

    after = [view.text() for view in views]
    assert all(current != previous for previous, current in zip(before, after))
    assert all("verification" in current.lower() for current in after)
    assert output.feed_fn.released
