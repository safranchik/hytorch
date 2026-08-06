import json
import os
import subprocess
import threading

import pytest
from conftest import (
    commit_agent_changes,
    find_agent_workspace,
    merge_agent_inputs,
    run_git,
)

import hytorch
from hytorch import mn


class RecordingHarness(hytorch.harness.Harness):
    def __init__(self, name="recording"):
        super().__init__(name)
        self.prompts = []
        self.layouts = []
        self.lock = threading.Lock()

    def start(self, directory, prompt, mtype, **kwargs):
        with self.lock:
            self.prompts.append(prompt)
            self.layouts.append(sorted(os.listdir(directory)))
        statespace = os.path.join(directory, "statespace")
        merge_agent_inputs(statespace)
        with open(os.path.join(statespace, "output.txt"), "a") as file:
            file.write("done\n")
        commit_agent_changes(statespace, "agent: finish forward")
        session = hytorch.harness.Session(self.name, f"session-{id(directory)}", "")
        return hytorch.harness.Result("done", session)

    def resume(self, session, directory, prompt, mtype, **kwargs):
        return "done"

    def close(self, session):
        pass


class Model(mn.Module):
    def __init__(self, in_features, out_features, bias="Decide carefully."):
        super().__init__()
        self.layer = mn.Linear(in_features, out_features, bias=bias)

    def forward(self, spaces, task=""):
        return self.layer(spaces, task=task)


def test_linear_registers_one_workspace_per_output():
    layer = mn.Linear(3, 4, bias="Decide carefully.")
    assert layer.weight.shape == (4,)
    assert layer.bias == "Decide carefully."
    assert [name for name, _ in layer.named_parameters(recurse=False)] == ["weight"]


def test_default_priors_are_general_and_seeded():
    hytorch.manual_seed(7)
    first = mn.Linear(3, 3, bias="Do the task.")
    values = [first.weight[i].text() for i in range(3)]

    hytorch.manual_seed(7)
    second = mn.Linear(3, 3, bias="Do the task.")
    assert values == [second.weight[i].text() for i in range(3)]
    used = [
        prior
        for prior in mn.init.DEFAULT_PRIORS
        if any(prior in value for value in values)
    ]
    assert len(used) == 9
    assert all("space" not in value.lower() for value in mn.init.DEFAULT_PRIORS)


def test_parameter_tree_is_one_directory_per_agent():
    model = Model(2, 3, bias="Resolve the task.")
    list(model.parameters())

    assert model.layer.weight[1].relative_path == "layers/layer/1"
    assert os.path.isdir(model.layer.weight[1].path)
    assert "## Input 0" in model.layer.weight[1].text()
    assert "## Input 1" in model.layer.weight[1].text()
    assert model.layer.bias == "Resolve the task."
    store = model.layer.weight._store
    assert store.root.startswith(os.path.abspath("hytorch/workspaces") + os.sep)
    with open(os.path.join(store.root, "MODEL.json"), encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["modules"]["layer"]["parameters"]["weight"]["shape"] == [3]
    assert manifest["modules"]["layer"]["parameters"]["weight"]["input_features"] == 2


def test_init_functions_can_reset_materialized_parameters():
    model = Model(2, 1)
    list(model.parameters())
    before = model.layer.weight[0].revision

    mn.init.constant_(model.layer.weight, "# Initial policy\n\nStay concise.\n")

    assert model.layer.weight[0].text().endswith("Stay concise.\n")
    assert model.layer.weight[0].revision != before


def test_dense_linear_runs_one_agent_per_output_without_state_injection(new_repo):
    hytorch.manual_seed(3)
    harness = hytorch.harness.register(RecordingHarness())
    spaces = hytorch.space([new_repo.root, new_repo.root], harness=harness)
    model = Model(2, 3).to(harness)
    expected = model.layer.weight[0].text().strip()

    outputs = model(spaces, task="answer")

    assert len(outputs) == 3
    assert len(harness.prompts) == 3
    assert all(
        "workspace/" in prompt and "statespace/" in prompt for prompt in harness.prompts
    )
    assert all("# Task\n\nanswer" in prompt for prompt in harness.prompts)
    assert all(expected not in prompt for prompt in harness.prompts)
    assert all(os.path.isdir(output.feed_fn.workspace) for output in outputs)
    assert all(os.path.basename(output.dir) == "statespace" for output in outputs)
    assert harness.layouts == [["statespace", "workspace"]] * 3
    for output in outputs:
        assert os.path.isdir(os.path.join(output.dir, ".git"))
        assert not os.path.exists(os.path.join(output.feed_fn.root, "inputs"))
        assert (
            subprocess.run(
                [
                    "git",
                    "-C",
                    output.dir,
                    "merge-base",
                    "--is-ancestor",
                    "refs/hytorch/inputs/0",
                    "HEAD",
                ],
                check=False,
            ).returncode
            == 0
        )
        assert not os.path.exists(os.path.join(output.dir, "AGENTS.md"))
    for index in range(3):
        assert "Decide carefully." in model.layer.weight[index].text()


def test_forward_rejects_workspace_mutation(new_repo):
    class InvalidHarness(RecordingHarness):
        def start(self, directory, prompt, mtype, **kwargs):
            path = os.path.join(
                find_agent_workspace(os.path.join(directory, "workspace")),
                "AGENTS.md",
            )
            os.chmod(path, 0o644)
            with open(path, "w") as file:
                file.write("illegal\n")
            session = hytorch.harness.Session(self.name, "invalid", "")
            return hytorch.harness.Result("done", session)

    harness = hytorch.harness.register(InvalidHarness("invalid-forward"))
    model = Model(1, 1).to(harness)
    with pytest.raises(RuntimeError, match="forward modified read-only workspace"):
        model(hytorch.space(new_repo.root, harness=harness))


def test_forward_requires_committed_agent_changes(new_repo):
    class UncommittedHarness(RecordingHarness):
        def start(self, directory, prompt, mtype, **kwargs):
            statespace = os.path.join(directory, "statespace")
            merge_agent_inputs(statespace)
            with open(os.path.join(statespace, "agent.txt"), "w") as file:
                file.write("uncommitted\n")
            session = hytorch.harness.Session(self.name, "uncommitted", "")
            return hytorch.harness.Result("uncommitted", session)

    harness = hytorch.harness.register(UncommittedHarness("uncommitted"))
    model = Model(1, 1).to(harness)

    with pytest.raises(RuntimeError, match="uncommitted statespace changes"):
        model(hytorch.space(new_repo.root, harness=harness))


def test_dense_linear_validates_input_width(new_repo):
    harness = hytorch.harness.register(RecordingHarness("width"))
    model = Model(2, 1).to(harness)
    with pytest.raises(ValueError, match="in_features=2"):
        model(hytorch.space(new_repo.root, harness=harness))


def test_linear_fetches_inputs_from_independent_repositories(new_repo, tmp_path):
    other = str(tmp_path / "independent")
    os.makedirs(other)
    run_git(other, "init", "-b", "main")
    run_git(other, "config", "user.name", "Test")
    run_git(other, "config", "user.email", "test@example.com")
    with open(os.path.join(other, "other.txt"), "w") as file:
        file.write("independent state\n")
    run_git(other, "add", "-A")
    run_git(other, "commit", "-m", "independent input")

    harness = hytorch.harness.register(RecordingHarness("independent-inputs"))
    inputs = hytorch.space([new_repo.root, other], harness=harness)
    output = Model(2, 1).to(harness)(inputs)[0]

    assert os.path.realpath(output.repo.root) == os.path.realpath(output.dir)
    assert os.path.realpath(output.repo.git_dir) == os.path.realpath(
        os.path.join(output.dir, ".git")
    )
    for index in range(2):
        assert (
            subprocess.run(
                [
                    "git",
                    "-C",
                    output.dir,
                    "merge-base",
                    "--is-ancestor",
                    f"refs/hytorch/inputs/{index}",
                    "HEAD",
                ],
                check=False,
            ).returncode
            == 0
        )
