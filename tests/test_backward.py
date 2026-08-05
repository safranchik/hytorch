import json
import os
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


class DirectionHarness(hytorch.harness.Harness):
    def __init__(self, name="direction"):
        super().__init__(name)
        self.started = []
        self.resumed = []
        self.closed = []
        self.received = {}
        self.max_tokens = None

    def start(self, directory, prompt, mtype, **kwargs):
        session = hytorch.harness.Session(self.name, f"session-{len(self.started)}", "")
        self.started.append(session)
        statespace = os.path.join(directory, "statespace")
        merge_agent_inputs(statespace)
        with open(os.path.join(statespace, "answer.txt"), "a") as file:
            file.write(session.id + "\n")
        commit_agent_changes(statespace, "agent: finish forward")
        return hytorch.harness.Result("finished forward", session)

    def resume(self, session, directory, prompt, mtype, **kwargs):
        self.resumed.append(session)
        self.max_tokens = kwargs.get("max_tokens")
        self.received[session.id] = prompt
        workspace = os.path.join(directory, "workspace")
        selected = find_agent_workspace(workspace)
        with open(os.path.join(selected, f"update-{session.id}.md"), "w") as file:
            file.write("Apply the received direction in future passes.\n")
        commit_agent_changes(workspace, "agent: update workspace")
        input_count = len(
            os.listdir(
                os.path.join(
                    directory, "statespace", ".git", "refs", "hytorch", "inputs"
                )
            )
        )
        return json.dumps(
            {
                "feedback": [
                    f"Improve input {index} for {session.id}."
                    for index in range(input_count)
                ]
            }
        )

    def close(self, session):
        self.closed.append(session)


class OneNode(mn.Module):
    def __init__(self):
        super().__init__()
        self.layer = mn.Linear(1, 1, bias="Answer correctly.")

    def forward(self, value):
        return self.layer(value, task="answer")[0]


def test_loss_contains_only_output_and_directional_feedback(new_repo):
    harness = hytorch.harness.register(DirectionHarness())
    model = OneNode().to(harness)
    hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    loss = hytorch.Loss(output, "Handle malformed inputs more carefully.")

    assert loss.output is output
    assert loss.feedback == "Handle malformed inputs more carefully."
    with pytest.raises(TypeError):
        hytorch.Loss(output, score=0.0, critique="old API")


def test_backward_updates_candidate_and_step_promotes_it(new_repo):
    harness = hytorch.harness.register(DirectionHarness("promote"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters(), temp=0.4)
    output = model(hytorch.space(new_repo.root, harness=harness))
    before = model.layer.weight[0].revision

    hytorch.Loss(output, "Use a stricter validation rule.").backward()

    assert model.layer.weight[0].revision == before
    assert not os.path.exists(
        os.path.join(model.layer.weight[0].path, "update-session-0.md")
    )
    assert output.feed_fn.released
    assert {session.id for session in harness.closed} == {
        session.id for session in harness.started
    }
    assert model.layer.weight[0].feed == ("Use a stricter validation rule.",)
    agent_commit = run_git(output.feed_fn.workspace, "rev-parse", "HEAD")
    history = run_git(output.feed_fn.workspace, "log", "--format=%s")
    assert "hytorch: initialize meta-network workspaces" in history

    optimizer.step()

    assert model.layer.weight[0].revision != before
    assert model.layer.weight._store.repo.is_ancestor(
        agent_commit, model.layer.weight[0].revision
    )
    assert os.path.isfile(
        os.path.join(model.layer.weight[0].path, "update-session-0.md")
    )


def test_backward_propagates_separate_feedback_and_accumulates_fanout(new_repo):
    class Fanout(mn.Module):
        def __init__(self):
            super().__init__()
            self.first = mn.Linear(1, 1, bias="Analyze.")
            self.split = mn.Linear(1, 2, bias="Propose.")
            self.final = mn.Linear(2, 1, bias="Select.")

        def forward(self, value):
            first = self.first(value)
            split = self.split(first)
            return self.final(split)[0]

    harness = hytorch.harness.register(DirectionHarness("fanout"))
    model = Fanout().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Prefer the smallest correct solution.").backward()

    assert model.final.weight[0].feed == ("Prefer the smallest correct solution.",)
    assert len(model.split.weight[0].feed) == 1
    assert len(model.split.weight[1].feed) == 1
    assert len(model.first.weight[0].feed) == 2
    assert {session.id for session in harness.closed} == {
        session.id for session in harness.started
    }
    optimizer.step()


def test_backward_runs_ready_distinct_workspaces_in_parallel(new_repo):
    class ParallelDirection(DirectionHarness):
        def __init__(self):
            super().__init__("backward-parallel")
            self.split_sessions = set()
            self.barrier = threading.Barrier(2)

        def start(self, directory, prompt, mtype, **kwargs):
            result = super().start(directory, prompt, mtype, **kwargs)
            if "Run node split[" in prompt:
                self.split_sessions.add(result.session.id)
            return result

        def resume(self, session, directory, prompt, mtype, **kwargs):
            if session.id in self.split_sessions:
                self.barrier.wait(timeout=5)
            return super().resume(session, directory, prompt, mtype, **kwargs)

    class ParallelLayer(mn.Module):
        def __init__(self):
            super().__init__()
            self.split = mn.Linear(1, 2, bias="Propose.")
            self.final = mn.Linear(2, 1, bias="Select.")

        def forward(self, value):
            return self.final(self.split(value))[0]

    harness = hytorch.harness.register(ParallelDirection())
    model = ParallelLayer().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Make the result more precise.").backward()
    optimizer.step()

    for view in model.split.weight:
        assert any(name.startswith("update-session-") for name in os.listdir(view.path))


def test_zero_feed_discards_unpromoted_candidate(new_repo):
    harness = hytorch.harness.register(DirectionHarness("discard"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))
    before = model.layer.weight[0].revision

    hytorch.Loss(output, "Use fewer steps.").backward()
    optimizer.zero_feed()
    optimizer.step()

    assert model.layer.weight[0].revision == before
    assert model.layer.weight[0].feed is None


def test_backward_allows_an_unchanged_workspace(new_repo):
    class NoMutation(DirectionHarness):
        def resume(self, session, directory, prompt, mtype, **kwargs):
            return json.dumps(
                {"feedback": ["Preserve more useful detail in the input."]}
            )

    harness = hytorch.harness.register(NoMutation("no-mutation"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))
    before = model.layer.weight[0].revision

    hytorch.Loss(output, "Keep the current behavior.").backward()
    optimizer.step()

    assert model.layer.weight[0].revision == before
    assert output.feed_fn.released


def test_backward_uses_max_tokens_as_agent_budget(new_repo):
    harness = hytorch.harness.register(DirectionHarness("budget"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters(), max_tokens=20)
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Use a more direct method.").backward()

    assert harness.max_tokens == 20
    optimizer.step()


def test_backward_rejects_statespace_mutation(new_repo):
    class InvalidDirection(DirectionHarness):
        def resume(self, session, directory, prompt, mtype, **kwargs):
            path = os.path.join(directory, "statespace", "answer.txt")
            os.chmod(path, 0o644)
            with open(path, "w") as file:
                file.write("illegal\n")
            return super().resume(session, directory, prompt, mtype, **kwargs)

    harness = hytorch.harness.register(InvalidDirection("invalid-state"))
    model = OneNode().to(harness)
    hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    with pytest.raises(ValueError, match="modified the read-only statespace"):
        hytorch.Loss(output, "Preserve the output state.").backward()


def test_backward_requires_one_feedback_string_per_input(new_repo):
    class MissingFeedback(DirectionHarness):
        def resume(self, session, directory, prompt, mtype, **kwargs):
            workspace = os.path.join(directory, "workspace")
            selected = find_agent_workspace(workspace)
            with open(os.path.join(selected, "changed.md"), "w") as file:
                file.write("changed\n")
            commit_agent_changes(workspace, "agent: incomplete response")
            return json.dumps({"feedback": []})

    harness = hytorch.harness.register(MissingFeedback("missing-feedback"))
    model = OneNode().to(harness)
    hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    with pytest.raises(RuntimeError, match="requires 1 feedback strings"):
        hytorch.Loss(output, "Explain the required input change.").backward()
