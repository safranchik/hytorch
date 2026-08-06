import json
import os
import threading

import pytest
from conftest import (
    commit_agent_changes,
    find_agent_workspace,
    merge_agent_inputs,
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
        self.owner_reductions = []

    def start(self, directory, prompt, mtype, **kwargs):
        session = hytorch.harness.Session(self.name, f"session-{len(self.started)}", "")
        self.started.append(session)
        workspace = find_agent_workspace(os.path.join(directory, "workspace"))
        if prompt.startswith("Update your persistent native state"):
            self.owner_reductions.append(prompt)
            with open(os.path.join(workspace, "owner-update.md"), "w") as file:
                file.write("Reduced all accumulated feed.\n")
            return hytorch.harness.Result("finished owner update", session)
        statespace = os.path.join(directory, "statespace")
        merge_agent_inputs(statespace)
        with open(os.path.join(statespace, "answer.txt"), "a") as file:
            file.write(session.id + "\n")
        commit_agent_changes(statespace, "agent: finish forward")
        with open(os.path.join(workspace, "forward-memory.md"), "w") as file:
            file.write("This session completed a forward turn.\n")
        return hytorch.harness.Result("finished forward", session)

    def resume(self, session, directory, prompt, mtype, **kwargs):
        self.resumed.append(session)
        self.max_tokens = kwargs.get("max_tokens")
        self.received[session.id] = prompt
        workspace = os.path.join(directory, "workspace")
        selected = find_agent_workspace(workspace)
        with open(os.path.join(selected, f"update-{session.id}.md"), "w") as file:
            file.write("Apply the received direction in future passes.\n")
        input_count = len(
            os.listdir(
                os.path.join(
                    directory, "statespace", ".git", "refs", "hytorch", "inputs"
                )
            )
        )
        return hytorch.harness.Result(
            json.dumps(
                {
                    "update": "Apply the received direction in future passes.",
                    "feedback": [
                        f"Improve input {index} for {session.id}."
                        for index in range(input_count)
                    ],
                }
            ),
            session,
        )

    def close(self, session):
        self.closed.append(session)


def test_two_forwards_accumulate_feed_and_step_reduces_owner_once(new_repo):
    harness = hytorch.harness.register(DirectionHarness("accumulate"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())

    first = model(hytorch.space(new_repo.root, harness=harness))
    second = model(hytorch.space(new_repo.root, harness=harness))
    hytorch.Loss(first, "Keep exact evidence.").backward()
    hytorch.Loss(second, "Use a smaller proof.").backward()

    assert model.layer.weight[0].feed == (
        "Apply the received direction in future passes.",
        "Apply the received direction in future passes.",
    )
    records = next(iter(optimizer._records.values()))
    assert len(records) == 2
    assert len({record.digest for record in records}) == 2

    optimizer.step()

    assert len(harness.owner_reductions) == 1


def test_backward_retain_graph_matches_pytorch_lifecycle(new_repo):
    harness = hytorch.harness.register(DirectionHarness("retain"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Check correctness.").backward(retain_graph=True)
    assert not output.feed_fn.consumed
    assert not output.feed_fn.released
    hytorch.Loss(output, "Check efficiency.").backward()
    assert output.feed_fn.consumed
    assert output.feed_fn.released
    assert len(model.layer.weight[0].feed) == 2

    with pytest.raises(RuntimeError, match="second time"):
        hytorch.Loss(output, "Run again.").backward()
    optimizer.step()


def test_step_is_atomic_and_retains_feed_after_reducer_failure(new_repo):
    class FailingReducer(DirectionHarness):
        def start(self, directory, prompt, mtype, **kwargs):
            if prompt.startswith(
                "Update your persistent native state for Parameter layers/layer/1"
            ):
                raise RuntimeError("reducer failed")
            return super().start(directory, prompt, mtype, **kwargs)

    harness = hytorch.harness.register(FailingReducer("atomic"))
    model = mn.Linear(1, 2, bias="Be exact.")

    class TwoOwners(mn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, value):
            return self.layer(value)

    network = TwoOwners(model).to(harness)
    optimizer = hytorch.optim.DFM(network.parameters())
    outputs = network(hytorch.space(new_repo.root, harness=harness))
    before = model.weight[0].revision
    hytorch.Loss(outputs[0], "Improve owner zero.").backward()
    hytorch.Loss(outputs[1], "Improve owner one.").backward()

    with pytest.raises(RuntimeError, match="reducer failed"):
        optimizer.step()

    assert model.weight[0].revision == before
    assert model.weight[0].feed is not None
    assert model.weight[1].feed is not None
    optimizer.zero_feed()


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


def test_backward_accumulates_feed_and_step_updates_parameter(new_repo):
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
    assert not os.path.exists(
        os.path.join(model.layer.weight[0].path, "forward-memory.md")
    )
    assert output.feed_fn.released
    assert {session.id for session in harness.closed} == {
        session.id for session in harness.started
    }
    assert model.layer.weight[0].feed == (
        "Apply the received direction in future passes.",
    )
    assert not os.path.exists(output.feed_fn.workspace)

    optimizer.step()

    assert model.layer.weight[0].revision != before
    assert os.path.isfile(os.path.join(model.layer.weight[0].path, "owner-update.md"))
    assert not os.path.exists(
        os.path.join(model.layer.weight[0].path, "forward-memory.md")
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

    assert model.final.weight[0].feed == (
        "Apply the received direction in future passes.",
    )
    assert len(model.split.weight[0].feed) == 1
    assert len(model.split.weight[1].feed) == 1
    assert len(model.first.weight[0].feed) == 1
    assert {session.id for session in harness.closed} == {
        session.id for session in harness.started
    }
    optimizer.step()


def test_frozen_intermediate_parameter_propagates_without_feed(new_repo):
    class FrozenMiddle(mn.Module):
        def __init__(self):
            super().__init__()
            self.first = mn.Linear(1, 1, bias="Learn.")
            self.middle = mn.Linear(1, 1, bias="Stay fixed.")
            self.middle.weight.requires_feed = False

        def forward(self, value):
            return self.middle(self.first(value))[0]

    harness = hytorch.harness.register(DirectionHarness("frozen"))
    model = FrozenMiddle().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Improve the learned input.").backward()

    assert model.middle.weight[0].feed is None
    assert model.first.weight[0].feed is not None
    optimizer.step()
    assert len(harness.owner_reductions) == 1


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
        assert os.path.isfile(os.path.join(view.path, "owner-update.md"))


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
            return hytorch.harness.Result(
                json.dumps(
                    {
                        "update": "Keep the useful behavior.",
                        "feedback": ["Preserve more useful detail in the input."],
                    }
                ),
                session,
            )

    harness = hytorch.harness.register(NoMutation("no-mutation"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))
    before = model.layer.weight[0].revision

    hytorch.Loss(output, "Keep the current behavior.").backward()
    optimizer.step()

    assert model.layer.weight[0].revision != before
    assert os.path.isfile(os.path.join(model.layer.weight[0].path, "owner-update.md"))
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
            return hytorch.harness.Result(
                json.dumps({"update": "Use exact evidence.", "feedback": []}),
                session,
            )

    harness = hytorch.harness.register(MissingFeedback("missing-feedback"))
    model = OneNode().to(harness)
    hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    with pytest.raises(RuntimeError, match="requires 1 feedback strings"):
        hytorch.Loss(output, "Explain the required input change.").backward()


def test_backward_accepts_a_rotated_native_session_tip(new_repo):
    class CompactingHarness(DirectionHarness):
        def resume(self, session, directory, prompt, mtype, **kwargs):
            result = super().resume(session, directory, prompt, mtype, **kwargs)
            rotated = hytorch.harness.Session(
                session.harness, session.id + "-compacted", session.storage
            )
            return hytorch.harness.Result(result.text, rotated)

    harness = hytorch.harness.register(CompactingHarness("compacting"))
    model = OneNode().to(harness)
    optimizer = hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    hytorch.Loss(output, "Retain the useful context.").backward()

    assert harness.closed[-1].id.endswith("-compacted")
    optimizer.step()


def test_backward_rejects_git_metadata_in_native_agent_state(new_repo):
    class NestedGitHarness(DirectionHarness):
        def resume(self, session, directory, prompt, mtype, **kwargs):
            result = super().resume(session, directory, prompt, mtype, **kwargs)
            os.makedirs(os.path.join(directory, "workspace", ".git"))
            return result

    harness = hytorch.harness.register(NestedGitHarness("nested-git"))
    model = OneNode().to(harness)
    hytorch.optim.DFM(model.parameters())
    output = model(hytorch.space(new_repo.root, harness=harness))

    with pytest.raises(ValueError, match="must not contain Git metadata"):
        hytorch.Loss(output, "Keep state portable.").backward()
