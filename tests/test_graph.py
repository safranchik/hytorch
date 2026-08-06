import os
import threading

import pytest
from conftest import merge_agent_inputs

import hytorch
from hytorch import mn


def test_public_version_matches_release():
    assert hytorch.__version__ == "0.1.0"


def test_all_builtin_native_harnesses_are_executable_adapters():
    builtins = {
        "pi": hytorch.harness.PiHarness,
        "codex": hytorch.harness.CodexHarness,
        "claude-code": hytorch.harness.ClaudeCodeHarness,
        "opencode": hytorch.harness.OpenCodeHarness,
        "hermes": hytorch.harness.HermesHarness,
        "prime-agent": hytorch.harness.PrimeAgentHarness,
    }

    registered = hytorch.harness.registered()
    for name, adapter in builtins.items():
        assert isinstance(registered[name], adapter)


class NullHarness(hytorch.harness.Harness):
    def start(self, directory, prompt, mtype, **kwargs):
        merge_agent_inputs(os.path.join(directory, "statespace"))
        session = hytorch.harness.Session(self.name, f"session-{id(directory)}", "")
        return hytorch.harness.Result("done", session)

    def resume(self, session, directory, prompt, mtype, **kwargs):
        return hytorch.harness.Result("done", session)

    def close(self, session):
        pass


def test_module_registers_nested_modules_and_parameters():
    class Block(mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = mn.Linear(1, 2, bias="Explore.")

        def forward(self, value):
            return self.layer(value)

    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, value):
            return self.block(value)

    model = Model()
    assert [name for name, _ in model.named_children()] == ["block"]
    assert [name for name, _ in model.named_modules()] == ["", "block", "block.layer"]
    assert [name for name, _ in model.named_parameters()] == [
        "block.layer.weight",
    ]


def test_assigning_module_or_parameter_requires_super_init():
    class Broken(mn.Module):
        def __init__(self):
            self.layer = mn.Linear(1, 1)

        def forward(self, value):
            return value

    try:
        Broken()
    except AttributeError as exc:
        assert "super().__init__()" in str(exc)
    else:
        raise AssertionError("expected assignment before Module.__init__ to fail")


def test_parameter_can_wrap_a_complete_space(new_repo):
    class Model(mn.Module):
        def __init__(self, value):
            super().__init__()
            self.state = mn.Parameter(value)

        def forward(self, value):
            return value

    model = Model(hytorch.space(new_repo.root))
    list(model.parameters())

    assert model.state.shape == (1,)
    assert os.path.isfile(os.path.join(model.state[0].path, "base.txt"))
    assert not os.path.exists(os.path.join(model.state[0].path, ".git"))


def test_linear_outputs_run_in_parallel(new_repo):
    barrier = threading.Barrier(2)

    class BarrierHarness(hytorch.harness.Harness):
        def start(self, directory, prompt, mtype, **kwargs):
            barrier.wait(timeout=2)
            merge_agent_inputs(os.path.join(directory, "statespace"))
            session = hytorch.harness.Session(self.name, f"session-{id(directory)}", "")
            return hytorch.harness.Result("done", session)

        def resume(self, session, directory, prompt, mtype, **kwargs):
            return hytorch.harness.Result("done", session)

        def close(self, session):
            pass

    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = mn.Linear(1, 2)

        def forward(self, value):
            return self.layer(value)

    harness = hytorch.harness.register(BarrierHarness("parallel"))
    root = hytorch.space(new_repo.root, harness=harness)
    assert len(Model().to(harness)(root)) == 2


def test_arbitrary_forward_topology_composes(new_repo):
    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.research = mn.Linear(1, 2, bias="Research.")
            self.verify = mn.Linear(2, 1, bias="Verify.")

        def forward(self, value):
            branches = self.research(value)
            return self.verify(branches)[0]

    harness = hytorch.harness.register(NullHarness("compose"))
    output = Model().to(harness)(hytorch.space(new_repo.root, harness=harness))
    assert output.layer == "verify"
    assert len(output.feed_fn.parameters) == 1
    assert len(output.feed_fn.parents) == 2


def test_training_forks_one_episode_per_repeated_parameter_execution(new_repo):
    class Recurrent(mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = mn.Linear(1, 1)

        def forward(self, value):
            return self.layer(self.layer(value)[0])[0]

    harness = hytorch.harness.register(NullHarness("recurrent-state"))
    model = Recurrent().to(harness)

    output = model(hytorch.space(new_repo.root, harness=harness))

    second = output.feed_fn
    first = second.parents[0]
    assert second.parameters[0].relative_path == first.parameters[0].relative_path
    assert second.workspace != first.workspace
    assert second.session.id != first.session.id


def test_module_to_supplies_harness_and_mtype(new_repo):
    class RecordingHarness(hytorch.harness.Harness):
        def __init__(self):
            super().__init__("placement")
            self.mtypes = []

        def start(self, directory, prompt, mtype, **kwargs):
            self.mtypes.append(mtype)
            merge_agent_inputs(os.path.join(directory, "statespace"))
            session = hytorch.harness.Session(self.name, "placement-session", "")
            return hytorch.harness.Result("done", session)

        def resume(self, session, directory, prompt, mtype, **kwargs):
            return hytorch.harness.Result("done", session)

        def close(self, session):
            pass

    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = mn.Linear(1, 1)

        def forward(self, value):
            return self.layer(value)[0]

    harness = RecordingHarness()
    model = Model().to(harness=harness)
    assert model.to(mtype="gpt-5.6-terra") is model
    assert model.to() is model
    model(hytorch.space(new_repo.root))
    assert harness.mtypes == ["gpt-5.6-terra"]


def test_model_rejects_mixed_harnesses(new_repo):
    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.first = mn.Linear(1, 1)
            self.second = mn.Linear(1, 1)

        def forward(self, value):
            return self.second(self.first(value))[0]

    first = hytorch.harness.register(NullHarness("model-harness-a"))
    second = hytorch.harness.register(NullHarness("model-harness-b"))
    model = Model().to(first)
    model.second.to(second)

    with pytest.raises(RuntimeError, match="one model must use one harness"):
        model(hytorch.space(new_repo.root, harness=first))


def test_inference_mode_releases_sessions_without_recording_feed(new_repo):
    class ClosingHarness(NullHarness):
        def __init__(self):
            super().__init__("inference")
            self.closed = []

        def close(self, session):
            self.closed.append(session.id)

    class Model(mn.Module):
        def __init__(self):
            super().__init__()
            self.first = mn.Linear(1, 2)
            self.final = mn.Linear(2, 1)

        def forward(self, value):
            return self.final(self.first(value))[0]

    harness = hytorch.harness.register(ClosingHarness())
    model = Model().to(harness)
    with hytorch.inference_mode():
        assert hytorch.is_inference_mode_enabled()
        output = model(hytorch.space(new_repo.root, harness=harness))

    assert not hytorch.is_inference_mode_enabled()
    assert output.feed_fn is None
    assert not output.requires_feed
    assert len(harness.closed) == 3
