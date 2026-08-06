import json
import os
from dataclasses import FrozenInstanceError

import pytest
from conftest import run_git

import hytorch
import hytorch.state_dir as state_dir_module
from hytorch import mn
from hytorch._git import Repo


class Model(mn.Module):
    def __init__(self, width: int = 2, bias: str = "source"):
        super().__init__()
        self.layer = mn.Linear(1, width, bias=bias)

    def forward(self, value):
        return self.layer(value)


def test_state_dir_save_and_load_round_trip(tmp_path):
    source = Model(bias="trained policy")
    source.layer.weight[0]._set_text("print('check')", "tools/check.py")
    source._ensure_parameter_store().commit("test: add workspace tool")
    source_state = source.state_dir()
    checkpoint = tmp_path / "model-state"

    hytorch.save(source_state, checkpoint)
    loaded = hytorch.load(checkpoint)

    assert isinstance(source_state, hytorch.StateDir)
    assert loaded.commit == source_state.commit
    assert os.path.isdir(checkpoint / ".git")
    assert json.loads((checkpoint / "MODEL.json").read_text())["format"] == (
        "hytorch-model-v1"
    )

    restored = Model(bias="different policy")
    result = restored.load_state_dir(loaded)
    assert repr(result) == "<All keys matched successfully>"
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert restored.layer.weight[0].text() == source.layer.weight[0].text()
    assert restored.layer.weight[1].text() == source.layer.weight[1].text()
    assert restored.layer.weight[0].text("tools/check.py") == "print('check')\n"
    assert restored.state_dir().commit != source_state.commit

    with pytest.raises(FrozenInstanceError):
        source_state.commit = "changed"


def test_save_uses_committed_revision_not_worktree(tmp_path):
    model = Model(width=1)
    state = model.state_dir()
    agents = os.path.join(state.dir, "layers", "layer", "0", "AGENTS.md")
    with open(agents, "w", encoding="utf-8") as file:
        file.write("uncommitted\n")

    checkpoint = tmp_path / "model-state"
    hytorch.save(state, checkpoint)

    assert (checkpoint / "layers/layer/0/AGENTS.md").read_text() != "uncommitted\n"
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        model.state_dir()


def test_save_rejects_existing_or_nested_destination(tmp_path):
    state = Model(width=1).state_dir()
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        hytorch.save(state, existing)
    with pytest.raises(ValueError, match="inside the model state"):
        hytorch.save(state, os.path.join(state.dir, "nested"))


def test_load_rejects_non_model_repository(new_repo):
    with pytest.raises(ValueError, match="MODEL.json"):
        hytorch.load(new_repo.root)


def test_strict_load_reports_missing_and_unexpected_keys(tmp_path):
    class ExpandedModel(Model):
        def __init__(self):
            super().__init__(width=1)
            self.extra = mn.Linear(1, 1, bias="extra")

    one = Model(width=1)
    checkpoint = tmp_path / "one"
    hytorch.save(one.state_dir(), checkpoint)
    state = hytorch.load(checkpoint)
    two = ExpandedModel()

    with pytest.raises(RuntimeError, match="Missing key.*extra.weight.0"):
        two.load_state_dir(state)

    result = two.load_state_dir(state, strict=False)
    assert result.missing_keys == ["extra.weight.0"]
    assert result.unexpected_keys == []
    assert two.layer.weight[0].text() == one.layer.weight[0].text()

    expanded_checkpoint = tmp_path / "expanded"
    hytorch.save(ExpandedModel().state_dir(), expanded_checkpoint)
    simple = Model(width=1)
    result = simple.load_state_dir(hytorch.load(expanded_checkpoint), strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == ["extra.weight.0"]


def test_load_rejects_shape_mismatch_even_when_not_strict(tmp_path):
    source = Model(width=1)
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)
    destination = Model(width=2)

    with pytest.raises(RuntimeError, match="size mismatch"):
        destination.load_state_dir(hytorch.load(checkpoint), strict=False)


def test_load_rejects_logical_input_width_mismatch(tmp_path):
    class WideInputModel(mn.Module):
        def __init__(self):
            super().__init__()
            self.layer = mn.Linear(2, 1)

        def forward(self, *values):
            return self.layer(*values)

    source = Model(width=1)
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)

    with pytest.raises(RuntimeError, match="input feature mismatch"):
        WideInputModel().load_state_dir(hytorch.load(checkpoint))


def test_load_accepts_v1_manifest_without_logical_width(tmp_path):
    source = Model(width=1)
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)
    manifest_path = checkpoint / "MODEL.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["modules"]["layer"]["parameters"]["weight"]["input_features"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run_git(str(checkpoint), "add", "MODEL.json")
    run_git(str(checkpoint), "commit", "-m", "test: emulate old manifest")

    restored = Model(width=1)
    restored.load_state_dir(hytorch.load(checkpoint))
    assert restored.layer.weight[0].text() == source.layer.weight[0].text()


def test_state_dir_preserves_model_generation_history(tmp_path):
    model = Model(width=1)
    initial = model.state_dir()
    model.layer.weight[0]._set_text("new policy")
    model._parameter_store.commit("test: update policy")
    updated = model.state_dir()
    checkpoint = tmp_path / "history"
    hytorch.save(updated, checkpoint)
    saved_repo = Repo.discover(str(checkpoint))

    assert saved_repo.is_ancestor(initial.commit, updated.commit)
    assert saved_repo.resolve("HEAD") == updated.commit


def test_load_rejects_pending_optimizer_update(tmp_path):
    source = Model(width=1)
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)
    destination = Model(width=1)
    optimizer = hytorch.optim.DFM(destination.parameters())
    optimizer._pending = object()

    with pytest.raises(RuntimeError, match="pending optimizer update"):
        destination.load_state_dir(hytorch.load(checkpoint))


def test_missing_saved_workspace_does_not_partially_load(tmp_path):
    source = Model(width=2, bias="source")
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)
    os.unlink(checkpoint / "layers/layer/1/AGENTS.md")
    run_git(str(checkpoint), "add", "-A")
    run_git(str(checkpoint), "commit", "-m", "test: remove workspace")
    state = hytorch.load(checkpoint)
    destination = Model(width=2, bias="destination")
    before = [view.text() for view in destination.layer.weight.views()]

    with pytest.raises(RuntimeError, match="missing workspace"):
        destination.load_state_dir(state)

    assert [view.text() for view in destination.layer.weight.views()] == before


def test_copy_failure_does_not_change_canonical_model(tmp_path, monkeypatch):
    source = Model(width=2, bias="source")
    checkpoint = tmp_path / "source"
    hytorch.save(source.state_dir(), checkpoint)
    destination = Model(width=2, bias="destination")
    before = [view.text() for view in destination.layer.weight.views()]
    before_commit = destination.state_dir().commit
    original = state_dir_module.copy_tree
    copies = 0

    def fail_second_copy(source, destination):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("simulated copy failure")
        original(source, destination)

    monkeypatch.setattr(state_dir_module, "copy_tree", fail_second_copy)
    with pytest.raises(OSError, match="simulated copy failure"):
        destination.load_state_dir(hytorch.load(checkpoint))

    assert destination.state_dir().commit == before_commit
    assert [view.text() for view in destination.layer.weight.views()] == before
