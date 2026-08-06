import os
import shutil

import pytest

from hytorch._native_view import native_node_view


def _node(root):
    (root / "statespace").mkdir(parents=True)
    (root / "parameter").mkdir()
    (root / "workspace").mkdir()
    return root


def test_native_view_keeps_one_working_path_across_materializations(tmp_path):
    first = _node(tmp_path / "first")
    with native_node_view(str(first)) as first_view:
        assert os.path.realpath(os.path.join(first_view, "statespace")) == str(
            (first / "statespace").resolve()
        )
        assert os.path.realpath(os.path.join(first_view, "parameter")) == str(
            (first / "parameter").resolve()
        )

    second = tmp_path / "second"
    (second / "statespace").mkdir(parents=True)
    (second / "parameter").mkdir()
    shutil.copytree(first / "workspace", second / "workspace")
    with native_node_view(str(second)) as second_view:
        assert second_view == first_view
        assert os.path.realpath(os.path.join(second_view, "workspace")) == str(
            (second / "workspace").resolve()
        )


def test_native_view_rejects_agent_changes_to_its_identity(tmp_path):
    root = _node(tmp_path / "node")

    with pytest.raises(RuntimeError, match="changed its native node identity"):
        with native_node_view(str(root)):
            (root / "workspace" / ".hytorch" / "node-id").write_text(
                "00000000000000000000000000000000\n"
            )
