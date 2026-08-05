import os

import pytest

from hytorch._git import MergeConflictError


def test_branch_and_worktree_lifecycle(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")

    new_repo.branch("feature", base)
    wt = str(tmp_path / "wt")
    new_repo.add_worktree(wt, "feature")
    assert os.path.exists(os.path.join(wt, "base.txt"))

    with open(os.path.join(wt, "new.txt"), "w") as f:
        f.write("hi\n")
    commit, changed = new_repo.commit_all_workdir(wt, "add new.txt")
    assert changed and commit != base

    commit2, changed2 = new_repo.commit_all_workdir(wt, "no-op")
    assert not changed2 and commit2 == commit

    new_repo.remove_worktree(wt)
    assert not os.path.exists(wt)


def test_merge_branches_clean_and_conflict(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")
    new_repo.branch("a", base)
    new_repo.branch("b", base)

    dir_a = str(tmp_path / "a")
    new_repo.add_worktree(dir_a, "a")
    with open(os.path.join(dir_a, "a.txt"), "w") as f:
        f.write("from a\n")
    new_repo.commit_all_workdir(dir_a, "a change")

    dir_b = str(tmp_path / "b")
    new_repo.add_worktree(dir_b, "b")
    with open(os.path.join(dir_b, "b.txt"), "w") as f:
        f.write("from b\n")
    new_repo.commit_all_workdir(dir_b, "b change")

    new_repo.branch("merged", base)
    dir_merged = str(tmp_path / "merged")
    new_repo.add_worktree(dir_merged, "merged")
    new_repo.merge_branches(dir_merged, ["a", "b"])
    assert os.path.exists(os.path.join(dir_merged, "a.txt"))
    assert os.path.exists(os.path.join(dir_merged, "b.txt"))

    new_repo.branch("c", base)
    dir_c = str(tmp_path / "c")
    new_repo.add_worktree(dir_c, "c")
    with open(os.path.join(dir_c, "base.txt"), "w") as f:
        f.write("c version\n")
    new_repo.commit_all_workdir(dir_c, "c edits base")

    new_repo.branch("d", base)
    dir_d = str(tmp_path / "d")
    new_repo.add_worktree(dir_d, "d")
    with open(os.path.join(dir_d, "base.txt"), "w") as f:
        f.write("d version\n")
    new_repo.commit_all_workdir(dir_d, "d edits base")

    new_repo.branch("conflict-target", base)
    dir_conflict = str(tmp_path / "conflict")
    new_repo.add_worktree(dir_conflict, "conflict-target")
    with pytest.raises(MergeConflictError) as exc_info:
        new_repo.merge_branches(dir_conflict, ["c", "d"])
    assert exc_info.value.files == ["base.txt"]


def test_diff_commits(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")
    wt = str(tmp_path / "wt")
    new_repo.branch("main2", base)
    new_repo.add_worktree(wt, "main2")
    with open(os.path.join(wt, "base.txt"), "w") as f:
        f.write("changed\n")
    child, _ = new_repo.commit_all_workdir(wt, "change base.txt")
    diff = new_repo.diff_commits(base, child)
    assert "-base" in diff and "+changed" in diff


def test_blame_line_and_file(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")
    wt = str(tmp_path / "wt")
    new_repo.branch("blame-branch", base)
    new_repo.add_worktree(wt, "blame-branch")
    with open(os.path.join(wt, "multi.txt"), "w") as f:
        f.write("line one\nline two\n")
    first, _ = new_repo.commit_all_workdir(wt, "add multi.txt")

    with open(os.path.join(wt, "multi.txt"), "w") as f:
        f.write("line one\nline TWO edited\n")
    second, _ = new_repo.commit_all_workdir(wt, "edit line two")

    assert new_repo.blame_line(second, "multi.txt", 1) == first
    assert new_repo.blame_line(second, "multi.txt", 2) == second

    all_commits = new_repo.blame_file(second, "multi.txt")
    assert all_commits == [first, second]


def test_tag_commit_and_restore_path_from_ref(new_repo, tmp_path):
    base = new_repo.resolve("HEAD")
    wt = str(tmp_path / "wt")
    new_repo.branch("tag-branch", base)
    new_repo.add_worktree(wt, "tag-branch")

    new_repo.tag_commit("baseline", base)
    assert new_repo.resolve("baseline") == base

    with open(os.path.join(wt, "base.txt"), "w") as f:
        f.write("mutated\n")
    new_repo.commit_all_workdir(wt, "mutate base.txt")

    new_repo.restore_path_from_ref(wt, "baseline", "base.txt")
    with open(os.path.join(wt, "base.txt")) as f:
        assert f.read() == "base\n"
