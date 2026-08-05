"""Private Git plumbing used by spaces, modules, and optimizers.

Ported from the Go implementation's internal/gitx package: branch/worktree
creation, merge, diff, blame, tag, and commit primitives, all implemented as
plain ``subprocess`` calls to the real ``git`` binary (no GitPython
dependency). This is not a general-purpose Git wrapper.
"""

from __future__ import annotations

import dataclasses
import io
import os
import subprocess
import tarfile


class GitError(RuntimeError):
    """A git command failed."""


class MergeConflictError(GitError):
    """A merge left conflict markers in the worktree.

    This is not a failure of the merge operation itself — resolving the
    listed files is expected to be the agent's own job, as part of its
    normal work (see AGENTS.md's forward-pass mechanics).
    """

    def __init__(self, files: list[str]):
        self.files = files
        super().__init__(
            f"merge left {len(files)} file(s) conflicted: {', '.join(files)}"
        )


def _hy_env() -> dict[str, str]:
    """A stable commit identity used for every commit/tag hytorch creates, so
    runs are reproducible regardless of the operator's local git identity.
    """
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="HyTorch",
        GIT_AUTHOR_EMAIL="hytorch@localhost",
        GIT_COMMITTER_NAME="HyTorch",
        GIT_COMMITTER_EMAIL="hytorch@localhost",
    )
    return env


def _run(
    cwd: str,
    args: list[str],
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> str:
    result = subprocess.run(
        ["git", "-C", cwd, *args],
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"git {' '.join(args)} failed"
        raise GitError(detail)
    return result.stdout.strip()


@dataclasses.dataclass
class Repo:
    """A Git repository that backs a HyTorch filesystem state."""

    root: str
    git_dir: str

    @classmethod
    def discover(cls, path: str) -> Repo:
        try:
            root = _run(path, ["rev-parse", "--show-toplevel"])
        except GitError as exc:
            raise GitError("not inside a git repository") from exc
        git_dir = _run(root, ["rev-parse", "--absolute-git-dir"])
        return cls(root=root, git_dir=git_dir)

    def resolve(self, ref: str) -> str:
        try:
            return _run(
                self.root,
                ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            )
        except GitError as exc:
            raise GitError(f"resolve git ref {ref!r}: {exc}") from exc

    def current_branch(self) -> str:
        branch = _run(self.root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
        if not branch:
            raise GitError("detached HEAD is not supported; check out a branch")
        return branch

    def worktree_head(self, directory: str) -> str:
        return _run(directory, ["rev-parse", "HEAD"])

    def export_tree(self, commit: str, destination: str) -> None:
        """Materialize one committed tree without its Git control directory."""
        os.makedirs(destination, exist_ok=True)
        result = subprocess.run(
            ["git", "-C", self.root, "archive", "--format=tar", commit],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise GitError(result.stderr.decode(errors="replace").strip())
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                normalized = os.path.normpath(member.name)
                if os.path.isabs(member.name) or normalized.startswith(".."):
                    raise GitError(f"unsafe path in Git tree: {member.name}")
            archive.extractall(destination)

    def is_clean(self) -> bool:
        status = _run(self.root, ["status", "--porcelain", "--untracked-files=normal"])
        return status == ""

    def branch(self, name: str, commit: str) -> None:
        try:
            _run(self.root, ["branch", name, commit])
        except GitError as exc:
            raise GitError(f"create branch {name}: {exc}") from exc

    @classmethod
    def create_integration(
        cls,
        destination: str,
        inputs: list[tuple[str, str]],
    ) -> tuple[Repo, str]:
        """Create a fresh node repository and fetch each independent input."""
        os.makedirs(destination)
        _run(destination, ["init", "-b", "hytorch-integration"])
        node = cls.discover(destination)
        _run(destination, ["config", "user.name", "HyTorch"])
        _run(destination, ["config", "user.email", "hytorch@localhost"])
        _run(
            destination,
            ["commit", "--allow-empty", "-m", "hytorch: begin node integration"],
            env=_hy_env(),
        )
        base = node.resolve("HEAD")
        for index, (source, commit) in enumerate(inputs):
            _run(
                destination,
                [
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    source,
                    f"{commit}:refs/hytorch/inputs/{index}",
                ],
            )
        return node, base

    def import_commit(self, source: str, commit: str, branch: str) -> str:
        """Import one self-contained node history under a canonical branch."""
        try:
            _run(
                self.root,
                [
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    source,
                    f"{commit}:refs/heads/{branch}",
                ],
            )
            return self.resolve(branch)
        except GitError as exc:
            raise GitError(f"import node commit {commit}: {exc}") from exc

    def commit_allow_empty(self, directory: str, message: str) -> str:
        _run(directory, ["add", "-A"])
        _run(directory, ["commit", "--allow-empty", "-m", message], env=_hy_env())
        return _run(directory, ["rev-parse", "HEAD"])

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = subprocess.run(
            [
                "git",
                "-C",
                self.root,
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise GitError(result.stderr.strip() or "git merge-base --is-ancestor failed")

    def delete_branch(self, name: str) -> None:
        try:
            _run(self.root, ["branch", "-D", name])
        except GitError as exc:
            raise GitError(f"delete branch {name}: {exc}") from exc

    def add_worktree(self, directory: str, branch: str) -> None:
        try:
            _run(self.root, ["worktree", "add", directory, branch])
        except GitError as exc:
            raise GitError(f"add worktree for {branch}: {exc}") from exc

    def remove_worktree(self, directory: str) -> None:
        try:
            _run(self.root, ["worktree", "remove", "--force", directory])
        except GitError as exc:
            raise GitError(f"remove worktree {directory}: {exc}") from exc

    def merge_branches(self, worktree_dir: str, branches: list[str]) -> None:
        """Merge every branch in branches into whatever branch is currently
        checked out in worktree_dir. If the merge leaves conflict markers,
        raises MergeConflictError listing the conflicted paths (the merge is
        intentionally left uncommitted so the caller/agent can resolve
        markers and commit as part of its own work).
        """
        if not branches:
            return
        result = subprocess.run(
            ["git", "-C", worktree_dir, "merge", "--no-edit", "--no-ff", *branches],
            env=_hy_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            conflicts = _run(worktree_dir, ["diff", "--name-only", "--diff-filter=U"])
            if conflicts:
                raise MergeConflictError(conflicts.split("\n"))
            raise GitError(f"merge {','.join(branches)}: {result.stderr.strip()}")

    def fast_forward(self, worktree_dir: str, commit: str) -> None:
        try:
            _run(worktree_dir, ["merge", "--ff-only", commit], env=_hy_env())
        except GitError as exc:
            raise GitError(f"fast-forward to {commit}: {exc}") from exc

    def worktree_status(self, directory: str) -> str:
        """Return porcelain status for one checked-out worktree."""
        return _run(directory, ["status", "--porcelain", "--untracked-files=normal"])

    def commit_all_workdir(self, directory: str, message: str) -> tuple[str, bool]:
        """Stage every change in directory (a checked-out worktree) and
        commit it, returning (new_commit, changed). If nothing changed,
        returns (current HEAD commit, False).
        """
        before = _run(directory, ["rev-parse", "HEAD"])
        _run(directory, ["add", "-A"])
        status = _run(directory, ["status", "--porcelain"])
        if status == "":
            return before, False
        _run(directory, ["commit", "-m", message], env=_hy_env())
        after = _run(directory, ["rev-parse", "HEAD"])
        return after, True

    def diff_commits(self, base: str, head: str) -> str:
        try:
            return _run(self.root, ["diff", base, head])
        except GitError as exc:
            raise GitError(f"diff {base}..{head}: {exc}") from exc

    def changed_paths(self, base: str, head: str) -> list[str]:
        """Return paths changed between two commits."""
        try:
            value = _run(self.root, ["diff", "--name-only", base, head])
        except GitError as exc:
            raise GitError(f"list changed paths {base}..{head}: {exc}") from exc
        return [path for path in value.split("\n") if path]

    def blame_line(self, ref: str, file: str, line: int) -> str:
        try:
            value = _run(
                self.root,
                ["blame", "--porcelain", "-L", f"{line},{line}", ref, "--", file],
            )
        except GitError as exc:
            raise GitError(f"blame {file}:{line} @ {ref}: {exc}") from exc
        fields = value.split()
        if not fields:
            raise GitError(f"blame {file}:{line} @ {ref}: no output")
        return fields[0]

    def blame_file(self, ref: str, file: str) -> list[str]:
        """For every line of file as of ref, the commit hash that last
        touched it. Index 0 of the result is line 1.
        """
        try:
            value = _run(self.root, ["blame", "--porcelain", ref, "--", file])
        except GitError as exc:
            raise GitError(f"blame {file} @ {ref}: {exc}") from exc
        commits = []
        for line in value.split("\n"):
            if not line:
                continue
            fields = line.split()
            # A porcelain header line starts with a 40-char hex hash
            # followed by three integers (orig-line, final-line,
            # num-lines-in-group).
            if len(fields) >= 4 and len(fields[0]) == 40 and _is_hex(fields[0]):
                try:
                    int(fields[1])
                except ValueError:
                    continue
                commits.append(fields[0])
        return commits

    def tag_commit(self, tag: str, commit: str) -> None:
        try:
            _run(self.root, ["tag", "-f", tag, commit], env=_hy_env())
        except GitError as exc:
            raise GitError(f"tag {tag}: {exc}") from exc

    def restore_path_from_ref(self, directory: str, ref: str, path: str) -> None:
        try:
            _run(directory, ["checkout", ref, "--", path])
        except GitError as exc:
            raise GitError(f"restore {path} from {ref}: {exc}") from exc

    def commit_message(self, commit: str) -> str:
        try:
            return _run(self.root, ["log", "-1", "--format=%B", commit])
        except GitError as exc:
            raise GitError(f"read commit message for {commit}: {exc}") from exc

    def parent(self, commit: str) -> str:
        try:
            return _run(self.root, ["rev-parse", f"{commit}^1"])
        except GitError:
            return ""

    def rev_list(self, commit: str) -> list[str]:
        """Return commits reachable from ``commit`` in topological order."""
        try:
            value = _run(self.root, ["rev-list", "--topo-order", commit])
        except GitError as exc:
            raise GitError(f"list ancestors of {commit}: {exc}") from exc
        return [item for item in value.splitlines() if item]


def _is_hex(s: str) -> bool:
    return all(c in "0123456789abcdef" for c in s)
