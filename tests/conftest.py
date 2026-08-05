import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import hytorch


def run_git(root: str, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args}: {result.stderr}")
    return result.stdout.strip()


def merge_agent_inputs(directory: str) -> None:
    refs = run_git(
        directory,
        "for-each-ref",
        "--sort=refname",
        "--format=%(refname)",
        "refs/hytorch/inputs",
    ).splitlines()
    for ref in refs:
        try:
            run_git(
                directory,
                "merge",
                "--no-ff",
                "--allow-unrelated-histories",
                "-m",
                f"agent: merge {ref}",
                ref,
            )
        except RuntimeError:
            conflicts = run_git(
                directory, "diff", "--name-only", "--diff-filter=U"
            ).splitlines()
            if not conflicts:
                raise
            for path in conflicts:
                run_git(directory, "checkout", "--theirs", "--", path)
            run_git(directory, "add", "-A")
            run_git(directory, "commit", "-m", f"agent: resolve {ref}")


def commit_agent_changes(directory: str, message: str) -> None:
    run_git(directory, "add", "-A")
    if run_git(directory, "status", "--porcelain"):
        run_git(directory, "commit", "-m", message)


def find_agent_workspace(directory: str) -> str:
    for current, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if name != ".git"]
        if "AGENTS.md" in files:
            return current
    raise RuntimeError("agent workspace has no AGENTS.md")


@pytest.fixture
def new_repo(tmp_path_factory) -> "hytorch.Repo":
    root = str(tmp_path_factory.mktemp("repo"))
    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.name", "Test")
    run_git(root, "config", "user.email", "test@example.com")
    with open(os.path.join(root, "base.txt"), "w") as f:
        f.write("base\n")
    with open(os.path.join(root, "protected.txt"), "w") as f:
        f.write("protected\n")
    with open(os.path.join(root, "output.txt"), "w") as f:
        f.write("")
    run_git(root, "add", ".")
    run_git(root, "commit", "-m", "base")
    return hytorch.Repo.discover(root)
