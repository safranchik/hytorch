"""Official Terminal-Bench 2.1 task preparation and verification."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

REPOSITORY = "https://github.com/harbor-framework/terminal-bench-2-1.git"
REVISION = "5c8eadf1f393183288fa08b8f73ca9a469cc5e00"

# These tasks produce files that a statespace commit can preserve. They do not
# require a nested Git object database, a background service, or a system-wide
# package installation as part of the submitted answer.
TRAIN_TASKS = (
    "regex-log",
    "cancel-async-tasks",
    "log-summary-date-ranges",
    "constraints-scheduling",
    "openssl-selfsigned-cert",
)
EVAL_TASKS = (
    "cobol-modernization",
    "extract-elf",
    "polyglot-c-py",
    "gcode-to-text",
    "raman-fitting",
)


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    image: str
    files: str
    tests: str


@dataclasses.dataclass(frozen=True)
class Score:
    task: str
    reward: float
    output: str


def ensure_suite(cache: str | None = None) -> str:
    """Return a pinned checkout of the public Terminal-Bench 2.1 tasks."""
    root = os.path.realpath(
        cache or os.path.join(Path.home(), ".cache", "hytorch", "terminal-bench-2-1")
    )
    if not os.path.isdir(os.path.join(root, ".git")):
        os.makedirs(os.path.dirname(root), exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                REPOSITORY,
                root,
            ]
        )
    _run(["git", "-C", root, "fetch", "--quiet", "origin", REVISION])
    _run(["git", "-C", root, "checkout", "--quiet", "--detach", REVISION])
    return root


def load_task(name: str, suite: str) -> Task:
    """Extract one public task and copy its initial /app filesystem."""
    package = tempfile.mkdtemp(prefix=f"hytorch-tbench-{name}-package-")
    archive = os.path.join(package, "task.tar")
    with open(archive, "wb") as file:
        _run(
            [
                "git",
                "-C",
                suite,
                "archive",
                "--format=tar",
                REVISION,
                f"tasks/{name}",
            ],
            stdout=file,
        )
    with tarfile.open(archive) as value:
        value.extractall(package, filter="data")
    task_root = os.path.join(package, "tasks", name)
    instruction = Path(task_root, "instruction.md").read_text(encoding="utf-8")
    image = _docker_image(Path(task_root, "task.toml").read_text(encoding="utf-8"))
    _run(["docker", "pull", "--quiet", image])

    container = _capture(["docker", "create", image]).strip()
    files = tempfile.mkdtemp(prefix=f"hytorch-tbench-{name}-state-")
    try:
        _run(["docker", "cp", f"{container}:/app/.", files])
    finally:
        _run(["docker", "rm", "--force", container], check=False)
    Path(files, "TASK.md").write_text(instruction.rstrip() + "\n", encoding="utf-8")
    _git_init(files)
    return Task(
        name=name,
        instruction=instruction.strip(),
        image=image,
        files=files,
        tests=os.path.join(task_root, "tests"),
    )


def verify(task: Task, candidate: str) -> Score:
    """Run the task's official verifier against one committed candidate tree."""
    clean = tempfile.mkdtemp(prefix=f"hytorch-tbench-{task.name}-candidate-")
    logs = tempfile.mkdtemp(prefix=f"hytorch-tbench-{task.name}-logs-")
    container = "hytorch-tbench-" + uuid.uuid4().hex
    try:
        # Forward returns the complete committed statespace. Copy its working
        # tree, rather than using git archive, because Git only records the
        # executable bit and would erase required modes such as 0600.
        shutil.copytree(
            candidate,
            clean,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        _run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                container,
                "--workdir",
                "/app",
                "--mount",
                f"type=bind,src={os.path.realpath(task.tests)},dst=/tests,readonly",
                "--mount",
                f"type=bind,src={logs},dst=/logs/verifier",
                "--entrypoint",
                "sh",
                task.image,
                "-c",
                "while :; do sleep 3600; done",
            ]
        )
        _run(["docker", "cp", f"{clean}/.", f"{container}:/app/"])
        result = subprocess.run(
            ["docker", "exec", container, "bash", "/tests/test.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
        reward_path = os.path.join(logs, "reward.txt")
        reward = (
            float(Path(reward_path).read_text(encoding="utf-8").strip())
            if os.path.isfile(reward_path)
            else 0.0
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return Score(task.name, reward, output[-12_000:])
    finally:
        _run(["docker", "rm", "--force", container], check=False)
        shutil.rmtree(clean, ignore_errors=True)
        shutil.rmtree(logs, ignore_errors=True)


def task_prompt(task: Task) -> str:
    return (
        "Solve this public Terminal-Bench 2.1 task. The original /app directory "
        "is statespace/ in this node. Work only in statespace/. Inspect and test "
        "the files there. Preserve TASK.md. Commit all changes before you finish.\n\n"
        + task.instruction
    )


def feedback(score: Score) -> str:
    if score.reward >= 1:
        return (
            f"Keep the successful behavior used for {score.task}. Improve the reusable "
            "workflow: inspect inputs first, implement the smallest complete artifact, "
            "test it when possible, and check every requirement before finishing."
        )
    detail = score.output or "The official verifier produced no diagnostic output."
    return (
        f"Improve the workflow after failing public Terminal-Bench task {score.task}. "
        "Use the verifier diagnostics below to identify the missed requirements. Add "
        "general procedures or tools to workspace/ that help on future tasks. Do not "
        "memorize a task-specific final answer.\n\n" + detail
    )


def cleanup(task: Task) -> None:
    shutil.rmtree(task.files, ignore_errors=True)
    package = Path(task.tests).parents[2]
    if package.name.startswith(f"hytorch-tbench-{task.name}-package-"):
        shutil.rmtree(package, ignore_errors=True)


def _docker_image(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("docker_image"):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError("task.toml has no environment.docker_image")


def _git_init(root: str) -> None:
    _run(["git", "-C", root, "init", "--quiet", "--initial-branch=main"])
    _run(["git", "-C", root, "config", "user.name", "HyTorch"])
    _run(["git", "-C", root, "config", "user.email", "hytorch@localhost"])
    _run(["git", "-C", root, "add", "-A"])
    _run(["git", "-C", root, "commit", "--quiet", "-m", "terminal-bench task input"])


def _capture(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _run(
    args: list[str], *, check: bool = True, stdout=None
) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, stdout=stdout, stderr=subprocess.PIPE)


__all__ = [
    "EVAL_TASKS",
    "REVISION",
    "Score",
    "TRAIN_TASKS",
    "Task",
    "cleanup",
    "ensure_suite",
    "feedback",
    "load_task",
    "task_prompt",
    "verify",
]
