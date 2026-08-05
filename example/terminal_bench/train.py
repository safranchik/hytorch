"""Train a 1 -> 3 -> 1 HyTorch network on ten public Terminal-Bench tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import hytorch

from .benchmark import (
    EVAL_TASKS,
    TRAIN_TASKS,
    Score,
    Task,
    cleanup,
    ensure_suite,
    feedback,
    load_task,
    task_prompt,
    verify,
)

MODEL = "gpt-5.6-terra"
AGENT_IMAGE = "hytorch-terminal-bench-mvp:latest"


class TerminalBenchMLP(hytorch.mn.Module):
    """One input, three parallel workers, and one dense integrator."""

    def __init__(self) -> None:
        super().__init__()
        self.work = hytorch.mn.Linear(
            1,
            3,
            bias=(
                "Solve terminal tasks carefully. Inspect the complete statespace. "
                "Use tools to validate assumptions. Produce and commit a complete answer."
            ),
        )
        self.integrate = hytorch.mn.Linear(
            3,
            1,
            bias=(
                "Integrate all three candidate branches. Resolve conflicts. Inspect and "
                "test the merged result. Leave the strongest complete answer committed."
            ),
        )

    def forward(self, state: hytorch.Space, *, task: str) -> hytorch.Space:
        candidates = self.work(state, task=task)
        return self.integrate(*candidates, task=task)[0]


@dataclass(frozen=True)
class Run:
    task: Task
    output: hytorch.Space
    score: Score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--backward-tokens", type=int, default=2_000)
    parser.add_argument("--output-dir", default="example/terminal_bench/results")
    args = parser.parse_args()

    ensure_agent_image()
    os.environ["HYTORCH_PI_IMAGE"] = AGENT_IMAGE
    preparation_started = time.perf_counter()
    suite = ensure_suite()
    all_names = TRAIN_TASKS + EVAL_TASKS
    tasks = parallel_map(
        lambda name: load_task(name, suite), all_names, args.max_workers
    )
    by_name = {task.name: task for task in tasks}
    train_tasks = [by_name[name] for name in TRAIN_TASKS]
    eval_tasks = [by_name[name] for name in EVAL_TASKS]
    print(f"task_preparation_seconds={time.perf_counter() - preparation_started:.1f}")

    # An explicit provider keeps Codex-login auth selected even when an unrelated
    # OPENAI_API_KEY exists in the operator's shell.
    harness = hytorch.harness.PiHarness(provider="openai-codex")
    hytorch.manual_seed(42)
    model = TerminalBenchMLP().to(harness, mtype=MODEL)
    optimizer = hytorch.optim.DFM(
        model.parameters(), temp=0.4, max_tokens=args.backward_tokens
    )
    print(f"architecture=1->3->1 model={MODEL} train=5 eval=5")
    print("workspace_store=" + model._parameter_store.root)
    history: list[tuple[int, str, Score]] = []

    try:
        for generation in range(args.epochs + 1):
            forward_started = time.perf_counter()
            usage_before = harness.usage()
            measured = measure(
                model,
                harness,
                train_tasks,
                eval_tasks,
                args.max_workers,
                record_train=generation < args.epochs,
            )
            forward_usage = harness.usage() - usage_before
            print(
                f"generation={generation} forward_verify_seconds="
                f"{time.perf_counter() - forward_started:.1f} "
                + format_usage(forward_usage),
                flush=True,
            )
            runs = [run for split, run in measured if split == "train"]
            train_scores = [run.score for run in runs]
            eval_scores = [run.score for split, run in measured if split == "eval"]
            label = "baseline" if generation == 0 else f"epoch {generation}"
            report(label + " train", train_scores)
            report(label + " eval", eval_scores)
            history.extend((generation, split, run.score) for split, run in measured)
            write_results(args.output_dir, history)
            print(
                "plot="
                + os.path.realpath(os.path.join(args.output_dir, "learning_curve.png")),
                flush=True,
            )
            if generation == args.epochs:
                break
            # Forward work is concurrent. DFM promotion is ordered because every
            # task updates the same canonical workspace history.
            backward_started = time.perf_counter()
            usage_before = harness.usage()
            for run in runs:
                optimizer.zero_feed()
                hytorch.Loss(run.output, feedback=feedback(run.score)).backward()
                optimizer.step()
            backward_usage = harness.usage() - usage_before
            print(
                f"generation={generation + 1} backward_seconds="
                f"{time.perf_counter() - backward_started:.1f} "
                + format_usage(backward_usage),
                flush=True,
            )
    finally:
        for task in tasks:
            cleanup(task)


def measure(
    model: TerminalBenchMLP,
    harness: hytorch.harness.Harness,
    train_tasks: list[Task],
    eval_tasks: list[Task],
    workers: int,
    *,
    record_train: bool,
) -> list[tuple[str, Run]]:
    items = [("train", task) for task in train_tasks] + [
        ("eval", task) for task in eval_tasks
    ]
    return parallel_map(
        lambda item: (
            item[0],
            forward_and_score(
                model,
                harness,
                item[1],
                inference=item[0] == "eval" or not record_train,
            ),
        ),
        items,
        workers,
    )


def forward_and_score(
    model: TerminalBenchMLP,
    harness: hytorch.harness.Harness,
    task: Task,
    *,
    inference: bool,
) -> Run:
    state = hytorch.space(task.files, harness=harness)
    if inference:
        # Context variables do not cross ThreadPoolExecutor thread boundaries.
        # Enter inference mode in the worker that performs the forward pass.
        with hytorch.inference_mode():
            output = model(state, task=task_prompt(task))
    else:
        output = model(state, task=task_prompt(task))
    result = verify(task, output.dir)
    return Run(task, output, result)


def report(label: str, scores: list[Score]) -> None:
    mean = sum(score.reward for score in scores) / len(scores)
    detail = " ".join(f"{score.task}={score.reward:g}" for score in scores)
    print(f"{label}: score={mean:.0%} {detail}", flush=True)


def format_usage(usage: hytorch.harness.Usage) -> str:
    return (
        f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens} "
        f"cache_read_tokens={usage.cache_read_tokens} "
        f"cache_write_tokens={usage.cache_write_tokens}"
    )


def parallel_map(fn, values, workers: int):
    values = list(values)
    lock = threading.Lock()

    def run(value):
        result = fn(value)
        with lock:
            if isinstance(value, str):
                name = value
            elif isinstance(value, tuple):
                name = value[1].name
            else:
                name = value.name
            print(f"finished={name}", flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(values))
    ) as executor:
        return list(executor.map(run, values))


def write_results(output_dir: str, history: list[tuple[int, str, Score]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with Path(root, "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "split", "task", "reward"])
        for generation, split, score in history:
            writer.writerow([generation, split, score.task, score.reward])
    generations = sorted({generation for generation, _, _ in history})
    grouped: dict[tuple[int, str], list[Score]] = {}
    for generation, split, score in history:
        grouped.setdefault((generation, split), []).append(score)

    figure, axis = plt.subplots(figsize=(10, 6))
    for split in ("train", "eval"):
        available = [g for g in generations if (g, split) in grouped]
        means = [
            sum(score.reward for score in grouped[(g, split)])
            / len(grouped[(g, split)])
            for g in available
        ]
        axis.plot(available, means, marker="o", linewidth=3, label=f"{split} mean")

    eval_names = sorted({score.task for _, split, score in history if split == "eval"})
    for name in eval_names:
        available = [
            g
            for g in generations
            if any(score.task == name for score in grouped.get((g, "eval"), []))
        ]
        values = [
            next(score.reward for score in grouped[(g, "eval")] if score.task == name)
            for g in available
        ]
        axis.plot(available, values, linestyle="--", alpha=0.35, label=name)

    axis.set_title("HyTorch Terminal-Bench 2.1 MVP: 1→3→1 Luna")
    axis.set_xlabel("Optimizer generations")
    axis.set_ylabel("Official task score")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks(generations)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "learning_curve.png", dpi=160)
    plt.close(figure)


def ensure_agent_image() -> None:
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    subprocess.run(
        [
            "docker",
            "build",
            "--tag",
            AGENT_IMAGE,
            "--file",
            os.path.join(root, "example", "terminal_bench", "Dockerfile"),
            root,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
