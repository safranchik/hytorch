"""Lazy dependency scheduler used by Module.__call__."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from .harness import name_of
from .space import Space, SpaceBatch


@dataclasses.dataclass(frozen=True)
class DeferredState:
    """A symbolic Space produced while forward() declares the graph."""

    scheduler: Scheduler
    node: int

    def to(self, harness) -> DeferredState:
        """Declare a lossless harness-placement boundary for this result."""
        target = name_of(harness)
        return self.scheduler.add([self], lambda states: states[0].to(target))


@dataclasses.dataclass
class _Node:
    dependencies: tuple[Space | DeferredState, ...]
    run: Callable[[list[Space]], Space]


class Scheduler:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self._nodes: list[_Node] = []

    def add(
        self,
        dependencies: list[Space | DeferredState],
        run: Callable[[list[Space]], Space],
    ) -> DeferredState:
        for dependency in dependencies:
            if (
                isinstance(dependency, DeferredState)
                and dependency.scheduler is not self
            ):
                raise ValueError(
                    "hytorch: cannot combine deferred states from different graph runs"
                )
            if not isinstance(dependency, (Space, DeferredState)):
                raise TypeError("hytorch.Linear inputs must be hytorch.Space values")
        index = len(self._nodes)
        self._nodes.append(_Node(tuple(dependencies), run))
        return DeferredState(self, index)

    def materialize(self, value):
        if not self._nodes:
            return value

        dependents: dict[int, list[int]] = {i: [] for i in range(len(self._nodes))}
        remaining: dict[int, int] = {}
        for index, node in enumerate(self._nodes):
            ids = {d.node for d in node.dependencies if isinstance(d, DeferredState)}
            remaining[index] = len(ids)
            for dependency in ids:
                dependents[dependency].append(index)

        results: dict[int, Space] = {}
        running: dict[Future[Space], int] = {}
        workers = self.max_workers or max(1, len(self._nodes))

        def launch(executor: ThreadPoolExecutor, index: int) -> None:
            node = self._nodes[index]
            inputs = [
                results[d.node] if isinstance(d, DeferredState) else d
                for d in node.dependencies
            ]
            running[executor.submit(node.run, inputs)] = index

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, count in remaining.items():
                if count == 0:
                    launch(executor, index)

            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    index = running.pop(future)
                    results[index] = future.result()
                    for child in dependents[index]:
                        remaining[child] -= 1
                        if remaining[child] == 0:
                            launch(executor, child)

        if len(results) != len(self._nodes):
            raise RuntimeError("hytorch: computation graph contains a dependency cycle")
        return _replace_deferred(value, results, self)


def _replace_deferred(value, results: dict[int, Space], scheduler: Scheduler):
    if isinstance(value, DeferredState):
        if value.scheduler is not scheduler:
            raise ValueError("hytorch: forward returned a state from another graph run")
        return results[value.node]
    if isinstance(value, SpaceBatch):
        return SpaceBatch(_replace_deferred(item, results, scheduler) for item in value)
    if isinstance(value, list):
        return [_replace_deferred(item, results, scheduler) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_deferred(item, results, scheduler) for item in value)
    if isinstance(value, dict):
        return {
            key: _replace_deferred(item, results, scheduler)
            for key, item in value.items()
        }
    return value
