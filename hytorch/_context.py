"""Execution-local state shared by Graphs and their registered modules."""

from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._scheduler import Scheduler
    from .harness import Harness


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Resources belonging to one graph invocation."""

    harnesses: Mapping[str, Harness]
    default_harness: str
    default_mtype: str | None
    scheduler: Scheduler
    inference: bool = False

    def harness_for(self, harness: str) -> Harness:
        try:
            return self.harnesses[harness]
        except KeyError as exc:
            available = ", ".join(sorted(self.harnesses)) or "(none)"
            raise RuntimeError(
                f"hytorch: harness {harness!r} is not registered for this run; "
                f"available harnesses: {available}"
            ) from exc


_run_context: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "hy_run_context", default=None
)


def current_run() -> RunContext:
    context = _run_context.get()
    if context is None:
        raise RuntimeError("hytorch: no active graph run; call model(space)")
    return context
