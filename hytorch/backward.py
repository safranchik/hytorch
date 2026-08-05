"""Directional feedback and workspace-update reports."""

from __future__ import annotations

import dataclasses

from ._autofeed import ancestors
from .space import Space


@dataclasses.dataclass
class Loss:
    """One directional feedback signal for an output statespace."""

    output: Space
    feedback: str

    def __post_init__(self) -> None:
        if not isinstance(self.output, Space):
            raise TypeError("hytorch.Loss output must be one hytorch.Space")
        if not isinstance(self.feedback, str) or not self.feedback.strip():
            raise ValueError("hytorch.Loss feedback must be non-empty text")
        self.feedback = self.feedback.strip()

    def backward(self) -> None:
        """Update candidate workspaces and propagate feedback through the graph."""
        if self.output.feed_fn is None:
            raise RuntimeError("hytorch.Loss output has no executed graph to traverse")
        nodes = ancestors([self.output.feed_fn])
        for node in nodes:
            if node.released or node.consumed:
                raise RuntimeError(
                    "Trying to backward through the graph a second time. "
                    "Run a new forward pass first."
                )
        optimizers = {
            parameter.parameter._optimizer
            for node in nodes
            for parameter in node.parameters
            if parameter.parameter._optimizer is not None
        }
        if not optimizers:
            raise RuntimeError(
                "hytorch.Loss.backward requires an optimizer for the model parameters"
            )
        if len(optimizers) != 1:
            raise RuntimeError(
                "hytorch.Loss.backward requires one optimizer for the executed graph"
            )
        optimizer = next(iter(optimizers))
        optimizer._backward(self.output, self.feedback)


@dataclasses.dataclass(frozen=True)
class WorkspaceRevision:
    before: str
    after: str


@dataclasses.dataclass
class Report:
    """Workspace mutations promoted by one optimizer step."""

    revised: dict[str, WorkspaceRevision] = dataclasses.field(default_factory=dict)
    commits: list[str] = dataclasses.field(default_factory=list)


__all__ = ["Loss", "Report", "WorkspaceRevision"]
