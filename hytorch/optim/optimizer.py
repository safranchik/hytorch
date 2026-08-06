"""Base optimizer for HyTorch Parameters."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable

from ..harness import registered
from ..parameter import Parameter, set_tree_writable


class Optimizer:
    """Base class for optimizers over registered workspace Parameters."""

    def __init__(self, params: Iterable[Parameter]) -> None:
        self.params = tuple(params)
        if not self.params:
            raise ValueError("hytorch.optim.Optimizer: params must not be empty")
        if any(not isinstance(parameter, Parameter) for parameter in self.params):
            raise TypeError(
                "hytorch.optim.Optimizer: params must be hytorch.mn.Parameter values"
            )
        for parameter in self.params:
            current = parameter._optimizer
            if current is not None and current is not self:
                raise RuntimeError(
                    "hytorch.optim.Optimizer: a Parameter already belongs to another optimizer"
                )
            parameter._optimizer = self
        self.state: dict = {}

    def zero_feed(self) -> None:
        """Clear accumulated feedback and discard an unpromoted update."""
        self._discard_pending()
        for parameter in self.params:
            parameter.zero_feed()

    def _backward(self, output, feedback: str, *, retain_graph: bool = False) -> None:
        raise NotImplementedError

    def _discard_pending(self) -> None:
        return None


def _release(context) -> None:
    if context.released:
        return
    try:
        harness = registered()[context.harness]
    except KeyError as exc:
        raise RuntimeError(
            f"hytorch.optim: harness {context.harness!r} is not registered"
        ) from exc
    try:
        harness.close(context.session)
    finally:
        for path in (context.workspace, context.parameter):
            if os.path.isdir(path):
                set_tree_writable(path, True)
                shutil.rmtree(path, ignore_errors=True)
        context.released = True


__all__ = ["Optimizer"]
