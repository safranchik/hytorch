"""Dynamic forward graph retained for feed propagation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._git import Repo
    from .harness import Session
    from .parameter import ParameterView
    from .space import Space


@dataclasses.dataclass(eq=False)
class Node:
    """One executed agent operation and the context needed to train it."""

    layer: str
    agent: int
    harness: str
    mtype: str | None
    session: Session
    parameters: tuple[ParameterView, ...]
    inputs: tuple[Space, ...]
    parents: tuple[Node, ...]
    repo: Repo
    commit: str
    root: str
    workspace: str
    parameter: str
    statespace: str
    workspace_revision: str
    consumed: bool = False
    released: bool = False
    applied: bool = False
    feed: tuple[str, ...] = ()


def ancestors(outputs: list[Node]) -> list[Node]:
    """Return unique nodes in reverse execution order."""
    ordered: list[Node] = []
    seen: set[int] = set()

    def visit(node: Node) -> None:
        identity = id(node)
        if identity in seen:
            return
        seen.add(identity)
        ordered.append(node)
        for parent in node.parents:
            visit(parent)

    for output in outputs:
        visit(output)
    return ordered
