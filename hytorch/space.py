"""Git-backed statespace values flowing through HyTorch networks."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

from ._git import Repo
from .harness import name_of

if TYPE_CHECKING:
    from ._autofeed import Node
    from .harness import Harness


@dataclasses.dataclass(frozen=True, init=False)
class Space:
    """An immutable Git-backed agent statespace.

    The positional ``data`` is one directory, mirroring ``torch.tensor``'s
    positional data argument. A Space is one Git-backed directory value.
    """

    repo: Repo
    commit: str
    path: str
    #: The name of the branch backing this state's worktree, if one was
    #: created for it (empty for a plain input state).
    branch: str = ""
    #: The filesystem path of the worktree checked out for this state, if
    #: any (empty for a plain input state that was never materialized on
    #: disk).
    dir: str = ""
    #: Identifies which registered hytorch.mn.Linear produced this value.
    layer: str = ""
    #: The compact handoff text the producing agent left for its
    #: successors: why it chose what it chose. Never a full session
    #: transcript, never a binary artifact.
    summary: str = ""
    #: The runtime harness where this state is available. This names a
    #: registered Harness (for example ``"pi"``); it is not a Docker image
    #: or a live agent session.
    harness: str = ""
    #: Default model type inherited by modules consuming this statespace.
    mtype: str | None = None
    #: Whether backward feed is recorded for this value.
    requires_feed: bool = False
    #: Operation that produced this activation, analogous to Tensor.grad_fn.
    feed_fn: Node | None = None

    def __init__(
        self,
        data: str,
        *,
        mtype: str | None = None,
        harness: Harness | str | None = None,
        requires_feed: bool = False,
        repo: Repo | None = None,
        commit: str | None = None,
        branch: str = "",
        dir: str = "",
        layer: str = "",
        summary: str = "",
        feed_fn: Node | None = None,
    ) -> None:
        if not isinstance(data, str) or not data:
            raise TypeError("hytorch.Space: data must be one non-empty directory path")
        if repo is None:
            repo = Repo.discover(os.path.realpath(data))
        if commit is None:
            commit = repo.resolve("HEAD")
        if mtype is not None and (not isinstance(mtype, str) or not mtype.strip()):
            raise ValueError("hytorch.Space: mtype must be a non-empty string")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "commit", commit)
        object.__setattr__(
            self, "path", os.path.realpath(data) if os.path.exists(data) else data
        )
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "dir", dir)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self, "harness", name_of(harness) if harness is not None else ""
        )
        object.__setattr__(self, "mtype", mtype.strip() if mtype else None)
        object.__setattr__(self, "requires_feed", bool(requires_feed))
        object.__setattr__(self, "feed_fn", feed_fn)

    def to(
        self,
        harness: Harness | str | None = None,
        *,
        mtype: str | None = None,
    ) -> Space:
        """Place this unchanged activation on another agent harness.

        A Git commit is portable, so this is a lossless placement change, not
        a forward pass: it neither invokes an agent nor creates a commit.  A
        target worktree/container is materialized lazily by the next layer.
        """
        if harness is None and mtype is None:
            return self
        if mtype is not None and (not isinstance(mtype, str) or not mtype.strip()):
            raise ValueError("hytorch.Space.to: mtype must be a non-empty string")
        return Space(
            self.path,
            repo=self.repo,
            commit=self.commit,
            branch=self.branch,
            dir=self.dir,
            layer=self.layer,
            summary=self.summary,
            harness=harness if harness is not None else self.harness or None,
            mtype=mtype.strip() if mtype else self.mtype,
            requires_feed=self.requires_feed,
            feed_fn=self.feed_fn,
        )


class SpaceBatch(list):
    """A fixed-width vector of statespaces produced by one ``hytorch.mn.Linear`` call.

    It remains iterable and indexable like a Python list, while batch-wide
    operations such as ``.to(harness)`` mirror Tensor ergonomics.
    """

    def to(
        self,
        harness: Harness | str | None = None,
        *,
        mtype: str | None = None,
    ) -> SpaceBatch:
        return SpaceBatch(state.to(harness, mtype=mtype) for state in self)


def space(
    data: str | os.PathLike[str] | list[str | os.PathLike[str]],
    *,
    mtype: str | None = None,
    harness: Harness | str | None = None,
    requires_feed: bool = False,
) -> Space | SpaceBatch:
    """Create one Space or a SpaceBatch from one directory or a list of directories.

    Signature deliberately mirrors ``torch.tensor(data, *, dtype, device,
    requires_grad)`` with ``mtype``, ``harness``, and ``requires_feed``.
    """
    if isinstance(data, (str, os.PathLike)):
        return Space(
            os.fspath(data),
            mtype=mtype,
            harness=harness,
            requires_feed=requires_feed,
        )
    if isinstance(data, list):
        return SpaceBatch(
            Space(
                os.fspath(path),
                mtype=mtype,
                harness=harness,
                requires_feed=requires_feed,
            )
            for path in data
        )
    raise TypeError("hytorch.space: data must be a directory or list of directories")
