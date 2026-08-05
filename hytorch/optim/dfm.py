"""Directional Feedback Mutation over candidate workspace branches."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from .._autofeed import Node, ancestors
from .._git import Repo
from ..backward import Report, WorkspaceRevision
from ..harness import registered
from ..parameter import (
    Parameter,
    ParameterStore,
    create_workspace_checkout,
    set_tree_writable,
    tree_manifest,
)
from ..space import Space
from .optimizer import Optimizer, _release


@dataclasses.dataclass
class _PendingUpdate:
    store: ParameterStore
    branch: str
    root: str
    base: str
    before: dict[str, dict[str, bytes]]


@dataclasses.dataclass
class _NodeUpdate:
    context: Node
    upstream: list[str]
    workspace_base: str
    workspace_head: str
    workspace_path: str


class DFM(Optimizer):
    """Generate workspace candidates during backward and promote them in step."""

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        temp: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(params)
        if not isinstance(temp, (int, float)) or isinstance(temp, bool) or temp < 0:
            raise ValueError("hytorch.optim.DFM: temp must be a non-negative number")
        if max_tokens is not None and (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ValueError(
                "hytorch.optim.DFM: max_tokens must be a positive integer or None"
            )
        self.temp = float(temp)
        self.max_tokens = max_tokens
        self._pending: _PendingUpdate | None = None

    def _backward(self, output: Space, feedback: str) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "hytorch.optim.DFM: call step() or zero_feed() before another backward pass"
            )
        nodes = ancestors([output.feed_fn])
        views = [view for node in nodes for view in node.parameters]
        if not views:
            raise RuntimeError(
                "hytorch.optim.DFM: executed graph has no trainable workspace"
            )
        stores = {id(view.parameter._store): view.parameter._store for view in views}
        if len(stores) != 1 or None in stores.values():
            raise RuntimeError(
                "hytorch.optim.DFM: one backward pass requires one model workspace store"
            )
        if any(view.parameter not in self.params for view in views):
            raise RuntimeError(
                "hytorch.optim.DFM: optimizer does not own every executed Parameter"
            )

        store = next(iter(stores.values()))
        base = store.repo.resolve("HEAD")
        branch = f"hytorch/dfm/{time.time_ns()}"
        candidate_root = tempfile.mkdtemp(prefix="hytorch-dfm-")
        store.repo.branch(branch, base)
        store.repo.add_worktree(candidate_root, branch)
        before = {
            view.relative_path: tree_manifest(
                os.path.join(store.root, view.relative_path)
            )
            for parameter in self.params
            for view in parameter.views()
        }
        pending = _PendingUpdate(store, branch, candidate_root, base, before)

        try:
            self._run_backward(nodes, output.feed_fn, feedback, pending)
        except Exception:
            for node in nodes:
                if not node.released:
                    _release(node)
            self._remove_pending(pending)
            raise
        self._pending = pending

    def _run_backward(
        self,
        nodes: list[Node],
        output_node: Node,
        feedback: str,
        pending: _PendingUpdate,
    ) -> None:
        children: dict[Node, set[Node]] = {node: set() for node in nodes}
        for node in nodes:
            for parent in node.parents:
                if parent in children:
                    children[parent].add(node)
        remaining = {node: len(values) for node, values in children.items()}
        queue = [node for node in nodes if remaining[node] == 0]
        downstream: dict[Node, list[str]] = defaultdict(list)
        downstream[output_node].append(feedback)
        processed = 0

        while queue:
            ready = queue
            queue = []
            while ready:
                selected: list[Node] = []
                deferred: list[Node] = []
                workspace_paths: set[str] = set()
                for node in ready:
                    if len(node.parameters) != 1:
                        raise RuntimeError(
                            "hytorch.optim.DFM: one agent must own one workspace"
                        )
                    path = node.parameters[0].relative_path
                    if path in workspace_paths:
                        deferred.append(node)
                    else:
                        workspace_paths.add(path)
                        selected.append(node)

                workspace_base = pending.store.repo.resolve(pending.branch)
                for node in selected:
                    if not downstream[node]:
                        raise RuntimeError(
                            f"hytorch.optim.DFM: node {node.layer}[{node.agent}] "
                            "received no feedback"
                        )
                with ThreadPoolExecutor(max_workers=len(selected)) as executor:
                    futures = [
                        executor.submit(
                            self._update_node,
                            node,
                            downstream[node],
                            pending,
                            workspace_base,
                        )
                        for node in selected
                    ]
                    updates = [future.result() for future in futures]
                self._integrate_updates(updates, pending)

                for update in updates:
                    node = update.context
                    processed += 1
                    for input_index, value in enumerate(node.inputs):
                        parent = value.feed_fn
                        if parent is not None and parent in remaining:
                            downstream[parent].append(update.upstream[input_index])
                    for parent in node.parents:
                        if parent not in remaining:
                            continue
                        remaining[parent] -= 1
                        if remaining[parent] == 0:
                            queue.append(parent)
                ready = deferred

        if processed != len(nodes):
            raise RuntimeError("hytorch.optim.DFM: backward graph contains a cycle")

    def _update_node(
        self,
        context: Node,
        messages: list[str],
        pending: _PendingUpdate,
        workspace_base: str,
    ) -> _NodeUpdate:
        if len(context.parameters) != 1:
            raise RuntimeError("hytorch.optim.DFM: one agent must own one workspace")
        view = context.parameters[0]
        set_tree_writable(context.workspace, True)
        workspace_repo = create_workspace_checkout(
            pending.store.root,
            workspace_base,
            view.relative_path,
            context.workspace,
        )
        set_tree_writable(context.statespace, False)
        statespace_repo = Repo.discover(context.statespace)
        statespace_head = statespace_repo.resolve("HEAD")
        statespace_before = tree_manifest(context.statespace)

        prompt = _backward_prompt(
            context.layer,
            context.agent,
            len(context.inputs),
            messages,
            view.relative_path,
            self.temp,
        )
        try:
            try:
                harness = registered()[context.harness]
            except KeyError as exc:
                raise RuntimeError(
                    f"hytorch.optim.DFM: harness {context.harness!r} is not registered"
                ) from exc
            summary = harness.resume(
                context.session,
                context.root,
                prompt,
                context.mtype,
                temperature=self.temp,
                max_tokens=self.max_tokens,
                read_only=(context.statespace,),
            )
            if tree_manifest(context.statespace) != statespace_before:
                raise ValueError(
                    "hytorch.optim.DFM: backward modified the read-only statespace"
                )
            if statespace_repo.resolve("HEAD") != statespace_head:
                raise ValueError(
                    "hytorch.optim.DFM: backward changed statespace Git history"
                )
            if not statespace_repo.is_clean():
                raise ValueError("hytorch.optim.DFM: backward left statespace changes")
            _validate_node_root(context.root)
            if not workspace_repo.is_clean():
                raise RuntimeError(
                    f"hytorch.optim.DFM: node {context.layer}[{context.agent}] "
                    "finished with uncommitted workspace changes"
                )
            upstream = _read_upstream(summary, len(context.inputs))
            workspace_head = workspace_repo.resolve("HEAD")
            if not workspace_repo.is_ancestor(workspace_base, workspace_head):
                raise RuntimeError(
                    f"hytorch.optim.DFM: node {context.layer}[{context.agent}] "
                    "rewrote global workspace history"
                )
            changed_paths = set(
                workspace_repo.changed_paths(workspace_base, workspace_head)
            )
            prefix = view.relative_path
            illegal = sorted(
                path
                for path in changed_paths
                if path != prefix and not path.startswith(prefix + "/")
            )
            if illegal:
                raise ValueError(
                    "hytorch.optim.DFM: backward changed paths outside its workspace: "
                    + ", ".join(illegal)
                )
            for message in messages:
                view._accumulate_feed(message)
            context.feed = tuple(messages)
            context.applied = True
            context.consumed = True
            return _NodeUpdate(
                context=context,
                upstream=upstream,
                workspace_base=workspace_base,
                workspace_head=workspace_head,
                workspace_path=view.relative_path,
            )
        finally:
            _release(context)

    def _integrate_updates(
        self,
        updates: list[_NodeUpdate],
        pending: _PendingUpdate,
    ) -> None:
        for update in updates:
            if update.workspace_head == update.workspace_base:
                continue
            imported_branch = f"hytorch/import/{time.time_ns()}-{update.context.agent}"
            imported = pending.store.repo.import_commit(
                update.context.workspace,
                update.workspace_head,
                imported_branch,
            )
            try:
                current = pending.store.repo.resolve(pending.branch)
                if current == update.workspace_base:
                    pending.store.repo.fast_forward(pending.root, imported)
                else:
                    pending.store.repo.merge_branches(pending.root, [imported])
            finally:
                pending.store.repo.delete_branch(imported_branch)

    def step(self) -> None:
        """Promote the complete candidate workspace branch."""
        pending = self._pending
        if pending is None:
            return None
        report = Report()
        try:
            if pending.store.repo.resolve("HEAD") != pending.base:
                raise RuntimeError(
                    "hytorch.optim.DFM: canonical model changed after backward; "
                    "discarding the stale candidate"
                )
            candidate = pending.store.repo.resolve(pending.branch)
            if candidate != pending.base:
                pending.store.repo.merge_branches(pending.store.root, [candidate])
                canonical = pending.store.repo.resolve("HEAD")
                report.commits.append(canonical)
            for relative, before in pending.before.items():
                after = tree_manifest(os.path.join(pending.store.root, relative))
                for path in sorted(set(before) | set(after)):
                    if before.get(path) == after.get(path):
                        continue
                    report.revised[os.path.join(relative, path)] = WorkspaceRevision(
                        before=_as_text(before.get(path)),
                        after=_as_text(after.get(path)),
                    )
            self.state["last_report"] = report
        finally:
            self._remove_pending(pending)
            self._pending = None
        return None

    def _discard_pending(self) -> None:
        if self._pending is None:
            return
        self._remove_pending(self._pending)
        self._pending = None

    @staticmethod
    def _remove_pending(pending: _PendingUpdate) -> None:
        if os.path.isdir(pending.root):
            pending.store.repo.remove_worktree(pending.root)
        try:
            pending.store.repo.delete_branch(pending.branch)
        except Exception:
            pass


def _backward_prompt(
    layer: str,
    agent: int,
    inputs: int,
    messages: list[str],
    workspace_path: str,
    temp: float,
) -> str:
    rendered = "\n\n".join(
        f"## Direction {index}\n\n{message}" for index, message in enumerate(messages)
    )
    return (
        f"Resume node {layer}[{agent}] for backward.\n\n"
        "The node root contains two self-contained Git states:\n"
        "- statespace/: read-only forward result and input refs\n"
        "- workspace/: writable sparse checkout with full model history\n\n"
        f"Your writable workspace is workspace/{workspace_path}/. "
        "Do not change any other model path.\n\n"
        f"# Directional feedback\n\n{rendered}\n\n"
        "Use the feedback to improve workspace/. You can leave the workspace "
        "unchanged. Commit all workspace changes before ending your turn and leave "
        "the repository clean. Each commit becomes part of the global model history. "
        "HyTorch combines commits from dependency-ready agents and step() promotes "
        "the completed candidate. Do not modify statespace/.\n\n"
        "Return only one JSON object with this shape:\n"
        f'{{"feedback": [<one imperative direction for each of the {inputs} input refs>]}}\n\n'
        f"Mutation temperature: {temp}. Use this as the semantic scale of the change.\n"
        "Finish the workspace commits and JSON response before ending your turn."
    )


def _read_upstream(text: str, count: int) -> list[str]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "hytorch.optim.DFM: backward response must be one JSON object"
        ) from exc
    if not isinstance(value, dict) or set(value) != {"feedback"}:
        raise RuntimeError(
            "hytorch.optim.DFM: backward response must contain only 'feedback'"
        )
    feedback = value["feedback"]
    if not isinstance(feedback, list) or len(feedback) != count:
        raise RuntimeError(
            f"hytorch.optim.DFM: backward response requires {count} feedback strings"
        )
    if any(not isinstance(item, str) or not item.strip() for item in feedback):
        raise RuntimeError(
            "hytorch.optim.DFM: every upstream feedback value must be non-empty text"
        )
    return [item.strip() for item in feedback]


def _validate_node_root(root: str) -> None:
    unexpected = sorted(set(os.listdir(root)) - {"statespace", "workspace"})
    if unexpected:
        raise ValueError(
            "hytorch.optim.DFM: backward changed paths outside the workspace: "
            + ", ".join(unexpected)
        )


def _as_text(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace")


__all__ = ["DFM"]
