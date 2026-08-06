"""Directional Feedback Mutation for persistent agent Parameters."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
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
    copy_tree,
    set_tree_writable,
    tree_manifest,
    validate_agent_state,
)
from ..space import Space
from .optimizer import Optimizer, _release


@dataclasses.dataclass(frozen=True)
class FeedRecord:
    """One reproducible mutation direction for an owner Parameter."""

    text: str
    layer: str
    agent: int
    session: str
    output_commit: str
    input_commits: tuple[str, ...]
    downstream: tuple[str, ...]
    harness: str
    mtype: str | None
    digest: str


@dataclasses.dataclass
class _NodeUpdate:
    context: Node
    update: str
    upstream: list[str]
    downstream: tuple[str, ...]


@dataclasses.dataclass
class _Candidate:
    store: ParameterStore
    branch: str
    root: str
    base: str


class DFM(Optimizer):
    """Accumulate directional feed and update each Parameter once in step()."""

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
        self._records: dict[tuple[int, str], list[FeedRecord]] = defaultdict(list)
        self._views = {}
        self._pending: _Candidate | None = None

    def _backward(
        self,
        output: Space,
        feedback: str,
        *,
        retain_graph: bool = False,
    ) -> None:
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
        if self._pending is not None:
            raise RuntimeError(
                "hytorch.optim.DFM: an earlier step candidate has not been resolved"
            )

        try:
            updates = self._run_backward(nodes, output.feed_fn, feedback)
            for value in updates:
                view = value.context.parameters[0]
                if not view.parameter.requires_feed:
                    value.context.feed = value.downstream
                    value.context.applied = True
                    continue
                record = _feed_record(value)
                key = (id(view.parameter._store), view.relative_path)
                self._records[key].append(record)
                self._views[key] = view
                view._accumulate_feed(value.update)
                value.context.feed = value.downstream
                value.context.applied = True
            if not retain_graph:
                for node in nodes:
                    node.consumed = True
                    if not node.released:
                        _release(node)
        except Exception:
            for node in nodes:
                node.consumed = True
                if not node.released:
                    _release(node)
            raise

    def _run_backward(
        self, nodes: list[Node], output_node: Node, feedback: str
    ) -> list[_NodeUpdate]:
        children: dict[Node, set[Node]] = {node: set() for node in nodes}
        for node in nodes:
            for parent in node.parents:
                if parent in children:
                    children[parent].add(node)
        remaining = {node: len(values) for node, values in children.items()}
        queue = [node for node in nodes if remaining[node] == 0]
        downstream: dict[Node, list[str]] = defaultdict(list)
        downstream[output_node].append(feedback)
        completed: list[_NodeUpdate] = []

        while queue:
            ready, queue = queue, []
            for node in ready:
                if not downstream[node]:
                    raise RuntimeError(
                        f"hytorch.optim.DFM: node {node.layer}[{node.agent}] received no feedback"
                    )
            with ThreadPoolExecutor(max_workers=len(ready)) as executor:
                futures = [
                    executor.submit(self._update_node, node, downstream[node])
                    for node in ready
                ]
                updates = [future.result() for future in futures]
            completed.extend(updates)
            for update in updates:
                node = update.context
                for index, value in enumerate(node.inputs):
                    parent = value.feed_fn
                    if parent is not None and parent in remaining:
                        downstream[parent].append(update.upstream[index])
                for parent in node.parents:
                    if parent in remaining:
                        remaining[parent] -= 1
                        if remaining[parent] == 0:
                            queue.append(parent)
        if len(completed) != len(nodes):
            raise RuntimeError("hytorch.optim.DFM: backward graph contains a cycle")
        return completed

    def _update_node(self, context: Node, messages: list[str]) -> _NodeUpdate:
        if len(context.parameters) != 1:
            raise RuntimeError("hytorch.optim.DFM: one agent must own one workspace")
        set_tree_writable(context.statespace, False)
        set_tree_writable(context.parameter, False)
        statespace_repo = Repo.discover(context.statespace)
        statespace_head = statespace_repo.resolve("HEAD")
        statespace_before = tree_manifest(context.statespace, include_git=True)
        parameter_before = tree_manifest(context.parameter)
        view = context.parameters[0]
        prompt = _backward_prompt(
            context.layer,
            context.agent,
            len(context.inputs),
            messages,
            view.relative_path,
            self.temp,
        )
        try:
            harness = registered()[context.harness]
        except KeyError as exc:
            raise RuntimeError(
                f"hytorch.optim.DFM: harness {context.harness!r} is not registered"
            ) from exc
        result = harness.resume(
            context.session,
            context.root,
            prompt,
            context.mtype,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            read_only=(context.statespace, context.parameter),
        )
        context.session = result.session
        if tree_manifest(context.statespace, include_git=True) != statespace_before:
            raise ValueError(
                "hytorch.optim.DFM: backward modified the read-only statespace"
            )
        if (
            statespace_repo.resolve("HEAD") != statespace_head
            or not statespace_repo.is_clean()
        ):
            raise ValueError(
                "hytorch.optim.DFM: backward changed statespace Git history"
            )
        if tree_manifest(context.parameter) != parameter_before:
            raise ValueError(
                "hytorch.optim.DFM: backward modified the read-only Parameter"
            )
        _validate_node_root(context.root)
        validate_agent_state(context.workspace)
        update, upstream = _read_response(result.text, len(context.inputs))
        return _NodeUpdate(context, update, upstream, tuple(messages))

    def step(self) -> None:
        """Reduce all accumulated feed into one atomic Parameter generation."""
        if not self._records:
            return None
        stores = {
            id(view.parameter._store): view.parameter._store
            for view in self._views.values()
        }
        if len(stores) != 1 or None in stores.values():
            raise RuntimeError(
                "hytorch.optim.DFM: step requires one model workspace store"
            )
        store = next(iter(stores.values()))
        candidate = self._create_candidate(store)
        self._pending = candidate
        before = {
            view.relative_path: tree_manifest(
                os.path.join(store.root, view.relative_path)
            )
            for view in self._views.values()
        }
        try:
            keys = sorted(self._records, key=lambda item: item[1])
            with ThreadPoolExecutor(max_workers=len(keys)) as executor:
                futures = [
                    executor.submit(
                        self._reduce_owner,
                        candidate,
                        self._views[key],
                        self._records[key],
                    )
                    for key in keys
                ]
                for future in futures:
                    future.result()
            candidate.store.repo.commit_all_workdir(
                candidate.root, "hytorch: apply directional feedback"
            )
            if candidate.store.repo.resolve("HEAD") != candidate.base:
                raise RuntimeError(
                    "hytorch.optim.DFM: canonical model changed during step"
                )
            revision = candidate.store.repo.resolve(candidate.branch)
            if revision != candidate.base:
                candidate.store.repo.merge_branches(candidate.store.root, [revision])
            report = Report()
            canonical = candidate.store.repo.resolve("HEAD")
            if canonical != candidate.base:
                report.commits.append(canonical)
            for relative, old in before.items():
                new = tree_manifest(os.path.join(store.root, relative))
                for path in sorted(set(old) | set(new)):
                    if old.get(path) != new.get(path):
                        report.revised[os.path.join(relative, path)] = (
                            WorkspaceRevision(
                                before=_as_text(old.get(path)),
                                after=_as_text(new.get(path)),
                            )
                        )
            self.state["last_report"] = report
        except Exception:
            # Feed remains available so the caller can retry or call zero_feed().
            raise
        finally:
            self._remove_candidate(candidate)
            self._pending = None
        return None

    def _reduce_owner(
        self, candidate: _Candidate, view, records: list[FeedRecord]
    ) -> None:
        root = tempfile.mkdtemp(prefix="hytorch-owner-")
        statespace = os.path.join(root, "statespace")
        parameter = os.path.join(root, "parameter")
        workspace = os.path.join(root, "workspace")
        source = os.path.join(candidate.root, view.relative_path)
        copy_tree(source, parameter)
        copy_tree(source, workspace)
        set_tree_writable(parameter, False)
        repo, _ = Repo.create_integration(statespace, [])
        evidence = [
            dataclasses.asdict(value) for value in sorted(records, key=_record_key)
        ]
        with open(
            os.path.join(statespace, "feeds.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(evidence, file, indent=2, sort_keys=True)
            file.write("\n")
        repo.commit_allow_empty(statespace, "hytorch: record accumulated feed")
        set_tree_writable(statespace, False)
        harness_names = {value.harness for value in records}
        model_types = {value.mtype for value in records}
        if len(harness_names) != 1 or len(model_types) != 1:
            raise RuntimeError(
                "hytorch.optim.DFM: one Parameter cannot mix harnesses or model types before step"
            )
        harness = registered()[next(iter(harness_names))]
        prompt = _step_prompt(view.relative_path, len(records), self.temp)
        result = None
        try:
            result = harness.start(
                root,
                prompt,
                next(iter(model_types)),
                temperature=self.temp,
                max_tokens=self.max_tokens,
                read_only=(statespace, parameter),
            )
            if tree_manifest(parameter) != tree_manifest(source):
                raise ValueError(
                    "hytorch.optim.DFM: step modified the read-only Parameter"
                )
            validate_agent_state(workspace)
            _validate_node_root(root)
            copy_tree(workspace, source)
        finally:
            if result is not None:
                harness.close(result.session)
            set_tree_writable(parameter, True)
            set_tree_writable(statespace, True)
            shutil.rmtree(root, ignore_errors=True)

    def _create_candidate(self, store: ParameterStore) -> _Candidate:
        base = store.repo.resolve("HEAD")
        branch = f"hytorch/dfm/{time.time_ns()}"
        root = tempfile.mkdtemp(prefix="hytorch-dfm-")
        store.repo.branch(branch, base)
        store.repo.add_worktree(root, branch)
        return _Candidate(store, branch, root, base)

    def _discard_pending(self) -> None:
        if self._pending is not None:
            self._remove_candidate(self._pending)
            self._pending = None
        self._records.clear()
        self._views.clear()

    @staticmethod
    def _remove_candidate(candidate: _Candidate) -> None:
        if os.path.isdir(candidate.root):
            candidate.store.repo.remove_worktree(candidate.root)
        try:
            candidate.store.repo.delete_branch(candidate.branch)
        except Exception:
            pass


def _feed_record(value: _NodeUpdate) -> FeedRecord:
    context = value.context
    payload = {
        "text": value.update,
        "layer": context.layer,
        "agent": context.agent,
        "session": context.session.id,
        "output_commit": context.commit,
        "input_commits": [item.commit for item in context.inputs],
        "downstream": list(value.downstream),
        "harness": context.harness,
        "mtype": context.mtype,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FeedRecord(
        text=value.update,
        layer=context.layer,
        agent=context.agent,
        session=context.session.id,
        output_commit=context.commit,
        input_commits=tuple(item.commit for item in context.inputs),
        downstream=value.downstream,
        harness=context.harness,
        mtype=context.mtype,
        digest=digest,
    )


def _record_key(value: FeedRecord) -> tuple:
    return (value.layer, value.agent, value.output_commit, value.session, value.digest)


def _backward_prompt(
    layer: str,
    agent: int,
    inputs: int,
    messages: list[str],
    parameter_path: str,
    temp: float,
) -> str:
    rendered = "\n\n".join(
        f"## Direction {index}\n\n{message}" for index, message in enumerate(messages)
    )
    return (
        f"Resume temporary episode {layer}[{agent}] for backward.\n\n"
        "statespace/ and parameter/ are read-only. workspace/ is temporary episode "
        "state. HyTorch will discard it after backward. Do not update your persistent "
        f"state now. Propose one update for Parameter {parameter_path}.\n\n"
        f"# Directional feedback\n\n{rendered}\n\n"
        "Return only one JSON object with this shape:\n"
        f'{{"update": "<one imperative owner update>", "feedback": [<one imperative direction for each of the {inputs} inputs>]}}\n\n'
        f"Mutation temperature: {temp}."
    )


def _step_prompt(parameter_path: str, count: int, temp: float) -> str:
    return (
        f"Update your persistent native state for Parameter {parameter_path}.\n\n"
        "parameter/ is the read-only state before this optimizer step. workspace/ is "
        "your writable state. statespace/feeds.json contains all accumulated feed with "
        "provenance. Inspect it. Resolve duplicate or conflicting directions. Update "
        "workspace/ once. You may change memories, instructions, skills, settings, "
        "sessions, databases, or other useful native state. Do not use Git in workspace/. "
        f"There are {count} feed records. Mutation temperature: {temp}. Finish the update."
    )


def _read_response(text: str, count: int) -> tuple[str, list[str]]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "hytorch.optim.DFM: backward response must be one JSON object"
        ) from exc
    if not isinstance(value, dict) or set(value) != {"update", "feedback"}:
        raise RuntimeError(
            "hytorch.optim.DFM: backward response must contain 'update' and 'feedback'"
        )
    feedback = value.get("feedback")
    if not isinstance(feedback, list) or len(feedback) != count:
        raise RuntimeError(
            f"hytorch.optim.DFM: backward response requires {count} feedback strings"
        )
    if any(not isinstance(item, str) or not item.strip() for item in feedback):
        raise RuntimeError(
            "hytorch.optim.DFM: every upstream feedback value must be non-empty text"
        )
    update = value["update"]
    if not isinstance(update, str) or not update.strip():
        raise RuntimeError("hytorch.optim.DFM: owner update must be non-empty text")
    return update.strip(), [item.strip() for item in feedback]


def _validate_node_root(root: str) -> None:
    unexpected = sorted(
        set(os.listdir(root)) - {"statespace", "parameter", "workspace"}
    )
    if unexpected:
        raise ValueError(
            "hytorch.optim.DFM: agent changed paths outside the node state: "
            + ", ".join(unexpected)
        )


def _as_text(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


__all__ = ["DFM", "FeedRecord"]
