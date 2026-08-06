"""hytorch.mn.Linear: a dense layer of parallel filesystem agents."""

from __future__ import annotations

import os
import shutil
import tempfile

from ._autofeed import Node
from ._context import current_run
from ._git import Repo
from ._scheduler import DeferredState
from .graph import Module
from .parameter import (
    Parameter,
    ParameterStore,
    materialize_agent_state,
    set_tree_writable,
    tree_manifest,
    validate_agent_state,
)
from .space import Space, SpaceBatch

HYTORCH_LAYER_TRAILER = "HyTorch-Layer"
HYTORCH_WORKSPACE_TRAILER = "HyTorch-Workspace"
HYTORCH_AGENT_TRAILER = "HyTorch-Agent"
HYTORCH_SESSION_TRAILER = "HyTorch-Session"


class Linear(Module):
    """Apply a dense transformation with one agent per output feature.

    Every output agent receives every input Space. Each agent owns one
    monolithic, directory-backed ``weight`` workspace.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: str | None = "",
        harness: str | None = None,
        mtype: str | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(in_features, int)
            or isinstance(in_features, bool)
            or in_features <= 0
        ):
            raise ValueError(
                "hytorch.mn.Linear: in_features must be a positive integer"
            )
        if (
            not isinstance(out_features, int)
            or isinstance(out_features, bool)
            or out_features <= 0
        ):
            raise ValueError(
                "hytorch.mn.Linear: out_features must be a positive integer"
            )
        if bias is not None and not isinstance(bias, str):
            raise TypeError("hytorch.mn.Linear: bias must be text or None")
        if mtype is not None and (not isinstance(mtype, str) or not mtype.strip()):
            raise ValueError("hytorch.mn.Linear: mtype must be a non-empty string")

        self.in_features = in_features
        self.out_features = out_features
        object.__setattr__(self, "_harness", harness)
        object.__setattr__(self, "_mtype", mtype.strip() if mtype else None)
        self.weight = Parameter.empty(out_features, input_features=in_features)
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset each output agent's workspace."""
        from .mn import init

        init.uniform_(self.weight)
        if self.bias:
            for view in self.weight.views():
                policy = view.text().strip()
                view._set_text(f"# Bias\n\n{self.bias.strip()}\n\n{policy}\n")
            if self.weight._store is not None:
                self.weight._store.commit(f"hytorch: initialize {self.weight.name}")

    def _bind_parameters(self, store: ParameterStore) -> None:
        layer = self._qualified_name.replace(".", "/")
        paths = {(i,): f"layers/{layer}/{i}" for i in range(self.out_features)}
        self.weight._bind(store, f"{self._qualified_name}.weight", paths)

    def id(self) -> str:
        return self._qualified_name

    def forward(
        self, *predecessors: Space | DeferredState, task: str = ""
    ) -> SpaceBatch:
        run = current_run()
        if not predecessors:
            raise ValueError("hytorch.mn.Linear: at least one input space is required")
        inputs = (
            list(predecessors[0])
            if len(predecessors) == 1 and isinstance(predecessors[0], SpaceBatch)
            else list(predecessors)
        )
        if len(inputs) != self.in_features:
            raise ValueError(
                f"hytorch.mn.Linear {self.id()}: in_features={self.in_features} requires "
                f"exactly {self.in_features} input spaces, got {len(inputs)}"
            )

        results = SpaceBatch()
        for index in range(self.out_features):
            results.append(
                run.scheduler.add(
                    inputs,
                    lambda resolved, output_index=index: self._execute(
                        run, resolved, task, output_index
                    ),
                )
            )
        return results

    def _execute(self, run, inputs: list[Space], task: str, index: int) -> Space:
        input_harnesses = {value.harness for value in inputs if value.harness}
        target = run.default_harness
        if self._harness is not None and self._harness != target:
            raise RuntimeError(
                "hytorch.mn.Module: one model must use one harness for forward and backward"
            )
        if len(input_harnesses) > 1 or (
            input_harnesses and input_harnesses != {target}
        ):
            raise ValueError(
                f"hytorch.mn.Linear {self.id()}: inputs are on {sorted(input_harnesses)} but "
                f"the layer runs on {target!r}; move the value explicitly with .to()"
            )
        mtype = self._mtype if self._mtype is not None else run.default_mtype
        return self._run_instance(
            run.harness_for(target),
            inputs,
            task,
            index,
            target,
            mtype,
            run.inference,
        )

    def _run_instance(
        self,
        harness,
        inputs: list[Space],
        task: str,
        index: int,
        harness_name: str,
        mtype: str | None,
        inference: bool,
    ) -> Space:
        branch = "hytorch-integration"
        root = tempfile.mkdtemp(prefix="hytorch-node-")
        statespace = os.path.join(root, "statespace")
        parameter = os.path.join(root, "parameter")
        workspace = os.path.join(root, "workspace")
        commits = [value.commit for value in inputs]
        statespace_repo, integration_base = Repo.create_integration(
            statespace,
            [(value.repo.root, value.commit) for value in inputs],
        )

        weight = self.weight[index]
        workspace_revision = weight.revision
        materialize_agent_state(
            weight.parameter._store,
            workspace_revision,
            weight.relative_path,
            parameter,
        )
        validate_agent_state(parameter)
        shutil.copytree(parameter, workspace, symlinks=True)
        episode_identity = os.path.join(workspace, ".hytorch", "node-id")
        if os.path.isfile(episode_identity):
            os.remove(episode_identity)
        parameter_before = tree_manifest(parameter)
        set_tree_writable(parameter, False)

        prompt = self._build_prompt(
            task, index, workspace_revision, weight.relative_path
        )
        try:
            result = harness.start(
                root,
                prompt,
                mtype,
                read_only=(parameter,),
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        try:
            validate_agent_state(workspace)
            if tree_manifest(parameter) != parameter_before:
                raise RuntimeError(
                    f"hytorch.mn.Linear {self.id()}: forward modified the read-only Parameter"
                )
            unexpected = sorted(
                set(os.listdir(root)) - {"parameter", "workspace", "statespace"}
            )
            if unexpected:
                raise RuntimeError(
                    f"hytorch.mn.Linear {self.id()}: forward changed paths outside the statespace: "
                    + ", ".join(unexpected)
                )
            if not statespace_repo.is_clean():
                raise RuntimeError(
                    f"hytorch.mn.Linear {self.id()}: forward finished with uncommitted "
                    "statespace changes"
                )
            agent_head = statespace_repo.resolve("HEAD")
            if agent_head == integration_base:
                raise RuntimeError(
                    f"hytorch.mn.Linear {self.id()}: forward did not merge any input"
                )
            for input_index, expected in enumerate(commits):
                ref = f"refs/hytorch/inputs/{input_index}"
                actual = statespace_repo.resolve(ref)
                if actual != expected:
                    raise RuntimeError(
                        f"hytorch.mn.Linear {self.id()}: forward changed input ref {input_index}"
                    )
                if not statespace_repo.is_ancestor(actual, agent_head):
                    raise RuntimeError(
                        f"hytorch.mn.Linear {self.id()}: forward did not merge input {input_index}"
                    )
            message = (
                f"hytorch: {self.id()} agent {index}\n\n"
                f"{HYTORCH_LAYER_TRAILER}: {self.id()}\n"
                f"{HYTORCH_AGENT_TRAILER}: {index}\n"
                f"{HYTORCH_WORKSPACE_TRAILER}: {workspace_revision}\n"
                f"{HYTORCH_SESSION_TRAILER}: {result.session.id}\n\n{result.text}"
            )
            commit = statespace_repo.commit_allow_empty(statespace, message)
        except Exception:
            harness.close(result.session)
            shutil.rmtree(root, ignore_errors=True)
            raise

        if inference:
            harness.close(result.session)
            shutil.rmtree(workspace, ignore_errors=True)
            set_tree_writable(parameter, True)
            shutil.rmtree(parameter, ignore_errors=True)
            return Space(
                statespace,
                repo=statespace_repo,
                commit=commit,
                branch=branch,
                dir=statespace,
                layer=self.id(),
                summary=result.text,
                harness=harness_name,
                mtype=mtype,
                requires_feed=False,
                feed_fn=None,
            )

        parameters = (weight,)
        parents = _deduplicate_nodes(
            [value.feed_fn for value in inputs if value.feed_fn is not None]
        )
        node = Node(
            layer=self.id(),
            agent=index,
            harness=harness_name,
            mtype=mtype,
            session=result.session,
            parameters=parameters,
            inputs=tuple(inputs),
            parents=parents,
            repo=statespace_repo,
            commit=commit,
            root=root,
            workspace=workspace,
            parameter=parameter,
            statespace=statespace,
            workspace_revision=workspace_revision,
        )
        return Space(
            statespace,
            repo=statespace_repo,
            commit=commit,
            branch=branch,
            dir=statespace,
            layer=self.id(),
            summary=result.text,
            harness=harness_name,
            mtype=mtype,
            requires_feed=bool(weight.parameter.requires_feed or parents),
            feed_fn=node if weight.parameter.requires_feed or parents else None,
        )

    def _build_prompt(
        self,
        task: str,
        index: int,
        workspace_revision: str,
        workspace_path: str,
    ) -> str:
        parts = [
            f"Run node {self.id()}[{index}] at workspace revision {workspace_revision}.\n\n",
            "The node root contains three directories:\n",
            "- statespace/: writable self-contained Git state\n",
            "- parameter/: read-only canonical native state\n",
            "- workspace/: writable temporary episode fork\n\n",
            f"HyTorch materialized parameter/ from private model path {workspace_path}. ",
            "The directories have no model Git metadata. Use workspace/ for temporary "
            "session state. Forward changes to workspace/ are never promoted. "
            "Do not modify parameter/.\n",
            f"The {self.in_features} inputs are Git refs in statespace/:\n",
            *[
                f"- refs/hytorch/inputs/{input_index}\n"
                for input_index in range(self.in_features)
            ],
            "\nInspect workspace/ and use it to transform statespace/. Choose the input "
            "merge order. Merge every input ref with --no-ff and "
            "--allow-unrelated-histories. Resolve and commit each merge before the "
            "next merge. You can make more statespace changes after merging. Commit "
            "all statespace changes before ending your turn. Leave the statespace repository "
            "clean. Your final committed statespace HEAD is this node's output and is "
            "passed to successor agents. Do not run Git commands in workspace/.\n",
        ]
        if task:
            parts.extend(["\n# Task\n\n", task, "\n"])
        return "".join(parts)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _deduplicate_nodes(values: list[Node]) -> tuple[Node, ...]:
    result: list[Node] = []
    seen: set[int] = set()
    for value in values:
        identity = id(value)
        if identity not in seen:
            seen.add(identity)
            result.append(value)
    return tuple(result)


__all__ = [
    "HYTORCH_AGENT_TRAILER",
    "HYTORCH_LAYER_TRAILER",
    "HYTORCH_SESSION_TRAILER",
    "HYTORCH_WORKSPACE_TRAILER",
    "Linear",
]
