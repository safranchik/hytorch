"""Directory-native model state serialization."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import namedtuple
from dataclasses import dataclass, field

from ._git import GitError, Repo
from .parameter import copy_tree


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    __slots__ = ()

    def __repr__(self) -> str:
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()


@dataclass(frozen=True, init=False)
class StateDir:
    """An immutable model-state revision stored in a Git directory."""

    repo: Repo = field(repr=False, compare=False)
    commit: str
    dir: str
    _manifest: dict = field(repr=False, compare=False)

    def __init__(self, directory: str | os.PathLike[str], commit: str = "HEAD"):
        resolved_dir = os.path.realpath(os.fspath(directory))
        try:
            repo = Repo.discover(resolved_dir)
        except GitError as exc:
            raise ValueError(
                "hytorch.StateDir: directory must be a Git repository"
            ) from exc
        if repo.root != resolved_dir:
            raise ValueError(
                "hytorch.StateDir: directory must be a Git repository root"
            )
        resolved_commit = repo.resolve(commit)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "commit", resolved_commit)
        object.__setattr__(self, "dir", resolved_dir)
        object.__setattr__(self, "_manifest", _read_manifest(repo, resolved_commit))


def save(state_dir: StateDir, path: str | os.PathLike[str]) -> None:
    """Save one exact model-state revision as a self-contained Git directory."""
    if not isinstance(state_dir, StateDir):
        raise TypeError("hytorch.save: value must be a hytorch.StateDir")
    destination = os.path.abspath(os.fspath(path))
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if os.path.commonpath((state_dir.dir, destination)) == state_dir.dir:
        raise ValueError("hytorch.save: destination cannot be inside the model state")

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        _git(
            state_dir.dir,
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--no-tags",
            state_dir.dir,
            destination,
        )
        _git(destination, "checkout", "--quiet", "-B", "main", state_dir.commit)
        _git(destination, "remote", "remove", "origin")
        StateDir(destination)
    except Exception:
        if os.path.lexists(destination):
            shutil.rmtree(destination)
        raise


def load(path: str | os.PathLike[str]) -> StateDir:
    """Load a model-state directory for use with ``Module.load_state_dir``."""
    return StateDir(path)


def load_module_state(module, state_dir: StateDir, strict: bool) -> _IncompatibleKeys:
    if not isinstance(state_dir, StateDir):
        raise TypeError(
            "hytorch.mn.Module.load_state_dir: state_dir must be a hytorch.StateDir"
        )
    if not isinstance(strict, bool):
        raise TypeError("hytorch.mn.Module.load_state_dir: strict must be a bool")

    store = module._ensure_parameter_store()
    if not store.repo.is_clean():
        raise RuntimeError(
            "hytorch.mn.Module.load_state_dir: model state has uncommitted changes"
        )
    optimizers = {
        parameter._optimizer
        for parameter in module.parameters()
        if parameter._optimizer is not None
    }
    if any(
        getattr(optimizer, "_pending", None) is not None for optimizer in optimizers
    ):
        raise RuntimeError(
            "hytorch.mn.Module.load_state_dir: discard or promote the pending optimizer update first"
        )

    destination_manifest = _read_manifest(store.repo, store.repo.resolve("HEAD"))
    source_entries = _workspace_entries(state_dir._manifest)
    destination_entries = _workspace_entries(destination_manifest)
    source_keys = set(source_entries)
    destination_keys = set(destination_entries)
    missing = sorted(destination_keys - source_keys)
    unexpected = sorted(source_keys - destination_keys)
    errors = _compatibility_errors(
        state_dir._manifest,
        destination_manifest,
        source_entries,
        destination_entries,
    )
    if strict:
        if missing:
            errors.append("Missing key(s): " + ", ".join(repr(key) for key in missing))
        if unexpected:
            errors.append(
                "Unexpected key(s): " + ", ".join(repr(key) for key in unexpected)
            )
    if errors:
        detail = "\n\t".join(errors)
        raise RuntimeError(
            f"Error(s) in loading state_dir for {type(module).__name__}:\n\t{detail}"
        )

    with tempfile.TemporaryDirectory(prefix="hytorch-state-load-") as snapshot:
        state_dir.repo.export_tree(state_dir.commit, snapshot)
        matched = sorted(source_keys & destination_keys)
        for key in matched:
            source = os.path.join(snapshot, source_entries[key]["path"])
            if not os.path.isdir(source):
                raise RuntimeError(
                    f"hytorch.load: state directory is missing workspace {key!r}"
                )
        _promote_loaded_workspaces(
            store,
            snapshot,
            matched,
            source_entries,
            destination_entries,
        )

    for parameter in module.parameters():
        parameter.zero_feed()
    return _IncompatibleKeys(missing, unexpected)


def _promote_loaded_workspaces(
    store,
    snapshot: str,
    matched: list[str],
    source_entries: dict[str, dict],
    destination_entries: dict[str, dict],
) -> None:
    base = store.repo.resolve("HEAD")
    branch = f"hytorch/load/{uuid.uuid4().hex}"
    candidate = tempfile.mkdtemp(prefix="hytorch-state-candidate-")
    branched = False
    added = False
    try:
        store.repo.branch(branch, base)
        branched = True
        store.repo.add_worktree(candidate, branch)
        added = True
        for key in matched:
            source = os.path.join(snapshot, source_entries[key]["path"])
            destination = os.path.join(candidate, destination_entries[key]["path"])
            copy_tree(source, destination)
        commit, _ = store.repo.commit_all_workdir(
            candidate, "hytorch: load model state"
        )
        store.repo.fast_forward(store.root, commit)
    except Exception:
        _discard_load_candidate(store.repo, candidate, branch, branched, added)
        raise
    _discard_load_candidate(store.repo, candidate, branch, branched, added)


def _discard_load_candidate(
    repo: Repo, candidate: str, branch: str, branched: bool, added: bool
) -> None:
    if added:
        try:
            repo.remove_worktree(candidate)
        except GitError:
            return
    elif os.path.isdir(candidate):
        shutil.rmtree(candidate)
    if branched:
        try:
            repo.delete_branch(branch)
        except GitError:
            pass


def _read_manifest(repo: Repo, commit: str) -> dict:
    try:
        value = json.loads(repo.read_file(commit, "MODEL.json"))
    except (GitError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            "hytorch.StateDir: committed MODEL.json is missing or invalid"
        ) from exc
    if not isinstance(value, dict) or value.get("format") != "hytorch-model-v1":
        raise ValueError("hytorch.StateDir: unsupported model-state format")
    if not isinstance(value.get("modules"), dict):
        raise ValueError("hytorch.StateDir: MODEL.json modules must be an object")
    _workspace_entries(value)
    return value


def _workspace_entries(manifest: dict) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for module_name, module in manifest["modules"].items():
        if not isinstance(module_name, str) or not isinstance(module, dict):
            raise ValueError("hytorch.StateDir: invalid module entry")
        parameters = module.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("hytorch.StateDir: invalid parameter entries")
        for parameter_name, parameter in parameters.items():
            if not isinstance(parameter_name, str) or not isinstance(parameter, dict):
                raise ValueError("hytorch.StateDir: invalid parameter entry")
            shape = parameter.get("shape")
            input_features = parameter.get("input_features")
            workspaces = parameter.get("workspaces")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in shape
                )
                or (
                    input_features is not None
                    and (
                        not isinstance(input_features, int)
                        or isinstance(input_features, bool)
                        or input_features <= 0
                    )
                )
                or not isinstance(workspaces, list)
                or len(workspaces) != shape[0]
            ):
                raise ValueError("hytorch.StateDir: invalid workspace metadata")
            for index, relative in enumerate(workspaces):
                if not isinstance(relative, str) or not _safe_relative_path(relative):
                    raise ValueError("hytorch.StateDir: invalid workspace path")
                key = f"{module_name}.{parameter_name}.{index}"
                if key in entries or any(
                    item["path"] == relative for item in entries.values()
                ):
                    raise ValueError("hytorch.StateDir: duplicate workspace entry")
                entries[key] = {
                    "path": relative,
                    "shape": shape,
                    "input_features": input_features,
                    "module_type": module.get("type"),
                }
    return entries


def _compatibility_errors(
    source_manifest: dict,
    destination_manifest: dict,
    source_entries: dict[str, dict],
    destination_entries: dict[str, dict],
) -> list[str]:
    del source_manifest, destination_manifest
    errors = []
    for key in sorted(set(source_entries) & set(destination_entries)):
        source = source_entries[key]
        destination = destination_entries[key]
        if source["shape"] != destination["shape"]:
            errors.append(
                f"size mismatch for {key}: source shape {source['shape']} "
                f"does not match model shape {destination['shape']}"
            )
        if (
            source["input_features"] is not None
            and destination["input_features"] is not None
            and source["input_features"] != destination["input_features"]
        ):
            errors.append(
                f"input feature mismatch for {key}: source "
                f"{source['input_features']} does not match model "
                f"{destination['input_features']}"
            )
        if source["module_type"] != destination["module_type"]:
            errors.append(
                f"module type mismatch for {key}: source {source['module_type']!r} "
                f"does not match model {destination['module_type']!r}"
            )
    return errors


def _safe_relative_path(path: str) -> bool:
    normalized = os.path.normpath(path)
    return (
        bool(path)
        and not os.path.isabs(path)
        and normalized not in {".", ".."}
        and not normalized.startswith(".." + os.sep)
    )


def _git(directory: str, *args: str) -> None:
    result = subprocess.run(
        ["git", "-C", directory, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or f"git {' '.join(args)} failed")


__all__ = ["StateDir", "load", "save"]
