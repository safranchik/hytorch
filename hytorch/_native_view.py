"""Stable local working-directory views for persistent native sessions."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the process-local lock.
    fcntl = None

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


@contextlib.contextmanager
def native_node_view(directory: str) -> Iterator[str]:
    """Expose one changing node workspace at one stable local path."""
    root = os.path.realpath(directory)
    statespace = os.path.join(root, "statespace")
    workspace = os.path.join(root, "workspace")
    if not os.path.isdir(statespace) or not os.path.isdir(workspace):
        raise RuntimeError("hytorch harness: node state is unavailable")

    identity = _node_identity(workspace)
    with _locks_guard:
        lock = _locks.setdefault(identity, threading.Lock())
    with lock, _process_lock(identity):
        view = _view_path(identity)
        os.makedirs(view, mode=0o700, exist_ok=True)
        targets = [("statespace", statespace), ("workspace", workspace)]
        parameter = os.path.join(root, "parameter")
        if os.path.isdir(parameter):
            targets.append(("parameter", parameter))
        for name, target in targets:
            link = os.path.join(view, name)
            if os.path.lexists(link):
                if os.path.isdir(link) and not os.path.islink(link):
                    shutil.rmtree(link)
                else:
                    os.remove(link)
            os.symlink(target, link, target_is_directory=True)
        failed = False
        try:
            yield view
        except BaseException:
            failed = True
            raise
        finally:
            for name, _ in targets:
                link = os.path.join(view, name)
                if os.path.lexists(link):
                    os.remove(link)
            try:
                os.rmdir(view)
            except OSError:
                pass
            if not failed and _read_identity(workspace) != identity:
                raise RuntimeError(
                    "hytorch harness: agent changed its native node identity"
                )


def _node_identity(workspace: str) -> str:
    metadata = os.path.join(workspace, ".hytorch")
    path = os.path.join(metadata, "node-id")
    if os.path.isfile(path):
        return _read_identity(workspace)
    os.makedirs(metadata, exist_ok=True)
    value = uuid.uuid4().hex
    with open(path, "w", encoding="ascii") as file:
        file.write(value + "\n")
    return value


def _read_identity(workspace: str) -> str:
    path = os.path.join(workspace, ".hytorch", "node-id")
    if not os.path.isfile(path):
        raise RuntimeError("hytorch harness: native node identity is unavailable")
    with open(path, encoding="ascii") as file:
        value = file.read().strip()
    try:
        return uuid.UUID(value).hex
    except ValueError as exc:
        raise RuntimeError("hytorch harness: native node identity is invalid") from exc


def _view_path(identity: str) -> str:
    return os.path.join(tempfile.gettempdir(), "hytorch-native", identity)


@contextlib.contextmanager
def _process_lock(identity: str) -> Iterator[None]:
    root = os.path.join(tempfile.gettempdir(), "hytorch-native")
    os.makedirs(root, mode=0o700, exist_ok=True)
    path = os.path.join(root, identity + ".lock")
    with open(path, "a", encoding="ascii") as file:
        if fcntl is not None:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


__all__ = ["native_node_view"]
