"""Directory-backed Parameters and attributed backward feed."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator

from ._git import Repo


class ParameterView:
    """One output agent's complete trainable workspace."""

    def __init__(self, parameter: Parameter, index: tuple[int, ...]) -> None:
        self.parameter = parameter
        self.index = index

    @property
    def feed(self) -> tuple[str, ...] | None:
        values = self.parameter._feed[self.index]
        return tuple(values) if values else None

    def _accumulate_feed(self, value: str) -> None:
        self.parameter._feed[self.index].append(value)

    @property
    def path(self) -> str:
        if self.parameter._store is None:
            raise RuntimeError("hytorch.mn.Parameter is not attached to a Module store")
        return os.path.join(self.parameter._store.root, self.relative_path)

    @property
    def relative_path(self) -> str:
        try:
            return self.parameter._paths[self.index]
        except KeyError as exc:
            raise RuntimeError(
                "hytorch.mn.Parameter is not attached to a Module store"
            ) from exc

    @property
    def revision(self) -> str:
        if self.parameter._store is None:
            return ""
        return self.parameter._store.repo.resolve("HEAD")

    def text(self, relative: str = "AGENTS.md") -> str:
        """Read one text file from the workspace."""
        if self.parameter._store is None:
            data = self.parameter._trees[self.index].get(relative)
            if data is None:
                raise FileNotFoundError(relative)
            return data.decode("utf-8")
        with open(os.path.join(self.path, relative), encoding="utf-8") as file:
            return file.read()

    def _set_text(self, text: str, relative: str = "AGENTS.md") -> None:
        if not isinstance(text, str):
            raise TypeError("hytorch.mn.Parameter text must be a string")
        value = text.rstrip().encode("utf-8") + b"\n"
        self.parameter._trees[self.index][relative] = value
        if self.parameter._store is not None:
            self.parameter._store.write_file(
                os.path.join(self.relative_path, relative), value
            )

    def __repr__(self) -> str:
        suffix = ", ".join(str(value) for value in self.index)
        return f"ParameterView({self.parameter.name or 'unbound'}[{suffix}])"


class Parameter:
    """A registered vector of directory-backed trainable workspaces.

    Each element is one complete filesystem owned by one output agent. The
    filesystem can contain arbitrary files. ``AGENTS.md`` is only its seeded
    initial content.
    """

    def __init__(
        self,
        data,
        requires_feed: bool = True,
        *,
        input_features: int = 1,
    ) -> None:
        trees = _from_data(data)
        if (
            not isinstance(input_features, int)
            or isinstance(input_features, bool)
            or input_features <= 0
        ):
            raise ValueError("hytorch.mn.Parameter: input_features must be positive")
        self.shape = (len(trees),)
        self.input_features = input_features
        self.requires_feed = bool(requires_feed)
        self.name = ""
        self.owner = None
        self._trees = {(i,): tree for i, tree in enumerate(trees)}
        self._feed: dict[tuple[int, ...], list[str]] = {
            (i,): [] for i in range(len(trees))
        }
        self._paths: dict[tuple[int, ...], str] = {}
        self._store: ParameterStore | None = None
        self._optimizer = None

    @classmethod
    def empty(cls, count: int, *, input_features: int) -> Parameter:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("hytorch.mn.Parameter.empty: count must be positive")
        return cls([{} for _ in range(count)], input_features=input_features)

    def __getitem__(self, index: int | tuple[int, ...]) -> ParameterView:
        normalized = (index,) if isinstance(index, int) else tuple(index)
        if normalized not in self._trees:
            raise IndexError(
                f"hytorch.mn.Parameter index {normalized} is out of bounds for {self.shape}"
            )
        return ParameterView(self, normalized)

    def views(self) -> Iterator[ParameterView]:
        for index in self._trees:
            yield ParameterView(self, index)

    def zero_feed(self) -> None:
        for index in self._feed:
            self._feed[index].clear()

    def _bind(
        self,
        store: ParameterStore,
        name: str,
        paths: dict[tuple[int, ...], str],
    ) -> None:
        if self._store is not None and self._store is not store:
            raise RuntimeError(
                "hytorch.mn.Parameter cannot belong to two Module stores"
            )
        self.name = name
        self._store = store
        self._paths = paths
        for index, relative in paths.items():
            store.write_tree(relative, self._trees[index])

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, name={self.name!r})"


class ParameterStore:
    """Private Git repository containing one model's workspaces."""

    def __init__(self) -> None:
        parent = os.path.abspath(os.path.join("hytorch", "workspaces"))
        os.makedirs(parent, exist_ok=True)
        self.root = os.path.join(parent, uuid.uuid4().hex)
        os.makedirs(self.root)
        _git(self.root, "init", "-b", "main")
        _git(self.root, "config", "user.name", "HyTorch")
        _git(self.root, "config", "user.email", "hytorch@localhost")
        _git(
            self.root,
            "commit",
            "--allow-empty",
            "-m",
            "hytorch: create workspace store",
        )
        self.repo = Repo.discover(self.root)
        self._committed = False

    def write_file(self, relative: str, data: bytes) -> None:
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(data)

    def write_tree(self, relative: str, tree: dict[str, bytes]) -> None:
        root = os.path.join(self.root, relative)
        os.makedirs(root, exist_ok=True)
        for path, data in tree.items():
            self.write_file(os.path.join(relative, path), data)

    def commit_initial(self) -> str:
        commit, changed = self.commit("hytorch: initialize meta-network workspaces")
        if not changed and not self._committed:
            raise RuntimeError("hytorch: workspace initialization produced no files")
        self._committed = True
        return commit

    def commit(self, message: str) -> tuple[str, bool]:
        return self.repo.commit_all_workdir(self.root, message)


def copy_tree(source: str, destination: str) -> None:
    """Replace ``destination`` with a copy of one directory tree."""
    if os.path.lexists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=True)


def create_workspace_checkout(
    source: str,
    revision: str,
    relative_path: str,
    destination: str,
) -> Repo:
    """Create a sparse model checkout with full global workspace history."""
    if os.path.lexists(destination):
        set_tree_writable(destination, True)
        shutil.rmtree(destination)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            source,
            destination,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "clone workspace store failed")
    _git(destination, "config", "user.name", "HyTorch")
    _git(destination, "config", "user.email", "hytorch@localhost")
    _git(destination, "sparse-checkout", "init", "--no-cone")
    _git(
        destination,
        "sparse-checkout",
        "set",
        "--no-cone",
        f"/{relative_path}/",
    )
    _git(destination, "switch", "-c", "hytorch-workspace", revision)
    _git(destination, "remote", "remove", "origin")
    return Repo.discover(destination)


def set_tree_writable(root: str, writable: bool) -> None:
    """Set best-effort host permissions for a node tree."""
    directory_mode = 0o755 if writable else 0o555
    file_write = 0o200 if writable else 0
    for current, directories, files in os.walk(root):
        os.chmod(current, directory_mode)
        for name in directories:
            path = os.path.join(current, name)
            if not os.path.islink(path):
                os.chmod(path, directory_mode)
        for name in files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                continue
            mode = os.stat(path, follow_symlinks=False).st_mode & 0o777
            os.chmod(path, (mode | file_write) if writable else (mode & ~0o222))


def tree_manifest(root: str) -> dict[str, bytes]:
    """Return a stable file manifest for mutation-boundary checks."""
    result: dict[str, bytes] = {}
    for current, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name != ".git")
        for name in sorted(value for value in files if value != ".git"):
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root)
            if os.path.islink(path):
                result[relative] = b"link:" + os.readlink(path).encode("utf-8")
            else:
                with open(path, "rb") as file:
                    result[relative] = file.read()
    return result


def _from_data(data) -> list[dict[str, bytes]]:
    from .space import Space, SpaceBatch

    if isinstance(data, Space):
        return [_tree_from_directory(data.dir or data.path)]
    if isinstance(data, SpaceBatch):
        if not data:
            raise ValueError("hytorch.mn.Parameter: data must not be empty")
        return [_tree_from_directory(value.dir or value.path) for value in data]
    if (
        isinstance(data, list)
        and data
        and all(isinstance(value, dict) for value in data)
    ):
        result = []
        for tree in data:
            normalized = {}
            for path, value in tree.items():
                if (
                    not isinstance(path, str)
                    or not path
                    or os.path.isabs(path)
                    or ".." in path.split(os.sep)
                ):
                    raise ValueError(
                        "hytorch.mn.Parameter: tree paths must be safe relative paths"
                    )
                if isinstance(value, str):
                    value = value.encode("utf-8")
                if not isinstance(value, bytes):
                    raise TypeError(
                        "hytorch.mn.Parameter: tree values must be text or bytes"
                    )
                normalized[path] = value
            result.append(normalized)
        return result
    raise TypeError(
        "hytorch.mn.Parameter: data must be a Space, SpaceBatch, or non-empty list of file trees"
    )


def _tree_from_directory(root: str) -> dict[str, bytes]:
    if not os.path.isdir(root):
        raise ValueError(
            f"hytorch.mn.Parameter: workspace directory does not exist: {root}"
        )
    return tree_manifest(root)


def _git(directory: str, *args: str) -> None:
    result = subprocess.run(
        ["git", "-C", directory, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")


__all__ = [
    "Parameter",
    "ParameterView",
    "copy_tree",
    "set_tree_writable",
    "tree_manifest",
]
