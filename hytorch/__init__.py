"""PyTorch-shaped meta-networks over Git-backed statespaces and workspaces."""

from . import harness, mn, optim
from ._git import GitError, MergeConflictError, Repo
from ._inference import inference_mode, is_inference_mode_enabled
from ._random import manual_seed
from .backward import Loss, Report, WorkspaceRevision
from .space import Space, SpaceBatch, space
from .state_dir import StateDir, load, save

__version__ = "0.1.0"

__all__ = [
    "GitError",
    "Loss",
    "MergeConflictError",
    "Repo",
    "Report",
    "Space",
    "SpaceBatch",
    "StateDir",
    "WorkspaceRevision",
    "__version__",
    "harness",
    "inference_mode",
    "is_inference_mode_enabled",
    "load",
    "manual_seed",
    "mn",
    "optim",
    "save",
    "space",
]
