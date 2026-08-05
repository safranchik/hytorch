"""Thread-local inference mode for forward passes without backward state."""

from __future__ import annotations

import contextlib
import contextvars

_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hytorch_inference_mode", default=False
)


class inference_mode(contextlib.ContextDecorator):
    """Disable feed recording and release forward sessions after each node."""

    def __init__(self, mode: bool = True) -> None:
        if not isinstance(mode, bool):
            raise TypeError("hytorch.inference_mode: mode must be a bool")
        self.mode = mode
        self._token = None

    def __enter__(self):
        self._token = _enabled.set(self.mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is None:
            raise RuntimeError("hytorch.inference_mode: context was not entered")
        _enabled.reset(self._token)
        self._token = None


def is_inference_mode_enabled() -> bool:
    """Return whether the current thread is declaring an inference graph."""
    return _enabled.get()


__all__ = ["inference_mode", "is_inference_mode_enabled"]
