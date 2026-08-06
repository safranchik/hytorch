"""Agent harnesses, resumable sessions, and the global registry."""

from __future__ import annotations

import abc
import dataclasses
import threading


@dataclasses.dataclass(frozen=True)
class Session:
    """Opaque reference to a harness-native persisted conversation."""

    harness: str
    id: str
    storage: str


@dataclasses.dataclass(frozen=True)
class Usage:
    """Aggregate model tokens consumed by harness turns."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cache_read_tokens=self.cache_read_tokens - other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens - other.cache_write_tokens,
        )


@dataclasses.dataclass(frozen=True)
class Result:
    """Text and opaque native session tip returned by one harness turn."""

    text: str
    session: Session


class Harness(abc.ABC):
    """An execution environment for agent layers.

    A harness owns the agent runtime, container setup, tool availability, and
    default model. It is HyTorch's analogue of an execution placement, but is
    intentionally named for what it actually is: an agent harness.
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        resolved = name or self.name or type(self).__name__.lower()
        if not isinstance(resolved, str) or not resolved.strip():
            raise ValueError("hytorch.harness.Harness: name must be a non-empty string")
        self.name = resolved.strip()

    @abc.abstractmethod
    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        """Start a persisted session and execute its first prompt."""
        raise NotImplementedError

    @abc.abstractmethod
    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        """Resume a saved session and return its new opaque tip."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, session: Session) -> None:
        """Release runtime resources without deleting persisted agent state."""
        raise NotImplementedError


class UnavailableHarness(Harness):
    """A built-in harness identity whose runtime is not shipped yet."""

    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        raise RuntimeError(f"hytorch harness {self.name!r} is built in but unavailable")

    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        raise RuntimeError(f"hytorch harness {self.name!r} is built in but unavailable")

    def close(self, session: Session) -> None:
        raise RuntimeError(f"hytorch harness {self.name!r} is built in but unavailable")


class PiHarness(Harness):
    """The built-in Pi harness."""

    name = "pi"

    def __init__(self, name: str | None = None, **kwargs) -> None:
        from .pi_harness import PiRuntime

        super().__init__(name)
        self._runtime = PiRuntime(harness_name=self.name, **kwargs)

    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        return self._runtime.start(
            directory,
            prompt,
            mtype,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )

    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        return self._runtime.resume(
            session,
            directory,
            prompt,
            mtype,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )

    def close(self, session: Session) -> None:
        self._runtime.close(session)

    def usage(self) -> Usage:
        """Return aggregate token use for this harness instance."""
        return self._runtime.usage()


class PrimeAgentHarness(Harness):
    """The built-in Prime Agent harness."""

    name = "prime-agent"

    def __init__(self, name: str | None = None, **kwargs) -> None:
        from .prime_harness import PrimeRuntime

        super().__init__(name)
        self._runtime = PrimeRuntime(harness_name=self.name, **kwargs)

    def start(
        self,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        return self._runtime.start(
            directory,
            prompt,
            mtype,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )

    def resume(
        self,
        session: Session,
        directory: str,
        prompt: str,
        mtype: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        read_only: tuple[str, ...] = (),
    ) -> Result:
        return self._runtime.resume(
            session,
            directory,
            prompt,
            mtype,
            temperature=temperature,
            max_tokens=max_tokens,
            read_only=read_only,
        )

    def close(self, session: Session) -> None:
        self._runtime.close(session)

    def usage(self) -> Usage:
        """Return aggregate token use for this harness instance."""
        return self._runtime.usage()


_lock = threading.Lock()
_harnesses: dict[str, Harness] = {}


def register(harness: Harness) -> Harness:
    """Register a custom harness globally and return it."""
    if not isinstance(harness, Harness):
        raise TypeError(
            "hytorch.harness.register: harness must be a hytorch.harness.Harness"
        )
    if not harness.name:
        harness.name = type(harness).__name__.lower()
    with _lock:
        _harnesses[harness.name] = harness
    return harness


def registered() -> dict[str, Harness]:
    """Return a snapshot of globally registered harnesses."""
    with _lock:
        return dict(_harnesses)


def name_of(harness: Harness | str) -> str:
    if isinstance(harness, Harness):
        register(harness)
        return harness.name or type(harness).__name__.lower()
    if isinstance(harness, str) and harness.strip():
        return harness.strip()
    raise ValueError(
        "hytorch: harness must be a hytorch.harness.Harness or non-empty string"
    )


from .claude_harness import ClaudeCodeHarness  # noqa: E402
from .codex_harness import CodexHarness  # noqa: E402
from .hermes_harness import HermesHarness  # noqa: E402
from .opencode_harness import OpenCodeHarness  # noqa: E402

# Stable executable built-ins.
pi = register(PiHarness())
codex = register(CodexHarness())
claude_code = register(ClaudeCodeHarness())
opencode = register(OpenCodeHarness())
hermes = register(HermesHarness())
prime_agent = register(PrimeAgentHarness())

__all__ = [
    "Harness",
    "ClaudeCodeHarness",
    "CodexHarness",
    "HermesHarness",
    "OpenCodeHarness",
    "PiHarness",
    "PrimeAgentHarness",
    "Result",
    "Session",
    "Usage",
    "UnavailableHarness",
    "claude_code",
    "codex",
    "hermes",
    "name_of",
    "pi",
    "opencode",
    "prime_agent",
    "register",
    "registered",
]
