"""A small, dependency-free circuit breaker.

Purpose: when a provider is down, stop paying its timeout on every request. The
breaker is per provider (one instance in the fallback chain), and its state is
observable so the trace can say *why* a request never reached the network
(NFR-PORT-01, NFR-OPS-01).

State machine (the conventional one):

```
closed --(failures >= threshold)--> open --(recovery_timeout elapsed)--> half_open
half_open --(success)--> closed
half_open --(failure)--> open
```

``half_open`` admits at most ``half_open_max_calls`` probes; every other caller is
rejected until the probe resolves. The clock is injected so tests never wait.

    # AT-PORT-01
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-provider breaker. Not thread-safe on purpose: one event loop per worker."""

    name: str = "provider"
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_max_calls: int = 1
    clock: Callable[[], float] = time.monotonic
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    opened_at: float | None = None
    half_open_calls: int = 0
    rejected: int = 0
    events: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_s < 0:
            raise ValueError("recovery_timeout_s must be >= 0")

    # -- state transitions --------------------------------------------------------
    def _record_event(self, event: str, reason: str = "") -> None:
        self.events.append((event, reason))

    def _open(self, reason: str) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = self.clock()
        self.half_open_calls = 0
        self._record_event("open", reason)

    def _close(self) -> None:
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self.failures = 0
        self.half_open_calls = 0
        self._record_event("close")

    # -- API ----------------------------------------------------------------------
    def allow(self) -> bool:
        """Whether a call may proceed right now. Counts the rejection otherwise."""
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            if self.opened_at is not None and (
                self.clock() - self.opened_at >= self.recovery_timeout_s
            ):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self._record_event("half_open", "recovery timeout elapsed")
            else:
                self.rejected += 1
                return False
        if self.half_open_calls >= self.half_open_max_calls:
            self.rejected += 1
            return False
        self.half_open_calls += 1
        return True

    def record_success(self) -> None:
        self.successes += 1
        if self.state is CircuitState.HALF_OPEN:
            self._close()
        else:
            self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.state is CircuitState.HALF_OPEN:
            self._open("half-open probe failed")
            return
        if self.failures >= self.failure_threshold:
            self._open(f"{self.failures} consecutive failures")

    # -- observability ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Plain data for the trace/metrics layer (no secrets, no clock objects)."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "successes": self.successes,
            "rejected": self.rejected,
            "half_open_calls": self.half_open_calls,
        }


__all__ = ["CircuitBreaker", "CircuitState"]
