"""Retry policy: exponential backoff with jitter, and the decision of *what* to retry.

Two things are deliberately explicit here:

* the delay formula, so a reviewer can check it without running a test
  ``delay = min(max_delay, base_delay * 2**attempt) * (1 - jitter + 2*jitter*rng())``
  — a symmetric multiplicative jitter, i.e. with ``jitter=0.5`` the delay lands in
  ``[0.5x, 1.5x]`` of the nominal value;
* the retryable set, because retrying a 401 five times is how CI gets rate-limited.

The random source and the sleep function are injected, so tests exercise the policy
deterministically and never actually sleep (NFR-DET-01).

    # AT-PORT-01
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

#: Statuses worth a second attempt: rate limiting, lock contention and server faults.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Statuses where a retry would waste money and time (bad key, bad request, no model).
FATAL_STATUS: frozenset[int] = frozenset({400, 401, 403, 404, 405, 413, 422})

#: Assumed server cooldown when a 429/503 carries no ``Retry-After`` header.
_DEFAULT_RETRY_AFTER_S = 1.0

#: Never sleep longer than this because a server asked us to (CI has a 10-minute gate).
_MAX_RETRY_AFTER_S = 30.0


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff parameters. ``max_retries`` is the number of *additional* attempts."""

    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("delays must be >= 0")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be within [0, 1]")

    def nominal_delay(self, attempt: int) -> float:
        """Delay before attempt ``attempt + 1`` without jitter (attempt is 0-based)."""
        if self.base_delay_s == 0.0:
            return 0.0
        return min(self.max_delay_s, self.base_delay_s * (2.0**attempt))

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Jittered delay in seconds for the given 0-based retry attempt."""
        nominal = self.nominal_delay(attempt)
        if nominal == 0.0:
            return 0.0
        if self.jitter == 0.0:
            return nominal
        source = rng or random.Random()
        factor = 1.0 - self.jitter + 2.0 * self.jitter * source.random()
        return min(self.max_delay_s, nominal * factor)

    def should_retry(
        self,
        *,
        attempt: int,
        status_code: int | None = None,
        timed_out: bool = False,
        transport_error: bool = False,
    ) -> bool:
        """Whether another attempt is allowed and *sensible* for this failure."""
        if attempt >= self.max_retries:
            return False
        if status_code is not None:
            if status_code in FATAL_STATUS:
                return False
            return status_code in RETRYABLE_STATUS
        # No status: a timeout or a connection failure is transient by assumption.
        return bool(timed_out or transport_error)


@dataclass(frozen=True)
class Runner:
    """Injected side effects: sleeping, randomness and the monotonic clock."""

    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)
    clock: Callable[[], float] = time.monotonic

    def wait(self, seconds: float) -> None:
        if seconds > 0:
            self.sleep(seconds)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header; only the delta-seconds form is honoured."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_S)


def retry_after_or_default(value: str | None) -> float:
    """``Retry-After`` when the server sent one, otherwise a conservative default."""
    parsed = parse_retry_after(value)
    return _DEFAULT_RETRY_AFTER_S if parsed is None else parsed


__all__ = [
    "FATAL_STATUS",
    "RETRYABLE_STATUS",
    "RetryPolicy",
    "Runner",
    "parse_retry_after",
    "retry_after_or_default",
]
