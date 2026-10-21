"""Per-client rate limiting: a token bucket with a documented retry hint.

FR-SVC-06 asks for "rate limiting per client with documented limits and a ``429``
envelope". A fixed window is easy but produces the classic 2× burst at the window edge; a
token bucket is one line more code and behaves the way an operator expects:

* capacity ``burst`` (default = the per-minute limit);
* refill ``limit_per_minute / 60`` tokens per second;
* a request costs one token; if none is available the response is ``429`` with
  ``Retry-After`` computed from the time to the next whole token.

Identity is the ``X-Client-Id`` header when present (the console sets it) and the socket
peer otherwise. A caller can therefore only exhaust its own budget — there is no shared
global bucket that lets one client deny service to another.

The limiter is in-process by design: this is a reference stack, not a cluster. Running
several workers means the effective budget is multiplied by the worker count; that is
documented in ``docs/ops.md`` rather than hidden.

    # AT-SVC-06
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RateDecision:
    """The outcome of one check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float = 0.0
    reset_after_s: float = 0.0


class RateLimiter:
    """Token bucket keyed by client identity."""

    def __init__(
        self,
        *,
        limit_per_minute: int,
        burst: int | None = None,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        self.limit = int(limit_per_minute)
        self.capacity = float(burst or limit_per_minute)
        self.refill_per_second = self.limit / 60.0
        self.enabled = enabled
        self.clock = clock
        self.max_clients = max_clients
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    # -- checks --------------------------------------------------------------------
    def check(self, client_id: str, *, cost: float = 1.0) -> RateDecision:
        """Consume ``cost`` tokens for ``client_id`` (or report the retry delay)."""
        if not self.enabled:
            return RateDecision(allowed=True, limit=self.limit, remaining=self.limit)
        now = self.clock()
        key = client_id or "anonymous"
        with self._lock:
            self._prune(now)
            tokens, updated = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.refill_per_second)
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return RateDecision(
                    allowed=True,
                    limit=self.limit,
                    remaining=int(tokens),
                    reset_after_s=max(0.0, (self.capacity - tokens) / self.refill_per_second),
                )
            self._buckets[key] = (tokens, now)
            deficit = cost - tokens
            retry_after = deficit / self.refill_per_second if self.refill_per_second else 60.0
            return RateDecision(
                allowed=False,
                limit=self.limit,
                remaining=int(tokens),
                retry_after_s=round(retry_after, 3),
                reset_after_s=round(retry_after, 3),
            )

    def remaining(self, client_id: str) -> int:
        with self._lock:
            tokens, _updated = self._buckets.get(client_id or "anonymous", (self.capacity, 0.0))
            return int(tokens)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    # -- internals -----------------------------------------------------------------
    def _prune(self, now: float) -> None:
        """Drop buckets that have been full for a minute (bounded memory)."""
        if len(self._buckets) < self.max_clients:
            return
        full_after = self.capacity / self.refill_per_second if self.refill_per_second else 60.0
        stale = [
            key
            for key, (tokens, updated) in self._buckets.items()
            if tokens >= self.capacity and now - updated > full_after
        ]
        for key in stale:
            del self._buckets[key]
        if len(self._buckets) >= self.max_clients:  # still full: drop the oldest half
            ordered = sorted(self._buckets.items(), key=lambda item: item[1][1])
            for key, _value in ordered[: max(1, len(ordered) // 2)]:
                del self._buckets[key]


__all__ = ["RateDecision", "RateLimiter"]
