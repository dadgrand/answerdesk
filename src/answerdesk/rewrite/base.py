"""The rewriter seam: one tiny Protocol, one result object, one identity baseline.

The rest of the stack (retrieval, service) only ever sees :class:`QueryRewriter` and
:class:`RewriteResult`; the concrete backends (deterministic, LoRA) are interchangeable and
may be absent. ``IdentityRewriter`` is the default and the fallback: it is a no-op that still
reports latency and provenance so the trace is uniform across arms.

    # AT-GEN-13  # AT-PORT-01
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import Field

from answerdesk.contracts import Contract


class RewriteResult(Contract):
    """What a rewriter returns: the query to retrieve with, and how it got there."""

    original: str
    query: str = Field(min_length=1)
    backend: str = "identity"
    available: bool = True
    fallback: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    notes: str = ""

    @property
    def changed(self) -> bool:
        return self.query != self.original


@runtime_checkable
class QueryRewriter(Protocol):
    """Structural seam: name, availability and a pure rewrite call."""

    name: str

    @property
    def available(self) -> bool: ...

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult: ...


class IdentityRewriter:
    """The no-op baseline (and the graceful fallback when a backend is unavailable)."""

    name = "identity"
    available = True

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult:
        return RewriteResult(
            original=query,
            query=query or " ",
            backend=self.name,
            available=True,
            fallback=False,
            notes="identity (rewriter disabled or unavailable)",
        )


class TimedRewriter:
    """Wrap any rewriter with a latency budget and never let an exception escape.

    A rewriter that raises or times out must not break retrieval: the wrapper records the
    reason and returns the identity rewrite, flagged ``fallback=True``.
    """

    def __init__(
        self,
        inner: QueryRewriter,
        *,
        clock: Callable[[], float] = time.perf_counter,
        timeout_ms: float | None = None,
    ) -> None:
        self.inner = inner
        self.name = f"timed[{inner.name}]"
        self._clock = clock
        self.timeout_ms = timeout_ms

    @property
    def available(self) -> bool:
        return self.inner.available

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult:
        started = self._clock()
        try:
            result = self.inner.rewrite(query, locale=locale)
        except Exception as exc:  # a rewriter must never break the request path
            elapsed = (self._clock() - started) * 1000.0
            return RewriteResult(
                original=query,
                query=query or " ",
                backend=f"fallback[{self.inner.name}]",
                available=False,
                fallback=True,
                latency_ms=round(elapsed, 3),
                notes=f"{type(exc).__name__}: {exc}",
            )
        elapsed = (self._clock() - started) * 1000.0
        result = result.model_copy(update={"latency_ms": round(elapsed, 3)})
        if self.timeout_ms is not None and elapsed > self.timeout_ms:
            return RewriteResult(
                original=query,
                query=query or " ",
                backend=f"fallback[{self.inner.name}]",
                available=False,
                fallback=True,
                latency_ms=round(elapsed, 3),
                notes=f"exceeded {self.timeout_ms:.0f} ms budget; original query kept",
            )
        return result


__all__ = ["IdentityRewriter", "QueryRewriter", "RewriteResult", "TimedRewriter"]
