"""Streaming latency accounting: TTFT separate from total, per provider, per arm.

NFR-LAT-01 asks for time-to-first-token; NFR-LAT-02 for end-to-end p95; NFR-COST-01 for
tokens and USD of **every** call. The existing ledger (`llm/cost.py`) records total
`latency_ms` and cost but nothing about *when the first token arrived*, which is the number a
streaming product is judged on: a 35B model can have a 900 ms TTFT and a 6 s total, and a user
perceives only the former.

This module adds the missing measurement without touching the ledger's contract:

* :class:`ChunkTimer` wraps a chunk iterator; :attr:`ChunkTimer.ttft_ms` is set when the first
  chunk carrying a non-empty delta arrives, :attr:`ChunkTimer.total_ms` when the iterator is
  exhausted. `generate.pipeline.stream_answer` uses it, so the SSE path cannot report a TTFT it
  did not observe.
* :class:`LatencySample` is one measured call; :class:`LatencyStats` is the p50/p95/mean summary
  of a list of them (nearest-rank percentile, the same definition as `llm/cost.py` uses for the
  ledger and as `docs/latency.md` documents).
* :class:`LatencyLedger` groups samples by ``(provider, model, streamed)`` and produces the
  before/after table in `reports/latency-v2/`.

Percentiles are *nearest-rank* (no interpolation): with ≤ 40 samples an interpolated p95 invents
a number between observations, and this project's whole point is that published numbers are
reproducible from the recorded samples. The samples themselves are in the artifact.

    # AT-LAT-01  # AT-COST-01
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from pydantic import Field

from answerdesk.contracts import CompletionChunk, Contract


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in ``[0, 1]``); raises on an empty list."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if q == 0.0:
        return ordered[0]
    rank = max(1, int(-(-q * len(ordered) // 1)))  # ceil(q * n) without float error
    return ordered[min(rank, len(ordered)) - 1]


class LatencyStats(Contract):
    """p50/p95/mean for one metric of one arm."""

    n: int = Field(ge=0)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def of(cls, values: Iterable[float]) -> LatencyStats:
        samples = [float(value) for value in values]
        if not samples:
            return cls(n=0)
        return cls(
            n=len(samples),
            p50_ms=round(percentile(samples, 0.50), 3),
            p95_ms=round(percentile(samples, 0.95), 3),
            mean_ms=round(statistics.fmean(samples), 3),
            min_ms=round(min(samples), 3),
            max_ms=round(max(samples), 3),
        )


class LatencySample(Contract):
    """One measured model call (streamed or not)."""

    arm: str = "baseline"
    provider: str = ""
    model: str = ""
    streamed: bool = False
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    chunks: int = 0
    priced: bool = True
    note: str = ""


class ArmLatency(Contract):
    """Aggregate row of the latency table for one arm."""

    arm: str
    provider: str = ""
    model: str = ""
    streamed: bool = False
    n: int = 0
    ttft: LatencyStats = Field(default_factory=lambda: LatencyStats(n=0))
    total: LatencyStats = Field(default_factory=lambda: LatencyStats(n=0))
    input_tokens_p50: float = 0.0
    output_tokens_p50: float = 0.0
    output_tokens_mean: float = 0.0
    usd_mean: float = 0.0
    usd_total: float = 0.0
    chunks_p50: float = 0.0
    priced: bool = True
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "provider": self.provider,
            "model": self.model,
            "streamed": self.streamed,
            "n": self.n,
            "ttft_p50_ms": self.ttft.p50_ms,
            "ttft_p95_ms": self.ttft.p95_ms,
            "total_p50_ms": self.total.p50_ms,
            "total_p95_ms": self.total.p95_ms,
            "input_tokens_p50": self.input_tokens_p50,
            "output_tokens_p50": self.output_tokens_p50,
            "output_tokens_mean": self.output_tokens_mean,
            "usd_mean": self.usd_mean,
            "usd_total": self.usd_total,
            "priced": self.priced,
            "chunks_p50": self.chunks_p50,
            "note": self.note,
        }


def summarise_arm(samples: Iterable[LatencySample]) -> ArmLatency:
    """Collapse the samples of one arm into the row the report prints."""
    rows = list(samples)
    if not rows:
        raise ValueError("no samples for this arm")
    streamed = all(row.streamed for row in rows)
    return ArmLatency(
        arm=rows[0].arm,
        provider=rows[0].provider,
        model=rows[0].model,
        streamed=streamed,
        n=len(rows),
        ttft=LatencyStats.of(row.ttft_ms for row in rows if row.streamed),
        total=LatencyStats.of(row.total_ms for row in rows),
        input_tokens_p50=round(statistics.median(row.input_tokens for row in rows), 1),
        output_tokens_p50=round(statistics.median(row.output_tokens for row in rows), 1),
        output_tokens_mean=round(statistics.fmean(row.output_tokens for row in rows), 2),
        usd_mean=round(statistics.fmean(row.usd for row in rows), 8),
        usd_total=round(sum(row.usd for row in rows), 8),
        chunks_p50=round(statistics.median(row.chunks for row in rows), 1),
        priced=all(row.priced for row in rows),
    )


class ChunkTimer:
    """Iterate chunks, recording time-to-first-token and total wall-clock.

    ``ttft_ms`` is set on the first chunk whose ``delta`` is non-empty — an SSE keep-alive or an
    empty delta is not a token and must not be reported as one.
    """

    def __init__(
        self,
        source: Iterable[CompletionChunk],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._source = iter(source)
        self._clock = clock
        self._started = clock()
        self._first_token_at: float | None = None
        self._finished_at: float | None = None
        self.chunks = 0
        self.text = ""

    def __iter__(self) -> Iterator[CompletionChunk]:
        for chunk in self._source:
            now = self._clock()
            self.chunks += 1
            if chunk.delta:
                if self._first_token_at is None:
                    self._first_token_at = now
                self.text += chunk.delta
            yield chunk
        if self._finished_at is None:
            self._finished_at = self._clock()

    @property
    def ttft_ms(self) -> float | None:
        if self._first_token_at is None:
            return None
        return round((self._first_token_at - self._started) * 1000.0, 3)

    @property
    def total_ms(self) -> float:
        end = self._finished_at if self._finished_at is not None else self._clock()
        return round((end - self._started) * 1000.0, 3)

    @property
    def saw_first_token(self) -> bool:
        return self._first_token_at is not None


class LatencyLedger:
    """Collect samples in memory and emit the report rows (artifact-friendly)."""

    def __init__(self) -> None:
        self._samples: list[LatencySample] = []

    def add(self, sample: LatencySample) -> LatencySample:
        self._samples.append(sample)
        return sample

    @property
    def samples(self) -> list[LatencySample]:
        return list(self._samples)

    def arms(self) -> dict[str, ArmLatency]:
        grouped: dict[str, list[LatencySample]] = {}
        for sample in self._samples:
            grouped.setdefault(sample.arm, []).append(sample)
        return {name: summarise_arm(rows) for name, rows in grouped.items()}

    def rows(self) -> list[dict[str, Any]]:
        return [arm.as_row() for arm in self.arms().values()]


__all__ = [
    "ArmLatency",
    "ChunkTimer",
    "LatencyLedger",
    "LatencySample",
    "LatencyStats",
    "percentile",
    "summarise_arm",
]
