"""Offline load & concurrency verification for the answerdesk service.

The harness drives the **real ASGI app** in this process through ``httpx.ASGITransport``:
middleware, routes, the SQLite store, the trace writer and the metrics registry all run,
and the only double is the model provider (the offline fake). No socket is opened, so the
run is deterministic, network-free and safe for CI.

What it measures, per concurrency level and per cache phase:

* client-observed latency (p50/p95/p99/max) and throughput for a mixed workload
  (answerable/unanswerable × streaming/JSON);
* per-stage latency (retrieval, generation, verification, policy, cache, total) read from
  the **service traces**, so a bottleneck is attributed to a stage, not guessed;
* cache separation: the cold phase replays distinct questions, the warm phase repeats them,
  and the hit/miss counts come from the traces themselves (`cache_status`);
* process resources (RSS, handles, threads) before, during and after the run, plus leak
  waves at the highest concurrency;
* rate limiting under load and the executor's backpressure (requests queue instead of the
  process growing).

Usage::

    # CI gate (< 30 s by contract, typically ~5 s)
    uv run python scripts/load.py --smoke

    # Full pass: 1 / 8 / 32 / 64 clients, cold + warm, three leak waves
    uv run python scripts/load.py --full --json reports/load/service-load.json

    # One concurrency level only
    uv run python scripts/load.py --levels 8

Exit code is 0 only when every assertion holds: no 5xx, no timeouts, no unhandled
exception, bounded memory/handle growth, no leaked idempotency locks, and the rate limiter
returning ``429`` with a ``Retry-After`` hint.

    # AT-OPS-07  # NFR-LAT-01  # NFR-OPS-01
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import json
import math
import os
import platform
import random
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # the api extra is required; the message below is the operator-facing error
    import httpx
except ModuleNotFoundError:  # pragma: no cover - exercised by the base install only
    print(
        "answerdesk: the load harness needs the API stack: uv sync --extra api --extra dev",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

from answerdesk.config import load_settings
from answerdesk.generate.demo import demo_responses
from answerdesk.llm.presets import provider_config
from answerdesk.service.app import create_app
from answerdesk.service.config import ServiceSettings
from answerdesk.service.runtime import ServiceRuntime
from answerdesk.telemetry.tracing import JsonlTraceWriter
from answerdesk.testing.doubles import FakeProvider

#: The question the demo corpus answers with two verified citations.
ANSWERABLE_MESSAGES: tuple[str, ...] = (
    "обслуживание карты",
    "Сколько стоит годовое обслуживание карты «Вектор»?",
    "карта «Вектор» обслуживание тариф",
    "тарифы по карте Вектор",
    "Какая комиссия за перевод?",
    "лимиты переводов по карте",
    "как заменить карту",
    "KYC requirements documents",
    "Kartani almashtirish kerak",
    "Kartani qanday almashtirish mumkin?",
    "Картаны кантип алмаштырса болот?",
    "Комиссия за обслуживание",
)

#: Questions the corpus cannot answer (out-of-scope): they must escalate, not fail.
UNANSWERABLE_MESSAGES: tuple[str, ...] = (
    "Какие часы работы у офиса в Антарктиде?",
    "Сколько стоит билет на Марс?",
    "Расскажи рецепт борща",
    "What is the weather in Reykjavik tomorrow?",
    "Кто выиграл чемпионат мира по шахматам в 1972 году?",
    "Сколько планет в Солнечной системе?",
)

#: Workload kinds, as (label, answerable, stream).
WORKLOAD_KINDS: tuple[tuple[str, bool, bool], ...] = (
    ("answerable-stream", True, True),
    ("answerable-json", True, False),
    ("unanswerable-stream", False, True),
    ("unanswerable-json", False, False),
)

#: Hardware/software facts recorded with every run (docs/load.md quotes them).
LIMITATIONS: tuple[str, ...] = (
    "the model provider is the in-process fake (no network, no GPU); a real provider adds "
    "time-to-first-token and is the dominant cost of a production turn",
    "the app runs in one process on one event loop; uvicorn worker count, TLS and a reverse "
    "proxy are out of scope",
    "requests are driven through httpx.ASGITransport, so there is no loopback TCP/UDP stack, "
    "no kernel socket buffers and no client-side connection pool contention",
    "the workload is synthetic and repeats a small set of questions, so cache hit rates are "
    "an upper bound, not a forecast",
    "numbers come from one machine and one run; percentiles of a 160-request phase are "
    "indicative, not statistically tight",
    "traces are collected in memory instead of the JSONL writer, which removes the per-turn "
    "fsync from the measured latency (the fsync cost is reported separately)",
)


# ---------------------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Request:
    """One planned request (deterministic: same seed -> same sequence)."""

    kind: str
    message: str
    stream: bool
    answerable: bool

    @property
    def label(self) -> str:
        return self.kind


def build_warm_workload(count: int, *, seed: int = 7) -> list[Request]:
    """The cache-hit phase: the exact demo questions, repeated.

    Only an ``answer`` is cached (an escalation has no answer to store), so a warm phase
    built from the answerable pool is what actually exercises the exact-cache path. The
    cold phase keeps the full mix, and both phases report the cache status they observed
    instead of assuming one.
    """
    rng = random.Random(seed)
    requests: list[Request] = []
    for index in range(count):
        label, _answerable, stream = WORKLOAD_KINDS[
            index % 2
        ]  # answerable-stream / answerable-json
        message = ANSWERABLE_MESSAGES[rng.randrange(len(ANSWERABLE_MESSAGES))]
        requests.append(Request(kind=label, message=message, stream=stream, answerable=True))
    return requests


def answered_queries(samples: Sequence[Sample], sink: CollectingTraceWriter) -> list[str]:
    """The queries the prefill actually answered (only an answer is cached)."""
    seen: list[str] = []
    for sample in samples:
        line = sink.read(sample.trace_id, kind="turn") if sample.trace_id else None
        if line is None:
            continue
        answered = line.get("action") == "answer" or line.get("reason_code") == "ANSWERED"
        query = str(line.get("query") or "")
        if answered and query and query not in seen:
            seen.append(query)
    return seen


def build_cache_hits(queries: Sequence[str], count: int) -> list[Request]:
    """The measured warm phase: replay the prefill-answered queries, alternating stream/JSON."""
    if not queries:
        return build_warm_workload(count)
    requests: list[Request] = []
    for index in range(count):
        label, _answerable, stream = WORKLOAD_KINDS[index % 2]
        requests.append(
            Request(
                kind=label,
                message=queries[index % len(queries)],
                stream=stream,
                answerable=True,
            )
        )
    return requests


def build_warmup() -> list[Request]:
    """One pass over every distinct answerable question (not measured)."""
    return [
        Request(
            kind=WORKLOAD_KINDS[index % 2][0],
            message=message,
            stream=WORKLOAD_KINDS[index % 2][2],
            answerable=True,
        )
        for index, message in enumerate(ANSWERABLE_MESSAGES)
    ]


def build_workload(count: int, *, seed: int = 7) -> list[Request]:
    """Interleave the four workload kinds deterministically, cycling the message pools."""
    rng = random.Random(seed)
    requests: list[Request] = []
    for index in range(count):
        label, answerable, stream = WORKLOAD_KINDS[index % len(WORKLOAD_KINDS)]
        pool = ANSWERABLE_MESSAGES if answerable else UNANSWERABLE_MESSAGES
        message = pool[rng.randrange(len(pool))]
        if answerable:
            # distinct surface forms keep the *cold* phase genuinely cold while the warm
            # phase repeats the identical sequence and hits the exact cache
            message = f"{message} #{index // len(WORKLOAD_KINDS)}"
        requests.append(Request(kind=label, message=message, stream=stream, answerable=answerable))
    return requests


@dataclass
class Sample:
    """One completed request, as the client saw it."""

    kind: str
    status: int
    duration_ms: float
    trace_id: str = ""
    ok: bool = True
    timed_out: bool = False
    error: str = ""
    cache_status: str = ""
    action: str = ""
    stage_ms: dict[str, float] = field(default_factory=dict)
    done_frames: int = 0


# ---------------------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------------------
class ProcessProbe:
    """RSS / handles / threads without a third-party dependency.

    Windows: ``psapi.GetProcessMemoryInfo`` and ``kernel32.GetProcessHandleCount`` through
    ``ctypes``. POSIX: ``resource.getrusage`` and ``/proc/self/fd``. Any failure degrades to
    ``None`` rather than breaking the run.
    """

    def __init__(self) -> None:
        self._windows = sys.platform == "win32"
        self._kernel32: Any = None
        self._psapi: Any = None
        if self._windows:
            try:  # pragma: no cover - platform specific
                from ctypes import wintypes

                self._wintypes = wintypes
                self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
                self._kernel32.GetCurrentProcess.restype = wintypes.HANDLE
                self._psapi.GetProcessMemoryInfo.argtypes = [
                    wintypes.HANDLE,
                    ctypes.c_void_p,
                    wintypes.DWORD,
                ]
                self._psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                self._kernel32.GetProcessHandleCount.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                self._kernel32.GetProcessHandleCount.restype = wintypes.BOOL
            except Exception:
                self._kernel32 = self._psapi = None

    def rss_mb(self) -> float | None:
        if self._windows and self._psapi is not None:  # pragma: no cover - platform specific
            try:
                counters = _ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
                handle = self._kernel32.GetCurrentProcess()
                if not self._psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return None
                return round(counters.WorkingSetSize / (1024 * 1024), 1)
            except Exception:
                return None
        try:  # pragma: no cover - POSIX
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
            return round(usage.ru_maxrss / divisor, 1)
        except Exception:
            return None

    def peak_rss_mb(self) -> float | None:
        """The OS-recorded peak working set (no sampling, no overhead)."""
        if self._windows and self._psapi is not None:  # pragma: no cover - platform specific
            try:
                counters = _ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
                handle = self._kernel32.GetCurrentProcess()
                if not self._psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return None
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
            except Exception:
                return None
        return self.rss_mb()  # POSIX ru_maxrss is already the peak

    def handles(self) -> int | None:
        if self._windows and self._kernel32 is not None:  # pragma: no cover - platform specific
            try:
                from ctypes import wintypes

                count = wintypes.DWORD(0)
                if not self._kernel32.GetProcessHandleCount(
                    self._kernel32.GetCurrentProcess(), ctypes.byref(count)
                ):
                    return None
                return int(count.value)
            except Exception:
                return None
        try:  # pragma: no cover - POSIX
            return len(list(Path("/proc/self/fd").iterdir()))
        except Exception:
            return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "rss_mb": self.rss_mb(),
            "handles": self.handles(),
            "threads": threading.active_count(),
        }


# ---------------------------------------------------------------------------------------
# Windows process metrics (module level: ctypes needs the structure defined once)
# ---------------------------------------------------------------------------------------
if sys.platform == "win32":  # pragma: no cover - platform specific
    from ctypes import wintypes as _wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", _wintypes.DWORD),
            ("PageFaultCount", _wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
else:  # pragma: no cover - POSIX never touches it

    class _ProcessMemoryCounters:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------------------
# event loop
# ---------------------------------------------------------------------------------------
def _run_async(coro: Any) -> Any:
    """``asyncio.run`` with the loopback self-pipe allowed.

    ``tests/conftest.py`` blocks every ``socket.connect`` (NFR-DET-01). On Windows the
    proactor loop builds its self-pipe with ``socket.socketpair()``, which is a loopback
    connect and not network access; the harness therefore re-allows loopback for the
    duration of the loop and nothing else. Standalone runs are unaffected.
    """
    import _socket
    import socket

    original = _socket.socket.connect
    guard = socket.socket.connect

    def _loopback_allowed(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else address
        if host in ("127.0.0.1", "::1", "localhost", ""):
            return original(self, address, *args, **kwargs)
        return guard(self, address, *args, **kwargs)

    socket.socket.connect = _loopback_allowed  # type: ignore[method-assign]
    try:
        return asyncio.run(coro)
    finally:
        socket.socket.connect = guard  # type: ignore[method-assign]


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default), ``None`` for an empty sample."""
    if not values:
        return None
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be within 0..100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": _round(percentile(values, 50)),
        "p95": _round(percentile(values, 95)),
        "p99": _round(percentile(values, 99)),
        "max": _round(max(values) if values else None),
        "mean": _round(sum(values) / len(values) if values else None),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _stage_summary(samples: Sequence[Sample]) -> dict[str, dict[str, float | None]]:
    stages: dict[str, list[float]] = {}
    for sample in samples:
        for stage, value in sample.stage_ms.items():
            stages.setdefault(stage, []).append(float(value))
    return {stage: latency_summary(values) for stage, values in sorted(stages.items())}


# ---------------------------------------------------------------------------------------
# trace sink
# ---------------------------------------------------------------------------------------
class CollectingTraceWriter:
    """An in-memory ``TraceWriter``: the service writes the same dict line as JSONL."""

    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []

    def write(self, line: Any) -> None:
        self.lines.append(dict(line))

    def read(self, trace_id: str, *, kind: str | None = None) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        for line in reversed(self.lines):
            if line.get("trace_id") != trace_id:
                continue
            if kind is None or line.get("kind") == kind:
                return line
            fallback = fallback or line
        return fallback

    def close(self) -> None:  # parity with JsonlTraceWriter
        return None

    @property
    def path(self) -> str:
        return "memory://load-harness"


# ---------------------------------------------------------------------------------------
# app factory + one phase
# ---------------------------------------------------------------------------------------
def build_app(
    *,
    workdir: Path,
    rate_limit_per_minute: int = 100_000,
    rate_limit_burst: int | None = None,
    tracer: Any = None,
    real_tracer: bool = False,
) -> tuple[Any, ServiceRuntime, Any]:
    """Build the real app around an isolated SQLite store and an in-memory trace sink."""
    core = load_settings(profile="offline")
    settings = ServiceSettings(
        db_path=str(workdir / "load.db"),
        trace_dir=str(workdir / "traces"),
        rate_limit_per_minute=rate_limit_per_minute,
        rate_limit_burst=rate_limit_burst or rate_limit_per_minute,
        idempotency_ttl_s=3600,
        sse_token_chars=24,
        calibration_max_age_days=365,
    )
    if real_tracer:
        # the production writer: one fsync per turn line (durability cost included)
        sink: Any = JsonlTraceWriter(workdir / "traces.jsonl")
    else:
        sink = tracer or CollectingTraceWriter()
    provider = FakeProvider(provider_config("fake"), responses=demo_responses())
    runtime = ServiceRuntime.create(core, settings, fake=True, provider=provider, tracer=sink)
    app = create_app(settings=core, service_settings=settings, runtime=runtime)
    return app, runtime, sink


def _count_done_frames(body: str) -> int:
    return body.count("event: done")


async def _drive(app: Any, requests: Sequence[Request], concurrency: int) -> list[Sample]:
    """Run ``requests`` with at most ``concurrency`` in flight; order is preserved."""
    results: list[Sample | None] = [None] * len(requests)
    transport = httpx.ASGITransport(app=app)
    queue: asyncio.Queue[tuple[int, Request]] = asyncio.Queue()
    for item in enumerate(requests):
        queue.put_nowait(item)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://load.test", timeout=60.0
    ) as client:

        async def worker() -> None:
            while True:
                try:
                    index, request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results[index] = await _one(client, request)

        await asyncio.gather(*[worker() for _ in range(max(1, concurrency))])
    return [sample for sample in results if sample is not None]


async def _one(client: httpx.AsyncClient, request: Request) -> Sample:
    started = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            client.post(
                "/v1/chat",
                json={"message": request.message, "stream": request.stream},
                headers={
                    "Accept": "text/event-stream" if request.stream else "application/json",
                    "X-Client-Id": "load-harness",
                },
            ),
            timeout=60.0,
        )
    except TimeoutError:
        return Sample(
            kind=request.kind,
            status=0,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            ok=False,
            timed_out=True,
            error="timeout",
        )
    except Exception as exc:
        return Sample(
            kind=request.kind,
            status=0,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    duration_ms = (time.perf_counter() - started) * 1000.0
    sample = Sample(
        kind=request.kind,
        status=response.status_code,
        duration_ms=duration_ms,
        trace_id=response.headers.get("X-Trace-Id", ""),
        ok=response.status_code < 500,
        done_frames=_count_done_frames(response.text) if request.stream else 0,
    )
    if response.status_code >= 400:
        sample.error = response.text[:200]
    return sample


def _enrich(samples: Sequence[Sample], sink: CollectingTraceWriter) -> None:
    """Attach the trace's stage timings, cache status and decision to each sample."""
    for sample in samples:
        if not sample.trace_id:
            continue
        line = sink.read(sample.trace_id, kind="turn")
        if line is None:
            continue
        sample.cache_status = str(line.get("cache_status") or "")
        sample.action = str(line.get("reason_code") or line.get("action") or "")
        stage = line.get("stage_latency_ms") or {}
        sample.stage_ms = {
            str(name): float(value)
            for name, value in stage.items()
            if isinstance(value, (int, float))
        }


# ---------------------------------------------------------------------------------------
# one level
# ---------------------------------------------------------------------------------------
@dataclass
class PhaseResult:
    concurrency: int
    phase: str
    samples: list[Sample]
    wall_s: float
    resources_before: dict[str, Any]
    resources_after: dict[str, Any]
    peak_rss_mb: float | None = None
    pending_tasks: int = 0

    def as_dict(self) -> dict[str, Any]:
        durations = [sample.duration_ms for sample in self.samples]
        statuses = Counter(str(sample.status) for sample in self.samples)
        cache = Counter(sample.cache_status or "unknown" for sample in self.samples)
        actions = Counter(sample.action or "unknown" for sample in self.samples)
        return {
            "concurrency": self.concurrency,
            "phase": self.phase,
            "requests": len(self.samples),
            "wall_s": round(self.wall_s, 3),
            "throughput_rps": round(len(self.samples) / self.wall_s, 2) if self.wall_s else None,
            "status_counts": dict(sorted(statuses.items())),
            "errors": sum(1 for sample in self.samples if not sample.ok),
            "timeouts": sum(1 for sample in self.samples if sample.timed_out),
            "error_rate": round(
                sum(1 for sample in self.samples if not sample.ok) / len(self.samples), 4
            )
            if self.samples
            else None,
            "latency_ms": latency_summary(durations),
            "latency_by_kind": {
                kind: latency_summary(
                    [sample.duration_ms for sample in self.samples if sample.kind == kind]
                )
                for kind in sorted({sample.kind for sample in self.samples})
            },
            "stages_ms": _stage_summary(self.samples),
            "cache": dict(sorted(cache.items())),
            "decisions": dict(sorted(actions.items())),
            "done_frames_missing": sum(
                1
                for sample in self.samples
                if sample.kind.endswith("stream") and sample.ok and not sample.done_frames
            ),
            "resources_before": self.resources_before,
            "resources_after": self.resources_after,
            "peak_rss_mb": self.peak_rss_mb,
            "pending_tasks": self.pending_tasks,
        }


async def _phase(
    app: Any,
    requests: Sequence[Request],
    *,
    concurrency: int,
    phase: str,
    probe: ProcessProbe,
    sink: CollectingTraceWriter,
) -> PhaseResult:
    """Measure one phase. The caller owns the app lifespan (the cache must stay warm)."""
    before = probe.snapshot()
    gc.collect()
    started = time.perf_counter()
    samples = await _drive(app, requests, concurrency)
    wall_s = time.perf_counter() - started
    # every request task must have finished: a growing set here would be an unbounded queue
    pending = len([task for task in asyncio.all_tasks() if not task.done()])
    after = probe.snapshot()
    _enrich(samples, sink)
    return PhaseResult(
        concurrency=concurrency,
        phase=phase,
        samples=samples,
        wall_s=wall_s,
        resources_before=before,
        resources_after=after,
        peak_rss_mb=probe.peak_rss_mb(),
        pending_tasks=pending,
    )


# ---------------------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------------------
class CallTimer:
    """Time a method call (including the lock wait) without changing its behaviour."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._lock = threading.Lock()
        self._original: Any = None
        self._target: Any = None
        self._name = ""

    def wrap(self, target: Any, name: str) -> None:
        self._target = target
        self._name = name
        self._original = getattr(target, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return self._original(*args, **kwargs)
            finally:
                with self._lock:
                    self.samples.append((time.perf_counter() - started) * 1000.0)

        setattr(target, name, timed)

    def restore(self) -> None:
        if self._target is not None and self._original is not None:
            setattr(self._target, self._name, self._original)

    def summary(self) -> dict[str, Any]:
        return {"samples": len(self.samples), "latency_ms": latency_summary(self.samples)}


def run_level(
    *,
    workdir: Path,
    concurrency: int,
    count: int,
    probe: ProcessProbe,
    real_tracer: bool = False,
) -> dict[str, Any]:
    """Cold then warm phase at one concurrency level, on a fresh app."""
    workload = build_workload(count)
    app, runtime, sink = build_app(workdir=workdir / f"c{concurrency}", real_tracer=real_tracer)
    warm_pool: list[str] = []
    persist_timer = CallTimer()
    trace_timer = CallTimer()
    persist_timer.wrap(runtime.store, "append_turn")
    trace_timer.wrap(runtime.tracer, "write")

    async def _run() -> tuple[PhaseResult, PhaseResult]:
        async with app.router.lifespan_context(app):
            # warm-up: first-touch costs (thread pool, SQLite pages, index) are not the
            # service's steady state and would otherwise show up as a p99 outlier
            await _drive(app, build_warmup(), 1)
            cold = await _phase(
                app, workload, concurrency=concurrency, phase="cold", probe=probe, sink=sink
            )
            # prefill: ask every answerable question once, keep what was answered, then
            # measure the replay — that is the exact-cache path, not a guess about it
            prefill = await _drive(app, build_warm_workload(count), 1)
            _enrich(prefill, sink)
            warm_pool.extend(answered_queries(prefill, sink))
            warm_workload = build_cache_hits(warm_pool, count)
            warm = await _phase(
                app, warm_workload, concurrency=concurrency, phase="warm", probe=probe, sink=sink
            )
        return cold, warm

    try:
        cold, warm = _run_async(_run())
    finally:
        persist_timer.restore()
        trace_timer.restore()
        runtime.close()
    return {
        "level": {
            "concurrency": concurrency,
            "requests_per_phase": count,
            "warm_queries": len(warm_pool),
            "tracer": "jsonl" if real_tracer else "memory",
        },
        "cold": cold.as_dict(),
        "warm": warm.as_dict(),
        "instrumentation": {
            "append_turn_ms": persist_timer.summary(),
            "trace_write_ms": trace_timer.summary(),
            "note": (
                "time spent inside the call, including waiting for the store's lock; "
                "samples cover warm-up, cold, prefill and warm phases"
            ),
        },
    }


def run_leak_waves(
    *,
    workdir: Path,
    concurrency: int,
    count: int,
    waves: int,
    probe: ProcessProbe,
) -> dict[str, Any]:
    """Repeat the same workload at max concurrency and watch RSS/handles/threads."""
    workload = build_workload(count)
    app, runtime, sink = build_app(workdir=workdir / "leak")
    snapshots: list[dict[str, Any]] = []

    async def _run() -> None:
        async with app.router.lifespan_context(app):
            for wave in range(waves):
                await _phase(
                    app,
                    workload,
                    concurrency=concurrency,
                    phase=f"wave-{wave + 1}",
                    probe=probe,
                    sink=sink,
                )
                snapshots.append(probe.snapshot())

    try:
        _run_async(_run())
    finally:
        runtime.close()

    from answerdesk.service import routes as service_routes  # private on purpose: leak check

    rss_values = [snapshot["rss_mb"] for snapshot in snapshots if snapshot["rss_mb"] is not None]
    handle_values = [
        snapshot["handles"] for snapshot in snapshots if snapshot["handles"] is not None
    ]
    first, last = snapshots[0], snapshots[-1]
    return {
        "waves": snapshots,
        "rss_growth_mb": round(rss_values[-1] - rss_values[0], 1) if len(rss_values) >= 2 else None,
        "handle_growth": handle_values[-1] - handle_values[0] if len(handle_values) >= 2 else None,
        "thread_growth": last["threads"] - first["threads"],
        "idempotency_locks": len(service_routes._IDEMPOTENCY_LOCKS),
        "traces_collected": len(sink.lines),
    }


def run_rate_limit(workdir: Path, *, concurrency: int = 16, count: int = 96) -> dict[str, Any]:
    """A tight bucket under load: 429 + Retry-After, health exempt, never a 5xx."""
    app, runtime, _sink = build_app(
        workdir=workdir / "rate", rate_limit_per_minute=60, rate_limit_burst=60
    )
    workload = build_workload(count, seed=11)

    async def _run() -> tuple[list[Sample], int, int]:
        async with app.router.lifespan_context(app):
            samples = await _drive(app, workload, concurrency)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://load.test"
            ) as client:
                healthy = await client.get("/healthz")
                metrics = await client.get("/metrics")
            return samples, healthy.status_code, metrics.status_code

    try:
        samples, health_status, metrics_status = _run_async(_run())
    finally:
        runtime.close()

    limited = [sample for sample in samples if sample.status == 429]
    return {
        "requests": len(samples),
        "concurrency": concurrency,
        "status_counts": dict(sorted(Counter(str(sample.status) for sample in samples).items())),
        "http_429": len(limited),
        "server_errors": sum(1 for sample in samples if sample.status >= 500),
        "timeouts": sum(1 for sample in samples if sample.timed_out),
        "retry_after_present": all("retry_after" in sample.error for sample in limited)
        if limited
        else None,
        "health_status": health_status,
        "metrics_status": metrics_status,
    }


# ---------------------------------------------------------------------------------------
# assertions + reporting
# ---------------------------------------------------------------------------------------
def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    """Turn the numbers into the pass/fail assertions the CLI exits on."""
    failures: list[str] = []
    for level in report.get("levels", []):
        for phase in ("cold", "warm"):
            data = level[phase]
            if data["status_counts"].get("5", 0) or any(
                int(status) >= 500 for status in data["status_counts"]
            ):
                failures.append(f"5xx at concurrency {level['level']['concurrency']} ({phase})")
            if data["timeouts"]:
                failures.append(
                    f"timeouts at concurrency {level['level']['concurrency']} ({phase})"
                )
            if data["errors"]:
                failures.append(
                    f"{data['errors']} failing requests at concurrency "
                    f"{level['level']['concurrency']} ({phase})"
                )
            if data["done_frames_missing"]:
                failures.append(
                    f"{data['done_frames_missing']} streams without a done frame at "
                    f"{level['level']['concurrency']} ({phase})"
                )
            if data.get("pending_tasks", 0) > 2:
                failures.append(
                    f"{data['pending_tasks']} pending tasks after concurrency "
                    f"{level['level']['concurrency']} ({phase})"
                )
            if not data["stages_ms"]:
                failures.append(
                    f"no per-stage timings at concurrency {level['level']['concurrency']} ({phase})"
                )
    leaks = report.get("leaks", {})
    if leaks.get("rss_growth_mb") is not None and leaks["rss_growth_mb"] > 96:
        failures.append(f"RSS grew by {leaks['rss_growth_mb']} MB across leak waves")
    if leaks.get("handle_growth") is not None and leaks["handle_growth"] > 64:
        failures.append(f"handle count grew by {leaks['handle_growth']} across leak waves")
    if leaks.get("thread_growth", 0) > 8:
        failures.append(f"thread count grew by {leaks['thread_growth']} across leak waves")
    if leaks.get("idempotency_locks"):
        failures.append(f"{leaks['idempotency_locks']} idempotency locks leaked")
    rate = report.get("rate_limit", {})
    if rate.get("server_errors"):
        failures.append("the rate limiter produced 5xx responses")
    if rate.get("timeouts"):
        failures.append("the rate limiter produced timeouts")
    if rate.get("http_429") is None or rate["http_429"] <= 0:
        failures.append("the rate limiter never returned 429 under load")
    if rate.get("retry_after_present") is False:
        failures.append("a 429 response carried no Retry-After hint")
    if rate.get("health_status") != 200 or rate.get("metrics_status") != 200:
        failures.append("health/metrics were affected by the rate limiter")
    return {"passed": not failures, "failures": failures}


def _summary_table(report: dict[str, Any]) -> str:
    header = (
        f"{'conc':>4} {'phase':>5} {'req':>4} {'rps':>8} {'p50':>7} {'p95':>7} "
        f"{'p99':>7} {'err':>3} {'hit':>6}  peak-rss"
    )
    lines = [header, "-" * len(header)]
    for level in report["levels"]:
        for phase in ("cold", "warm"):
            data = level[phase]
            latency = data["latency_ms"]
            hits = data["cache"].get("hit", 0)
            lines.append(
                f"{level['level']['concurrency']:>4} {phase:>5} {data['requests']:>4} "
                f"{data['throughput_rps']:>8} {_fmt(latency['p50']):>7} {_fmt(latency['p95']):>7} "
                f"{_fmt(latency['p99']):>7} {data['errors']:>3} {hits:>6}  {data['peak_rss_mb']} MB"
            )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


def _service_info(runtime: ServiceRuntime) -> dict[str, Any]:
    info = runtime.info
    return {
        "profile": info.profile,
        "fake": info.fake,
        "corpus_version": info.corpus_version,
        "index_version": info.index_version,
        "calibration": info.calibration.threshold_ref,
        "model": info.model,
    }


def run(
    *,
    mode: str = "smoke",
    levels: Sequence[int] | None = None,
    requests_per_phase: int | None = None,
    leak_waves: int = 3,
    real_tracer: bool = False,
) -> dict[str, Any]:
    """Run the whole scenario set and return the serialisable report."""
    smoke = mode == "smoke"
    level_list = list(levels or ([1, 8] if smoke else [1, 8, 32, 64]))
    count = requests_per_phase or (24 if smoke else 160)
    max_concurrency = max(level_list)

    tmp = tempfile.TemporaryDirectory(prefix="answerdesk-load-")
    workdir = Path(tmp.name)
    probe = ProcessProbe()
    baseline = probe.snapshot()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "machine": _machine(),
        "workload": {
            "requests_per_phase": count,
            "levels": level_list,
            "kinds": [label for label, _answerable, _stream in WORKLOAD_KINDS],
            "cache": (
                "cold = mixed first-seen questions; warm = exact replay of the answerable "
                "pool after a prefill pass"
            ),
        },
        "limitations": list(LIMITATIONS),
        "baseline_resources": baseline,
        "tracer": "jsonl" if real_tracer else "memory",
        "executor": {
            "asyncio_to_thread": "default ThreadPoolExecutor",
            "max_workers": min(32, (os.cpu_count() or 1) + 4),
            "note": (
                "each turn runs in a worker thread; beyond max_workers requests queue in the "
                "executor, which is the service's backpressure mechanism (latency grows, "
                "nothing is dropped)"
            ),
        },
        "levels": [],
    }

    # one throwaway app for service identity (the levels build their own)
    _app, identity_runtime, _sink = build_app(workdir=workdir / "identity")
    report["service"] = _service_info(identity_runtime)
    identity_runtime.close()

    started = time.perf_counter()
    for concurrency in level_list:
        report["levels"].append(
            run_level(
                workdir=workdir,
                concurrency=concurrency,
                count=count,
                probe=probe,
                real_tracer=real_tracer,
            )
        )
    report["levels_wall_s"] = round(time.perf_counter() - started, 2)

    report["leaks"] = run_leak_waves(
        workdir=workdir,
        concurrency=max_concurrency,
        count=max(12, count // 2),
        waves=max(2, leak_waves),
        probe=probe,
    )
    report["rate_limit"] = run_rate_limit(
        workdir, concurrency=max(8, max_concurrency // 2), count=96
    )
    report["final_resources"] = probe.snapshot()
    report["assertions"] = evaluate(report)
    report["total_wall_s"] = round(time.perf_counter() - started, 2)
    tmp.cleanup()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--smoke", action="store_true", help="CI mode: 1 and 8 clients, < 30 s")
    parser.add_argument("--full", action="store_true", help="full mode: 1, 8, 32 and 64 clients")
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help="comma-separated concurrency levels (overrides --smoke/--full)",
    )
    parser.add_argument("--requests", type=int, default=0, help="requests per phase")
    parser.add_argument("--waves", type=int, default=3, help="leak waves at max concurrency")
    parser.add_argument("--json", type=str, default="", help="write the full report to this path")
    parser.add_argument(
        "--tracer",
        choices=("memory", "jsonl"),
        default="memory",
        help="memory (default, measures the service without fsync) or jsonl (production writer)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = parser.parse_args(argv)

    if args.levels:
        levels = [int(part) for part in args.levels.split(",") if part.strip()]
    elif args.full or not args.smoke:
        levels = [1, 8, 32, 64]
    else:
        levels = [1, 8]
    mode = "smoke" if args.smoke and not args.full and not args.levels else "full"
    if args.requests:
        count = args.requests
        mode = "custom"
    else:
        count = 24 if mode == "smoke" else 160

    report = run(
        mode=mode,
        levels=levels,
        requests_per_phase=count,
        leak_waves=args.waves,
        real_tracer=args.tracer == "jsonl",
    )

    if not args.quiet:
        print(_summary_table(report))
        print()
        leaks = report["leaks"]
        print(
            f"leak waves: rss {leaks['rss_growth_mb']} MB, handles {leaks['handle_growth']}, "
            f"threads {leaks['thread_growth']}, idempotency locks {leaks['idempotency_locks']}"
        )
        rate = report["rate_limit"]
        print(
            f"rate limit: {rate['http_429']}/{rate['requests']} → 429, "
            f"5xx {rate['server_errors']}, retry-after present {rate['retry_after_present']}, "
            f"health {rate['health_status']}, metrics {rate['metrics_status']}"
        )
        print(f"total wall: {report['total_wall_s']} s")
        print()
    verdict = report["assertions"]
    print(f"LOAD {'PASS' if verdict['passed'] else 'FAIL'} ({report['mode']})")
    for failure in verdict["failures"]:
        print(f"  - {failure}")

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report: {target}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - the CLI entry point
    raise SystemExit(main())
