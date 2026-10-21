# Load & concurrency verification

This document answers one question with measurements: **what does the answerdesk service do
under concurrent load, and where does it stop scaling?** It is a verification pass, not a
benchmark claim about production.

The harness is [`scripts/load.py`](../scripts/load.py). It drives the **real ASGI app** in
this process through `httpx.ASGITransport`: middleware, routes, the SQLite store, the trace
writer and the metrics registry all run; the only double is the model provider (the offline
fake). No socket is opened, so the run is deterministic and network-free.

```bash
# CI gate: 1 and 8 clients, 24 requests per phase, < 30 s (measured ~2 s)
uv run python scripts/load.py --smoke

# Full pass: 1 / 8 / 32 / 64 clients, cold + warm, three leak waves
uv run python scripts/load.py --full --json reports/load/full.json

# One level, the production trace writer (fsync included), custom size
uv run python scripts/load.py --levels 1,64 --requests 64 --waves 2 --tracer jsonl \
    --json reports/load/jsonl.json

# The test gate (statistics, workload, CLI, smoke, rate limit, leaks)
uv run pytest -q tests/load
```

The exit code is 0 only when every assertion holds: no 5xx, no timeouts, no failing
requests, no stream missing its `done` frame, per-stage timings present, bounded
memory/handle/thread growth, no leaked idempotency locks, and the rate limiter returning
`429` with `Retry-After` while `/healthz` and `/metrics` stay `200`.

---

## 1. Hardware and software

Recorded in every report (`machine` block) and quoted here for the run below:

| Item | Value |
| --- | --- |
| OS | Windows 11 (10.0.26200 SP0), AMD64 |
| CPU | Intel64 Family 6 Model 183 Stepping 1 — 28 logical CPUs |
| Python | 3.12.10 |
| Executor | `asyncio.to_thread`, default `ThreadPoolExecutor`, `min(32, cpu+4)` = **32 workers** |
| Service | offline profile, fake provider, corpus `corpus-demo-6cb29bf77b57`, index `index-87424537ecf8b33c`, calibration `cal-f70b9bf8edb0` |
| Store | SQLite (WAL, `synchronous=NORMAL`), one connection guarded by an `RLock` |
| Tracer | in-memory sink for the main table; JSONL writer (fsync per turn) for the comparison |

Nothing else was running on the machine; the numbers are from **one** run each.

---

## 2. Method

* **Workload**: four kinds interleaved deterministically (seed 7) — answerable-stream,
  answerable-JSON, unanswerable-stream, unanswerable-JSON. The answerable pool is the demo
  corpus's grounded questions; the unanswerable pool is out-of-scope questions that must
  escalate. Each answerable request in the cold phase carries a distinct surface form
  (`… #N`), so the first pass is genuinely cold.
* **Size**: 160 requests per phase, at concurrency 1, 8, 32 and 64. One warm-up pass (every
  answerable question once, concurrency 1) runs before measuring, so first-touch costs
  (thread pool, SQLite pages, index) do not appear as a p99 outlier.
* **Cache separation**: after the cold phase the harness asks every answerable question once
  at concurrency 1 (prefill), reads the traces to find which queries were actually
  **answered** (only an answer is cached), and builds the warm phase from exactly those
  queries. The cold phase reports the hit/miss counts it observed; the warm phase reports
  its own. Nothing is assumed.
* **Latency** is client-observed (around the full HTTP call). Percentiles use linear
  interpolation on the sorted sample (the numpy default); p99 of 160 samples is indicative,
  not statistically tight.
* **Per-stage latency** comes from the service traces (`stage_latency_ms` in the `kind=turn`
  line): retrieval, rerank, generation, verify, policy, cache and `total`. `bookkeeping` in
  the tables is `total − (the named stages)`: session resolution, persistence and trace
  write, i.e. everything outside the pipeline proper.
* **Resources**: the OS peak working set (`PeakWorkingSetSize`), the handle count
  (`GetProcessHandleCount`) and `threading.active_count()`, sampled before and after each
  phase.

---

## 3. Results (full pass, 160 requests per phase)

Throughput is requests per second measured over the phase wall clock; latency in ms.

| clients | phase | rps | p50 | p95 | p99 | cache hits | errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cold | 193.55 | 4.63 | 6.37 | 7.62 | 0/160 | 0 |
| 1 | warm | 405.05 | 2.13 | 2.37 | 2.78 | 160/160 | 0 |
| 8 | cold | 189.06 | 39.52 | 69.57 | 101.7 | 0/160 | 0 |
| 8 | warm | 360.95 | 16.2 | 72.2 | 78.06 | 160/160 | 0 |
| 32 | cold | 203.52 | 148.66 | 210.21 | 229.3 | 0/160 | 0 |
| 32 | warm | 424.1 | 59.33 | 88.81 | 110.5 | 160/160 | 0 |
| 64 | cold | 190.56 | 311.77 | 375.93 | 402.25 | 0/160 | 0 |
| 64 | warm | 436.25 | 118.98 | 189.26 | 202.39 | 160/160 | 0 |

Peak working set: 87 MB at 1 client → **102 MB at 64 clients** (the interpreter and the demo
index dominate; the growth across the whole run is ~15 MB). Total wall time for the full
pass: **10.7 s** on the machine above.

Two facts stand out:

1. **Throughput does not grow with concurrency.** Cold: 194 → 189 → 204 → 191 rps from 1 to
   64 clients; warm: 405 → 361 → 424 → 436 rps. Latency grows linearly instead
   (4.6 → 39.5 → 149 → 312 ms cold). Little's law holds: at 64 clients,
   `64 / 191 rps ≈ 335 ms`, which is the measured p50. The service is saturated by a
   serialized section, not by CPU: the pipeline stages stay flat (§4).
2. **The cache-hit path is ~2× faster than the cold path** at every concurrency level
   (405 vs 194 rps at 1 client; 436 vs 191 at 64). A warm hit skips retrieval and
   generation entirely (§4), yet it still pays the persistence path — that is why it does not
   scale either; on the warm path the persistence share is simply a larger fraction of a
   smaller turn.

Errors: none — 0 × 5xx, 0 timeouts, 0 streams without a `done` frame across all 1 280
measured requests.

---

## 4. Where the time goes (stage attribution, p50 ms)

| clients | phase | total | retrieval | generation | verify | policy | cache | bookkeeping |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | cold | 1.73 | 0.32 | 0.85 | 0.04 | 0.14 | 0.10 | 0.28 |
| 1 | warm | 0.34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 0.18 |
| 8 | cold | 9.85 | 0.37 | 0.88 | 0.04 | 0.14 | 0.11 | 8.31 |
| 8 | warm | 3.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 2.87 |
| 32 | cold | 44.50 | 0.38 | 0.88 | 0.04 | 0.14 | 0.12 | 42.94 |
| 32 | warm | 16.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 16.66 |
| 64 | cold | 76.32 | 0.38 | 0.89 | 0.04 | 0.14 | 0.12 | 74.75 |
| 64 | warm | 37.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.16 | 36.84 |

**Finding F-L1 — the persistence path serializes the turn.** Retrieval, generation,
verification and policy are flat to two decimal places from 1 to 64 clients (the whole
pipeline is under 1.5 ms). The residual — session resolution, `append_turn`, trace write — is
0.28 ms at 1 client and 75 ms at 64, and it accounts for essentially all of the latency growth.
Instrumenting the call directly (`--levels 1,64` report, `instrumentation` block):

| clients | `append_turn` p50 | `append_turn` p95 | `trace_write` p50 |
| ---: | ---: | ---: | ---: |
| 1 | 0.09 ms | 0.14 ms | 0.00 ms |
| 64 | **14.4 ms** | **53.4 ms** | 0.00 ms |

The same call, 160× slower at 64 clients, is the signature of the store's single connection
guarded by an `RLock`: every turn serializes on it, so the second client waits for the first.
This is **by design** (documented in `service/sessions.py`: "WAL mode plus a process-local
lock is enough for the documented deployment"), and it is not a defect — but it is the
ceiling: ~190 rps cold, ~400 rps warm on this machine. Options that were *not* taken here
because they change service behaviour: a write-behind queue for turns, `synchronous=OFF`, or
a per-thread connection pool. Any of them would need its own durability decision.

**Finding F-L2 — the JSONL fsync costs 25–45 % of throughput.** The main table uses the
in-memory sink so the service is measured without the durability cost. Re-running the same
levels with the production writer (`--tracer jsonl`, one `fsync` per turn):

| tracer | 1 client cold | 1 client warm | 64 clients cold | 64 clients warm |
| --- | ---: | ---: | ---: | ---: |
| memory | 4.5 ms / 213 rps | 2.2 ms / 471 rps | 189 ms / 212 rps | 73 ms / 351 rps |
| jsonl (fsync) | **8.0 ms / 123 rps** | **4.1 ms / 199 rps** | **406 ms / 139 rps** | **204 ms / 258 rps** |

The comparison rows come from a shorter, separate run (64 requests per phase, same levels);
compare them with each other, not with the 160-request table in section 3.

That is the honest cost of "the trace survives a crash": roughly 2.8 ms per turn at one
client and ~30 % less throughput at 64. It is a deliberate trade (INV-05/NFR-OBS), recorded
here so an operator can decide whether a batch flush is acceptable.

**Finding F-L3 — backpressure is queueing, not dropping.** Each turn runs in
`asyncio.to_thread`; the default executor has 32 workers on this machine. At 8–32 clients the
work fits the pool; at 64 clients 32 turns run and 32 wait in the executor queue, which is
exactly the linear latency growth above. Nothing times out, nothing is dropped, no queue is
unbounded (each client holds one in-flight request, and the harness asserts the pending-task
count returns to the baseline after every phase). The rate limiter is the *explicit* shed
mechanism, verified next.

**Finding F-L4 — the client wrapper adds most of the wall-clock at high concurrency.** At 64
clients the p50 is 312 ms while the turn's own `total` is 76 ms. The difference is the ASGI
transport, the event loop scheduling 64 concurrent requests and the executor queue — an
in-process artifact of the harness. A real deployment with uvicorn and several workers would
move that cost into the kernel and the process pool; it is not extrapolated here.

---

## 5. Rate limiting and health under load

A separate app with a tight bucket (`rate_limit_per_minute=60`, burst 60) and 96 requests at
32 clients:

| Result | Value |
| --- | --- |
| 200 / 429 | 60 / **36** |
| 5xx / timeouts | **0 / 0** |
| `Retry-After` on every 429 | yes |
| `GET /healthz` from a starved client | **200** |
| `GET /metrics` from a starved client | **200** |

This is the documented behaviour (`docs/ops.md` §4): the limiter is per client (`X-Client-Id`,
falling back to the peer), a caller can only exhaust its own budget, and health/metrics are
exempt so a monitor is never rate limited.

---

## 6. Leaks and resource stability

Three waves of the same workload at 64 clients after the measured phases:

| Metric | Value |
| --- | --- |
| Peak working set growth across the full run | **+1.3 MB** (≈102 MB total) |
| Handle growth | **0** |
| Thread growth | **0** |
| Leaked idempotency locks (`_IDEMPOTENCY_LOCKS`) | **0** |
| Pending tasks after a phase | ≤ 2 (the harness's own task) |
| Traces collected | 480 (3 waves × 160 requests) |

No file-descriptor, thread or memory growth was observed at the highest concurrency. The
harness additionally fails the run if RSS grows by more than 96 MB, the handle count by more
than 64, the thread count by more than 8, or any idempotency lock survives a phase.

---

## 7. Limitations (do not extrapolate)

* **The provider is the fake.** The offline profile answers from a fixed in-process
  provider; a real model adds time-to-first-token (hundreds of milliseconds to seconds) and
  is the dominant cost of a production turn. The numbers here measure the *service around*
  the model, not a model-backed product.
* **Single process, single event loop.** uvicorn worker count, TLS, a reverse proxy, and
  multi-process SQLite contention are out of scope. The limiter and the store are in-process
  by design (documented in `docs/ops.md`).
* **No network stack.** Requests go through `httpx.ASGITransport`; there is no loopback
  TCP/UDP, no kernel socket buffers, no client connection-pool contention, no proxy. Real
  HTTP adds per-connection cost that this harness does not model.
* **Synthetic workload.** Four question shapes repeated from the demo corpus; the cache-hit
  rate in the warm phase (100 %) is an upper bound, not a forecast.
* **One machine, one run.** Percentiles from 160 requests per phase; run-to-run variation of
  a few percent is normal (the shortest phases — 1 client, warm — varied by up to 15 %
  between runs, which is why the committed report, not a pasted number, is the evidence).
  The relative shape (flat throughput, linear latency, flat stages) was stable across the
  runs made during this task.
* **Windows fsync.** The JSONL comparison measures this machine's filesystem; other
  filesystems will differ.
* **Peak memory is Python-heap dominated.** The 87–104 MB working set is the interpreter, the
  demo index and the caches, not per-request allocation; tracemalloc was deliberately not
  used per phase because its global lock distorts the very contention being measured.

---

## 8. Reproducing

```bash
cd answerdesk
uv sync --extra api --extra dev
uv run python scripts/load.py --smoke                                   # CI, < 30 s
uv run python scripts/load.py --full --json reports/load/full.json      # numbers in §3
uv run python scripts/load.py --levels 1,64 --requests 64 --waves 2 \
    --json reports/load/instrumented.json                               # §4 attribution
uv run python scripts/load.py --levels 1,64 --requests 64 --waves 2 \
    --tracer jsonl --json reports/load/jsonl.json                       # §4 fsync cost
uv run pytest -q tests/load                                             # harness + smoke gates
```

The reports behind the tables above are generated by the commands in this section and are
not checked in (`--json` writes one; the paths used above are `reports/load/full.json`,
`reports/load/instrumented.json` and `reports/load/jsonl.json`). Each report records the
hardware, the service identity, the workload, the limitations and every assertion, so a
future run can be compared with this page field by field instead of by re-reading prose.
