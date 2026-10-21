# answerdesk operations guide

Everything an operator needs to *run* the service: configuration, the three profiles, the
failure-mode matrix, what the metrics mean, and a runbook for the situations that actually
happen. The endpoint reference is [`api.md`](api.md); the design rationale is
[`architecture.md`](architecture.md).

This guide covers the v1.0 reference stack (requirements NFR-OPS-01, NFR-OBS-01,
NFR-LAT-01/03, NFR-COST-03). It is deliberately **not** a deployment guide: hosting,
containers and SLAs are out of scope (NG-02). It runs on a laptop, in CI and on a single
internal VM.

---

## 1. Quickstart

### 1.1 Offline (no key, no network, no corpus build)

```bash
uv sync --extra dev
uv run pytest -q                     # the CI gate
uv run --extra api answerdesk serve --fake --port 8765 --healthcheck
uv run --extra api answerdesk serve --fake --port 8765
# in another shell:
curl -s -X POST http://127.0.0.1:8765/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"обслуживание карты","stream":false}' | jq '.answer.decision'
```

The offline profile wires an **in-memory** demo corpus, the deterministic `FakeProvider`,
the sealed calibration artifact and the real store/tracer/metrics. It is the profile CI
uses (INV-08) and the one that proves the service wiring (`--healthcheck` exits 0 only when
`/healthz` answered).

### 1.2 API model

```bash
export ANSWERDESK_PROFILE=api
export ANSWERDESK_LLM__PRESET=openai          # openai | deepseek | openrouter | openai_compatible
export OPENAI_API_KEY=sk-...                  # or ANSWERDESK_LLM__API_KEY
answerdesk config check                       # resolves the preset and prints it redacted
answerdesk index --demo                       # retrieval needs an index on disk
answerdesk serve --port 8765
```

The profile `api` refuses to start with the `fake` preset; the offline profile refuses a
real preset. That contradiction check is deliberate (`answerdesk.config.apply_profile`).

### 1.3 Local model (vLLM / llama.cpp / Ollama)

All three are OpenAI-compatible servers, so they are the same code path with different
defaults (`configs/default.yaml`, `answerdesk.llm.presets`):

```bash
# Ollama
export ANSWERDESK_PROFILE=local
export ANSWERDESK_LLM__PRESET=ollama
export ANSWERDESK_LLM__MODEL=qwen3:8b
answerdesk serve --port 8765

# vLLM (OpenAI server on :8000)
export ANSWERDESK_LLM__PRESET=vllm
export ANSWERDESK_LLM__BASE_URL=http://127.0.0.1:8000/v1
export ANSWERDESK_LLM__MODEL=Qwen/Qwen3-8B

# llama.cpp server
export ANSWERDESK_LLM__PRESET=llama_cpp
export ANSWERDESK_LLM__BASE_URL=http://127.0.0.1:8080/v1
```

The `local` profile also enables local embeddings (`bge_m3`) and the cross-encoder
reranker, which need the `local` extra (`uv sync --extra local`). On a CPU-only machine,
leave `retrieval.rerank.enabled=false`: reranking is the latency hot spot (requirements
NFR-LAT-02).

> **Why `--extra api` on some commands?** FastAPI/uvicorn live in the `api` extra so the
> base install stays CLI/library-only (NFR-PORT-03). `answerdesk serve` therefore needs
> `uv run --extra api …` (or an installed extra). The test suite imports the API stack only
> in `tests/integration/service`.

---

## 2. Configuration

### 2.1 Precedence

`explicit overrides` > `environment` > `configs/default.yaml` > built-in defaults.
`ANSWERDESK_CONFIG` points at another YAML file; `--profile` selects a profile file. Every
value is visible (redacted) with:

```bash
answerdesk config show
```

### 2.2 Core variables (owned by `answerdesk.config`)

| Variable | Example | Meaning |
| --- | --- | --- |
| `ANSWERDESK_PROFILE` | `offline` \| `api` \| `local` | profile defaults (provider, rerank, embedder) |
| `ANSWERDESK_LLM__PRESET` | `openai`, `ollama`, … | provider preset |
| `ANSWERDESK_LLM__MODEL` | `gpt-4o-mini` | model id |
| `ANSWERDESK_LLM__BASE_URL` | `http://127.0.0.1:8000/v1` | override the preset URL |
| `ANSWERDESK_LLM__API_KEY` | `sk-…` | key when the preset's env var is not used |
| `ANSWERDESK_LLM__TIMEOUT_S` | `60` | per-request timeout |
| `ANSWERDESK_LLM__MAX_RETRIES` | `3` | retries with jittered backoff |
| `ANSWERDESK_LLM__FALLBACK_PRESETS` | `["deepseek"]` | fallback chain (JSON list) |
| `ANSWERDESK_APP__CORPUS_DIR` | `data/corpus` | corpus directory |
| `ANSWERDESK_APP__INDEX_DIR` | `data/index` | index root |
| `ANSWERDESK_APP__REPORTS_DIR` | `reports` | evidence artifacts |
| `ANSWERDESK_CACHE__ENABLED` | `false` | cache kill switch (FR-CAC-08) |
| `ANSWERDESK_POLICY__ALPHA` | `{...}` | per-bucket α (JSON object) |
| `ANSWERDESK_LOGGING__LEVEL` | `INFO` | log level |

Full list: `answerdesk config show | jq keys` and `configs/default.yaml`.

### 2.3 Service variables (owned by `answerdesk.service`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANSWERDESK_SERVICE_HOST` | `127.0.0.1` | bind address (local-first) |
| `ANSWERDESK_SERVICE_PORT` | `8765` | bind port |
| `ANSWERDESK_SERVICE_DB` | `data/service/answerdesk.db` | SQLite file (sessions, turns, tickets, idempotency) |
| `ANSWERDESK_SERVICE_TRACE_DIR` | `logs/traces` | JSONL trace directory |
| `ANSWERDESK_SERVICE_ADMIN_TOKEN` | *(unset)* | bearer token; **admin is disabled until this is set** |
| `ANSWERDESK_SERVICE_RATE_LIMIT` | `60` | requests/minute per client |
| `ANSWERDESK_SERVICE_RATE_BURST` | `= limit` | token-bucket capacity |
| `ANSWERDESK_SERVICE_MAX_BODY_BYTES` | `262144` | request body cap (`413`) |
| `ANSWERDESK_SERVICE_IDEMPOTENCY_TTL_S` | `86400` | replay window |
| `ANSWERDESK_SERVICE_TRACE_MAX_BYTES` | `33554432` | trace rotation threshold (3 backups) |
| `ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT` | `reports/policy-smoke/calibration-artifact.json` | sealed calibration state |
| `ANSWERDESK_SERVICE_CALIBRATION_MAX_AGE_DAYS` | `365` | staleness ceiling (production: 7) |
| `ANSWERDESK_SERVICE_CACHE_THRESHOLDS` | `reports/cache-smoke/cache-thresholds.json` | per-cohort cache thresholds (else exact-only hits) |
| `ANSWERDESK_SERVICE_SSE_TOKEN_CHARS` | `24` | size of each `tokens` delta |
| `ANSWERDESK_SERVICE_DOCS` | `true` | serve `/docs`, `/redoc`, `/openapi.json` |
| `ANSWERDESK_SERVICE_CORS_ORIGINS` | `http://localhost:5173` | comma-separated console origins |

Paths are resolved against the repository root (or the current directory when installed).

> **The shipped calibration artifact is a synthetic study.** `reports/policy-smoke/` is
> t8's calibration *machinery* demonstration, not a claim about a model's calibration
> (INV-10; `docs/escalation.md` §7). A real deployment must run `answerdesk calibrate` on
> its own dev split, point `ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT` at the result, and set
> `CALIBRATION_MAX_AGE_DAYS=7`. `/healthz` reports the artifact's provenance and age so this
> is never invisible.

### 2.4 Admin token

```bash
export ANSWERDESK_SERVICE_ADMIN_TOKEN="$(openssl rand -hex 32)"
```

* absent → `/admin/*` always `403 admin_disabled` (the default is safe);
* wrong token → `403 forbidden`; missing header → `401 unauthorized`;
* comparison is constant-time; the token never appears in `/healthz`, logs or OpenAPI.

Rotate by restarting with a new value; there is no in-process reload (a restart is the
simplest auditable rotation).

---

## 3. Running the service

### 3.1 Process model

One process serves the API, the SQLite store and the JSONL writer.

```bash
# foreground, logs to stderr as JSON (ANSWERDESK_LOGGING__JSON=false for text)
answerdesk serve --fake --port 8765
```

**Multiple workers.** `uvicorn --workers N` would open N SQLite writers and N rate-limit
buckets: the per-client budget multiplies by N and WAL write contention appears. The
supported topology is one process; scale out by running several processes behind a proxy
that pins a session to a process, or move the store to Postgres (out of scope for v1.0).
This is a documented limitation, not a silent one.

**Healthcheck mode.** `--healthcheck` starts the server in a background thread, polls
`/healthz` for up to 30 s, prints the payload, shuts the server down and returns 0 (or 1 on
failure). It is the CI proof that the app, lifespan, runtime and health wiring all work:

```bash
uv run --extra api answerdesk serve --fake --port 8765 --healthcheck
# {"url": "http://127.0.0.1:8765", "healthy": true, "health": {...}}
```

**ASGI entry point** (for a process manager that expects an app object):

```bash
uvicorn answerdesk.service.app:app_factory --factory --port 8765
```

### 3.2 Startup sequence

1. configuration is loaded and validated (`config_error` exits 2 before binding);
2. the lifespan creates a `ServiceRuntime`: embedder → index (or in-memory demo index) →
   retriever → provider → `AnswerPipeline` → decision engine (sealed artifact) → cache →
   SQLite store → trace writer → metrics;
3. missing dependencies are recorded as **degraded**, not fatal: the process starts,
   `/healthz` explains what is missing, and the answer path returns `503` with the matching
   code (this is what lets an operator see *why* a fresh deployment cannot answer);
4. a log line `answerdesk service ready` carries profile, corpus/index versions,
   calibration reference and the degraded set.

### 3.3 Files and state

| Path | Content | Backup? |
| --- | --- | --- |
| `data/service/answerdesk.db` (+ `-wal`, `-shm`) | sessions, turns, feedback, tickets, idempotency | yes — it is the only non-rebuildable state |
| `logs/traces/traces.jsonl` (+ rotated) | redacted JSONL traces | optional (evidence) |
| `data/corpus/`, `data/index/` | corpus and index | rebuildable (`ingest`/`index`) |
| `reports/policy-smoke/`, `reports/cache-smoke/` | sealed calibration/threshold artifacts | version them with the corpus |

SQLite backup while running: `sqlite3 data/service/answerdesk.db ".backup backup.db"` (WAL
mode makes a file copy unsafe without the `-wal` file).

---

## 4. Failure modes (NFR-OPS-01)

Every row has a defined behaviour, a test that proves it and an operator action. The test
names are in `tests/integration/service/test_failure_modes.py` unless stated otherwise.

| # | Failure | Behaviour | HTTP | Test | Operator action |
| --- | --- | --- | --- | --- | --- |
| 1 | **provider timeout** | retry policy exhausts, `provider_timeout`; mid-stream it is an `error`+`done` pair | `504` | `test_provider_timeout_is_a_504`, `test_provider_failure_mid_stream_emits_error_then_done` | check the provider's latency, raise `ANSWERDESK_LLM__TIMEOUT_S`, or switch preset/fallback |
| 2 | **provider 5xx** | retries, then `provider_error`; the turn is not persisted as answered | `502` | `test_provider_5xx_is_a_502` | check the provider status page; the fallback chain (`ANSWERDESK_LLM__FALLBACK_PRESETS`) exists for this |
| 3 | **malformed model json** | one strict re-ask, then the generator escalates `PROVIDER_ERROR` — never prose | `200` (escalation) | `test_malformed_model_json_escalates_instead_of_answering` | usually a model/prompt mismatch: check `prompt_version` in the trace and whether the provider honours `json_mode` |
| 4 | **empty retrieval** | no passage survives packing → `NO_EVIDENCE` escalation, no prose | `200` (escalation) | `test_empty_retrieval_escalates_with_no_evidence` | the query is out of scope, or the index is stale/empty — check `/healthz` `index` and re-run `answerdesk index` |
| 5 | **index missing** | runtime starts degraded; the answer path refuses to invent evidence | `503 index_missing` | `test_index_missing_is_a_503`, `test_health_strict_mode_returns_503_when_degraded` | `answerdesk ingest --demo` (or your corpus), then `answerdesk index`; verify `/healthz` shows the corpus version |
| 6 | **calibration missing** | runtime starts degraded; the service refuses to answer as if calibrated | `503 calibration_missing` | `test_calibration_missing_is_a_503`, `test_calibration_missing_is_a_503_even_with_an_index` | run `answerdesk calibrate --synthetic` (machinery) or calibrate on real data and point `ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT` at the sealed artifact |
| 7 | **corpus version mismatch** | the engine is stale (`STALE_CALIBRATION`), health is `degraded`, every turn escalates | `200` (escalation) + `degraded` health | `test_corpus_version_mismatch_escalates_and_degrades_health` | re-calibrate on the current corpus, or restore the corpus the artifact was fitted on; `answerdesk calibrate` refuses dev/test leakage (INV-09) |

### 4.1 Error envelope: what the caller may see (F-5, closed)

The envelope is a contract with two audiences, and t35 made the split explicit:

| Field | Caller (HTTP/SSE) | Operator (trace + log) |
| --- | --- | --- |
| `code` | taxonomy code, stable | same |
| `retryable` / `retry_after_s` | from the taxonomy row | same |
| `message` | ours, or a **generic** per-code message for provider-family failures | — |
| `detail` | validation: field path + pydantic type slug (`body.locale: string_type`); provider: an HTTP status hint (`HTTP 429`) at most | full upstream text, redacted |
| `trace_id` | always | `trace_id` on the JSONL line |
| diagnostic | **never** | `ServiceError.diagnostic` → `service error diagnostic` log record and the trace `error` field |

Two guarantees are enforced in code, not by convention:

1. `ServiceError.__init__`/`envelope` run every caller-facing string through
   `telemetry.redaction.redact_str` and cap it at 240 characters, so even a message assembled
   elsewhere cannot carry an e-mail, card, phone or credential-shaped value;
2. provider-family errors (`provider_error`, `provider_timeout`, `provider_unreachable`,
   `provider_bad_response`, `provider_circuit_open`) never copy the upstream text into the
   envelope: the message is a fixed sentence and the only upstream fact exposed is the HTTP
   status. Validation errors expose `loc` + `type` only — never the offending value or the
   pydantic message.

**Logs.** The service does not log `exc_info` for an arbitrary exception: the JSON formatter
masks secret-named keys but not free text, so a traceback could copy an upstream message (and
any PII it quotes) into the log stream. Unhandled failures log the redacted diagnostic and the
exception type instead (limitation L-10 in `docs/robustness.md`).

The trace still records `"{ExcType}: {message} ({detail})"` (redacted by the trace writer), so
debugging an upstream failure is unchanged; `tests/property/test_error_redaction.py` covers the
4xx, 5xx, rate-limit, timeout and connection paths plus two validation payloads.

Adjacent failures that are also tested:

| Failure | Behaviour | Test |
| --- | --- | --- |
| rate limit exceeded | `429 rate_limited` + `Retry-After`; per client; `/healthz` and `/metrics` exempt | `test_rate_limited_response_carries_a_retry_hint` |
| oversized body | `413 payload_too_large` before the body is read | `test_oversized_body_is_413` |
| unknown route / method | `404 not_found` / `405 method_not_allowed` | `test_unknown_route_is_404_and_wrong_method_is_405` |
| idempotency key reuse with a different body | `409 idempotency_conflict`, the first response stays | `test_same_key_with_a_different_body_is_a_conflict` |
| concurrent retries with the same key | one turn, one model call, the rest replay the stored JSON (`X-Idempotent-Replay: true`); no `UNIQUE` violation | `test_concurrent_retries_of_one_key_run_the_turn_once` (property) |
| `Idempotency-Key` longer than 200 chars or with control characters | `400 bad_request`, nothing stored | `test_huge_idempotency_key_is_rejected` (property) |
| **cache lookup raises** (e.g. a query that normalises to `""`) | the turn continues with `cache.status=bypassed` and a reason in the trace — a cache defect never fails a request (F-02) | `test_torture_corpus_never_breaks_the_stream` (property) |
| invalid UTF-8 body / unpaired surrogate | `400 bad_request` / `422 validation_error`; an optional field is sanitised, nothing reaches SQLite unencodable (F-01) | `test_invalid_utf8_bytes_are_a_traced_400`, `test_lone_surrogate_in_the_message_is_a_traced_validation_error` |
| malformed body (empty, `null`, array, deep nesting, `NaN`, duplicate keys) | `400/413/422` with a taxonomy code and a trace id, never a 500 | `test_every_malformed_body_is_a_traced_4xx` (property) |
| unsafe `X-Trace-Id` (spaces, control characters, >128 chars) | a fresh `tr-<24 hex>` id replaces it; the header is always a safe token (F-08) | `test_client_trace_ids_are_echoed_only_when_safe` |
| provider transport error escaping the LLM layer (`httpx` timeout/reset/status) | `504 provider_timeout` / `502 provider_unreachable` / `502 provider_error`, `retryable=true` (F-05) | `test_provider_failure_is_a_taxonomised_error` (property) |
| provider SSE truncated mid-frame | the completion is empty, the generator escalates `PROVIDER_ERROR`, nothing is displayed (limitation L-3) | `test_truncated_provider_sse_is_an_empty_completion_not_a_hang` |
| client disconnects mid-stream | no exception escapes, the idempotency lock is released, the next request succeeds | `test_client_disconnect_does_not_raise_and_the_service_stays_up` |
| directive-shaped text in retrieved evidence | `INJECTION_SUSPECTED` escalation with **zero** provider calls | `test_injection_in_retrieved_evidence_is_escalated_without_a_model_call` |
| PII in retrieved evidence | redacted (`[CARD]`, `[EMAIL]`, `[PHONE]`) before every provider attempt, including retries | `test_pii_in_evidence_is_redacted_before_the_provider` |
| secret-shaped value pasted into a question | `[SECRET]` in the trace, the log and the provider body (F-06) | `test_secrets_in_the_query_are_redacted_in_traces_and_prompts` |
| U+2028/U+2029 in a query | escaped in the SSE frame, so strict clients can still parse it (F-07) | `test_torture_corpus_never_breaks_the_stream` |
| malformed config file / env var | CLI exits 2 with `config error: …`, no traceback, nothing starts | `test_bad_config_exits_two_without_a_traceback` (property) |
| admin disabled / wrong token | `403 admin_disabled` / `403 forbidden` / `401 unauthorized` | `test_admin_is_disabled_without_a_configured_token`, `test_admin_requires_the_bearer_token` |
| SQLite write failure | `500 internal_error`; the turn is not reported as persisted | (unit) `tests/integration/service/test_errors.py` covers the mapping |

**The rule behind the table:** a dependency failure never becomes a confident answer. The
service answers only when the decision engine says `answer`; everything else escalates with
a reason code and an operator ticket.

The adversarial pass behind these rows — what was fuzzed, every defect it found (F-01…F-08)
and the limits it does **not** cover — is documented in
[`docs/robustness.md`](robustness.md). Its suite is `tests/property/` (102 tests; run it with
`uv run pytest -q tests/property tests/integration/service`).

---

## 5. Observability

### 5.1 Metrics (`GET /metrics`, Prometheus text format)

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `answerdesk_http_requests_total` | counter | `method`, `route`, `status` | requests handled; `route` is the template (`/v1/chat`), not the raw path |
| `answerdesk_http_request_duration_seconds` | histogram | `method`, `route` | wall clock, arrival → final byte |
| `answerdesk_http_errors_total` | counter | `code` | error responses by taxonomy code |
| `answerdesk_stage_duration_seconds` | histogram | `stage` | per-stage latency (`analysis`, `retrieval`, `rerank`, `generation`, `verify`, `policy`, `cache`, `total`) |
| `answerdesk_decisions_total` | counter | `action`, `reason` | policy decisions |
| `answerdesk_escalations_total` | counter | `reason` | turns escalated to a human |
| `answerdesk_refusals_total` | counter | `reason` | turns refused |
| `answerdesk_cache_hits_total` | counter | `kind` (`exact`, `semantic`, `negative`) | cache hits |
| `answerdesk_cache_misses_total` | counter | — | misses (including guard demotions) |
| `answerdesk_cache_bypass_total` | counter | — | bypassed (kill switch / high-risk intent) |
| `answerdesk_cache_wrong_hits_total` | counter | — | hits later labelled wrong (the error-budget numerator) |
| `answerdesk_cache_hit_latency_seconds` | histogram | — | hit latency (NFR-LAT-04) |
| `answerdesk_cache_entries` | gauge | — | entries held |
| `answerdesk_model_tokens_total` | counter | `kind` (`input`, `output`) | billable tokens |
| `answerdesk_model_cost_usd_total` | counter | `model` | spend in USD (pinned pricing table) |
| `answerdesk_sessions_created_total` | counter | — | sessions created |
| `answerdesk_feedback_total` | counter | `rating` | feedback submissions |
| `answerdesk_rate_limited_total` | counter | — | requests rejected by the limiter |
| `answerdesk_traces_written_total` | counter | `outcome` (`ok`, `error`) | JSONL lines written |
| `answerdesk_idempotent_replays_total` | counter | — | requests served from the idempotency store |

Every family is declared even before its first sample (`# HELP`/`# TYPE`), so a dashboard
does not appear broken on a fresh instance. Rendering is deterministic (sorted families and
label values), which is what the scrape test asserts.

Useful queries:

```promql
# request rate and error ratio
sum(rate(answerdesk_http_requests_total[5m]))
sum(rate(answerdesk_http_errors_total[5m])) / sum(rate(answerdesk_http_requests_total[5m]))

# p95 end-to-end latency
histogram_quantile(0.95, sum by (le) (rate(answerdesk_http_request_duration_seconds_bucket[5m])))

# escalation rate by reason
sum by (reason) (rate(answerdesk_escalations_total[30m]))

# cache health: hit rate and wrong-hit budget
sum(rate(answerdesk_cache_hits_total[30m])) /
  (sum(rate(answerdesk_cache_hits_total[30m])) + sum(rate(answerdesk_cache_misses_total[30m])))
rate(answerdesk_cache_wrong_hits_total[1h]) / rate(answerdesk_cache_hits_total[1h])

# spend per hour
sum(rate(answerdesk_model_cost_usd_total[1h])) * 3600
```

Suggested alerts: error ratio > 5 % for 10 min; p95 latency > 4 s (NFR-LAT-01) for 10 min;
`answerdesk_escalations_total{reason="STALE_CALIBRATION"}` > 0 (re-calibrate now);
wrong-hit ratio > 1 % over 24 h (the cache budget ε = 0.01).

### 5.2 Traces

One JSONL line per request in `logs/traces/traces.jsonl`; `kind=turn` carries the rich
answer trace (retrieval scores, fusion, per-stage latency, tokens, USD, decision, reason,
cache status, escalation), `kind=http` the routing/timing audit line. The full field list is
[`api.md`](api.md) §6. Lookup: `GET /v1/traces/{trace_id}`.

The trace is the *primary* debugging artifact: a wrong answer, a slow turn and a cache
decision are all explained by one line.

### 5.3 Logs

Structured JSON on stderr (`ANSWERDESK_LOGGING__JSON=false` for text):

```json
{"ts":"2026-10-21T01:02:03.004+00:00","level":"INFO","logger":"answerdesk.service",
 "message":"answerdesk service ready","profile":"offline","fake":true,
 "corpus_version":"corpus-demo-…","index_version":"index-…","calibration":"cal-…","degraded":[]}
```

Provider chatter is pinned to `WARNING` (httpx logs full URLs at `INFO`). Secrets are
redacted by key name; PII by value.

### 5.4 PII boundaries (INV-05)

Redaction happens at **three** boundaries, all sharing the t8 detector
(`answerdesk.policy.pii`) so they agree about what PII is:

1. **store** — `append_turn`, feedback comments and ticket notes are redacted before they
   are written (`[CARD]`, `[EMAIL]`, `[PHONE]`, `[ACCOUNT]`);
2. **trace/log** — every JSONL line and log payload is passed through
   `redact_payload`/`redact_value`; numeric observability fields are *not* redacted (a
   `input_tokens` value is a count, not a credential — a name-based secret rule keeps the
   cost trace intact);
3. **provider request** — the service wraps the configured provider in
   `RedactingProvider`, so no PII leaves the process even in a repair prompt
   (`provider_redactions` in the trace counts what was removed).

A PII query is also stopped *before* retrieval/generation: the guard pre-flight returns
`PII_DETECTED` and costs no model call (`answerdesk.service.signals.preflight_decision`).

### 5.5 What is *not* instrumented (honest scope)

* no OpenTelemetry spans, no distributed tracing (the trace id is the correlation handle);
* no per-token streaming latency (the answer path is one structured model call, §5.1 of
  `api.md`);
* metrics are in-process and reset on restart (scrape them into a real TSDB).

---

## 6. Runbook

### 6.1 `/healthz` says `degraded`

```bash
curl -s localhost:8765/healthz | jq '.checks | to_entries[] | select(.value.status != "ok")'
```

| Component | Typical cause | Fix |
| --- | --- | --- |
| `index` / `corpus` | no index for the current corpus, embedder mismatch | `answerdesk ingest --demo` (or your corpus) then `answerdesk index`; `POST /admin/refresh` reloads without a restart |
| `calibration` | artifact missing, stale, or fitted on another corpus | `answerdesk calibrate --synthetic` (machinery) or calibrate on real data; point `ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT` at it |
| `cache` | disabled by configuration or thresholds unreadable | check `ANSWERDESK_CACHE__ENABLED`; regenerate thresholds (`answerdesk.cache.calibrate`) |
| `database` | path unwritable, disk full | check `ANSWERDESK_SERVICE_DB` and free space; the service keeps serving reads but turns cannot be persisted |
| `provider` | key missing (offline profile with a real preset, or vice versa) | `answerdesk config check --profile api` prints the actionable error |

### 6.2 Re-ingest after a knowledge-base change

```bash
answerdesk ingest ./kb/                 # or POST /admin/ingest {"source":"./kb"}
answerdesk index                        # or POST /admin/refresh
answerdesk calibrate                    # if the corpus changed materially
```

`/admin/ingest` and `/admin/refresh` reload the retriever in place and invalidate the answer
cache for the new corpus version; the old index keeps serving until the new one loads.

### 6.3 Re-calibrate

1. `answerdesk calibrate --out reports/policy` (fits on the dev split only; INV-09);
2. check the gate report (`gate-checks.json`) — a failed risk/coverage gate is a stop sign,
   not a warning;
3. restart with `ANSWERDESK_SERVICE_CALIBRATION_ARTIFACT=<new artifact>`, or copy it over
   the default path;
4. confirm `/healthz.checks.calibration` shows the new `threshold_ref` and a small age.

### 6.4 Cache misbehaving

* kill switch: `ANSWERDESK_CACHE__ENABLED=false` (restart) — `/healthz` shows `disabled`,
  `done.cache.status` becomes `bypassed` and `/metrics` shows `cache_bypass_total`;
* wrong hits: `answerdesk_cache_wrong_hits_total` increments on a 👎 on a cache hit; above
  the ε = 0.01 budget, raise the per-cohort thresholds or disable the semantic path
  (`thresholds=None` → exact-only);
* stale after a re-index: `POST /admin/refresh` invalidates by corpus version.

### 6.5 Ticket queue

```bash
curl -s 'localhost:8765/v1/escalations?status=open&limit=50' | jq '.tickets[] | {ticket_id,priority,reason}'
```

Auto-created tickets have `status: "auto"`; operator-created ones `"open"`. The trace id on
each ticket resolves to the full evidence trail (`GET /v1/traces/{trace_id}`).

### 6.6 Disk and retention

* traces rotate at 32 MiB with three backups (≈128 MiB worst case); lower
  `ANSWERDESK_SERVICE_TRACE_MAX_BYTES` on a small disk;
* `idempotency` rows expire after the replay window but are only purged lazily per key;
  a monthly `DELETE FROM idempotency WHERE created_at < <iso>` is safe;
* `answerdesk.db` grows with turns; export and prune old sessions offline (stop the
  service or use `.backup`).

---

## 7. Performance budget and measurement

| Budget | Target | How to check |
| --- | --- | --- |
| end-to-end p95 (API model, n ≥ 200) | ≤ 4 000 ms; p50 ≤ 1 500 ms (NFR-LAT-01) | release evidence run; the trace's `timings_ms.total` and `answerdesk_http_request_duration_seconds` |
| time-to-first-token p95 | ≤ 1 200 ms (NFR-LAT-01) | the `retrieval` event arrives first; measure its arrival time client-side |
| offline pipeline p95, deterministic doubles | ≤ 1 500 ms (NFR-LAT-03) | the service tests assert the wiring, not a number; CI uses the double path |
| cache hit p95 | ≤ 150 ms semantic, ≤ 50 ms exact (NFR-LAT-04) | `answerdesk_cache_hit_latency_seconds` |
| stage attribution | present in ≥ 95 % of traces (NFR-COST-03) | every `kind=turn` line has `timings_ms` with `retrieval`, `generation`, `verify` |

Offline measurements on the reference laptop (t9, `--fake`, demo corpus): a full turn
(guards → cache → retrieval → generation → verify → policy → persistence → trace) completes
in single-digit milliseconds of service overhead; the dominant cost is the model call in a
real profile.

Measure a real deployment with a load generator of your choice and read the histogram:

```bash
curl -s localhost:8765/metrics | grep answerdesk_http_request_duration_seconds
```

---

## 8. Security notes

* **Local-first by default.** `127.0.0.1` bind, no outbound connection except the configured
  provider. Exposing the service is an explicit `ANSWERDESK_SERVICE_HOST` change behind your
  own proxy/TLS.
* **Admin is off until configured.** No default token, no default password, constant-time
  comparison, token never logged or serialised.
* **No tool calls, no account actions.** Retrieved content can never trigger an action
  (INV-01); the service has no write path to any customer system.
* **CORS** is a console allow-list (`ANSWERDESK_SERVICE_CORS_ORIGINS`), not `*`.
* **Injection** is treated as data: the generation layer spotlights untrusted context and
  the guards escalate `INJECTION_SUSPECTED`; the service adds no interpretation of its own.
* **Rate limiting is in-process.** Behind a reverse proxy, set `X-Client-Id` from the real
  client identity (or accept a single shared bucket).

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: fastapi` when running `serve` | `api` extra not installed | `uv run --extra api …` or `uv sync --extra api --extra dev` |
| `uv` cannot download wheels (`dns error`, `tunnel error`) while `curl` works | a Windows SOCKS proxy is set in Internet Settings and `uv` routes through it | `NO_PROXY=* uv sync --extra api --extra dev` (one-off), or fix the system proxy |
| `answerdesk serve --healthcheck` exits 1 with "connection refused" | previous process still bound to the port, or startup failed | check the printed `error`; `netstat -ano | findstr 8765`; read the lifespan log line |
| every turn escalates with `LOW_CONFIDENCE` | `p̂` below the cost band (0.95 for informational) — often a paraphrase in the offline demo | expected for the strict demo scorer; a real profile with a calibrated scorer answers more |
| every turn escalates with `STALE_CALIBRATION` | artifact older than the ceiling or fitted on another corpus | re-calibrate, or align the corpus |
| `503 index_missing` on a fresh checkout | no index built | `answerdesk ingest --demo && answerdesk index --demo` |
| duplicate answers after a retry | the client did not send `Idempotency-Key` | send it; a replay costs no model call |
| `409 idempotency_conflict` | same key, different body | use a new key per distinct request |
| 429 on the console but not on curl | shared bucket per socket peer | send `X-Client-Id` per browser tab/user |
| Cyrillic/CJK garbled in the console | response consumed as latin-1 | read as UTF-8; SSE frames are `ensure_ascii=False` but valid JSON |

---

## 10. Reproducing this guide

| Section | Command |
| --- | --- |
| offline boot + health (CMD-M) | `uv run --extra api answerdesk serve --fake --port 8765 --healthcheck` |
| service test suite | `uv run pytest -q tests/integration/service` |
| failure matrix | `uv run pytest -q tests/integration/service/test_failure_modes.py` |
| metrics families | `uv run pytest -q tests/integration/service/test_health_metrics.py` |
| PII boundaries | `uv run pytest -q tests/integration/service/test_tracing.py` |
| OpenAPI document | `answerdesk serve --export-openapi reports/openapi.json` |
| full offline gate | `uv run pytest -q` |

## 11. Cache-miss similarity semantics

CacheInfo.similarity is display-only and must stay in 0..1 (frozen contract). On a
semantic-cache **miss** the raw cosine can be negative (the hashing embedder is not a
metric embedding); service/turn.py therefore clamps it to 0..1 and may emit None
when the cache reports no similarity. Regression test:
	ests/integration/service/test_multiturn_cache.py (two distinct turns, second one
after the cache is non-empty). Do not feed a raw miss similarity into CacheInfo.
