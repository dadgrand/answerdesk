# Robustness: fuzzing, hostile input and graceful failure

This document records the adversarial pass over the answerdesk service (NFR-OPS-01,
INV-05). It answers one question with evidence: **what happens when the input is not the
input we designed for?**

The suite lives in `tests/property/` (9 modules, 102 tests) and runs offline, deterministically
and without `hypothesis` (not a dependency; the generators are seeded literal corpora, which
is also what keeps the cases readable in a failure):

```bash
uv sync --extra api --extra dev
uv run pytest -q tests/property tests/integration/service   # 169 passed
uv run pytest -q                                            # full suite
```

Every case asserts an **observable contract**, not an implementation detail: the HTTP status
class, the taxonomy code, the presence of a `trace_id`, the SSE frame order, the absence of a
secret in a log line, the number of provider calls, or the content of the prompt.

---

## 1. What was fuzzed

| # | Input class | Examples | Expected behaviour | Tests |
| --- | --- | --- | --- | --- |
| 1 | **arbitrary unicode** | empty, whitespace, punctuation, control chars (`\x01`, `\x7f`), NUL, RTL (he/ar), emoji + ZWJ, combining marks, bidi overrides, BOM, U+2028/U+2029, mixed scripts, 8 000 chars, 8 001 chars | 200 with a well-formed stream, or a 422 with a taxonomy code; never 500 | `test_unicode_inputs.py` (9) |
| 2 | **lone surrogates / invalid UTF-8** | JSON `\ud800` in `message` and in `session_id`; raw `\xff\xfe` bytes; a truncated multi-byte sequence | `message`: 422 `validation_error`; an optional field: sanitised to U+FFFD and accepted; invalid bytes: 400 `bad_request` | `test_unicode_inputs.py` |
| 3 | **size boundaries** | 1 MB body; exactly 262 144 B; 262 145 B; 8 000-char message; 1 000 000-char query to the runner; 1 MB retrieved chunk | 413 before parsing; 422 at the schema; the pipeline returns a decision in < 30 s (measured 3.5 s) | `test_large_inputs.py` (8) |
| 4 | **malformed bodies** | empty, `{`, `null`, `[]`, `"str"`, numbers, `NaN`, `Infinity`, `1e999`, duplicate keys, trailing garbage, unknown fields, bad enums, 2 000-deep nesting, BOM-prefixed JSON | 4xx with `code`/`message`/`trace_id`/`retryable`; the code is in `ERROR_TAXONOMY` | `test_malformed_http.py` (8) |
| 5 | **methods, routes, content types, headers** | GET/PUT/DELETE on `/v1/chat`, unknown paths, path traversal, 5 000-char `X-Trace-Id`, duplicate headers, `text/plain`/form bodies, query-string garbage | 404/405/415/422 in the taxonomy; unsafe trace ids are regenerated | `test_malformed_http.py` |
| 6 | **provider failures** | timeout, 429, 500, 503, connection reset, protocol error, circuit open, all-providers-failed, raw `httpx` errors | 502/503/504 with a taxonomy code, `retryable=true`, a trace id; mid-stream: `error` + `done` with the same code | `test_provider_failures.py` (16) |
| 7 | **malformed provider payloads** | prose instead of JSON, empty completion, truncated JSON, `{}`, control characters, truncated SSE frame, JSON where SSE was expected, unknown SSE events, comments | never an answer: the generator escalates (`PROVIDER_ERROR`/`NO_EVIDENCE`) or the provider layer yields an empty completion; no hang, no 500 | `test_provider_failures.py`, `test_stream_and_cancellation.py` |
| 8 | **cancellation and timeouts** | client sends `http.disconnect` mid-stream; a streamed response is closed after the first chunk; a provider times out | no exception escapes the app; the idempotency lock is released; the follow-up request works; a failure is fast (< 5 s in the test) | `test_stream_and_cancellation.py` (7) |
| 9 | **duplicate / concurrent identifiers** | 4 concurrent requests with one `Idempotency-Key`; same key + different body; same `X-Trace-Id` twice; duplicate feedback; duplicate tickets; duplicate chunk ids in retrieval | one turn, one provider call, N identical bodies, 3 replays; 409 on a conflicting body; unique citation ids | `test_duplicate_ids.py` (7) |
| 10 | **injection** | 9 directive payloads in the query; a canary in retrieved text; a marker sentence in evidence | matched queries/evidence escalate `INJECTION_SUSPECTED` with **zero** provider calls; anything that does reach the prompt is inside `<passage>` in the user message and never in the system message | `test_injection_and_pii.py` (16) |
| 11 | **PII and secrets** | card/email/phone in the query; PII inside retrieved evidence; a secret-shaped value in the query; a cache warm with an answer, then a PII variant | guard fires before the cache and the provider; `[CARD]`/`[EMAIL]`/`[SECRET]` in traces, logs and every provider message; no raw value anywhere | `test_injection_and_pii.py` |
| 12 | **malformed config / CLI** | invalid YAML, list at the top level, wrong types, unknown keys, bad profile, missing file, directory, 200 KB value, garbage env var | exit 2 with `config error: …`, never a traceback; an empty config is valid | `test_config_and_cli.py` (13) |
| 13 | **log and trace sinks** | all of the above, while the root logger is captured and the JSONL files are read back | every log message is UTF-8 and free of PII/secrets; every trace line is valid UTF-8 JSON with a `trace_id` | `test_logs_and_traces.py` (5) |

Counts in parentheses are tests per module; `uv run pytest -q tests/property` reports 102.

---

## 2. Defects found and fixed

**The lesson first:** every finding below was found *after* the service already had a green
integration suite (65 tests), a green smoke run and a published API contract. Happy-path
tests exercise the shapes we designed for; they cannot find the shapes a hostile client
sends. None of these is a redesign — endpoints, SSE order, taxonomy codes and the trace
schema are unchanged — but each one was a real 500, a real silent mutation or a real leak
until it was fixed. Each has a regression test in `tests/property/`.

### F-01 — a lone surrogate could kill a turn and its trace

* **Trigger:** `POST /v1/chat` with `{"message": "bad \ud800 text"}` (valid JSON, invalid UTF-8).
* **Observed failure:** `UnicodeEncodeError` escaped the app — no response at all; SQLite and
  the JSONL writer cannot encode an unpaired surrogate.
* **Fix:** the request boundary normalises every string (`SanitisedStr` in
  `service/schemas.py`: surrogates → U+FFFD, NUL → U+FFFD); telemetry repeats the
  normalisation defensively (`sanitise_text` in `telemetry/redaction.py`, applied by
  `redact_text`, `redact_value` and `tracing.write`). A surrogate in `message` is a 422; in
  an optional field it is repaired.
* **Regression test:** `test_lone_surrogate_in_the_message_is_a_traced_validation_error`,
  `test_surrogate_in_an_optional_field_is_sanitised`.

### F-02 — a punctuation-only query failed the turn

* **Trigger:** `"?!?!..."`, `"   "`, `"..."` (empty after normalisation).
* **Observed failure:** a stream `error` event with `internal_error`; the non-streaming call
  was a 500.
* **Fix:** `TurnRunner` wraps the lookup (`_cache_lookup`); a cache-side exception degrades
  to `BYPASSED` with a reason in the trace. The cache is an optimisation, never a dependency.
* **Regression test:** `test_torture_corpus_never_breaks_the_stream` (the `punctuation` and
  `space` cases).

### F-03 — concurrent retries with one idempotency key raised a 500

* **Trigger:** two or more simultaneous requests with the same `Idempotency-Key`.
* **Observed failure:** `sqlite3.IntegrityError: UNIQUE constraint failed:
  turns.idempotency_key` — a 500 caused by a client doing the right thing.
* **Fix:** a per-key async mutex (`_KeyLock` in `service/routes.py`) is acquired **before**
  the replay check and released when the response (or the stream) finishes; entries are
  reference counted and evicted, so a hostile client cannot grow the map without bound.
* **Regression test:** `test_concurrent_retries_of_one_key_run_the_turn_once` (4 requests →
  1 provider call, 3 replays, identical JSON).

### F-04 — an unbounded `Idempotency-Key`

* **Trigger:** a 5 000-character header value.
* **Observed failure:** accepted and stored, an easy way to grow a table by megabytes.
* **Fix:** `_validated_idempotency_key` trims, caps at 200 characters and rejects control
  characters with `400 bad_request`.
* **Regression test:** `test_huge_idempotency_key_is_rejected`.

### F-05 — raw transport errors escaped the taxonomy

* **Trigger:** a provider raising `httpx.ConnectError`, `httpx.ReadTimeout`,
  `httpx.RemoteProtocolError` or `httpx.HTTPStatusError`.
* **Observed failure:** `internal_error` 500 (`retryable=false`) — the client was told to
  give up on a transient outage.
* **Fix:** `to_service_error` maps `httpx.TimeoutException` → `provider_timeout` (504),
  `httpx.TransportError` → `provider_unreachable` (502), `httpx.HTTPStatusError` →
  `provider_error` (502), plus `RecursionError`/`JSONDecodeError`/`UnicodeDecodeError` →
  `bad_request` (400), `sqlite3.IntegrityError` → `conflict` (409) and `sqlite3.Error` →
  `service_unavailable` (503).
* **Regression test:** `test_provider_failure_is_a_taxonomised_error` (12 provider failure
  shapes, each asserting status, code and `retryable`).

### F-06 — a secret pasted into a question was written to the trace

* **Trigger:** `"Мой ключ sk-live-…"` in a customer message.
* **Observed failure:** the value appeared verbatim in the trace JSONL, in the logs and in
  the prompt sent to the provider.
* **Fix:** redaction now matches credential *values* (`sk-…`, `ghp_…`, `xox…`, `AKIA…`,
  `Bearer …`, PEM headers, `api_key=…`) as well as secret-bearing key names, replacing them
  with `[SECRET]` in traces, logs and provider messages; the trace records
  `redaction.kinds: ["secret"]`.
* **Regression test:** `test_secrets_in_the_query_are_redacted_in_traces_and_prompts`.

### F-07 — U+2028 broke strict SSE clients

* **Trigger:** a query containing U+2028 (LINE SEPARATOR), e.g. parsed from a pasted document.
* **Observed failure:** the `data:` line was split by `str.splitlines()`-style clients and the
  frame's JSON became unparseable (the commit's own decoder hit this first).
* **Fix:** `sse_event` escapes U+2028, U+2029 and U+0085 (`\u2028`, `\u2029`, `\u0085`), so
  every field stays on one physical line.
* **Regression test:** `test_torture_corpus_never_breaks_the_stream` (the `line_separators`
  case; the decoder uses `splitlines()` on purpose).

### F-08 — an unsafe `X-Trace-Id` was echoed into a response header

* **Trigger:** `X-Trace-Id: "bad id with spaces"` or a value containing control characters.
* **Observed failure:** echoed verbatim into the response header and used as the trace id.
* **Fix:** a strict token regex (`^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$`); anything else is
  replaced by a fresh `tr-<24 hex>` id.
* **Regression test:** `test_client_trace_ids_are_echoed_only_when_safe`.

### Cross-scope fix (captain-approved)

`scripts/train_query_rewriter.py` read `four_bit_stderr` unconditionally but assigned it only
inside the CUDA branch, so `tests/unit/rewrite/test_script.py` failed on a machine without
CUDA (`UnboundLocalError`). Fixed with a single initialisation (`four_bit_stderr = ""`
before the branch); the existing test covers it, no new test added.

## 3. Honest limitations

* **L-1 — a denylist is not a proof.** The injection guard recognises directive-shaped text
  (`docs/security.md` / `generate/guards.py`). A paraphrase it does not match still reaches
  the prompt — but only inside the spotlighted `<passage>` block, and the grounded pipeline
  refuses to display any claim that is not verified against the evidence. The suite asserts
  the *outcome* for every payload (no answer, no citations), not the detection.
* **L-2 — RTL scripts outside the six locales fall back to `en`.** Hebrew/Arabic messages are
  answered or escalated normally; the language field is a fallback, not a proof of language
  detection. Extending the banks is a corpus task, not a service task.
* **L-3 — a truncated provider SSE frame ends the stream silently.** The LLM layer treats an
  unterminated `data:` line as "stream over" and yields an empty completion; the generation
  pipeline then escalates `PROVIDER_ERROR` and displays nothing. That is safe but not
  diagnostic: the trace shows an empty completion rather than a `ProviderResponseError`.
  Changing it belongs to `llm/` (not in this task's scope).
* **L-4 — the concurrency guarantee is per process.** The idempotency mutex is an
  `asyncio.Lock` in the serving process. With several uvicorn workers (not the default; the
  documented deployment is one process) a retry could still race into the UNIQUE index, which
  now maps to `409 conflict` instead of a 500 — correct, but the client sees a retry hint
  rather than a replay.
* **L-5 — the fuzz corpus is curated, not coverage-guided.** There is no AFL/libFuzzer loop
  and no `hypothesis` dependency; the cases are literal and deterministic, chosen from the
  failure modes of the actual stack (JSON decoding, SQLite encoding, SSE framing, httpx).
  A coverage-guided pass is a reasonable follow-up, not a claim made here.
* **L-6 — resource exhaustion beyond the body limit is delegated.** The 256 KiB body limit,
  the per-client token bucket and uvicorn's own header/time limits are the defence; this pass
  verifies the limits fire, not that a 10 Gbit flood is survivable (that is a deployment
  concern: `docs/ops.md` §6).

---

## 4. What the operator should take away

* A bad request is a **4xx with a code**; a dependency failure is a **502/503/504 with
  `retryable`**; an internal fault is the only 500, and it is logged with a trace id and no
  customer text.
* The trace is safe to share: PII and credentials are replaced, and the `redaction`/`query_pii`
  blocks say how many values were replaced without storing them.
* A stream that fails after the status line sends `error` + `done`; clients should render the
  error and keep the connection state, not retry blindly.
* A retry with the same `Idempotency-Key` is always safe, including under concurrency.
