# answerdesk HTTP API (v1)

The operator console, the CLI and any integration talk to one FastAPI service. This
document is the endpoint reference: request and response shapes, the Server-Sent Events
(SSE) protocol, the error taxonomy, the trace document and the commands that reproduce all
three. The operational side — environment variables, failure modes, runbook — is
[`ops.md`](ops.md).

Status: **v1.0 reference implementation.** The service is the `answerdesk.service` package
(requirements FR-SVC-01 … FR-SVC-08, NFR-OBS-01, NFR-OPS-01). It is local-first: the
default bind address is `127.0.0.1:8765`, and nothing here phones home.

---

## 1. Run it

```bash
# offline demo: deterministic doubles, no API key, no corpus build, no network
uv run --extra api answerdesk serve --fake --port 8765

# prove the wiring and exit 0 (this is CMD-M / AT-SVC-07)
uv run --extra api answerdesk serve --fake --port 8765 --healthcheck

# API model (needs OPENAI_API_KEY, or any configured preset)
ANSWERDESK_PROFILE=api ANSWERDESK_LLM__PRESET=openai answerdesk serve --port 8765

# local server (vLLM / llama.cpp / Ollama all speak the same wire format)
ANSWERDESK_PROFILE=local ANSWERDESK_LLM__PRESET=ollama answerdesk serve --port 8765
```

Interactive docs are served at `/docs` (Swagger UI) and `/redoc`; machine-readable schema
at `/openapi.json`. `ANSWERDESK_SERVICE_DOCS=false` turns all three off for a hardened
deployment.

### 1.1 Export the OpenAPI document (no server)

```bash
answerdesk serve --export-openapi reports/openapi.json
```

`export_openapi()` is also importable:

```python
from answerdesk.service import export_openapi

export_openapi("reports/openapi.json")   # writes the schema, starts nothing
```

The export is exercised by `tests/unit/test_cli.py::test_serve_exports_the_openapi_schema_without_a_server`
and `tests/integration/service/test_openapi.py`.

### 1.2 Offline profile, in one paragraph

`--fake` (or `ANSWERDESK_PROFILE=offline`) serves an **in-memory** demo corpus, the
deterministic `FakeProvider`, the *real* decision engine loaded from the sealed calibration
artifact in `reports/policy-smoke/`, the *real* SQLite store and the *real* JSONL tracer.
No index build, no key, no network. The demo corpus is deliberately tiny and the overlap
scorer is deliberately strict, so in the offline profile most paraphrases **escalate** —
that is the cost model working, not a bug (see §5.1). A near-verbatim question such as
`обслуживание карты` produces a cited answer; anything outside the corpus escalates with a
reason code.

---

## 2. Conventions

| Concern | Rule |
| --- | --- |
| Base URL | `http://127.0.0.1:8765` by default; every path below is relative to it |
| Content type | requests and non-streaming responses are `application/json; charset=utf-8` |
| Streaming | `POST /v1/chat` defaults to `text/event-stream`; send `stream: false` or `Accept: application/json` for a single JSON object |
| Trace id | every response carries `X-Trace-Id`; every SSE event and every trace line carries the same `trace_id` |
| Client-supplied trace id | `X-Trace-Id: <id>` on the request is honoured (≤128 chars), so a gateway can correlate |
| Idempotency | `Idempotency-Key: <key>` on `POST /v1/chat`; replay returns the stored response and costs no model call |
| Rate limit | per client, token bucket, `X-Client-Id` header or socket peer; `429` + `Retry-After` |
| Errors | one envelope: `{"error": {code, message, trace_id, retryable[, detail][, retry_after_s]}}` |
| Versions | the frozen contracts schema is `SCHEMA_VERSION = 1.0.0`; `/v1` is the HTTP surface version |
| Unknown fields | request models forbid them: a typo is a `422 validation_error`, never silently ignored |

### 2.1 Limits

| Limit | Default | Configuration |
| --- | --- | --- |
| Request body | 256 KiB | `ANSWERDESK_SERVICE_MAX_BODY_BYTES` |
| Message length | 8 000 chars | fixed in the schema |
| Requests per client | 60/min, burst 60 | `ANSWERDESK_SERVICE_RATE_LIMIT`, `..._BURST` |
| Idempotency replay window | 24 h | `ANSWERDESK_SERVICE_IDEMPOTENCY_TTL_S` |
| Session history replayed | last 6 turns | fixed; the last 3 are used for query resolution |

`/healthz` and `/metrics` are exempt from rate limiting: a monitor must never be throttled.

---

## 3. Endpoints

### 3.1 `POST /v1/chat` — answer one turn

Streams SSE (default) or returns one JSON object.

**Request**

```json
{
  "message": "обслуживание карты",
  "session_id": "s-0123456789abcdef",
  "locale": "ru",
  "intent_bucket": "informational",
  "intent_label": null,
  "stream": true
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `message` | string, 1–8000 | required; any supported locale (ru, tg, uz, ky, tr, zh, en) |
| `session_id` | string? | omit to start a new session; a new id is created on first use |
| `locale` | string? | cache-cohort hint; detected from the text when omitted |
| `intent_bucket` | `informational` \| `transactional` \| `account_legal` | selects the conformal α (§8 of the requirements) |
| `intent_label` | string? | optional fine-grained label for the high-risk guard |
| `stream` | bool (default `true`) | SSE when true; JSON when false (or `Accept: application/json`) |

**Response, `stream: false`** — `200` `application/json`:

```json
{
  "trace_id": "tr-3f9c0b7a2d4e4f1a9b8c7d6e",
  "session_id": "s-0123456789abcdef",
  "turn_id": "t-7c1e2d3f4a5b6c7d",
  "answer": {
    "answer_id": "ans-9a8b7c6d5e4f3a2b",
    "trace_id": "tr-3f9c0b7a2d4e4f1a9b8c7d6e",
    "language": "ru",
    "text": "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. [1]",
    "claims": [
      {"claim_id": "c1", "text": "Обслуживание карты стоит 120 рублей в месяц.",
       "citation_ids": [1], "support": "supported"}
    ],
    "citations": [
      {"citation_id": 1, "chunk_id": "chunk-card-fee", "doc_id": "doc-card-fee",
       "doc_title": "Тарифы по карте «Вектор»",
       "heading_path": ["Тарифы", "Обслуживание"],
       "quote": "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
       "start_char": 0, "end_char": 58, "doc_hash": "…64 hex…"}
    ],
    "decision": {"action": "answer", "reason_code": "ANSWERED", "score": 1.0,
                 "alpha": 0.05, "threshold_ref": "cal-f70b9bf8edb0"},
    "confidence": {"p_correct": 1.0, "calibrator_version": "cal-f70b9bf8edb0"},
    "cache": {"status": "miss", "entry_id": null, "similarity": null},
    "usage": {"input_tokens": 812, "output_tokens": 96, "usd": 0.000042,
              "model": "fake-model", "prompt_version": "grounded-answer-1.0.0"},
    "provenance": {"corpus_version": "corpus-demo-6cb29bf77b57",
                   "index_version": "index-87424537ecf8b33c",
                   "prompt_version": "grounded-answer-1.0.0",
                   "verifier_version": "lexical-1.0.0",
                   "scorer_version": "overlap-1.0.0"}
  },
  "usage": {"input_tokens": 812, "output_tokens": 96, "usd": 0.000042,
            "model": "fake-model", "prompt_version": "grounded-answer-1.0.0"},
  "timings_ms": {"history": 0.05, "cache": 0.06, "retrieval": 0.41,
                 "generation": 0.95, "verify": 0.04, "policy": 0.12, "total": 1.9},
  "history_turns": 0,
  "follow_up": false,
  "ticket_id": null,
  "idempotent_replay": false,
  "cache": {"status": "miss", "entry_id": null, "similarity": null, "reason": ""}
}
```

`answer` is the frozen [`Answer` contract](../src/answerdesk/contracts.py): the citation
ids in `claims` always resolve to `citations`, and an answered turn never carries an
uncited claim (INV-03). An escalated turn has `text: ""`, no claims and no citations.

**Response, streaming** — `200` `text/event-stream`, see §5.

**Errors**: `422` (schema), `429` (rate limit), `502`/`504` (provider), `503`
(index/calibration unavailable), `500` (unexpected). Streaming failures after the first
byte arrive as an `error` event followed by `done` (§5.4).

**Example**

```bash
curl -sN http://127.0.0.1:8765/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"message":"обслуживание карты"}'
```

### 3.2 `POST /v1/sessions` — create a conversation

`201` with an empty session. Optional `?locale=ru`.

```bash
curl -s -X POST 'http://127.0.0.1:8765/v1/sessions?locale=ru'
```

### 3.3 `GET /v1/sessions/{session_id}` — read a conversation

Returns the session and its turns (oldest first), each with the `trace_id` that produced
it, the stored customer message (PII-redacted) and the answer text. `?limit=` (1–500,
default 50) returns the last *N* turns, still oldest-first.

```json
{
  "session_id": "s-follow-up",
  "created_at": "2026-10-21T01:02:03.004+00:00",
  "updated_at": "2026-10-21T01:02:09.114+00:00",
  "locale": "ru",
  "turn_count": 2,
  "turns": [
    {"turn_id": "t-…", "seq": 1, "message": "обслуживание карты",
     "answer_text": "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. [1]",
     "trace_id": "tr-…", "action": "answer", "reason_code": "ANSWERED",
     "detail": "p_hat 1.0000 >= answer threshold 0.9500",
     "created_at": "2026-10-21T01:02:03.099+00:00"}
  ]
}
```

`404 not_found` for an unknown id.

**History replay.** The last turns are replayed into the pipeline: a short message (≤4
words) or one that opens with an anaphoric/conjunction cue (`и`, `а`, `ещё`, `what about`,
`and`, `那`, `peki`, …) is resolved against the previous customer message before retrieval
and generation. `done.history_turns` and `done.follow_up` report what happened, and the
trace records `history.turns_used` and `history.cues`.

### 3.4 `POST /v1/feedback` — attach a verdict to a trace

```json
{"trace_id": "tr-…", "rating": "up", "comment": "Точно ответили", "session_id": "s-…"}
```

`201` with `{"feedback_id": "fb-…", "trace_id": "tr-…", "rating": "up", "created_at": "…"}`.
`404` when the trace is unknown (a feedback row without a trace is not auditable).
A `down` vote on a **cache hit** increments `answerdesk_cache_wrong_hits_total` and records
the wrong hit against the cache entry — that is the live signal for the cache error budget
(requirements §9.5).

### 3.5 Escalation tickets

**`POST /v1/escalations`** — open a ticket against a trace:

```json
{"trace_id": "tr-…", "reason": "Клиент просит человека", "priority": "high",
 "note": "Проверить вручную", "session_id": "s-…"}
```

`201` with the ticket (`ticket_id`, `status`, `priority`, `reason`, `note`, timestamps).
`404` for an unknown trace.

An **auto-escalated turn** (action `escalate`/`refuse`) also opens a ticket, with
`status: "auto"`; the `done` event carries its `ticket_id`, and the trace line records
`escalation.reason`.

**`GET /v1/escalations/{ticket_id}`** — one ticket (`404` if unknown).

**`GET /v1/escalations?status=open&limit=50`** — newest first:
`{"tickets": [ … ]}`.

### 3.6 `GET /v1/traces/{trace_id}` — read the redacted trace line

Returns `{"trace_id": "tr-…", "trace": { … }}`. The turn line is preferred; the HTTP audit
line is returned only if no turn line exists. `404` when the id is unknown.

### 3.7 Admin: `POST /admin/ingest`, `POST /admin/refresh`

Bearer-token protected. **Disabled unless `ANSWERDESK_SERVICE_ADMIN_TOKEN` is set**:
with no token configured every call is `403 admin_disabled`; a missing header is `401
unauthorized`; a wrong token is `403 forbidden`. The comparison is constant-time.

```bash
curl -s -X POST http://127.0.0.1:8765/admin/ingest \
  -H "Authorization: Bearer $ANSWERDESK_SERVICE_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"demo": true, "force": false}'
```

```json
{
  "corpus_version": "corpus-55d57ee3c986ff8d",
  "documents": 60,
  "chunks": 3110,
  "index_version": "index-87424537ecf8b33c",
  "reused": false,
  "elapsed_ms": 2143.5,
  "manifest_path": "…/data/corpus/manifest.json",
  "cache_invalidated": 12
}
```

`ingest` accepts `{"source": "path/to/dir"}` (an explicit source) or `{"demo": true}`
(build/refresh the synthetic corpus). Both rebuild or reuse the index and reload the
retriever in place; the previous index keeps serving until the new one loads.

`refresh` is the index-only variant: `POST /admin/refresh?force=true`. It invalidates the
answer cache for the new corpus version and returns the same manifest stats.

### 3.8 `GET /healthz` — dependency status

`200` with a per-component status; `?strict=true` returns `503` when anything is degraded
(for a load balancer that understands only status codes). No secret is ever included.

```json
{
  "status": "ok",
  "service": "answerdesk",
  "version": "0.1.0",
  "schema_version": "1.0.0",
  "uptime_s": 12.34,
  "checks": {
    "service":     {"status": "ok", "version": "0.1.0", "detail": "offline"},
    "database":    {"status": "ok", "detail": "3 sessions, 7 turns"},
    "index":       {"status": "ok", "version": "index-8742…", "detail": "8 chunks (hashing-64)"},
    "corpus":      {"status": "ok", "version": "corpus-demo-6cb2…", "detail": "demo corpus (in-memory)"},
    "provider":    {"status": "ok", "version": "fake-model",
                    "detail": "preset=fake base_url=memory://fake api_key=absent"},
    "calibration": {"status": "ok", "version": "cal-f70b9bf8edb0",
                    "detail": "…/reports/policy-smoke/calibration-artifact.json; age 0.1d; artifact corpus 'synthetic-corpus-1'"},
    "cache":       {"status": "ok", "version": "…/reports/cache-smoke/cache-thresholds.json",
                    "detail": "0 entries, 0 hits, 2 misses, 0 guard rejections"},
    "admin":       {"status": "ok", "detail": "bearer token configured"},
    "tracing":     {"status": "ok", "detail": "…/logs/traces"}
  }
}
```

### 3.9 `GET /metrics` — Prometheus exposition

`200` `text/plain; version=0.0.4; charset=utf-8`. See `ops.md` §5.1 for the metric list.

### 3.10 `GET /v1/info` — non-secret identity

`{"service", "version", "profile", "fake", "corpus_version", "index_version", "chunks",
"model", "preset", "cache_enabled", "admin_enabled"}`. Useful for the console's status bar;
contains no secret.

---

## 4. Error taxonomy

One closed table (`answerdesk.service.errors.ERROR_TAXONOMY`). A code outside it cannot be
emitted: `ServiceError` refuses it at construction, so the table cannot silently drift.

| `code` | HTTP | retryable | Meaning |
| --- | --- | --- | --- |
| `bad_request` | 400 | no | syntactically valid but unusable request |
| `validation_error` | 422 | no | a field failed schema validation (`detail` names it) |
| `unauthorized` | 401 | no | a bearer token is required (`WWW-Authenticate: Bearer`) |
| `forbidden` | 403 | no | the bearer token is not accepted |
| `admin_disabled` | 403 | no | admin endpoints disabled (no token configured) |
| `not_found` | 404 | no | the referenced resource does not exist |
| `method_not_allowed` | 405 | no | route exists, not for this method |
| `conflict` | 409 | no | request conflicts with current state |
| `idempotency_conflict` | 409 | no | the key was used with a different body |
| `payload_too_large` | 413 | no | body exceeds `ANSWERDESK_SERVICE_MAX_BODY_BYTES` |
| `unsupported_media_type` | 415 | no | body is not JSON |
| `rate_limited` | 429 | yes | per-client budget exhausted (`Retry-After`) |
| `provider_error` | 502 | yes | provider failed after retries/fallback |
| `provider_unreachable` | 502 | yes | provider could not be reached |
| `provider_bad_response` | 502 | yes | provider returned an unusable payload |
| `provider_timeout` | 504 | yes | provider exceeded the configured timeout |
| `provider_circuit_open` | 503 | yes | circuit breaker open: fail fast |
| `provider_chain_exhausted` | 502 | yes | every provider in the fallback chain failed |
| `corpus_missing` | 503 | no | corpus or manifest missing (operator action) |
| `index_missing` | 503 | no | no index for this corpus (operator action) |
| `calibration_missing` | 503 | no | no sealed calibration artifact (operator action) |
| `stale_calibration` | 503 | no | calibration too old / wrong corpus (operator action) |
| `service_unavailable` | 503 | yes | service starting or shutting down |
| `not_implemented` | 501 | no | surface exists but is not implemented |
| `contract_violation` | 500 | no | a module produced a contract violation |
| `internal_error` | 500 | no | unexpected failure; details are in the logs |
| `stream_error` | 500 | no | the SSE stream failed after it started |

`retryable` is advice for the client: retry `429`/`502`/`503`/`504` with backoff (honour
`Retry-After`), never retry a `4xx` from the taxonomy except `429`.

```json
{"error": {"code": "rate_limited", "message": "rate limit exceeded (60 requests/minute); retry in 0.8s", "trace_id": "tr-…", "retryable": true, "retry_after_s": 0.8}}
```

---

## 5. The SSE protocol

### 5.1 Event order

```
retrieval ─▶ tokens* ─▶ citations ─▶ decision ─▶ done
```

Every event carries `trace_id`; the order is fixed (FR-SVC-01) and asserted by
`tests/integration/service/test_chat_stream.py`. An escalated turn emits `retrieval`,
`citations` (empty), `decision`, `done` — no `tokens`, because a refusal carries no prose
(CR-8).

**Why tokens arrive after generation.** The answer is a structured object whose claims must
be verified against packed evidence before any text is displayed (CR-1, INV-03). Streaming
raw provider tokens would display unverified text. The service therefore emits the
*verified* answer text progressively — a client renders it exactly like a live token
stream, and the concatenation of all `tokens` deltas equals `done.answer.text` (asserted in
the tests). A future provider-level streaming path can reuse this framing unchanged.

### 5.2 Event payloads

Each frame is `event: <name>\ndata: <one line of JSON>\n\n`; `data` is never empty. All
payloads are validated by `answerdesk.service.schemas` before they are written.

**`retrieval`**

```json
{"event":"retrieval","trace_id":"tr-…","query":"обслуживание карты",
 "normalized_query":"обслуживание карты","language":"ru",
 "passages":[{"chunk_id":"chunk-card-fee","doc_id":"doc-card-fee",
              "doc_title":"Тарифы по карте «Вектор»","heading_path":["Тарифы","Обслуживание"],
              "breadcrumb":"Тарифы > Обслуживание","channel":"bm25","stage":"pack",
              "score":0.0325,"rank":1,"snippet":"Обслуживание дебетовой карты …"}],
 "channels":{"bm25":8,"dense":8},
 "fusion":{"fuser":"rrf","params":{"k":60}},
 "candidates":16,"dropped_duplicates":0,
 "cache":{"status":"miss","entry_id":null,"similarity":null,"reason":""},
 "timings_ms":{"cache":0.06,"retrieval":0.41,"generation":0.95,"total":1.9}}
```

`query` is the *resolved* query (a follow-up already contains the previous turn). `snippet`
is capped at 400 characters; the full text stays on the server.

**`tokens`**

```json
{"event":"tokens","trace_id":"tr-…","delta":"Обслуживание дебетовой ","index":0}
```

`index` is the zero-based delta counter; deltas are ordered and lossless.

**`citations`** — emitted only for a verified answer:

```json
{"event":"citations","trace_id":"tr-…",
 "citations":[{"citation_id":1,"chunk_id":"chunk-card-fee","doc_id":"doc-card-fee",
               "doc_title":"Тарифы по карте «Вектор»","heading_path":["Тарифы","Обслуживание"],
               "quote":"Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
               "start_char":0,"end_char":58,"doc_hash":"…"}],
 "claims":[{"claim_id":"c1","text":"Обслуживание карты стоит 120 рублей в месяц.",
            "citation_ids":[1],"support":"supported"}]}
```

Every `citation.chunk_id` is present in the `retrieval` event's `passages` (INV-02).

**`decision`**

```json
{"event":"decision","trace_id":"tr-…","action":"answer","reason_code":"ANSWERED",
 "score":1.0,"alpha":0.05,"threshold_ref":"cal-f70b9bf8edb0","escalation":null}
```

For an escalation the object is not null:

```json
{"action":"escalate","reason_code":"PII_DETECTED","score":0.0,"alpha":0.05,
 "threshold_ref":"cal-f70b9bf8edb0",
 "escalation":{"ticket_id":"tk-…","priority":"normal","detail":"","reason":"PII_DETECTED"}}
```

`score` is the calibrated `p̂` the decision consumed; `alpha` is the bucket's final α;
`threshold_ref` resolves to the sealed calibration artifact (`cal-<hash12>`), which is what
makes a decision reproducible from disk.

**`done`** — terminal, always last:

```json
{"event":"done","trace_id":"tr-…","session_id":"s-…","turn_id":"t-…",
 "answer":{…the Answer object…},
 "usage":{"input_tokens":812,"output_tokens":96,"usd":0.000042,"model":"fake-model",
          "prompt_version":"grounded-answer-1.0.0"},
 "timings_ms":{"history":0.05,"cache":0.06,"retrieval":0.41,"generation":0.95,
               "verify":0.04,"policy":0.12,"total":1.9},
 "history_turns":0,"follow_up":false,"ticket_id":null,"idempotent_replay":false}
```

A client that only needs the answer may ignore every event except `done`.

### 5.3 Client sketch

```python
import httpx

with httpx.Client(timeout=None) as client:
    with client.stream("POST", "http://127.0.0.1:8765/v1/chat",
                       json={"message": "обслуживание карты"}) as response:
        trace_id = response.headers["x-trace-id"]
        event = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "tokens":
                print(__import__("json").loads(line[6:])["delta"], end="", flush=True)
```

### 5.4 Failures inside a stream

If the pipeline fails *after* the response started (a provider failure is the common case),
the stream emits one `error` frame and then a terminal `done` frame — never a truncated
stream:

```
event: error
data: {"event":"error","trace_id":"tr-…","error":{"code":"provider_timeout","message":"…","trace_id":"tr-…","retryable":true}}

event: done
data: {"event":"done","trace_id":"tr-…","session_id":"s-…","turn_id":"","error":{…},"idempotent_replay":false}
```

Dependency failures (no index, no calibration, rate limit, validation) happen *before* the
first byte and are ordinary HTTP errors with a status code. Both paths write a trace line,
the failing one with `"error": "<ExceptionType>: <message>"`.

### 5.5 Reconnect / retry

Use `Idempotency-Key` and re-POST: the replay returns the stored JSON response (not a
replayed stream) without a second model call. The SSE frames intentionally carry no `id:`
field — resuming a stream would mean re-running the generation, which the idempotency store
already covers.

---

## 6. Trace document (JSONL)

Every request writes one line to `<trace_dir>/traces.jsonl` (`logs/traces` by default).
Two kinds share the file:

* `kind = "turn"` — the rich answer trace (one per `POST /v1/chat`);
* `kind = "http"` — a routing/timing audit line for every other non-exempt request
  (`GET /metrics` and `/healthz` are exempt).

**Turn line fields** (all PII-redacted before the line is written):

| Field | Meaning |
| --- | --- |
| `kind`, `schema_version` | `"turn"`, trace document version `1.1.0` |
| `trace_id`, `session_id`, `turn_id` | identifiers the client also sees |
| `query` | the customer message, redacted (`[CARD]`, `[EMAIL]`, `[PHONE]`, `[ACCOUNT]`) |
| `query_pii` | `{"redacted": n, "kinds": [...]}` for the incoming message |
| `pii` | redactions applied to the stored line itself (should be 0: it is already clean) |
| `language`, `intent_bucket` | detected locale, risk tier |
| `decision` | action, reason code, `score`, `alpha`, `threshold_ref` |
| `p_hat`, `confidence` | the calibrated probability the decision consumed |
| `retrieval` | the full `RetrievalTrace`: candidates, fused, reranked, packed, `fusion` |
| `retrieval_scores` | `[{chunk_id, doc_id, channel, score, rank, stage}]` in packed order |
| `fusion` | `{"fuser": "rrf"|"cc"|"none", "params": {...}}` |
| `cache` | `{status, entry_id, similarity, reason}` (a guard demotion says why) |
| `history` | `{follow_up, turns_used, cues, resolved_query_chars}` |
| `generation` | `{repaired, repair_attempts, dropped_claims, numeric_violations_pre_repair, verifier_version, language_notice}` |
| `escalation` | `{ticket_id, reason}` for an escalated turn, else `null` |
| `usage` | `{input_tokens, output_tokens, usd, model, prompt_version}` |
| `timings_ms` | `analysis`, `retrieval`, `rerank`, `generation`, `verify`, `policy`, `cache`, `total` |
| `stage_latency_ms` | the raw per-stage map the service measured (superset of `timings_ms`) |
| `corpus_version`, `index_version`, `prompt_version` | provenance of the answer |
| `service` | profile, model, preset, calibration provenance, scorer identity |
| `provider_redactions` | PII spans removed at the provider boundary (INV-05) |
| `error` | `null`, or `<ExceptionType>: <message>` for a failed turn |

**HTTP audit line**: `{"kind":"http","trace_id","method","route","status","duration_ms","created_at"}`
— no body, no query, no headers.

Rotation: the file rotates at `ANSWERDESK_SERVICE_TRACE_MAX_BYTES` (default 32 MiB) into
`traces.1.jsonl` … up to three generations. `GET /v1/traces/{id}` scans newest-first.

---

## 7. Versioning and compatibility

* The HTTP surface is `/v1`; a breaking change means `/v2`, not a silent field change.
* The pydantic contracts in `answerdesk/contracts.py` are frozen at schema `1.0.0`; the
  trace document is a versioned *extension* (`schema_version: "1.1.0"`) so the frozen
  models do not need to grow.
* `GET /healthz` reports both versions, and `GET /v1/info` reports the corpus/index versions
  an integration can pin.

## 8. Reproducing this document

| Claim | Command |
| --- | --- |
| endpoint set and schemas | `answerdesk serve --export-openapi reports/openapi.json` |
| SSE order and payload shapes | `uv run pytest -q tests/integration/service/test_chat_stream.py` |
| error table | `uv run pytest -q tests/integration/service/test_errors.py` |
| sessions, history replay, persistence | `uv run pytest -q tests/integration/service/test_sessions.py` |
| idempotency | `uv run pytest -q tests/integration/service/test_idempotency.py` |
| admin guard and manifest stats | `uv run pytest -q tests/integration/service/test_admin.py` |
| feedback and tickets | `uv run pytest -q tests/integration/service/test_feedback_escalations.py` |
| health and metrics | `uv run pytest -q tests/integration/service/test_health_metrics.py` |
| trace document and PII | `uv run pytest -q tests/integration/service/test_tracing.py` |
| failure-mode matrix | `uv run pytest -q tests/integration/service/test_failure_modes.py` |
| full offline service suite | `uv run pytest -q tests/integration/service` |
