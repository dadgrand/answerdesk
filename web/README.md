# answerdesk operator console

The React + Vite + TypeScript console operators use to answer customers with
**verifiable citations**, a calibrated **confidence/escalation** decision, and the retrieval
diagnostics that explain both. It talks to the answerdesk service

* `POST /v1/chat` (SSE: `retrieval` → `tokens`* → `citations` → `decision` → `done`)
* `GET /v1/info`, `GET /healthz`, `GET /metrics`, `GET /v1/traces/{trace_id}`
* `POST /v1/feedback`, `POST /v1/escalations`

and runs fully offline against [MSW](https://mswjs.io) fixtures.

```bash
npm ci
npm run dev              # http://localhost:5173 (proxies /v1, /metrics, /healthz to :8765)
npm run test             # vitest + Testing Library (61 tests, no network)
npm run build            # tsc --noEmit + vite build → dist/
npm run screenshots      # Playwright: docs/assets/*.png (needs the Python service, see below)
```

## Screens

| Screen | What it shows |
| --- | --- |
| **Live queue** (left) | every waiting customer, language badge, channel, priority, SLA timer (2-minute target, at-risk after 75 s, breached after 120 s) and the turn status |
| **Conversation** (centre) | the transcript, the suggested reply with inline `[n]` citation chips, the calibrated p̂ meter against the answer threshold, the decision badge with its reason, cache status, claim-by-claim verification, and one-click **Send reply / Edit / Escalate** |
| **Diagnostics** (right drawer) | the policy decision (action, reason code, p̂, α, `threshold_ref`), retrieval candidates with scores/channels, fusion parameters, cache status and guard reason, per-stage latency, token/USD cost, provenance, raw redacted trace JSON |
| **Analytics** | latency distribution and p50/p95 against the 4 s budget, USD per answer, escalation rate, cache hit rate and per-stage p95 — parsed from the live Prometheus exposition |

Citations are the point of the UI: an `[n]` marker only becomes a button **after** the
service's `citations` event, i.e. after verification (`INV-02`/`INV-03`). Clicking it opens
the source passage (document, heading path, verbatim quote, chunk id, span, doc hash), and
`Escape` closes it. Keyboard focus is preserved and every control has an accessible name.

## Mock mode (no backend)

```bash
npm run dev                                    # then open http://localhost:5173/?mock=1
VITE_USE_MOCK=true npm run build               # bake mock mode into a build
```

`?mock=1` (also inside the hash route: `#/console?mock=1`) starts the MSW worker with the
same handlers the vitest suite uses — a real streaming `ReadableStream`, real SSE framing,
the trace document and the Prometheus text. `?mock=1` is also how the screenshots fall back
if Python/uv is unavailable.

## Deep links

* `#/console?conversation=c-1043` — open a queue item directly.
* `&session=returning` — open a fresh session for the same customer (the exact cache hit
  case; cache entries are keyed on the query, not the session).

## Real service

```bash
# terminal 1 — offline profile, deterministic doubles, no key
cd .. && uv run --extra api answerdesk serve --fake --port 8765

# terminal 2
npm run dev            # the Vite dev server proxies /v1, /metrics and /healthz
```

For a build served from another origin set `VITE_API_BASE=http://127.0.0.1:8765` at build
time; the service's CORS default allows `http://localhost:5173`.

## Screenshots (CL-UI-01/02)

`npm run screenshots` builds the app if needed, starts the **real service** (`uv run --extra
api answerdesk serve --fake`), serves `dist/` through a same-origin proxy, drives Chrome
through six screens and writes PNGs + a manifest to [`docs/assets/`](../docs/assets/):

`console-answer`, `console-citation`, `console-diagnostics`, `console-escalation`,
`console-light`, `analytics`.

State (SQLite, traces) goes to `web/.screenshots/`, which is deleted after a successful run
and git-ignored. Flags: `--mock` (MSW instead of the service), `--base <url>` (reuse a
running service), `--out <dir>` (default `docs/assets`), `--port`, `--service-port`. `PLAYWRIGHT_CHANNEL=msedge` selects another
installed browser; no browser download is required if Chrome or Edge is present.

All screenshot data is synthetic (requirements C-10): fictional customers, reserved-range
test identifiers. The manifest records the mode, viewport, screen descriptions and the
cache/decision counters from the live scrape.

## Architecture

```
src/
  api/        wire types, SSE decoder, Prometheus parser + KPI derivation, fetch client, MSW handlers
  state/      pure reducer (streaming), useConsole hook (network effects, abort, actions)
  components/ queue, conversation, answer card, citations/passages, diagnostics, charts, shell
  pages/      ConsolePage, AnalyticsPage
  demo/       synthetic inbox (7 conversations, 6 locales, one PII guard case)
  lib/        formatting, SLA arithmetic, hash-route parameter helper
```

Two design decisions worth knowing:

1. **The reducer is transport-free.** Every SSE event is a transition (`retrieval` sets the
   trace and passages, `tokens` append, `citations` attaches verified sources, `done`
   finalises), which is what makes progressive rendering testable without timing.
2. **Analytics parses Prometheus, not a bespoke endpoint.** Each KPI names its metric family
   in the UI (`answerdesk_decisions_total`, `answerdesk_stage_duration_seconds`, …), so a
   number can be traced to the scrape in one step (NFR-OBS-02).

## Accessibility & theming

Semantic landmarks (`nav`, `main`, headings), `aria-current` on the selected queue item,
`aria-expanded`/`aria-label` on citation chips, `role="status"`/`aria-live` for streaming and
notices, visible focus rings, `Escape` closes the passage and the drawer, a skip link, and
`prefers-reduced-motion` disabling animations. The theme is a `data-theme` attribute with
dark/light tokens, persisted in `localStorage` and set before first paint.

## Note on the service bug found while building this

While wiring the console, the multi-turn flow exposed a service defect: a cache **miss**
similarity (a raw cosine, possibly negative) was passed into `CacheInfo(similarity=...)`,
whose contract is `0..1`, so the second distinct question in a process failed with
`internal_error / ValidationError`. It was reported and fixed in the service
(`service/turn.py` clamps the miss similarity; regression test
`tests/integration/service/test_multiturn_cache.py`). The console also treats a terminal SSE
`done` frame without an `answer` as an error state instead of rendering a blank answer
(`src/state/console.ts`, covered by a reducer test).
