# answerdesk v0.1.0

**A multilingual customer-support RAG reference stack that shows its work: verified citations, a
calibrated answer/escalate decision, and evidence with confidence intervals — including the
results that failed.**

This is the first public release. It is a reference implementation for people who want to see how
grounded retrieval-augmented generation is measured, not a product and not a hosted service.

- Local-first: `uv` only — no Docker, no database server, no GPU, no API key for the offline path.
- Six locales first-class: ru, uz, tg, ky, tr, zh.
- One OpenAI-compatible provider layer: hosted APIs by default, local servers (vLLM / llama.cpp /
  Ollama) without code changes.
- Every published number is generated from committed artifacts; three scripts fail the build when
  the docs and the evidence disagree.

## What is in it

Ingestion (structure-first chunking, tables and FAQ pairs never split), hybrid retrieval
(BM25 + exact dense, RRF and tuned convex fusion, optional rerank/MMR), grounded generation
(span-level claims and citations, verifier, numeric guard), a split-conformal escalation policy
with cost-sensitive routing and guards, a guarded semantic cache with a wrong-hit budget, a
FastAPI + SSE service with traces and metrics, a React operator console, and an evaluation harness
with frozen splits, adversarial sets, judge calibration protocol, external-validity arm and a
cross-language consistency audit.

Start here: [`README.md`](../README.md) · [`docs/architecture.md`](architecture.md) ·
[`docs/results.md`](results.md) · [`docs/requirements.md`](requirements.md).

## The honest part (read this before quoting anything)

**No risk-control guarantee at 5%.** Both attempts fail their own pre-stated gate and ship the
failure. v2 (`n = 133` calibration rows, base error 69%) has an **absent** certificate at α = 0.05;
the largest feasible level is α = 0.25 with 8.3% coverage, reported as a fallback. v3 (persisted
answers, `n = 485`) reaches AUROC 0.8286 but coverage at risk ≤ 5% of only 1.26%, and the
certificate is still **absent** (`feasible = false`). The repository contains no "calibrated" or
"risk-controlled" claim. → [`docs/risk-control.md`](risk-control.md)

**NFR-LAT-01 is missed.** Target p50 ≤ 1,500 ms / p95 ≤ 4,000 ms; measured **2,618.22 / 14,116.76
ms** (primary arm, n = 200, generation stage, judge excluded). The target is not amended — the miss
is reported next to the numbers it belongs to. → [`docs/results.md`](results.md)

**Consistency is a real limitation.** Cross-language fact agreement is 0.14–0.27 depending on
locale; a correct answer in one language does not imply the same answer in another.
→ [`docs/consistency.md`](consistency.md)

**Judged metrics are diagnostic.** No human κ exists in this checkout, so groundedness, rubric and
citation numbers are labelled diagnostic and are not release gates.
→ [`docs/eval-methodology.md`](eval-methodology.md)

**What the evidence does support.** Grounded answers with citations that resolve: judged
groundedness 0.908 [0.858, 0.950] (n = 141), rubric 4.69, must-escalate recall 1.000 [0.862, 1.000]
with 0 false negatives on 24 unsafe turns, and 84% auto-answer coverage on the synthetic corpus.
External, judge-free evidence: Belebele comprehension improves with retrieval in ru (+0.267), uz
(+0.233), tg (+0.258), ky (+0.283) and shows **no claim** in zh; Mr. TyDi ru nDCG@10 improves
+0.1435 [0.1290, 0.1581] — the only external retrieval-quality claim.
→ [`docs/results.md`](results.md) · [`docs/external-validity.md`](external-validity.md)

**Also documented, not hidden:** the secondary generator arm ran through a schema-fallback path and
is not comparable; the cache produced 0 hits in the near-miss replay (guards are conservative); the
experimental query-rewriter did not help and ships disabled; the corpus is synthetic, so internal
numbers are optimistic; per-package eval coverage is 68% against the 70% wording while the
aggregate gate passes (R-39); the vulnerability audit is a networked release step (R-08). The
independent audit's verdict is **PASS** with those findings recorded:
[`docs/review-report.md`](review-report.md).

## Frozen inputs of this release

| input | value |
|---|---|
| corpus | `corpus-77d7d081813738a6` (sha256 `ae30066e…`) |
| `answerdesk-bench` v0.1 | `self_sha256 65de8498b494b653b82be70bc854c7cb24f543fd911cfeda61705b522d356671` |
| golden set v1 | manifest `2fd9eb4c…` · split `643a28c8…` |
| judge gate | pending / diagnostic (no human κ) |
| risk-v2 certificate | **ABSENT** (`feasible = false`, α = 0.05, n = 133) |
| risk-v3 certificate | **ABSENT** (`feasible = false`, α = 0.05, n = 485) |

## Quickstart

```bash
uv sync --extra dev
uv run answerdesk ingest --demo          # 810 docs / 7,229 chunks, six locales
uv run answerdesk index --demo
uv run answerdesk ask --fake "Сколько стоит обслуживание карты?"
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke
pwsh -File scripts/demo.ps1              # the whole offline loop in one command
```

Service + console:

```bash
$env:NO_PROXY='*'; uv run --extra api answerdesk serve --fake --port 8765
cd web && npm ci && npm run dev          # http://localhost:5173
```

## Upgrade / compatibility notes

First release, so nothing to upgrade from. Two things worth knowing early:

- The base install is light on purpose; `torch`, `sentence-transformers`, `fastapi` and
  `qdrant-client` live in extras.
- Windows with a system `socks4://` proxy: `httpx` cannot drive it, so loopback traffic needs
  `NO_PROXY` (the scripts already set it).

## Licence

MIT for code; the synthetic corpus and golden sets are CC0-1.0; benchmark data keeps its upstream
licences (CC BY-SA 4.0 / Apache-2.0 / CC0-1.0) and is kept separate from the MIT code. No employer,
bank or client code or data is included.

**Full changelog:** [`CHANGELOG.md`](../CHANGELOG.md).
