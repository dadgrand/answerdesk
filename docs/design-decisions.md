# Architecture decision records

Decisions that shape the boundaries in `docs/architecture.md`. Each ADR states the context,
the decision, the consequences (including what gets harder) and the alternatives rejected.
Format: Status / Date / Context / Decision / Consequences / Alternatives.

Decisions inherited from the research stage (structure-first chunking, RRF default, tuned
convex combination, conformal escalation, cache guards, verified citations) are recorded in
`docs/requirements.md` §1.4 (D-01…D-20); their implementation ADRs will be added by the
owning tasks (t5/t6/t8) where the maths needs its own record.

---

## ADR-001 — Local-first on Windows with no Docker and no service dependencies

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** The primary reader runs Windows 11 with Python 3.12. Every comparable OSS project
requires a container stack (RAGFlow: Elasticsearch/Infinity + MySQL + MinIO + Redis) or a
cloud account. That gate is exactly where a curious reader stops.

**Decision.** The reference stack runs from `uv sync` + `uv run` on a laptop: no Docker, no
database server, no message queue, no external vector store required. SQLite and files are the
default persistence; HTTP is used only for model calls. CI runs the same commands on Linux and
Windows, so the Windows path is exercised, not assumed.

**Consequences.** *Good:* one-command quickstart, trivially reproducible benchmark numbers,
no operations story to maintain. *Harder:* no horizontal scaling story (explicitly a
non-goal, NG-02/NG-10); SQLite concurrency limits the service to a single process; heavy
optional dependencies must be extras. *Accepted risk:* performance numbers on a laptop are
not production numbers — documented in `docs/results.md` limitations.

**Alternatives rejected.** Docker Compose (adds a dependency we would have to support);
managed vector DB (breaks the offline story); Postgres (operationally heavier than SQLite for
a single-machine reference).

---

## ADR-002 — SQLite + FTS5 for storage and lexical search; exact `numpy` dense search; Qdrant optional

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Requirements target a ≥3,000-document / ≥20,000-chunk knowledge base (C-07).
At that size, an exact cosine scan over a float32 matrix is milliseconds on CPU, and BM25 can
be served by SQLite FTS5 without an extra process.

**Decision.**
1. Documents, chunks, sessions, turns, cache entries and feedback live in one SQLite file
   (`data/answerdesk.sqlite3`, WAL mode).
2. Lexical retrieval is SQLite FTS5 (BM25 ranking); `bm25s` is allowed as an in-memory
   alternative if FTS5 tokenisation proves insufficient for the six locales (t5 measures).
3. Dense retrieval is exact cosine over `data/index/vectors.npy` plus an id array. No ANN
   index in v1.0.
4. A Qdrant adapter is allowed behind the `qdrant` extra and must satisfy the same
   `Retriever` protocol and produce identical ordering on the exact-case test.

**Consequences.** *Good:* deterministic retrieval (no ANN recall mystery in the eval), zero
services, trivially inspectable artifacts, honest latency numbers. *Harder:* memory grows
with the corpus (≈30 MB for 20k chunks at 1024 dims float32); no incremental ANN updates.
*Mitigation:* the interface is `Retriever`, so a Qdrant adapter can be swapped in without
touching callers; NFR-LAT-02 measures both points.

**Alternatives rejected.** FAISS/HNSW by default (approximate recall would contaminate the
retrieval metrics we are trying to defend); Chroma (extra dependency, weak multilingual
filters); Postgres+pgvector (server process).

---

## ADR-003 — One universal OpenAI-compatible provider layer

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** The product must run against an API by default and against local servers
(vLLM, llama.cpp server, Ollama) without code changes (C-02, NFR-PORT-01). Every one of those
speaks the OpenAI chat-completions shape.

**Decision.** Exactly one HTTP client implementation (`answerdesk.llm.providers`, t6) built on
`httpx`, parameterised by `ProviderConfig` (base_url, model, key, timeout, retries, JSON mode).
Presets are *data*: `answerdesk.config.PROVIDER_PRESETS` maps names to defaults. No module may
construct its own model HTTP client (INV-07); no code branches on preset name.

**Consequences.** *Good:* one code path to test for retries, streaming, JSON mode and error
mapping; local mode needs no key; swapping providers is a config edit. *Harder:* provider
specific features outside the common subset (e.g. vendor-only tool formats) are unavailable;
a genuinely different API would require an adapter that still satisfies `ChatProvider`.

**Alternatives rejected.** Per-provider SDKs (five dependency trees, five retry semantics);
LiteLLM-style router (heavy dependency, hides the wire format); a local-only design (fails
C-02).

---

## ADR-004 — Deterministic doubles are first-class: `HashingEmbedder`, `FakeProvider`, `FakeJudge`

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** The repository's promise is a reproducible eval. CI cannot call a model (no key,
no network, no cost, no flakiness), yet the pipeline must be exercised end to end on every
commit.

**Decision.** The skeleton ships doubles that satisfy the production protocols:
`HashingEmbedder` (BLAKE2b feature hashing → unit vector, identical across processes),
`FakeProvider` (reply derived only from the last user message, canned overrides, streaming),
`FakeJudge` (lexical-overlap rubric score), `FakeVerifier` (marker rule), `InMemoryCache`,
`InMemoryTraceSink`. CI runs the whole pipeline on them. Quality numbers produced this way are
plumbing evidence, never release evidence (requirements §0.3: G-CI vs G-REL).

**Consequences.** *Good:* hermetic, fast, deterministic tests; module tasks can start before
any model access exists; the doubles double as fixtures for docs examples. *Harder:* doubles
must be maintained alongside the protocols (a protocol change breaks them — which is the
point); a passing CI says nothing about model quality and the docs must keep saying so.

**Alternatives rejected.** Recorded HTTP cassettes only (brittle, large, leak prompts);
live API in CI (costs money, needs a secret, flakes); no end-to-end CI (then the pipeline
rots).

---

## ADR-005 — Two gate classes and `AT-` traceability markers

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Requirements demand numeric quality targets *and* deterministic offline CI. Those
cannot be the same gate: an offline suite cannot certify retrieval quality.

**Decision.** Every requirement is proven by a **G-CI** gate (offline, deterministic: structure,
maths, contracts, guards, harness correctness), a **G-REL** gate (frozen test split, real
models, committed artifacts) or a **G-REV** checklist item. Every G-REL requirement carries a
G-CI companion. Each acceptance test gets an id (`AT-RET-04`) and the id appears as a marker
(`# AT-RET-04`) in the code; `scripts/check-traceability.py` (t12) fails when a marker is
missing.

**Consequences.** *Good:* no silently unverified requirement; the traceability matrix is
machine-checked; reviewers can grep evidence. *Harder:* every module task must carry markers
and keep the matrix updated (one line per test).

**Alternatives rejected.** Single "all tests" gate (would force either faked quality numbers
or untested plumbing); prose-only traceability (rots immediately).

---

## ADR-006 — Frozen pydantic contracts with `extra="forbid"` and an explicit schema version

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Five parallel streams edit five packages. The failure mode is silent interface
drift: one module adds a field, another stops reading it, the console shows nothing.

**Decision.** All cross-module types live in `answerdesk/contracts.py`, subclass `Contract`
(`extra="forbid"`, `validate_assignment=True`), and carry invariants that encode the
requirements (citations resolve, no uncited claim displayed, reason code matches action).
`SCHEMA_VERSION` is printed by `answerdesk version` and recorded in reports. Changing a shape
means editing contracts, bumping the version, and noting it in the change log.

**Consequences.** *Good:* drift fails loudly and immediately; the console/service/eval share
one truth. *Harder:* adding a field requires a small cross-team edit; `validate_assignment`
costs a little performance on mutation (acceptable: contracts are boundary objects).

**Alternatives rejected.** Dataclasses/TypedDicts (no validation, no JSON schema);
untyped dicts (the reason most RAG demos cannot be debugged); protobuf (overkill, no Python
story for this scale).

---

## ADR-007 — typer CLI with honest stubs and fixed exit codes

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Downstream tasks' verify commands already name the CLI verbs (`ingest --demo`,
`eval --retrieval-only`, `serve --fake`, …). If the surface only appears at the end, every
task invents its own flags.

**Decision.** The skeleton declares the complete documented command surface with typer, one
entry point (`answerdesk = answerdesk.cli:main`). Unimplemented commands exit **2** with a
message naming the owning task; configuration errors exit 2; unexpected failures are caught at
the top level with the error code. `--help` works for every command (AT-OPS-01).

**Consequences.** *Good:* parallel tasks implement against a stable surface; a user never gets
a silent no-op. *Harder:* flag changes ripple through docs and tests (accepted: the surface is
part of the contract).

**Alternatives rejected.** argparse (more boilerplate, worse help); Click directly (typer is
the thin layer over Click with type-driven help); one script per module (no shared surface).

---

## ADR-008 — pydantic-settings with YAML, environment overrides and three profiles

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Operators must switch provider/embedder/reranker without code edits, and CI must
run a deterministic profile with no secrets.

**Decision.** One `Settings` tree (pydantic-settings, prefix `ANSWERDESK_`, nested delimiter
`__`). Precedence: CLI overrides > environment > `.env` > `configs/*.yaml` > code defaults.
Unknown keys are rejected at every level. Profiles: `offline` (fake provider, hashing embedder,
rerank off, no network), `api` (API provider), `local` (local server, real embedder, rerank on).
Profile/preset coherence is enforced (`offline` + a remote preset is an error).

**Consequences.** *Good:* one place to change behaviour; the same knobs are testable; a typo
fails instead of silently defaulting. *Harder:* adding a key requires touching `config.py` and
`configs/default.yaml` together; env vars are global (tests must isolate them — the `hermetic
env` fixture does).

**Alternatives rejected.** Hydra (heavy, dynamic, hard to type-check); plain `os.environ`
(scattered defaults, no validation); TOML-only config (no env override for keys).

---

## ADR-009 — Light base install; heavy dependencies only behind extras

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** A reader on a CPU-only laptop must be able to run the tests and the CLI. Torch and
sentence-transformers are ~2 GB of wheels; fastapi is only needed for the service.

**Decision.** Base dependencies: pydantic, pydantic-settings, typer, pyyaml, numpy, httpx.
Extras: `dev` (pytest, ruff, mypy), `local` (torch, sentence-transformers), `qdrant`,
`api` (fastapi, uvicorn, sse-starlette). Nothing in the base install may import an extra's
package (tested in `tests/smoke/test_package.py`). Dependency licences must be permissive
(requirements §13.3); no Dify/FastGPT code is vendored.

**Consequences.** *Good:* fast install, wide compatibility, small attack surface. *Harder:*
runtime imports of optional features need lazy loading and a clear error when the extra is
missing (`pip install answerdesk[local]`).

**Alternatives rejected.** One fat install (slow, CUDA pain on Windows); a plugin system with
entry points (premature for v1.0).

---

## ADR-010 — stdlib structured logging with redaction; JSONL traces as the audit surface

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Support traffic is PII-heavy (requirements §11.1). Logs are the most likely leak
path, and traces are what the operator console reads.

**Decision.** `answerdesk.logging` provides a JSON formatter and a `redact()` function that
masks secret-looking keys recursively; no third-party logging dependency. Tracing uses the
`TraceRecord` contract, written as JSONL with PII already redacted before the write (INV-05).
The console reads that same shape.

**Consequences.** *Good:* one record type for logs/traces/console; redaction is testable;
no dependency. *Harder:* no log-shipping integration out of the box (explicitly out of scope,
NG-02).

**Alternatives rejected.** structlog/loguru (extra dependency for a formatter); OpenTelemetry
(right idea, too much machinery for a reference stack, and it encourages shipping traces
somewhere we do not have).

---

## ADR-011 — Retrieval defaults: RRF (k=60) ships, convex combination is the tuned path

**Status:** accepted (t3) · **Date:** 2026-10-20 · **Owner of the ablation:** t5/t11

**Context.** Research (`research/answerdesk/findings.md` §2.1) found RRF is the robust
cold-start default, while tuned convex combination beats it in and out of domain
(arXiv:2210.11934 [S]). Min–max normalisation is query-relative and can let a flat-score
retriever dominate.

**Decision.** `retrieval.fuser` defaults to `rrf` with `rrf_k=60`; `cc_alpha` has **no**
default and must come from a fitted artifact on the dev split. The ablation (RT-7) decides
which one ships, and the requirement is written so "RRF wins" is a valid outcome.

**Consequences.** *Good:* a working default with no labels, and a measurable upgrade path.
*Harder:* fusion parameters must be persisted with provenance (FR-RET-05); the ablation must
run every release.

**Alternatives rejected.** Shipping a tuned α without an artifact (unfalsifiable); learned
fusion in v1.0 (deferred to v1.1, Q7).

---

## ADR-012 — Interface seams are protocols, not base classes

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** Module tasks must be able to implement a component without importing a shared
abstract base that might change under them, and tests must be able to substitute doubles
without inheritance gymnastics.

**Decision.** Use `typing.Protocol` (runtime-checkable where cheap) plus pydantic value models.
Concrete implementations never inherit from a shared base; they are checked structurally in
tests (`isinstance(HashingEmbedder(), Embedder)`).

**Consequences.** *Good:* zero import coupling, trivial fakes, multiple implementations coexist.
*Harder:* protocol drift is only caught where it is used or asserted; the contract tests must
keep covering each seam.

**Alternatives rejected.** ABC hierarchies (import coupling, diamond problems); duck typing
without protocols (no static checking, no runtime assertion in tests).

---

## ADR-013 — Open questions this skeleton deliberately leaves open

**Status:** accepted (t3) · **Date:** 2026-10-20

**Context.** A skeleton that guesses at internals becomes a merge conflict generator.

**Decision.** The skeleton does **not** decide: chunk size and overlap (t4 measures), BM25 vs
FTS5 internals (t5), prompt text (t6), the calibration scorer (t8), the judge model (t7), the
console's state management (t10). It does fix the *seams* those decisions must respect: the
contracts, the config keys, the CLI verbs, the trace shape.
