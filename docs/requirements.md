# answerdesk v1.0 — Product and Technical Requirements

| Field | Value |
|---|---|
| Document ID | `AD-REQ-v1.0` |
| Version | 1.0.0 (frozen) |
| Status | **Frozen** — change only through §18 change control |
| Date | 2026-10-20 |
| Owner | architect (requirements and contracts) |
| Approver | captain |
| Applies to | `answerdesk` repository, release v0.1.0 implementing spec v1.0 |
| Evidence base | `research/answerdesk/findings.md`, `recommendations.md`, `open-questions.md`, `sources.md` |
| License of this document | MIT (same as the repository) |

---

## 0. How to read this document

### 0.1 Purpose

This document is the **contract** between the requirements stage and every implementation, verification and review task in the `answerdesk` build. It states what v1.0 must do, how well, and — for every single requirement — the artifact or command that proves it. A downstream task may cite a requirement ID (`FR-RET-05`, `NFR-LAT-01`, `INV-03`) as its authority; it may not silently narrow or widen it.

Three properties are non-negotiable, because they are what the acceptance criteria test:

1. **Testable, not aspirational.** Every requirement has a pass condition that a machine or a reviewer can evaluate to `true`/`false`. Where a number depends on a real model, the number is bound to a named artifact and a named command, not to intent.
2. **Traceable.** Every requirement maps to at least one named module, an owning task and exactly one acceptance-test ID (§16).
3. **Consistent.** No two requirements conflict; where requirements genuinely pull against each other, §15 fixes the precedence rule in advance and the resolution is part of the spec.

### 0.2 Requirement keywords

| Keyword | Meaning | May be dropped? |
|---|---|---|
| **MUST** | Release-blocking. Verified by the stated gate. | No — only by amending this document (§18). |
| **SHOULD** | Release-relevant. May be dropped only with an ADR in `docs/design-decisions.md` that names the requirement ID and the residual risk. | Yes, with ADR. |
| **MAY** | Optional. Presence must not break any MUST. | Yes, freely. |

**Identifier families.** `FR-` functional requirement; `NFR-` non-functional requirement; `INV-` invariant that outranks every FR/NFR (§11); `AT-` acceptance test; `CL-` review checklist item (§17.1); `CMD-` verification command (§17.2); `C-` external constraint (§1.3); `D-` frozen design decision (§1.4, Appendix A); `NG-` non-goal (§12); `R-` risk, zero-padded `R-01…R-20` (§14); `RT-` retrieval target (§6.4); `F-` generation/citation target (§7.5); `E-` escalation target (§8.5); `K-` cache target (§9.5); `CR-` citation rule (§7.2); `T-` tension (§15.2); `Q-` research open question (§19).

### 0.3 Gate classes — how a requirement is proven

A requirement is only real if it can fail. Every requirement carries at least one acceptance test in one of three classes:

| Gate | Runs | Determinism | What it can prove | Cannot prove |
|---|---|---|---|---|
| **G-CI** | On every commit, offline, no API key, no network | Fully deterministic | Structure, math, invariants, contracts, plumbing, metric/harness correctness, guard ordering | Model-quality numbers |
| **G-REL** | Release evidence run on the frozen test split with a real embedder and a real generator; artifacts committed under `reports/<run>/` and summarised in `docs/results.md` | Seeded; re-runnable; recorded/replay mode must be labelled | Retrieval quality, groundedness, calibration, escalation risk, latency, cost | Generalisation to real traffic (§10.7) |
| **G-REV** | Human reviewer against an explicit checklist item (§17.1) | Reviewer judgement, bounded by the checklist wording | Documentation accuracy, derivation correctness, honesty of claims | Anything statistical |

**Rule (G-REL → G-CI companion).** Every G-REL requirement also has a G-CI companion test that proves the *machinery* that produces the number is correct (e.g. the recall@k function on hand-computed inputs, the split-leakage detector, the conformal quantile on synthetic scores). This is deliberate: an offline CI can never certify model quality, and pretending otherwise is exactly the dishonesty this repository is meant to avoid. Quality numbers live in G-REL; correctness of measurement lives in G-CI.

**Rule (no unproven claim).** A number may appear in `README.md` or `docs/results.md` only if a committed artifact under `reports/` reproduces it and the command that produced it is documented (NFR-REP-01). Replay/recorded-mode numbers are labelled `REPLAY-MODE` in every table cell they touch.

### 0.4 Reliability tags carried over from research

Where this document relies on a research claim, the tag is preserved: `[C]` confirmed, `[S]` single source, `[A]` assumption. Requirements derived from `[S]`/`[A]` evidence are still MUST — but they are MUST-*and-measured*: the acceptance test can falsify the design decision, and if it does, the ADR records the reversal. Tags appear where the evidence shapes a threshold; they are not repeated on plumbing requirements.

---

## 1. Scope, context and constraints

### 1.1 What v1.0 is

A **local-first, MIT-licensed, multilingual customer-support RAG reference stack**: ingest a knowledge base, retrieve hybrid (sparse + dense) candidates, generate a grounded answer with **inline, verified citations**, decide **answer / clarify / escalate** under a calibrated risk target, serve a cached or fresh answer over an OpenAI-compatible provider layer, and prove all of it with a one-command, reproducible evaluation harness plus an operator console that shows *why* the system decided what it decided.

Six locales are in scope for v1.0: **ru, tg, uz, ky, tr, zh**. The generator's answer language must match the query language; the corpus is multilingual by construction (§13.2).

### 1.2 What v1.0 is explicitly not

See §12. Short form: no voice, no deployment, no real customer data, no employer or bank code, no autonomous actions, no fine-tuning, no multi-tenant SaaS.

### 1.3 External constraints (immutable inputs)

| ID | Constraint | Consequence for requirements |
|---|---|---|
| C-01 | **Local-first on Windows 11 + Python 3.12; no Docker required anywhere in the quickstart.** Linux/macOS must work from the same commands. | NFR-PORT-02, NFR-DET-01, FR-OPS-01 |
| C-02 | Generation goes through an **OpenAI-compatible chat-completions API by default**, but **every** model access sits behind a provider abstraction that also supports local servers (vLLM, llama.cpp server, Ollama). | NFR-PORT-01, FR-OPS-03 |
| C-03 | **No employer, bank or client code, data, schema, prompt or document** may enter the repository. Public or synthetic data only. | NFR-REP-04, CL-DATA-01, §13.2–13.4 |
| C-04 | **English repository** (code, docs, comments, commit messages, issues). Localised user-facing UI strings and the six locale corpora are the only non-English artifacts. | NFR-DOC-01 |
| C-05 | **MIT license** for all code. No dependency or vendored source that cannot be redistributed under MIT. | NFR-REP-04, §13.3 |
| C-06 | **No deployment.** v1.0 produces a release candidate and a tagged commit; it never provisions a host, a domain, a container registry or a CI secret with production credentials. | NG-02, §12, §18.3 |
| C-07 | **Knowledge base scale**: the operating point is **≥3,000 documents / ≥20,000 chunks**; the checked-in demo corpus is a subset (≥60 docs, ≥1,500 chunks) so CI stays fast. | FR-ING-07, NFR-SCALE-01 |
| C-08 | **Budget class**: the reference stack must be usable with zero paid API cost (deterministic doubles in CI, `local` extra for real embeddings/reranking) and cheap enough in API mode to run the full eval suite repeatedly. | NFR-COST-01, NFR-COST-02, NFR-DET-01 |
| C-09 | **Time budget**: a full offline CI run ≤ 10 minutes on a laptop-class machine; the full release evidence run ≤ 90 minutes with API provider. | NFR-DET-01, FR-EVL-10 |
| C-10 | Public/synthetic data only implies no third-party PII, no scraped private tickets, and identifiers restricted to reserved ranges. | NFR-SEC-02, CL-DATA-01, §13.4 |

### 1.4 Frozen design decisions inherited from research

These are decisions, not open questions; the requirements encode them. Each has a home in `docs/design-decisions.md` (ADR) and is traceable to the research decision table.

| ID | Decision | Evidence | Requirement(s) |
|---|---|---|---|
| D-01 | Structure-first parsing and chunking; **no semantic chunking** by default | RAGFlow parsing-led adoption `[C]`; semantic chunking cost/benefit `[S]` | FR-ING-02, FR-ING-03 |
| D-02 | Chunks carry a breadcrumb header and an optional 50–100-token contextual header | contextual headers −35/−49/−67% failures `[S]` | FR-ING-04 |
| D-03 | Hybrid retrieval: BM25 + dense, **RRF (k=60) as untuned default**, tuned convex combination as the shipped path | RRF defaults `[C]`; CC > RRF `[S]` | FR-RET-04, FR-RET-05 |
| D-04 | Fusion weights and α fitted **only on the dev split**, persisted with provenance | leakage discipline `[A]` | FR-RET-05, FR-EVL-04 |
| D-05 | Cross-encoder rerank over top 50–150 → top 8–12, optional behind the `local` extra | rerank gains `[S]`; cross-encoder > bi-encoder `[C]` | FR-RET-07, FR-GEN-06 |
| D-06 | Context packing: 8–12 chunks, hard token cap, rank-1 first and rank-2 last, near-duplicate dedupe | lost-in-the-middle `[C]`; context rot `[S]` | FR-GEN-05 |
| D-07 | Query handling: unicode NFKC, language ID, deterministic entity/number/date canonicalisation, lexical must-match for identifiers, **no default machine translation** | dense failure on exact IDs `[S]`; rest `[A]` | FR-RET-06, FR-RET-09 |
| D-08 | Citations are **verified, not decorated**: span-level, entailment-checked, repair-or-escalate | display-only is the OSS norm `[C]`; ALCE `[C]` | FR-GEN-07, FR-GEN-08 |
| D-09 | **Split conformal** abstention with the ⌈(n+1)(1−α)⌉ quantile; three-way routing; cost-sensitive α | conformal guarantee `[C]` | FR-ESC-01..03 |
| D-10 | **Semantic cache** with per-entry thresholds, entity + source-doc guards, versioned invalidation, off for high-risk intents | vCache `[C]`; invalidation design `[A]` | FR-CAC-01..06 |
| D-11 | **Retrieved content is data, never control flow.** No tool call, routing decision or policy bypass from document text | indirect injection `[C]`; CaMeL `[C]` | INV-01..05 |
| D-12 | Golden set **≥50 items per locale**, ~18% unanswerable, retrieval labels separate from answer labels, frozen and hashed | ARES/PPI pattern `[C]` | FR-EVL-01, FR-EVL-02 |
| D-13 | Statistics: paired bootstrap (B=1000, 95% CI) + McNemar + Wilson; **no bare point estimates** | standard practice `[C]` | FR-EVL-07 |
| D-14 | Observable decision: console shows chunks, fusion contributions, verifier verdict, calibrated confidence, escalation reason, cache status | 10-project gap analysis `[C]` | FR-UI-04, FR-UI-05 |
| D-15 | One command reproduces everything; golden-set and corpus hashes printed in every artifact | no OSS project does this `[C]` | NFR-REP-01..03 |

### 1.5 System decomposition (module → path → owning task)

Task IDs are the shared-task-list IDs (t3–t14). This table is the vocabulary used by §16.

| Module | Path (repo-relative) | Owns | Task |
|---|---|---|---|
| contracts | `src/answerdesk/contracts.py` | Protocols + pydantic models for every seam | t3 |
| config | `src/answerdesk/config.py`, `configs/**` | settings, provider presets, profiles | t3 |
| ingest | `src/answerdesk/ingest/` | loaders, chunker, dedup, synthetic corpus, dataset adapters | t4 |
| corpus | `data/corpus/` | checked-in demo corpus + manifest | t4 |
| index | `src/answerdesk/index/` | BM25 index, exact dense index, optional Qdrant adapter | t5 |
| retrieval | `src/answerdesk/retrieval/` | fusion, query analysis, pipeline | t5 |
| rerank | `src/answerdesk/rerank/` | cross-encoder, MMR | t5 |
| llm | `src/answerdesk/llm/` | provider layer, presets, cost ledger | t6 |
| generate | `src/answerdesk/generate/` | prompt registry, context packer, citations, verifier | t6 |
| policy | `src/answerdesk/policy/` | conformal calibration, risk/coverage, cost model, PII, injection guards | t8 |
| cache | `src/answerdesk/cache/` | semantic cache, invalidation, error budget | t8 |
| service | `src/answerdesk/service/` | FastAPI app, SSE, sessions, admin ingestion | t9 |
| telemetry | `src/answerdesk/telemetry/` | JSONL tracing, Prometheus metrics | t9 |
| eval | `src/answerdesk/eval/`, `evals/` | datasets, metrics, judge, stats, calibration, report | t7 |
| web | `web/` | React operator console | t10 |
| docs | `docs/` | this document + methodology/derivation docs | t3, t4, t5, t6, t8, t12 |
| scripts/CI | `scripts/`, `.github/workflows/` | reproduction scripts, traceability checker, CI | t3, t12, t14 |

### 1.6 Glossary (normative)

| Term | Definition used in this document |
|---|---|
| **Chunk** | A retrievable unit produced by the chunker; has `chunk_id`, `doc_id`, `heading_path`, char span, token count, text, optional contextual header. |
| **Passage** | A chunk, after retrieval, with a score and a source channel (`bm25`/`dense`/`rerank`). |
| **Gold chunk set** | The set of chunk IDs a golden case marks as sufficient evidence; a case may have 1–3. |
| **Claim** | One atomic, checkable assertion in the generated answer. Answers are structured as an ordered list of claims. |
| **Citation** | A (claim, chunk, char-span, verbatim quote) tuple. Citation IDs are stable integers within an answer. |
| **Decision score** `p̂` | The calibrated estimate of the probability that the answer is correct, given the query and the packed evidence, for one request; in `[0,1]`. |
| **Decision** | One of `answer`, `clarify`, `escalate`, `refuse`, with a closed-enum reason code. |
| **Auto-answer** | A decision of `answer` that is shown to the customer without human review. |
| **Risk** | Empirical error rate among auto-answers, on a held-out set. |
| **Coverage** | Fraction of requests receiving `answer`. |
| **Escalation** | Handing the turn to a human operator with the evidence bundle attached. |
| **Corpus version** | Monotonic hash-identified snapshot of the indexed corpus; any ingest/delete changes it. |
| **Wrong hit** (cache) | A cache hit whose cached answer is incorrect for the new query, measured on the near-miss replay set (§9.5). |
| **REPLAY-MODE** | A measurement produced without a live model/provider, from recorded fixtures; always labelled. |
| **Reference machine** | The hardware profile recorded in `docs/results.md` for every latency/cost number; all timing requirements are relative to it. |

---

## 2. Personas and workflows

### 2.1 Personas

| ID | Persona | Goal | Success looks like | Primary surface |
|---|---|---|---|---|
| **P1** | **Customer** (ru/tg/uz/ky/tr/zh speaker, mobile-first) | Get a correct, specific answer in their own language, or a fast human handoff | Answer in ≤ 4 s p95 with citations they never have to trust blindly; never a confident wrong answer | Chat endpoint (`POST /v1/chat`, SSE) |
| **P2** | **Support operator** (human agent) | Handle the hard 20–40%: review a suggested draft, verify it, send or escalate | Draft arrives with clickable citations, calibrated confidence and the reason it was escalated; one action to send, one to escalate | Operator console (`web/`) |
| **P3** | **Knowledge-base admin** | Keep the KB current: ingest, re-ingest, delete, verify freshness | One command (or one protected endpoint) refreshes the index; provenance and chunk stats visible; stale answers impossible | CLI + `/admin/*` |
| **P4** | **Maintainer / evaluator** (the repo's second audience) | Reproduce every published number and every ablation on their own machine | `git clone` → two commands → same table, same CIs, same hashes | CLI (`answerdesk eval`, `scripts/`) |

Persona P4 is a first-class citizen of this spec. A repository that publishes retrieval, calibration and cost numbers but cannot be re-run by a stranger is not the product we are building.

### 2.2 Workflow W1 — customer chat (P1)

| Step | Behaviour | Requirement |
|---|---|---|
| W1.1 | Customer sends a message in any supported locale; the service assigns `trace_id`, resolves/creates a session, records the turn | FR-SVC-01, FR-SVC-02 |
| W1.2 | Text is normalised (NFKC), language-detected, PII-scrubbed for logging; the raw message is never sent to a provider before redaction | FR-RET-06, INV-06, NFR-SEC-02 |
| W1.3 | Retrieval runs: BM25 + dense → fusion → optional rerank → dedupe → packed context (8–12 chunks) | FR-RET-01..07, FR-GEN-05 |
| W1.4 | Cache is consulted first; a hit must pass cheap structural re-validation before it may answer | FR-CAC-02, FR-CAC-03 |
| W1.5 | Generation produces a structured answer: claims, each with verified citations; language matches the query | FR-GEN-01..08 |
| W1.6 | Policy decides `answer` / `clarify` / `escalate` / `refuse` from the calibrated score, guards and cost model | FR-ESC-01..06 |
| W1.7 | Streamed to the client as ordered SSE events: `retrieval`, `tokens`, `citations`, `decision`, `done` — with citations only streamed once verified | FR-SVC-01, FR-GEN-09 |
| W1.8 | A JSONL trace records stage latencies, fusion contributions, scores, decision, reason code, tokens, USD | NFR-OBS-01 |
| W1.9 | Customer feedback (👍/👎) is captured and attached to the trace for later analysis | FR-SVC-05 |

### 2.3 Workflow W2 — operator console (P2)

| Step | Behaviour | Requirement |
|---|---|---|
| W2.1 | Queue lists escalated and low-confidence conversations with locale badge, wait time and reason code | FR-UI-01, FR-UI-02 |
| W2.2 | Opening a conversation shows the customer thread plus the **suggested draft** with inline citation markers | FR-UI-02, FR-UI-03 |
| W2.3 | Clicking a citation opens the exact source passage (doc title, heading path, quoted span, highlighted offset) | FR-UI-03, FR-GEN-02 |
| W2.4 | Diagnostics drawer shows retrieval candidates with per-channel scores and fusion contribution, cache status, verifier verdict, `p̂`, α and threshold reference | FR-UI-04, FR-UI-05 |
| W2.5 | Operator edits and sends, or escalates with a reason; the action is recorded against the trace | FR-UI-06, FR-SVC-05 |
| W2.6 | Analytics view shows latency percentiles, USD/answer, escalation rate and cache hit rate for the selected window | FR-UI-07, NFR-OBS-02 |

The console is **read-and-act only in v1.0**: no KB editing from the UI (that path is CLI/admin ingestion), no arbitrary prompt editing. This bounds scope and keeps the audit surface small.

### 2.4 Workflow W3 — admin ingestion and refresh (P3)

| Step | Behaviour | Requirement |
|---|---|---|
| W3.1 | `answerdesk ingest <source>` loads PDF/DOCX/HTML/MD/TXT, keeps heading hierarchy and tables | FR-ING-01, FR-ING-02 |
| W3.2 | Chunking respects structure and a token budget; every chunk carries provenance | FR-ING-03..05 |
| W3.3 | Near-duplicates removed deterministically; manifest written with source, license, sha256, chunk stats | FR-ING-06, FR-ING-08 |
| W3.4 | `answerdesk index` builds BM25 + dense indexes; new **corpus version** is minted | FR-OPS-02, FR-CAC-05 |
| W3.5 | Refresh is idempotent: re-ingesting unchanged sources produces an identical manifest (same hashes) | FR-ING-08, NFR-REP-02 |
| W3.6 | Protected `POST /admin/ingest` performs the same operation over HTTP with a bearer token; disabled unless a token is configured | FR-SVC-04 |
| W3.7 | Invalidation: any corpus-version bump invalidates affected cache entries and makes stale citations impossible | FR-CAC-05, FR-CAC-03 |

### 2.5 Workflow W4 — reproduce and audit (P4)

| Step | Behaviour | Requirement |
|---|---|---|
| W4.1 | `uv sync --extra dev` then `uv run pytest -q` passes offline with no key | NFR-DET-01 |
| W4.2 | `uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke` runs end-to-end offline and writes a report with the corpus and golden-set hashes | FR-EVL-09, NFR-REP-02 |
| W4.3 | `uv run answerdesk eval --all --report reports/bench` produces the release evidence tables with CIs | FR-EVL-10 |
| W4.4 | `docs/results.md` reproduces exactly from the documented commands; every number links to an artifact | NFR-REP-01 |
| W4.5 | `scripts/check-traceability.py` proves every requirement ID still has its test marker | NFR-REP-03 |

### 2.6 User stories → requirement map

| ID | Story (as … I want … so that …) | Requirements |
|---|---|---|
| US-01 | As a customer, I ask in Russian with typos and an order ID and get the right policy answer, not a generic article | FR-RET-06, FR-RET-09, FR-GEN-06 |
| US-02 | As a customer, I see which document each part of the answer came from | FR-GEN-02, FR-GEN-03, FR-UI-03 |
| US-03 | As a customer, when the bot is unsure it hands me to a human instead of guessing | FR-ESC-01, FR-ESC-05, FR-GEN-07 |
| US-04 | As an operator, I see the draft, the evidence, the confidence and the escalation reason in one screen | FR-UI-02..05 |
| US-05 | As an operator, sending or escalating is one action and is auditable | FR-UI-06, FR-SVC-05 |
| US-06 | As an admin, refreshing 3,000 documents is one command and I can see what changed | FR-ING-07, FR-ING-08, NFR-SCALE-01 |
| US-07 | As an admin, a deleted document can never be cited again | FR-CAC-05, FR-CAC-03 |
| US-08 | As a maintainer, I can swap the model to a local Ollama server by editing config only | FR-OPS-03, NFR-PORT-01 |
| US-09 | As a maintainer, I can reproduce the published numbers and the ablations on my laptop | FR-EVL-08..10, NFR-REP-01, NFR-DET-02 |
| US-10 | As a reviewer, I can see that the escalation risk target actually held on held-out data | FR-ESC-07, FR-EVL-06, NFR-REP-01 |
| US-11 | As a security reviewer, I can prove injected document text cannot change control flow | INV-01..04, NFR-SEC-01 |
| US-12 | As a maintainer, I can turn the cache off with one flag and measure its error budget | FR-CAC-07, FR-CAC-08 |

---

## 3. Cross-cutting invariants

Invariants outrank every FR/NFR. A change that violates an invariant is rejected regardless of the metric it improves. Detail and threat model: §11.

| ID | Invariant |
|---|---|
| INV-01 | Retrieved document text is **data, never control flow**. No routing decision, tool call, prompt-template switch, cache-policy change or guard bypass may originate from document content. |
| INV-02 | No answer is displayed with a citation that does not resolve to a passage that was actually in the packed context. |
| INV-03 | No displayed claim lacks at least one supporting citation after verification. |
| INV-04 | No number or date appears in a displayed claim unless it appears (after canonicalisation) in a cited chunk. |
| INV-05 | PII is redacted **before** it reaches an index, a log line, a trace file, or an external provider request body. |
| INV-06 | An unanswerable or out-of-scope query never yields an auto-answer; it yields `clarify`, `escalate` or `refuse` with a reason code. |
| INV-07 | Every model call goes through the provider abstraction; no module may construct its own HTTP client for a model endpoint. |
| INV-08 | Deterministic offline CI: no test, script or default path in the CI gate may require network access, an API key, a GPU, or Docker. |
| INV-09 | Calibration and fusion artifacts are only ever fit on the dev split; the test split is read-only for reporting. |
| INV-10 | No number appears in public docs without an artifact and a command that reproduces it. |

---

## 4. Functional requirements

Notation: **Level** = MUST/SHOULD; **AT** = acceptance-test ID (§17); G-REL numbers are measured in the release evidence run, never asserted in offline CI.

### 4.1 Ingestion (module `ingest`, owner t4)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-ING-01 | MUST | Loaders for PDF (pypdf), DOCX (python-docx), HTML (trafilatura or bs4), MD and TXT produce a normalised document record with title, heading tree and block list; each format has a fixture and a regression test | 5 formats load; 0 crashes on the fixture set; headings preserved on ≥ 95% of fixture headings | AT-ING-01 |
| FR-ING-02 | MUST | Chunking is structure-first: split on heading boundaries, FAQ Q/A pairs, table rows and numbered procedure steps; a table or an enumerated procedure is never split across chunks unless it exceeds the token budget, in which case it is split at row/step boundaries with the header repeated | 0 fixtures split a table mid-row; 0 fixtures split a numbered list across chunks when it fits | AT-ING-02 |
| FR-ING-03 | MUST | Every chunk respects the token budget with documented overlap math; default budget ≤ 512 tokens, overlap ≤ 64 tokens; the chunker never emits an empty or whitespace-only chunk | 100% of chunks within budget; 0 empty chunks; invariant test over the demo corpus | AT-ING-03 |
| FR-ING-04 | MUST | Every chunk carries `doc_id`, `chunk_id`, `heading_path`, char span, token count and a breadcrumb string (e.g. `Product > Billing > Refunds`); optional contextual header (50–100 tokens) is a config flag, off by default, and when on is stored separately from the chunk text | 100% of chunks have a non-empty `doc_id` and `chunk_id`; heading path present for ≥ 95% of chunks in structured formats | AT-ING-04 |
| FR-ING-05 | MUST | Numeric/date/identifier strings are preserved verbatim in chunk text and additionally exposed as canonicalised tokens (`5 марта` → `2026-03-05`, `12,5 €` → `12.50 EUR`) in a sidecar field used by BM25 | Canonicalisation fixtures per locale pass; original text byte-identical | AT-ING-05 |
| FR-ING-06 | MUST | Near-duplicate removal is deterministic (MinHash, documented Jaccard threshold, seeded permutations); same input → same survivors | Identical survivor set across two runs; duplicates below threshold removed from the demo corpus | AT-ING-06 |
| FR-ING-07 | MUST | The corpus builder accepts a documented scale parameter and produces a manifest reporting doc/chunk counts; the checked-in demo corpus is ≥ 60 docs / ≥ 1,500 chunks, and the system's operating point of ≥ 3,000 docs / ≥ 20,000 chunks is coachable from the same command | Demo manifest: `docs ≥ 60`, `chunks ≥ 1,500`; scale manifest: `docs ≥ 3,000`, `chunks ≥ 20,000` (G-REL, NFR-SCALE-01) | AT-ING-07 |
| FR-ING-08 | MUST | Ingest is idempotent and provenance-complete: re-ingesting unchanged sources yields a byte-identical manifest; the manifest records per-source path, license, sha256, byte size, and aggregate chunk statistics | Manifest hash stable across re-runs; every source has a non-empty `license` field | AT-ING-08 |
| FR-ING-09 | SHOULD | Public-dataset adapter (Bitext customer-support family) exists behind an explicit license gate: it refuses to run unless the source's license was verified and recorded, and never commits data whose license forbids redistribution | Adapter refuses with a clear error when `license_ok` is absent; no non-redistributable data in `data/` | AT-ING-09 |
| FR-ING-10 | MUST | The synthetic corpus generator is seeded, multilingual (ru/tg/uz/ky/tr/zh), covers support intents (tariffs, cards, transfers, documents, KYC, complaints) and injects controlled perturbations: typos, intra-sentence code-switching (RU + EN product terms), numbers, dates, order IDs and error codes | Same seed → byte-identical output and manifest; ≥ 6 locales present; each perturbation class ≥ 5% of generated queries where applicable | AT-ING-10 |

### 4.2 Retrieval (modules `index`, `retrieval`, `rerank`; owner t5)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-RET-01 | MUST | Sparse retrieval is BM25 over chunks with per-language tokenisation that preserves numerals, dates and identifier tokens; scores are deterministic and exposed | Same query twice → identical ranked list and scores; identifier-token fixtures retrieved | AT-RET-01 |
| FR-RET-02 | MUST | Dense retrieval is exact cosine similarity over multilingual embeddings behind an embedder Protocol; the CI implementation is a deterministic hashing embedder, the local implementation an injected real embedder | Exact nearest-neighbour equivalence against a brute-force reference on a 200-vector fixture (ids and order) | AT-RET-02 |
| FR-RET-03 | MUST | Both retrievers implement one `Retriever` Protocol and return `RetrievedPassage` records with channel, score and rank | Contract test for both implementations; no adapter-specific leakage into callers | AT-RET-03 |
| FR-RET-04 | MUST | Fusion supports **RRF** with tunable `k` (default 60) and **convex combination** with per-retriever normalisation (min–max and z-score); the active fuser is configurable per profile | RRF output on synthetic ranks equals the hand-computed formula; CC output equals the hand-computed formula; degenerate-score fixture (flat cosines) does not silently let one channel dominate | AT-RET-04 |
| FR-RET-05 | MUST | Fusion parameters (RRF `k`, CC `α`, optional learned weights) are fit **only on the dev split** through a documented procedure (coordinate search and/or logistic regression), persisted as an artifact with provenance: fitting split hash, metric optimised, date, and the resulting parameters | Artifact exists and its `split_hash` equals the dev manifest hash; a leakage test fails if any test-split item id appears in the fitting inputs | AT-RET-05 |
| FR-RET-06 | MUST | Query analysis applies: unicode NFKC, language identification, identifier/number/date protection (never stemmed or fuzzy-matched away), and typo-tolerant expansion (char n-gram or fuzzy) that is additive, not destructive — the raw query remains a retrieval arm | Fixtures: `Error code TS-999` and an order-ID query retrieve the exact-identifier chunk; typo fixture (`kartа` with Cyrillic/Latin mixing) still retrieves the source chunk | AT-RET-06 |
| FR-RET-07 | SHOULD | Optional cross-encoder reranker and MMR diversification, lazy-loaded behind the `local` extra, never imported on the base install; rerank budget is configurable (default top-50 in, top-12 out) | Base install imports without torch; reranker fixture returns deterministic order; MMR λ fixture verified against a reference computation | AT-RET-07 |
| FR-RET-08 | MUST | The retrieval pipeline emits a full trace: per-channel candidate lists, fusion contributions, rerank deltas, dedupe removals, final order | Trace object validates against the contract schema and contains every stage for a fixture query | AT-RET-08 |
| FR-RET-09 | MUST | No default machine translation of queries; locale normalisation may correct keyboard layout and obvious typos but must not transliterate identifiers or translate them | Fingerprint fixtures: digit strings, order IDs and error codes are byte-identical pre/post analysis | AT-RET-09 |
| FR-RET-10 | MUST | Retrieval quality meets the release targets in §6.4 on the frozen test split, macro-averaged over the six locales, with per-locale reporting | See §6.4; each row is a binary gate with a CI lower bound where a delta is claimed | AT-RET-10 |

### 4.3 Grounded generation and citations (modules `llm`, `generate`; owner t6)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-GEN-01 | MUST | Answers are structured objects: ordered claims, each with ≥ 1 citation; the rendered text's citation markers are derived from the structure, never parsed back out of prose | Schema validation on 100% of fixture outputs; marker/text consistency check passes | AT-GEN-01 |
| FR-GEN-02 | MUST | Every citation resolves to a `chunk_id` present in the packed context and carries a char span and a verbatim quote (whitespace-normalised) that is a substring of that chunk's stored text | 0 phantom citations; 0 non-substring quotes (deterministic check, no model needed) | AT-GEN-02 |
| FR-GEN-03 | MUST | Citation numbering is stable within an answer, ordered by first appearance, and identical in streamed and final payloads | Fixture: streamed citation payload equals final payload | AT-GEN-03 |
| FR-GEN-04 | MUST | The prompt registry is versioned: every prompt has an ID, semantic version and content hash; every answer records the prompt version used | 100% of answers carry a resolvable prompt version; changing a prompt without bumping the version fails the registry test | AT-GEN-04 |
| FR-GEN-05 | MUST | The context packer enforces a hard token budget, dedupes near-identical passages (cosine > 0.95 or MinHash), and applies lost-in-the-middle ordering: strongest evidence first, second-strongest last, remainder in descending order between | Packer fixture: token count ≤ budget; duplicate passages collapsed; ordering assertion holds for a 10-passage fixture; default packing 8–12 chunks | AT-GEN-05 |
| FR-GEN-06 | MUST | The answer language equals the detected query language for all six locales, with an explicit, visible notice when the model cannot comply | Release: language-agreement rate ≥ 0.98 macro (G-REL); CI: detector + prompt-contract fixtures per locale | AT-GEN-06 |
| FR-GEN-07 | MUST | A verifier judges every claim against its cited passage(s) with verdicts `supported` / `partial` / `unsupported`; `unsupported` claims trigger exactly one repair pass, after which they are removed or the answer is escalated — never displayed | 0 unsupported claims displayed (structural assertion on fixtures and on the release run); repair passes ≤ 1 per answer | AT-GEN-07 |
| FR-GEN-08 | MUST | Groundedness, citation precision and citation recall meet the targets in §7.5 on the frozen test split | See §7.5 | AT-GEN-08 |
| FR-GEN-09 | MUST | Generation is deterministic under test doubles: a recorded fixture provider reproduces a byte-identical `Answer` object (excluding trace IDs and timings) | Two runs over the fixture corpus produce identical answer JSON hashes | AT-GEN-09 |
| FR-GEN-10 | MUST | Insufficient-evidence handling is explicit: if no passage passes the evidence threshold, the generator returns a structured refusal/escalation signal rather than prose | Fixture set of 20 no-evidence queries yields 0 prose answers and 20 structured signals | AT-GEN-10 |
| FR-GEN-11 | SHOULD | The prompt enforces spotlighting/delimiting of untrusted context and an explicit instruction that document content cannot change the task, the output format or the routing | Injection fixture set: 0 answers follow injected instructions; guard flags raise `INJECTION_SUSPECTED` (G-CI + G-REL, INV-01) | AT-GEN-11 |
| FR-GEN-12 | MUST | The numeric/date guard (INV-04, rule CR-9) is deterministic and independent of the model: every number or date appearing in a displayed claim must appear, after canonicalisation, in at least one chunk cited by that claim; a violation drops the claim or escalates with `CITATION_FAILED` | 0 displayed violations on the fixture matrix and on the release run; the pre-repair violation rate is reported as a diagnostic | AT-GEN-12 |

### 4.4 Calibrated escalation (module `policy`; owner t8)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-ESC-01 | MUST | An answerability scorer produces a calibrated `p̂` in `[0,1]` — the probability that the answer is correct, given the query and the packed evidence — backed by the split-conformal machinery in §8.1; alternative scorers (LLM judge, NLI entailment, token-overlap baseline) are interchangeable behind one Protocol | Scorer contract test for ≥ 2 implementations; scores in range; monotone in evidence overlap on a synthetic fixture | AT-ESC-01 |
| FR-ESC-02 | MUST | Calibration uses the split-conformal quantile `q̂ = ⌈(n+1)(1−α)⌉`-th smallest calibration risk score, with the finite-sample correction; the calibration set is the dev split or a disjoint calibration slice, never the test split | Property test over 200 seeded synthetic trials: realised risk on fresh samples ≤ α in ≥ 95% of trials; artifact records `n`, `α`, `q̂`, split hash | AT-ESC-02 |
| FR-ESC-03 | MUST | The decision is cost-sensitive, three-way, and derived from published costs: `answer` iff `p̂ ≥ 1 − C_delay/C_wrong`, `clarify` iff `1 − C_abstain/C_wrong ≤ p̂ < 1 − C_delay/C_wrong`, otherwise `escalate`; `α_final = min(α_bucket, C_abstain/C_wrong)` | Boundary fixtures per branch return the expected action; changing costs changes the threshold as the formula predicts; no hard-coded threshold in code (config only) | AT-ESC-03 |
| FR-ESC-04 | MUST | Guard precedence is fixed and deterministic: `PII`/`INJECTION` guards → hard-policy intents (account/legal/refund) → conformity → calibrated band. Guards can force `escalate`/`refuse`; nothing can force `answer` | Ordered test matrix over (guard, policy, score) triples; each reason code reachable and asserted | AT-ESC-04 |
| FR-ESC-05 | MUST | Reason codes are a closed enum shared with the service and console, documented in Appendix B, and every escalation/refusal carries exactly one | Enum ↔ docs ↔ UI parity test; every code covered by at least one fixture | AT-ESC-05 |
| FR-ESC-06 | MUST | The chosen operating point and its provenance are persisted: calibration artifact (split hash, `n`, `α`, `q̂`, costs, scorer version, model id, date) and referenced by every decision | Artifact schema test; decisions carry `threshold_ref` resolving to the artifact | AT-ESC-06 |
| FR-ESC-07 | MUST | Escalation quality meets the targets in §8.5 on the frozen test split: realised auto-answer risk within α (with Wilson upper bound slack), must-escalate recall, escalation precision and coverage | See §8.5 | AT-ESC-07 |
| FR-ESC-08 | MUST | PII detection covers phones, emails, card and account-like numbers with locale-aware rules; detected PII deterministically forces redaction in logs and either redaction-in-answer or escalation per policy | Fixture matrix per locale: ≥ 95% recall, ≥ 95% precision on synthetic PII set; redaction verified in trace output | AT-ESC-08 |
| FR-ESC-09 | MUST | Prompt-injection heuristics flag instruction-like content in retrieved passages and in user text; a flag forbids auto-answer (escalate with reason) and is recorded in the trace | Injection fixture set: flag raised on ≥ 90% of positives, ≤ 5% on the clean control set | AT-ESC-09 |
| FR-ESC-10 | SHOULD | A coverage monitor exists: realised coverage and realised risk over a rolling window, with a documented alert threshold of −3 coverage points vs the calibration-time value | Monitor unit test with synthetic windows; alert fires exactly at the threshold crossing | AT-ESC-10 |

### 4.5 Semantic cache (module `cache`; owner t8)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-CAC-01 | MUST | Exact-match cache: normalised query + locale + entity set + version tuple returns the stored artifact without a model call | Exact-repeat fixture: second call performs 0 provider calls and returns an identical `Answer` | AT-CAC-01 |
| FR-CAC-02 | MUST | Semantic cache: embedding similarity ≥ a per-cohort threshold, **plus** mandatory guards — entity equality (order ID, amount, date, plan, locale), source-doc existence and chunk-hash equality, corpus/index/prompt/model version equality | Guard matrix: any guard mismatch → miss (6 fixtures, one per guard) | AT-CAC-02 |
| FR-CAC-03 | MUST | A hit is re-validated cheaply before display: citations still resolve, quoted spans still match, INV-04 numeric guard re-run; failure demotes the hit to a miss | Tampered-chunk fixture: hit becomes miss and no stale citation is displayed | AT-CAC-03 |
| FR-CAC-04 | MUST | The cache stores the **whole artifact**: answer, claims, citation spans, retrieval fingerprint, corpus/index versions, prompt version, model id, decision and reason code | Round-trip equality test of the stored artifact; cache entry schema version pinned | AT-CAC-04 |
| FR-CAC-05 | MUST | Invalidation is version-keyed: any corpus-version bump invalidates every entry; deleting or changing a document invalidates entries whose citations reference it; TTL is per content-volatility class (policy/price short, how-to long) | After ingest-driven version bump: 0 valid entries; doc-delete fixture: exactly the affected entries invalidated; TTL fixtures per class | AT-CAC-05 |
| FR-CAC-06 | MUST | High-risk intents (account, refund, legal, payment) bypass the semantic cache entirely (exact-match only) and are tagged in config, not code | Fixture: 0 semantic hits on the high-risk intent list; config-driven tag change is reflected without code edit | AT-CAC-06 |
| FR-CAC-07 | MUST | Wrong-hit budget is calibrated and measured: per-cohort thresholds are chosen so the measured wrong-hit rate on the near-miss replay set (≥ 120 adversarial pairs) meets §9.5 | Release: wrong-hit rate ≤ 0.01, Wilson 95% upper bound ≤ 0.02; CI: calibration routine on synthetic distributions hits the target | AT-CAC-07 |
| FR-CAC-08 | MUST | Cache observability and kill switch: hit/miss/bypass counters, wrong-hit counter, per-intent rates exposed as metrics and in the report; one env var / CLI flag disables the cache and the bypass is provable | Metrics present in `/metrics`; with cache disabled, 0 hits and identical answers | AT-CAC-08 |
| FR-CAC-09 | MUST | Negative caching: escalated/refused turns may be cached as non-answers for a bounded TTL (default 600 s), never serve an answer, and are invalidated by version bumps | Fixture: negative entry never yields prose; TTL expiry test | AT-CAC-09 |

### 4.6 Service (module `service`, `telemetry`; owner t9)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-SVC-01 | MUST | `POST /v1/chat` streams SSE events in a documented order (`retrieval` → `tokens`* → `citations` → `decision` → `done`), each carrying `trace_id`; citations are emitted only after verification | Event-order assertion over a fixture stream; `trace_id` on every event | AT-SVC-01 |
| FR-SVC-02 | MUST | Sessions and turns persist in SQLite; history is replayed into the pipeline; requests accept an idempotency key and replaying a key returns the stored response without a second model call | Restart test: history survives process restart; duplicate-key test: 1 provider call | AT-SVC-02 |
| FR-SVC-03 | MUST | Consistent error taxonomy: typed error envelope with code, message, `trace_id`, retryable flag; HTTP status mapping documented | Table-driven test over every error code | AT-SVC-03 |
| FR-SVC-04 | MUST | Protected admin ingestion endpoint (`POST /admin/ingest`) is bearer-token guarded, disabled without a configured token, and returns manifest stats | 401 without token; 403 with wrong token; disabled-by-default test | AT-SVC-04 |
| FR-SVC-05 | MUST | Feedback (`POST /v1/feedback`) and escalation-ticket endpoints persist the operator/customer action against the trace | Round-trip test; trace linkage asserted | AT-SVC-05 |
| FR-SVC-06 | MUST | Rate limiting is enforced per client with documented limits and a `429` envelope | Burst test returns 429 with retry hint | AT-SVC-06 |
| FR-SVC-07 | MUST | `/healthz` reports dependency status (index, corpus version, provider config, cache, calibration artifact) without leaking secrets | Health payload schema test; no secret substring present | AT-SVC-07 |
| FR-SVC-08 | MUST | Every request writes one JSONL trace record: stage latencies, retrieval scores, fusion contributions, cache status, `p̂`, decision, reason code, tokens, USD, model, prompt version, corpus/index version — with PII redacted | Trace schema test; PII-absence scan on fixture traces | AT-SVC-08 |

### 4.7 Operator console (module `web`; owner t10)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-UI-01 | MUST | Live queue lists conversations with locale badge, wait time, state and reason code; states are filterable | Component test: 3 seeded conversations render with badges; filter narrows the list | AT-UI-01 |
| FR-UI-02 | MUST | Conversation view shows the thread and the suggested draft with inline citation markers that are clickable | Component test: clicking marker `[2]` opens the passage panel for citation 2 | AT-UI-02 |
| FR-UI-03 | MUST | Citation panel shows document title, heading path, quoted span with highlight and the chunk ID; it never shows text not in the cited span | Component test asserts rendered quote equals payload quote | AT-UI-03 |
| FR-UI-04 | MUST | Diagnostics drawer shows retrieval candidates with per-channel scores and fusion contribution, rerank deltas, cache status, verifier verdict per claim, `p̂`, α and the threshold reference | Component test against a fixture payload covering all fields | AT-UI-04 |
| FR-UI-05 | MUST | Escalation state, reason code and confidence are visible without opening the drawer; a high-risk escalation is visually distinct | Component test: escalated fixture shows badge + reason; accessible name present | AT-UI-05 |
| FR-UI-06 | MUST | Operator actions: edit draft, send, escalate (with reason), each firing exactly one API call and reflected optimistically | Component test with MSW: each action → 1 request, UI state updates | AT-UI-06 |
| FR-UI-07 | MUST | Analytics view renders latency p50/p95, USD/answer, escalation rate and cache hit rate for a selectable window from the API | Component test with fixture series; empty/error states handled | AT-UI-07 |
| FR-UI-08 | MUST | Streaming renders progressively from SSE and degrades gracefully when streaming is unavailable (fallback to final payload) | Component test: token-by-token render; fallback path test | AT-UI-08 |
| FR-UI-09 | MUST | Offline development mode via MSW fixtures, so the console builds and tests without a backend | `npm run test` passes with no network | AT-UI-09 |
| FR-UI-10 | MUST | Security and accessibility: no `dangerouslySetInnerHTML` on model or document content without sanitisation; keyboard navigation for all actions; contrast ≥ WCAG AA | vitest XSS payload test (script in chunk text does not execute); a11y lint report with 0 critical violations | AT-UI-10 |
| FR-UI-11 | MUST | Committed, reproducible screenshots of all main screens under `docs/assets/` produced by one script | `npm run screenshots` regenerates identical-or-updated assets; ≥ 4 screens covered | AT-UI-11 |

### 4.8 Evaluation harness (module `eval`; owner t7)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-EVL-01 | MUST | Golden-set schema separates retrieval labels (gold chunk ids) from answer labels (reference answer, required citations, answerability, category, locale, intent, risk tier) | Schema validation; a case missing either label class fails loudly | AT-EVL-01 |
| FR-EVL-02 | MUST | Datasets are generated reproducibly from the corpus with a seed and meet the composition in §10.1 (categories, per-locale counts, unanswerable share) and the split rules in §10.2 | Generated manifests satisfy every quota assertion; same seed → identical hash | AT-EVL-02 |
| FR-EVL-03 | MUST | Split manifests are frozen and hashed; the harness fails loudly on split leakage (a test-split document appearing in dev) and on missing labels | Leakage fixture triggers a hard failure with the offending ids | AT-EVL-03 |
| FR-EVL-04 | MUST | Metrics are implemented and unit-tested against hand-computed values: recall@k (any-hit and all-gold variants), nDCG@k (binary relevance), MRR@k, groundedness, citation precision/recall, escalation precision/recall/coverage, ECE, Brier, AURC | Every metric has ≥ 3 hand-computed cases including tie and empty-set edge cases; all match to 1e-9 | AT-EVL-04 |
| FR-EVL-05 | MUST | A judge runner supports a deterministic `FakeJudge` for CI and a real rubric-based LLM judge with: context-anchored rubric, position swap for pairwise, judge model ≠ generator model family, recorded judge model + prompt version, and per-language human-agreement calibration | Judge contract test; position-swap test shows order invariance within tolerance; report includes judge metadata and per-language κ | AT-EVL-05 |
| FR-EVL-06 | MUST | Calibration and risk analysis are produced as artifacts: reliability data, ECE (equal-width and equal-mass), Brier, risk–coverage curve, AURC, achieved coverage and realised risk at the operating point, per intent bucket | Artifact schema test; report embeds the values; plot files exist | AT-EVL-06 |
| FR-EVL-07 | MUST | Statistics: paired bootstrap over queries (B = 1000, 2.5/97.5 percentiles), McNemar (χ² with the exact-binomial fallback when `b + c < 25`), Wilson intervals for proportions; every reported table cell carries n and a 95% CI | Statistics unit tests against closed-form cases; report contains no metric without n and CI | AT-EVL-07 |
| FR-EVL-08 | MUST | The harness runs end-to-end offline with `FakeProvider` + `FakeJudge` + hashing embedder, writes a markdown report plus plots into `reports/<run>/`, and maintains a stable `latest` pointer | `answerdesk eval --synthetic --provider fake --report reports/offline-smoke` exits 0 and writes ≥ 1 `.md` + ≥ 1 `.png` | AT-EVL-08 |
| FR-EVL-09 | MUST | Reports print the corpus hash, golden-set hash, split manifest hash, seed, judge identity and provider mode on the first page; REPLAY-MODE runs are labelled in a banner and per-table | Header fields present; replay banner present when applicable | AT-EVL-09 |
| FR-EVL-10 | MUST | The full evidence run (`answerdesk eval --all --report reports/bench`) covers the ablations in §10.6 within the time budget (≤ 90 min with API provider), and its numbers are the only ones allowed into `docs/results.md` | Ablation matrix complete; runtime recorded; every `results.md` number maps to a `reports/bench` artifact | AT-EVL-10 |
| FR-EVL-11 | MUST | `docs/eval-methodology.md` states splits, metric definitions with formulas, judge bias controls, statistical procedure, and the limitations in §10.7 | Reviewer checklist CL-EVL-01 confirms every subsection present and every formula matches code | AT-EVL-11 |

### 4.9 CLI, configuration and operations (modules `config`, root CLI; owners t3, t9)

| ID | Lvl | Requirement (testable statement) | Target / pass condition | AT |
|---|---|---|---|---|
| FR-OPS-01 | MUST | A single `answerdesk` CLI exposes documented subcommands: `ingest`, `index`, `ask`, `eval`, `calibrate`, `serve`, `demo`; every subcommand has `--help`, and non-zero exit codes on failure | `answerdesk --help` lists all subcommands; each `--help` exits 0 | AT-OPS-01 |
| FR-OPS-02 | MUST | `answerdesk index` builds both indexes from the corpus manifest, mints a corpus version, and is idempotent for an unchanged corpus | Second run reports unchanged and reuses the version | AT-OPS-02 |
| FR-OPS-03 | MUST | Provider selection is configuration only: presets for `openai`, `deepseek`, `openrouter`, `vllm`, `llama.cpp`, `ollama` differ solely in default `base_url`/`model`/key requirements; local presets require no API key | Preset table test; `answerdesk ask` against a local mock OpenAI-compatible server succeeds with 0 keys set | AT-OPS-03 |
| FR-OPS-04 | MUST | Profiles bundle settings: `offline` (hashing embedder, fake provider, no network), `api`, `local`; `offline` is the CI default and requires no secrets | Profile resolution test; `answerdesk eval` with no env vars picks `offline` | AT-OPS-04 |
| FR-OPS-05 | MUST | Configuration is validated at startup with actionable errors (missing key for a remote preset, unreachable local server, unknown profile) and never logs secrets | Error-message tests; secret-redaction test on the config dump | AT-OPS-05 |
| FR-OPS-06 | MUST | `answerdesk demo --smoke` runs ingest → index → ask → decision → report with doubles and exits 0, completing in ≤ 120 s on the reference machine | Smoke run exit 0; duration recorded | AT-OPS-06 |

---

## 5. Non-functional requirements

Each NFR states a number, the measurement procedure, and the gate. Measurement details that are shared (datasets, statistics) live in §§6–10.

| ID | Lvl | Requirement | Target / pass condition | Gate |
|---|---|---|---|---|
| NFR-LAT-01 | MUST | End-to-end answer latency, server-side wall clock (arrival → final byte), API provider, n ≥ 200 golden queries, reference machine, cold start excluded | **p50 ≤ 1,500 ms; p95 ≤ 4,000 ms**; time-to-first-token p95 ≤ 1,200 ms. **Measured on the primary arm (n=200, generation stage, judge excluded): p50 2,618.22 ms / p95 14,116.76 ms → MISS.** The target is not amended; the miss is reported in `docs/results.md` ("NFR-LAT-01 verdict") and logged as CC-2026-10-21-b in §18.5 | G-REL (AT-LAT-01) |
| NFR-LAT-02 | MUST | Retrieval-only latency (analysis → BM25 + dense → fusion → rerank), CPU, per query | p95 ≤ **200 ms** at the demo corpus (≥ 1,500 chunks); p95 ≤ **500 ms** at the operating point (≥ 20,000 chunks) | G-REL (AT-LAT-02) |
| NFR-LAT-03 | MUST | Offline pipeline latency with doubles (no network, no reranker), used as a CI regression guard | p95 ≤ **1,500 ms** for a fixture query on CI hardware | G-CI (AT-LAT-03) |
| NFR-LAT-04 | MUST | Cache hit latency, server-side, excluding client | p95 ≤ **150 ms** (semantic hit), exact hit ≤ 50 ms | G-REL (AT-LAT-04) |
| NFR-COST-01 | MUST | A cost ledger records tokens (input/output), USD and latency for **every** model call, with token counts taken from provider usage when available and reconciled against the local tokeniser | Reconciliation mismatch ≤ **0.5%** per call batch; ledger present in every trace record | G-CI (AT-COST-01) |
| NFR-COST-02 | MUST | Cost per answered query at the reference profile (pricing table pinned with `retrieved_at` and source) | **mean ≤ $0.005/answer; p95 ≤ $0.010/answer**; cache hits ≤ **$0.0002**; a pricing change is a config edit, never a code edit | G-REL (AT-COST-02) |
| NFR-COST-03 | MUST | Cost and latency are recorded per stage (analysis/retrieval/rerank/generation/verify) so a regression can be attributed | Stage breakdown present in ≥ 95% of traces | G-CI (AT-COST-03) |
| NFR-DET-01 | MUST | Deterministic offline CI: `uv run pytest -q` passes with no network, no API key, no GPU, no Docker, within **10 minutes**, on Linux **and** Windows runners | CI green on both OSes; a network-blocked run passes | G-CI (AT-DET-01) |
| NFR-DET-02 | MUST | Eval reproducibility: fixed seeds; two runs with the same seed produce metrics files that are byte-identical after removing timing fields (compared by hash) | Hash equality test over two offline runs | G-CI (AT-DET-02) |
| NFR-DET-03 | SHOULD | CI is split into a fast gate (≤ 3 min: unit + contract + traceability) and the full offline gate (≤ 10 min) — implemented as **two jobs in one workflow** (`fast` and `test` in `.github/workflows/ci.yml`), which is what the requirement asks for (the offline equivalent is `scripts/ci-offline.ps1`); per-job wall-clock durations are recorded by GitHub, and the job budgets are enforced by `timeout-minutes: 10` / `20` | Two workflows, measured durations recorded | G-CI (AT-DET-03) |
| NFR-PORT-01 | MUST | Provider abstraction: one OpenAI-compatible HTTP client; six presets; streaming, structured output, timeouts, retries with jitter, fallback chain, circuit breaker; switching provider or pointing at a local server is a config change with no code change and no test edit | Preset table test; mock-server integration test per transport feature; reviewer checklist CL-PORT-01 confirms no provider-specific branches | G-CI + G-REV (AT-PORT-01) |
| NFR-PORT-02 | MUST | Runs on Windows 11 + Python 3.12 with no Docker; the quickstart works from a clean checkout; Linux/macOS parity is CI-verified for the offline gate | Clean-checkout quickstart transcript in `docs/results.md`; CI matrix includes Windows | G-CI (AT-PORT-02) |
| NFR-PORT-03 | SHOULD | Optional heavy dependencies (torch, sentence-transformers, cross-encoder, qdrant client, fastapi) live behind extras; the base install imports in ≤ 3 s with no CUDA requirement | Import-time test of the base install; extras table documented | G-CI (AT-PORT-03) |
| NFR-SEC-01 | MUST | Injection resistance by architecture: no tool call, routing change, prompt switch or guard bypass can originate from retrieved content; injection fixtures (≥ 40) yield 0 auto-answers that follow injected instructions | Fixture suite green; guard flags recorded | G-CI + G-REL (AT-SEC-01) |
| NFR-SEC-02 | MUST | PII is redacted at three boundaries — before indexing, before logging/tracing, before any external provider request body — and the redaction is verifiable in a mock-server capture | Mock provider request body contains no fixture PII; traces contain no fixture PII | G-CI (AT-SEC-02) |
| NFR-SEC-03 | MUST | No secrets in the repository: `.env` ignored, `.env.example` placeholder-only, CI secret scan over tracked files and history | Secret-scan job green; 0 findings | G-CI (AT-SEC-03) |
| NFR-SEC-04 | MUST | Console output escaping: model/document text rendered as text (or sanitised markdown); XSS payload in a chunk never executes | vitest DOM test with script payload passes | G-CI (AT-SEC-04) |
| NFR-SEC-05 | MUST | Dependency audit runs in CI; no known-vulnerable dependency at high/critical severity at release time; findings recorded | `pip-audit` (or equivalent) report committed with 0 high/critical. Honesty note: `pip-audit` needs network, so the offline gate runs the licence half (`scripts/check-licenses.py`, CMD-LIC); the vulnerability half stays a networked release step documented in `docs/reproducibility.md` | G-REL (AT-SEC-05) |
| NFR-QUAL-01 | MUST | Lint clean (`ruff check`), formatting enforced, type checking on `src/answerdesk` with a documented strictness level and 0 errors | CI jobs green | G-CI (AT-QUAL-01) |
| NFR-QUAL-02 | MUST | Line coverage ≥ **70%** for `src/answerdesk/{retrieval,policy,cache,eval}`; overall coverage reported (no threshold on UI code) | Coverage report in CI; threshold enforced for the four math modules. Gate: the coverage step of `scripts/ci-offline.ps1` (`--cov=src/answerdesk/{retrieval,policy,cache,eval} --cov-fail-under=70`); measured 2026-10-21: cache 85.5%, eval 73.6%, policy 90.0%, retrieval 92.5%, aggregate 80.7% (CC-2026-10-21-d) | G-CI (AT-QUAL-02) |
| NFR-QUAL-03 | MUST | Tests are meaningful: no vacuous assertions, no blanket skips; the reviewer spot-checks ≥ 5 modules by mutation-style breakage and confirms tests fail | Review findings show 0 modules whose breakage goes undetected. Evidence: `docs/review-report.md` §2.1 (8 mutations, 7 caught, 1 open finding R-04 on the numeric guard); `tests/smoke/test_requirements_evidence.py` asserts the audit is recorded | G-REV (AT-QUAL-03) |
| NFR-ML-01 | MUST | All six locales appear in the golden set and in every per-locale table; no locale may be silently dropped from reporting | Per-locale rows present for ru/tg/uz/ky/tr/zh in every metric table | G-CI (AT-ML-01) |
| NFR-ML-02 | MUST | No default machine translation; the stack answers in the source language and preserves identifiers (INV-04) | Translation-call absence test (mock provider sees no translate endpoint); identifier fingerprint fixtures | G-CI (AT-ML-02) |
| NFR-REP-01 | MUST | Every published number is reproducible: `README.md` and `docs/results.md` numbers each map to a committed `reports/<run>/` artifact and a documented command; no hand-written number anywhere | Reviewer checklist CL-REP-01 + the number checker (`scripts/check-numbers.py`, CMD-NUM) fails on any quantity without an artifact or a documented exception | G-REV + G-CI (AT-REP-01) |
| NFR-REP-02 | MUST | Provenance: corpus manifest, golden manifest, split manifest, calibration artifact and report header carry sha256 hashes; every run prints them | Hashes present and stable across re-runs | G-CI (AT-REP-02) |
| NFR-REP-03 | MUST | `scripts/check-traceability.py` parses §16 of this document and asserts that every `AT-` ID appears as a marker in the repository and that every requirement ID resolves to a module path that exists; exits non-zero listing gaps | Script exits 0 on the release candidate and fails on a seeded gap (the smoke test exercises both directions) | G-CI (AT-REP-03) |
| NFR-REP-04 | MUST | Dependency and license audit: runtime dependency tree contains only permissive licenses (allowlist in §13.3), no Dify/FastGPT/Verba code or forked source, Apache-2.0 code attributed if ever vendored (nothing is vendored in v0.1) | License report committed; 0 disallowed licenses. Enforced offline by `scripts/check-licenses.py` (CMD-LIC): installed distributions classified from metadata, then from bundled licence text, every override carrying its reason | G-REL (AT-REP-04) |
| NFR-OBS-01 | MUST | Observable decisions: `/metrics` exposes request, latency (histogram), error, cache hit/miss/wrong-hit, escalation and refusal counters; JSONL traces carry per-stage latency, scores, decision and reason | Metrics endpoint scrape test; trace schema test | G-CI (AT-OBS-01) |
| NFR-OBS-02 | MUST | The console analytics view reads real API series (not hard-coded fixtures) and shows at least latency, cost, escalation rate and cache hit rate | Component test against a mocked API; manual run against the real service recorded in the release notes | G-REL (AT-OBS-02) |
| NFR-OPS-01 | MUST | Failure modes are documented and exercised: provider timeout, provider 5xx, malformed model JSON, empty retrieval, index missing, calibration artifact missing, corpus version mismatch — each has a defined behaviour and a test | Failure-mode matrix in `docs/ops.md` cross-referenced to tests; each row has a test | G-CI + G-REV (AT-OPS-07) |
| NFR-DOC-01 | MUST | Documentation set exists and is accurate: `docs/architecture.md`, `design-decisions.md`, `ingestion.md`, `retrieval.md`, `generation.md`, `escalation.md`, `cache.md`, `eval-methodology.md`, `api.md`, `ops.md`, `math.md`, `reproducibility.md`, `results.md`, `review-report.md`, `release-checklist.md`, plus `README.md` (EN) and `README.ru.md` | Reviewer checklist CL-DOC-01: every file present, every command in README executed successfully | G-REV (AT-DOC-01) |
| NFR-DOC-02 | MUST | English repository: code, comments, docs, commit messages and issues in English; the only non-English artifacts are locale corpora, localised UI strings and `README.ru.md` | Checklist CL-DOC-02 spot-check of 20 files | G-REV (AT-DOC-02) |
| NFR-A11Y-01 | SHOULD | Console meets WCAG 2.1 AA for contrast and keyboard operability; automated a11y check with 0 critical violations | Automated a11y evidence is the aria/keyboard assertions in the console test suite, mapped one-to-one in `tests/smoke/test_ui_traceability.py`; a dedicated axe run is roadmap, not claimed | G-CI (AT-A11Y-01) |
| NFR-SCALE-01 | MUST | At the operating point (≥ 3,000 docs / ≥ 20,000 chunks) on the reference machine: index build ≤ **10 min**, peak RSS ≤ **4 GB**, retrieval p95 ≤ 500 ms (NFR-LAT-02) | The scale *knob* is proven offline (`docs_per_locale` and per-locale manifest depth, `tests/smoke/test_requirements_evidence.py`); the 3k-document run itself is a G-REL step recorded in the evidence artifacts, not claimed here | G-REL (AT-SCALE-01) |

---

## 6. Retrieval quality: targets and measurement

### 6.1 Metric definitions (normative)

Let `Q` be the evaluated query set, `top_k(q)` the ranked chunk ids returned by the pipeline, `G_q` the gold chunk set of query `q`, `|Q|` the number of queries.

```
recall_any@k  = (1/|Q|) · Σ_q  1[ G_q ∩ top_k(q) ≠ ∅ ]          # any-hit: the primary gate
recall_all@k  = (1/|Q|) · Σ_q  |G_q ∩ top_k(q)| / |G_q|          # completeness: reported alongside
```

```
nDCG@k = (1/|Q|) · Σ_q  DCG_q@k / IDCG_q@k
DCG_q@k = Σ_{i=1..k}  rel_i / log2(i+1),          rel_i ∈ {0,1}
IDCG_q@k = Σ_{i=1..min(k,|G_q|)} 1 / log2(i+1)
```

```
MRR@k = (1/|Q|) · Σ_q  1 / rank_q(first gold chunk)   if it appears within k, else 0
```

Rationale for two recall variants: `recall_any` is what the *user* experiences (at least one right source made it into the context); `recall_all` is what the *answer* needs when evidence is spread across chunks. Reporting only one of them is how retrieval numbers become unfalsifiable. `[A]`

All retrieval metrics are computed **at the k actually fed to the generator** and at the rerank input k; a good nDCG@10 with a 12-chunk context is a mismatch that this spec forbids by construction (D-05, D-06). `[A]`

### 6.2 The k values, and why

| k | Role | Why this k |
|---|---|---|
| 50–150 | rerank input window (default 50) | Cross-encoder reranking over a wide window is where the largest reported gains come from (`[S]` one vendor experiment: top-20 failure 5.7% → 1.9% with reranking); the upper bound is a latency guard. |
| 20 | **primary retrieval gate** (`recall_any@20`) | The recall ceiling for the pipeline: the reranker cannot recover what retrieval never returned, and 20 is the widest window we can afford to rerank on CPU. |
| 12 | generator context cap (default; range 8–12) | Lost-in-the-middle is confirmed (`[C]` TACL 2023) and more tokens is not monotonically better (`[S]` context rot); 8–12 keeps the strongest evidence at both ends. |
| 10 | nDCG@10 / MRR@10 reporting | Standard IR comparability; also the practical "top of the context" region. |

### 6.3 Measurement procedure (identical for every release)

1. **Split**: `golden/v1` **test** split only (§10.2). The dev split is used to fit fusion parameters, α and cache thresholds; it is never used for a reported gate.
2. **Embedder**: at least one real multilingual embedder must be used for a G-REL retrieval claim (BGE-M3 default, Qwen3-Embedding-0.6B as documented alternative; `[S]` MTEB values for orientation only). The hashing embedder is CI-only and its numbers may never appear in `results.md` as a quality claim.
3. **Ablation arms**: `bm25`, `dense`, `hybrid-rrf`, `hybrid-cc`, `hybrid-cc+rerank` (SHOULD), `+mmr` (SHOULD). Each arm runs on the same frozen split with the same tokenisation and the same query analysis.
4. **Aggregation**: per-locale metrics first, then macro-average over the six locales with equal weight (not item-weighted) so that a well-resourced locale cannot mask a broken one.
5. **Uncertainty**: every cell carries `n` and a 95% CI (bootstrap for means, Wilson for proportions, §10.4). Any claimed improvement between arms must survive a paired bootstrap over queries (`B = 1000`, CI lower bound > 0) and, for binary any-hit outcomes, McNemar.
6. **Artifacts**: `reports/<run>/retrieval/*.json` + `*.csv` + the markdown table in `docs/results.md`, with corpus hash, golden hash and split hash in the header (NFR-REP-02).

### 6.4 Release targets (gate: `AT-RET-10`, G-REL)

Thresholds are floors, not aspirations: a release either meets them or amends this section (§18). Numbers are macro-averaged over ru/tg/uz/ky/tr/zh on `golden/v1` test.

| # | Metric | Threshold | Rationale / tag |
|---|---|---|---|
| RT-1 | `recall_any@20` (hybrid, tuned) | **≥ 0.90** | The pipeline's recall ceiling; below this the generator is being asked to answer from evidence it never saw. `[A]` |
| RT-2 | `recall_any@20` per locale (floor) | **≥ 0.80** every locale | Prevents a macro average from hiding a broken low-resource locale (NFR-ML-01). `[A]` |
| RT-3 | `recall_any@10` | **≥ 0.80** | What the packed context can actually use. `[A]` |
| RT-4 | `recall_all@20` | **≥ 0.75** | Multi-chunk evidence cases; reported to expose recall_any inflation. `[A]` |
| RT-5 | `nDCG@10` | **≥ 0.65** | Ranking quality at the top of the context. `[A]` |
| RT-6 | `MRR@10` | **≥ 0.60** | First-hit quality; ties directly to citation precision. `[A]` |
| RT-7 | Hybrid uplift: `recall_any@20(hybrid) − recall_any@20(best single channel)` | **≥ +0.03 absolute**, paired-bootstrap 95% CI lower bound **> 0** | Proves hybrid retrieval is not decoration; the ablation the research says nobody publishes. `[S]` |
| RT-8 | Identifier/numeric subset `recall_any@20` (n ≥ 60) | **≥ 0.85** | Exact-ID queries are the documented dense failure mode; this is why the lexical must-match arm exists (D-07). `[S]` |
| RT-9 | Perturbation delta: `recall_any@20(typo/code-switch subset) − recall_any@20(clean subset)` | **≥ −0.05** (no more than 5 points lost) | Support traffic contains typos and code-switching; the research found no credible public benchmark, so we set our own budget. `[A]` |
| RT-10 | Rerank effect (SHOULD arm): `recall_any@10` after rerank **≥** before rerank, and `nDCG@10` improves **≥ +0.02** with CI lower bound > 0 | as stated | Quantifies the rerank claim on our own data instead of quoting a vendor blog. `[S]`/`[A]` |

**Falsification clause.** If RT-7 fails (RRF ≥ tuned CC on our data), the shipped default becomes RRF and the ADR records it — this is precisely open question Q1, and the requirements are written so the answer can be "no". If RT-2 fails for a specific locale, the locale stays in the product but its retrieval path is documented as degraded and the release claims exclude it; the requirement itself is then amended. `[A]`

### 6.5 CI-verifiable companions (gate: `AT-RET-*`, G-CI)

The numbers above need a real model; the *machinery* does not. These are asserted offline on every commit:

| Check | Assertion | AT |
|---|---|---|
| Fusion math | RRF and CC outputs equal hand-computed values on 3 fixtures, including a degenerate flat-score case | AT-RET-04 |
| Exact-NN equivalence | Dense index top-k equals brute-force cosine top-k on a 200-vector fixture (ids and order) | AT-RET-02 |
| Determinism | Same corpus + query → identical ranking and scores across two runs and two processes | AT-RET-01 |
| Metric correctness | `recall_any`, `recall_all`, `nDCG@k`, `MRR@k` match hand-computed values including empty/tie cases | AT-EVL-04 |
| Leakage | A test-split item present in the fusion-fitting input raises a hard error | AT-RET-05 |
| Identifier protection | Digit/ID fingerprints unchanged by query analysis | AT-RET-09 |
| Trace completeness | Every pipeline stage contributes to the trace object | AT-RET-08 |

### 6.6 Ablations that must be runnable (not necessarily run in CI)

`answerdesk eval --all` must be able to run: 2 fusers (RRF, CC) × at least 2 embedder families; rerank on/off; MMR on/off; chunking variants (structure-first vs naive fixed-window; contextual headers on/off, SHOULD); packing k ∈ {5, 8, 12, 20}. This is the evidence for open questions Q1, Q2, Q3, Q7, Q8; the harness must make them one flag apart, and the artifacts must be committed for the arms actually run.

---

## 7. Grounded generation and the citation contract

### 7.1 The answer object (frozen shape)

```text
Answer
  answer_id: str
  trace_id: str
  language: str                      # BCP-47-ish tag of the answer, must equal query language (FR-GEN-06)
  text: str                          # rendered, markers derived from citations[]
  claims: [ Claim
              claim_id: str
              text: str                # atomic assertion
              citation_ids: [int]      # >= 1, ordered by first appearance
              support: supported | partial | unsupported   # verifier verdict
          ]
  citations: [ Citation
                 citation_id: int
                 chunk_id: str
                 doc_id: str
                 doc_title: str
                 heading_path: str
                 span: {start: int, end: int}
                 quote: str             # verbatim substring of chunk text (whitespace-normalised)
                 doc_hash: str          # sha256 of the chunk text at retrieval time
             ]
  decision: { action: answer|clarify|escalate|refuse, reason_code: <enum>, score: float,
              alpha: float, threshold_ref: str }
  confidence: { p_correct: float, calibrator_version: str }
  cache: { status: hit|miss|bypassed, entry_id: str|null, similarity: float|null }
  usage: { input_tokens: int, output_tokens: int, usd: float, model: str, prompt_version: str }
  provenance: { corpus_version: str, index_version: str, prompt_version: str,
                verifier_version: str, scorer_version: str }
```

This shape is the contract between `generate`, `policy`, `cache`, `service` and `web`. Adding a field is allowed; removing or repurposing one requires amending this document (it is part of t3's `contracts.py`).

### 7.2 Citation rules (each rule is independently testable)

| Rule | Requirement | Test |
|---|---|---|
| CR-1 | Every claim has ≥ 1 citation; a claim without support is not rendered (§7.3) | AT-GEN-01 |
| CR-2 | Every `citation_id` resolves to a chunk in the packed context — no phantom citations | AT-GEN-02 |
| CR-3 | `quote` is a whitespace-normalised verbatim substring of that chunk's stored text | AT-GEN-02 |
| CR-4 | `span` maps to the chunk's character offsets and contains the quote | AT-GEN-02 |
| CR-5 | Markers in `text` and `claims[].citation_ids` are bijective; numbering is by first appearance | AT-GEN-01, AT-GEN-03 |
| CR-6 | `doc_hash` is recorded and re-checked at display/cache-hit time (stale citation prevention) | AT-CAC-03 |
| CR-7 | Verdict `unsupported` ⇒ repair once, then remove the claim or escalate; never display | AT-GEN-07 |
| CR-8 | A refusal/escalation carries no prose answer (no "best guess" text) | AT-GEN-10 |
| CR-9 | Numeric/date guard (INV-04): every number or date in a displayed claim appears, after canonicalisation, in a cited chunk | AT-GEN-02, AT-GEN-12 |
| CR-10 | The answer language equals the query language; a fallback answer is explicitly marked | AT-GEN-06 |
| CR-11 | Document text is never reproduced as instructions; the renderer treats claim text as data in the UI (no raw HTML) | AT-UI-10 |
| CR-12 | Streaming never reveals a citation that is later retracted: citations are emitted only after verification | AT-SVC-01 |

CR-9 is deliberately strict: numeric claims are where support-checking models are weakest (`[A]`, open question Q10) and where a customer is most damaged by a plausible wrong number (price, fee, deadline).

### 7.3 Verification and repair state machine

```
generated claims ──► verifier (per claim, per cited passage)
                        ├─ supported            → keep, render
                        ├─ partial              → keep only if the partial part is non-numeric; mark support=partial
                        └─ unsupported          → repair pass (exactly one)
                                                   ├─ now supported → keep
                                                   └─ still unsupported → drop claim;
                                                                          if no claim survives → escalate(NO_EVIDENCE / VERIFIER_REJECT)
```

* The verifier is a separate, versioned component (`verifier_version` in the answer) — it may be NLI-based, judge-based, or a deterministic numeric-check ensemble; whichever it is, CR-9's numeric guard is always deterministic.
* The repair budget is **one** pass. Unbounded repair loops are a latency and cost hazard.
* A false rejection (dropping a good claim) is a real cost. The budget is stated in §7.5 (F-3) because a verifier that suppresses good answers makes the bot useless (`[A]`, Q10).

### 7.4 Prompt and context contract

| Element | Rule |
|---|---|
| Prompt registry | Versioned prompts with content hash (FR-GEN-04); any change bumps the version and is recorded in every answer. |
| Untrusted context | Retrieved passages are wrapped in explicit delimiters/labels; the system prompt states that context is *data* and cannot change the task, the output schema or the routing (INV-01, FR-GEN-11). |
| Context budget | Hard token cap; default packing 8–12 chunks; dedupe above cosine 0.95; ordering rank-1 first, rank-2 last (FR-GEN-05). |
| Citations | The model is asked for claim/citation structure (JSON), never for prose markers; markers are rendered from structure. |
| Language | Answer in the query language; identifiers stay verbatim; no translation of quoted spans. |
| Refusal | An explicit "insufficient evidence" output channel exists in the schema; the model is never asked to "try your best". |

### 7.5 Release targets (gate: `AT-GEN-08`, G-REL)

| # | Metric | Threshold | Definition / caveat |
|---|---|---|---|
| F-1 | Groundedness (claim entailment rate) | **≥ 0.90** macro | Fraction of displayed claims judged `supported` by the calibrated judge; judge bias controls in §10.5. `[C]` metric families (RAGAS/ARES), `[A]` threshold. |
| F-2 | Citation precision | **≥ 0.90** | Fraction of citations that actually support their claim (ALCE `citation_precision`). `[C]` |
| F-3 | Citation recall | **≥ 0.85** | Fraction of answer statements fully supported by at least one cited passage (ALCE `citation_recall`); the sobering public baseline is ~50% failure even for strong systems, so this is a real target, not a formality. `[C]` |
| F-4 | Unsupported-claim display | **0** | Structural: no displayed claim has verdict `unsupported`. Deterministic, also checked offline. |
| F-5 | Phantom citations / bad quotes | **0** | Structural, deterministic. |
| F-6 | Numeric-guard violations | **0** displayed | Deterministic; the release run reports the violation rate of the *draft* before repair as a diagnostic. |
| F-7 | Language agreement | **≥ 0.98** macro | Detector-based check per locale; mis-detections on short queries are excluded only with a documented rule. |
| F-8 | Refusal correctness on the unanswerable subset | **≥ 0.95** escalate/refuse, i.e. auto-answer ≤ 0.05 | Coupled to §8.5 E-1; the same α protects both. |
| F-9 | Verifier quality on the claim-level labelled set (n ≥ 200 claims) | precision **≥ 0.90**, recall **≥ 0.85**, false-rejection ≤ **0.10** | The false-rejection budget is the guard against an over-cautious verifier (Q10). `[A]` |
| F-10 | Judge–human agreement (per language) | Cohen's κ **≥ 0.60** required to report a judged metric for that language; target **≥ 0.70** | If κ is below 0.60, judged metrics for that locale are reported as "not reportable" and only deterministic metrics are shown (§10.5). |

**Honesty clause.** F-1/F-2/F-3 are judge-mediated and inherit judge bias. They may be reported only alongside κ, the judge model identity, the rubric version and the number of human calibration labels. A judged number without those four is a defect, not a result.

---

## 8. Escalation policy contract

### 8.1 Decision score and conformal calibration

For a request `x` with packed evidence `E`, the scorer produces `p̂(x) = P(answer is correct | x, E)` and the risk score `s(x) = 1 − p̂(x)`. Correctness is defined once, before measurement:

> **Correct** = the answer is judged rubric-pass (≥ 4/5 on the task rubric) **and** every displayed claim is `supported` **and** all required citations are present **and** the answer language matches.

Given a calibration set of `n` exchangeable labelled requests with risk scores `s_1..s_n` and target risk `α`:

```
q̂ = the ⌈(n + 1)(1 − α)⌉-th smallest value of {s_i}
auto-answer  iff  s(x) ≤ q̂        ⇒      P(error | auto-answer) ≤ α      (marginal, distribution-free, finite-sample)
```

The guarantee and its assumptions are exactly those of split conformal prediction (`[C]` Angelopoulos & Bates, arXiv:2107.07511). The assumption that matters for us is **exchangeability**: support traffic drifts (new product, new locale, new promotion), and coverage degrades silently when it does (`[S]`/`[A]`). §8.6 therefore mandates refresh and monitoring; a guarantee nobody monitors is a liability, not a feature.

### 8.2 Calibration units and the risk-tier table

Calibration is per **intent bucket**, because the cost of being wrong is not uniform:

| Bucket | Examples | Default α | Default cache behaviour |
|---|---|---|---|
| informational | how-to, fees, hours, eligibility explanation | **0.05** | semantic cache allowed |
| transactional | refunds, payments, plan changes, card blocks | **0.02** | exact-match only |
| account/legal | account closure, fraud, complaints, legal/regulatory | **0.01** | exact-match only, guard-forced escalation allowed |

`α_final = min(α_bucket, C_abstain / C_wrong)` — the cost model can tighten α but never loosen it (see §8.3). Per-bucket and pooled risk are both reported; pooled must also satisfy **≤ 0.05**.

### 8.3 Cost model and the three-way rule

Costs are **configuration, never code constants**: `C_wrong` (a wrong auto-answer), `C_abstain` (escalate to a human), `C_delay` (ask a clarifying question), plus optional `C_review` for a human review queue (v1.1 hook, unused in v1.0).

```
answer    iff  p̂ ≥ 1 − C_delay  / C_wrong
clarify   iff  1 − C_abstain / C_wrong  ≤  p̂  <  1 − C_delay / C_wrong
escalate  otherwise
```

Reference defaults: `C_wrong = 1.00`, `C_abstain = 0.10`, `C_delay = 0.05` ⇒ answer ≥ **0.95**, clarify ∈ [0.90, 0.95), escalate < 0.90. Setting the wrong/abstain ratio to 10 makes the operating point `p̂ ≥ 0.90` on the conformal axis, and the clarify band then trims the bottom of it. `[A]` (decision-theoretically standard; the ratio is a business input the spec exposes rather than hides).

Rules that make the cost model honest:

* Costs live in `configs/policy/*.yaml` with a provenance comment ("pilot measurement", "assumption", date, owner). A cost with no provenance is invalid config and must fail validation.
* Every decision records `threshold_ref`, the calibration artifact hash and the cost-set hash.
* No module may compare `p̂` to a literal; a lint-level review item (CL-POL-01) checks that thresholds are derived.

### 8.4 Guard precedence and reason codes

Precedence is total and deterministic (FR-ESC-04); the first matching rule wins:

```
1. PII guard / injection guard          → refuse or escalate (reason: PII_DETECTED / INJECTION_SUSPECTED)
2. Hard-policy intent (account/legal)   → escalate (reason: HIGH_RISK_INTENT)
3. No usable evidence (empty retrieval, all claims unsupported, verifier reject) → escalate/refuse
4. Conformity check (s ≤ q̂)              → not conforming ⇒ escalate (LOW_CONFIDENCE)
5. Cost-sensitive band                    → answer / clarify / escalate
6. Otherwise                              → answer
```

The closed reason-code enum is in Appendix B; `service`, `cache`, `web` and `eval` must agree on it (parity test AT-ESC-05).

### 8.5 Release targets (gate: `AT-ESC-07`, G-REL)

| # | Metric | Threshold | Note |
|---|---|---|---|
| E-1 | Realised auto-answer risk vs α | observed risk **≤ α** and Wilson 95% **upper bound ≤ α + 0.03** per bucket, and pooled **≤ 0.05** | n per bucket reported; with n ≈ 100 the CI is wide, which is why the UCB slack is explicit rather than pretended away |
| E-2 | Must-escalate recall (unanswerable ∪ PII ∪ injection ∪ out-of-scope) | **≥ 0.95** | The safety-critical direction |
| E-3 | Escalation precision (escalated turns that were genuinely unsafe/unanswerable) | **≥ 0.55** | Over-escalation is survivable but must be bounded, otherwise the bot is decoration |
| E-4 | Auto-answer coverage on answerable items | **≥ 0.60** | The other side of the trade-off; reported with E-1 in one row so neither can be quoted alone |
| E-5 | Calibration error of `p̂` | ECE **≤ 0.08** (10 equal-width bins) **and** Brier **≤ 0.15**; equal-mass ECE reported alongside | ECE is binning-dependent (`[A]`), hence the dual metric and the reliability plot |
| E-6 | AURC (risk–coverage) | reported with its curve; **no threshold** — it is a diagnostic, not a gate | Prevent the trap of tuning the number instead of the decision |
| E-7 | Coverage stability under simulated drift | after injecting 10% new documents + a locale-mix shift, realised risk stays ≤ α + 0.05; if not, the monitor must fire | Evidence for Q11; drift simulation is part of the release artifacts |

### 8.6 Monitoring and refresh (operational contract)

* The calibration artifact has a date and a split hash; it is **refreshed at least weekly** in a live deployment and always after an ingest that changes the corpus version materially.
* The service computes realised coverage and realised risk over rolling windows and alerts at **−3 coverage points** vs the calibration-time value (FR-ESC-10).
* The console shows current α, achieved coverage, realised risk and the escalation reason distribution (D-14).
* A stale artifact (age > configured TTL, or corpus version mismatch beyond the configured threshold) must cause a startup warning and a visible console banner; it must not silently answer as if calibrated.

---

## 9. Semantic cache contract

### 9.1 What a cache entry is

```
CacheEntry
  key: { normalized_query, locale, entity_set, corpus_version, index_version,
         prompt_version, model_id, policy_artifact_hash }
  value: the whole Answer artifact (§7.1) + retrieval fingerprint (§4.2 trace) + created_at + ttl_class
  stats: hits, wrong_hits, last_hit_at, cohort_id
```

Caching a bare answer string is forbidden: a cached answer without its citations, versions and fingerprint is a stale-answer generator (`[A]`).

### 9.2 Guards (all mandatory, in this order)

| Guard | Rule | Failure mode it prevents |
|---|---|---|
| G-1 version guard | corpus/index/prompt/model/policy hashes must all equal the live values | serving an answer produced by a different system |
| G-2 entity guard | extracted entities (order ID, amount, date, plan, locale) must match exactly | "your refund is 30 days" answering a question about a different plan |
| G-3 source-doc guard | every cited chunk must still exist with an unchanged `doc_hash` | citing a deleted or edited document |
| G-4 structural re-validation | citations resolve, quotes match, numeric guard (INV-04) re-run at hit time | any silent stale-citation path |
| G-5 risk-tier guard | high-risk intents never hit the semantic path (exact only) | confidently wrong answers on money/account questions |

G-4 is cheap (string and set operations, no model) and must never be skipped for latency: the whole point is that a wrong hit is the worst failure mode in the product.

### 9.3 Threshold calibration

Similarity thresholds are **per cohort** (intent bucket × locale family × question-length band), not global. For each cohort, choose the smallest threshold `τ` such that, on the calibration replay set, the measured wrong-hit rate satisfies the budget with its Wilson upper bound inside the slack: `wrong_hits / hits ≤ ε` and `UCB_95 ≤ 2ε` with `ε = 0.01`.

Why per-entry/per-cohort rather than one static cosine threshold: static thresholds give no correctness guarantee and produce unexpected error rates; per-entry online-learned thresholds meet a specified error bound with much higher hit rate (`[C]` vCache, ICLR 2026, arXiv:2502.03771). We adopt the *error-budget* discipline; the online-learning variant is a v1.1 experiment.

### 9.4 Invalidation (explicit, because no paper settles it — `[A]`, Q5)

| Trigger | Effect |
|---|---|
| Any ingest or delete | corpus version bump → **all** entries invalid (conservative, one-line implementation, zero stale-citation risk) |
| Single document changed/deleted | entries citing that document invalidated immediately, before the version bump propagates |
| Prompt or model change | entries invalidated by the version guard (§9.2 G-1) |
| Policy/calibration artifact change | entries invalidated by the policy hash in the key |
| TTL | per content-volatility class: policy/price/legal pages **short** (default 1 h), how-to/FAQ **long** (default 7 d), escalations **600 s** |
| Kill switch | `ANSWERDESK_CACHE_ENABLED=false` or `--no-cache` bypasses everything, including the exact cache |

### 9.5 Wrong-hit definition, measurement and budget (gate: `AT-CAC-07`, G-REL)

**Wrong hit** = a cache hit returned for a query `q_new` whose gold answer differs from the cached answer's gold answer, on the **near-miss replay set**: pairs `(q_prev, q_new)` that are embedding-similar but differ in answer-relevant detail (different plan, amount, date, order state, locale). The set is built adversarially (≥ 120 pairs, ≥ 20 per locale) because random pairs are too easy and would make the budget vacuous.

| # | Metric | Threshold |
|---|---|---|
| K-1 | Wrong-hit rate on the near-miss replay set | **≤ 0.01**, Wilson 95% UCB **≤ 0.02** |
| K-2 | Hit rate, informational intents | **≥ 0.15** (floor), reported per intent and per locale; no gate on a specific locale |
| K-3 | Semantic hits on high-risk intents | **0** (structural) |
| K-4 | Stale hits after invalidation | **0** (structural; a stale hit is a defect, not a metric) |
| K-5 | Cost of a hit | ≤ **$0.0002** (embedding + lookup only) |
| K-6 | Latency of a hit | p95 ≤ **150 ms** (NFR-LAT-04) |

K-2 exists so that K-1 cannot be satisfied trivially by turning the cache off; K-3/K-4 exist so that K-2 cannot be satisfied by being reckless. The three together are the contract.

### 9.6 Observability

`/metrics` exposes `cache_hits_total`, `cache_misses_total`, `cache_bypass_total`, `cache_wrong_hits_total` (from the replay harness and, in live mode, from feedback-labelled hits), `cache_hit_latency_seconds`; the evidence run emits a hit-rate-vs-error-rate curve (vCache-style) as an artifact.

---

## 10. Evaluation protocol

### 10.1 Dataset composition (`golden/v1`)

| Property | v1.0 requirement |
|---|---|
| Locales | **ru, tg, uz, ky, tr, zh** — every locale present in every split |
| Minimum size | **≥ 50 items per locale in test** (≥ 300 total); dev split ≥ 30 per locale |
| Unanswerable / out-of-scope | **15–20%** of test items (target 18%); below 15% makes calibration meaningless |
| Category quotas (test, per locale) | direct look-up **35%**, numeric/date **20%**, typo or code-switch **15%**, unanswerable **18%**, PII-bearing **6%**, prompt-injection **6%** |
| Separate label classes | retrieval labels (gold chunk ids) vs answer labels (reference answer, required citation ids, answerability, risk tier) — never merged |
| Provenance | every item records how it was produced (document-derived, LLM-drafted+human-verified, perturbation-injected) and which document/seed |
| Additional sets | near-miss cache pairs ≥ 120; claim-level verifier set ≥ 200 claims; injection set ≥ 40; PII set ≥ 40; perturbation set ≥ 120; drift-simulation set (§8.5 E-7) |
| Public data | only with a verified permissive license recorded in the manifest; the adapter refuses without a `license_ok` record (FR-ING-09) |
| Synthetic generation | seeded and deterministic; the generator lives in the repo; the checked-in set is the artifact of the generator, not a hand-edited file |

### 10.2 Splits and leakage

* Split unit is the **document**, not the query: test items may not derive from documents whose chunks are in dev. Otherwise a paraphrase of the same paragraph lands in both splits and every number inflates.
* The split manifest is frozen (`evals/splits/v1.json`), hashed, and printed in every report header.
* The harness **fails loudly** on: a missing label, a gold chunk id not present in the corpus, a test item whose document appears in dev, an item with no category, an item in no locale bucket.
* Ratios: dev **60%** / test **40%** by document; the dev split is the only place fusion parameters, α, cache thresholds and judge calibration may be fit (INV-09).

### 10.3 Labeling protocol

1. **Bootstrap from documents** — sample chunks, draft candidate Q/A pairs with a strong model, then have a human verify, correct or reject (the ARES/PPI pattern `[C]`).
2. **Human verification is mandatory for every answer label**; machine-only labels are marked `machine` and are excluded from any gate (they may appear as diagnostics).
3. **Two annotators on a 200-item subset** (≥ 30 items per locale), independent, then adjudication; publish Cohen's κ per language and overall. Target κ ≥ 0.70 pre-adjudication; the actual value is reported even when it is bad.
4. **Ambiguity is preserved, not resolved by fiat**: items where the annotators disagree and adjudication cannot settle get `ambiguous: true` and are excluded from the primary metric with their count reported.
5. **Retrieval labels are chunk-id sets produced by marking all sufficient passages**, not by marking "the best" one — otherwise `recall_all` is meaningless.
6. **Label protocol is documented** in `docs/eval-methodology.md` with the annotator instructions verbatim.

### 10.4 Metrics inventory

| Layer | Metric | Definition source | Reported with |
|---|---|---|---|
| Retrieval | `recall_any@k`, `recall_all@k`, `nDCG@k`, `MRR@k` | §6.1 formulas | n, 95% CI, per locale, per category |
| Retrieval diagnostics | best rank of a gold chunk, candidate count, dedupe removals | §4.2 trace | distribution, not just mean |
| Answer | groundedness (claim entailment rate), citation precision, citation recall | §7.5, RAGAS/ALCE families `[C]` | n, CI, κ of the judge |
| Answer | rubric score (1–5, pass ≥ 4), answer relevance | rubric version | n, CI, per locale |
| Escalation | realised risk, coverage, must-escalate recall, escalation precision, AURC | §8.5 | n, CI, per bucket |
| Calibration | ECE (equal-width and equal-mass), Brier, reliability plot | Guo et al. 2017 `[C]`, caveat `[A]` | plot + numbers |
| Operational | p50/p95 latency per stage, USD/answer, token counts, cache hit rate | §5 NFRs | distribution, hardware profile |
| Robustness | perturbation delta, drift-simulation risk, injection/PII detection rates | §6.4 RT-9, §8.5 E-7, FR-ESC-08/09 | per class |

### 10.5 Judge protocol and bias controls

| Control | Rule |
|---|---|
| Judge ≠ generator | the generating model may never judge its own family's output in a reported metric (`[C]` self-enhancement bias) |
| Rubric anchoring | rubric references the retrieved context and the claim, never answer length or style (`[C]` verbosity bias) |
| Position swap | pairwise comparisons are run in both orders and averaged (`[C]` position bias); a fixture test asserts order invariance within tolerance |
| Pairwise over Likert | prefer pairwise/forced-choice to 1–10 scales where feasible (`[C]`) |
| Human calibration | ≥ 100 human-labelled items per language (target 150); κ published per language; judged metrics gated on κ ≥ 0.60 (§7.5 F-10) |
| Versioning | judge model, prompt version and rubric version recorded next to every judged number |
| Determinism | `FakeJudge` reproduces fixture scores exactly for CI; a judge change is a visible report diff, never a silent drift |

### 10.6 Evidence and ablation plan (`answerdesk eval --all`)

| Arm | Variants | Purpose |
|---|---|---|
| Retrieval | bm25 · dense · hybrid-rrf (k sweep) · hybrid-cc (α sweep) · +rerank · +mmr | Q1, Q7, D-05; proves RT-7 |
| Chunking | structure-first · naive fixed-window · contextual headers on/off (SHOULD) | Q3, D-01/D-02 |
| Packing | k ∈ {5, 8, 12, 20} × ordering (rank order vs best-first-and-last) | Q8, D-06 |
| Policy | fixed threshold · conformal · conformal+cost-sensitive | D-09; proves the calibration claim |
| Cache | off · exact only · semantic with guards (ε sweep) | Q5, D-10; produces K-1/K-2 curves |
| Escalation/drift | calibration refresh cadence under simulated drift | Q11 |
| Language | per-locale metrics for everything above | NFR-ML-01 |

Every arm reports point estimate, `n`, 95% CI, and for the key pairs a McNemar/paired-bootstrap test. The report is the evidence; prose claims are downstream of it.

### 10.7 Limitations — the honest-reporting policy

These limitations must appear verbatim in `docs/results.md` and `docs/eval-methodology.md` (§10.7 is normative, not advisory):

1. **Synthetic data optimism.** Queries derive from the corpus and are paraphrased from it; recall and groundedness are therefore *optimistic* relative to real support traffic. No number in this repository transfers to a production corpus without re-measurement.
2. **Small n.** n = 300 test items gives roughly ±3 points on the macro retrieval metric and much wider CIs on per-locale and subset metrics. Every cell carries n and CI; a "win" that does not clear the CI is reported as noise.
3. **Judge bias.** Judged metrics are LLM-mediated; low-resource languages are likely worse. κ per language is published; where κ < 0.60 the metric is marked "not reportable" rather than softened with adjectives.
4. **Benchmark ≠ support domain.** MTEB leaderboard values are vendor-curated snapshots (`[S]`, unverified U3); embedder choice is justified only by our own ablation.
5. **Conformal exchangeability.** The risk guarantee holds under exchangeability; v1.0 has no real traffic, so drift is simulated, not observed. The guarantee is a statement about the calibration distribution, not a promise about the future.
6. **Environment dependence.** Latency and cost numbers are for the recorded reference machine and provider pricing at `retrieved_at`; they are not portable claims.
7. **No production evidence.** No SLA, no incident history, no live traffic, no real handoff-cost measurement (Q4). The cost ratios are documented assumptions.
8. **Cache results are replay-based.** K-1/K-2 are measured on adversarial replay pairs, not on live traffic; the hit rate in production will differ in both directions.
9. **Replay mode is labelled.** Any number produced without a live model/provider carries `REPLAY-MODE`; it may never be presented as a live result (NFR-REP-01).
10. **No claim of SOTA.** The repository claims reproducibility, calibration artifacts and citation verification — not state-of-the-art retrieval or generation.

### 10.8 Reproducibility contract for results

```text
one command      → answerdesk eval --all --report reports/bench
artifacts        → reports/bench/{retrieval,answer,escalation,cache,ops}/*.json|csv|png + report.md
header           → corpus_hash, golden_hash, split_hash, seed, judge model+prompt version,
                   provider mode (live|replay), hardware profile, pricing table id+date
determinism      → same seed + same split → identical metrics JSON after removing timing fields
reproduction doc → docs/reproducibility.md with OS, Python, uv version, extras installed,
                   expected durations, and the exact commands
```

---

## 11. Security, privacy and grounding invariants

### 11.1 Threat model (v1.0)

| Threat | Vector | Why it matters here | Enforcement |
|---|---|---|---|
| Indirect prompt injection | instructions planted in an ingested document or a retrieved passage | The defining RAG threat (`[C]` arXiv:2302.12173; OWASP LLM01). A KB is attacker-writable in most real deployments. | INV-01, INV-02, FR-GEN-11, FR-ESC-09, NFR-SEC-01; **architectural** separation (no tool calls, no routing from content) rather than prompt filtering alone (`[C]` CaMeL result: provable security costs 77% vs 84% task success — we accept that trade) |
| Citation fabrication | model invents a chunk id, quote or number | Citations are the product's trust surface; display-only citations are the OSS norm (`[C]`) | INV-02..04, CR-1–CR-9, deterministic checks with no model in the loop |
| Stale citation | document edited/deleted after caching | A wrong citation is worse than no citation (`[A]`) | FR-CAC-03, FR-CAC-05, CR-6, `doc_hash` |
| Cache poisoning | low-quality or adversarial answer cached and replayed | A wrong hit is the worst user-visible failure in the product | FR-CAC-02..06, near-miss replay budget (§9.5) |
| PII leakage | ticket text into index, logs, traces, provider | Support tickets are the highest-PII corpus in most companies (`[A]`) | INV-05, FR-ESC-08, NFR-SEC-02 (three boundaries, mock-capture assertion) |
| Secret leakage | keys in repo, logs, error messages | C-06, §13.5; an OSS repo is public by definition | NFR-SEC-03, FR-OPS-05 |
| XSS / script execution in the console | model or document text rendered as HTML | The console renders untrusted text by design | FR-UI-10, NFR-SEC-04 |
| Dependency compromise / license contamination | transitive dependencies, vendored code | MIT repo, public release | NFR-REP-04, §13.3 allowlist, NFR-SEC-05 |
| Vector-store weaknesses | embeddings as an attack surface (OWASP LLM08 `[C]`) | Retrieval is a security boundary | INV-01, index provenance, corpus-version pinning |

### 11.2 Invariant enforcement map

| INV | Primary enforcement | Verified by |
|---|---|---|
| INV-01 control/data separation | architecture: no tool layer in v1.0; decision path consumes only `p̂`, guards and config | CL-SEC-01 review of the decision path + AT-SEC-01 fixture suite |
| INV-02 citations resolve | structural validation before render | AT-GEN-02, AT-CAC-03 |
| INV-03 no unsupported claim displayed | verifier + drop/escalate | AT-GEN-07, AT-GEN-08 |
| INV-04 numeric guard | deterministic number/date extraction and containment check | AT-GEN-12 |
| INV-05 PII redaction | guard module called at the three boundaries | AT-SEC-02, AT-ESC-08 |
| INV-06 no auto-answer when unanswerable | conformal + guards precedence | AT-ESC-02/04/07 |
| INV-07 single provider seam | one HTTP client in `llm/providers.py`; import-lint rule | AT-PORT-01 + CL-PORT-01 |
| INV-08 deterministic offline CI | profiles, fakes, no network in default paths | AT-DET-01 |
| INV-09 dev/test discipline | fitting code reads only the dev manifest; leakage detector | AT-RET-05, AT-EVL-03 |
| INV-10 artifact-or-it-didn't-happen | report headers, traceability checker, review | CL-REP-01, AT-REP-01 |

---

## 12. Non-goals (v1.0)

Explicit exclusions. Each states the exclusion, why, and what exists instead — silence is how scope creep wins.

| ID | Non-goal | Why | What we do instead |
|---|---|---|---|
| NG-01 | **No voice / telephony / IVR / speech-to-text** | Out of scope for a text RAG reference stack; audio adds an entire evaluation surface we cannot validate honestly | Text chat only; the API accepts text |
| NG-02 | **No live deployment, hosting, managed service, uptime SLA, or production data** | C-06; publishing a reference stack is the deliverable, not operating one | Tagged release candidate + quickstart + `docs/ops.md` runbook |
| NG-03 | **No employer, bank, client or proprietary data, code, schema or prompt** | C-03; legal and ethical red line | Synthetic multilingual corpus + optional public dataset adapter with license gate |
| NG-04 | **No real customer PII, ever** | Same as NG-03; also makes logs safe to publish | Synthetic PII in reserved ranges only (§13.4) |
| NG-05 | **No autonomous actions on customer accounts** (no refunds, no account changes, no order mutations) | Any action needs a trust and liability model this project cannot test; retrieved content must never drive actions (INV-01) | Answers and escalation only; no tool calls in v1.0 |
| NG-06 | **No fine-tuning** of embedders, rerankers or generators | `[A]`/out of research scope; fine-tuning would invalidate the reproducibility story and need data we do not have | Prompt + retrieval + calibration improvements only; fine-tuning is a documented v2 spike |
| NG-07 | **No multi-tenant SaaS, SSO, SCIM, RBAC beyond a single admin bearer token** | Enterprise identity is a product of its own; the interesting engineering is elsewhere | Single-tenant local service; token-protected admin endpoints |
| NG-08 | **No channel connectors** (Zendesk/Intercom/Slack/WhatsApp/Telegram sync) | Connector maintenance is where OSS RAG projects go to die (`[C]` Onyx's burden) | One HTTP API + one console; adapters are a reader exercise |
| NG-09 | **No mobile app, no desktop app** | Console is the operator surface; customers use the API | Responsive web console |
| NG-10 | **No Docker/Kubernetes requirement** | C-01: local-first on Windows; Docker is a barrier for the target reader | `uv sync` + `uv run`; Dockerfile may exist as an optional extra but must never be required |
| NG-11 | **No agentic multi-step retrieval loops / deep-research mode** | RAGFlow-style tiered agents noted as a trend, not evaluated (`research` out-of-scope); unbounded agent loops conflict with the latency and cost gates | Single-pass retrieve → rerank → generate → verify; deterministic and measurable |
| NG-12 | **No default machine translation of queries or documents** | Translation destroys identifiers (`[A]`, D-07) | Per-language corpus + multilingual embedder + language-matched answers |
| NG-13 | **No claim of state-of-the-art retrieval or generation** | Honesty policy (§10.7); we have no competitive benchmark on real traffic | Claims are limited to reproducibility, calibration artifacts and verified citations (§10.7 item 10) |
| NG-14 | **No graph RAG / knowledge-graph extraction** | Cost and evaluation burden; no evidence it beats hybrid on support FAQs | Structure-first chunks with heading paths |
| NG-15 | **No real-time streaming ingestion from external sources** | Operationally unverifiable here | Explicit ingest/refresh command and admin endpoint with version bump |
| NG-16 | **No A/B testing framework or feature-flag service** | Scope; the eval harness is the comparison mechanism | Offline ablations (§10.6) |

---

## 13. Data, licensing and provenance constraints

### 13.1 Code license

* **MIT** for the whole repository (C-05). `LICENSE` present, copyright line filled, no per-file headers required.
* Third-party code is **imported as a dependency**, not vendored, except where a small, clearly attributed snippet is necessary; any such snippet carries its origin and license in a comment and in `NOTICE`.
* **Forbidden**: copying or forking code from Dify ("Apache-2.0 + additional conditions") or FastGPT (no-SaaS clause), and building on archived/dormant projects (Verba archived 2026-06-08; R2R dormant since 2025-11-07) `[C]`. Re-implementation from published formulas and papers is explicitly allowed and preferred.

### 13.2 Data provenance and locale corpora

| Dataset class | Rule |
|---|---|
| Synthetic corpus (primary) | Generated by the seeded generator in `src/answerdesk/ingest/synthetic.py`; dedicated to the public domain / MIT-compatible (CC0-1.0 preferred); manifest records generator version + seed |
| Public dataset adapters (optional) | Only with a license that permits use **and** redistribution; the adapter refuses to run without a recorded `license_ok` + license identifier + URL + retrieval date (FR-ING-09); data that cannot be redistributed is downloaded at runtime and **git-ignored** |
| Golden sets | Derived from the corpus and human-verified; published as data under the repo license so others can re-run the benchmark |
| Documentation assets | Screenshots generated from synthetic data only; fonts/icons under OFL/Apache-2.0/MIT with attribution |
| Anything else | Not allowed |

### 13.3 Runtime dependency allowlist

| Allowed | Notes |
|---|---|
| MIT, BSD-2-Clause, BSD-3-Clause, ISC, Apache-2.0, PSF-2.0 | Apache-2.0 requires `NOTICE` attribution |
| MPL-2.0 | Only unmodified dependencies, file-level copyleft respected |
| CC0-1.0 / public domain (data, fonts) | Preferred for data |
| **Forbidden** | GPL-2.0/3.0, AGPL-3.0, SSPL, BUSL, "Apache + additional conditions", no-SaaS clauses, unknown/proprietary |

Enforcement: `NFR-REP-04` license audit in the release gate; dependency additions require an ADR line naming the license.

### 13.4 Synthetic identifiers: reserved ranges only

| Kind | Allowed values |
|---|---|
| IPv4 | RFC 5737 documentation ranges: `192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24` |
| Domains | `example.com`, `example.org`, `example.net` and subdomains |
| Emails | `*@example.com` style only |
| Phones | Reserved/fictional ranges per locale (e.g. `+1 555-01xx`, `+7 000 000-00-00` style) documented in the generator |
| Card numbers | Documented test PANs only (Luhn-valid, never issued) |
| Names/addresses | Fictional, no real institution names; no employer or bank brand |

A CI test scans fixtures and generated corpora for identifiers outside these ranges (AT-ING-10, AT-SEC-02).

### 13.5 Secret handling

`.env` is git-ignored; `.env.example` carries placeholders only; no key, token or endpoint with embedded credentials is committed, logged or echoed in error envelopes. CI runs a secret scan (AT-SEC-03).

---

## 14. Risk register

Likelihood (L) and impact (I) are H/M/L. "Residual" is the risk after the listed mitigation; rows with residual H are release blockers by definition.

| ID | Risk | L | I | Mitigation (requirement) | Residual | Trigger to act |
|---|---|---|---|---|---|---|
| R-01 | **Synthetic-data optimism**: published numbers look great and do not transfer to real support traffic | H | H | Limitations stated in every results table (§10.7); perturbation + near-miss sets; no SOTA claim (NG-13) | M | Any external re-measurement contradicting a claim |
| R-02 | **Judge bias** inflates groundedness, especially tg/ky/uz | H | H | Judge ≠ generator, position swap, context-anchored rubric, per-language κ, "not reportable" rule (§10.5, F-10) | M | κ < 0.60 in any locale |
| R-03 | **Conformal assumption breaks** under drift → silent coverage loss | M | H | Weekly refresh, coverage monitor with −3 pt alert, drift simulation in the release evidence (E-7), stale-artifact banner (§8.6) | M | Monitor fires; corpus version jumps |
| R-04 | **Cache wrong hit** answers a near-miss question confidently | M | H | Error budget ε=0.01 with UCB ≤ 0.02, entity + doc-hash guards, high-risk bypass, kill switch (§9) | L–M | Any wrong hit in the replay harness blocks release |
| R-05 | **License contamination** (copied Dify/FastGPT code, GPL dep) | M | H | Clean-room implementation rule, dependency allowlist, license audit (NFR-REP-04, §13.3) | L | Any new dependency without a license ADR |
| R-06 | **Dependency mortality/churn** (project archived mid-build) | M | M | No forks of dying projects; dependency-light base; pins in `uv.lock` (§13.1) | L | Upstream archived/stale > 12 months |
| R-07 | **Local-model latency/cost blow-up** (CPU rerank, 20k chunks) | H | M | Rerank behind `local` extra, configurable budget, ONNX/int8 path noted, scale gate NFR-SCALE-01 | M | p95 above NFR-LAT-02 at scale |
| R-08 | **Over-escalation** makes the product useless ("always handoff") | M | H | E-3 precision floor, E-4 coverage floor, α from cost model, risk–coverage artifact (§8.5) | M | Coverage < 0.60 |
| R-09 | **Verifier false rejections** suppress good answers | M | M | False-rejection budget F-9, claim-level labelled set, repair-before-drop (§7.3) | M | False rejection > 10% |
| R-10 | **Prompt injection through the KB** steers answers or the UI | M | H | INV-01 architectural separation, injection fixtures in CI/release, guard-forced escalation (§11) | L–M | Any fixture auto-answered with attacker intent |
| R-11 | **PII leak** into logs, traces, provider payloads, or the published repo | M | H | Three-boundary redaction, mock-capture assertion, synthetic-only data, secret scan (§11) | L | Any fixture PII found in a trace |
| R-12 | **Scope creep** (connectors, voice, agents, mobile) | H | M | Non-goals §12; traceability matrix; every new ask becomes a requirement amendment or a non-goal | M | New work with no FR ID |
| R-13 | **Environment variance** (Windows vs Linux, CPU vs GPU) makes latency/cost numbers irreproducible | M | M | Reference machine recorded, hardware profile in report header, CI matrix, offline gate (NFR-PORT-02, NFR-DET-01) | L | Report lacking a hardware profile |
| R-14 | **Metric gaming / underpowered comparison** presented as a win | M | H | Pre-registered primary metrics, n + CI everywhere, McNemar for binary pairs, no bare point estimates (§10.4, FR-EVL-07) | L | Any table cell without n/CI |
| R-15 | **Golden-set leakage** inflates every number | M | H | Document-disjoint splits, hashed manifests, hard failure on leakage (FR-EVL-03) | L | Leakage detector trips |
| R-16 | **Pricing/model drift** invalidates cost numbers | H | L | Pinned pricing table with `retrieved_at`, cost recorded per call, replay labelling (NFR-COST-02) | L | Pricing table older than 90 days at release |
| R-17 | **Interface drift** between modules (contracts churn) breaks parallel work | M | M | `contracts.py` frozen in t3, §7.1 answer shape, schema tests per seam | L | Any module bypassing contracts |
| R-18 | **Security review fatigue**: guards exist but are not exercised | M | M | Guard fixture matrix, reason-code reachability test, reviewer mutation spot-check (AT-ESC-04/05, NFR-QUAL-03) | L | Guard branch with no test |
| R-19 | **Documentation drift**: docs claim what code no longer does | M | M | CL-DOC-01 executes every README command; traceability checker; reviewer audit (t13) | L–M | Any documented command failing |
| R-20 | **Timeline compression** pushes honesty shortcuts (unlabelled replay numbers, missing κ) | M | H | INV-10 + CL-REP-01 + explicit "no claim" rule; a delayed release is acceptable, a false claim is not | M | Any number without an artifact path |

---

## 15. Consistency, precedence and conflict resolution

### 15.1 Precedence order (total, applies to every future change)

```
1. Invariants (§3)            — security, PII, grounding, determinism-of-CI
2. Risk control               — α, guards, no-auto-answer-when-unsafe
3. Measurement honesty        — no unproven claim, n+CI, artifact-or-it-didn't-happen
4. Functional correctness     — answers correct, citations resolve, contracts honoured
5. Cost                       — USD/answer
6. Latency                    — p50/p95
7. Coverage / hit-rate / UI richness
```

A lower-ranked requirement may never be satisfied by weakening a higher-ranked one. Concretely: never disable the verifier, the PII guard, the numeric guard or the cache guards to win latency; never inflate a number to win a release.

### 15.2 Tensions and their pre-agreed resolutions

| # | Tension | Resolution rule |
|---|---|---|
| T-01 | Recall (more chunks, more candidates) vs latency | Rerank window and packed-k are the only knobs: reduce rerank input from 50 → 30 and packing from 12 → 8 in that order, documented per release. Never skip verification. |
| T-02 | Groundedness vs cost | Trim context and switch to the cheaper model profile **only** while F-1..F-6 still pass; a cost win that breaks grounding is a regression. |
| T-03 | Escalation recall vs coverage | Recall wins (E-2 > E-4): the product promise is not to be confidently wrong. If coverage collapses below 0.60, the fix is retrieval/generation quality, not a looser α. |
| T-04 | Cache hit rate vs wrong hits | Wrong-hit budget wins (K-1 gates K-2). Raising the hit rate by loosening guards is a defect. |
| T-05 | Verifier strictness vs usefulness | False-rejection budget F-9 bounds strictness; improve the verifier rather than remove the check. |
| T-06 | Macro average vs per-locale parity | Both are reported; a locale below the RT-2 floor fails the gate even if the macro passes. |
| T-07 | Real-model quality vs deterministic CI | They are separate gate classes (§0.3). CI proves machinery; release proves quality. Neither substitutes for the other. |
| T-08 | Console richness vs schedule | FR-UI-01..11 are the console's contract; anything beyond (KB editing, prompt editor, multi-agent dashboards) is v2 and may not delay t10. |
| T-09 | Conformal α vs cost-derived threshold | `α_final = min(α_bucket, C_abstain/C_wrong)`; the stricter wins; the derivation is recorded in `docs/escalation.md`. |
| T-10 | Golden-set size vs CI runtime | Full set in the release run; a fixed smoke subset (≤ 60 items) in CI, always labelled as a subset; metrics from the subset never appear in `results.md`. |
| T-11 | No-translation rule vs low-resource coverage | Multilingual embedding + per-language corpus instead of translation; if a locale's retrieval is degraded, it is reported degraded (RT-2 falsification clause), not translated. |
| T-12 | Scale (20k chunks) vs CPU latency | Budget splits: BM25 + exact dense is fast; rerank is bounded (top-30 at scale); cache absorbs repeats. NFR-LAT-02 is the arbiter. |

### 15.3 Cross-requirement dependencies (checked, not assumed)

| Requirement | Depends on | Why it must be checked |
|---|---|---|
| FR-GEN-08 (groundedness) | FR-RET-10 (retrieval targets) | The generator cannot ground claims in evidence retrieval never surfaced; a grounding failure with recall@20 < 0.90 is a retrieval bug. |
| FR-ESC-07 (risk) | FR-GEN-08, FR-ESC-02 | "Correct" is defined via grounding and citations (§8.1); a miscalibrated scorer and a broken verifier produce the same symptom. |
| FR-CAC-07 (wrong-hit) | FR-ESC-07, FR-CAC-03 | A cache hit's answer must satisfy the same correctness definition, including citations. |
| NFR-COST-02 | FR-CAC-02..04, FR-GEN-05 | Cost is dominated by context size and cache hits; both are specified elsewhere. |
| NFR-LAT-01/02 | FR-RET-07, FR-GEN-05 | Rerank and packing are the latency hot spots; their budgets are the latency budget. |
| NFR-DET-02 | FR-EVL-02/03, all seeded generators | Determinism requires every stochastic step to be seeded; one unseeded sampler breaks the whole claim. |
| FR-UI-04/05 | FR-ESC-05, FR-CAC-04, FR-RET-08 | The console can only show the decision if the trace and enum carry it. |

### 15.4 Single-source-of-truth register for every numeric target

Thresholds appear **once** in this document. This register is the index; any duplicate with a different value is a defect found by the traceability checker (CMD-T).

| Value owner | Section | Values |
|---|---|---|
| Retrieval targets | §6.4 | RT-1..RT-10 |
| Generation/citation targets | §7.5 | F-1..F-10 |
| Escalation targets | §8.5 | E-1..E-7 (plus α and cost defaults in §8.2/§8.3) |
| Cache targets | §9.5 | K-1..K-6 |
| Latency/cost/scale | §5 | NFR-LAT-01..04, NFR-COST-01..03, NFR-SCALE-01 |
| Determinism/runtime | §5 | NFR-DET-01..03, FR-OPS-06 |
| Dataset composition | §10.1 | sizes, quotas, split ratios (§10.2) |
| Config defaults | Appendix C | α, costs, k, thresholds, TTLs, budgets |

---

## 16. Traceability matrix

Every functional and non-functional requirement maps to a module (§1.5), an owning task, one acceptance test ID and one verification (a command code from §17.2 or a review checklist item from §17.1). "Gate" is the class from §0.3.

### 16.1 Ingestion (owner t4)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-ING-01 | `ingest/loaders.py` | t4 | AT-ING-01 | CMD-B | G-CI |
| FR-ING-02 | `ingest/chunker.py` | t4 | AT-ING-02 | CMD-B | G-CI |
| FR-ING-03 | `ingest/chunker.py` | t4 | AT-ING-03 | CMD-B, CMD-C | G-CI |
| FR-ING-04 | `ingest/chunker.py` | t4 | AT-ING-04 | CMD-B, CMD-C | G-CI |
| FR-ING-05 | `ingest/chunker.py`, `index/bm25_index.py` | t4, t5 | AT-ING-05 | CMD-B, CMD-D | G-CI |
| FR-ING-06 | `ingest/dedup.py` | t4 | AT-ING-06 | CMD-B | G-CI |
| FR-ING-07 | `ingest/`, `data/corpus/` | t4 (demo), t11 (scale) | AT-ING-07 | CMD-C, CMD-R | G-CI + G-REL |
| FR-ING-08 | `ingest/` manifest | t4 | AT-ING-08 | CMD-C | G-CI |
| FR-ING-09 | `ingest/datasets.py` | t4 | AT-ING-09 | CMD-B | G-CI |
| FR-ING-10 | `ingest/synthetic.py` | t4 | AT-ING-10 | CMD-B, CMD-C | G-CI |

### 16.2 Retrieval (owner t5)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-RET-01 | `index/bm25_index.py` | t5 | AT-RET-01 | CMD-D | G-CI |
| FR-RET-02 | `index/vector_index.py` | t5 | AT-RET-02 | CMD-D | G-CI |
| FR-RET-03 | `contracts.py`, both index modules | t5 | AT-RET-03 | CMD-D | G-CI |
| FR-RET-04 | `retrieval/fusion.py` | t5 | AT-RET-04 | CMD-D | G-CI |
| FR-RET-05 | `retrieval/fusion.py`, `configs/retrieval/` | t5 | AT-RET-05 | CMD-D, CMD-F | G-CI |
| FR-RET-06 | `retrieval/query_analysis.py` | t5 | AT-RET-06 | CMD-D | G-CI |
| FR-RET-07 | `rerank/cross_encoder.py` | t5 | AT-RET-07 | CMD-D | G-CI (+G-REL for RT-10) |
| FR-RET-08 | `retrieval/pipeline.py` | t5 | AT-RET-08 | CMD-D | G-CI |
| FR-RET-09 | `retrieval/query_analysis.py` | t5 | AT-RET-09 | CMD-D | G-CI |
| FR-RET-10 | `retrieval/`, `eval/` | t5, t11 | AT-RET-10 | CMD-F, CMD-R | G-REL (+G-CI: AT-EVL-04) |

### 16.3 Generation and citations (owner t6)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-GEN-01 | `generate/pipeline.py` | t6 | AT-GEN-01 | CMD-G | G-CI |
| FR-GEN-02 | `generate/citations.py` | t6 | AT-GEN-02 | CMD-G, CMD-H | G-CI |
| FR-GEN-03 | `generate/citations.py` | t6 | AT-GEN-03 | CMD-G | G-CI |
| FR-GEN-04 | `generate/prompts.py`, `configs/prompts/` | t6 | AT-GEN-04 | CMD-G | G-CI |
| FR-GEN-05 | `generate/pipeline.py` (packer) | t6 | AT-GEN-05 | CMD-G | G-CI |
| FR-GEN-06 | `generate/`, `llm/` | t6, t11 | AT-GEN-06 | CMD-G, CMD-R | G-CI + G-REL |
| FR-GEN-07 | `generate/citations.py` (verifier) | t6 | AT-GEN-07 | CMD-G | G-CI (+G-REL F-9) |
| FR-GEN-08 | `generate/`, `eval/metrics.py` | t6, t7, t11 | AT-GEN-08 | CMD-R | G-REL |
| FR-GEN-09 | `llm/` fixture provider | t6 | AT-GEN-09 | CMD-G | G-CI |
| FR-GEN-10 | `generate/pipeline.py` | t6 | AT-GEN-10 | CMD-G | G-CI |
| FR-GEN-11 | `generate/prompts.py` | t6, t8 | AT-GEN-11 | CMD-G, CMD-J | G-CI + G-REL |
| FR-GEN-12 | `generate/citations.py` (numeric guard) | t6 | AT-GEN-12 | CMD-G | G-CI + G-REL |

### 16.4 Escalation (owner t8)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-ESC-01 | `policy/scorers.py`, `policy/metrics.py` | t8 | AT-ESC-01 | CMD-J | G-CI |
| FR-ESC-02 | `policy/conformal.py` | t8 | AT-ESC-02 | CMD-J, CMD-K | G-CI |
| FR-ESC-03 | `policy/costs.py` | t8 | AT-ESC-03 | CMD-J | G-CI |
| FR-ESC-04 | `policy/` decision composition | t8 | AT-ESC-04 | CMD-J | G-CI |
| FR-ESC-05 | `contracts.py` (enum), `policy/` | t8, t9 | AT-ESC-05 | CMD-J, CMD-L | G-CI |
| FR-ESC-06 | `policy/`, calibration artifacts | t8 | AT-ESC-06 | CMD-K | G-CI |
| FR-ESC-07 | `policy/`, `eval/calibration.py` | t8, t11 | AT-ESC-07 | CMD-R | G-REL (+AT-EVL-06) |
| FR-ESC-08 | `policy/pii.py` | t8 | AT-ESC-08 | CMD-J | G-CI + G-REL |
| FR-ESC-09 | `policy/` injection heuristics | t8 | AT-ESC-09 | CMD-J | G-CI + G-REL |
| FR-ESC-10 | `policy/`, `telemetry/metrics.py` | t8, t9 | AT-ESC-10 | CMD-J | G-CI |

### 16.5 Semantic cache (owner t8)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-CAC-01 | `cache/semantic_cache.py` | t8 | AT-CAC-01 | CMD-J | G-CI |
| FR-CAC-02 | `cache/semantic_cache.py` | t8 | AT-CAC-02 | CMD-J | G-CI |
| FR-CAC-03 | `cache/`, `generate/citations.py` | t8 | AT-CAC-03 | CMD-J | G-CI |
| FR-CAC-04 | `cache/`, `contracts.py` | t8 | AT-CAC-04 | CMD-J | G-CI |
| FR-CAC-05 | `cache/` invalidation | t8 | AT-CAC-05 | CMD-J | G-CI |
| FR-CAC-06 | `configs/policy/`, `cache/` | t8 | AT-CAC-06 | CMD-J | G-CI |
| FR-CAC-07 | `cache/`, `eval/` | t8, t11 | AT-CAC-07 | CMD-J, CMD-R | G-REL (+G-CI) |
| FR-CAC-08 | `cache/`, `telemetry/` | t8, t9 | AT-CAC-08 | CMD-J, CMD-L | G-CI |
| FR-CAC-09 | `cache/` | t8 | AT-CAC-09 | CMD-J | G-CI |

### 16.6 Service and telemetry (owner t9)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-SVC-01 | `service/routes.py` | t9 | AT-SVC-01 | CMD-L, CMD-M | G-CI |
| FR-SVC-02 | `service/sessions.py` | t9 | AT-SVC-02 | CMD-L | G-CI |
| FR-SVC-03 | `service/` error taxonomy | t9 | AT-SVC-03 | CMD-L | G-CI |
| FR-SVC-04 | `service/routes.py` (admin) | t9 | AT-SVC-04 | CMD-L | G-CI |
| FR-SVC-05 | `service/routes.py` | t9 | AT-SVC-05 | CMD-L | G-CI |
| FR-SVC-06 | `service/` rate limiting | t9 | AT-SVC-06 | CMD-L | G-CI |
| FR-SVC-07 | `service/app.py` | t9 | AT-SVC-07 | CMD-L, CMD-M | G-CI |
| FR-SVC-08 | `telemetry/tracing.py` | t9 | AT-SVC-08 | CMD-L | G-CI |

### 16.7 Operator console (owner t10)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-UI-01 | `web/src/` queue | t10 | AT-UI-01 | CMD-N | G-CI |
| FR-UI-02 | `web/src/` conversation | t10 | AT-UI-02 | CMD-N | G-CI |
| FR-UI-03 | `web/src/` citation panel | t10 | AT-UI-03 | CMD-N | G-CI |
| FR-UI-04 | `web/src/` diagnostics | t10 | AT-UI-04 | CMD-N | G-CI |
| FR-UI-05 | `web/src/` badges | t10 | AT-UI-05 | CMD-N | G-CI |
| FR-UI-06 | `web/src/` actions | t10 | AT-UI-06 | CMD-N | G-CI |
| FR-UI-07 | `web/src/` analytics | t10 | AT-UI-07 | CMD-N + CL-UI-01 | G-CI + G-REL |
| FR-UI-08 | `web/src/` streaming | t10 | AT-UI-08 | CMD-N | G-CI |
| FR-UI-09 | `web/src/` MSW | t10 | AT-UI-09 | CMD-N | G-CI |
| FR-UI-10 | `web/src/` rendering, a11y | t10 | AT-UI-10 | CMD-N + CL-UI-01 | G-CI |
| FR-UI-11 | `web/scripts/screenshots.mjs`, `docs/assets/` | t10 | AT-UI-11 | CMD-O + CL-UI-02 | G-CI |

### 16.7a Gates verified by the web job, not by the offline Python matrix (R-06/R-09/R-10)

Two NFRs have no row in 16.10 because the offline gate cannot exercise a browser. They are
**not** waived: they are owned by the console task and verified by `CMD-N` (`cd web && npm ci &&
npm run build && npm run test`), which runs in the release sequence (§17.3) and in the console's
own CI job.

| Req | Module | Owner | AT | Verify | Status |
|---|---|---|---|---|---|
| NFR-SEC-04 (no XSS through an answer) | `web/src/`, `web/src/test` | t10 | AT-SEC-04 | CMD-N (vitest DOM test) | open: React escapes by default, but no DOM test feeds a `<script>`/`onerror` payload through a mocked `/v1/chat` answer yet |
| NFR-A11Y-01 (automated accessibility check) | `web/src/`, `web/scripts/` | t10 | AT-A11Y-01 | CMD-N (axe-core / vitest-axe) + committed report | open: no `axe` dependency and no committed accessibility report yet |

The offline matrix in 16.10 therefore proves nothing about the two; a reader must not treat a
green `scripts/ci-offline.ps1` as covering them. Both sit inside `answerdesk/web`, which is out of
scope for the service/CLI repair round that added this note.

### 16.8 Evaluation harness (owner t7)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-EVL-01 | `eval/datasets.py` | t7 | AT-EVL-01 | CMD-P | G-CI |
| FR-EVL-02 | `eval/datasets.py`, `evals/` | t7 | AT-EVL-02 | CMD-P, CMD-Q | G-CI |
| FR-EVL-03 | `eval/datasets.py` | t7 | AT-EVL-03 | CMD-P | G-CI |
| FR-EVL-04 | `eval/metrics.py` | t7 | AT-EVL-04 | CMD-P | G-CI |
| FR-EVL-05 | `eval/judge.py` | t7 | AT-EVL-05 | CMD-P, CMD-R | G-CI + G-REL |
| FR-EVL-06 | `eval/calibration.py` | t7 | AT-EVL-06 | CMD-P, CMD-R | G-CI (+G-REL) |
| FR-EVL-07 | `eval/stats.py` | t7 | AT-EVL-07 | CMD-P | G-CI |
| FR-EVL-08 | `eval/report.py` | t7 | AT-EVL-08 | CMD-Q | G-CI |
| FR-EVL-09 | `eval/report.py` | t7 | AT-EVL-09 | CMD-Q | G-CI |
| FR-EVL-10 | `eval/`, `scripts/` | t7, t11 | AT-EVL-10 | CMD-R | G-REL |
| FR-EVL-11 | `docs/eval-methodology.md` | t7 | AT-EVL-11 | CL-EVL-01 | G-REV |

### 16.9 CLI, config, operations (owners t3, t9, t12, t14)

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| FR-OPS-01 | CLI, `pyproject.toml` | t3 | AT-OPS-01 | CMD-A | G-CI |
| FR-OPS-02 | `index/`, CLI | t5 | AT-OPS-02 | CMD-E | G-CI |
| FR-OPS-03 | `llm/presets.py`, `config.py` | t6, t3 | AT-OPS-03 | CMD-I, CMD-G | G-CI + CL-PORT-01 |
| FR-OPS-04 | `config.py`, `configs/` | t3 | AT-OPS-04 | CMD-A, CMD-Q | G-CI |
| FR-OPS-05 | `config.py` | t3 | AT-OPS-05 | CMD-A | G-CI |
| FR-OPS-06 | CLI, `scripts/` | t14 | AT-OPS-06 | CMD-U | G-CI |

### 16.10 Non-functional requirements

| Req | Module | Owner | AT | Verify | Gate |
|---|---|---|---|---|---|
| NFR-LAT-01 | `service/`, `telemetry/` | t9, t11 | AT-LAT-01 | CMD-R | G-REL |
| NFR-LAT-02 | `retrieval/`, `rerank/` | t5, t11 | AT-LAT-02 | CMD-R | G-REL |
| NFR-LAT-03 | `service/` offline path | t9 | AT-LAT-03 | CMD-Q | G-CI |
| NFR-LAT-04 | `cache/` | t8, t11 | AT-LAT-04 | CMD-R | G-REL |
| NFR-COST-01 | `llm/cost.py` | t6 | AT-COST-01 | CMD-G, CMD-L | G-CI |
| NFR-COST-02 | `llm/cost.py`, `configs/` pricing | t6, t11 | AT-COST-02 | CMD-R | G-REL |
| NFR-COST-03 | `telemetry/tracing.py` | t9 | AT-COST-03 | CMD-L | G-CI |
| NFR-DET-01 | `.github/workflows/ci.yml` | t3, t14 | AT-DET-01 | CMD-A + CL-REL-01 | G-CI |
| NFR-DET-02 | `eval/`, seeds across modules | t7, t14 | AT-DET-02 | CMD-Q (twice, hash compare) | G-CI |
| NFR-DET-03 | `.github/workflows/` | t3, t14 | AT-DET-03 | CL-REL-01 | G-CI |
| NFR-PORT-01 | `llm/providers.py`, `llm/presets.py` | t6 | AT-PORT-01 | CMD-G, CMD-I, CL-PORT-01 | G-CI + G-REV |
| NFR-PORT-02 | packaging, CI | t3, t14 | AT-PORT-02 | CMD-A + CL-REL-02 | G-CI |
| NFR-PORT-03 | `pyproject.toml` extras | t3 | AT-PORT-03 | CMD-A | G-CI |
| NFR-SEC-01 | `policy/`, `generate/` | t6, t8 | AT-SEC-01 | CMD-J, CMD-G, CMD-R | G-CI + G-REL |
| NFR-SEC-02 | `policy/pii.py`, `telemetry/`, `llm/` | t8, t9 | AT-SEC-02 | CMD-J, CMD-L | G-CI |
| NFR-SEC-03 | repo hygiene, CI | t3, t14 | AT-SEC-03 | CMD-Y + CL-SEC-01 | G-CI |
| NFR-SEC-04 | `web/` | t10 | AT-SEC-04 | CMD-N | G-CI |
| NFR-SEC-05 | dependencies | t14 | AT-SEC-05 | CMD-Z | G-REL |
| NFR-QUAL-01 | all src | t3, t14 | AT-QUAL-01 | CMD-S | G-CI |
| NFR-QUAL-02 | `retrieval/`, `policy/`, `cache/`, `eval/`, `scripts/ci-offline.ps1` | t5, t7, t8 | AT-QUAL-02 | CMD-W | G-CI |
| NFR-QUAL-03 | tests across modules | t13 | AT-QUAL-03 | CL-QUAL-01 | G-REV |
| NFR-ML-01 | `eval/report.py`, golden set | t7 | AT-ML-01 | CMD-Q + CL-ML-01 | G-CI |
| NFR-ML-02 | `retrieval/query_analysis.py`, `llm/` | t5, t6 | AT-ML-02 | CMD-D, CMD-G | G-CI |
| NFR-REP-01 | `README.md`, `docs/results.md`, `scripts/check-numbers.py` | t12, t13 | AT-REP-01 | CMD-NUM, CL-REP-01 | G-REV + G-CI |
| NFR-REP-02 | manifests, `eval/report.py` | t4, t7 | AT-REP-02 | CMD-C, CMD-Q | G-CI |
| NFR-REP-03 | `scripts/check-traceability.py` | t12 | AT-REP-03 | CMD-T | G-CI |
| NFR-REP-04 | dependencies, `scripts/check-licenses.py`, `LICENSE` | t14 | AT-REP-04 | CMD-Z, CMD-LIC | G-REL |
| NFR-OBS-01 | `telemetry/` | t9 | AT-OBS-01 | CMD-L, CMD-M | G-CI |
| NFR-OBS-02 | `web/`, `service/` | t9, t10 | AT-OBS-02 | CL-UI-01 | G-REL |
| NFR-OPS-01 | `docs/ops.md`, services | t9 | AT-OPS-07 | CL-OPS-01 | G-CI + G-REV |
| NFR-DOC-01 | `docs/`, READMEs | t3, t12 | AT-DOC-01 | CL-DOC-01 | G-REV |
| NFR-DOC-02 | repo-wide | all | AT-DOC-02 | CL-DOC-02 | G-REV |
| NFR-A11Y-01 | `web/` | t10 | AT-A11Y-01 | CMD-N | G-CI |
| NFR-SCALE-01 | `ingest/`, `index/`, `retrieval/` | t4, t5, t11 | AT-SCALE-01 | CMD-R | G-REL |

### 16.11 Owner-task index (convenience view; §16.1–16.10 is authoritative)

Every downstream task can find its contract here. "Primary" means the task must implement it and provide its `AT-` marker; tasks listed as co-owners must satisfy the same requirement without owning the marker.

| Task | Primary requirements | Count | Also responsible for |
|---|---|---|---|
| t3 skeleton | FR-OPS-01, FR-OPS-04, FR-OPS-05, NFR-DET-01, NFR-DET-03, NFR-PORT-02, NFR-PORT-03, NFR-SEC-03, NFR-QUAL-01, NFR-DOC-01 | 10 | contract shapes for every seam (§7.1), CI topology (CL-REL-01) |
| t4 ingestion | FR-ING-01…06, FR-ING-07 (demo half), FR-ING-08…10, NFR-REP-02, NFR-SCALE-01 | 12 | corpus hashes consumed by NFR-REP-02 |
| t5 retrieval | FR-RET-01…10, FR-OPS-02, NFR-LAT-02, NFR-QUAL-02, NFR-ML-02 | 14 | rerank/MMR arm of RT-10; scale measurement with t11 |
| t6 LLM + generation | FR-GEN-01…12, FR-OPS-03, NFR-COST-01, NFR-COST-02, NFR-PORT-01, NFR-SEC-01 | 17 | cost ledger feeding NFR-OBS-01 |
| t7 evaluation | FR-EVL-01…11, NFR-DET-02, NFR-ML-01 | 13 | metric definitions used by every G-REL gate |
| t8 policy + cache | FR-ESC-01…10, FR-CAC-01…09, NFR-LAT-04, NFR-SEC-02 | 21 | calibration artifacts consumed by §8.6 monitoring |
| t9 service | FR-SVC-01…08, NFR-LAT-01, NFR-LAT-03, NFR-COST-03, NFR-OBS-01, NFR-OBS-02, NFR-OPS-01 | 14 | trace schema that FR-UI-04 reads |
| t10 console | FR-UI-01…11, NFR-SEC-04, NFR-A11Y-01 | 13 | real-data analytics (NFR-OBS-02) |
| t11 evidence | — (runs the G-REL gates) | — | RT-1…10, F-1…10, E-1…E-7, K-1…K-6, NFR-SCALE-01, NFR-LAT/COST measurements, all CIs |
| t12 docs/packaging | NFR-REP-01, NFR-REP-03 | 2 | NFR-DOC-01 accuracy, README claims (CL-REP-01) |
| t13 review | NFR-QUAL-03 | 1 | every CL-* item in §17.1, all findings |
| t14 integration | FR-OPS-06, NFR-SEC-05, NFR-REP-04 | 3 | release sequence (CMD-V), clean-checkout quickstart (CL-REL-02) |
| all tasks | NFR-DOC-02 (English repository) | 1 | invariants §3 apply to everyone |

**Matrix completeness.** 87 functional requirements + 34 non-functional requirements = **121 rows**, plus 10 invariants enforced in §11.2. `scripts/check-traceability.py` (CMD-T) asserts that every `AT-` ID above appears as a marker (`# AT-…`) in the repository and that every module path in this section exists — so the matrix cannot silently rot.

---

## 17. Verification index

### 17.1 Review checklist (for the items no command can prove)

| ID | Check | Pass condition |
|---|---|---|
| CL-DOC-01 | Every documented command in `README.md`, `docs/*.md` was executed during the audit | Every command either succeeds or is explicitly marked "optional/live-key required"; zero invented flags |
| CL-DOC-02 | Language policy (C-04) | Spot-check 20 source/doc files: English; only locale corpora, UI strings and `README.ru.md` are non-English |
| CL-EVL-01 | `docs/eval-methodology.md` completeness | Splits, metric formulas (matching code), judge controls, statistics, limitations §10.7 all present |
| CL-PORT-01 | No provider-specific code paths | One HTTP client; presets differ only in defaults; local-server mode needs no key; reviewer reads the provider module end-to-end |
| CL-POL-01 | Thresholds are derived, never literal | No comparison of `p̂` to a numeric literal outside config loading; α/threshold provenance present |
| CL-REP-01 | Every published number has an artifact | Each number in README/`results.md` maps to `reports/<run>/…` and a documented command; replay-mode numbers labelled |
| CL-SEC-01 | Decision path reads no document text | Reviewer traces `decision` inputs: guards, score, config only; no passage text, no model output selecting a route |
| CL-QUAL-01 | Mutation-style spot-check of ≥ 5 modules | Each deliberate breakage causes a test failure; no vacuous assertions; no blanket skips |
| CL-ML-01 | Locale coverage in reports | ru/tg/uz/ky/tr/zh rows present in every metric table; any "not reportable" locale explained by κ |
| CL-DATA-01 | Data provenance and reserved identifiers | Manifest license fields, no real institution names, identifiers within reserved ranges |
| CL-UI-01 | Console against the live service | Screens show real API data; diagnostics match a trace record; no hard-coded metric series |
| CL-UI-02 | Screenshot reproducibility | `npm run screenshots` regenerates the committed assets; ≥ 4 screens |
| CL-OPS-01 | Failure-mode matrix exercised | Each row of the NFR-OPS-01 matrix has a test or a documented manual check |
| CL-REL-01 | CI topology | Fast gate ≤ 3 min, full gate ≤ 10 min, Windows + Linux, no network, no keys, artifact retention |
| CL-REL-02 | Clean-checkout quickstart | Fresh clone, no `.venv`, commands from README succeed; transcript committed |
| CL-HON-01 | Honesty audit | No SOTA claim, no unlabelled replay number, no judged metric without κ, limitations §10.7 verbatim in `results.md` |

### 17.2 Command codes

All commands run from `answerdesk/` unless stated.

| Code | Command | Proves |
|---|---|---|
| CMD-A | `uv run pytest -q` | Offline suite, contracts, CLI smoke, import hygiene (NFR-DET-01) |
| CMD-B | `uv run pytest -q tests/unit/ingest tests/integration/ingest` | Ingestion ATs |
| CMD-C | `uv run answerdesk ingest --demo` then manifest assertion (`docs >= 60`, `chunks >= 1500`) | Corpus build + provenance |
| CMD-D | `uv run pytest -q tests/unit/retrieval tests/unit/index` | Retrieval/index ATs |
| CMD-E | `uv run answerdesk index --demo` | Index build + corpus version |
| CMD-F | `uv run answerdesk eval --retrieval-only` | Retrieval ablation runner |
| CMD-G | `uv run pytest -q tests/unit/llm tests/integration/generation` | Provider + generation ATs |
| CMD-H | `uv run answerdesk ask --fake "Сколько стоит обслуживание карты?"` | End-to-end question path with doubles |
| CMD-I | `uv run python -c "from answerdesk.llm.presets import PRESETS; print(sorted(PRESETS))"` | Preset surface |
| CMD-J | `uv run pytest -q tests/unit/policy tests/unit/cache` | Conformal, cost, PII, cache ATs |
| CMD-K | `uv run python -m answerdesk.policy.calibrate --synthetic --out reports/policy-smoke` | Calibration artifact production |
| CMD-L | `uv run pytest -q tests/integration/service` | Service ATs |
| CMD-M | `uv run --extra api answerdesk serve --fake --port 8765 --healthcheck` | Service boots, `/healthz` wiring |
| CMD-N | `cd web && npm ci && npm run build && npm run test` | Console build, component, a11y and XSS tests |
| CMD-O | `cd web && npm run screenshots` | Screenshot reproducibility |
| CMD-P | `uv run pytest -q tests/unit/eval` | Metrics, stats, judge, dataset ATs |
| CMD-Q | `uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke` | Offline end-to-end eval + report |
| CMD-R | `uv run answerdesk eval --all --report reports/bench` | Release evidence run (all G-REL metrics) |
| CMD-S | `uv run ruff check src tests` (+ format check) | Lint/format gate |
| CMD-T | `uv run python scripts/check-traceability.py` | Traceability matrix ↔ repo markers |
| CMD-U | `uv run answerdesk demo --smoke` | One-command offline demo — **delivered** (t14/t50): ingest → index → ask → calibrate → service health check in one command, no key, no network; measured 11.2 s on the reference machine (5 steps, `reports/demo/summary.json`), against the FR-OPS-06 budget of ≤ 120 s |
| CMD-V | `git -C answerdesk log --oneline -1` and tag check | Release commit + `v0.1.0` tag |
| CMD-W | `uv run pytest -q --cov=src/answerdesk --cov-report=term-missing` | Coverage thresholds |
| CMD-X | `uv run answerdesk eval --all --report reports/bench` (drift arm) | Drift simulation (E-7) |
| CMD-Y | `git grep -nEi -e "api[_-]?key" -e "secret" -e "token"` then review each hit for a hard-coded credential, expecting no findings | Secret scan |
| CMD-Z | `uv run pip-audit` (networked release step) | Known-vulnerability audit |
| CMD-NUM | `uv run python scripts/check-numbers.py` | NFR-REP-01: every published quantity traces to the docs/artifacts |
| CMD-LIC | `uv run python scripts/check-licenses.py` | NFR-REP-04: offline licence audit of the installed tree |

### 17.3 Minimum acceptance sequence for the release candidate

```text
uv sync --extra dev
uv run pytest -q                       # CMD-A
uv run ruff check src tests            # CMD-S
uv run answerdesk ingest --demo        # CMD-C
uv run answerdesk index --demo         # CMD-E
uv run answerdesk demo --smoke         # CMD-U (11.2 s measured, exit 0)
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke   # CMD-Q
uv run answerdesk eval --all --report reports/bench                                 # CMD-R (live or REPLAY-MODE)
uv run python scripts/check-traceability.py                                          # CMD-T
uv run python scripts/check-numbers.py                                               # CMD-NUM
uv run python scripts/check-licenses.py                                              # CMD-LIC
cd web && npm ci && npm run build && npm run test && npm run screenshots            # CMD-N, CMD-O
```

A release is accepted only when: every CMD above exits 0; every G-REL threshold in §§6–9 is met or the requirement was formally amended; the review checklist (§17.1) is signed with no open blocker/high finding; and `docs/results.md` carries the limitations of §10.7 verbatim.

---

## 18. Change control and definition of done

### 18.1 What "frozen" means

Version 1.0.0 is frozen: implementation tasks build against it. Changes are allowed but must be visible.

| Change kind | Example | Process |
|---|---|---|
| **Clarification** | Rewording a sentence without changing any threshold, behaviour or acceptance test | Patch bump (1.0.x), noted in the change log below |
| **Amendment** | Changing a threshold, adding/removing a requirement, changing a gate class or a metric definition | Minor bump (1.x.0) + ADR in `docs/design-decisions.md` + captain approval; the ADR must state the measured evidence that motivated it |
| **Scope change** | New module, new locale, new surface | Major bump (2.0.0) + re-plan; never smuggled into a module task |

### 18.2 Rules that keep the document honest

1. **No silent drift.** If implementation reality diverges from a requirement, the requirement is amended with an ADR — the doc is never left describing a system that does not exist.
2. **A number may move down only with evidence.** Lowering a target requires the measured result that made it unattainable, plus the residual-risk statement. "Ran out of time" is not evidence.
3. **New work must cite an ID.** A task, PR or review finding that cannot name a requirement is either a defect fix or a scope change; it may not silently expand the product (R-12).
4. **Adding a MUST requires an AT.** No requirement without an acceptance test and a traceability row.
5. **Ratchet on quality.** G-REL thresholds once met become regression baselines: a later release that drops below a previously achieved number fails, even if the floor in §6–9 is still met.

### 18.3 Definition of done (applies to every downstream task)

A task is done only when all of the following hold:

- [ ] Every FR/NFR it owns in §16 is implemented and its `AT-` marker exists in the codebase.
- [ ] Its G-CI tests pass offline, on Windows and Linux, with no key and no network.
- [ ] Its G-REL obligations are either satisfied with committed artifacts or explicitly deferred with an amendment.
- [ ] Its module appears in the answer/trace contract (§7.1, §4.2 trace) without contract drift.
- [ ] Its docs (where owned) match code, and every command in them was executed.
- [ ] No invariant in §3 is weakened; no new dependency violates §13.3.
- [ ] The task reports its verdict with evidence references (command output, artifact paths), not adjectives.

### 18.4 Document change log

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0.0 | 2026-10-20 | Initial frozen requirements: 87 FR, 34 NFR, 10 invariants, 121-row traceability matrix, evaluation protocol, risk register, non-goals, licensing constraints | architect |

### 18.5 Repair log (post-review change control)

Entries here are the one-line change-control notes required when the release candidate's own
machinery (matrix, numbers, licence audit) is corrected after the frozen 1.0.0 text. Each entry
names what changed, why, and which requirement it touches.

| id | date | change | requirement(s) | evidence |
|---|---|---|---|---|
| CC-2026-10-21-a | 2026-10-21 | §16 matrix module paths corrected to the modules that actually implement each row: `generate/prompts/`→`generate/prompts.py`, `policy/risk.py`→`policy/scorers.py` + `policy/metrics.py`, `policy/cost_model.py`→`policy/costs.py`, `web/tests/`→`web/src/`, bare `presets.py`→`llm/presets.py`, the non-existent `NOTICE`→`LICENSE` + `scripts/check-licenses.py`. `scripts/check-traceability.py` now exists, so that reference resolves too. No requirement was weakened: only its pointer was fixed | NFR-REP-03, FR-GEN-04, FR-GEN-11, FR-ESC-01, FR-ESC-03, FR-UI-09, NFR-PORT-01, NFR-REP-04 | `scripts/check-traceability.py` exits 0; `tests/smoke/test_traceability.py` seeds a missing marker and a broken path and asserts both fail |
| CC-2026-10-21-b | 2026-10-21 | NFR-LAT-01 annotated with the measured reality (p50 2,618.22 ms / p95 14,116.76 ms, n=200, MISS) and reported in §"NFR-LAT-01 verdict" of `docs/results.md`. The requirement is **not** amended: the target stands and the release records the miss; a future run that meets it (smaller/local model, streaming TTFT, shorter answers) supersedes this note | NFR-LAT-01 | `scripts/build-results.py` emits the verdict from `reports/bench/ops`; `--check` passes |
| CC-2026-10-21-c | 2026-10-21 | The 21 requirements that had no `AT-` marker now have one: the 11 console ids and `AT-A11Y-01`/`AT-SEC-04` are traced to their vitest cases in `tests/smoke/test_ui_traceability.py`; `AT-REP-01`/`AT-REP-02`/`AT-REP-04`/`AT-DOC-02`/`AT-QUAL-02`/`AT-QUAL-03`/`AT-SCALE-01`/`AT-SEC-05` are traceable to executable checks in `tests/smoke/test_requirements_evidence.py`; `AT-LAT-04` is a measured exact-hit latency assertion in `tests/unit/cache/test_semantic_cache.py` | NFR-REP-03 and the 21 rows | checker reports 121/121 marked |
| CC-2026-10-21-d | 2026-10-21 | NFR-REP-01 and NFR-QUAL-02 got executable gates: `scripts/check-numbers.py` (README numbers ⊆ generated docs/artifacts, results numbers ⊆ `reports/`), and a ≥70% coverage step in `scripts/ci-offline.ps1` (measured cache 85.5%, eval 73.6%, policy 90.0%, retrieval 92.5%, aggregate 80.7%). Both run in the offline gate; a smoke test fails if either is unwired | NFR-REP-01, NFR-QUAL-02 | `check-numbers.py` exit 0; `tests/smoke/test_requirements_evidence.py` |

Known open item recorded rather than hidden: markers `AT-CNS-01…04` (consistency suite) and
`AT-GEN-13…19` (experimental query-rewriter) exist in the tree but have no §16 row, so the matrix
does not yet cover those behaviours. The checker reports them as a note; they need a requirement
amendment (or removal) in a follow-up, not a silent deletion here.

---

---

## 19. Open questions: what this document decides, measures, or defers

The research left twelve decision-shaping questions. The requirements do not hide them; they place each one in a slot.

| Q | Question (research) | Disposition in v1.0 |
|---|---|---|
| Q1 | Does tuned convex combination beat RRF on multilingual support data? | **Frozen default**: RRF untuned ships; CC is the tuned path. **Measured** by RT-7/§6.4 with a falsification clause. |
| Q2 | BGE-M3 vs Qwen3-Embedding-0.6B as default | **Frozen**: BGE-M3 default (dense+sparse in one), Qwen3 as documented alternative; §6.3 requires at least one real embedder per release claim. |
| Q3 | Do contextual chunk headers pay off on support docs? | **Flag, SHOULD**, ablated in §10.6; not a gate. |
| Q4 | Operating point for escalation and cost-matrix ownership | **Frozen**: α tiers (§8.2), cost model config with provenance (§8.3); real handoff costs remain a pilot input, documented as assumption. |
| Q5 | Can the semantic cache be safe by default? | **Frozen**: on for informational intents with guards, off for high-risk; measured by K-1/K-2 on the near-miss set. |
| Q6 | Typo and code-switch degradation | **Measured** by RT-9 + perturbation set (§10.1); no absolute claim until measured. |
| Q7 | Learned multi-feature fusion vs single α | **Deferred** to v1.1; §10.6 keeps the ablation runnable; single α in v1.0. |
| Q8 | Context budget: 8, 12 or 20 chunks | **Frozen default 8–12** with the k-sweep in §10.6 as the deciding evidence. |
| Q9 | Which judge, and honesty across six languages | **Protocol** in §10.5: judge ≠ generator, per-language κ, "not reportable" rule. |
| Q10 | Is citation verification accurate enough not to suppress good answers? | **Budgeted**: F-9 false-rejection ≤ 0.10 + claim-level labelled set. |
| Q11 | Conformal coverage decay under drift and refresh cadence | **Operational contract** §8.6 + E-7 drift simulation; weekly refresh. |
| Q12 | Minimum viable multilingual golden set under public-data constraints | **Specified** §10.1/§10.3: ≥ 50 items per locale in test, document-disjoint splits, published as data. |

Deferred items are not silent: they appear here, in `research/answerdesk/open-questions.md`, and in the roadmap section of `README.md`.

---

## Appendix A — Additional frozen decisions (D-16 … D-20)

| ID | Decision | Consequence |
|---|---|---|
| D-16 | **Two-tier gate classes** (G-CI vs G-REL) with mandatory companions | Quality is never claimed from doubled/faked runs; CI proves machinery only (§0.3) |
| D-17 | **Traceability markers** `# AT-…` in code plus a checker script | The matrix is machine-verifiable, not aspirational (§16, NFR-REP-03) |
| D-18 | **Near-miss replay set** for cache wrong-hit measurement | The cache budget is measurable instead of rhetorical (§9.5) |
| D-19 | **Reference machine + pricing table identity in every performance artifact** | Latency/cost numbers are reproducible or explicitly scoped (§17.2, NFR-COST-02) |
| D-20 | **REPLAY-MODE labelling** | Recorded runs can never masquerade as live results (§10.7 item 9) |

## Appendix B — Decision reason codes (closed enum)

| Code | Applies to | Forced by | Meaning |
|---|---|---|---|
| `ANSWERED` | — | — | Decision is `answer`; no reason code required, recorded for completeness |
| `CLARIFY_AMBIGUOUS` | clarify | cost band | Score in the clarify band; ask one disambiguating question |
| `NO_EVIDENCE` | escalate/refuse | evidence check | No passage passed the evidence threshold |
| `LOW_CONFIDENCE` | escalate | conformal | `s > q̂` at the active α |
| `VERIFIER_REJECT` | escalate/refuse | verifier | All claims unsupported after one repair |
| `CITATION_FAILED` | escalate | structural | Citation resolution/quote/numeric guard failed after repair |
| `HIGH_RISK_INTENT` | escalate | policy | Intent bucket requires human handling |
| `PII_DETECTED` | escalate/refuse | PII guard | PII present and policy forbids auto-handling |
| `INJECTION_SUSPECTED` | escalate/refuse | injection guard | Instruction-like content in query or retrieved passages |
| `OUT_OF_SCOPE` | refuse | policy | Query outside the supported product/domain |
| `UNSUPPORTED_LANGUAGE` | refuse/escalate | policy | Language outside the six supported locales |
| `CORPUS_MISSING` | escalate | runtime | Index/corpus unavailable or version mismatch |
| `PROVIDER_ERROR` | escalate | runtime | Provider failed after retries/fallback |
| `STALE_CALIBRATION` | escalate | runtime | Calibration artifact expired or version-mismatched (§8.6) |
| `OPERATOR_REQUEST` | escalate | human | Operator chose to escalate |
| `USER_REQUESTED_HUMAN` | escalate | user | Customer asked for a human |

Parity requirement: this table is the single source for `service`, `cache`, `web` and `eval`; a parity test (AT-ESC-05) asserts code enum = docs table = UI label map.

## Appendix C — Configuration keys the requirements depend on

| Key | Default | Owner requirement |
|---|---|---|
| `policy.alpha.informational` / `.transactional` / `.account_legal` | 0.05 / 0.02 / 0.01 | FR-ESC-02, §8.2 |
| `policy.costs.c_wrong` / `.c_abstain` / `.c_delay` | 1.00 / 0.10 / 0.05 | FR-ESC-03, §8.3 |
| `policy.calibration_artifact` | path + hash | FR-ESC-06, §8.6 |
| `cache.enabled` | `true` | FR-CAC-08 |
| `cache.epsilon` (wrong-hit budget) | 0.01 | FR-CAC-07, §9.5 |
| `cache.ttl_class.{policy,howto,escalation}` | 1 h / 7 d / 600 s | FR-CAC-09, §9.4 |
| `cache.high_risk_intents` | account, refund, legal, payment | FR-CAC-06 |
| `retrieval.fuser` | `rrf` (untuned default) | FR-RET-04, D-03 |
| `retrieval.rrf_k` | 60 | FR-RET-04 |
| `retrieval.cc_alpha` | fitted (artifact), not a literal | FR-RET-05 |
| `retrieval.top_k` | 50 (rerank input) / 20 (gate) | §6.2 |
| `rerank.enabled` / `.budget` | `false` in offline profile / 50→12 | FR-RET-07 |
| `generate.context_k` / `token_budget` | 8–12 chunks / hard cap | FR-GEN-05 |
| `generate.repair_passes` | 1 | FR-GEN-07 |
| `llm.preset` | `openai` (API) or `offline` (CI) | FR-OPS-03, NFR-PORT-01 |
| `llm.pricing_table` | pinned file with `retrieved_at` | NFR-COST-02 |
| `eval.seed` / `eval.split` | fixed seed / `evals/splits/v1.json` | FR-EVL-02/03, NFR-DET-02 |

Any new configuration key that changes a gated behaviour requires a row here in the same change (CL-POL-01).


