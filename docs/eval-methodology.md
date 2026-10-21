# Evaluation methodology (answerdesk v1.0)

This document is the reference for **what the numbers mean, how they are produced, and what
they cannot say**. It is normative for the harness in `src/answerdesk/eval/` and is the
checklist item `CL-EVL-01` of `docs/requirements.md` §17.1.

Requirements covered here: FR-EVL-01 … FR-EVL-11, NFR-DET-02, NFR-ML-01, plus the metric
definitions every G-REL gate in §§6–10 of the requirements depends on.

> **One sentence summary.** The harness can prove the *machinery* offline and repeatably; it
> cannot prove quality without a real embedder, a real provider, a calibrated judge and
> human-verified labels. Every report says which of those it had.

---

## 1. Commands and artifacts

```bash
cd answerdesk

# offline smoke run: no network, no API key, no GPU (NFR-DET-01)
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke

# re-freeze the golden set from the corpus (deliberate, reviewed act)
uv run answerdesk eval --synthetic --provider fake --report reports/offline-smoke --regenerate

# ablation arms (bm25, dense-hash, hybrid-rrf) with paired statistics
uv run answerdesk eval --all --report reports/bench

# retrieval machinery only, no judge calls
uv run answerdesk eval --retrieval-only --report reports/retrieval
```

A run writes, into `reports/<run>/`:

| File | Content |
|---|---|
| `report.md` | the report: header, retrieval, answer quality, escalation/calibration, operations, statistics, targets, limitations |
| `metrics.json` | every cell, comparison and target, machine-readable |
| `turns.jsonl` | one line per `(case, arm)` — decisions, citations, latencies, cost |
| `cells.csv` | the same cells as CSV |
| `plots/*.png` | retrieval recall by arm, reliability diagram, risk–coverage curve, stage latency |
| `../latest.json` | stable pointer to the newest run (and a `latest` symlink when the OS allows) |

The report header carries the **corpus hash, golden hash, split hash, seed, judge identity
(model + prompt version + rubric version), provider mode, hardware, pricing table and the
deterministic digest**. A report without those fields is a defect (`FR-EVL-09`,
`NFR-REP-02`).

### 1.1 Replay mode

A run that used deterministic doubles prints `REPLAY-MODE` in a banner *and* in its header
notes, and every target it cannot speak to is marked `diagnostic` rather than `pass`/`FAIL`
(§10.7 item 9, `NFR-REP-01`). The harness refuses to write a replay report without that
note (`require_statistics`).

### 1.2 Determinism

Fixed seed + fixed split + fixed corpus ⇒ identical `metrics.json` after removing timing
fields. The digest printed in the header and in `metrics.json` is computed over exactly that
projection: metric cells (latency cells excluded), comparisons, and the per-turn decisions,
citations and rankings. Latency and USD are excluded because they are properties of the
machine, not of the system under test (`NFR-DET-02`).

Bootstrap resampling uses `numpy.random.default_rng(seed)` (PCG64), whose stream is stable
across numpy releases; the same seed therefore reproduces the same interval.

---

## 2. Golden set (`golden/v1`)

### 2.1 Composition

Per `docs/requirements.md` §10.1:

| Property | Implemented as |
|---|---|
| Locales | `ru, tg, uz, ky, tr, zh`; every locale present in **every** split |
| Size | test ≥ 50 items/locale (the frozen set: 60), dev ≥ 30 (the frozen set: 36) |
| Unanswerable share (test) | 18% target, enforced inside 15–20% |
| Category quotas (test, per locale) | direct 35%, numeric/date 20%, typo-or-code-switch 15%, unanswerable 18%, PII 6%, prompt-injection 6% |
| Separate label classes | retrieval labels (`gold_chunk_ids`, `required_citation_chunk_ids`) vs answer labels (`reference_answer`, `answerable`, `intent_bucket`, `category`, `locale`, split) — never merged |
| Provenance | every case records `source`, `generator`, `seed`, `document_ids` and the labelling rule |
| Additional sets | near-miss cache pairs ≥ 120 (24/locale), injection ≥ 40, PII ≥ 40, perturbation ≥ 120, claim-level verifier labels ≥ 200 |

**Multilingual is the locale axis, not a seventh category.** The frozen `EvalCategory` enum
has six members; §10.1 requires the six locales in every split and every reporting table. The
generator therefore asserts all 36 locale × category cells of the test split are non-empty
and writes the matrix into the manifest (`multilingual_cells`), and every metric table has a
per-locale section plus a macro row (`NFR-ML-01`).

### 2.2 How a case is produced

The demo corpus (`answerdesk ingest --demo`) is generated from locale banks: each document
carries sections, a table, and an FAQ block; the query pool (`data/corpus/queries/queries.jsonl`)
holds two questions per document — one *direct* question taken from the document's first FAQ
pair, and one *perturbed* question (typo, code-switch, numeric/date or identifier
substitution). The golden generator consumes that pool; it never invents text in a language
it cannot check.

Families built **from** the corpus, not authored by hand:

| Family | Construction | Label |
|---|---|---|
| direct, numeric/date, typo/code-switch | sampled from the corpus query pool | answerable; retrieval labels from the FAQ structure |
| unanswerable | a locale-native request for an order/operation id that is **verified absent** from every chunk (`anchor_index`) | `answerable=false`, `must_escalate=true` |
| PII | the same request plus reserved-range PII (4111… PAN, example.* e-mail, reserved phone range) | `answerable=false`, `must_escalate=true`, `pii_kind` recorded |
| prompt-injection | the same request plus one of five English directive payloads that each trip a different rule in `generate.guards` | `answerable=false`, `must_escalate=true`, `injection_payload_id` |

### 2.3 Retrieval labels: the equivalence class

The primary evidence of a generated question is the **FAQ chunk of its source document** —
structural ground truth, because the chunker keeps FAQ pairs atomic and the question was
generated from that FAQ block.

The demo corpus asks *generic template questions* ("how do I block my card?") in several
documents per locale, and the template's digits are drawn from a different slot draw than the
document's. A document-specific gold chunk would therefore be an arbitrary choice among
equally valid answers, and would punish a correct retrieval. The harness labels instead the
**equivalence class**: every FAQ chunk whose question normalises to the same stem
(digits masked, punctuation dropped, case folded) as the query's base question.

Consequences, stated so nobody over-reads a number:

* `recall_any@k` asks "did the ranking return at least one document that answers this
  question" — the user-facing property;
* `recall_all@k` measures **coverage of the equivalence class**, not completeness of one
  document's evidence; a class of ten documents makes it low by construction;
* `required_citation_chunk_ids` keeps the *source* document's FAQ chunk for auditing, and a
  correct answer that cites any acceptable chunk counts as citing evidence
  (`TurnRecord.evidence_cited`).

### 2.4 Machine labels are not gate labels

Every generated case carries `metadata.machine_label = true` and `label_origin`. §10.3
requires human verification for every answer label and excludes machine-only labels from any
gate. Therefore:

* a replay report's numbers are diagnostics;
* `docs/results.md` may not quote a number from a run without human-verified labels;
* the tooling for the human pass exists (per-language κ, `evals/judge/human_v1.jsonl`) but the
  labels themselves are a human deliverable, not a generated artifact.

### 2.5 Adversarial companion sets

| Set | File | Size (frozen v1) | Purpose |
|---|---|---|---|
| near-miss cache pairs | `adversarial/near_miss.jsonl` | 144 (24/locale) | §9.5 wrong-hit replay: same intent, near-identical wording, *different* acceptable evidence |
| injection | `adversarial/injection.jsonl` | 78 | guard coverage (`NFR-SEC-01`) |
| PII | `adversarial/pii.jsonl` | 72 | redaction boundaries (`NFR-SEC-02`) |
| perturbation | `adversarial/perturbation.jsonl` | 198 | RT-9 delta (typo/code-switch robustness) |
| claim labels | `adversarial/claims.jsonl` | 242 | verifier precision/recall (F-9): verbatim sentences (`supported`) and mutated twins (`unsupported`) |

Adversarial sets are written to their own files and are **not** part of the stratified test
count: counting a replay pair as a test item would inflate every per-locale share.

### 2.6 Frozen artifacts

`evals/golden/v1/{test,dev}.jsonl` + `manifest.json`, and `evals/splits/v1.json`. Both are
hashed: the split manifest carries its own self-hash (a hand-edited split is rejected) and the
golden manifest carries the corpus hash, the split hash and a sha256 per file.

`--regenerate` is the only way to change them, and it is a reviewed act: the same seed
reproduces byte-identical files, so a diff means the generator changed.

---

## 3. Splits and leakage

* The split unit is the **document**, never the query (§10.2). Two paraphrases of the same
  paragraph in different halves would leak the answer into the fitting half and inflate every
  number.
* The split is **stratified by deficit coverage**: documents are chosen for the test half so
  that each locale has enough numeric/date and typo questions to meet its quotas, ties broken
  by `sha256(seed:doc_id)`; the remaining documents form dev. Ratio 60/40 by document.
* The harness fails loudly (§10.2) on:
  * a missing answer or retrieval label (`MissingLabelError`),
  * a gold chunk id that does not exist in the corpus (`UnknownChunkError`),
  * a test item whose document also feeds dev (`SplitLeakageError`),
  * an unknown locale, an empty question, a duplicate case id, an empty selection,
  * a corpus that no longer matches the frozen manifest's `corpus_hash`,
  * a golden set whose split hash and split manifest disagree.
* Nothing is fitted on the test split by the harness. Fusion parameters, α, cache thresholds
  and judge calibration are dev-split work (`INV-09`); the harness exposes dev loading for
  that purpose (`--split dev`).

---

## 4. Metrics

All formulas below match the code; the tests compare them against hand-computed values
including ties and empty sets (`AT-EVL-04`). Let `top_k(q)` be the ranked chunk ids returned
by an arm, `G_q` the gold set.

### 4.1 Retrieval (§6.1)

```
recall_any@k  = (1/|Q|) Σ_q 1[G_q ∩ top_k(q) ≠ ∅]
recall_all@k  = (1/|Q|) Σ_q |G_q ∩ top_k(q)| / |G_q|
nDCG@k        = (1/|Q|) Σ_q DCG_q@k / IDCG_q@k,
                DCG_q@k = Σ_{i≤k} rel_i / log2(i+1),   rel_i ∈ {0,1}
                IDCG_q@k = Σ_{i≤min(k,|G_q|)} 1 / log2(i+1)
MRR@k         = (1/|Q|) Σ_q 1 / rank_q(first gold in top_k)  (0 if absent)
```

k ∈ {10, 20}: 20 is the retrieval gate (`RT-1`), 10 is what the packed context can use
(`RT-3`). An empty gold set is a labelling bug and raises, never a zero.

### 4.2 Answer quality (§7.5)

* **Groundedness** (`F-1`) — share of displayed claims the verifier/judge accepts:
  `groundedness_verifier = supported claims / displayed claims` (deterministic, from the
  pipeline's verifier) and `groundedness_judged` = share of claims the judge scores ≥ 4
  against their cited passage. Both are reported; they are different instruments.
* **Citation precision** (`F-2`) — of the citations attached to a claim, the share whose claim
  is `supported`: `|citations of supported claims| / |citations|`, averaged per answer.
* **Citation recall** (`F-3`) — the ALCE definition: `claims fully supported by ≥ 1 citation /
  claims`.
* **Structural counters** — `phantom_citations` (a citation whose chunk is not in the packed
  evidence) and `unsupported_displayed_claims` must both be 0 (`F-4`, `F-5`); they are
  reported as rates over answered turns.
* **Rubric score** — 1–5 from the judge, `pass ≥ 4`; per-turn value is the mean over claims.
* **Language agreement** (`F-7`) — the answer language equals the detected query language.

### 4.3 Escalation (§8.5)

* `realised_risk` = incorrect auto-answered turns / auto-answered turns (`E-1`), per intent
  bucket and pooled.
* `coverage` = answered answerable items / answerable items (`E-4`).
* `must_escalate_recall` = escalated among unanswerable ∪ PII ∪ injection ∪ out-of-scope
  (`E-2`).
* `escalation_precision` = escalated turns that were genuinely unsafe / escalated (`E-3`).

"Correct" is defined once, in `runner._is_correct`, exactly as §8.1 says: rubric pass **and**
every displayed claim supported **and** evidence cited **and** language agreement. For a
must-escalate case the same slot records the safe outcome (escalated); that is what makes risk
and coverage comparable across arms.

### 4.4 Calibration (§8.5 E-5/E-6)

* `ECE` with equal-width **and** equal-mass bins: `Σ (n_b/n)·|accuracy_b − confidence_b|`.
* `Brier` = mean squared error of the confidence forecast.
* Reliability diagram data, **risk–coverage curve** and `AURC` (trapezoid over the curve).
* `coverage_at_risk(target)` and `risk_at_coverage(c)` for the operating point.

Confidence in this harness is the **packed evidence score** (`PackedContext.evidence_score`),
not a conformal `p̂`. The conformal calibrator is t8's; until it is wired in, the calibration
section is a diagnostic and no risk guarantee is claimed. Calibration correctness is defined
as "answering was the right call and the answer was right", so a refusal on an unsafe turn
counts as a correctly predicted non-answer.

### 4.5 Operations

`p50`/`p95` latency per stage (retrieval, generation, judge, total), mean and p95 USD per
answer, mean token counts, and the cache hit rate. Latency is measured in-process around the
pipeline call: it is a regression guard (`NFR-LAT-03`), not a server-side latency claim
(`NFR-LAT-01`). The bracket printed for a percentile is the p(q−5)…p(q+5) spread, not a
confidence interval. The cache hit rate in a replay run is an **exact-normalised-query probe**
over the case list — the semantic cache is t8's, so `K-1…K-6` are not measured here.

### 4.6 Statistics (FR-EVL-07)

| Tool | Use | Detail |
|---|---|---|
| Wilson score interval | every proportion | no continuity correction; never collapses to zero width at 0/1 |
| Paired bootstrap | "is arm A better than arm B" (paired over queries) | resamples **queries** (`B = 1000`, 2.5/97.5 percentiles), same items in both arms |
| Two-level bootstrap | the macro row | resamples queries inside each locale, then recomputes the equal-weight mean over locales |
| McNemar | binary outcomes (any-hit@k) | χ² with continuity correction; exact-binomial fallback when `b + c < 25` |
| Cohen's κ | judge ↔ human agreement | per language and pooled; gates reportability at κ ≥ 0.60 |

A difference whose 95% interval contains 0 is reported as **noise**, not as an improvement in the paired bootstrap (§6.3 step 5). There is no multiplicity correction: with several arms and several metrics, p
values are descriptive, and the report says so.

---

## 5. The judge

### 5.1 Rubric and anchoring

The rubric (`evals/judge/rubric_v1.md`, `RUBRIC_VERSION = rubric-v1`) grades claim support
against the **cited passage**, not against the model's own text and not against style,
length, tone or confidence — those are the documented verbosity and self-preference biases.
Grading is **per claim** and averaged, because whole-answer grading against a short reference
measures length.

### 5.2 Position swap

Pairwise comparisons are implemented once, in `judge.compare_pair`, which runs both orders
(A vs B and B vs A) and averages the two scores. `compare_pair` is the only path: the
deterministic double and a live judge are audited by the same code. `position_swap_audit`
reports order invariance, and the run notes print it (the offline smoke run prints
`8/8 comparisons kept their verdict across both orders`). A judge that always prefers the first
slot scores an invariance of 0.0 in the unit test — the audit catches it instead of the
leaderboard quietly absorbing it.

### 5.3 Judge identity and reportability

Every judged number carries `judge_model`, `prompt_version`, `rubric_version` and the κ block
(§7.5 honesty clause). A judged metric is **reportable** only when per-language κ ≥ 0.60. The
harness ships no human labels for the demo corpus, so its offline judged metrics are
explicitly `not reportable`, and the report says so instead of softening the claim.

`build_judge` refuses a configuration where the judge model equals the generator model
(§10.5: a model family may not grade its own output in a reported metric).

### 5.4 The deterministic double

`answerdesk.testing.doubles.FakeJudge` scores lexical overlap between the candidate and the
reference (or the context when there is no reference). It is deterministic and cheap; it is
**not** a measurement of quality, and its agreement with humans is unknown. The offline run
uses it to prove the plumbing.

`eval.fixture_provider.EvidenceAnchoredProvider` is the offline *generator*: a `FakeProvider`
subclass that emits schema-valid drafts by quoting a sentence verbatim from the highest-
overlap packed passage, refusing when no passage shares a content word with the question or
when the question names an identifier no passage contains. Consequences: groundedness and
citation precision are high by construction in replay mode. That is what a plumbing run
should look like; it is not evidence about a real model.

---

## 6. Known biases and limitations

The honest-reporting policy of §10.7 applies verbatim and is printed in every report:

1. **Synthetic data optimism.** Queries derive from the corpus; recall and groundedness are
   optimistic relative to real support traffic.
2. **Small n.** A few hundred items give roughly ±3 points on the macro retrieval metric and
   much wider intervals per locale and per subset.
3. **Judge bias.** Judged metrics are LLM-mediated; low-resource languages are likely worse.
   κ per language is published; below 0.60 the metric is "not reportable".
4. **Benchmark ≠ support domain.** Public leaderboard values do not justify an embedder
   choice; only our own ablation does.
5. **Conformal exchangeability.** The risk guarantee holds under exchangeability; v1.0 has no
   real traffic, so drift is simulated, not observed.
6. **Environment dependence.** Latency and cost numbers belong to the recorded machine and the
   pinned pricing table.
7. **No production evidence.** No SLA, no incident history, no live traffic, no measured
   handoff cost.
8. **Cache results are replay-based.** Wrong-hit rates come from adversarial replay pairs.
9. **Replay mode is labelled.** Any number produced without a live model carries
   `REPLAY-MODE`.
10. **No claim of SOTA.** The repository claims reproducibility, calibration artifacts and
    citation verification.

Harness-specific limitations on top of §10.7:

11. **Equivalence-class retrieval labels.** A gold set is "all documents carrying this
    question template", so `recall_all` measures class coverage. The wording of the metric is
    printed with every table for that reason.
12. **Reference answers are not verifiable in the demo corpus.** The generator draws the
    operator reply from the intent's FAQ bank with its own slot draw, so its amounts, dates and
    identifiers often occur in no document. `reference_is_verifiable` records this per case and
    the judge grades such answers against their cited evidence instead. The diagnostic cell
    `rubric_reference_overlap` shows the gap: when it is low, the reference — not the system —
    is the problem.
13. **Retrieval arms are reference implementations.** `eval.arms` provides BM25, a hashed
    dense arm and RRF. They exist so CI measures a real ranking over the real corpus; the
    shipped hybrid retriever with a real multilingual embedder is t5's, and t11 must re-run the
    evidence arms with it.
14. **No conformal guarantee is measured here.** `p̂` is the evidence score; ECE/Brier/AURC are
    diagnostics of that score, not of a calibrated policy.
15. **No multiplicity correction.** Many cells, many arms; intervals are per-cell.
16. **The demo corpus is not reviewed by native speakers** (see `docs/ingestion.md` §12), so
    language-dependent metrics on it inherit the generator's wording, not a language's.

---

## 7. Labeling protocol for the human pass (§10.3)

The harness supports the protocol; it cannot perform it. Steps, in the order they must happen:

1. **Bootstrap from documents** — sample chunks, draft candidate Q/A pairs with a strong
   model, then have a human verify, correct or reject (the ARES/PPI pattern).
2. **Mark all sufficient passages**, not "the best" one: `recall_all` is meaningless
   otherwise. A passage is sufficient if it contains the fact the answer states.
3. **Two annotators on a 200-item subset** (≥ 30 per locale), independently, then
   adjudication. Publish Cohen's κ per language and overall; target κ ≥ 0.70 pre-adjudication.
4. **Preserve ambiguity** — items that adjudication cannot settle get `ambiguous: true` and are
   excluded from the primary metric, with the count reported.
5. **Record provenance** for every item (document-derived, LLM-drafted + human-verified,
   perturbation-injected) and the seed.
6. **Drop the reference answer when it is not entailed by the evidence**, or fix the corpus;
   do not grade a model against a reference the knowledge base cannot support.

Annotator instructions (verbatim, for the 200-item subset):

> You are given a customer question, the document it came from, and a candidate answer.
>
> 1. Mark every passage that would let a careful agent answer the question. Mark all of them,
>    not the best one.
> 2. Write the reference answer in the question's language, using only facts from the marked
>    passages. Copy numbers, dates and identifiers exactly.
> 3. If the documents cannot answer the question, say so and label the item unanswerable.
> 4. If the question is genuinely ambiguous (two readings with different answers), label it
>    ambiguous and write both readings; do not pick one.
> 5. Judge the answer, not the writing: length, tone and confidence are irrelevant.

---

## 8. Extending the harness for the evidence run (t11)

1. Replace `eval.arms` with the shipped retriever (`answerdesk.retrieval`, t5) — the runner
   accepts any `answerdesk.contracts.Retriever`.
2. Use a real multilingual embedder (BGE-M3 or Qwen3-Embedding-0.6B) and a real provider;
   `build_provider` already routes through the universal provider layer.
3. Wire the conformal calibrator (t8) into `TurnRecord.confidence`; then ECE/Brier/AURC and
   `E-1` become gateable instead of diagnostic.
4. Add human-verified labels plus `evals/judge/human_v1.jsonl`; judged metrics become
   reportable per language once κ ≥ 0.60.
5. Add the semantic-cache replay (t8) for `K-1…K-6`.
6. Run `answerdesk eval --all --report reports/bench` and quote **only** those artifacts in
   `docs/results.md` (`NFR-REP-01`).

# AT-EVL-11
