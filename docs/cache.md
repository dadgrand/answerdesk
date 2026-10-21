# Semantic cache: guards, the wrong-hit budget, and invalidation

Owner: **t8** (`answerdesk.cache`). Normative source: `docs/requirements.md` §9 (cache
contract), §3 (invariants), §7.2 (citation rules that G-4 re-runs), Appendix C (keys: `cache.
enabled`, `cache.epsilon`, `cache.ttl_class.*`, `cache.high_risk_intents`).

```text
query ──► exact key ──hit──► guards G-1…G-4 ──pass──► serve
   │            │ miss                                    ▲
   │            ▼                                         │
   └──► high-risk intent? ──yes──► BYPASS (exact only)     │
                │ no                                      │
                ▼                                         │
        embedding search k=1 ──► similarity ≥ τ(cohort)? ─┘
                     │ no
                     └──► MISS (regenerate)
```

A hit is re-validated on **every** read. The check is string and set work, so "skipping it for
latency" would be false economy: a wrong hit is the worst failure this product can produce.

---

## 1. What an entry is (§9.1)

`CacheEntry` stores the whole answer artifact — text, claims, citation spans and hashes,
decision, usage, provenance — plus the retrieval fingerprint, the TTL class, doc hashes,
the embedding and hit counters. Caching a bare string is forbidden by the contract, and the
type makes it impossible: there is no field for one.

The key is `CacheKey`:

```
normalised_query   NFKC + case-fold + whitespace collapse + edge-punctuation trim
locale             BCP-47-ish tag
intent_bucket      informational | transactional | account_legal
entities           identifiers (ORD-12345, TS-999) + canonical numbers (amounts, dates, times)
versions           corpus, index, prompt, model, policy-artifact hashes
```

`exact_key()` hashes the tuple. Normalisation never transliterates or translates (NFR-ML-02):
`Сколько стоит карта?` and `сколько стоит карта` are one key; `«Вектор»` keeps its angle
brackets inside the string and only the edges are trimmed.

Entity extraction reuses the same numeric canonicalisation as the numeric guard, so `12,5 €`
and `12.50` are the same entity and `05.03.2026` equals `5 марта`. The guard compares the two
sets for **equality** (not subset): a query that adds an entity is a different question.

---

## 2. Guards (§9.2), in order

| guard | rule | failure it prevents |
|---|---|---|
| G-1 version | `entry.key.versions == live versions` (corpus, index, prompt, model, policy hash) | serving an answer produced by a different system |
| G-2 entity | extracted entities equal | "your refund is 30 days" answering a question about another plan/amount/date |
| G-3 source doc | every cited chunk still exists and its `doc_hash` is unchanged | citing a deleted or edited document |
| G-4 structural | citations resolve, quotes still substrings, numeric guard (INV-04) re-run, markers bijective | any silent stale-citation path |
| G-5 risk tier | account/legal and the configured high-risk labels never take a *semantic* hit | confidently wrong answers on money/account questions |

G-4 is `cache/revalidate.py::revalidate_answer`: it re-runs `find_quote_span` for every
citation, `numeric_violations` for every claim, and `validate_render` for the marker structure.
A failure demotes the read to a miss with `guard="G-4"` and the reason recorded — the caller
then takes the normal path.

G-5 is implemented as *exact-only*, not *no-cache*: an identical repeat on a refund question is
still served (`test_high_risk_intents_bypass_the_semantic_path`).

---

## 3. Thresholds and the wrong-hit budget (§9.3, §9.5)

A single global cosine threshold gives no error guarantee. For each **cohort** (intent bucket ×
locale family × question-length band) the calibration picks

```
τ(coh) = min { τ : wrong_hits(τ) / hits(τ) ≤ ε }        ε = 0.01
```

on an adversarial replay set, subject to:

* **the entity guard is inside the loop** — a pair whose entities differ is not a hit, so a
  near miss blocked by G-2 does not force τ upward;
* **exact-equivalent pairs always hit** (`normalise_query` equal), because that is FR-CAC-01's
  exact path, independent of any threshold;
* **a floor** `τ ≥ 0.80`. The replay set is finite and not exhaustive; a sweep that "proves" a
  near-zero threshold is not evidence that every future query belongs to the cached one.
  Without the floor the transactional cohorts calibrate to τ = 0, which is indefensible;
* **infeasible cohorts** get `τ = 1.0` (no semantic hits) and `feasible=False`. K-1 cannot be
  met by guessing.

### 3.1 Where the Wilson bound is applied

§9.3 asks for `wrong_hits/hits ≤ ε` **and** `UCB₉₅ ≤ 2ε` *per cohort*. A 2% Wilson upper bound
needs roughly 190 hits in that cohort; a realistic replay set (the requirements' own "≥ 120
pairs, ≥ 20 per locale") has far fewer. So:

* the **per-cohort** gate is the measured rate (≤ ε), with the per-cohort UCB kept in the
  report as a diagnostic;
* the **pooled** UCB (≤ 2ε) is asserted on the whole replay set
  (`report.CacheEvaluation.passes_k1`), which is the statistically meaningful test at this
  sample size.

This is stated here because a reader will otherwise wonder why a cohort with 26 hits reports
`UCB = 0.13`. With 0 wrong hits out of 527 pooled hits the pooled bound is 0.0072 — inside 2ε.

---

## 4. The near-miss replay set (§9.5)

`cache/replay.py` builds it deterministically per cohort with three families:

| family | construction | similarity | gold answer |
|---|---|---|---|
| paraphrase | identical text, or the same tokens reordered | highest for identical, lower for reordered | same |
| near miss | one answer-relevant entity changed (plan suffix, card type, amount, date, legal-entity wording) | high — deliberately close | different |
| distractor | related but different intent | lower | different |

The similarity of each pair is whatever the configured embedder says; the calibration only sees
`(similarity, wrong?, entities-compatible?, exact-equivalent?)` and must find a threshold that
keeps the budget. That is the honest direction of the test: labels are ground truth, similarities
are measurements.

`cache-report.json`/`.md` report, pooled and per cohort: wrong-hit rate, Wilson UCB, hit rate,
precision (correct hits / hits) and recall (correct hits / pairs whose answer still applies).
`hit-rate-vs-error.svg` is the vCache-style curve (§9.6).

---

## 5. Invalidation (§9.4) — the matrix, as code

| trigger | implementation | effect |
|---|---|---|
| any ingest/delete (corpus bump) | `invalidate(corpus_version=…)` | **all** entries dropped (conservative) |
| one document changed/deleted | `invalidate(document_ids={…})` | entries citing that document dropped immediately |
| prompt/model/policy change | G-1 on every read | entry never served again |
| TTL | `CacheEntry.is_expired` with per-class TTL (`policy` 1 h, `howto` 7 d, `escalation` 600 s) | expired entry evicted on access |
| kill switch | `enabled=False` | every read BYPASSED; `put` refuses |

Negative caching (§9.4): an escalated/refused turn is stored with class `escalation` and
`negative=True`; a negative hit returns the *escalation artifact* (empty text, `action=escalate`),
never prose.

---

## 6. Observability (§9.6)

`CacheStats` exposes `entries`, `hits`, `misses`, `bypassed`, `exact_hits`, `semantic_hits`,
`negative_hits`, `wrong_hits`, `guard_rejections`, `invalidations`, `hit_rate`, `by_intent`.
`record_wrong_hit(entry_id)` is the live feedback hook for K-1 (an operator marking a hit wrong).
t9 maps these onto `/metrics`; the names are in `tests/unit/cache/test_semantic_cache.py`.

---

## 7. Measurements from the smoke run

`python -m answerdesk.cache.calibrate --synthetic --out reports/cache-smoke` (seed 20261020,
120 pairs/cohort, 12 cohorts, ε = 0.01; numbers from `reports/cache-smoke/cache-report.json`):

| metric | value | target |
|---|---|---|
| pairs | 2160 | ≥ 120 (adversarial) |
| hits | 527 | — |
| wrong hits | 0 | — |
| wrong-hit rate | 0.0000 | ≤ 0.01 (K-1) |
| wrong-hit UCB (pooled) | 0.0072 | ≤ 0.02 (K-1) |
| hit rate (all) | 0.2440 | — |
| hit rate (informational) | 0.2986 | ≥ 0.15 (K-2) |
| precision / recall | 1.0000 / 0.7319 | reported |
| calibrated τ | 1.0 in every cohort | — |

**Why τ = 1.0 everywhere.** The CI embedder is a hashing bag-of-words double. Under it, a
one-token near miss (`«Вектор»` → `«Вектор Плюс»`) is *more* similar than a reordered
paraphrase, so the budget can only be met by admitting exact-equivalent queries; the entity
guard is what blocks the entity-differing near misses (which is why the transactional cohorts
would otherwise accept τ = 0). With a real dense embedder (BGE-M3 / Qwen3, t11) τ will sit in a
useful band — the machinery, the budget and the report are what this task proves on data whose
truth is known. This is a limitation of the *study*, stated rather than hidden.

---

## 8. Limitations

1. **Synthetic replay set.** Paraphrases are token reorderings; real traffic is harder. t11
   replays the HITL-labelled near-miss set and re-calibrates.
2. **The embedder decides.** With the offline double, thresholds converge to exact-match (see
   above); a dense embedder is required for the semantic path to earn its keep.
3. **Per-cohort UCB is statistically out of reach** at realistic replay sizes (§3.1); the pooled
   bound is the gate.
4. **K-5/K-6 (cost and latency of a hit)** are measured by t11 on the reference machine; the
   cache itself does no model call and one cosine scan (`ExactCosineIndex`).
5. **Exact search, not ANN.** At thousands of entries exact cosine is faster than an index
   build and fully reproducible; `SimilarityIndex` is the seam for an ANN backend.

---

## 9. Command reference and map

```bash
uv run pytest -q tests/unit/cache                      # AT-CAC-01…09
uv run python -m answerdesk.cache.calibrate --synthetic --out reports/cache-smoke
```

| requirement | code | test |
|---|---|---|
| FR-CAC-01 exact cache | `semantic_cache.get/put` | `test_exact_repeat_hits_without_recomputing` |
| FR-CAC-02 semantic + guards | `keys.py`, `semantic_cache.py` | `test_semantic_hit_above_the_calibrated_threshold`, `test_entity_guard_blocks_a_near_miss`, `test_version_guard_rejects_a_different_live_version` |
| FR-CAC-03 hit re-validation | `revalidate.py` | `test_structural_revalidation_rejects_a_tampered_quote`, `test_source_doc_guard_detects_deleted_and_edited_chunks` |
| FR-CAC-04 whole artifact | `CacheEntry` | type-level (no string field); round-trip via `CacheEntry.model_validate` |
| FR-CAC-05 invalidation | `invalidate`, TTL | `test_corpus_version_bump_invalidates_everything`, `test_document_deletion_invalidates_only_affected_entries`, `test_ttl_expiry_demotes_to_a_miss` |
| FR-CAC-06 high-risk bypass | `is_high_risk` | `test_high_risk_intents_bypass_the_semantic_path` |
| FR-CAC-07 wrong-hit budget | `thresholds.py`, `report.py` | `test_calibrated_table_meets_the_budget_on_the_replay_set` |
| FR-CAC-08 counters/report | `CacheStats`, `report.write_report` | `test_counters_track_hits_misses_bypass_and_wrong_hits`, `test_report_files_are_written` |
| FR-CAC-09 negative cache | `put(negative=…)`, TTL class | `test_negative_entries_are_cached_with_the_escalation_ttl` |
