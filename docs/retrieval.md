# Retrieval: tokenisation, indexes, fusion, query analysis, reranking and ablation

Status: implemented by task **t5**. Requirements: `FR-RET-01…10`, `FR-OPS-02`,
`NFR-LAT-02`, `NFR-ML-02`, `NFR-QUAL-02`; invariants `INV-01`, `INV-08`, `INV-09`, `INV-10`.
Quality targets: `docs/requirements.md` §6.4 (`RT-1…RT-10`). Module map: §13 of this file.

This document derives the maths the code implements, states which numbers are *measured* and
which are *asserted*, and reports the latency and memory footprint of the 3k-chunk demo
corpus. Every claim here is reproducible with a command in §12; anything that is not is
marked as a limitation in §11.

---

## 1. What this layer is, and what it refuses to be

The layer turns a natural-language support question into a ranked, traceable list of chunks:

```
query ──▶ analysis ──▶ ┌ sparse probe (expanded terms)
                       ├ sparse probe (identifier must-match)  ──▶ fusion ──▶ dedupe ──▶ [rerank] ──▶ [mmr] ──▶ pack
                       └ dense probe (untouched query)
```

Design commitments, in the order they constrain the code:

1. **Exactness over approximation.** The dense index is exact cosine over a float32 matrix
   (§4). At the operating point that costs milliseconds; an ANN index would cost recall we
   cannot explain per query. `AT-RET-02` compares the fast path against a brute-force oracle.
2. **Determinism as a feature.** Same corpus, same query, same ranking and scores — in any
   process (`AT-RET-01`). Ties break on `chunk_id`, never on dict iteration order; the
   BM25 top-k is taken from the *full* score vector, not from the library's internal heap.
3. **Nothing destructive.** Query expansion only *adds* candidate terms; the raw normalised
   query stays the dense arm and identifiers are never stemmed, transliterated or translated
   (`FR-RET-06`, `FR-RET-09`, `NFR-ML-02`). Document text is never rewritten (`INV-01`).
4. **Records are data.** Chunk text is fed to the rankers and to fusion as *text*, never as
   instructions; no ranking, filtering or routing decision is derived from commands inside a
   chunk. Injection defence lives in generation (t6), but this layer already treats chunk
   content as untrusted input.
5. **Measured, not asserted.** Fusion parameters are fitted on the dev split; the reported
   table is the test split; the CI numbers say out loud that they are a machinery check
   (`INV-09`, `INV-10`, §10).

---

## 2. Tokenisation (`answerdesk.index.text`)

One function, `tokenize(text)`, serves **both** the corpus and the query. A BM25 score is
meaningless if the two sides disagree about what a term is, so there is no second path.

**Word segmentation.** `[^\W_]+` runs over NFKC-normalised text, lower-cased.

* Latin, Cyrillic, Greek, Hebrew… → whitespace-delimited word runs.
* Han, Kana, Hangul, Thai (no inter-word spaces) → **one token per character plus character
  bigrams**, on both sides. A whole Han run as a single token would never match a query; a
  segmenter would be a model dependency the base install does not have (`NFR-PORT-03`).
  Unigrams give recall, bigrams give precision (`AT-RET-06`).

**Identifier preservation (`FR-RET-01`, `FR-ING-05`).** Every text is also passed through
`answerdesk.ingest.canonical.canonicalize`, and its canonical tokens are appended:

| in the text | appended canonical token |
|---|---|
| `ORD-2026-10009`, `err-trf-4012` | `ORD-2026-10009` (upper-cased) |
| `5 марта 2026`, `05.03.2026`, `3月5日` | `2026-03-05` |
| `12,5 €`, `€12.50`, `1200 руб.` | `12.50 EUR`, `1200.00 RUB` |
| `4000 0000 0000 0002` | `4000000000000002` + `****0002` |

Canonicalisation is **additive**: the raw token stays, the canonical token is extra evidence.
That is why `ts-999` and `TS-999` retrieve the same chunk, and why a money amount written
three ways in three locales still matches one chunk.

**No stemming — a stated limitation.** `карта` and `карты` are different terms (§11). The
consequence is visible in `test_vocabulary_is_built_once_and_exposes_corpus_terms`, which
asserts exactly that. Typo/inflection tolerance is handled at *query* time by additive
expansion (§5.4), not by mutilating the index.

**The indexed representation.** `ChunkStore.text_for(chunk)` = `breadcrumb` + optional
`context_header` + `chunk.text`. The breadcrumb (`Product > Billing > Refunds`) is included
because support questions usually name the section; the chunk's own text is untouched. The
choice is recorded in the index fingerprint as `representation=chunk-text-with-breadcrumb-v1`,
so a change to it invalidates the index instead of silently changing scores.

---

## 3. Sparse retrieval (`answerdesk.index.bm25_index`)

### 3.1 Scoring

BM25 with the Lucene variant, as implemented by `bm25s` (`k1 = 1.5`, `b = 0.75`):

```
idf(t)    = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
score(q,d)= Σ_{t ∈ q} idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
```

`k1` controls term-frequency saturation, `b` the length normalisation. Both are frozen in
`BM25Params` and serialised with the index: changing either changes the index fingerprint
(`compute_index_version`), so a stale index can never masquerade as a fresh one.

### 3.2 Deterministic top-k

The library returns its own `k` best documents, and ties at the cut-off are resolved by
whatever order its heap produced. This index instead requests the **full score vector**
(`BM25.get_scores`, one sparse matvec) and selects with

```
order = argsort(−scores, kind="stable")     # ties → ascending corpus position
take the first k with score > 0
```

Two consequences worth stating: chunks that share no term with the query are never returned
(a zero-score chunk would corrupt fusion ranks), and ties are reproducible across processes
and platforms (`AT-RET-01`).

### 3.3 The identifier must-match probe

`search_terms(..., required=[...])` keeps only chunks that contain **every** required
canonical token, then ranks normally. The pipeline issues the sparse channel twice — the
expanded query, and the identifier-restricted variant — and merges them by best rank
(`merge_probes`, ties on `chunk_id`). That is the mechanism behind RT-8 and D-07: exact
identifiers are the documented dense failure mode, so they get a lexical must-match arm that
cannot be outvoted.

### 3.4 Persistence

The index persists **tokens**, not library internals (`tokens.jsonl` + `bm25.json`), and
rebuilds postings on load. A `bm25s` upgrade therefore cannot change the on-disk format, and
a rebuild is byte-reproducible. Load asserts that the stored chunk ids equal the corpus chunk
ids and refuses otherwise (`IndexMissingError`).

---

## 4. Dense retrieval (`answerdesk.index.vector_index`)

Exact cosine over pre-normalised rows:

```
d̂ = d / ‖d‖₂ ,  q̂ = q / ‖q‖₂ ,  s(q,d) = q̂ · d̂ ∈ [−1, 1]
```

so a query costs one matrix–vector product: `O(N·D)` with `N` chunks and `D` dimensions. For
the demo corpus that is 3,110 × 64 ≈ 199k multiply-adds per query — well under a millisecond
(§9).

* Zero rows (empty text) stay zero, so the score is `0` rather than `nan`.
* Non-positive cosines are dropped by default: an orthogonal or opposed chunk is not
  evidence. `search_vector(..., positive_only=False)` exposes the raw ranking, which is what
  the exactness test compares against `brute_force_top_k` (a plain Python loop — slow on
  purpose, it is the oracle).
* The embedder is injected (`Embedder` protocol). CI uses `HashingEmbedder` (BLAKE2b hashed
  bag of features, deterministic, no weights); `local` mode uses sentence-transformers; the
  `api` backend uses a hosted OpenAI-compatible endpoint such as NeuralDeep's `bge-m3`
  (§14). The embedder identity (name + dimension) is part of the index fingerprint, and
  loading an index built by a different embedder is refused.
* The optional Qdrant adapter (`answerdesk.index.qdrant_index`, `qdrant` extra) implements
  the same `Retriever` shape with a lazy import; without the extra it raises a `ConfigError`
  that names the command to install it. It is not the default and it is not needed for
  `NFR-SCALE-01`.

---

## 5. Query analysis (`answerdesk.retrieval.query_analysis`)

`QueryAnalyzer.analyze(query)` produces a `QueryAnalysis` record that the retrievers consume.

### 5.1 Normalisation

NFKC + whitespace folding. **Case is preserved** in `normalized`, because identifier
fingerprints must stay byte-identical (`FR-RET-09`); lower-casing happens inside
tokenisation, where it belongs.

### 5.2 Language identification

A cue-based detector over the six shipped locales plus `en` and `und`: function-word hits
(weight 1.0) + distinctive-letter hits (weight 1.5). `ы` is deliberately **not** a Russian cue
— Kyrgyz writes it, and including it made Kyrgyz queries look Russian.

Measured on the checked-in query set (720 items, 6 locales × 120): **100% accuracy** overall
and per locale after the cue tuning. The test asserts ≥ 0.90 overall and ≥ 0.85 per locale,
so it fails on a regression rather than on one ambiguous item. Text without evidence returns
`("und", 0.0)` — the detector reports ignorance instead of guessing, and the pipeline then
skips locale-specific handling.

### 5.3 Identifier protection

`identifier_tokens(text)` = canonical tokens containing a digit. They are (a) appended to the
lexical terms, (b) lower-cased and used as the must-match probe input, and (c) reported in
the trace (`RetrievalTrace.identifier_tokens`). Fingerprint fixtures assert that `TS-999`,
`ORD-2026-10009`, `12.50 EUR`, `2026-03-05` and 16-digit card strings survive analysis
unchanged.

### 5.4 Additive typo and layout repair

For a term that the corpus vocabulary does not contain and that is at least 5 characters long:

1. **Keyboard-layout repair** — if the other-layout spelling exists in the corpus
   vocabulary, add it (`ghbdtn` → `привет`). It is offered, never applied.
2. **Character-trigram fuzzy match** — candidates by trigram Jaccard ≥ `fuzzy_threshold`
   (default 0.6), best first, ties on the term. Covers typos *and* mild inflection
   (`обслуживани` → `обслуживание`).

Both are bounded by `max_expansions` (default 3) and are recorded in `QueryAnalysis.notes`.
The original term always remains in `lexical_terms`, and `dense_text` is always the
unexpanded normalised query — so an expansion can never change what the dense arm means.
Multi-query expansion exists (`multi_query=True`) but is **off by default**: it triples the
dense cost for an unmeasured gain, and the ablation in §10 is where that would be justified.

---

## 6. Fusion (`answerdesk.retrieval.fusion`)

Two fusers implement the `FusionStrategy` protocol; a third (`SingleChannelFusion`) is the
identity for one-channel arms.

### 6.1 Reciprocal Rank Fusion

```
RRF(d) = Σ_{r ∈ R} 1 / (k_rrf + rank_r(d)) ,        rank starts at 1,  k_rrf = 60
```

Properties that matter here: it uses only ranks, so it needs no score calibration and no
corpus; and it therefore throws score magnitude away, which is why a *bad* channel can only
dilute a good one (see the measured example in §10.3). Bruch et al. (arXiv:2210.11934) show
RRF is parameter-sensitive and loses to a tuned convex combination in and out of domain — so
RRF ships as the untuned cold start (`D-03`) and CC is the tuned path.

Hand-computed fixture used by `test_rrf_matches_the_hand_computed_formula` (k=60,
A = [c1, c2, c3], B = [c3, c1]):

| chunk | ranks | RRF | 
|---|---|---|
| c1 | 1 in A, 2 in B | 1/61 + 1/62 = 0.03252247 |
| c3 | 3 in A, 1 in B | 1/63 + 1/61 = 0.03226646 |
| c2 | 2 in A | 1/62 = 0.01612903 |

`k_rrf` is tunable and *does* change orderings: the test constructs a case where `k=1` ties
three candidates while `k=60` gives `y` an outright win (`test_rrf_rank_constant_changes_the_ordering`).

### 6.2 Convex combination

```
CC(d) = Σ_{r ∈ R} w_r · ŝ_r(d) ,   w_r ≥ 0 ,  Σ_r w_r = 1
```

Default weights: `w_dense = α`, `w_bm25 = 1 − α` (α = 0.75, the Weaviate convention), or
explicit per-channel weights from a fitted artifact. Weights are renormalised over the
channels actually present.

**Normalisers.** Two, both per channel and per query:

```
minmax:  ŝ = (s − min s) / (max s − min s)      ∈ [0, 1]
zscore:  ŝ = (s − μ_s) / σ_s                     unbounded
```

*`minmax` is the default and it is **query-relative**: the same chunk gets a different
normalised score depending on the rest of that query's candidate list, and one huge outlier
compresses everything else. This is the documented pitfall, not an accident — it is why the
fitted `α` from one corpus does not transfer to another and why the artifact records the
corpus hash. `zscore` keeps spacing under outliers but is unbounded, so a channel with a long
tail can exceed its nominal weight.*

Hand-computed fixture (`test_convex_combination_matches_the_hand_computed_formula`, α = 0.75):

| chunk | BM25 raw → minmax | DENSE raw → minmax | CC |
|---|---|---|---|
| c0 | 10 → 1.0 | 0.9 → 1.0 | 0.25·1.0 + 0.75·1.0 = **1.0** |
| c1 | 5 → 0.4444 | 0.2 → 0.0 | 0.25·0.4444 + 0 = **0.111111** |
| c2 | 1 → 0.0 | — | 0.25·0 = **0.0** |

### 6.3 Degenerate score distributions

A channel whose scores are all equal carries no ranking information. The rule is explicit:

| situation | normalised value | rationale |
|---|---|---|
| flat vector, **n > 1** | all `0.0`, channel listed in `degenerate_channels` | mapping to `1.0` would let a useless channel dominate every fused ranking |
| **n = 1** | `1.0` | that single chunk *is* the channel's whole evidence |

This asymmetry is the answer to `FR-RET-04`'s "flat cosines" fixture: with a flat dense
channel, `test_flat_dense_scores_do_not_let_the_dense_channel_dominate` shows the fused order
equal to the sparse order and the dense contribution exactly `0.0`.

### 6.4 Determinism and channel attribution

Fused passages are sorted by `(−score, chunk_id)`, truncated to `k`, and re-ranked 1..n. Each
passage reports the channel that contributed most to its score, plus a `contributions` map
(`{"bm25": …, "dense": …}`) so a trace still says where a candidate came from (`FR-RET-08`).

---

## 7. Ranking metrics (`answerdesk.retrieval.metrics`)

Transcribed from requirements §6.1, with the edge cases specified:

```
recall_any@k = (1/|Q|) Σ_q 1[ G_q ∩ top_k(q) ≠ ∅ ]
recall_all@k = (1/|Q|) Σ_q |G_q ∩ top_k(q)| / |G_q|
nDCG@k       = (1/|Q|) Σ_q DCG_q@k / IDCG_q@k
               DCG_q@k  = Σ_{i=1..k} rel_i / log2(i+1) ,  rel_i ∈ {0,1}
               IDCG_q@k = Σ_{i=1..min(k,|G_q|)} 1 / log2(i+1)
MRR@k        = (1/|Q|) Σ_q 1/rank_q(first gold)   if found within k, else 0
```

* **Empty gold scores 0 on every metric.** An unanswerable item must never be counted as a
  hit (§6.1); the retrieval eval harness only reports answerable items, but the function must
  not silently reward a system that returns everything.
* **`k` beyond the ranking** uses the ranking that exists (§6.1 says nothing about padding).
* **`IDCG` uses `min(k, |G_q|)`**, so at `k=1` with two gold chunks a gold at rank 1 is a
  perfect score — the definition, not a bug (`test_ndcg_matches_the_hand_computed_dcg_over_idcg`).
* **Aggregation** is per-locale first, then macro-average with equal weights, so a
  well-resourced locale cannot mask a broken one (§6.3 step 4, `NFR-ML-01`).
* **Uncertainty**: paired bootstrap over per-query vectors (`B` configurable, seed 1337) for
  "arm A beats arm B" claims; Wilson interval for proportions; exact McNemar for paired
  binary outcomes. A retrieval table without error bars is decoration (§6.3 step 5).

---

## 8. Fitting fusion parameters (`answerdesk.retrieval.fit`)

`INV-09` says fusion parameters are only ever fitted on the dev split. The module enforces it
rather than promising it:

1. `assert_disjoint(dev_ids, test_ids)` raises `LeakageError` if any item id appears on both
   sides — before any evaluation happens;
2. `fit_fusion_artifact` re-checks that the recorded split hashes equal `sha256(sorted ids)`
   of the data actually passed in, so an artifact cannot claim provenance it did not have;
3. the objective is a caller-supplied `evaluate(params) -> float`, and the fitting code has no
   other data source — the runner passes a dev-only closure.

Two procedures (FR-RET-05):

**Coordinate search (default).** Grid over `rrf_k ∈ {10, 20, 40, 60, 80, 100}` and
`α ∈ {0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9}`, objective `nDCG@10` on dev. Best per family,
then the better family. Tie-break: **convex combination wins ties**, because the requirements
ship tuned CC as the path (`D-03`); a tie means RRF bought nothing. The full grid is stored
in `search_history`, so a flat optimum is visible instead of hidden.

**Logistic regression** (`--method logistic_regression`). Feature vector = per-channel
normalised scores for one candidate, label = "this candidate is gold".
`P(gold|x) = σ(b + w·x)`, L2-regularised, 400 gradient-descent steps from zero initialisation
(numpy only, no scikit-learn, no stochasticity). The fitted `w` (clipped at 0, renormalised to
sum 1) becomes `FusionParams.weights`; `b` is fitted but never weighted.

The **intercept** is not cosmetic: roughly 8 of 50 candidates are gold, so a model without a
bias term can only reach that base rate by making every feature weight negative — which the
non-negativity clip turns into the equal-weight fallback, silently discarding the signal. With
the intercept, this run's logistic fit returns `w_bm25 = 1.0`, `w_dense = 0.0` on the hashing
embedder (dev `nDCG@10 = 0.1124` vs `0.1133` for the coordinate search), i.e. it independently
reaches the same conclusion as the grid search: the noise channel deserves no weight.

The persisted artifact (`data/index/ablation/fusion.json`) is strict JSON:

| field | meaning |
|---|---|
| `artifact_id` | `fusion-<method>-<fuser>-<params hash>` |
| `method` | `coordinate_search` \| `logistic_regression` |
| `metric`, `metric_value` | objective and its dev value |
| `params` | `fuser`, `rrf_k`, `alpha`, `normalization`, `weights` |
| `split` | dev: name, item hash, `n_items`, `n_documents`, salt |
| `excluded_split` | test: same shape — the explicit "never fitted on this" record |
| `corpus` | corpus version + chunk hash |
| `embedder` | name + dimension |
| `query_analysis` | analyser version (`qa-v2`) |
| `seed`, `fitted_at`, `notes`, `search_history` | reproducibility |

`FusionArtifact.fingerprint()` hashes everything **except** `fitted_at`, so two runs on the
same data produce the same fingerprint (NFR-DET-02) while the timestamp still records when it
happened. Loading is `extra="forbid"`: a hand-edited artifact that gains a field is rejected.

---

## 9. The pipeline (`answerdesk.retrieval.pipeline`)

| stage | what it does | budget (default) |
|---|---|---|
| analysis | NFKC, language, identifiers, expansion | — |
| sparse | expanded-query probe + identifier must-match probe, merged | `top_k = 50` |
| dense | exact cosine over the untouched normalised query (+ optional multi-query probes) | `top_k = 50` |
| fusion | RRF or CC over the channels present | `top_k = 50` |
| dedupe | greedy cosine (or token-Jaccard) near-duplicate removal | `dedupe_cosine = 0.95` |
| rerank (optional) | cross-encoder or the deterministic lexical stand-in | `top_in = 50 → top_out = 12` |
| mmr (optional) | maximal marginal relevance | `final_k = 12` |
| pack | truncate, rank 1..n, `stage="pack"` | `final_k = min(12, …)` |

**Trace.** `RetrievalTrace` (frozen contract) carries `candidates` (both channels with their
per-channel ranks), `fused`, `reranked`, `packed`, `fusion` (`FusionInfo`: fuser + parameters),
`dropped_duplicates`, `identifier_tokens`, `normalized_query`, `language`. Timings are
returned separately (`retrieve_with_timings`) because the contract has no timing field; keys
are `analysis_ms`, `sparse_ms`, `dense_ms`, `fusion_ms`, `dedupe_ms`, optional `rerank_ms` /
`mmr_ms`, and `total_ms`.

**Seam.** `RetrievalPipeline` implements `Retriever` (`search(query, k) -> list[RetrievedPassage]`),
so the service (t9), the cache (t8) and the evaluation harness (t7) depend on "a retriever",
not on this class. `build_pipeline(settings, index)` resolves configuration into a pipeline.

**Dedupe.** Greedy in rank order: keep a passage unless something already kept is ≥ threshold
similar. Cosine over the dense index vectors when available (the same geometry the retriever
used), token Jaccard otherwise. The backend is chosen once for the whole call — mixing them
mid-pass would make the result depend on which passage happened to be missing a vector.

**MMR.**

```
MMR(i) = λ · rel(i) − (1 − λ) · max_{j ∈ S} sim(i, j)
```

with `rel` the min–max normalised fused score and `λ = 0.7` by default; `λ = 1` is pure
relevance, `λ = 0` is pure diversity. Similarity is the dense cosine when available, token
Jaccard otherwise, and **cached per pair**: without that cache the selection is
`O(k·n²)` text tokenisations, which on the demo corpus is the difference between a 2.5 s and a
150 s arm (measured while building this module). Ties break on `chunk_id`.

**Why MMR is off by default.** Diversification trades first-hit quality for coverage; on
support data the answer usually lives in one article. It is an ablation arm (§10), not a
default.

---

## 10. Ablations

### 10.1 How to run them

```bash
uv run answerdesk ingest --demo                    # corpus + 720 labelled queries
uv run answerdesk index --demo                     # BM25 + exact dense index
uv run python -m answerdesk.retrieval.ablation --bootstrap 1000
```

Flags: `--corpus DIR`, `--out DIR`, `--arms configs/retrieval/ablations.yaml`, `--no-fit`
(use cold-start parameters), `--method coordinate_search|logistic_regression`, `--bootstrap N`
(paired CI on RT-7), `--latency-sample N`.

Outputs, under `data/index/ablation/` by default: `fusion.json` (the fitted artifact),
`retrieval-ablation.json` (all numbers + provenance), `retrieval-metrics.md` (the table
below). The runner prints the table and exits 0.

### 10.2 The arms

| arm | channels | fusion | extra stage |
|---|---|---|---|
| `bm25` | sparse | none | — |
| `dense` | dense | none | — |
| `hybrid-rrf` | both | RRF, `k` fitted on dev | — |
| `hybrid-convex` | both | CC, `α` fitted on dev (shipped path) | — |
| `hybrid-convex+rerank` | both | CC | reranker (offline: `lexical-overlap-v1`) |
| `hybrid-convex+mmr` | both | CC | MMR (λ = 0.7) |

### 10.3 A recorded run, and how to read it

Recorded on 2026-10-21 from a **hashing-64** embedder, corpus `corpus-55d57ee3c986ff8d`
(360 documents / 3,110 chunks), test split 346 queries / 173 documents:

| arm | recall_any@20 | strict recall@20 | recall_any@10 | recall_all@20 | nDCG@10 | MRR@10 | id recall@20 | perturb. Δ | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.991 | 0.939 | 0.988 | 0.276 | 0.222 | 0.462 | 1.000 (n=85) | −0.017 | 1.0 | 2.7 |
| dense | 0.564 | 0.445 | 0.462 | 0.085 | 0.065 | 0.192 | 0.518 (n=85) | −0.204 | 0.4 | 0.4 |
| hybrid-rrf | 0.965 | 0.841 | 0.786 | 0.221 | 0.139 | 0.376 | 1.000 (n=85) | −0.068 | 1.4 | 3.1 |
| hybrid-convex (α = 0.25 fitted) | 0.991 | 0.939 | 0.988 | 0.244 | 0.144 | 0.412 | 1.000 (n=85) | −0.017 | 1.4 | 3.1 |
| hybrid-convex+rerank | 0.983 | 0.902 | 0.983 | 0.269 | 0.246 | 0.538 | 1.000 (n=85) | −0.051 | 12.9 | 16.4 |
| hybrid-convex+mmr | 0.991 | 0.939 | 0.988 | 0.238 | 0.177 | 0.427 | 1.000 (n=85) | −0.017 | 2.5 | 4.2 |

RT-7 uplift (best hybrid − best single channel, `recall_any@20`): **−0.026** (95% CI available
with `--bootstrap 1000`); on the strict label **−0.098**.

The two latency columns come from a 20-query subsample of the test split
(`--latency-sample`) and move by ±1 ms between runs; §12 is the larger, stable measurement.

**These are machinery numbers, not a quality claim** (§6.3 step 2), and they must be read as
such:

* The `hashing` embedder is a deterministic hashed bag of features. Its dense channel is
  near-noise (`recall_any@20 = 0.564`, identifier recall `0.518` — it fails exactly where the
  research says dense retrieval fails), so fusing it *into* a working BM25 channel can only
  dilute: hence the negative RT-7 uplift. The fit agrees: `α = 0.25` is the **lower edge of the
  search grid**, i.e. the optimum wants the dense weight to be even smaller than the grid
  allows. On a real multilingual embedder that boundary behaviour would be the signal to widen
  the grid or to rethink the channel, which is exactly why the artifact records the whole
  `search_history`.
* `recall_any@20 ≈ 0.99` for BM25 is the corpus's redundancy, not a triumph: the synthetic
  articles share boilerplate, and the gold label is *document-level*, so any chunk of the right
  article counts.
* The identifier arm is the load-bearing evidence here: sparse `1.000` vs dense `0.518` with
  `n = 85` is precisely why the lexical must-match probe exists (`D-07`, RT-8).
* The rerank arm is the one arm that clearly earns its cost on this data: `nDCG@10` 0.144 →
  0.246 (+0.10), `MRR@10` 0.412 → 0.538, `recall_all@20` 0.244 → 0.269, for ~9× latency. That
  is the shape RT-10 predicts, measured with a *lexical stand-in*, so the real cross-encoder
  arm (t11, `local` extra) is the one that decides RT-10.
* `nDCG@10` between 0.06 and 0.25 looks alarming and is a labelling artefact: with ~8 gold
  chunks per query and `IDCG` built from `min(k, |G|)`, the ideal list fills all 10 slots, so a
  realistic ranking cannot approach 1.0. The golden set (t7) uses single-chunk labels; a
  cross-corpus nDCG comparison is meaningless.

**What would falsify the design:** a *real* multilingual embedder making
`recall_any@20(dense) ≥ recall_any@20(bm25)` **and** RT-7 failing (hybrid ≤ best channel) on the
frozen test split. That is the falsification clause of §6.4: the shipped default would then be
the single best channel or RRF, and the ADR would record it. Nothing in this table supports
that conclusion, because the dense channel here is not a real one. `RT-1…RT-6` and `RT-10` are
G-REL gates measured by t11 on `golden/v1` with a real embedder; this runner exists so the
machinery behind them is not a black box.

---

## 11. Limitations (stated, not hidden)

1. **No stemming / lemmatisation.** `карта` ≠ `карты` for BM25; inflection is compensated only
   by query-side trigram expansion. A stemmer is a legitimate future ablation, not a silent
   default.
2. **`minmax` normalisation is query-relative.** A single outlier compresses the rest of the
   list; a fitted `α` does not transfer across corpora. `zscore` is implemented and tested but
   is not the default.
3. **Language ID is a heuristic.** It is measured at 100% on the checked-in query set and
   returns `und` when it has no evidence, but it is not a statistical classifier and must not
   be treated as ground truth downstream.
4. **Trigram expansion replaces nothing.** A query whose only content word is misspelled
   *and* absent from the vocabulary by more than the trigram threshold will not match. Recall
   for adversarial typos is an assumption, not a demonstrated result — the same gap the
   research flagged (RT-9, Q6).
5. **The offline reranker is not a cross-encoder.** `lexical-overlap-v1` exercises the
   machinery (budgets, batching, stage bookkeeping, determinism); RT-10 evidence requires the
   `local` extra.
6. **The ablation's gold label is document-level** and derived from the generator's source
   document. The strict (answer-chunk) label is a *heuristic* — the chunk with maximal overlap
   with the generator's operator reply — and is reported separately, never mixed in.
7. **Numbers in §10.3 are corpus- and time-specific.** The corpus is generated (`data/corpus`
   is derived and git-ignored); re-running the ablation after a corpus change will produce
   different numbers with the same structure. The authoritative numbers of any run live in
   that run's `retrieval-ablation.json`.
8. **No ANN, no sharding.** Exact search is a deliberate choice at ≤ 20k chunks
   (`NFR-SCALE-01`); beyond that, memory and latency grow linearly and the Qdrant adapter
   becomes the escape hatch.

---

## 12. Latency and memory at the demo corpus (3k chunks)

Measured on the reference machine (Windows 11, Python 3.12, CPU only, no GPU), corpus
`corpus-55d57ee3c986ff8d`: **3,110 chunks / 360 documents / 221,868 tokens / 5,005-term
vocabulary**, `hashing-64` embedder.

The corpus is generated and is regenerated whenever ingestion changes (t4, t15), so treat
these as a recorded snapshot rather than constants. Per-query cost is linear in the chunk
count (one sparse matvec and one dense matvec); at the 5,739-chunk corpus current at the time
of writing the same pipeline runs roughly 2× slower and remains an order of magnitude inside
`NFR-LAT-02`'s p95 ≤ 200 ms. `NFR-LAT-02` asks for p95 ≤ 200 ms at the demo corpus;
every number below is at least an order of magnitude inside that budget *with* the optional
stages on.

**Build** (`uv run answerdesk index --demo`):

| stage | measured |
|---|---|
| BM25 build (tokenise 3,110 chunks + postings) | 747 ms |
| dense build (3,110 embed calls in batches of 32, then normalise) | 250 ms |
| total (incl. chunk-set fingerprint + serialisation) | **≈ 1.0 s** |

**Per-query retrieval** (60 test-split queries, `top_k = 50`, `final_k = 12`):

| configuration | p50 | p95 |
|---|---|---|
| sparse only | 1.07 ms | 2.92 ms |
| dense only | 0.37 ms | 0.57 ms |
| hybrid-RRF | 1.37 ms | 3.16 ms |
| hybrid-CC | 1.40 ms | 4.29 ms |
| hybrid-CC + lexical rerank | 12.85 ms | 16.00 ms |
| hybrid-CC + MMR | 2.48 ms | 4.32 ms |

The reranker dominates whenever it is on: it re-tokenises 50 candidates per query in the
offline stand-in, and a real cross-encoder on CPU is *slower* still — which is why the rerank
window is bounded by `top_in` and why `NFR-LAT-01` measures end-to-end latency, not retrieval
alone. MMR costs ~1 ms over fusion because its similarity matrix is cached per pair;
without that cache the same arm takes ~2 minutes (§9).

**Memory** (resident, in-process):

| structure | size |
|---|---|
| dense matrix, 3,110 × 64 × float32 | 796,160 B ≈ 0.76 MiB |
| BM25 token lists (221,868 tokens as Python strings) | ~13 MiB (dominant cost) |
| vocabulary + trigram index (5,005 terms) | ~1.7 MiB |
| **index directory on disk** (`chunks.jsonl` + tokens + vectors + manifest) | **7.4 MiB** |

At the operating point (≥ 20,000 chunks) the matrix grows to ~5 MiB for a 64-dimensional
embedder or ~80 MiB for BGE-M3's 1,024 dimensions — still small next to the token lists, which
is why `NFR-SCALE-01`'s 4 GB peak-RSS bound is not at risk from *this* layer.

---

## 13. Module map and traceability

| requirement | module | acceptance test |
|---|---|---|
| FR-RET-01 sparse retrieval, determinism, identifiers | `index/text.py`, `index/bm25_index.py` | `AT-RET-01` (`tests/unit/index/test_bm25.py`) |
| FR-RET-02 exact dense retrieval | `index/vector_index.py` | `AT-RET-02` (`tests/unit/index/test_vector.py`) |
| FR-RET-03 one `Retriever` protocol | both index modules, `retrieval/pipeline.py` | `AT-RET-03` |
| FR-RET-04 RRF + CC + normalisers + degenerate case | `retrieval/fusion.py` | `AT-RET-04` (`tests/unit/retrieval/test_fusion.py`) |
| FR-RET-05 dev-only fitting + artifact + leakage guard | `retrieval/fit.py`, `configs/retrieval/` | `AT-RET-05` (`test_fit.py`, `test_ablation.py`) |
| FR-RET-06 query analysis | `retrieval/query_analysis.py` | `AT-RET-06` (`test_query_analysis.py`) |
| FR-RET-07 rerank + MMR behind `local` | `rerank/cross_encoder.py`, `rerank/mmr.py` | `AT-RET-07` (`test_rerank_mmr.py`) |
| FR-RET-08 full trace | `retrieval/pipeline.py` | `AT-RET-08` (`test_retrieval_pipeline.py`) |
| FR-RET-09 no translation / identifier fingerprints | `retrieval/query_analysis.py` | `AT-RET-09` |
| FR-RET-10 quality targets | `retrieval/ablation.py` + t7/t11 | `AT-RET-10` (G-REL, §10.3 machinery) |
| FR-OPS-02 idempotent `index` + corpus version | `index/store.py`, `index/cli.py` | `AT-OPS-02` (`tests/unit/index/test_store.py`, `tests/unit/test_cli.py`) |
| NFR-LAT-02 retrieval latency | pipeline timings, §12 | `AT-LAT-02` |
| NFR-ML-02 no machine translation | `retrieval/query_analysis.py` | `AT-ML-02` |
| NFR-QUAL-02 coverage of `retrieval/` | `tests/unit/retrieval` | `AT-QUAL-02` |
| INV-09 fusion fitted on dev only | `retrieval/fit.py` | `AT-RET-05` |

| file | role |
|---|---|
| `index/text.py` | tokenisation, canonical tokens, `TermVocabulary` (fuzzy + layout) |
| `index/bm25_index.py` | BM25 index, deterministic top-k, must-match probe, persistence |
| `index/vector_index.py` | exact cosine index, brute-force oracle, persistence |
| `index/qdrant_index.py` | optional Qdrant adapter (`qdrant` extra, lazy import) |
| `index/embedders.py` | backend → `Embedder` (hashing / sentence-transformers / hosted API) |
| `index/store.py` | chunk store, index fingerprint, build/reuse, load |
| `index/cli.py` | `answerdesk index` |
| `retrieval/query_analysis.py` | normalisation, language ID, identifiers, expansion |
| `retrieval/fusion.py` | RRF, convex combination, normalisers, degenerate handling |
| `retrieval/fit.py` | coordinate search, logistic fit, artifact, leakage guard |
| `retrieval/pipeline.py` | the stage pipeline + trace + timings |
| `retrieval/dedupe.py` | greedy near-duplicate removal |
| `retrieval/metrics.py` | recall/nDCG/MRR, bootstrap, Wilson, McNemar |
| `retrieval/splits.py` | document-disjoint hashed split |
| `retrieval/datasets.py` | retrieval cases from the corpus query set |
| `retrieval/ablation.py` | the six-arm runner and its report |
| `rerank/cross_encoder.py` | lazy cross-encoder + deterministic offline stand-in |
| `rerank/api.py` | hosted Cohere-style `/rerank` adapter (bge-reranker) |
| `rerank/mmr.py` | MMR with a cached similarity matrix |
| `configs/retrieval/ablations.yaml` | the arm definitions |
| `configs/retrieval/fusion.yaml` | cold start, fitting grids, artifact paths |

---

## 14. Hosted API mode: bge-m3 embeddings and bge-reranker

Real multilingual vectors are the difference between measuring retrieval and measuring a
hash function. Downloading a multi-gigabyte model is a bad first run (and impossible in
this CI), so the layer also speaks to a hosted endpoint. Both adapters are the *same*
protocol seam as the local ones — `Embedder` and `Reranker` — so nothing downstream knows
the difference (`FR-RET-02`, `FR-RET-07`, `NFR-PORT-01`).

### 14.1 Embeddings — `POST {base_url}/embeddings`

OpenAI-compatible wire format:

```http
POST https://api.neuraldeep.ru/v1/embeddings
Authorization: Bearer $OPENAI_COMPATIBLE_API_KEY
{"model": "bge-m3", "input": ["...", "..."]}

200 {"data": [{"index": 0, "embedding": [0.013, ...]}, ...]}
```

| behaviour | implementation |
|---|---|
| batching | `embedding.batch_size` inputs per request (default 32); batches are sequential and ordered |
| result order | re-sorted by the response's own `index` when it is a permutation, so a server that answers out of order cannot misalign vectors and chunks |
| dimension | `embedding.api_dimension` declares it; `null` discovers it from the first response and pins it. A later change is a `ConfigError` |
| retries | transport errors, 429 and 5xx, with exponential backoff (`0.5 s · 2^n`, capped at 8 s) **plus jitter**; `Retry-After` is honoured as a floor |
| timeouts | `embedding.api_timeout_s` (httpx client timeout) |
| failures | 400/401/403/404 and malformed payloads raise `ConfigError` with the fix in the message — never a silent degraded vector space |
| key | `embedding.api_key` (`SecretStr`) or `os.environ[embedding.api_key_env]`; required for a non-loopback host, optional for loopback |
| proxies | `trust_env` stays on, so a real corporate proxy (or the Windows registry proxy) works; an **unusable** ambient proxy is ignored with a warning instead of crashing construction (§14.4.1) |

Verified endpoint: NeuralDeep `https://api.neuraldeep.ru/v1/embeddings` with
`model=bge-m3` returns **1024-dimensional** vectors. Set `api_dimension: 1024` (or leave it
`null` and let the first response pin it — the index fingerprint is written *after* the
dense build precisely so discovery is safe, see §14.4).

### 14.2 Reranking — `POST {base_url}/rerank`

Cohere-compatible wire format, as served by NeuralDeep's `bge-reranker`:

```http
POST https://api.neuraldeep.ru/v1/rerank
{"model": "bge-reranker", "query": "...", "documents": ["...", "..."], "top_n": 12}

200 {"results": [{"index": 3, "relevance_score": 0.91}, ...]}
```

* Results are mapped back by the server's `index`, so the score-to-chunk mapping cannot
  drift when the server answers in score order; duplicate or out-of-range indices raise.
* `rerank.batch_size` documents go per request; batches are merged by global index and only
  then sorted by `(-score, chunk_id)` and cut to `top_n` — batching never changes the order.
* `top_in` is applied by the pipeline before the call, `top_out` is the `top_n` argument.
* A response with **no** scored documents raises: an unexecuted stage must not be reported
  as executed in the trace (`FR-RET-08`).

### 14.3 Configuration and credentials

```yaml
embedding:
  backend: api                              # hashing | bge_m3 | qwen3 | api
  api_base_url: https://api.neuraldeep.ru/v1
  api_model: bge-m3                         # falls back to `model`
  api_dimension: 1024                       # null = discover and pin
  api_key: null                             # prefer the env var; SecretStr if set here
  api_key_env: OPENAI_COMPATIBLE_API_KEY
  api_timeout_s: 30.0
  api_max_retries: 3

retrieval:
  rerank:
    enabled: true
    backend: api                            # cross_encoder | lexical | api
    api_base_url: https://api.neuraldeep.ru/v1
    api_model: bge-reranker
    api_key_env: OPENAI_COMPATIBLE_API_KEY
```

Every key exists in `config.py` **and** `configs/default.yaml` (a test asserts the two key
sets are identical), unknown keys are still rejected, and `Settings.redacted()` renders the
key as `**********` — `SecretStr` means the value cannot leak through a dump, a log line or
an error message (`INV-05`).

The `offline` profile forces `embedding.backend=hashing`, so no test, demo or CI job can
reach the network (`INV-08`).

### 14.4 The index fingerprint stays honest

`index_version = H(corpus_version, chunk hash, embedder name+dimension, tokenizer,
BM25 params, dense metric)`. Switching `hashing-64` → `api:bge-m3` (1024) therefore mints a
*new* index instead of silently reusing one built in a different vector space, and
`IndexBundle.load` refuses an index whose stored embedder dimension differs from the
configured one. Because a hosted embedder may only learn its width from the first
response, `IndexBundle.build` reads the embedder identity **after** the dense build — a test
(`test_bundle_fingerprint_uses_the_discovered_dimension`) pins that ordering, and a
dimension of `0` after embedding is a hard error.

### 14.4.1 Proxies: an unusable ambient configuration must not crash retrieval

Both adapters build their client through `answerdesk.index.http.build_http_client`, which
mirrors the chat provider's policy so the whole stack behaves the same way:

1. build `httpx.Client` with `trust_env` **untouched** — a valid `HTTP_PROXY` / `HTTPS_PROXY` /
   `ALL_PROXY` (or the Windows registry proxy) is respected and never silently bypassed;
2. if httpx rejects the ambient configuration — `socks4://…`, or `socks5://…` without the
   optional `socksio` package, raise `ValueError: Unknown scheme for proxy URL` **at
   construction** — log `ignoring unusable ambient proxy configuration for the hosted
   retrieval client` and rebuild the client with `trust_env=False` (go direct);
3. if even the direct client cannot be built, raise `ConfigError` naming the variables to fix.

This was a real defect: the development machine exposes `socks4://127.0.0.1:10808`, and
before the fix `ApiEmbedder(...).embed_query(...)` died with a `ValueError` before the first
request, so hosted retrieval could not start at all. The regression tests
(`tests/unit/index/test_api_proxy.py`, `tests/unit/retrieval/test_reranker_proxy.py`)
monkeypatch that exact environment and drive both adapters through `httpx.MockTransport`, plus
one integration-style case that runs `default_embedder(settings)` end-to-end under the bad
environment. A transport can be injected (`ApiEmbedder(transport=…)`) precisely so proxy
tolerance is testable without a socket.

### 14.5 Running a hosted benchmark

```bash
# one key serves the generator and the retrieval adapters
export OPENAI_COMPATIBLE_API_KEY=...        # never commit it
export ANSWERDESK_PROFILE=api
export ANSWERDESK_EMBEDDING__BACKEND=api
export ANSWERDESK_EMBEDDING__API_BASE_URL=https://api.neuraldeep.ru/v1
export ANSWERDESK_EMBEDDING__API_MODEL=bge-m3
export ANSWERDESK_RETRIEVAL__RERANK__ENABLED=true
export ANSWERDESK_RETRIEVAL__RERANK__BACKEND=api
export ANSWERDESK_RETRIEVAL__RERANK__API_BASE_URL=https://api.neuraldeep.ru/v1
export ANSWERDESK_RETRIEVAL__RERANK__API_MODEL=bge-reranker

uv run answerdesk index --demo --force      # new embedder -> new index version
uv run python -m answerdesk.retrieval.ablation --bootstrap 1000
```

This is the run that can satisfy `RT-1…RT-10` evidence (`§6.3` step 2 forbids hashing
numbers from appearing as a quality claim). Costs and rate limits are the operator's:
the adapters retry 429s, but they do not throttle.

---

## 15. Noisy input: telegraphic, run-on and typo'd messages

The golden set types like a specification; users do not. The owner's three messages are the
specimen this section answers:

```
как займ оформит                                              telegraphic, truncated verb
деньги как получит                                            telegraphic, mangled order
денги первод таджикистан сделал денги нет брату не дошли      run-on, no punctuation, two typos,
                                                              mixed intent (transfer + not arrived)
```

### 15.1 Pipeline

Everything below is **additive** (`FR-RET-06`): the user's own tokens stay in
`primary_terms`, and no stage can remove them.

```
                 ┌─ segmentation ──▶ clauses (bounded, ≤ max_clauses)
normalised query ┼─ focus view ────▶ content terms (function words out, tokens intact)
                 ├─ expansion ─────▶ layout repair │ transliteration │ trigram │ edit distance
                 └─ rewrite (opt) ─▶ cached LLM reformulation, identifier-guarded
```

| stage | mechanism | shipped |
|---|---|---|
| **clause segmentation** | punctuation → clause conjunctions → negation restart → repeated token → balanced long split (§15.2) | computed, *promotion off* (measured, §15.3) |
| **focus view** | content words only, same language filter table as the language detector | **on** (the one promoting probe) |
| **transliteration** | rule-based Latin→Cyrillic (`zaym`→`займ`, `dengi`→`денги`→`деньги`) | on, inside the primary query |
| **trigram expansion** | the t5 path, threshold 0.6 | on, first (highest precision) |
| **edit-distance fallback** | trigram-prefiltered, banded Levenshtein, k=1 (5–7 chars) / k=2 (≥8) | on, last resort |
| **LLM rewrite** | cached, `temperature 0`, identifier-guarded | **off by default** |

Probe promotion is what turns analysis into ranking. The design that shipped:

* the **primary probe** always carries `primary_terms` (user tokens + canonical identifiers +
  layout repairs) — the raw query is therefore always an arm;
* the **identifier must-match probe** merges with no penalty (exact evidence should win);
* the **focus probe** fires only for telegraphic input (`is_telegraphic`: ≤ 5 word tokens, at
  least one function word, at least one content word), and is merged **through a single
  "noise consensus" vote** (`noise_probe_fold`) so several probes cannot each promote their own
  favourite into the fused head;
* **clause** and **expansion promotion** are implemented, config-gated and *off*: measured
  neutral-to-negative (§15.3). Turning one on is a measurement decision, not a preference.

### 15.2 Segmentation, in the order the boundaries are tried

1. **punctuation** — a comma/semicolon/dash is an explicit boundary;
2. **clause-initial conjunction** (`а`, `и`, `но`, `and`, `but`, `ва`, `жана`, `ve`, …) — the
   conjunction is dropped from the following clause;
3. **negation restart** — run-on text often continues with a negation rather than a
   conjunction: `… сделал денги **нет** брату не дошли` splits before `нет`, which is what
   gets the whole message out of the fraud-document region;
4. **repeated content token** — a self-repair (`денги … денги`) marks a restart;
5. **balanced long split** — a ≥ 8-word message with no other boundary is cut near the middle
   at the most function-word-ish position, because two probes for two possible intents beat
   one diluted bag of words.

Each boundary requires at least two words on both sides, so a normal sentence is never
shredded, and the result is capped at `max_clauses` with the remainder appended (never
dropped).

### 15.3 Measurement (frozen corpus `corpus-77d7d081813738a6`)

Reproduce with `uv run python -m answerdesk.retrieval.ablation --bootstrap 1000`
(artifacts in `data/index/ablation/`; the section also lands in the printed table). Both
conditions share the same index, the same fusion parameters and the same arms — only
`QueryAnalysisConfig` differs, so a delta is attributable to the analysis.

| slice | arm | n | baseline recall@20 | noisy recall@20 | Δ@20 | 95% CI | baseline@10 | noisy@10 |
|---|---|---|---|---|---|---|---|---|
| noisy | bm25 | 474 | 0.951 | 0.962 | **+0.011** | [−0.002, +0.023] | 0.905 | 0.897 |
| noisy | dense | 474 | 0.363 | 0.363 | +0.000 | [0.000, 0.000] | 0.304 | 0.304 |
| noisy | hybrid-rrf | 474 | 0.897 | 0.899 | +0.002 | [−0.015, +0.017] | 0.643 | 0.650 |
| clean | bm25 | 600 | 0.967 | 0.968 | +0.002 | [−0.007, +0.010] | 0.917 | 0.918 |
| clean | dense | 600 | 0.475 | 0.475 | +0.000 | [0.000, 0.000] | 0.383 | 0.383 |
| clean | hybrid-rrf | 600 | 0.925 | 0.927 | +0.002 | [−0.003, +0.007] | 0.778 | 0.782 |

Per style (hybrid arm, `recall_any@20`):

| style | n | baseline | noisy | Δ |
|---|---|---|---|---|
| translit (`zaym kak oformit`) | 127 | 0.835 | 0.874 | **+0.039** |
| run_on (`… сделал денги нет …`) | 133 | 0.940 | 0.940 | +0.000 |
| dropped_preposition | 27 | 0.741 | 0.741 | +0.000 |
| telegraphic (`как займ оформит`) | 80 | 0.950 | 0.938 | −0.012 |
| typo_mix | 107 | 0.916 | 0.888 | −0.028 |

**Clean-input gate: PASS.** Every clean arm moves by at most +0.002 and all intervals contain
zero; the contract's threshold is −0.01. This is structural rather than lucky: a clean,
typo-free, multi-word question triggers *no* extra probe, so its candidate list is identical to
the pre-t23 one.

What the numbers say, including the parts that are not flattering:

* **Transliterated queries gain the most (+0.039).** A user typing Russian in a Latin layout
  was previously unreachable; the rule-based table plus the fuzzy fallback resolve it.
* **Aggregate gains are small** (+0.011 BM25, +0.002 hybrid at 20; +0.007 hybrid at 10)
  because the baseline is already near-saturated on document-level labels (0.95 for BM25):
  there is very little headroom to win at k=20 on this corpus. The intervals contain zero, so
  the honest claim is "no regression, small positive point estimate", not "significant".
* **typo_mix loses 2.8 points and telegraphic 1.2.** The corpus's typo generator mangles a
  question whose un-mangled words already retrieve the right article; the extra expansion
  then perturbs a working primary. The focus probe is what helps telegraphic messages
  individually (the owner's `деньги как получит` moved from rank 18 to rank 3) while the
  short-message aggregate slips slightly — both facts are reported rather than smoothed.
* **Three mechanisms were measured and rejected** (all implemented, all one flag away):
  clause promotion (−0.006 @20, −0.005 clean), expansion-as-a-separate-probe (−0.038 @20),
  and demoting probes by a rank offset (no gain, and it *removed* the owner-case improvement
  because a demoted probe can no longer outvote a diluted primary). Trying the loosest
  expansion first was also worse: precision order is trigram → transliteration → edit distance.
* **The owner's cases** (hybrid, `final_k = 20`): `деньги как получит` rank 18 → **3**;
  the run-on message rank 11 → **7**; `как займ оформит` 19 → 19 (it now retrieves a
  *different* loan article first — the corpus has several, and the document-level label is
  one of them).

Latency: the analyser stays at ~0.15 ms per query; the bounded fuzzy stage adds **≤ 0.3 ms**
for a telegraphic message (8 unknown terms × ≤ 24 candidates, banded DP with early exit —
measured, and a unit test fails if 8 terms exceed 5 ms). The rejected full-scan design cost
12 ms per term and is documented here as the reason the pool is capped.

### 15.4 Knobs and the optional rewrite hook

`configs/retrieval/noisy.yaml` (in-scope file, not the service configuration) carries the
analysis stages, the promotion flags and the rewrite settings:

```yaml
analysis: {clause_segmentation: true, focus_probe: true, fuzzy_edit_distance: true,
           transliteration: true, max_expansions: 4, fuzzy_threshold: 0.6}
pipeline: {noisy_probes: true, focus_probe: true, clause_probes: false,
           expansion_probe: false, noise_probe_offset: 0, noise_probe_fold: true}
rewrite:  {enabled: false, provider_preset: fake, prompt_version: query-rewrite-v1,
           cache_entries: 1024}
```

The rewrite hook (`answerdesk.retrieval.rewrite`) is a `ChatProvider` consumer: it asks the
model for a terse reformulation, then **validates** it — a candidate is discarded if it invents
any digit-bearing token or date/money/order-code form, drops an identifier the user typed, is
empty, or is model chatter. Accepted rewrites are cached on `(rewriter name, normalised query)`
and used as one extra probe; the raw query keeps its arm. It is off by default, deterministic
(`temperature 0`, seeded nothing — the provider is the only source of variation, and the cache
removes it on replay), and tested with `FakeProvider` offline.

### 15.5 Honest limits

1. **Word order and dropped prepositions are not recoverable from surface form.** `деньги как
   получит` has no signal saying *which* money; segmentation and the focus view can only
   improve the odds. `dropped_preposition` shows a flat 0.000 delta for exactly this reason.
2. **Multi-intent ambiguity is softened, not solved.** Each clause becomes a probe and fusion
   decides which intent wins; when both intents are equally supported, the context carries
   both and the generator has to disambiguate (t6's clarify path). There is no "ask the user"
   step in retrieval.
3. **Spam and off-topic messages are out of scope.** A message with no domain vocabulary
   retrieves nothing better after noise handling; escalation (t8) is the answer, not retrieval.
4. **Vocabulary is the hard ceiling.** A word that no document contains cannot be matched to
   anything: `займ` was unreachable until the corpus gained loan articles. Noise handling makes
   a *known* domain reachable through bad typing; it cannot invent knowledge.
5. **The numbers are corpus-specific.** Every figure above is from the frozen corpus
   `corpus-77d7d081813738a6` with the hashing embedder (a machinery check, §6.3). A real
   multilingual embedder (t16/t11) will change the dense column and therefore the hybrid one.
6. **`typo_mix` regresses on this corpus** (−0.028) and `telegraphic` slips −0.012 while the
   owner's individual cases improve sharply. Both are reported; a future iteration should
   gate the edit-distance fallback on a *typo-confidence* signal rather than on "unknown term".

---

## 16. Reproduce

```bash
cd answerdesk
uv sync --extra dev

# unit: fusion maths, exact NN, determinism, leakage, query analysis, rerank/MMR
uv run pytest -q tests/unit/retrieval tests/unit/index       # CMD-D

# indexes for the demo corpus (self-healing: builds the corpus if absent)
uv run answerdesk index --demo                               # CMD-E
uv run answerdesk index --demo                               # second run: "unchanged (reused)"

# fit on dev, measure on test, write artifacts + table
uv run python -m answerdesk.retrieval.ablation --bootstrap 1000
```

Expected: 157 tests pass in ~1 s; `index --demo` prints `index: built` then
`index: unchanged (reused)` with identical `index_version`; the ablation prints the §10.3
table and writes `data/index/ablation/{fusion.json,retrieval-ablation.json,retrieval-metrics.md}`.
