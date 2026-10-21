# The maths behind answerdesk

Every formula the stack actually uses, in one place, with what it decides and where it lives in the
code. This is deliberately short: the goal is that a reader can check a number in
[`docs/results.md`](results.md) against the formula that produced it.

Notation: `q` a query, `d` a document chunk, `G_q` the gold chunk set of `q`, `top_k(q)` the ranked
result, `n` a sample size, `α` a risk target, `p̂` the decision score. All intervals are 95% unless
stated.

---

## 1. Lexical retrieval: BM25

```
score(q, d) = Σ_{t ∈ q} IDF(t) · tf(t,d) · (k₁ + 1) / (tf(t,d) + k₁ · (1 − b + b · |d| / avgdl))
IDF(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
```

`k₁` controls term-frequency saturation, `b` length normalisation. The implementation is SQLite FTS5
(in-process) with number/date/identifier tokens preserved; see
[`docs/retrieval.md`](retrieval.md) and `src/answerdesk/index/`.

Why it matters here: on the synthetic corpus BM25 is very strong (`recall_any@10 = 0.996`), which is a
property of generated text, not of BM25 in general. The ablation reports it rather than hiding it.

## 2. Dense retrieval

Cosine similarity over L2-normalised vectors, exact search over a `float32` matrix — no ANN index, so
there is no recall mystery in the evaluation:

```
sim(q, d) = (e_q · e_d) / (‖e_q‖ · ‖e_d‖)
```

The CI embedder is a deterministic feature-hashing double (`hashing-64`); real runs use `bge-m3` or
`Qwen3-Embedding` behind the `local` extra. **A quality claim measured on the hashing embedder is not
a quality claim** — that distinction is enforced by the reporting rules in
[`docs/eval-methodology.md`](eval-methodology.md).

## 3. Fusion

### Reciprocal Rank Fusion (shipped default)

```
RRF(d) = Σ_{r ∈ retrievers} w_r / (k + rank_r(d))          k = 60, ranks start at 1
```

No score calibration, no labels, robust to score-scale differences — which is exactly why it is the
default. `k` is configurable (`retrieval.rrf_k`), and the ablation sweeps `k ∈ {10, 60, 100}`.

### Convex combination (tuned path)

```
CC(d) = α · ŝ_dense(d) + (1 − α) · ŝ_sparse(d)
ŝ(d)  = (s(d) − min S) / (max S − min S)                  # min–max over the candidate pool
```

`α` is fitted **on the dev split only** (`retrieval/fit.py`), persisted as an artifact with its
provenance, and reported in every run. Two caveats that the code documents instead of hiding:

- **Min–max is query-relative.** A retriever whose scores are nearly flat gets stretched to `[0, 1]`
  and can dominate the fusion. The fusion code falls back to rank-based scoring when a channel's
  distribution is degenerate.
- **More parameters is not more quality.** In this corpus's ablation, `bm25` alone beat every hybrid
  arm; the paired interval for `bm25 − rrf@60` includes zero. The tuning machinery exists so the claim
  can be tested, not so it can be assumed.

## 4. Ranking metrics

```
recall_any@k = (1/|Q|) Σ_q 1[ G_q ∩ top_k(q) ≠ ∅ ]        # did any gold chunk make it in
recall_all@k = (1/|Q|) Σ_q |G_q ∩ top_k(q)| / |G_q|        # how much of the evidence made it in
nDCG@k       = (1/|Q|) Σ_q DCG_q@k / IDCG_q@k
DCG_q@k      = Σ_{i=1..k} rel_i / log₂(i+1),  rel_i ∈ {0,1}
MRR@k        = (1/|Q|) Σ_q 1 / rank_q(first gold chunk within k)
```

`recall_any` is what the user experiences; `recall_all` is what a multi-chunk answer needs. Both are
reported at the `k` actually fed to the generator (10 and 20), never only at a flattering `k`.

## 5. Groundedness and citations

**Groundedness (claim support rate)** — the fraction of displayed claims whose cited passage supports
them, judged per claim:

```
groundedness = (1/|C|) Σ_{c ∈ claims} 1[verdict(c) = supported]
```

**Citation recall / precision** (ALCE-style, computed per answer statement):

```
citation_recall    = |{statements fully supported by ≥ 1 cited passage}| / |statements|
citation_precision = |{citations that support their statement}| / |citations|
```

Citation recall of `0.211` in [`docs/results.md`](results.md) is the honest weak spot: the verifier
accepts partial support, so many statements are not *fully* covered. Tightening the verifier is a
roadmap item, not a documentation fix.

**Numeric guard (deterministic, no model).** Every number or date in a displayed claim must appear,
after canonicalisation (`5 марта` → `2026-03-05`, `12,5 €` → `12.50 EUR`), in at least one chunk cited
by that claim. A violation drops the claim or escalates with `CITATION_FAILED`. This is the one
citation rule that cannot be softened by a judge.

## 6. Split conformal prediction (the escalation decision)

Given calibration risk scores `s_i = 1 − p̂_i` on `n` exchangeable labelled examples and a target
error rate `α`, let

```
q̂ = the ⌈(n + 1)(1 − α)⌉-th smallest of {s_1, …, s_n}
```

and answer only when `s(x) ≤ q̂`. Then, under exchangeability of calibration and test data,

```
P( error on an answered point ) ≤ α        (marginal, distribution-free, finite-sample)
```

`n` must satisfy `⌈(n+1)(1−α)⌉ ≤ n`, i.e. `n ≥ 1/α − 1` (for `α = 0.05`, `n ≥ 19`); the artifact
records `n`, `α`, `q̂`, the split hash and the scorer version.

**Per-bucket α.** The cost of a wrong answer is not uniform, so calibration is per intent bucket:
informational `α = 0.05`, transactional `0.02`, account/legal `0.01`. The effective target is
`α_final = min(α_bucket, c_abstain / c_wrong)` — the cost model may tighten α, never loosen it.

**What actually happened (published, not buried).** The decision score is a deterministic overlap
signal that saturates at `p̂ = 1.0` for ~74% of queries. A saturated score has no resolution, so `q̂`
collapses to `0` and the threshold answers exactly the saturated set. On held-out data that set has a
22.3% error rate against `α = 0.05`. A logistic scorer fitted on dev improved calibration
(`ECE 0.343 → 0.079`) but still failed the pre-stated gate (realised risk `0.410`; Wilson upper bound
`0.446 > 0.05`).

**The gate, stated before measurement.** Risk control is claimed only if the Wilson 95% upper bound of
the realised risk on the answered set is `≤ α`. A point estimate below α with an upper bound above it
does not count. This rule is why two "improvements" in this repository are labelled failures.

## 7. Cost-sensitive three-way rule

With `c_wrong` (a wrong auto-answer), `c_abstain` (escalate to a human) and `c_delay` (ask a
clarifying question), the expected-cost minimising rule is:

```
answer    iff  p̂ ≥ 1 − c_delay  / c_wrong
clarify   iff  1 − c_abstain / c_wrong ≤ p̂ < 1 − c_delay / c_wrong
escalate  otherwise
```

Derivation: answering costs `(1 − p̂)·c_wrong`; clarifying costs `c_delay` and defers the residual
risk; escalating costs `c_abstain`. Comparing those three expected costs gives the bands above.

| costs | answer ≥ | clarify band | escalate < |
|---|---|---|---|
| `c_wrong = 1.00`, `c_abstain = 0.10`, `c_delay = 0.05` (defaults) | 0.95 | [0.90, 0.95) | 0.90 |

Costs are configuration with provenance (`configs/policy/`), never literals in code; the ratio is a
business input, and this repository flags it as an assumption rather than a measurement.

## 8. Semantic cache: an error budget, not a similarity threshold

A cache hit is a **decision**, so it needs a risk target. For a cohort of cache entries, choose the
smallest similarity threshold `τ` such that, on the near-miss replay set,

```
wrong_hits / hits ≤ ε            and            UCB_Wilson(wrong_hits, hits) ≤ 2ε
```

with `ε = 0.01`. The upper-bound requirement prevents "0 wrong hits out of 3" from passing. A
wrong hit is defined adversarially: a hit for a query whose gold answer differs from the cached
answer's gold answer (different plan, amount, date, order state, locale).

On the current replay pairs the guard set rejected every near-miss, so the measured hit rate is
`0.0 [0.000, 0.121]` and the wrong-hit rate is `0.0 (UCB 0.0)`. An honest zero: conservative guards,
no positive evidence yet.

Guards that run before a hit is served (entity equality, source-document hash, version tuple,
structural re-validation of citations and the numeric guard) are not statistics — they are
deterministic checks, and a failure demotes the hit to a miss.

## 9. Risk–coverage and AURC

```
coverage(t)        = |{x : s(x) ≤ t}| / n
selective risk(t)  = mean error over covered points
AURC               = ∫ risk(c) dc   over coverage c ∈ [0, 1]      (lower is better)
```

Reported with the whole curve, never as a single tuned point, because a single point can always be
made to look good by moving the threshold after seeing the test set. The operating point in
[`docs/results.md`](results.md) is the one selected on **dev**; the test-optimal point is labelled
oracle/diagnostic and excluded from claims.

## 10. Calibration of the score itself

```
ECE   = Σ_b (n_b / n) · | acc(b) − conf(b) |        b over confidence bins
Brier = (1/n) Σ_i (p̂_i − y_i)²
```

`ECE` is binning-dependent, so the report carries both equal-width (10 bins) and equal-mass
reliability data plus the Brier score and the reliability plot. A low ECE with no discriminative power
is exactly the failure mode observed here: the logistic scorer is *well calibrated* and still useless
for risk control because it cannot separate correct from incorrect turns.

## 11. Statistics used for comparisons

**Wilson score interval** for a proportion `k/n` (better than the normal approximation at small `n`
and near 0/1):

```
centre = (p̂ + z²/2n) / (1 + z²/n),   half = z/(1 + z²/n) · √( p̂(1−p̂)/n + z²/4n² )
```

**Paired bootstrap** over queries for a difference of means: resample queries with replacement
`B = 1000` times, report the 2.5/97.5 percentiles of the metric difference. Pairing is what makes the
comparison sensitive; comparing two unpaired means throws away the pairing.

**McNemar** for paired binary outcomes (per-query correct/incorrect):

```
χ² = (b − c)² / (b + c)      b, c = discordant pairs
```

with the exact binomial test when `b + c < 25` (the implementation reports which was used, because the
`χ²` approximation is unreliable in that regime).

No comparison is reported as a bare point estimate: every cell carries `n` and an interval, and a
"win" whose interval crosses zero is written as **no claim** (the `zh` row of the external report is
the worked example).

## 12. Judge agreement (κ)

Cohen's κ between the LLM judge and human labels:

```
κ = (p_o − p_e) / (1 − p_e)        p_o = observed agreement, p_e = chance agreement
```

The repository's rule: a judged metric may be quoted as a finding only when `κ ≥ 0.6` for that
language. There are no human labels in this checkout, so every judged number in
[`docs/results.md`](results.md) is labelled **diagnostic** and is not a release gate.

## 13. What none of this gives you

1. **A guarantee about the future.** Conformal validity holds under exchangeability; support traffic
   drifts, and the calibration set must be refreshed and coverage monitored.
2. **External validity.** Every internal number is measured on a synthetic corpus.
3. **A calibrated probability from a saturated score.** The maths is correct; a degenerate input makes
   the output degenerate. That is a data/signal problem, and the roadmap says so.
4. **Anything about production cost.** Prices are a pinned snapshot; a free tariff is recorded as 0.
