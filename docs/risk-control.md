# Risk control v2: calibrated answerability with a finite-sample certificate

The escalation policy in `docs/results.md` was calibrated on a **saturated** signal: the overlap
score is 1.0 for most questions, so a conformal quantile on it collapsed and the realised risk
(22.3%) was four times the target. This document reports the replacement: a decision-time
answerability model, a threshold selected by **Learn-then-Test / Conformal Risk Control** with a
finite-sample **Bentkus** bound, and the honest risk–coverage frontier on frozen test splits.

Everything below is serialised in `reports/risk-v2/`:

| artefact | contents |
|---|---|
| `summary.json` | both populations, model weights, certificate, frontier point, arms, targets, `n_required` |
| `certificate.json` | the certificate for the primary population (threshold, calibration risk, UCB, grid, rule, claims) |
| `frontier.csv` | risk–coverage curve of the primary population (one row per accepted prefix) |
| `reliability.csv` | equal-mass reliability bins of the primary population |
| `features.csv`, `bench-features.csv` | the feature tables (dev + test rows, digests recorded in the summary) |
| `split.json` | the fact-level split rule and the dev/test fact lists, with `no_leakage` |
| `self-consistency.json` | the dev-only k=3 draft measurement and its ablation |
| `report.md` | the same numbers in prose |

## Reproduction

```bash
# 1. features for the consistency population (one bounded reranker call per row, ~340 rows)
. ./scripts/live-env.ps1 neuraldeep
uv run python scripts/risk-v2.py --build-features

# 2. features for the bench population (no reranker needed; see the caveat below)
uv run python scripts/risk-v2.py --build-bench-features --no-rerank

# 3. dev-only self-consistency ablation (k=3 drafts, bounded: 25 questions x 3 calls)
uv run python scripts/risk-v2.py --self-consistency --limit 25

# 4. fit on dev, certify, evaluate once on test, write the artefacts
uv run python scripts/risk-v2.py --evaluate

# 5. reproducibility gate: re-derives every artefact from the cached feature tables, no network
uv run python scripts/risk-v2.py --check
```

## Features (all computed before the answer is shown)

| feature | what it measures | availability |
|---|---|---|
| `rerank_top` | cross-encoder (hosted bge-reranker) relevance of the best passage | consistency: measured (340 calls); bench: **rate-limited, 0.0** |
| `retrieval_margin` | normalised gap between the best and second lexical score | both |
| `claim_support` | deterministic share of answer sentences whose values are supported by the cited text | consistency only (needs full answer text) |
| `citation_coverage` | how much evidence the answer cites (0 / 1 / 2+) | both |
| `numeric_consistency` | every amount, date, ID and percentage the answer asserts appears in the passages it cites | consistency only |
| `self_consistency` | mean pairwise agreement of k=3 sampled drafts (temperature 0.7) | **dev-only ablation**, never in the test-time model |

The deployable model is an L2 logistic regression on the decision-time features plus the t11
legacy set (`bm25_top`, `bm25_margin`, `bm25_above_gate`, `citation_density`, `log_claims`,
`overlap_score`), with **Platt calibration fitted on dev** so the reported ECE measures the model
rather than a logit offset. The old 6-feature logistic and the raw overlap score are reported as
arms.

## The certificate (not a nonconformity quantile)

For a calibration set of `n` rows with observed selective risk `R̂`, the upper bound at level
`δ` is

- **Bentkus**: `R̂ + sqrt(R̂ · ln(1/δ) / n) + ln(1/δ) / n`
- **Hoeffding**: `R̂ + sqrt(ln(1/δ) / (2n))`

The threshold is the largest calibration acceptance level whose **UCB ≤ α** (one pass over the
grid, most permissive first, stop at the first feasible level — Learn-then-Test / fixed
sequence). The certificate therefore covers the selection rule, not a threshold chosen after
looking at test data, and the test risk is reported separately.

**Why the 5% certificate is absent.** With `n_dev = 133` and a base error of 69%, the dev split's
zero-error region has 10 accepted rows, whose UCB is 0.23 — above 5%. That is a statement about
data size, not about the model: the summary carries the requirement explicitly.

| bound | rows needed at zero observed risk (α = 0.05, δ = 0.1) |
|---|---|
| Bentkus | ≈ 46 |
| Hoeffding | ≈ 461 |

More rows are needed when the observed risk is not zero. Producing a 5% certificate therefore
requires roughly 500–1000 dev answers with the current base error — the cost is stated, not
hidden.

**Operational fallback.** At the largest feasible level, **α = 0.25**, the certificate *is*
feasible (coverage 8.3%, calibration risk 0.0) — reported as the fallback with its own α, never
as a 5% guarantee.

## Results (frozen test splits, evaluated once)

| population | n (dev/test) | base error (test) | AUROC | ECE | coverage @ risk ≤ 5% | certificate @ α=0.05 |
|---|---|---|---|---|---|---|
| **consistency (primary)** | 133 / 207 | 0.76 | **0.965** | **0.057** | **0.116** (24 rows, risk 0.042) | absent (n too small) |
| bench (t11 turns) | 200 / 200 | 0.43 | 0.665 | 0.186 | 0.0 | absent |

Targets: AUROC ≥ 0.75 ✓ on the primary population, ECE ≤ 0.08 ✓, coverage ≥ 0.40 ✗ (**documented
miss**: the achieved frontier is reported, 0.116 at ≤5% risk).

Arms on the primary population: raw overlap AUROC 0.392 (the saturated signal is worse than
chance), legacy 6-feature 0.937, risk-v2 **0.965**.

### Which population works, and why the other does not

The signal exists where the model can *see its own work*: numeric consistency and deterministic
claim support are the two strongest features, and they need the full answer text. The consistency
population has it; the t11 bench population persisted only a 60-character answer head, so those
features are unavailable there and its reranker feature could not be built under the free-tariff
rate limit (HTTP 429 on `/rerank`). The bench population therefore keeps only four weaker
features and misses the AUROC target (0.665) — reported as a measured miss, not as a reason to
drop the population.

## Per-intent and per-language breakdown

`summary.json` → `populations.<name>.per_intent` and `.per_language` carry AUROC, accuracy and the
coverage-at-risk point per group, with pooled fallback flagged for groups under 20 test rows (the
per-language view of the primary population is thin: 25 facts × 6 locales, so several locales
fall back to pooled).

## Honest limitations

1. **Labels are cross-family LLM judgments, not human labels** (`deepseek-flash` judging
   `qwen3.6-35b-a3b-noreason` answers). The certificate is a certificate *about the judge's notion
   of correctness*; the human-label gate in `docs/external-validity.md` still applies before any
   user-facing claim.
2. **The 5% certificate is absent** on both populations; the achieved frontier (0.116 coverage at
   ≤5% risk on the primary population) is the honest product number for this data size.
3. **The primary population is hard**: its base error is dominated by retrieval landing on a
   sibling document (CNS-2 in `docs/consistency.md`), so the frontier is compressed toward low
   coverage. A production corpus without the sibling ambiguity should give a wider frontier at the
   same AUROC, but that is a hypothesis, not a measurement.
4. **The bench population is feature-limited** (two features unavailable, reranker rate-limited);
   its AUROC miss is a data-availability consequence, not evidence that the model is weak.
5. **Test risk is not covered by the certificate.** It is reported with a Wilson interval and is
   the number a reviewer should hold the system to.
6. **No test data was used for fitting**: the split is over facts (consistency) or the frozen
   t11 dev/test split (bench), `split.json` records `no_leakage: true`, and `--check` fails if the
   split leaks or the feature table drifts from the summary digest.
