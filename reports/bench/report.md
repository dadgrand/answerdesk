# answerdesk benchmark run — `bench`

| field | value |
|---|---|
| corpus version | corpus-77d7d081813738a6 |
| corpus hash | ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36 |
| documents / chunks | 810 / 7229 |
| test cases | 530 |
| calibration | {'alpha_by_bucket': {'account_legal': 0.01, 'informational': 0.05, 'transactional': 0.02}, 'buckets': [{'alpha': 0.05, 'bucket': 'informational', 'coverage_lower_bound': 0.95, 'feasible': True, 'n': 200, 'pooled_fallback': False, 'q_hat': 0.436364, 'rank': 191}, {'alpha': 0.02, 'bucket': 'transactional', 'coverage_lower_bound': 0.98, 'feasible': True, 'n': 200, 'pooled_fallback': True, 'q_hat': 0.533333, 'rank': 197}, {'alpha': 0.01, 'bucket': 'account_legal', 'coverage_lower_bound': 0.99, 'feasible': True, 'n': 200, 'pooled_fallback': True, 'q_hat': 0.6, 'rank': 199}], 'corpus_version': 'corpus-77d7d081813738a6', 'costs': {'c_abstain': 0.1, 'c_delay': 0.05, 'c_wrong': 1.0}, 'n': 200, 'note': 'split-conformal thresholds fitted on the dev split; the test split is never used for fitting (INV-09)', 'operating_point_selection': {'dev_coverage': 0.005, 'dev_coverage_at_min_risk': 0.005, 'dev_min_risk': 0.0, 'dev_n': 200, 'dev_risk': 0.0, 'feasible': True, 'source': 'dev-selected threshold (risk-constrained on the calibration split)', 'threshold': 1.0}, 'split': 'dev'} |
| hardware | Windows 11 / Python 3.12.10 |
| duration | 0.0 s |

## Retrieval ablation

| arm | recall_any@10 | recall_any@20 | recall_all@20 | nDCG@10 | MRR@10 | n |
|---|---|---|---|---|---|---|
| bm25 | 0.996 [0.988, 1.000] | 0.996 [0.988, 1.000] | 0.947 [0.928, 0.962] | 0.934 [0.915, 0.952] | 0.984 [0.970, 0.996] | 240 |
| cc@0.5 | 0.996 [0.988, 1.000] | 0.996 [0.988, 1.000] | 0.926 [0.904, 0.945] | 0.919 [0.898, 0.937] | 0.971 [0.953, 0.986] | 240 |
| cc@0.6 | 0.996 [0.988, 1.000] | 0.996 [0.988, 1.000] | 0.900 [0.875, 0.922] | 0.880 [0.854, 0.902] | 0.948 [0.925, 0.970] | 240 |
| cc@0.7 | 0.979 [0.963, 0.996] | 0.988 [0.975, 1.000] | 0.848 [0.817, 0.876] | 0.833 [0.799, 0.862] | 0.921 [0.891, 0.949] | 240 |
| cc@0.75 | 0.975 [0.954, 0.992] | 0.975 [0.954, 0.992] | 0.817 [0.782, 0.850] | 0.809 [0.774, 0.840] | 0.914 [0.880, 0.944] | 240 |
| cc@0.75+rerank | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.270 [0.236, 0.307] | 0.357 [0.331, 0.384] | 0.904 [0.863, 0.940] | 80 |
| cc@0.8 | 0.975 [0.954, 0.992] | 0.975 [0.954, 0.992] | 0.799 [0.762, 0.831] | 0.787 [0.750, 0.818] | 0.903 [0.868, 0.935] | 240 |
| cc@0.9 | 0.963 [0.938, 0.983] | 0.963 [0.938, 0.983] | 0.762 [0.724, 0.795] | 0.731 [0.693, 0.766] | 0.858 [0.819, 0.893] | 240 |
| dense | 0.950 [0.921, 0.975] | 0.954 [0.925, 0.979] | 0.707 [0.668, 0.743] | 0.672 [0.632, 0.706] | 0.820 [0.780, 0.860] | 240 |
| rrf@10 | 0.996 [0.988, 1.000] | 0.996 [0.988, 1.000] | 0.893 [0.870, 0.913] | 0.874 [0.850, 0.895] | 0.965 [0.946, 0.982] | 240 |
| rrf@100 | 0.992 [0.979, 1.000] | 0.996 [0.988, 1.000] | 0.869 [0.840, 0.894] | 0.856 [0.826, 0.881] | 0.957 [0.936, 0.977] | 240 |
| rrf@60 | 0.992 [0.979, 1.000] | 0.996 [0.988, 1.000] | 0.868 [0.839, 0.893] | 0.856 [0.827, 0.882] | 0.957 [0.936, 0.977] | 240 |

### Paired comparisons (recall_any@10)

| pair | n | Δ | 95% CI | McNemar p | CI excludes 0 |
|---|---|---|---|---|---|
| bm25 − dense | 240 | +0.046 | [+0.021, +0.075] | 0.0010 (exact_binomial) | yes |
| bm25 − rrf@60 | 240 | +0.004 | [+0.000, +0.013] | 1.0000 (exact_binomial) | no |
| rrf@60 − cc@0.75 | 240 | +0.017 | [+0.004, +0.033] | 0.1250 (exact_binomial) | yes |
| cc@0.75 − cc@0.75+rerank | 80 | +0.000 | [+0.000, +0.000] | 1.0000 (n/a) | no |

## Answer quality and escalation per arm

### `primary` — generator `qwen3.6-35b-a3b-noreason`, judge `deepseek-flash`

Cross-family judging: **yes** (qwen vs deepseek); reportable: **yes**. cross-family judge; human κ not established for this corpus, so judged numbers are reported with an explicit κ caveat, not as F-1/F-2/F-3 gates

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.907801 | [0.858, 0.950] | 141 |
| citation_recall | 0.210537 | [0.189, 0.233] | 141 |
| rubric_score | 4.691828 | [4.576, 4.790] | 141 |
| coverage | 0.840909 | [0.780, 0.888] | 176 |
| must_escalate_recall | 1.0 | [0.862, 1.000] | 24 |
| escalation_precision | 0.461538 | [0.333, 0.595] | 52 |
| realised_risk | 0.222973 | [0.163, 0.297] | 148 |

Confusion: TP 24 · FP 28 · FN 0 · TN 148 — false-escalation rate 0.159.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 28 | 0.538462 |
| INJECTION_SUSPECTED | 24 | 0.461538 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 176 | 28 | 0.159091 | [0.112, 0.220] |
| unsafe | 24 | 24 | 1.0 | [0.862, 1.000] |

Operating point: coverage **0.740** at realised risk **0.2230** (target ≤ 2%).

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.945 | 0.391534 | 189 |
| 5:1 | 0.8 | 0.84 | 0.315476 | 168 |
| 10:1 | 0.9 | 0.76 | 0.243421 | 152 |

### `secondary` — generator `deepseek-flash`, judge `qwen3.6-35b-a3b-noreason`

Cross-family judging: **yes** (deepseek vs qwen); reportable: **yes**. cross-family judge; human κ not established for this corpus, so judged numbers are reported with an explicit κ caveat, not as F-1/F-2/F-3 gates

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.390805 | [0.287, 0.494] | 87 |
| citation_recall | 0.224818 | [0.188, 0.263] | 87 |
| rubric_score | 3.61092 | [3.385, 3.825] | 87 |
| coverage | 0.840909 | [0.780, 0.888] | 176 |
| must_escalate_recall | 1.0 | [0.862, 1.000] | 24 |
| escalation_precision | 0.461538 | [0.333, 0.595] | 52 |
| realised_risk | 0.817568 | [0.748, 0.871] | 148 |

Confusion: TP 24 · FP 28 · FN 0 · TN 148 — false-escalation rate 0.159.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 28 | 0.538462 |
| INJECTION_SUSPECTED | 24 | 0.461538 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 176 | 28 | 0.159091 | [0.112, 0.220] |
| unsafe | 24 | 24 | 1.0 | [0.862, 1.000] |

Operating point: coverage **0.740** at realised risk **0.8176** (target ≤ 2%).

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.945 | 0.857143 | 189 |
| 5:1 | 0.8 | 0.84 | 0.839286 | 168 |
| 10:1 | 0.9 | 0.76 | 0.822368 | 152 |

## Semantic cache

| quantity | value |
|---|---|
| pairs | 30 |
| hits / attempts | 0 / 28 |
| hit rate | 0.00% |
| wrong-hit rate | 0.00% |
| hit latency p50/p95 | 0.0 / 0.0 ms |
| saved USD (hits) | 0.0 |

## Operations

| arm | stage | p50 ms | p95 ms | n |
|---|---|---|---|---|
| primary | retrieval | 1.439 | 2.541 | 200 |
| primary | generation | 2618.221 | 14116.757 | 200 |
| primary | judge | 3409.631 | 51578.15 | 200 |
| primary | total | 7748.476 | 55107.455 | 200 |
| secondary | retrieval | 1.387 | 2.423 | 200 |
| secondary | generation | 7097.864 | 13367.989 | 200 |
| secondary | judge | 651.958 | 4930.769 | 200 |
| secondary | total | 9588.746 | 14859.048 | 200 |

Token counts and USD are recorded per answer from the provider's own usage fields and the pinned pricing table; see `ops/` for the pricing snapshot.

## Noisy / free-form slice

Reported separately and never averaged into the headline numbers.

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.952381 | [0.857, 1.000] | 21 |
| groundedness_claim_pooled | 0.951613 | [0.867, 0.983] | 62 |
| citation_precision | 0.784127 | [0.640, 0.929] | 21 |
| citation_recall | 0.174206 | [0.136, 0.227] | 21 |
| rubric_score | 4.72619 | [4.428, 4.940] | 21 |
| phantom_citations | 0.0 | [0.000, 0.149] | 22 |
| unsupported_displayed_claims | 0.0 | [0.000, 0.149] | 22 |
| language_agreement | 1.0 | [0.851, 1.000] | 22 |

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.333333 | [0.133, 0.600] | 15 |
| groundedness_claim_pooled | 0.291667 | [0.182, 0.432] | 48 |
| citation_precision | 0.594444 | [0.444, 0.756] | 15 |
| citation_recall | 0.163889 | [0.115, 0.222] | 15 |
| rubric_score | 3.516667 | [3.064, 4.023] | 15 |
| phantom_citations | 0.0 | [0.000, 0.149] | 22 |
| unsupported_displayed_claims | 0.0 | [0.000, 0.149] | 22 |
| language_agreement | 1.0 | [0.851, 1.000] | 22 |

## Notes

* phased execution: the same functions as `answerdesk eval --all`, run in resumable chunks because this sandbox terminates long-lived processes
* answer arms retrieve with the convex-combination pipeline; the hosted reranker is measured in the retrieval ablation
