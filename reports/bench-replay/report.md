# answerdesk benchmark run — `bench`

> **REPLAY-MODE.** Deterministic doubles were used; these numbers validate the harness, not the system.

| field | value |
|---|---|
| corpus version | corpus-77d7d081813738a6 |
| corpus hash | ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36 |
| documents / chunks | 810 / 7229 |
| test cases | 530 |
| calibration | {'corpus_version': 'corpus-77d7d081813738a6', 'split': 'dev', 'n': 120, 'buckets': [{'bucket': 'informational', 'n': 108, 'alpha': 0.05, 'q_hat': 0.085714, 'rank': 104, 'feasible': True, 'pooled_fallback': False, 'coverage_lower_bound': 0.95}, {'bucket': 'transactional', 'n': 12, 'alpha': 0.02, 'q_hat': inf, 'rank': 13, 'feasible': False, 'pooled_fallback': False, 'coverage_lower_bound': 0.98}, {'bucket': 'account_legal', 'n': 120, 'alpha': 0.01, 'q_hat': 0.533333, 'rank': 120, 'feasible': True, 'pooled_fallback': True, 'coverage_lower_bound': 0.99}], 'alpha_by_bucket': {'informational': 0.05, 'transactional': 0.02, 'account_legal': 0.01}, 'costs': {'c_wrong': 1.0, 'c_abstain': 0.1, 'c_delay': 0.05}, 'note': 'split-conformal thresholds fitted on the dev split; the test split is never used for fitting (INV-09)', 'operating_point_selection': {'source': "no feasible point at the target risk on the calibration split; the conformal engine's own threshold is the shipped rule", 'threshold': 1.0, 'dev_coverage': 0.0, 'dev_risk': 0.0, 'dev_n': 120, 'dev_min_risk': 1.0, 'dev_coverage_at_min_risk': 0.008333, 'feasible': False}} |
| hardware | Windows 11 / Python 3.12.10 |
| duration | 33.2 s |

## Retrieval ablation

| arm | recall_any@10 | recall_any@20 | recall_all@20 | nDCG@10 | MRR@10 | n |
|---|---|---|---|---|---|---|
| bm25 | 0.992 [0.981, 0.998] | 0.992 [0.981, 0.998] | 0.936 [0.920, 0.948] | 0.905 [0.889, 0.919] | 0.976 [0.963, 0.987] | 464 |
| dense | 0.462 [0.421, 0.504] | 0.473 [0.431, 0.514] | 0.316 [0.280, 0.349] | 0.279 [0.247, 0.311] | 0.309 [0.275, 0.345] | 464 |
| rrf@10 | 0.992 [0.981, 0.998] | 0.992 [0.981, 0.998] | 0.765 [0.745, 0.783] | 0.685 [0.665, 0.706] | 0.822 [0.800, 0.844] | 464 |
| rrf@60 | 0.987 [0.976, 0.996] | 0.989 [0.980, 0.998] | 0.748 [0.727, 0.768] | 0.667 [0.644, 0.690] | 0.778 [0.753, 0.804] | 464 |
| rrf@100 | 0.987 [0.976, 0.996] | 0.989 [0.980, 0.998] | 0.748 [0.727, 0.768] | 0.666 [0.643, 0.689] | 0.776 [0.751, 0.802] | 464 |
| cc@0.5 | 0.989 [0.979, 0.998] | 0.989 [0.979, 0.998] | 0.919 [0.901, 0.936] | 0.852 [0.834, 0.869] | 0.836 [0.814, 0.859] | 464 |
| cc@0.6 | 0.981 [0.968, 0.992] | 0.987 [0.977, 0.996] | 0.824 [0.803, 0.847] | 0.644 [0.618, 0.669] | 0.585 [0.554, 0.617] | 464 |
| cc@0.7 | 0.748 [0.710, 0.784] | 0.825 [0.788, 0.857] | 0.566 [0.532, 0.599] | 0.447 [0.414, 0.481] | 0.466 [0.430, 0.505] | 464 |
| cc@0.75 | 0.598 [0.559, 0.641] | 0.662 [0.621, 0.703] | 0.439 [0.403, 0.475] | 0.381 [0.347, 0.415] | 0.426 [0.386, 0.465] | 464 |
| cc@0.8 | 0.526 [0.486, 0.568] | 0.562 [0.520, 0.605] | 0.389 [0.355, 0.425] | 0.348 [0.314, 0.382] | 0.392 [0.354, 0.429] | 464 |
| cc@0.9 | 0.493 [0.451, 0.535] | 0.506 [0.464, 0.548] | 0.343 [0.308, 0.377] | 0.310 [0.276, 0.342] | 0.350 [0.314, 0.388] | 464 |

### Paired comparisons (recall_any@10)

| pair | n | Δ | 95% CI | McNemar p | CI excludes 0 |
|---|---|---|---|---|---|
| bm25 − dense | 464 | +0.532 | [+0.489, +0.575] | 0.0000 (chi2) | yes |
| bm25 − rrf@60 | 464 | +0.004 | [-0.004, +0.015] | 0.6875 (exact_binomial) | no |
| rrf@60 − cc@0.75 | 464 | +0.392 | [+0.349, +0.433] | 0.0000 (chi2) | yes |

## Answer quality and escalation per arm

### `primary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 0.606635 | [0.559, 0.652] | 422 |
| must_escalate_recall | 1.0 | [0.966, 1.000] | 108 |
| escalation_precision | 0.394161 | [0.338, 0.453] | 274 |
| realised_risk | 1.0 | [0.985, 1.000] | 256 |

Confusion: TP 108 · FP 166 · FN 0 · TN 256 — false-escalation rate 0.393.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 223 | 0.813869 |
| INJECTION_SUSPECTED | 24 | 0.087591 |
| PII_DETECTED | 18 | 0.065693 |
| CLARIFY_AMBIGUOUS | 8 | 0.029197 |
| NO_EVIDENCE | 1 | 0.00365 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 422 | 166 | 0.393365 | [0.348, 0.441] |
| injection | 24 | 24 | 1.0 | [0.862, 1.000] |
| pii | 18 | 18 | 1.0 | [0.824, 1.000] |
| unanswerable | 66 | 66 | 1.0 | [0.945, 1.000] |

Operating point: coverage **0.483** at realised risk **1.0000** (target ≤ 2%).

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.928302 | 1.0 | 492 |
| 5:1 | 0.8 | 0.813208 | 1.0 | 431 |
| 10:1 | 0.9 | 0.539623 | 1.0 | 286 |

### `secondary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 0.606635 | [0.559, 0.652] | 422 |
| must_escalate_recall | 1.0 | [0.966, 1.000] | 108 |
| escalation_precision | 0.394161 | [0.338, 0.453] | 274 |
| realised_risk | 1.0 | [0.985, 1.000] | 256 |

Confusion: TP 108 · FP 166 · FN 0 · TN 256 — false-escalation rate 0.393.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 223 | 0.813869 |
| INJECTION_SUSPECTED | 24 | 0.087591 |
| PII_DETECTED | 18 | 0.065693 |
| CLARIFY_AMBIGUOUS | 8 | 0.029197 |
| NO_EVIDENCE | 1 | 0.00365 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 422 | 166 | 0.393365 | [0.348, 0.441] |
| injection | 24 | 24 | 1.0 | [0.862, 1.000] |
| pii | 18 | 18 | 1.0 | [0.824, 1.000] |
| unanswerable | 66 | 66 | 1.0 | [0.945, 1.000] |

Operating point: coverage **0.483** at realised risk **1.0000** (target ≤ 2%).

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.928302 | 1.0 | 492 |
| 5:1 | 0.8 | 0.813208 | 1.0 | 431 |
| 10:1 | 0.9 | 0.539623 | 1.0 | 286 |

## Semantic cache

Skipped: replay mode: the cache harness needs the hosted embedding model

## Operations

| arm | stage | p50 ms | p95 ms | n |
|---|---|---|---|---|
| primary | retrieval | 1.8 | 5.485 | 530 |
| primary | policy | 0.985 | 1.314 | 530 |
| primary | generation | 1.827 | 3.085 | 421 |
| primary | total | 4.765 | 7.849 | 530 |
| secondary | retrieval | 1.77 | 5.533 | 530 |
| secondary | policy | 0.97 | 1.291 | 530 |
| secondary | generation | 1.813 | 3.016 | 421 |
| secondary | total | 4.717 | 7.716 | 530 |

Token counts and USD are recorded per answer from the provider's own usage fields and the pinned pricing table; see `ops/` for the pricing snapshot.

## Noisy / free-form slice

Reported separately and never averaged into the headline numbers.

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| groundedness_claim_pooled | 0.0 | [0.000, 1.000] | 0 |
| citation_precision | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| phantom_citations | 0.0 | [0.000, 0.058] | 62 |
| unsupported_displayed_claims | 0.0 | [0.000, 0.058] | 62 |
| language_agreement | 1.0 | [0.942, 1.000] | 62 |
| realised_risk | 1.0 | [0.942, 1.000] | 62 |
| coverage | 0.317949 | [0.257, 0.386] | 195 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 0.028] | 133 |
| realised_risk_by_bucket | 1.0 | [0.942, 1.000] | 62 |
| coverage_by_bucket | 0.317949 | [0.257, 0.386] | 195 |

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| groundedness_claim_pooled | 0.0 | [0.000, 1.000] | 0 |
| citation_precision | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| phantom_citations | 0.0 | [0.000, 0.058] | 62 |
| unsupported_displayed_claims | 0.0 | [0.000, 0.058] | 62 |
| language_agreement | 1.0 | [0.942, 1.000] | 62 |
| realised_risk | 1.0 | [0.942, 1.000] | 62 |
| coverage | 0.317949 | [0.257, 0.386] | 195 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 0.028] | 133 |
| realised_risk_by_bucket | 1.0 | [0.942, 1.000] | 62 |
| coverage_by_bucket | 0.317949 | [0.257, 0.386] | 195 |

## Notes

* split v1 f4738e34a303 corpus corpus-77d7d081813738a6
* REPLAY: hashing embedder, no hosted models
