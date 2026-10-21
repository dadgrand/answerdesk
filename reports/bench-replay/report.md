# answerdesk benchmark run — `bench-replay`

> **REPLAY-MODE.** Deterministic doubles were used; these numbers validate the harness, not the system.

| field | value |
|---|---|
| corpus version | corpus-96f93db4f7f648b5 |
| corpus hash | 7ef3fc20400131dc966904bc68d517a805faf48e98b62584f44e086270e246d6 |
| documents / chunks | 810 / 7229 |
| test cases | 540 |
| calibration | {'corpus_version': 'corpus-96f93db4f7f648b5', 'split': 'dev', 'n': 120, 'buckets': [{'bucket': 'informational', 'n': 108, 'alpha': 0.05, 'q_hat': 0.675, 'rank': 104, 'feasible': True, 'coverage_lower_bound': 0.95}, {'bucket': 'transactional', 'n': 12, 'alpha': 0.02, 'q_hat': inf, 'rank': 13, 'feasible': False, 'coverage_lower_bound': 0.98}], 'alpha_by_bucket': {'informational': 0.05, 'transactional': 0.02, 'account_legal': 0.01}, 'costs': {'c_wrong': 1.0, 'c_abstain': 0.1, 'c_delay': 0.05}, 'note': 'split-conformal thresholds fitted on the dev split; the test split is never used for fitting (INV-09)'} |
| hardware | Windows 11 / Python 3.12.10 |
| duration | 24.6 s |

## Retrieval ablation

| arm | recall_any@10 | recall_any@20 | recall_all@20 | nDCG@10 | MRR@10 | n |
|---|---|---|---|---|---|---|
| bm25 | 0.962 [0.945, 0.979] | 0.964 [0.947, 0.979] | 0.915 [0.896, 0.933] | 0.745 [0.718, 0.772] | 0.757 [0.723, 0.790] | 474 |
| dense | 0.386 [0.344, 0.428] | 0.397 [0.352, 0.441] | 0.289 [0.254, 0.323] | 0.231 [0.202, 0.262] | 0.246 [0.213, 0.281] | 474 |
| rrf@10 | 0.888 [0.859, 0.916] | 0.914 [0.888, 0.939] | 0.750 [0.723, 0.778] | 0.563 [0.534, 0.591] | 0.625 [0.590, 0.660] | 474 |
| rrf@60 | 0.871 [0.842, 0.901] | 0.901 [0.876, 0.928] | 0.731 [0.700, 0.762] | 0.541 [0.511, 0.570] | 0.581 [0.543, 0.615] | 474 |
| rrf@100 | 0.871 [0.842, 0.901] | 0.901 [0.876, 0.928] | 0.731 [0.700, 0.762] | 0.541 [0.511, 0.569] | 0.579 [0.542, 0.614] | 474 |
| cc@0.5 | 0.939 [0.918, 0.960] | 0.945 [0.924, 0.964] | 0.892 [0.871, 0.915] | 0.695 [0.667, 0.722] | 0.643 [0.611, 0.677] | 474 |
| cc@0.6 | 0.892 [0.865, 0.920] | 0.916 [0.890, 0.939] | 0.795 [0.766, 0.824] | 0.529 [0.501, 0.559] | 0.456 [0.423, 0.489] | 474 |
| cc@0.7 | 0.591 [0.551, 0.633] | 0.686 [0.643, 0.728] | 0.509 [0.473, 0.549] | 0.355 [0.321, 0.388] | 0.356 [0.320, 0.392] | 474 |
| cc@0.75 | 0.498 [0.454, 0.542] | 0.532 [0.487, 0.576] | 0.397 [0.360, 0.435] | 0.312 [0.279, 0.345] | 0.333 [0.297, 0.370] | 474 |
| cc@0.8 | 0.443 [0.401, 0.487] | 0.464 [0.422, 0.508] | 0.356 [0.319, 0.392] | 0.286 [0.252, 0.319] | 0.304 [0.268, 0.341] | 474 |
| cc@0.9 | 0.414 [0.369, 0.456] | 0.430 [0.386, 0.475] | 0.323 [0.287, 0.359] | 0.257 [0.226, 0.289] | 0.277 [0.242, 0.313] | 474 |

### Paired comparisons (recall_any@10)

| pair | n | Δ | 95% CI | McNemar p | CI excludes 0 |
|---|---|---|---|---|---|
| bm25 − dense | 474 | +0.576 | [+0.532, +0.620] | 0.0000 (chi2) | yes |
| bm25 − rrf@60 | 474 | +0.091 | [+0.068, +0.118] | 0.0000 (chi2) | yes |
| rrf@60 − cc@0.75 | 474 | +0.373 | [+0.327, +0.411] | 0.0000 (chi2) | yes |

## Answer quality and escalation per arm

### `primary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 0.333333 | [0.180, 0.533] | 24 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 0.194] | 16 |
| realised_risk | 1.0 | [0.676, 1.000] | 8 |

Confusion: TP 0 · FP 16 · FN 0 · TN 8 — false-escalation rate 0.667.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 10 | 0.625 |
| CLARIFY_AMBIGUOUS | 6 | 0.375 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 24 | 16 | 0.666667 | [0.467, 0.820] |

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.75 | 1.0 | 18 |
| 5:1 | 0.8 | 0.666667 | 1.0 | 16 |
| 10:1 | 0.9 | 0.583333 | 1.0 | 14 |

### `secondary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 0.333333 | [0.180, 0.533] | 24 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 0.194] | 16 |
| realised_risk | 1.0 | [0.676, 1.000] | 8 |

Confusion: TP 0 · FP 16 · FN 0 · TN 8 — false-escalation rate 0.667.

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 10 | 0.625 |
| CLARIFY_AMBIGUOUS | 6 | 0.375 |

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 24 | 16 | 0.666667 | [0.467, 0.820] |

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.75 | 1.0 | 18 |
| 5:1 | 0.8 | 0.666667 | 1.0 | 16 |
| 10:1 | 0.9 | 0.583333 | 1.0 | 14 |

## Semantic cache

Skipped: replay mode: the cache harness needs the hosted embedding model

## Operations

| arm | stage | p50 ms | p95 ms | n |
|---|---|---|---|---|
| primary | retrieval | 1.92 | 2.626 | 24 |
| primary | policy | 0.943 | 1.254 | 24 |
| primary | generation | 2.179 | 2.804 | 24 |
| primary | total | 4.962 | 6.269 | 24 |
| secondary | retrieval | 2.007 | 2.607 | 24 |
| secondary | policy | 0.986 | 1.268 | 24 |
| secondary | generation | 2.154 | 2.841 | 24 |
| secondary | total | 5.1 | 6.209 | 24 |

Token counts and USD are recorded per answer from the provider's own usage fields and the pinned pricing table; see `ops/` for the pricing snapshot.

## Noisy / free-form slice

Reported separately and never averaged into the headline numbers.

`primary`: no noisy items in this corpus version.
`secondary`: no noisy items in this corpus version.
## Notes

* split v1 5e57276f1c92 corpus corpus-96f93db4f7f648b5
* REPLAY: hashing embedder, no hosted models
