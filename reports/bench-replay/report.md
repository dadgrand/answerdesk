# answerdesk benchmark run — `bench-replay`

> **REPLAY-MODE.** Deterministic doubles were used; these numbers validate the harness, not the system.

| field | value |
|---|---|
| corpus version | corpus-96f93db4f7f648b5 |
| corpus hash | 7ef3fc20400131dc966904bc68d517a805faf48e98b62584f44e086270e246d6 |
| documents / chunks | 810 / 7229 |
| test cases | 360 |
| calibration | no calibration artifact supplied: escalation decisions fall back to the configured cost bands only (diagnostic, not the shipped conformal path) |
| hardware | Windows 11 / Python 3.12.10 |
| duration | 17.3 s |

## Retrieval ablation

| arm | recall_any@10 | recall_any@20 | recall_all@20 | nDCG@10 | MRR@10 | n |
|---|---|---|---|---|---|---|
| bm25 | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 0.927 [0.909, 0.944] | 0.887 [0.866, 0.906] | 0.970 [0.955, 0.984] | 294 |
| dense | 0.425 [0.371, 0.480] | 0.446 [0.391, 0.503] | 0.290 [0.246, 0.334] | 0.256 [0.215, 0.296] | 0.285 [0.236, 0.331] | 294 |
| rrf@10 | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 0.748 [0.723, 0.773] | 0.664 [0.634, 0.693] | 0.806 [0.775, 0.835] | 294 |
| rrf@60 | 0.980 [0.963, 0.993] | 0.986 [0.973, 0.997] | 0.726 [0.697, 0.754] | 0.637 [0.602, 0.670] | 0.752 [0.714, 0.787] | 294 |
| rrf@100 | 0.980 [0.963, 0.993] | 0.986 [0.973, 0.997] | 0.726 [0.697, 0.754] | 0.637 [0.602, 0.670] | 0.751 [0.711, 0.785] | 294 |
| cc@0.5 | 0.997 [0.990, 1.000] | 0.997 [0.990, 1.000] | 0.920 [0.900, 0.939] | 0.840 [0.818, 0.860] | 0.825 [0.796, 0.852] | 294 |
| cc@0.6 | 0.983 [0.966, 0.997] | 0.997 [0.990, 1.000] | 0.811 [0.784, 0.840] | 0.616 [0.580, 0.648] | 0.558 [0.514, 0.601] | 294 |
| cc@0.7 | 0.687 [0.633, 0.735] | 0.793 [0.752, 0.833] | 0.524 [0.475, 0.565] | 0.407 [0.361, 0.452] | 0.433 [0.384, 0.483] | 294 |
| cc@0.75 | 0.561 [0.510, 0.616] | 0.609 [0.558, 0.663] | 0.403 [0.355, 0.452] | 0.351 [0.307, 0.398] | 0.401 [0.349, 0.454] | 294 |
| cc@0.8 | 0.500 [0.446, 0.558] | 0.514 [0.459, 0.571] | 0.360 [0.311, 0.409] | 0.324 [0.280, 0.370] | 0.365 [0.315, 0.417] | 294 |
| cc@0.9 | 0.473 [0.422, 0.527] | 0.486 [0.432, 0.541] | 0.326 [0.281, 0.373] | 0.288 [0.245, 0.331] | 0.327 [0.279, 0.376] | 294 |

### Paired comparisons (recall_any@10)

| pair | n | Δ | 95% CI | McNemar p | CI excludes 0 |
|---|---|---|---|---|---|
| bm25 − dense | 294 | +0.571 | [+0.517, +0.629] | 0.0000 (chi2) | yes |
| bm25 − rrf@60 | 294 | +0.017 | [+0.003, +0.034] | 0.0625 (exact_binomial) | yes |
| rrf@60 − cc@0.75 | 294 | +0.418 | [+0.364, +0.476] | 0.0000 (chi2) | yes |

## Answer quality and escalation per arm

### `primary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 1.0 | [0.758, 1.000] | 12 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 1.000] | 0 |
| realised_risk | 1.0 | [0.758, 1.000] | 12 |

Confusion: TP 0 · FP 0 · FN 0 · TN 12 — false-escalation rate n/a.

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 12 | 0 | 0.0 | [0.000, 0.242] |

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.75 | 1.0 | 9 |
| 5:1 | 0.8 | 0.583333 | 1.0 | 7 |
| 10:1 | 0.9 | 0.583333 | 1.0 | 7 |

### `secondary` — generator `fake-model`, judge `fake-judge`

Cross-family judging: **no** (fake vs fake); reportable: **no**. same-family judge (fake judging fake): diagnostic only, never reported (§10.5)

| metric | value | 95% CI | n |
|---|---|---|---|
| groundedness_judged | 0.0 | [0.000, 1.000] | 0 |
| citation_recall | 0.0 | [0.000, 1.000] | 0 |
| coverage | 1.0 | [0.758, 1.000] | 12 |
| must_escalate_recall | 0.0 | [0.000, 1.000] | 0 |
| escalation_precision | 0.0 | [0.000, 1.000] | 0 |
| realised_risk | 1.0 | [0.758, 1.000] | 12 |

Confusion: TP 0 · FP 0 · FN 0 · TN 12 — false-escalation rate n/a.

E-2 recall by unsafe family:

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 12 | 0 | 0.0 | [0.000, 0.242] |

| cost ratio | threshold | coverage | realised risk | n answered |
|---|---|---|---|---|
| 2:1 | 0.5 | 0.75 | 1.0 | 9 |
| 5:1 | 0.8 | 0.583333 | 1.0 | 7 |
| 10:1 | 0.9 | 0.583333 | 1.0 | 7 |

## Semantic cache

Skipped: replay mode: the cache harness needs the hosted embedding model

## Operations

| arm | stage | p50 ms | p95 ms | n |
|---|---|---|---|---|
| primary | retrieval | 1.982 | 2.798 | 12 |
| primary | generation | 1.902 | 2.649 | 12 |
| primary | total | 4.037 | 5.329 | 12 |
| secondary | retrieval | 1.911 | 2.062 | 12 |
| secondary | generation | 1.926 | 2.63 | 12 |
| secondary | total | 3.782 | 4.503 | 12 |

Token counts and USD are recorded per answer from the provider's own usage fields and the pinned pricing table; see `ops/` for the pricing snapshot.

## Noisy / free-form slice

Reported separately and never averaged into the headline numbers.

`primary`: no noisy items in this corpus version.
`secondary`: no noisy items in this corpus version.
## Notes

* split v1 5e57276f1c92 corpus corpus-96f93db4f7f648b5
* REPLAY: hashing embedder, no hosted models
