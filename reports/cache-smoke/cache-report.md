# Semantic-cache report (synthetic near-miss replay set)

* replay hash: `82cfe3b586091942…`
* threshold table: 1.0.0 (calibrated 2026-10-20T22:12:01+00:00)
* epsilon (wrong-hit budget): 0.01
* pairs: 2160; hits: 527; wrong hits: 0

## Release targets

| metric | value | target | pass |
|---|---|---|---|
| K-1 wrong-hit rate | 0.0000 | ≤ 0.01 | yes |
| K-1 wrong-hit UCB | 0.0072 | ≤ 0.0200 | yes |
| K-2 informational hit rate | 0.2986 | ≥ 0.15 | yes |
| precision (correct hits / hits) | 1.0000 | — | — |
| recall (correct hits / same-answer pairs) | 0.7319 | — | — |

## Per cohort

| cohort | threshold | pairs | hits | hit rate | wrong hits | precision | recall |
|---|---|---|---|---|---|---|---|
| account_legal:cjk:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |
| account_legal:cyrillic:len1 | 1.000 | 360 | 78 | 0.217 | 0 | 1.000 | 0.650 |
| account_legal:latin-tr:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |
| account_legal:latin-uz:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |
| informational:cjk:len1 | 1.000 | 120 | 37 | 0.308 | 0 | 1.000 | 0.925 |
| informational:cyrillic:len1 | 1.000 | 360 | 108 | 0.300 | 0 | 1.000 | 0.900 |
| informational:latin-tr:len1 | 1.000 | 120 | 33 | 0.275 | 0 | 1.000 | 0.825 |
| informational:latin-uz:len1 | 1.000 | 120 | 37 | 0.308 | 0 | 1.000 | 0.925 |
| transactional:cjk:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |
| transactional:cyrillic:len1 | 1.000 | 360 | 78 | 0.217 | 0 | 1.000 | 0.650 |
| transactional:latin-tr:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |
| transactional:latin-uz:len1 | 1.000 | 120 | 26 | 0.217 | 0 | 1.000 | 0.650 |

## Limitations

1. The replay set is synthetic: paraphrases are token reorderings and near misses
   change one entity. Real traffic is harder; t11 replays the HITL-labelled set.
2. A cohort with no feasible threshold reports `threshold=1.0` (no semantic hits),
   which satisfies K-1 vacuously; K-2 is what prevents that from being a strategy.
