# answerdesk evaluation report — `offline-smoke`

> **REPLAY-MODE.** This run used deterministic doubles (FakeProvider / FakeJudge) and machine-labelled synthetic data. It validates the harness, not the system: no number here may be presented as a live-model or production result (§10.7 item 9, NFR-REP-01).

## 1. Run header — REPLAY-MODE

| field | value |
|---|---|
| run id | `offline-smoke` |
| schema / contracts | 1.0.0 |
| corpus version | `corpus-ab58e8c0f5cf506b` |
| corpus hash (sha256) | `02671e0a484f8035231ecea73245ce27b9d60a4e1bb869736365e2e0ddbe477d` |
| golden version | v1 |
| golden hash (sha256) | `5a45594a9251e904f92c506d36e75b1082679d4d1917985682c2d18668cc61eb` |
| split version | v1 |
| split hash (sha256) | `e1a9164ceb3ba572f53730a618d6e0d1d158905ec538951ae6135c0adbb5449d` |
| seed | 20261021 |
| provider mode | fake (replay) |
| judge model | fake-judge |
| judge prompt version | none (deterministic double) |
| judge rubric version | fake-rubric-1 |
| judge position swap | on |
| judge κ reportable | no |
| arms | hybrid-rrf, bm25, dense-hash |
| locales | ky, ru, tg, tr, uz, zh |
| cases (test split) | 360 |
| wall clock | 4.5 s |
| hardware | Windows 11 / Python 3.12.10 |
| pricing table | configs/pricing.yaml |
| deterministic digest | `69dd478a8124f86c66d1077551ac4da249663e458633c37d60c811cb409afbbb` |

Every cell below carries `n` and a 95% interval. `method` says how the interval was computed: `wilson` for a proportion, `bootstrap` for a mean (B resamples, §10.4), `none` for a distribution summary such as p95 latency.

Judge note: no human calibration labels for this run: judged metrics are REPLAY-MODE diagnostics and are not reportable (§7.5 F-10).

Run notes:

* position-swap audit: 8/8 comparisons kept their verdict across both orders (invariance 1.00, mean order effect 0.000)
* human agreement: no calibration labels on disk (evals/judge/human_v1.jsonl); judged metrics stay not-reportable (§7.5 F-10)

Every section below is marked **REPLAY-MODE**: a number produced by deterministic doubles is a diagnostic, never a live-model result.

## 2. Retrieval — REPLAY-MODE

### 2.1 Arm `hybrid-rrf` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.792 | [0.755, 0.828] | 294 | bootstrap |
| MRR@20 | 0.795 | [0.759, 0.829] | 294 | bootstrap |
| nDCG@10 | 0.700 | [0.668, 0.733] | 294 | bootstrap |
| nDCG@20 | 0.814 | [0.788, 0.838] | 294 | bootstrap |
| recall_all@10 | 0.698 | [0.667, 0.732] | 294 | bootstrap |
| recall_all@20 | 0.907 | [0.889, 0.924] | 294 | bootstrap |
| recall_any@10 | 0.966 | [0.942, 0.986] | 294 | bootstrap |
| recall_any@20 | 1.000 | [1.000, 1.000] | 294 | bootstrap |

### 2.2 Arm `hybrid-rrf` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.946 | [0.889, 0.990] | 49 |
| MRR@10 | ru | 0.819 | [0.717, 0.904] | 49 |
| MRR@10 | tg | 0.893 | [0.805, 0.965] | 49 |
| MRR@10 | tr | 0.847 | [0.756, 0.929] | 49 |
| MRR@10 | uz | 0.736 | [0.640, 0.830] | 49 |
| MRR@10 | zh | 0.511 | [0.402, 0.617] | 49 |
| MRR@20 | ky | 0.946 | [0.889, 0.990] | 49 |
| MRR@20 | ru | 0.822 | [0.723, 0.905] | 49 |
| MRR@20 | tg | 0.895 | [0.806, 0.965] | 49 |
| MRR@20 | tr | 0.851 | [0.762, 0.930] | 49 |
| MRR@20 | uz | 0.736 | [0.640, 0.830] | 49 |
| MRR@20 | zh | 0.520 | [0.416, 0.622] | 49 |
| nDCG@10 | ky | 0.849 | [0.782, 0.910] | 49 |
| nDCG@10 | ru | 0.701 | [0.606, 0.785] | 49 |
| nDCG@10 | tg | 0.832 | [0.750, 0.906] | 49 |
| nDCG@10 | tr | 0.777 | [0.685, 0.862] | 49 |
| nDCG@10 | uz | 0.627 | [0.539, 0.715] | 49 |
| nDCG@10 | zh | 0.411 | [0.332, 0.488] | 49 |
| nDCG@20 | ky | 0.914 | [0.865, 0.954] | 49 |
| nDCG@20 | ru | 0.807 | [0.733, 0.865] | 49 |
| nDCG@20 | tg | 0.899 | [0.839, 0.952] | 49 |
| nDCG@20 | tr | 0.859 | [0.795, 0.916] | 49 |
| nDCG@20 | uz | 0.764 | [0.704, 0.827] | 49 |
| nDCG@20 | zh | 0.640 | [0.578, 0.697] | 49 |
| recall_all@10 | ky | 0.831 | [0.761, 0.896] | 49 |
| recall_all@10 | ru | 0.679 | [0.587, 0.757] | 49 |
| recall_all@10 | tg | 0.827 | [0.749, 0.905] | 49 |
| recall_all@10 | tr | 0.786 | [0.694, 0.871] | 49 |
| recall_all@10 | uz | 0.639 | [0.547, 0.729] | 49 |
| recall_all@10 | zh | 0.429 | [0.357, 0.500] | 49 |
| recall_all@20 | ky | 0.949 | [0.914, 0.980] | 49 |
| recall_all@20 | ru | 0.873 | [0.820, 0.914] | 49 |
| recall_all@20 | tg | 0.948 | [0.907, 0.983] | 49 |
| recall_all@20 | tr | 0.935 | [0.892, 0.971] | 49 |
| recall_all@20 | uz | 0.888 | [0.845, 0.933] | 49 |
| recall_all@20 | zh | 0.851 | [0.802, 0.892] | 49 |
| recall_any@10 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | ru | 0.959 | [0.898, 1.000] | 49 |
| recall_any@10 | tg | 0.980 | [0.939, 1.000] | 49 |
| recall_any@10 | tr | 0.959 | [0.898, 1.000] | 49 |
| recall_any@10 | uz | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | zh | 0.898 | [0.816, 0.980] | 49 |
| recall_any@20 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | ru | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | tg | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | tr | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | uz | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | zh | 1.000 | [1.000, 1.000] | 49 |

### 2.1 Arm `bm25` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.987 | [0.976, 0.995] | 294 | bootstrap |
| MRR@20 | 0.987 | [0.976, 0.995] | 294 | bootstrap |
| nDCG@10 | 0.991 | [0.985, 0.996] | 294 | bootstrap |
| nDCG@20 | 0.992 | [0.986, 0.997] | 294 | bootstrap |
| recall_all@10 | 0.997 | [0.995, 0.999] | 294 | bootstrap |
| recall_all@20 | 0.999 | [0.997, 1.000] | 294 | bootstrap |
| recall_any@10 | 1.000 | [1.000, 1.000] | 294 | bootstrap |
| recall_any@20 | 1.000 | [1.000, 1.000] | 294 | bootstrap |

### 2.2 Arm `bm25` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.990 | [0.969, 1.000] | 49 |
| MRR@10 | ru | 0.969 | [0.929, 1.000] | 49 |
| MRR@10 | tg | 0.990 | [0.969, 1.000] | 49 |
| MRR@10 | tr | 0.990 | [0.969, 1.000] | 49 |
| MRR@10 | uz | 0.983 | [0.949, 1.000] | 49 |
| MRR@10 | zh | 1.000 | [1.000, 1.000] | 49 |
| MRR@20 | ky | 0.990 | [0.969, 1.000] | 49 |
| MRR@20 | ru | 0.969 | [0.929, 1.000] | 49 |
| MRR@20 | tg | 0.990 | [0.969, 1.000] | 49 |
| MRR@20 | tr | 0.990 | [0.969, 1.000] | 49 |
| MRR@20 | uz | 0.983 | [0.949, 1.000] | 49 |
| MRR@20 | zh | 1.000 | [1.000, 1.000] | 49 |
| nDCG@10 | ky | 0.994 | [0.981, 1.000] | 49 |
| nDCG@10 | ru | 0.985 | [0.969, 0.999] | 49 |
| nDCG@10 | tg | 0.994 | [0.981, 1.000] | 49 |
| nDCG@10 | tr | 0.988 | [0.975, 0.999] | 49 |
| nDCG@10 | uz | 0.989 | [0.969, 1.000] | 49 |
| nDCG@10 | zh | 0.997 | [0.991, 1.000] | 49 |
| nDCG@20 | ky | 0.994 | [0.981, 1.000] | 49 |
| nDCG@20 | ru | 0.987 | [0.974, 0.997] | 49 |
| nDCG@20 | tg | 0.994 | [0.981, 1.000] | 49 |
| nDCG@20 | tr | 0.991 | [0.979, 1.000] | 49 |
| nDCG@20 | uz | 0.991 | [0.972, 1.000] | 49 |
| nDCG@20 | zh | 0.997 | [0.991, 1.000] | 49 |
| recall_all@10 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_all@10 | ru | 0.989 | [0.976, 0.998] | 49 |
| recall_all@10 | tg | 1.000 | [1.000, 1.000] | 49 |
| recall_all@10 | tr | 0.996 | [0.990, 1.000] | 49 |
| recall_all@10 | uz | 0.998 | [0.994, 1.000] | 49 |
| recall_all@10 | zh | 1.000 | [1.000, 1.000] | 49 |
| recall_all@20 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_all@20 | ru | 0.995 | [0.985, 1.000] | 49 |
| recall_all@20 | tg | 1.000 | [1.000, 1.000] | 49 |
| recall_all@20 | tr | 1.000 | [1.000, 1.000] | 49 |
| recall_all@20 | uz | 1.000 | [1.000, 1.000] | 49 |
| recall_all@20 | zh | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | ru | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | tg | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | tr | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | uz | 1.000 | [1.000, 1.000] | 49 |
| recall_any@10 | zh | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | ky | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | ru | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | tg | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | tr | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | uz | 1.000 | [1.000, 1.000] | 49 |
| recall_any@20 | zh | 1.000 | [1.000, 1.000] | 49 |

### 2.1 Arm `dense-hash` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.399 | [0.355, 0.450] | 294 | bootstrap |
| MRR@20 | 0.403 | [0.360, 0.454] | 294 | bootstrap |
| nDCG@10 | 0.349 | [0.308, 0.392] | 294 | bootstrap |
| nDCG@20 | 0.409 | [0.366, 0.453] | 294 | bootstrap |
| recall_all@10 | 0.352 | [0.311, 0.395] | 294 | bootstrap |
| recall_all@20 | 0.462 | [0.417, 0.506] | 294 | bootstrap |
| recall_any@10 | 0.558 | [0.507, 0.606] | 294 | bootstrap |
| recall_any@20 | 0.616 | [0.565, 0.663] | 294 | bootstrap |

### 2.2 Arm `dense-hash` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.573 | [0.463, 0.694] | 49 |
| MRR@10 | ru | 0.306 | [0.210, 0.409] | 49 |
| MRR@10 | tg | 0.570 | [0.455, 0.685] | 49 |
| MRR@10 | tr | 0.538 | [0.413, 0.668] | 49 |
| MRR@10 | uz | 0.320 | [0.209, 0.445] | 49 |
| MRR@10 | zh | 0.087 | [0.023, 0.168] | 49 |
| MRR@20 | ky | 0.576 | [0.469, 0.697] | 49 |
| MRR@20 | ru | 0.309 | [0.215, 0.413] | 49 |
| MRR@20 | tg | 0.571 | [0.456, 0.685] | 49 |
| MRR@20 | tr | 0.544 | [0.423, 0.672] | 49 |
| MRR@20 | uz | 0.331 | [0.219, 0.453] | 49 |
| MRR@20 | zh | 0.088 | [0.023, 0.171] | 49 |
| nDCG@10 | ky | 0.491 | [0.390, 0.602] | 49 |
| nDCG@10 | ru | 0.267 | [0.180, 0.354] | 49 |
| nDCG@10 | tg | 0.521 | [0.411, 0.630] | 49 |
| nDCG@10 | tr | 0.446 | [0.342, 0.557] | 49 |
| nDCG@10 | uz | 0.292 | [0.187, 0.401] | 49 |
| nDCG@10 | zh | 0.077 | [0.019, 0.151] | 49 |
| nDCG@20 | ky | 0.548 | [0.452, 0.659] | 49 |
| nDCG@20 | ru | 0.346 | [0.249, 0.446] | 49 |
| nDCG@20 | tg | 0.608 | [0.496, 0.717] | 49 |
| nDCG@20 | tr | 0.530 | [0.417, 0.645] | 49 |
| nDCG@20 | uz | 0.335 | [0.228, 0.449] | 49 |
| nDCG@20 | zh | 0.087 | [0.023, 0.167] | 49 |
| recall_all@10 | ky | 0.486 | [0.384, 0.592] | 49 |
| recall_all@10 | ru | 0.274 | [0.190, 0.361] | 49 |
| recall_all@10 | tg | 0.532 | [0.427, 0.635] | 49 |
| recall_all@10 | tr | 0.441 | [0.339, 0.549] | 49 |
| recall_all@10 | uz | 0.306 | [0.200, 0.418] | 49 |
| recall_all@10 | zh | 0.076 | [0.018, 0.149] | 49 |
| recall_all@20 | ky | 0.588 | [0.486, 0.696] | 49 |
| recall_all@20 | ru | 0.420 | [0.311, 0.527] | 49 |
| recall_all@20 | tg | 0.688 | [0.575, 0.799] | 49 |
| recall_all@20 | tr | 0.598 | [0.482, 0.716] | 49 |
| recall_all@20 | uz | 0.382 | [0.263, 0.504] | 49 |
| recall_all@20 | zh | 0.094 | [0.027, 0.176] | 49 |
| recall_any@10 | ky | 0.714 | [0.592, 0.837] | 49 |
| recall_any@10 | ru | 0.592 | [0.449, 0.735] | 49 |
| recall_any@10 | tg | 0.816 | [0.714, 0.918] | 49 |
| recall_any@10 | tr | 0.673 | [0.551, 0.816] | 49 |
| recall_any@10 | uz | 0.429 | [0.306, 0.571] | 49 |
| recall_any@10 | zh | 0.122 | [0.041, 0.224] | 49 |
| recall_any@20 | ky | 0.755 | [0.633, 0.878] | 49 |
| recall_any@20 | ru | 0.633 | [0.490, 0.776] | 49 |
| recall_any@20 | tg | 0.837 | [0.735, 0.939] | 49 |
| recall_any@20 | tr | 0.735 | [0.612, 0.857] | 49 |
| recall_any@20 | uz | 0.592 | [0.449, 0.715] | 49 |
| recall_any@20 | zh | 0.143 | [0.041, 0.245] | 49 |

## 3. Answer quality — REPLAY-MODE

| arm / metric | value | 95% CI | n | method |
|---|---|---|---|---|
| hybrid-rrf / groundedness_verifier | 1.000 | [1.000, 1.000] | 202 | bootstrap |
| hybrid-rrf / groundedness_claim_pooled | 1.000 | [0.990, 1.000] | 390 | wilson |
| hybrid-rrf / citation_precision | 1.000 | [1.000, 1.000] | 202 | bootstrap |
| hybrid-rrf / citation_recall | 1.000 | [1.000, 1.000] | 202 | bootstrap |
| hybrid-rrf / rubric_score | 5.000 | [5.000, 5.000] | 202 | bootstrap |
| hybrid-rrf / rubric_pass_rate | 1.000 | [0.981, 1.000] | 202 | wilson |
| hybrid-rrf / phantom_citations | 0.000 | [0.000, 0.019] | 202 | wilson |
| hybrid-rrf / unsupported_displayed_claims | 0.000 | [0.000, 0.019] | 202 | wilson |
| hybrid-rrf / language_agreement | 0.950 | [0.911, 0.973] | 202 | wilson |
| hybrid-rrf / answer_accuracy_by_category [direct] | 0.897 | [0.831, 0.939] | 126 | wilson |
| hybrid-rrf / answer_accuracy_by_category [numeric_date] | 0.389 | [0.285, 0.504] | 72 | wilson |
| hybrid-rrf / answer_accuracy_by_category [typo_code_switch] | 0.852 | [0.734, 0.923] | 54 | wilson |
| hybrid-rrf / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| hybrid-rrf / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| hybrid-rrf / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| bm25 / groundedness_verifier | 1.000 | [1.000, 1.000] | 200 | bootstrap |
| bm25 / groundedness_claim_pooled | 1.000 | [0.990, 1.000] | 389 | wilson |
| bm25 / citation_precision | 1.000 | [1.000, 1.000] | 200 | bootstrap |
| bm25 / citation_recall | 1.000 | [1.000, 1.000] | 200 | bootstrap |
| bm25 / rubric_score | 5.000 | [5.000, 5.000] | 200 | bootstrap |
| bm25 / rubric_pass_rate | 1.000 | [0.981, 1.000] | 200 | wilson |
| bm25 / phantom_citations | 0.000 | [0.000, 0.019] | 200 | wilson |
| bm25 / unsupported_displayed_claims | 0.000 | [0.000, 0.019] | 200 | wilson |
| bm25 / language_agreement | 0.920 | [0.874, 0.950] | 200 | wilson |
| bm25 / answer_accuracy_by_category [direct] | 0.889 | [0.822, 0.933] | 126 | wilson |
| bm25 / answer_accuracy_by_category [numeric_date] | 0.375 | [0.272, 0.490] | 72 | wilson |
| bm25 / answer_accuracy_by_category [typo_code_switch] | 0.833 | [0.713, 0.910] | 54 | wilson |
| bm25 / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| bm25 / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| bm25 / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| dense-hash / groundedness_verifier | 1.000 | [1.000, 1.000] | 177 | bootstrap |
| dense-hash / groundedness_claim_pooled | 1.000 | [0.988, 1.000] | 309 | wilson |
| dense-hash / citation_precision | 1.000 | [1.000, 1.000] | 177 | bootstrap |
| dense-hash / citation_recall | 1.000 | [1.000, 1.000] | 177 | bootstrap |
| dense-hash / rubric_score | 5.000 | [5.000, 5.000] | 177 | bootstrap |
| dense-hash / rubric_pass_rate | 1.000 | [0.979, 1.000] | 177 | wilson |
| dense-hash / phantom_citations | 0.000 | [0.000, 0.021] | 177 | wilson |
| dense-hash / unsupported_displayed_claims | 0.000 | [0.000, 0.021] | 177 | wilson |
| dense-hash / language_agreement | 0.960 | [0.921, 0.981] | 177 | wilson |
| dense-hash / answer_accuracy_by_category [direct] | 0.675 | [0.589, 0.750] | 126 | wilson |
| dense-hash / answer_accuracy_by_category [numeric_date] | 0.181 | [0.109, 0.285] | 72 | wilson |
| dense-hash / answer_accuracy_by_category [typo_code_switch] | 0.556 | [0.424, 0.680] | 54 | wilson |
| dense-hash / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| dense-hash / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| dense-hash / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |

`phantom_citations` and `unsupported_displayed_claims` are structural counts (F-4/F-5): they are reported as a rate over answered turns and must be 0.

## 4. Escalation and calibration — REPLAY-MODE

| arm / metric | value | 95% CI | n | note |
|---|---|---|---|---|
| hybrid-rrf / realised_risk | 0.074 | [0.046, 0.119] | 202 | 15/202 auto-answered turns were incorrect (E-1) |
| hybrid-rrf / coverage | 0.802 | [0.748, 0.846] | 252 | auto-answer coverage on answerable items (E-4) |
| hybrid-rrf / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| hybrid-rrf / escalation_precision | 0.684 | [0.607, 0.751] | 158 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| hybrid-rrf / realised_risk_by_bucket [informational] | 0.074 | [0.046, 0.119] | 202 | intent bucket informational (E-1 per bucket) |
| hybrid-rrf / coverage_by_bucket [informational] | 0.802 | [0.748, 0.846] | 252 | intent bucket informational |
| bm25 / realised_risk | 0.080 | [0.050, 0.126] | 200 | 16/200 auto-answered turns were incorrect (E-1) |
| bm25 / coverage | 0.794 | [0.739, 0.839] | 252 | auto-answer coverage on answerable items (E-4) |
| bm25 / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| bm25 / escalation_precision | 0.675 | [0.599, 0.743] | 160 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| bm25 / realised_risk_by_bucket [informational] | 0.080 | [0.050, 0.126] | 200 | intent bucket informational (E-1 per bucket) |
| bm25 / coverage_by_bucket [informational] | 0.794 | [0.739, 0.839] | 252 | intent bucket informational |
| dense-hash / realised_risk | 0.277 | [0.216, 0.347] | 177 | 49/177 auto-answered turns were incorrect (E-1) |
| dense-hash / coverage | 0.702 | [0.643, 0.755] | 252 | auto-answer coverage on answerable items (E-4) |
| dense-hash / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| dense-hash / escalation_precision | 0.590 | [0.518, 0.659] | 183 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| dense-hash / realised_risk_by_bucket [informational] | 0.277 | [0.216, 0.347] | 177 | intent bucket informational (E-1 per bucket) |
| dense-hash / coverage_by_bucket [informational] | 0.702 | [0.643, 0.755] | 252 | intent bucket informational |

### 4.1 Reliability (E-5/E-6 diagnostic)

| quantity | value |
|---|---|
| n | 360 |
| ECE equal-width | 0.503 |
| ECE equal-mass | 0.503 |
| Brier | 0.489 |
| AURC | 0.164 |
| mean confidence | 0.016 |
| accuracy | 0.519 |
| coverage at target risk | 0.533 |
| risk at target | 0.047 |

`p̂` here is the packed evidence score, not a conformal calibrator (t8 owns conformal prediction), so E-5 is a diagnostic and no risk guarantee is claimed.

## 5. Operations — REPLAY-MODE

| arm / metric | value | bracket | n | method |
|---|---|---|---|---|
| hybrid-rrf / latency_generation_p50_ms | 2.0930 | [1.9584, 2.1539] | 360 | none |
| hybrid-rrf / latency_generation_p95_ms | 3.6732 | [3.1023, 5.3274] | 360 | none |
| hybrid-rrf / latency_judge_p50_ms | 0.0428 | [0.0403, 0.0458] | 202 | none |
| hybrid-rrf / latency_judge_p95_ms | 0.0610 | [0.0570, 0.1125] | 202 | none |
| hybrid-rrf / latency_retrieval_p50_ms | 2.6989 | [2.6567, 2.7517] | 360 | none |
| hybrid-rrf / latency_retrieval_p95_ms | 4.0334 | [3.7149, 4.7714] | 360 | none |
| hybrid-rrf / latency_total_p50_ms | 4.9267 | [4.8531, 5.0327] | 360 | none |
| hybrid-rrf / latency_total_p95_ms | 6.8136 | [6.3937, 8.8828] | 360 | none |
| hybrid-rrf / latency_total_mean_ms | 5.0855 | [4.9879, 5.1800] | 360 | bootstrap |
| hybrid-rrf / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 360 | bootstrap |
| hybrid-rrf / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 360 | none |
| hybrid-rrf / tokens_input_mean | 710.6222 | [682.0622, 740.6992] | 360 | bootstrap |
| hybrid-rrf / tokens_output_mean | 32.8861 | [30.6163, 35.3029] | 360 | bootstrap |
| hybrid-rrf / cache_hit_rate | 0.1750 | [0.1392, 0.2176] | 360 | wilson |
| bm25 / latency_generation_p50_ms | 1.7792 | [1.6774, 1.8330] | 360 | none |
| bm25 / latency_generation_p95_ms | 3.0964 | [2.7358, 4.3771] | 360 | none |
| bm25 / latency_judge_p50_ms | 0.0435 | [0.0406, 0.0466] | 200 | none |
| bm25 / latency_judge_p95_ms | 0.0880 | [0.0604, 0.1681] | 200 | none |
| bm25 / latency_retrieval_p50_ms | 0.4190 | [0.3750, 0.5040] | 360 | none |
| bm25 / latency_retrieval_p95_ms | 1.5632 | [1.3361, 3.6681] | 360 | none |
| bm25 / latency_total_p50_ms | 2.3403 | [2.2634, 2.4396] | 360 | none |
| bm25 / latency_total_p95_ms | 3.8394 | [3.4861, 7.4093] | 360 | none |
| bm25 / latency_total_mean_ms | 2.5111 | [2.4270, 2.6077] | 360 | bootstrap |
| bm25 / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 360 | bootstrap |
| bm25 / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 360 | none |
| bm25 / tokens_input_mean | 611.7806 | [587.9785, 635.8088] | 360 | bootstrap |
| bm25 / tokens_output_mean | 31.7778 | [29.7021, 34.0647] | 360 | bootstrap |
| bm25 / cache_hit_rate | 0.1750 | [0.1392, 0.2176] | 360 | wilson |
| dense-hash / latency_generation_p50_ms | 2.4451 | [2.3342, 2.5915] | 360 | none |
| dense-hash / latency_generation_p95_ms | 4.2810 | [3.8346, 6.5643] | 360 | none |
| dense-hash / latency_judge_p50_ms | 0.0411 | [0.0350, 0.0459] | 177 | none |
| dense-hash / latency_judge_p95_ms | 0.0615 | [0.0578, 0.1210] | 177 | none |
| dense-hash / latency_retrieval_p50_ms | 1.9507 | [1.9316, 1.9822] | 360 | none |
| dense-hash / latency_retrieval_p95_ms | 2.4439 | [2.2930, 24.0979] | 360 | none |
| dense-hash / latency_total_p50_ms | 4.4069 | [4.2934, 4.5075] | 360 | none |
| dense-hash / latency_total_p95_ms | 6.4794 | [6.0451, 26.2549] | 360 | none |
| dense-hash / latency_total_mean_ms | 4.6021 | [4.4568, 4.7844] | 360 | bootstrap |
| dense-hash / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 360 | bootstrap |
| dense-hash / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 360 | none |
| dense-hash / tokens_input_mean | 816.4167 | [779.9126, 854.6158] | 360 | bootstrap |
| dense-hash / tokens_output_mean | 32.9111 | [29.3056, 37.0974] | 360 | bootstrap |
| dense-hash / cache_hit_rate | 0.1750 | [0.1392, 0.2176] | 360 | wilson |

Latency is measured in-process around the pipeline call, so it is a regression guard (NFR-LAT-03) and not a server-side latency claim (NFR-LAT-01). The bracket for a percentile is the p(q-5)..p(q+5) spread, not a confidence interval.

## 6. Statistics: paired bootstrap and McNemar — REPLAY-MODE

| metric | arms | n | mean A | mean B | Δ | Δ 95% CI (paired bootstrap) | McNemar | p (method) | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|---|
| recall_any@10 | hybrid-rrf − bm25 | 294 | 0.966 | 1.000 | -0.034 | [-0.055, -0.017] | b=0, c=10 | 0.0020 (exact_binomial) | yes |
| recall_any@20 | hybrid-rrf − bm25 | 294 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | b=0, c=0 | 1.0000 (n/a) | no |
| answer_ok | hybrid-rrf − bm25 | 360 | 0.819 | 0.811 | 0.008 | [-0.011, 0.031] | b=10, c=7 | 0.6291 (exact_binomial) | no |
| recall_any@10 | hybrid-rrf − dense-hash | 294 | 0.966 | 0.558 | 0.408 | [0.354, 0.469] | b=120, c=0 | 0.0000 (chi2) | yes |
| recall_any@20 | hybrid-rrf − dense-hash | 294 | 1.000 | 0.616 | 0.384 | [0.327, 0.442] | b=113, c=0 | 0.0000 (chi2) | yes |
| answer_ok | hybrid-rrf − dense-hash | 360 | 0.819 | 0.656 | 0.164 | [0.125, 0.206] | b=61, c=2 | 0.0000 (chi2) | yes |
| recall_any@10 | bm25 − dense-hash | 294 | 1.000 | 0.558 | 0.442 | [0.384, 0.500] | b=130, c=0 | 0.0000 (chi2) | yes |
| recall_any@20 | bm25 − dense-hash | 294 | 1.000 | 0.616 | 0.384 | [0.327, 0.442] | b=113, c=0 | 0.0000 (chi2) | yes |
| answer_ok | bm25 − dense-hash | 360 | 0.811 | 0.656 | 0.156 | [0.117, 0.197] | b=58, c=2 | 0.0000 (chi2) | yes |

Paired bootstrap resamples **queries**, not arms (B = 1000); McNemar uses the χ² approximation with the exact-binomial fallback when b + c < 25. A difference whose interval contains 0 is noise and is reported as such (§6.3 step 5).

## 7. Targets — REPLAY-MODE

| id | target | threshold | measured | verdict | note |
|---|---|---|---|---|---|
| RT-1 | recall_any@20 (hybrid, tuned) | >= 0.90 | 1.000 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-3 | recall_any@10 | >= 0.80 | 0.966 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-4 | recall_all@20 | >= 0.75 | 0.907 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-5 | nDCG@10 | >= 0.65 | 0.700 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-6 | MRR@10 | >= 0.60 | 0.792 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-2 | recall_any@20 per-locale floor | >= 0.80 every locale | 1.000 | diagnostic | locales: ky, ru, tg, tr, uz, zh |
| F-1 | groundedness (claim support rate) | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-2 | citation precision | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-3 | citation recall | >= 0.85 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| E-4 | auto-answer coverage | >= 0.60 | 0.802 | diagnostic | diagnostic: the policy layer is t8's |
| E-2 | must-escalate recall | >= 0.95 | 1.000 | diagnostic | diagnostic: the policy layer is t8's |
| E-3 | escalation precision | >= 0.55 | 0.684 | diagnostic | diagnostic: the policy layer is t8's |
| E-1 | realised auto-answer risk vs target | <= 0.05 (pooled <= 0.05) | 0.074 | diagnostic | diagnostic: no conformal calibration in this run (t8) |
| E-5 | calibration error of p̂ | ECE <= 0.08 and Brier <= 0.15 | 0.503 | diagnostic | diagnostic: p̂ is the evidence score, not a conformal calibrator; Brier=0.4892, ECE(mass)=0.5031 |

`diagnostic` means the run cannot speak to the gate: a replay-mode provider, a judge without human calibration labels, or a layer owned by another task. The number is printed anyway — hiding it would be the dishonesty §10.7 forbids.

## 8. Limitations (normative, §10.7) — REPLAY-MODE

1. **Synthetic data optimism.** Queries derive from the corpus and are paraphrased from it; recall and groundedness are therefore optimistic relative to real support traffic. No number here transfers to a production corpus without re-measurement.
2. **Small n.** A few hundred test items give roughly ±3 points on the macro retrieval metric and much wider intervals on per-locale and subset metrics. Every cell carries n and a CI; a 'win' that does not clear the CI is reported as noise.
3. **Judge bias.** Judged metrics are LLM-mediated; low-resource languages are likely worse. κ per language is published; where κ < 0.60 the metric is marked 'not reportable' rather than softened with adjectives.
4. **Benchmark ≠ support domain.** Public leaderboard values are vendor-curated snapshots; an embedder choice is justified only by our own ablation.
5. **Conformal exchangeability.** The risk guarantee holds under exchangeability. v1.0 has no real traffic, so drift is simulated, not observed.
6. **Environment dependence.** Latency and cost numbers are for the recorded machine and the pinned pricing table; they are not portable claims.
7. **No production evidence.** No SLA, no incident history, no live traffic, no measured handoff cost.
8. **Cache results are replay-based.** The semantic cache belongs to t8; this run measures an exact-normalised-query probe, not a production hit rate.
9. **Replay mode is labelled.** Any number produced without a live model/provider carries REPLAY-MODE and may never be presented as a live result.
10. **No claim of SOTA.** The repository claims reproducibility, calibration artifacts and citation verification — not state-of-the-art retrieval or generation.

## 9. Reproduction — REPLAY-MODE

```bash
uv run answerdesk eval --synthetic --provider fake --report reports\offline-smoke
```

Same seed + same split + same corpus ⇒ identical `metrics.json` after removing timing fields (NFR-DET-02). The deterministic digest above is computed over exactly that projection.