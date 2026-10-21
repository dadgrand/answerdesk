# answerdesk evaluation report — `offline-smoke`

> **REPLAY-MODE.** This run used deterministic doubles (FakeProvider / FakeJudge) and machine-labelled synthetic data. It validates the harness, not the system: no number here may be presented as a live-model or production result (§10.7 item 9, NFR-REP-01).

## 1. Run header — REPLAY-MODE

| field | value |
|---|---|
| run id | `offline-smoke` |
| schema / contracts | 1.0.0 |
| corpus version | `corpus-77d7d081813738a6` |
| corpus hash (sha256) | `ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36` |
| golden version | v1 |
| golden hash (sha256) | `ad1a9a3aeeaa88c75b7ae1725993dcb9856a640eb5e428b5576cde44aae73f13` |
| split version | v1 |
| split hash (sha256) | `f4738e34a3032806b5d94b212980436cb284d4feb920c9698cb9b7edceea6ae7` |
| seed | 20261021 |
| provider mode | fake (replay) |
| judge model | fake-judge |
| judge prompt version | none (deterministic double) |
| judge rubric version | fake-rubric-1 |
| judge position swap | on |
| judge κ reportable | no |
| arms | hybrid-rrf, bm25, dense-hash |
| locales | ky, ru, tg, tr, uz, zh |
| cases (test split) | 530 |
| wall clock | 9.8 s |
| hardware | Windows 11 / Python 3.12.10 |
| pricing table | configs/pricing.yaml |
| deterministic digest | `dfbb4924fae717b5fe68bbdbcab8ed8e379c36a9fa47c2c2db9ee753bc688a48` |

Every cell below carries `n` and a 95% interval. `method` says how the interval was computed: `wilson` for a proportion, `bootstrap` for a mean (B resamples, §10.4), `none` for a distribution summary such as p95 latency.

Judge note: generator model not recorded: cross-family judging cannot be established.

Run notes:

* position-swap audit: 8/8 comparisons kept their verdict across both orders (invariance 1.00, mean order effect 0.000)
* human agreement: no calibration labels on disk (evals/judge/human_v1.jsonl); judged metrics stay not-reportable (§7.5 F-10)

Every section below is marked **REPLAY-MODE**: a number produced by deterministic doubles is a diagnostic, never a live-model result.

## 2. Retrieval — REPLAY-MODE

### 2.1 Arm `hybrid-rrf` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.778 | [0.751, 0.804] | 464 | bootstrap |
| MRR@20 | 0.779 | [0.752, 0.805] | 464 | bootstrap |
| nDCG@10 | 0.703 | [0.679, 0.726] | 464 | bootstrap |
| nDCG@20 | 0.805 | [0.788, 0.823] | 464 | bootstrap |
| recall_all@10 | 0.730 | [0.708, 0.753] | 464 | bootstrap |
| recall_all@20 | 0.927 | [0.910, 0.942] | 464 | bootstrap |
| recall_any@10 | 0.980 | [0.967, 0.991] | 464 | bootstrap |
| recall_any@20 | 0.991 | [0.983, 0.998] | 464 | bootstrap |

### 2.2 Arm `hybrid-rrf` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.847 | [0.781, 0.910] | 74 |
| MRR@10 | ru | 0.852 | [0.786, 0.911] | 79 |
| MRR@10 | tg | 0.877 | [0.810, 0.935] | 74 |
| MRR@10 | tr | 0.831 | [0.766, 0.894] | 79 |
| MRR@10 | uz | 0.768 | [0.694, 0.841] | 79 |
| MRR@10 | zh | 0.491 | [0.434, 0.556] | 79 |
| MRR@20 | ky | 0.849 | [0.785, 0.912] | 74 |
| MRR@20 | ru | 0.852 | [0.786, 0.911] | 79 |
| MRR@20 | tg | 0.880 | [0.814, 0.936] | 74 |
| MRR@20 | tr | 0.831 | [0.766, 0.894] | 79 |
| MRR@20 | uz | 0.769 | [0.696, 0.841] | 79 |
| MRR@20 | zh | 0.491 | [0.434, 0.556] | 79 |
| nDCG@10 | ky | 0.755 | [0.693, 0.812] | 74 |
| nDCG@10 | ru | 0.773 | [0.714, 0.829] | 79 |
| nDCG@10 | tg | 0.797 | [0.726, 0.859] | 74 |
| nDCG@10 | tr | 0.758 | [0.698, 0.820] | 79 |
| nDCG@10 | uz | 0.667 | [0.604, 0.731] | 79 |
| nDCG@10 | zh | 0.464 | [0.420, 0.509] | 79 |
| nDCG@20 | ky | 0.841 | [0.801, 0.881] | 74 |
| nDCG@20 | ru | 0.859 | [0.814, 0.898] | 79 |
| nDCG@20 | tg | 0.866 | [0.814, 0.909] | 74 |
| nDCG@20 | tr | 0.840 | [0.794, 0.882] | 79 |
| nDCG@20 | uz | 0.776 | [0.722, 0.831] | 79 |
| nDCG@20 | zh | 0.649 | [0.611, 0.685] | 79 |
| recall_all@10 | ky | 0.758 | [0.697, 0.813] | 74 |
| recall_all@10 | ru | 0.782 | [0.726, 0.835] | 79 |
| recall_all@10 | tg | 0.796 | [0.724, 0.858] | 74 |
| recall_all@10 | tr | 0.765 | [0.706, 0.821] | 79 |
| recall_all@10 | uz | 0.705 | [0.647, 0.769] | 79 |
| recall_all@10 | zh | 0.574 | [0.530, 0.620] | 79 |
| recall_all@20 | ky | 0.924 | [0.892, 0.954] | 74 |
| recall_all@20 | ru | 0.946 | [0.908, 0.976] | 79 |
| recall_all@20 | tg | 0.925 | [0.883, 0.960] | 74 |
| recall_all@20 | tr | 0.920 | [0.883, 0.954] | 79 |
| recall_all@20 | uz | 0.915 | [0.866, 0.954] | 79 |
| recall_all@20 | zh | 0.931 | [0.890, 0.964] | 79 |
| recall_any@10 | ky | 0.973 | [0.932, 1.000] | 74 |
| recall_any@10 | ru | 0.987 | [0.962, 1.000] | 79 |
| recall_any@10 | tg | 0.959 | [0.905, 1.000] | 74 |
| recall_any@10 | tr | 0.987 | [0.962, 1.000] | 79 |
| recall_any@10 | uz | 0.987 | [0.962, 1.000] | 79 |
| recall_any@10 | zh | 0.987 | [0.949, 1.000] | 79 |
| recall_any@20 | ky | 1.000 | [1.000, 1.000] | 74 |
| recall_any@20 | ru | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | tg | 0.986 | [0.959, 1.000] | 74 |
| recall_any@20 | tr | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | uz | 1.000 | [1.000, 1.000] | 79 |
| recall_any@20 | zh | 0.987 | [0.949, 1.000] | 79 |

### 2.1 Arm `bm25` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.960 | [0.942, 0.976] | 464 | bootstrap |
| MRR@20 | 0.961 | [0.943, 0.977] | 464 | bootstrap |
| nDCG@10 | 0.931 | [0.913, 0.947] | 464 | bootstrap |
| nDCG@20 | 0.944 | [0.928, 0.959] | 464 | bootstrap |
| recall_all@10 | 0.927 | [0.907, 0.943] | 464 | bootstrap |
| recall_all@20 | 0.953 | [0.937, 0.967] | 464 | bootstrap |
| recall_any@10 | 0.976 | [0.961, 0.989] | 464 | bootstrap |
| recall_any@20 | 0.987 | [0.976, 0.996] | 464 | bootstrap |

### 2.2 Arm `bm25` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.950 | [0.896, 0.991] | 74 |
| MRR@10 | ru | 0.987 | [0.962, 1.000] | 79 |
| MRR@10 | tg | 0.959 | [0.919, 0.990] | 74 |
| MRR@10 | tr | 0.943 | [0.886, 0.987] | 79 |
| MRR@10 | uz | 0.968 | [0.924, 1.000] | 79 |
| MRR@10 | zh | 0.951 | [0.907, 0.985] | 79 |
| MRR@20 | ky | 0.950 | [0.896, 0.991] | 74 |
| MRR@20 | ru | 0.987 | [0.962, 1.000] | 79 |
| MRR@20 | tg | 0.959 | [0.919, 0.990] | 74 |
| MRR@20 | tr | 0.946 | [0.891, 0.988] | 79 |
| MRR@20 | uz | 0.969 | [0.927, 1.000] | 79 |
| MRR@20 | zh | 0.951 | [0.907, 0.985] | 79 |
| nDCG@10 | ky | 0.903 | [0.850, 0.947] | 74 |
| nDCG@10 | ru | 0.959 | [0.923, 0.985] | 79 |
| nDCG@10 | tg | 0.946 | [0.909, 0.977] | 74 |
| nDCG@10 | tr | 0.912 | [0.855, 0.957] | 79 |
| nDCG@10 | uz | 0.934 | [0.886, 0.971] | 79 |
| nDCG@10 | zh | 0.933 | [0.893, 0.964] | 79 |
| nDCG@20 | ky | 0.919 | [0.868, 0.962] | 74 |
| nDCG@20 | ru | 0.967 | [0.935, 0.990] | 79 |
| nDCG@20 | tg | 0.961 | [0.931, 0.985] | 74 |
| nDCG@20 | tr | 0.925 | [0.874, 0.965] | 79 |
| nDCG@20 | uz | 0.946 | [0.905, 0.977] | 79 |
| nDCG@20 | zh | 0.944 | [0.907, 0.972] | 79 |
| recall_all@10 | ky | 0.891 | [0.835, 0.936] | 74 |
| recall_all@10 | ru | 0.953 | [0.915, 0.982] | 79 |
| recall_all@10 | tg | 0.951 | [0.910, 0.981] | 74 |
| recall_all@10 | tr | 0.906 | [0.851, 0.952] | 79 |
| recall_all@10 | uz | 0.924 | [0.872, 0.966] | 79 |
| recall_all@10 | zh | 0.935 | [0.891, 0.968] | 79 |
| recall_all@20 | ky | 0.925 | [0.873, 0.969] | 74 |
| recall_all@20 | ru | 0.969 | [0.935, 0.994] | 79 |
| recall_all@20 | tg | 0.980 | [0.952, 0.999] | 74 |
| recall_all@20 | tr | 0.935 | [0.890, 0.973] | 79 |
| recall_all@20 | uz | 0.952 | [0.911, 0.983] | 79 |
| recall_all@20 | zh | 0.958 | [0.918, 0.986] | 79 |
| recall_any@10 | ky | 0.959 | [0.905, 1.000] | 74 |
| recall_any@10 | ru | 0.987 | [0.962, 1.000] | 79 |
| recall_any@10 | tg | 1.000 | [1.000, 1.000] | 74 |
| recall_any@10 | tr | 0.949 | [0.899, 0.987] | 79 |
| recall_any@10 | uz | 0.975 | [0.937, 1.000] | 79 |
| recall_any@10 | zh | 0.987 | [0.949, 1.000] | 79 |
| recall_any@20 | ky | 0.959 | [0.905, 1.000] | 74 |
| recall_any@20 | ru | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | tg | 1.000 | [1.000, 1.000] | 74 |
| recall_any@20 | tr | 1.000 | [1.000, 1.000] | 79 |
| recall_any@20 | uz | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | zh | 0.987 | [0.949, 1.000] | 79 |

### 2.1 Arm `dense-hash` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.348 | [0.308, 0.383] | 464 | bootstrap |
| MRR@20 | 0.353 | [0.315, 0.388] | 464 | bootstrap |
| nDCG@10 | 0.313 | [0.277, 0.346] | 464 | bootstrap |
| nDCG@20 | 0.351 | [0.316, 0.385] | 464 | bootstrap |
| recall_all@10 | 0.329 | [0.293, 0.363] | 464 | bootstrap |
| recall_all@20 | 0.402 | [0.366, 0.439] | 464 | bootstrap |
| recall_any@10 | 0.495 | [0.452, 0.536] | 464 | bootstrap |
| recall_any@20 | 0.577 | [0.538, 0.616] | 464 | bootstrap |

### 2.2 Arm `dense-hash` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.471 | [0.377, 0.566] | 74 |
| MRR@10 | ru | 0.411 | [0.318, 0.510] | 79 |
| MRR@10 | tg | 0.518 | [0.412, 0.623] | 74 |
| MRR@10 | tr | 0.345 | [0.256, 0.440] | 79 |
| MRR@10 | uz | 0.293 | [0.205, 0.393] | 79 |
| MRR@10 | zh | 0.049 | [0.014, 0.093] | 79 |
| MRR@20 | ky | 0.474 | [0.380, 0.567] | 74 |
| MRR@20 | ru | 0.421 | [0.328, 0.518] | 79 |
| MRR@20 | tg | 0.521 | [0.416, 0.624] | 74 |
| MRR@20 | tr | 0.356 | [0.269, 0.447] | 79 |
| MRR@20 | uz | 0.298 | [0.210, 0.397] | 79 |
| MRR@20 | zh | 0.051 | [0.016, 0.095] | 79 |
| nDCG@10 | ky | 0.422 | [0.336, 0.506] | 74 |
| nDCG@10 | ru | 0.387 | [0.297, 0.478] | 79 |
| nDCG@10 | tg | 0.469 | [0.374, 0.560] | 74 |
| nDCG@10 | tr | 0.292 | [0.213, 0.371] | 79 |
| nDCG@10 | uz | 0.263 | [0.182, 0.350] | 79 |
| nDCG@10 | zh | 0.047 | [0.012, 0.090] | 79 |
| nDCG@20 | ky | 0.465 | [0.381, 0.552] | 74 |
| nDCG@20 | ru | 0.423 | [0.332, 0.515] | 79 |
| nDCG@20 | tg | 0.522 | [0.424, 0.615] | 74 |
| nDCG@20 | tr | 0.349 | [0.268, 0.430] | 79 |
| nDCG@20 | uz | 0.294 | [0.210, 0.388] | 79 |
| nDCG@20 | zh | 0.053 | [0.016, 0.096] | 79 |
| recall_all@10 | ky | 0.438 | [0.353, 0.522] | 74 |
| recall_all@10 | ru | 0.402 | [0.311, 0.496] | 79 |
| recall_all@10 | tg | 0.486 | [0.390, 0.579] | 74 |
| recall_all@10 | tr | 0.312 | [0.227, 0.395] | 79 |
| recall_all@10 | uz | 0.276 | [0.194, 0.367] | 79 |
| recall_all@10 | zh | 0.059 | [0.016, 0.109] | 79 |
| recall_all@20 | ky | 0.522 | [0.427, 0.616] | 74 |
| recall_all@20 | ru | 0.473 | [0.377, 0.565] | 79 |
| recall_all@20 | tg | 0.586 | [0.479, 0.687] | 74 |
| recall_all@20 | tr | 0.423 | [0.333, 0.511] | 79 |
| recall_all@20 | uz | 0.334 | [0.245, 0.433] | 79 |
| recall_all@20 | zh | 0.075 | [0.026, 0.129] | 79 |
| recall_any@10 | ky | 0.662 | [0.568, 0.770] | 74 |
| recall_any@10 | ru | 0.544 | [0.430, 0.658] | 79 |
| recall_any@10 | tg | 0.703 | [0.581, 0.811] | 74 |
| recall_any@10 | tr | 0.544 | [0.430, 0.658] | 79 |
| recall_any@10 | uz | 0.418 | [0.316, 0.532] | 79 |
| recall_any@10 | zh | 0.101 | [0.038, 0.177] | 79 |
| recall_any@20 | ky | 0.703 | [0.595, 0.811] | 74 |
| recall_any@20 | ru | 0.684 | [0.582, 0.785] | 79 |
| recall_any@20 | tg | 0.757 | [0.649, 0.851] | 74 |
| recall_any@20 | tr | 0.696 | [0.595, 0.797] | 79 |
| recall_any@20 | uz | 0.481 | [0.380, 0.595] | 79 |
| recall_any@20 | zh | 0.139 | [0.063, 0.228] | 79 |

## 3. Answer quality — REPLAY-MODE

| arm / metric | value | 95% CI | n | method |
|---|---|---|---|---|
| hybrid-rrf / groundedness_verifier | 1.000 | [1.000, 1.000] | 390 | bootstrap |
| hybrid-rrf / groundedness_claim_pooled | 1.000 | [0.994, 1.000] | 689 | wilson |
| hybrid-rrf / citation_precision | 1.000 | [1.000, 1.000] | 390 | bootstrap |
| hybrid-rrf / citation_recall | 1.000 | [1.000, 1.000] | 390 | bootstrap |
| hybrid-rrf / rubric_score | 5.000 | [5.000, 5.000] | 390 | bootstrap |
| hybrid-rrf / rubric_pass_rate | 1.000 | [0.990, 1.000] | 390 | wilson |
| hybrid-rrf / phantom_citations | 0.000 | [0.000, 0.010] | 390 | wilson |
| hybrid-rrf / unsupported_displayed_claims | 0.000 | [0.000, 0.010] | 390 | wilson |
| hybrid-rrf / language_agreement | 0.949 | [0.922, 0.967] | 390 | wilson |
| hybrid-rrf / answer_accuracy_by_category [direct] | 0.929 | [0.870, 0.962] | 126 | wilson |
| hybrid-rrf / answer_accuracy_by_category [numeric_date] | 0.542 | [0.427, 0.652] | 72 | wilson |
| hybrid-rrf / answer_accuracy_by_category [typo_code_switch] | 0.902 | [0.856, 0.934] | 224 | wilson |
| hybrid-rrf / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| hybrid-rrf / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| hybrid-rrf / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| bm25 / groundedness_verifier | 1.000 | [1.000, 1.000] | 381 | bootstrap |
| bm25 / groundedness_claim_pooled | 1.000 | [0.994, 1.000] | 694 | wilson |
| bm25 / citation_precision | 1.000 | [1.000, 1.000] | 381 | bootstrap |
| bm25 / citation_recall | 1.000 | [1.000, 1.000] | 381 | bootstrap |
| bm25 / rubric_score | 5.000 | [5.000, 5.000] | 381 | bootstrap |
| bm25 / rubric_pass_rate | 1.000 | [0.990, 1.000] | 381 | wilson |
| bm25 / phantom_citations | 0.000 | [0.000, 0.010] | 381 | wilson |
| bm25 / unsupported_displayed_claims | 0.000 | [0.000, 0.010] | 381 | wilson |
| bm25 / language_agreement | 0.950 | [0.923, 0.968] | 381 | wilson |
| bm25 / answer_accuracy_by_category [direct] | 0.937 | [0.880, 0.967] | 126 | wilson |
| bm25 / answer_accuracy_by_category [numeric_date] | 0.472 | [0.361, 0.586] | 72 | wilson |
| bm25 / answer_accuracy_by_category [typo_code_switch] | 0.848 | [0.795, 0.889] | 224 | wilson |
| bm25 / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| bm25 / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| bm25 / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| dense-hash / groundedness_verifier | 1.000 | [1.000, 1.000] | 320 | bootstrap |
| dense-hash / groundedness_claim_pooled | 1.000 | [0.992, 1.000] | 479 | wilson |
| dense-hash / citation_precision | 1.000 | [1.000, 1.000] | 320 | bootstrap |
| dense-hash / citation_recall | 1.000 | [1.000, 1.000] | 320 | bootstrap |
| dense-hash / rubric_score | 5.000 | [5.000, 5.000] | 320 | bootstrap |
| dense-hash / rubric_pass_rate | 1.000 | [0.988, 1.000] | 320 | wilson |
| dense-hash / phantom_citations | 0.000 | [0.000, 0.012] | 320 | wilson |
| dense-hash / unsupported_displayed_claims | 0.000 | [0.000, 0.012] | 320 | wilson |
| dense-hash / language_agreement | 0.950 | [0.920, 0.969] | 320 | wilson |
| dense-hash / answer_accuracy_by_category [direct] | 0.611 | [0.524, 0.692] | 126 | wilson |
| dense-hash / answer_accuracy_by_category [numeric_date] | 0.250 | [0.164, 0.361] | 72 | wilson |
| dense-hash / answer_accuracy_by_category [typo_code_switch] | 0.478 | [0.413, 0.543] | 224 | wilson |
| dense-hash / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| dense-hash / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| dense-hash / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |

`phantom_citations` and `unsupported_displayed_claims` are structural counts (F-4/F-5): they are reported as a rate over answered turns and must be 0.

## 4. Escalation and calibration — REPLAY-MODE

| arm / metric | value | 95% CI | n | note |
|---|---|---|---|---|
| hybrid-rrf / realised_risk | 0.082 | [0.059, 0.114] | 390 | 32/390 auto-answered turns were incorrect (E-1) |
| hybrid-rrf / coverage | 0.924 | [0.895, 0.946] | 422 | auto-answer coverage on answerable items (E-4) |
| hybrid-rrf / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| hybrid-rrf / escalation_precision | 0.771 | [0.695, 0.833] | 140 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| hybrid-rrf / realised_risk_by_bucket [informational] | 0.082 | [0.059, 0.114] | 390 | intent bucket informational (E-1 per bucket) |
| hybrid-rrf / coverage_by_bucket [informational] | 0.924 | [0.895, 0.946] | 422 | intent bucket informational |
| bm25 / realised_risk | 0.102 | [0.076, 0.137] | 381 | 39/381 auto-answered turns were incorrect (E-1) |
| bm25 / coverage | 0.903 | [0.871, 0.928] | 422 | auto-answer coverage on answerable items (E-4) |
| bm25 / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| bm25 / escalation_precision | 0.725 | [0.648, 0.790] | 149 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| bm25 / realised_risk_by_bucket [informational] | 0.102 | [0.076, 0.137] | 381 | intent bucket informational (E-1 per bucket) |
| bm25 / coverage_by_bucket [informational] | 0.903 | [0.871, 0.928] | 422 | intent bucket informational |
| dense-hash / realised_risk | 0.369 | [0.318, 0.423] | 320 | 118/320 auto-answered turns were incorrect (E-1) |
| dense-hash / coverage | 0.758 | [0.715, 0.797] | 422 | auto-answer coverage on answerable items (E-4) |
| dense-hash / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| dense-hash / escalation_precision | 0.514 | [0.447, 0.581] | 210 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| dense-hash / realised_risk_by_bucket [informational] | 0.369 | [0.318, 0.423] | 320 | intent bucket informational (E-1 per bucket) |
| dense-hash / coverage_by_bucket [informational] | 0.758 | [0.715, 0.797] | 422 | intent bucket informational |

### 4.1 Reliability (E-5/E-6 diagnostic)

| quantity | value |
|---|---|
| n | 530 |
| ECE equal-width | 0.655 |
| ECE equal-mass | 0.655 |
| Brier | 0.638 |
| AURC | 0.101 |
| mean confidence | 0.021 |
| accuracy | 0.675 |
| coverage at target risk | 0.194 |
| risk at target | 0.049 |

`p̂` here is the packed evidence score, not a conformal calibrator (t8 owns conformal prediction), so E-5 is a diagnostic and no risk guarantee is claimed.

## 5. Operations — REPLAY-MODE

| arm / metric | value | bracket | n | method |
|---|---|---|---|---|
| hybrid-rrf / latency_generation_p50_ms | 1.9705 | [1.9156, 2.0208] | 530 | none |
| hybrid-rrf / latency_generation_p95_ms | 2.9602 | [2.7362, 3.7625] | 530 | none |
| hybrid-rrf / latency_judge_p50_ms | 0.0440 | [0.0415, 0.0462] | 390 | none |
| hybrid-rrf / latency_judge_p95_ms | 0.0655 | [0.0615, 0.1173] | 390 | none |
| hybrid-rrf / latency_retrieval_p50_ms | 5.7982 | [5.6625, 6.0675] | 530 | none |
| hybrid-rrf / latency_retrieval_p95_ms | 9.1480 | [8.2666, 43.2299] | 530 | none |
| hybrid-rrf / latency_total_p50_ms | 7.9800 | [7.7551, 8.1889] | 530 | none |
| hybrid-rrf / latency_total_p95_ms | 11.3650 | [10.3423, 46.1813] | 530 | none |
| hybrid-rrf / latency_total_mean_ms | 8.3297 | [8.1629, 8.5346] | 530 | bootstrap |
| hybrid-rrf / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 530 | bootstrap |
| hybrid-rrf / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 530 | none |
| hybrid-rrf / tokens_input_mean | 692.0660 | [671.0680, 710.6882] | 530 | bootstrap |
| hybrid-rrf / tokens_output_mean | 36.8604 | [34.9601, 38.7529] | 530 | bootstrap |
| hybrid-rrf / cache_hit_rate | 0.1132 | [0.0890, 0.1430] | 530 | wilson |
| bm25 / latency_generation_p50_ms | 1.7444 | [1.6482, 1.8400] | 530 | none |
| bm25 / latency_generation_p95_ms | 2.8625 | [2.7035, 3.5293] | 530 | none |
| bm25 / latency_judge_p50_ms | 0.0450 | [0.0436, 0.0464] | 381 | none |
| bm25 / latency_judge_p95_ms | 0.0633 | [0.0588, 0.1411] | 381 | none |
| bm25 / latency_retrieval_p50_ms | 0.7772 | [0.5975, 0.9563] | 530 | none |
| bm25 / latency_retrieval_p95_ms | 3.1511 | [2.7525, 4.2906] | 530 | none |
| bm25 / latency_total_p50_ms | 2.7279 | [2.6482, 2.8748] | 530 | none |
| bm25 / latency_total_p95_ms | 5.1747 | [4.7093, 6.7162] | 530 | none |
| bm25 / latency_total_mean_ms | 3.0021 | [2.9132, 3.0961] | 530 | bootstrap |
| bm25 / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 530 | bootstrap |
| bm25 / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 530 | none |
| bm25 / tokens_input_mean | 640.8245 | [620.1295, 660.3575] | 530 | bootstrap |
| bm25 / tokens_output_mean | 37.5019 | [35.4467, 39.6029] | 530 | bootstrap |
| bm25 / cache_hit_rate | 0.1132 | [0.0890, 0.1430] | 530 | wilson |
| dense-hash / latency_generation_p50_ms | 2.2835 | [2.1988, 2.3676] | 530 | none |
| dense-hash / latency_generation_p95_ms | 3.3709 | [3.0511, 4.5503] | 530 | none |
| dense-hash / latency_judge_p50_ms | 0.0335 | [0.0318, 0.0367] | 320 | none |
| dense-hash / latency_judge_p95_ms | 0.0692 | [0.0612, 0.1155] | 320 | none |
| dense-hash / latency_retrieval_p50_ms | 4.6043 | [4.5248, 4.7092] | 530 | none |
| dense-hash / latency_retrieval_p95_ms | 5.8799 | [5.5056, 8.5853] | 530 | none |
| dense-hash / latency_total_p50_ms | 6.8209 | [6.7316, 6.9339] | 530 | none |
| dense-hash / latency_total_p95_ms | 8.8960 | [8.3192, 11.0317] | 530 | none |
| dense-hash / latency_total_mean_ms | 6.9540 | [6.8635, 7.0445] | 530 | bootstrap |
| dense-hash / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 530 | bootstrap |
| dense-hash / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 530 | none |
| dense-hash / tokens_input_mean | 791.6547 | [767.3150, 816.0482] | 530 | bootstrap |
| dense-hash / tokens_output_mean | 33.6113 | [30.7750, 37.1374] | 530 | bootstrap |
| dense-hash / cache_hit_rate | 0.1132 | [0.0890, 0.1430] | 530 | wilson |

Latency is measured in-process around the pipeline call, so it is a regression guard (NFR-LAT-03) and not a server-side latency claim (NFR-LAT-01). The bracket for a percentile is the p(q-5)..p(q+5) spread, not a confidence interval.

## 6. Statistics: paired bootstrap and McNemar — REPLAY-MODE

| metric | arms | n | mean A | mean B | Δ | Δ 95% CI (paired bootstrap) | McNemar | p (method) | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|---|
| recall_any@10 | hybrid-rrf − bm25 | 464 | 0.981 | 0.976 | 0.004 | [-0.011, 0.019] | b=8, c=6 | 0.7905 (exact_binomial) | no |
| recall_any@20 | hybrid-rrf − bm25 | 464 | 0.991 | 0.987 | 0.004 | [-0.006, 0.015] | b=4, c=2 | 0.6875 (exact_binomial) | no |
| answer_ok | hybrid-rrf − bm25 | 530 | 0.879 | 0.849 | 0.030 | [0.013, 0.051] | b=23, c=7 | 0.0062 (chi2) | yes |
| recall_any@10 | hybrid-rrf − dense-hash | 464 | 0.981 | 0.491 | 0.489 | [0.448, 0.530] | b=227, c=0 | 0.0000 (chi2) | yes |
| recall_any@20 | hybrid-rrf − dense-hash | 464 | 0.991 | 0.573 | 0.418 | [0.373, 0.463] | b=194, c=0 | 0.0000 (chi2) | yes |
| answer_ok | hybrid-rrf − dense-hash | 530 | 0.879 | 0.585 | 0.294 | [0.255, 0.338] | b=162, c=6 | 0.0000 (chi2) | yes |
| recall_any@10 | bm25 − dense-hash | 464 | 0.976 | 0.491 | 0.485 | [0.442, 0.530] | b=230, c=5 | 0.0000 (chi2) | yes |
| recall_any@20 | bm25 − dense-hash | 464 | 0.987 | 0.573 | 0.414 | [0.369, 0.459] | b=195, c=3 | 0.0000 (chi2) | yes |
| answer_ok | bm25 − dense-hash | 530 | 0.849 | 0.585 | 0.264 | [0.223, 0.309] | b=157, c=17 | 0.0000 (chi2) | yes |

Paired bootstrap resamples **queries**, not arms (B = 1000); McNemar uses the χ² approximation with the exact-binomial fallback when b + c < 25. A difference whose interval contains 0 is noise and is reported as such (§6.3 step 5).

## 7. Targets — REPLAY-MODE

| id | target | threshold | measured | verdict | note |
|---|---|---|---|---|---|
| RT-1 | recall_any@20 (hybrid, tuned) | >= 0.90 | 0.991 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-3 | recall_any@10 | >= 0.80 | 0.980 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-4 | recall_all@20 | >= 0.75 | 0.927 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-5 | nDCG@10 | >= 0.65 | 0.703 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-6 | MRR@10 | >= 0.60 | 0.778 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-2 | recall_any@20 per-locale floor | >= 0.80 every locale | 0.986 | diagnostic | locales: ky, ru, tg, tr, uz, zh |
| F-1 | groundedness (claim support rate) | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-2 | citation precision | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-3 | citation recall | >= 0.85 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| E-4 | auto-answer coverage | >= 0.60 | 0.924 | diagnostic | diagnostic: the policy layer is t8's |
| E-2 | must-escalate recall | >= 0.95 | 1.000 | diagnostic | diagnostic: the policy layer is t8's |
| E-3 | escalation precision | >= 0.55 | 0.771 | diagnostic | diagnostic: the policy layer is t8's |
| E-1 | realised auto-answer risk vs target | <= 0.05 (pooled <= 0.05) | 0.082 | diagnostic | diagnostic: no conformal calibration in this run (t8) |
| E-5 | calibration error of p̂ | ECE <= 0.08 and Brier <= 0.15 | 0.655 | diagnostic | diagnostic: p̂ is the evidence score, not a conformal calibrator; Brier=0.6381, ECE(mass)=0.6548 |

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