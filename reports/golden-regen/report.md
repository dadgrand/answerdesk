# answerdesk evaluation report — `golden-regen`

> **REPLAY-MODE.** This run used deterministic doubles (FakeProvider / FakeJudge) and machine-labelled synthetic data. It validates the harness, not the system: no number here may be presented as a live-model or production result (§10.7 item 9, NFR-REP-01).

## 1. Run header — REPLAY-MODE

| field | value |
|---|---|
| run id | `golden-regen` |
| schema / contracts | 1.0.0 |
| corpus version | `corpus-96f93db4f7f648b5` |
| corpus hash (sha256) | `7ef3fc20400131dc966904bc68d517a805faf48e98b62584f44e086270e246d6` |
| golden version | v1 |
| golden hash (sha256) | `a2bfd45e45ffc62ac18671981a61e4abb826722451ac7ed5680ecf9863bb7ac2` |
| split version | v1 |
| split hash (sha256) | `5e57276f1c92ad3af6c99b13a8664d7d379b66985f5f2cc3400678d53a75df43` |
| seed | 20261021 |
| provider mode | fake (replay) |
| judge model | fake-judge |
| judge prompt version | none (deterministic double) |
| judge rubric version | fake-rubric-1 |
| judge position swap | on |
| judge κ reportable | no |
| arms | hybrid-rrf, bm25, dense-hash |
| locales | ky, ru, tg, tr, uz, zh |
| cases (test split) | 540 |
| wall clock | 9.9 s |
| hardware | Windows 11 / Python 3.12.10 |
| pricing table | configs/pricing.yaml |
| deterministic digest | `701cc7af2e13b47a239517f283bec798aaee904c1956f82829ba92549aade320` |

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
| MRR@10 | 0.596 | [0.561, 0.629] | 474 | bootstrap |
| MRR@20 | 0.602 | [0.568, 0.634] | 474 | bootstrap |
| nDCG@10 | 0.583 | [0.554, 0.612] | 474 | bootstrap |
| nDCG@20 | 0.669 | [0.643, 0.695] | 474 | bootstrap |
| recall_all@10 | 0.706 | [0.676, 0.735] | 474 | bootstrap |
| recall_all@20 | 0.904 | [0.885, 0.925] | 474 | bootstrap |
| recall_any@10 | 0.871 | [0.842, 0.901] | 474 | bootstrap |
| recall_any@20 | 0.951 | [0.932, 0.970] | 474 | bootstrap |

### 2.2 Arm `hybrid-rrf` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.719 | [0.628, 0.804] | 79 |
| MRR@10 | ru | 0.638 | [0.554, 0.723] | 79 |
| MRR@10 | tg | 0.670 | [0.575, 0.759] | 79 |
| MRR@10 | tr | 0.569 | [0.485, 0.660] | 79 |
| MRR@10 | uz | 0.626 | [0.541, 0.717] | 79 |
| MRR@10 | zh | 0.353 | [0.286, 0.425] | 79 |
| MRR@20 | ky | 0.723 | [0.635, 0.806] | 79 |
| MRR@20 | ru | 0.643 | [0.560, 0.725] | 79 |
| MRR@20 | tg | 0.674 | [0.582, 0.761] | 79 |
| MRR@20 | tr | 0.573 | [0.489, 0.663] | 79 |
| MRR@20 | uz | 0.632 | [0.549, 0.720] | 79 |
| MRR@20 | zh | 0.365 | [0.301, 0.433] | 79 |
| nDCG@10 | ky | 0.673 | [0.593, 0.746] | 79 |
| nDCG@10 | ru | 0.628 | [0.552, 0.698] | 79 |
| nDCG@10 | tg | 0.658 | [0.577, 0.731] | 79 |
| nDCG@10 | tr | 0.601 | [0.527, 0.674] | 79 |
| nDCG@10 | uz | 0.583 | [0.505, 0.668] | 79 |
| nDCG@10 | zh | 0.354 | [0.303, 0.406] | 79 |
| nDCG@20 | ky | 0.731 | [0.663, 0.794] | 79 |
| nDCG@20 | ru | 0.715 | [0.655, 0.778] | 79 |
| nDCG@20 | tg | 0.717 | [0.641, 0.786] | 79 |
| nDCG@20 | tr | 0.673 | [0.610, 0.742] | 79 |
| nDCG@20 | uz | 0.674 | [0.604, 0.743] | 79 |
| nDCG@20 | zh | 0.502 | [0.450, 0.551] | 79 |
| recall_all@10 | ky | 0.753 | [0.673, 0.822] | 79 |
| recall_all@10 | ru | 0.744 | [0.671, 0.807] | 79 |
| recall_all@10 | tg | 0.748 | [0.663, 0.830] | 79 |
| recall_all@10 | tr | 0.786 | [0.714, 0.856] | 79 |
| recall_all@10 | uz | 0.671 | [0.591, 0.754] | 79 |
| recall_all@10 | zh | 0.535 | [0.452, 0.615] | 79 |
| recall_all@20 | ky | 0.876 | [0.814, 0.926] | 79 |
| recall_all@20 | ru | 0.931 | [0.886, 0.970] | 79 |
| recall_all@20 | tg | 0.881 | [0.813, 0.937] | 79 |
| recall_all@20 | tr | 0.949 | [0.910, 0.977] | 79 |
| recall_all@20 | uz | 0.883 | [0.822, 0.941] | 79 |
| recall_all@20 | zh | 0.905 | [0.846, 0.955] | 79 |
| recall_any@10 | ky | 0.886 | [0.810, 0.949] | 79 |
| recall_any@10 | ru | 0.911 | [0.848, 0.962] | 79 |
| recall_any@10 | tg | 0.873 | [0.785, 0.949] | 79 |
| recall_any@10 | tr | 0.924 | [0.861, 0.975] | 79 |
| recall_any@10 | uz | 0.848 | [0.759, 0.924] | 79 |
| recall_any@10 | zh | 0.785 | [0.684, 0.873] | 79 |
| recall_any@20 | ky | 0.937 | [0.873, 0.987] | 79 |
| recall_any@20 | ru | 0.975 | [0.937, 1.000] | 79 |
| recall_any@20 | tg | 0.924 | [0.861, 0.975] | 79 |
| recall_any@20 | tr | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | uz | 0.937 | [0.886, 0.987] | 79 |
| recall_any@20 | zh | 0.949 | [0.899, 0.987] | 79 |

### 2.1 Arm `bm25` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.764 | [0.732, 0.797] | 474 | bootstrap |
| MRR@20 | 0.766 | [0.733, 0.798] | 474 | bootstrap |
| nDCG@10 | 0.782 | [0.755, 0.809] | 474 | bootstrap |
| nDCG@20 | 0.793 | [0.766, 0.819] | 474 | bootstrap |
| recall_all@10 | 0.919 | [0.898, 0.939] | 474 | bootstrap |
| recall_all@20 | 0.950 | [0.932, 0.965] | 474 | bootstrap |
| recall_any@10 | 0.956 | [0.937, 0.973] | 474 | bootstrap |
| recall_any@20 | 0.975 | [0.960, 0.987] | 474 | bootstrap |

### 2.2 Arm `bm25` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.814 | [0.740, 0.887] | 79 |
| MRR@10 | ru | 0.806 | [0.727, 0.881] | 79 |
| MRR@10 | tg | 0.764 | [0.683, 0.844] | 79 |
| MRR@10 | tr | 0.722 | [0.635, 0.804] | 79 |
| MRR@10 | uz | 0.782 | [0.706, 0.856] | 79 |
| MRR@10 | zh | 0.698 | [0.610, 0.780] | 79 |
| MRR@20 | ky | 0.817 | [0.744, 0.889] | 79 |
| MRR@20 | ru | 0.806 | [0.727, 0.881] | 79 |
| MRR@20 | tg | 0.764 | [0.683, 0.844] | 79 |
| MRR@20 | tr | 0.724 | [0.638, 0.805] | 79 |
| MRR@20 | uz | 0.784 | [0.707, 0.857] | 79 |
| MRR@20 | zh | 0.699 | [0.611, 0.780] | 79 |
| nDCG@10 | ky | 0.805 | [0.741, 0.869] | 79 |
| nDCG@10 | ru | 0.823 | [0.759, 0.886] | 79 |
| nDCG@10 | tg | 0.787 | [0.723, 0.851] | 79 |
| nDCG@10 | tr | 0.758 | [0.688, 0.825] | 79 |
| nDCG@10 | uz | 0.794 | [0.724, 0.863] | 79 |
| nDCG@10 | zh | 0.726 | [0.654, 0.791] | 79 |
| nDCG@20 | ky | 0.825 | [0.768, 0.882] | 79 |
| nDCG@20 | ru | 0.825 | [0.762, 0.886] | 79 |
| nDCG@20 | tg | 0.799 | [0.733, 0.863] | 79 |
| nDCG@20 | tr | 0.770 | [0.705, 0.831] | 79 |
| nDCG@20 | uz | 0.801 | [0.735, 0.866] | 79 |
| nDCG@20 | zh | 0.738 | [0.668, 0.803] | 79 |
| recall_all@10 | ky | 0.883 | [0.824, 0.932] | 79 |
| recall_all@10 | ru | 0.942 | [0.895, 0.978] | 79 |
| recall_all@10 | tg | 0.940 | [0.893, 0.979] | 79 |
| recall_all@10 | tr | 0.932 | [0.882, 0.971] | 79 |
| recall_all@10 | uz | 0.894 | [0.828, 0.952] | 79 |
| recall_all@10 | zh | 0.923 | [0.868, 0.972] | 79 |
| recall_all@20 | ky | 0.949 | [0.915, 0.979] | 79 |
| recall_all@20 | ru | 0.946 | [0.900, 0.983] | 79 |
| recall_all@20 | tg | 0.965 | [0.924, 1.000] | 79 |
| recall_all@20 | tr | 0.969 | [0.932, 0.993] | 79 |
| recall_all@20 | uz | 0.917 | [0.860, 0.967] | 79 |
| recall_all@20 | zh | 0.953 | [0.909, 0.992] | 79 |
| recall_any@10 | ky | 0.937 | [0.873, 0.987] | 79 |
| recall_any@10 | ru | 0.975 | [0.937, 1.000] | 79 |
| recall_any@10 | tg | 0.975 | [0.937, 1.000] | 79 |
| recall_any@10 | tr | 0.962 | [0.911, 1.000] | 79 |
| recall_any@10 | uz | 0.924 | [0.861, 0.975] | 79 |
| recall_any@10 | zh | 0.962 | [0.911, 1.000] | 79 |
| recall_any@20 | ky | 0.987 | [0.962, 1.000] | 79 |
| recall_any@20 | ru | 0.975 | [0.937, 1.000] | 79 |
| recall_any@20 | tg | 0.975 | [0.937, 1.000] | 79 |
| recall_any@20 | tr | 0.987 | [0.949, 1.000] | 79 |
| recall_any@20 | uz | 0.949 | [0.899, 0.987] | 79 |
| recall_any@20 | zh | 0.975 | [0.937, 1.000] | 79 |

### 2.1 Arm `dense-hash` — macro over locales

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| MRR@10 | 0.291 | [0.255, 0.327] | 474 | bootstrap |
| MRR@20 | 0.295 | [0.260, 0.332] | 474 | bootstrap |
| nDCG@10 | 0.275 | [0.243, 0.307] | 474 | bootstrap |
| nDCG@20 | 0.307 | [0.275, 0.339] | 474 | bootstrap |
| recall_all@10 | 0.324 | [0.290, 0.360] | 474 | bootstrap |
| recall_all@20 | 0.402 | [0.366, 0.438] | 474 | bootstrap |
| recall_any@10 | 0.443 | [0.401, 0.485] | 474 | bootstrap |
| recall_any@20 | 0.506 | [0.466, 0.546] | 474 | bootstrap |

### 2.2 Arm `dense-hash` — per locale (§6.3 step 4, NFR-ML-01)

| metric | locale | value | 95% CI | n |
|---|---|---|---|---|
| MRR@10 | ky | 0.348 | [0.250, 0.443] | 79 |
| MRR@10 | ru | 0.378 | [0.285, 0.477] | 79 |
| MRR@10 | tg | 0.392 | [0.298, 0.488] | 79 |
| MRR@10 | tr | 0.282 | [0.205, 0.373] | 79 |
| MRR@10 | uz | 0.302 | [0.217, 0.396] | 79 |
| MRR@10 | zh | 0.043 | [0.009, 0.085] | 79 |
| MRR@20 | ky | 0.357 | [0.262, 0.448] | 79 |
| MRR@20 | ru | 0.385 | [0.292, 0.483] | 79 |
| MRR@20 | tg | 0.394 | [0.300, 0.491] | 79 |
| MRR@20 | tr | 0.286 | [0.207, 0.378] | 79 |
| MRR@20 | uz | 0.305 | [0.221, 0.400] | 79 |
| MRR@20 | zh | 0.044 | [0.010, 0.086] | 79 |
| nDCG@10 | ky | 0.314 | [0.228, 0.399] | 79 |
| nDCG@10 | ru | 0.361 | [0.275, 0.452] | 79 |
| nDCG@10 | tg | 0.397 | [0.306, 0.485] | 79 |
| nDCG@10 | tr | 0.271 | [0.203, 0.349] | 79 |
| nDCG@10 | uz | 0.272 | [0.193, 0.353] | 79 |
| nDCG@10 | zh | 0.034 | [0.007, 0.069] | 79 |
| nDCG@20 | ky | 0.366 | [0.281, 0.451] | 79 |
| nDCG@20 | ru | 0.386 | [0.300, 0.472] | 79 |
| nDCG@20 | tg | 0.424 | [0.331, 0.517] | 79 |
| nDCG@20 | tr | 0.323 | [0.251, 0.404] | 79 |
| nDCG@20 | uz | 0.305 | [0.226, 0.392] | 79 |
| nDCG@20 | zh | 0.036 | [0.008, 0.070] | 79 |
| recall_all@10 | ky | 0.333 | [0.246, 0.422] | 79 |
| recall_all@10 | ru | 0.412 | [0.319, 0.510] | 79 |
| recall_all@10 | tg | 0.499 | [0.387, 0.601] | 79 |
| recall_all@10 | tr | 0.350 | [0.268, 0.441] | 79 |
| recall_all@10 | uz | 0.306 | [0.219, 0.395] | 79 |
| recall_all@10 | zh | 0.043 | [0.008, 0.088] | 79 |
| recall_all@20 | ky | 0.480 | [0.384, 0.572] | 79 |
| recall_all@20 | ru | 0.479 | [0.381, 0.574] | 79 |
| recall_all@20 | tg | 0.562 | [0.452, 0.667] | 79 |
| recall_all@20 | tr | 0.474 | [0.380, 0.574] | 79 |
| recall_all@20 | uz | 0.369 | [0.279, 0.466] | 79 |
| recall_all@20 | zh | 0.051 | [0.010, 0.098] | 79 |
| recall_any@10 | ky | 0.481 | [0.367, 0.595] | 79 |
| recall_any@10 | ru | 0.494 | [0.380, 0.608] | 79 |
| recall_any@10 | tg | 0.608 | [0.494, 0.722] | 79 |
| recall_any@10 | tr | 0.532 | [0.430, 0.646] | 79 |
| recall_any@10 | uz | 0.468 | [0.354, 0.570] | 79 |
| recall_any@10 | zh | 0.076 | [0.025, 0.139] | 79 |
| recall_any@20 | ky | 0.620 | [0.506, 0.722] | 79 |
| recall_any@20 | ru | 0.582 | [0.468, 0.696] | 79 |
| recall_any@20 | tg | 0.646 | [0.532, 0.747] | 79 |
| recall_any@20 | tr | 0.595 | [0.481, 0.709] | 79 |
| recall_any@20 | uz | 0.506 | [0.405, 0.608] | 79 |
| recall_any@20 | zh | 0.089 | [0.025, 0.152] | 79 |

## 3. Answer quality — REPLAY-MODE

| arm / metric | value | 95% CI | n | method |
|---|---|---|---|---|
| hybrid-rrf / groundedness_verifier | 1.000 | [1.000, 1.000] | 402 | bootstrap |
| hybrid-rrf / groundedness_claim_pooled | 1.000 | [0.995, 1.000] | 714 | wilson |
| hybrid-rrf / citation_precision | 1.000 | [1.000, 1.000] | 402 | bootstrap |
| hybrid-rrf / citation_recall | 1.000 | [1.000, 1.000] | 402 | bootstrap |
| hybrid-rrf / rubric_score | 5.000 | [5.000, 5.000] | 402 | bootstrap |
| hybrid-rrf / rubric_pass_rate | 1.000 | [0.991, 1.000] | 402 | wilson |
| hybrid-rrf / phantom_citations | 0.000 | [0.000, 0.009] | 402 | wilson |
| hybrid-rrf / unsupported_displayed_claims | 0.000 | [0.000, 0.009] | 402 | wilson |
| hybrid-rrf / language_agreement | 0.945 | [0.919, 0.964] | 402 | wilson |
| hybrid-rrf / answer_accuracy_by_category [direct] | 0.960 | [0.910, 0.983] | 126 | wilson |
| hybrid-rrf / answer_accuracy_by_category [numeric_date] | 0.597 | [0.482, 0.703] | 72 | wilson |
| hybrid-rrf / answer_accuracy_by_category [typo_code_switch] | 0.415 | [0.353, 0.479] | 234 | wilson |
| hybrid-rrf / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| hybrid-rrf / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| hybrid-rrf / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| bm25 / groundedness_verifier | 1.000 | [1.000, 1.000] | 398 | bootstrap |
| bm25 / groundedness_claim_pooled | 1.000 | [0.995, 1.000] | 736 | wilson |
| bm25 / citation_precision | 1.000 | [1.000, 1.000] | 398 | bootstrap |
| bm25 / citation_recall | 1.000 | [1.000, 1.000] | 398 | bootstrap |
| bm25 / rubric_score | 5.000 | [5.000, 5.000] | 398 | bootstrap |
| bm25 / rubric_pass_rate | 1.000 | [0.990, 1.000] | 398 | wilson |
| bm25 / phantom_citations | 0.000 | [0.000, 0.010] | 398 | wilson |
| bm25 / unsupported_displayed_claims | 0.000 | [0.000, 0.010] | 398 | wilson |
| bm25 / language_agreement | 0.960 | [0.936, 0.975] | 398 | wilson |
| bm25 / answer_accuracy_by_category [direct] | 0.960 | [0.910, 0.983] | 126 | wilson |
| bm25 / answer_accuracy_by_category [numeric_date] | 0.569 | [0.454, 0.677] | 72 | wilson |
| bm25 / answer_accuracy_by_category [typo_code_switch] | 0.427 | [0.366, 0.491] | 234 | wilson |
| bm25 / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| bm25 / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| bm25 / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |
| dense-hash / groundedness_verifier | 1.000 | [1.000, 1.000] | 325 | bootstrap |
| dense-hash / groundedness_claim_pooled | 1.000 | [0.992, 1.000] | 493 | wilson |
| dense-hash / citation_precision | 1.000 | [1.000, 1.000] | 325 | bootstrap |
| dense-hash / citation_recall | 1.000 | [1.000, 1.000] | 325 | bootstrap |
| dense-hash / rubric_score | 5.000 | [5.000, 5.000] | 325 | bootstrap |
| dense-hash / rubric_pass_rate | 1.000 | [0.988, 1.000] | 325 | wilson |
| dense-hash / phantom_citations | 0.000 | [0.000, 0.012] | 325 | wilson |
| dense-hash / unsupported_displayed_claims | 0.000 | [0.000, 0.012] | 325 | wilson |
| dense-hash / language_agreement | 0.954 | [0.925, 0.972] | 325 | wilson |
| dense-hash / answer_accuracy_by_category [direct] | 0.675 | [0.589, 0.750] | 126 | wilson |
| dense-hash / answer_accuracy_by_category [numeric_date] | 0.236 | [0.153, 0.346] | 72 | wilson |
| dense-hash / answer_accuracy_by_category [typo_code_switch] | 0.235 | [0.185, 0.293] | 234 | wilson |
| dense-hash / answer_accuracy_by_category [unanswerable] | 1.000 | [0.945, 1.000] | 66 | wilson |
| dense-hash / answer_accuracy_by_category [pii] | 1.000 | [0.824, 1.000] | 18 | wilson |
| dense-hash / answer_accuracy_by_category [injection] | 1.000 | [0.862, 1.000] | 24 | wilson |

`phantom_citations` and `unsupported_displayed_claims` are structural counts (F-4/F-5): they are reported as a rate over answered turns and must be 0.

## 4. Escalation and calibration — REPLAY-MODE

| arm / metric | value | 95% CI | n | note |
|---|---|---|---|---|
| hybrid-rrf / realised_risk | 0.351 | [0.306, 0.399] | 402 | 141/402 auto-answered turns were incorrect (E-1) |
| hybrid-rrf / coverage | 0.931 | [0.903, 0.951] | 432 | auto-answer coverage on answerable items (E-4) |
| hybrid-rrf / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| hybrid-rrf / escalation_precision | 0.783 | [0.707, 0.843] | 138 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| hybrid-rrf / realised_risk_by_bucket [informational] | 0.351 | [0.306, 0.399] | 402 | intent bucket informational (E-1 per bucket) |
| hybrid-rrf / coverage_by_bucket [informational] | 0.931 | [0.903, 0.951] | 432 | intent bucket informational |
| bm25 / realised_risk | 0.342 | [0.297, 0.390] | 398 | 136/398 auto-answered turns were incorrect (E-1) |
| bm25 / coverage | 0.921 | [0.892, 0.943] | 432 | auto-answer coverage on answerable items (E-4) |
| bm25 / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| bm25 / escalation_precision | 0.761 | [0.684, 0.823] | 142 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| bm25 / realised_risk_by_bucket [informational] | 0.342 | [0.297, 0.390] | 398 | intent bucket informational (E-1 per bucket) |
| bm25 / coverage_by_bucket [informational] | 0.921 | [0.892, 0.943] | 432 | intent bucket informational |
| dense-hash / realised_risk | 0.517 | [0.463, 0.571] | 325 | 168/325 auto-answered turns were incorrect (E-1) |
| dense-hash / coverage | 0.752 | [0.710, 0.791] | 432 | auto-answer coverage on answerable items (E-4) |
| dense-hash / must_escalate_recall | 1.000 | [0.966, 1.000] | 108 | 108/108 unsafe or unanswerable turns were escalated (E-2) |
| dense-hash / escalation_precision | 0.502 | [0.436, 0.569] | 215 | escalated turns that were genuinely unsafe/unanswerable (E-3) |
| dense-hash / realised_risk_by_bucket [informational] | 0.517 | [0.463, 0.571] | 325 | intent bucket informational (E-1 per bucket) |
| dense-hash / coverage_by_bucket [informational] | 0.752 | [0.710, 0.791] | 432 | intent bucket informational |

### 4.1 Reliability (E-5/E-6 diagnostic)

| quantity | value |
|---|---|
| n | 540 |
| ECE equal-width | 0.462 |
| ECE equal-mass | 0.462 |
| Brier | 0.456 |
| AURC | 0.336 |
| mean confidence | 0.021 |
| accuracy | 0.483 |
| coverage at target risk | 0.006 |
| risk at target | 0.000 |

`p̂` here is the packed evidence score, not a conformal calibrator (t8 owns conformal prediction), so E-5 is a diagnostic and no risk guarantee is claimed.

## 5. Operations — REPLAY-MODE

| arm / metric | value | bracket | n | method |
|---|---|---|---|---|
| hybrid-rrf / latency_generation_p50_ms | 1.9823 | [1.9290, 2.0580] | 540 | none |
| hybrid-rrf / latency_generation_p95_ms | 2.9170 | [2.7269, 5.2671] | 540 | none |
| hybrid-rrf / latency_judge_p50_ms | 0.0467 | [0.0441, 0.0488] | 402 | none |
| hybrid-rrf / latency_judge_p95_ms | 0.0637 | [0.0617, 0.1318] | 402 | none |
| hybrid-rrf / latency_retrieval_p50_ms | 5.6970 | [5.5825, 5.8038] | 540 | none |
| hybrid-rrf / latency_retrieval_p95_ms | 8.9737 | [8.3723, 12.6127] | 540 | none |
| hybrid-rrf / latency_total_p50_ms | 7.7946 | [7.6378, 7.9979] | 540 | none |
| hybrid-rrf / latency_total_p95_ms | 11.2457 | [10.4536, 14.7109] | 540 | none |
| hybrid-rrf / latency_total_mean_ms | 8.1856 | [8.0688, 8.3185] | 540 | bootstrap |
| hybrid-rrf / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 540 | bootstrap |
| hybrid-rrf / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 540 | none |
| hybrid-rrf / tokens_input_mean | 703.7130 | [685.7646, 723.7835] | 540 | bootstrap |
| hybrid-rrf / tokens_output_mean | 38.7185 | [36.8627, 40.7376] | 540 | bootstrap |
| hybrid-rrf / cache_hit_rate | 0.1407 | [0.1139, 0.1726] | 540 | wilson |
| bm25 / latency_generation_p50_ms | 1.7342 | [1.6247, 1.8377] | 540 | none |
| bm25 / latency_generation_p95_ms | 3.0278 | [2.7147, 4.1351] | 540 | none |
| bm25 / latency_judge_p50_ms | 0.0467 | [0.0453, 0.0479] | 398 | none |
| bm25 / latency_judge_p95_ms | 0.0663 | [0.0612, 0.1565] | 398 | none |
| bm25 / latency_retrieval_p50_ms | 0.6646 | [0.5617, 0.8585] | 540 | none |
| bm25 / latency_retrieval_p95_ms | 3.2074 | [2.6392, 42.6720] | 540 | none |
| bm25 / latency_total_p50_ms | 2.6442 | [2.5422, 2.7838] | 540 | none |
| bm25 / latency_total_p95_ms | 4.9390 | [4.4640, 44.9680] | 540 | none |
| bm25 / latency_total_mean_ms | 3.0061 | [2.8573, 3.1921] | 540 | bootstrap |
| bm25 / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 540 | bootstrap |
| bm25 / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 540 | none |
| bm25 / tokens_input_mean | 646.8019 | [628.4528, 666.1727] | 540 | bootstrap |
| bm25 / tokens_output_mean | 39.0889 | [37.1643, 41.0149] | 540 | bootstrap |
| bm25 / cache_hit_rate | 0.1407 | [0.1139, 0.1726] | 540 | wilson |
| dense-hash / latency_generation_p50_ms | 2.2709 | [2.2051, 2.3470] | 540 | none |
| dense-hash / latency_generation_p95_ms | 3.1666 | [2.9753, 4.4390] | 540 | none |
| dense-hash / latency_judge_p50_ms | 0.0357 | [0.0325, 0.0408] | 325 | none |
| dense-hash / latency_judge_p95_ms | 0.0652 | [0.0591, 0.3264] | 325 | none |
| dense-hash / latency_retrieval_p50_ms | 4.6453 | [4.5556, 4.7273] | 540 | none |
| dense-hash / latency_retrieval_p95_ms | 5.6728 | [5.4744, 8.1402] | 540 | none |
| dense-hash / latency_total_p50_ms | 6.8857 | [6.7675, 7.0208] | 540 | none |
| dense-hash / latency_total_p95_ms | 8.5374 | [8.1730, 10.1170] | 540 | none |
| dense-hash / latency_total_mean_ms | 6.9371 | [6.8598, 7.0090] | 540 | bootstrap |
| dense-hash / usd_per_answer | 0.0000 | [0.0000, 0.0000] | 540 | bootstrap |
| dense-hash / usd_per_answer_p95 | 0.0000 | [0.0000, 0.0000] | 540 | none |
| dense-hash / tokens_input_mean | 788.6667 | [764.0739, 814.6917] | 540 | bootstrap |
| dense-hash / tokens_output_mean | 34.4130 | [31.0808, 38.2668] | 540 | bootstrap |
| dense-hash / cache_hit_rate | 0.1407 | [0.1139, 0.1726] | 540 | wilson |

Latency is measured in-process around the pipeline call, so it is a regression guard (NFR-LAT-03) and not a server-side latency claim (NFR-LAT-01). The bracket for a percentile is the p(q-5)..p(q+5) spread, not a confidence interval.

## 6. Statistics: paired bootstrap and McNemar — REPLAY-MODE

| metric | arms | n | mean A | mean B | Δ | Δ 95% CI (paired bootstrap) | McNemar | p (method) | CI excludes 0 |
|---|---|---|---|---|---|---|---|---|---|
| recall_any@10 | hybrid-rrf − bm25 | 474 | 0.871 | 0.956 | -0.084 | [-0.112, -0.059] | b=2, c=42 | 0.0000 (chi2) | yes |
| recall_any@20 | hybrid-rrf − bm25 | 474 | 0.951 | 0.975 | -0.023 | [-0.040, -0.011] | b=1, c=12 | 0.0034 (exact_binomial) | yes |
| answer_ok | hybrid-rrf − bm25 | 540 | 0.683 | 0.685 | -0.002 | [-0.026, 0.019] | b=20, c=21 | 1.0000 (chi2) | no |
| recall_any@10 | hybrid-rrf − dense-hash | 474 | 0.871 | 0.443 | 0.428 | [0.382, 0.468] | b=203, c=0 | 0.0000 (chi2) | yes |
| recall_any@20 | hybrid-rrf − dense-hash | 474 | 0.951 | 0.506 | 0.445 | [0.397, 0.485] | b=211, c=0 | 0.0000 (chi2) | yes |
| answer_ok | hybrid-rrf − dense-hash | 540 | 0.683 | 0.491 | 0.193 | [0.156, 0.230] | b=109, c=5 | 0.0000 (chi2) | yes |
| recall_any@10 | bm25 − dense-hash | 474 | 0.956 | 0.443 | 0.513 | [0.466, 0.557] | b=245, c=2 | 0.0000 (chi2) | yes |
| recall_any@20 | bm25 − dense-hash | 474 | 0.975 | 0.506 | 0.468 | [0.422, 0.511] | b=223, c=1 | 0.0000 (chi2) | yes |
| answer_ok | bm25 − dense-hash | 540 | 0.685 | 0.491 | 0.194 | [0.156, 0.235] | b=117, c=12 | 0.0000 (chi2) | yes |

Paired bootstrap resamples **queries**, not arms (B = 1000); McNemar uses the χ² approximation with the exact-binomial fallback when b + c < 25. A difference whose interval contains 0 is noise and is reported as such (§6.3 step 5).

## 7. Targets — REPLAY-MODE

| id | target | threshold | measured | verdict | note |
|---|---|---|---|---|---|
| RT-1 | recall_any@20 (hybrid, tuned) | >= 0.90 | 0.951 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-3 | recall_any@10 | >= 0.80 | 0.871 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-4 | recall_all@20 | >= 0.75 | 0.904 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-5 | nDCG@10 | >= 0.65 | 0.583 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-6 | MRR@10 | >= 0.60 | 0.596 | diagnostic | diagnostic: replay-mode retrieval arms and machine labels |
| RT-2 | recall_any@20 per-locale floor | >= 0.80 every locale | 0.924 | diagnostic | locales: ky, ru, tg, tr, uz, zh |
| F-1 | groundedness (claim support rate) | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-2 | citation precision | >= 0.90 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| F-3 | citation recall | >= 0.85 | 1.000 | diagnostic | diagnostic: deterministic double or no judge calibration |
| E-4 | auto-answer coverage | >= 0.60 | 0.931 | diagnostic | diagnostic: the policy layer is t8's |
| E-2 | must-escalate recall | >= 0.95 | 1.000 | diagnostic | diagnostic: the policy layer is t8's |
| E-3 | escalation precision | >= 0.55 | 0.783 | diagnostic | diagnostic: the policy layer is t8's |
| E-1 | realised auto-answer risk vs target | <= 0.05 (pooled <= 0.05) | 0.351 | diagnostic | diagnostic: no conformal calibration in this run (t8) |
| E-5 | calibration error of p̂ | ECE <= 0.08 and Brier <= 0.15 | 0.462 | diagnostic | diagnostic: p̂ is the evidence score, not a conformal calibrator; Brier=0.4560, ECE(mass)=0.4623 |

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
uv run answerdesk eval --synthetic --provider fake --report reports\golden-regen
```

Same seed + same split + same corpus ⇒ identical `metrics.json` after removing timing fields (NFR-DET-02). The deterministic digest above is computed over exactly that projection.