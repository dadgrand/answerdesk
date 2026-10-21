# Results — answerdesk v1.0

This document is generated from the artefacts of one benchmark run. It states what was measured, on which data, with which models, and what does not generalize. Every number below comes from `reports/<run>/`; the document itself is `scripts/build-results.py`.

> **Internal evidence.** The corpus is synthetic; the numbers describe the reference stack on that corpus, not a production customer-support deployment.

## Run header

| field | value |
|---|---|
| run id | bench |
| corpus version | corpus-77d7d081813738a6 |
| corpus hash | `ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36` |
| documents / chunks | 810 / 7229 |
| test cases | 530 |
| dev cases | 216 |
| replay mode | no (live providers) |
| hardware | Windows 11 / Python 3.12.10 |
| duration | not measured (the run did not record it) |
| calibration | split-conformal thresholds fitted on the dev split; the test split is never used for fitting (INV-09) |
| arm `primary` | generator `qwen3.6-35b-a3b-noreason` (qwen) · judge `deepseek-flash` (deepseek) · cross-family yes · reportable yes |
| arm `secondary` | generator `deepseek-flash` (deepseek) · judge `qwen3.6-35b-a3b-noreason` (qwen) · cross-family yes · reportable yes |

Artefacts: `reports/bench` (summary.json, retrieval/, answer/, escalation/, cache/, ops/).

Every table below is generated from those artefacts by this script; the document itself is a build product (`scripts/build-results.py`).

## Provenance (one auditable block)

```text
corpus_version      corpus-77d7d081813738a6
corpus_hash         ae30066e61e1dc1402b4a51a12f4c4f66335d72c207073b496ef5cd9e5378a36
corpus_manifest     da33968a0d0087ecc0c4e8d966c1cdc47d40699dec139a00694818c0e583af69
golden_manifest     2fd9eb4ca0cdb0dbd5c45081e342dc9342697e1fff61c94f04e20bd09af7dff5
split_manifest      643a28c89c36abf8be343103caa70a06a8c5174414f29f7ebedcf0e8f3fc29ea
answerdesk_bench    65de8498b494b653b82be70bc854c7cb24f543fd911cfeda61705b522d356671
pricing_table       30c26f1f95551ba5d9043255a89ffe1ee881f45f37318bffc5c0eae1321e14f2
run_id              bench
summary_json        fe547d78a894d90ea5ccd6f37f7941ad29a65cb989df6ce6987fa1ab3cbf466f
```

Every hash above is of a committed file and is computed over **canonical LF bytes**, so a CRLF working tree and a clean LF checkout publish identical digests. Changing any input without re-running invalidates the chain, and `scripts/build-results.py --check` fails on a stale document.

## Datasets

| property | value |
|---|---|
| corpus version | corpus-77d7d081813738a6 |
| schema | 2.1.0 |
| documents | 810 |
| chunks | 7229 |
| queries | 3240 |
| query categories | {"code_switch": 204, "direct": 714, "identifier": 174, "injection": 174, "multi_hop": 264, "noisy_freeform": 474, "numeric_date": 234, "pii": 264, "typo": 204, "unanswerable": 534} |
| unanswerable share | 0.1648 |
| reference grounding | verified |
| license | CC0-1.0 |

Golden set `v1`: seed 20261021, 104 composition checks, categories {"direct": 126, "injection": 24, "numeric_date": 72, "pii": 18, "typo_code_switch": 224, "unanswerable": 66}, split {"dev": 216, "test": 530}. Adversarial companion sets (near-miss cache pairs, injection, PII, perturbation, claim labels) are listed in `evals/README.md`.

## Retrieval ablation

| arm | recall_any@10 | recall_any@20 | recall_all@20 | nDCG@10 | MRR@10 | n | mean retrieval |
|---|---|---|---|---|---|---|---|
| bm25 | 0.996 [0.988, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.947 [0.928, 0.962] (n=240) | 0.934 [0.915, 0.952] (n=240) | 0.984 [0.970, 0.996] (n=240) | 240 | 1.309 ms |
| cc@0.5 | 0.996 [0.988, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.926 [0.904, 0.945] (n=240) | 0.919 [0.898, 0.937] (n=240) | 0.971 [0.953, 0.986] (n=240) | 240 | 2.646 ms |
| cc@0.6 | 0.996 [0.988, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.900 [0.875, 0.922] (n=240) | 0.880 [0.854, 0.902] (n=240) | 0.948 [0.925, 0.970] (n=240) | 240 | 2.408 ms |
| cc@0.7 | 0.979 [0.963, 0.996] (n=240) | 0.988 [0.975, 1.000] (n=240) | 0.848 [0.817, 0.876] (n=240) | 0.833 [0.799, 0.862] (n=240) | 0.921 [0.891, 0.949] (n=240) | 240 | 2.646 ms |
| cc@0.75 | 0.975 [0.954, 0.992] (n=240) | 0.975 [0.954, 0.992] (n=240) | 0.817 [0.782, 0.850] (n=240) | 0.809 [0.774, 0.840] (n=240) | 0.914 [0.880, 0.944] (n=240) | 240 | 2.438 ms |
| cc@0.75+rerank | 1.000 [1.000, 1.000] (n=80) | 1.000 [1.000, 1.000] (n=80) | 0.270 [0.236, 0.307] (n=80) | 0.357 [0.331, 0.384] (n=80) | 0.904 [0.863, 0.940] (n=80) | 80 | 6599.142 ms |
| cc@0.8 | 0.975 [0.954, 0.992] (n=240) | 0.975 [0.954, 0.992] (n=240) | 0.799 [0.762, 0.831] (n=240) | 0.787 [0.750, 0.818] (n=240) | 0.903 [0.868, 0.935] (n=240) | 240 | 2.672 ms |
| cc@0.9 | 0.963 [0.938, 0.983] (n=240) | 0.963 [0.938, 0.983] (n=240) | 0.762 [0.724, 0.795] (n=240) | 0.731 [0.693, 0.766] (n=240) | 0.858 [0.819, 0.893] (n=240) | 240 | 2.411 ms |
| dense | 0.950 [0.921, 0.975] (n=240) | 0.954 [0.925, 0.979] (n=240) | 0.707 [0.668, 0.743] (n=240) | 0.672 [0.632, 0.706] (n=240) | 0.820 [0.780, 0.860] (n=240) | 240 | 1017.741 ms |
| rrf@10 | 0.996 [0.988, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.893 [0.870, 0.913] (n=240) | 0.874 [0.850, 0.895] (n=240) | 0.965 [0.946, 0.982] (n=240) | 240 | 2.544 ms |
| rrf@100 | 0.992 [0.979, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.869 [0.840, 0.894] (n=240) | 0.856 [0.826, 0.881] (n=240) | 0.957 [0.936, 0.977] (n=240) | 240 | 2.548 ms |
| rrf@60 | 0.992 [0.979, 1.000] (n=240) | 0.996 [0.988, 1.000] (n=240) | 0.868 [0.839, 0.893] (n=240) | 0.856 [0.827, 0.882] (n=240) | 0.957 [0.936, 0.977] (n=240) | 240 | 2.327 ms |

Paired comparisons (recall_any@10, paired bootstrap over queries + McNemar):

| pair | n | Δ | 95% CI | McNemar p | CI excludes 0 |
|---|---|---|---|---|---|
| `bm25` − `dense` | 240 | +0.0458 | [+0.0208, +0.0750] | 0.0010 (exact_binomial) | yes |
| `bm25` − `rrf@60` | 240 | +0.0042 | [+0.0000, +0.0125] | 1.0000 (exact_binomial) | no |
| `rrf@60` − `cc@0.75` | 240 | +0.0167 | [+0.0042, +0.0333] | 0.1250 (exact_binomial) | yes |
| `cc@0.75` − `cc@0.75+rerank` | 80 | +0.0000 | [+0.0000, +0.0000] | 1.0000 (n/a) | no |

### Per language (recall_any@10, Wilson interval)

| arm | locale | recall_any@10 | n |
|---|---|---|---|
| bm25 | ky | 1.000 [0.912, 1.000] | 40 |
| bm25 | ru | 1.000 [0.912, 1.000] | 40 |
| bm25 | tg | 1.000 [0.912, 1.000] | 40 |
| bm25 | tr | 1.000 [0.912, 1.000] | 40 |
| bm25 | uz | 0.975 [0.871, 0.996] | 40 |
| bm25 | zh | 1.000 [0.912, 1.000] | 40 |
| rrf@60 | ky | 1.000 [0.912, 1.000] | 40 |
| rrf@60 | ru | 1.000 [0.912, 1.000] | 40 |
| rrf@60 | tg | 1.000 [0.912, 1.000] | 40 |
| rrf@60 | tr | 1.000 [0.912, 1.000] | 40 |
| rrf@60 | uz | 0.950 [0.835, 0.986] | 40 |
| rrf@60 | zh | 1.000 [0.912, 1.000] | 40 |
| cc@0.75 | ky | 0.975 [0.871, 0.996] | 40 |
| cc@0.75 | ru | 0.975 [0.871, 0.996] | 40 |
| cc@0.75 | tg | 0.975 [0.871, 0.996] | 40 |
| cc@0.75 | tr | 1.000 [0.912, 1.000] | 40 |
| cc@0.75 | uz | 0.925 [0.801, 0.974] | 40 |
| cc@0.75 | zh | 1.000 [0.912, 1.000] | 40 |
| cc@0.75+rerank | ky | 1.000 [0.785, 1.000] | 14 |
| cc@0.75+rerank | ru | 1.000 [0.785, 1.000] | 14 |
| cc@0.75+rerank | tg | 1.000 [0.772, 1.000] | 13 |
| cc@0.75+rerank | tr | 1.000 [0.772, 1.000] | 13 |
| cc@0.75+rerank | uz | 1.000 [0.772, 1.000] | 13 |
| cc@0.75+rerank | zh | 1.000 [0.772, 1.000] | 13 |

### Noisy / free-form slice (its own block, never averaged in)

| arm | metric | value |
|---|---|---|
| bm25 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| bm25 | recall_all@10 | 0.943 [0.907, 0.976] (n=90) |
| bm25 | nDCG@10 | 0.950 [0.918, 0.979] (n=90) |
| bm25 | MRR@10 | 0.989 [0.967, 1.000] (n=90) |
| bm25 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| bm25 | recall_all@20 | 0.953 [0.917, 0.984] (n=90) |
| bm25 | nDCG@20 | 0.955 [0.922, 0.982] (n=90) |
| bm25 | MRR@20 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.5 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.5 | recall_all@10 | 0.946 [0.914, 0.975] (n=90) |
| cc@0.5 | nDCG@10 | 0.954 [0.925, 0.977] (n=90) |
| cc@0.5 | MRR@10 | 0.983 [0.956, 1.000] (n=90) |
| cc@0.5 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.5 | recall_all@20 | 0.952 [0.921, 0.978] (n=90) |
| cc@0.5 | nDCG@20 | 0.956 [0.929, 0.979] (n=90) |
| cc@0.5 | MRR@20 | 0.983 [0.956, 1.000] (n=90) |
| cc@0.6 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.6 | recall_all@10 | 0.930 [0.899, 0.960] (n=90) |
| cc@0.6 | nDCG@10 | 0.934 [0.903, 0.962] (n=90) |
| cc@0.6 | MRR@10 | 0.964 [0.931, 0.994] (n=90) |
| cc@0.6 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.6 | recall_all@20 | 0.943 [0.912, 0.970] (n=90) |
| cc@0.6 | nDCG@20 | 0.941 [0.913, 0.967] (n=90) |
| cc@0.6 | MRR@20 | 0.964 [0.931, 0.994] (n=90) |
| cc@0.7 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.7 | recall_all@10 | 0.901 [0.863, 0.937] (n=90) |
| cc@0.7 | nDCG@10 | 0.907 [0.870, 0.942] (n=90) |
| cc@0.7 | MRR@10 | 0.953 [0.914, 0.987] (n=90) |
| cc@0.7 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| cc@0.7 | recall_all@20 | 0.916 [0.882, 0.949] (n=90) |
| cc@0.7 | nDCG@20 | 0.915 [0.881, 0.947] (n=90) |
| cc@0.7 | MRR@20 | 0.953 [0.914, 0.987] (n=90) |
| cc@0.75 | recall_any@10 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.75 | recall_all@10 | 0.870 [0.826, 0.913] (n=90) |
| cc@0.75 | nDCG@10 | 0.883 [0.840, 0.922] (n=90) |
| cc@0.75 | MRR@10 | 0.948 [0.906, 0.985] (n=90) |
| cc@0.75 | recall_any@20 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.75 | recall_all@20 | 0.895 [0.854, 0.934] (n=90) |
| cc@0.75 | nDCG@20 | 0.896 [0.857, 0.933] (n=90) |
| cc@0.75 | MRR@20 | 0.948 [0.906, 0.985] (n=90) |
| cc@0.8 | recall_any@10 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.8 | recall_all@10 | 0.851 [0.806, 0.894] (n=90) |
| cc@0.8 | nDCG@10 | 0.864 [0.819, 0.904] (n=90) |
| cc@0.8 | MRR@10 | 0.940 [0.896, 0.980] (n=90) |
| cc@0.8 | recall_any@20 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.8 | recall_all@20 | 0.874 [0.830, 0.917] (n=90) |
| cc@0.8 | nDCG@20 | 0.876 [0.832, 0.916] (n=90) |
| cc@0.8 | MRR@20 | 0.940 [0.896, 0.980] (n=90) |
| cc@0.9 | recall_any@10 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.9 | recall_all@10 | 0.792 [0.739, 0.842] (n=90) |
| cc@0.9 | nDCG@10 | 0.807 [0.757, 0.853] (n=90) |
| cc@0.9 | MRR@10 | 0.918 [0.869, 0.965] (n=90) |
| cc@0.9 | recall_any@20 | 0.978 [0.944, 1.000] (n=90) |
| cc@0.9 | recall_all@20 | 0.831 [0.779, 0.880] (n=90) |
| cc@0.9 | nDCG@20 | 0.829 [0.781, 0.873] (n=90) |
| cc@0.9 | MRR@20 | 0.918 [0.869, 0.965] (n=90) |
| dense | recall_any@10 | 0.967 [0.933, 1.000] (n=90) |
| dense | recall_all@10 | 0.730 [0.674, 0.782] (n=90) |
| dense | nDCG@10 | 0.740 [0.683, 0.792] (n=90) |
| dense | MRR@10 | 0.857 [0.799, 0.915] (n=90) |
| dense | recall_any@20 | 0.967 [0.933, 1.000] (n=90) |
| dense | recall_all@20 | 0.777 [0.719, 0.828] (n=90) |
| dense | nDCG@20 | 0.766 [0.712, 0.815] (n=90) |
| dense | MRR@20 | 0.857 [0.799, 0.915] (n=90) |
| rrf@10 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| rrf@10 | recall_all@10 | 0.915 [0.880, 0.950] (n=90) |
| rrf@10 | nDCG@10 | 0.922 [0.887, 0.954] (n=90) |
| rrf@10 | MRR@10 | 0.970 [0.939, 0.994] (n=90) |
| rrf@10 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| rrf@10 | recall_all@20 | 0.930 [0.898, 0.962] (n=90) |
| rrf@10 | nDCG@20 | 0.930 [0.899, 0.961] (n=90) |
| rrf@10 | MRR@20 | 0.970 [0.939, 0.994] (n=90) |
| rrf@100 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| rrf@100 | recall_all@10 | 0.900 [0.861, 0.937] (n=90) |
| rrf@100 | nDCG@10 | 0.911 [0.873, 0.945] (n=90) |
| rrf@100 | MRR@10 | 0.969 [0.939, 0.994] (n=90) |
| rrf@100 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| rrf@100 | recall_all@20 | 0.918 [0.881, 0.953] (n=90) |
| rrf@100 | nDCG@20 | 0.922 [0.886, 0.954] (n=90) |
| rrf@100 | MRR@20 | 0.969 [0.939, 0.994] (n=90) |
| rrf@60 | recall_any@10 | 0.989 [0.967, 1.000] (n=90) |
| rrf@60 | recall_all@10 | 0.902 [0.862, 0.937] (n=90) |
| rrf@60 | nDCG@10 | 0.912 [0.875, 0.946] (n=90) |
| rrf@60 | MRR@10 | 0.969 [0.939, 0.994] (n=90) |
| rrf@60 | recall_any@20 | 0.989 [0.967, 1.000] (n=90) |
| rrf@60 | recall_all@20 | 0.918 [0.881, 0.953] (n=90) |
| rrf@60 | nDCG@20 | 0.922 [0.886, 0.955] (n=90) |
| rrf@60 | MRR@20 | 0.969 [0.939, 0.994] (n=90) |

## Answer quality and escalation

### Arm `primary` — generator `qwen3.6-35b-a3b-noreason`, judge `deepseek-flash`

Generator family **qwen**: `qwen3.6-35b-a3b-noreason` produced these answers. `qwen3.6-35b-a3b` is the reasoning variant; the planned side study at 8192/16384 tokens was not run in this window, so no reasoning-variant number is claimed.

Cross-family judging: **yes** (`qwen` vs `deepseek`); reportable: **yes**. cross-family judge; human κ not established for this corpus, so judged numbers are reported with an explicit κ caveat, not as F-1/F-2/F-3 gates

| metric | value (95% CI, n) |
|---|---|
| groundedness_judged | 0.908 [0.858, 0.950] (n=141) |
| citation_recall | 0.211 [0.189, 0.233] (n=141) |
| citation_precision | 0.671 [0.610, 0.727] (n=141) |
| rubric_score | 4.692 [4.576, 4.790] (n=141) |
| language_agreement | 0.986 [0.952, 0.996] (n=148) |
| coverage | 0.841 [0.780, 0.888] (n=176) |
| must_escalate_recall | 1.000 [0.862, 1.000] (n=24) |
| escalation_precision | 0.462 [0.333, 0.595] (n=52) |
| realised_risk | 0.223 [0.163, 0.297] (n=148) |

Confusion matrix: TP 24 · FP 28 · FN 0 · TN 148; false-escalation rate 15.91%, coverage 84.09%.

Escalation reason codes:

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 28 | 53.85% |
| INJECTION_SUSPECTED | 24 | 46.15% |

Per language:

| language | n | coverage | must-escalate recall | answer accuracy |
|---|---|---|---|---|
| ky | 34 | 0.900 [0.744,0.965] | 1.000 [0.510,1.000] | 0.926 [0.766,0.979] |
| ru | 34 | 0.800 [0.627,0.905] | 1.000 [0.510,1.000] | 0.750 [0.551,0.880] |
| tg | 33 | 0.897 [0.736,0.964] | 1.000 [0.510,1.000] | 0.808 [0.621,0.915] |
| tr | 33 | 0.828 [0.655,0.924] | 1.000 [0.510,1.000] | 0.875 [0.690,0.957] |
| uz | 33 | 0.862 [0.694,0.945] | 1.000 [0.510,1.000] | 0.760 [0.566,0.885] |
| zh | 33 | 0.759 [0.579,0.878] | 1.000 [0.510,1.000] | 0.500 [0.307,0.693] |

### Arm `secondary` — generator `deepseek-flash`, judge `qwen3.6-35b-a3b-noreason`

Generator family **deepseek**: `deepseek-flash` produced these answers.

_schema-fallback arm, NOT directly comparable: deepseek-flash rejects response_format=json_schema, so the generator ran through the schema-fallback path and 61/148 answered turns produced no parsed claims at all (primary: 7/148). Its judged metrics measure the parse path as much as the model; it is reported separately and excluded from headline comparisons._

Cross-family judging: **yes** (`deepseek` vs `qwen`); reportable: **yes**. cross-family judge; human κ not established for this corpus, so judged numbers are reported with an explicit κ caveat, not as F-1/F-2/F-3 gates

| metric | value (95% CI, n) |
|---|---|
| groundedness_judged | 0.391 [0.287, 0.494] (n=87) |
| citation_recall | 0.225 [0.188, 0.263] (n=87) |
| citation_precision | 0.641 [0.576, 0.705] (n=87) |
| rubric_score | 3.611 [3.385, 3.825] (n=87) |
| language_agreement | 0.986 [0.952, 0.996] (n=148) |
| coverage | 0.841 [0.780, 0.888] (n=176) |
| must_escalate_recall | 1.000 [0.862, 1.000] (n=24) |
| escalation_precision | 0.462 [0.333, 0.595] (n=52) |
| realised_risk | 0.818 [0.748, 0.871] (n=148) |

Confusion matrix: TP 24 · FP 28 · FN 0 · TN 148; false-escalation rate 15.91%, coverage 84.09%.

Escalation reason codes:

| reason code | count | share |
|---|---|---|
| LOW_CONFIDENCE | 28 | 53.85% |
| INJECTION_SUSPECTED | 24 | 46.15% |

Per language:

| language | n | coverage | must-escalate recall | answer accuracy |
|---|---|---|---|---|
| ky | 34 | 0.900 [0.744,0.965] | 1.000 [0.510,1.000] | 0.185 [0.082,0.367] |
| ru | 34 | 0.800 [0.627,0.905] | 1.000 [0.510,1.000] | 0.167 [0.067,0.359] |
| tg | 33 | 0.897 [0.736,0.964] | 1.000 [0.510,1.000] | 0.115 [0.040,0.290] |
| tr | 33 | 0.828 [0.655,0.924] | 1.000 [0.510,1.000] | 0.292 [0.149,0.492] |
| uz | 33 | 0.862 [0.694,0.945] | 1.000 [0.510,1.000] | 0.120 [0.042,0.300] |
| zh | 33 | 0.759 [0.579,0.878] | 1.000 [0.510,1.000] | 0.227 [0.101,0.434] |

## Risk control v2 (calibrated answerability, Learn-then-Test certificate)

Primary population `consistency` (133 dev / {'aurc': 0.446281, 'auroc': 0.964713, 'certified_operating_point': {'coverage': 0.004831, 'n': 207, 'n_accepted': 1, 'realised_risk': 0.0, 'risk_ci': [0.0, 0.793451], 'threshold': 0.968445, 'wrong': 0}, 'coverage_at_risk': {'alpha': 0.05, 'coverage': 0.115942, 'label': 'oracle frontier point (chosen on test for reporting only, not deployable)', 'n_accepted': 24, 'risk': 0.041667, 'risk_ci': [0.007393, 0.202418], 'threshold': 0.819824}, 'ece_equal_mass': 0.057409, 'ece_equal_width': 0.052773, 'evaluated_once': True} rows), full method in `docs/risk-control.md`.

| metric | value | target |
|---|---|---|
| AUROC | 0.964713 | >= 0.75 |
| ECE (equal-mass) | 0.057409 | <= 0.08 |
| coverage at realised risk <= 5% | 0.115942 | >= 0.40 |
| AURC | 0.446281 | - |

Certificate at alpha 0.05, delta 0.1: **ABSENT** (threshold 0.968445, calibration risk 1.0, UCB 1.0, n=133). Rows needed for a 5% certificate at zero observed risk: Bentkus ~46.1, Hoeffding ~460.5. The largest feasible level is alpha 0.25 (coverage 0.082707), reported as the operational fallback with its own alpha — no 5% guarantee is claimed.

## Calibration verdict (does risk control hold?)

Conformal α (informational bucket) = 0.05, operating-point target 0.02; realised risk on the answered set of the primary arm = **0.222973**. Risk control **did not hold**.

_informational bucket alpha=0.05, operating-point target 2%. The overlap confidence signal saturates at 1.0 for 74% of queries, so the dev-fitted conformal quantile collapses to q_hat=0.0 and the threshold (1.0) answers exactly the saturated set, whose realised error rate on the held-out split is 22.3% - four times the 5% alpha. Risk control did not hold._

This is reported as a measured negative result, not smoothed over: the deterministic overlap confidence saturates at 1.0 on most queries, so the split-conformal quantile collapses to zero and the threshold separates almost nothing. A calibrated confidence signal (the t8 judge-based scorer, or a monotone calibration map fitted on dev) is required before any risk-control claim can be made; the shipped default in this repository remains the config-driven threshold, and README/claim language must not assert calibrated escalation on this evidence.

## Calibrated answerability scorer (replaces the saturated overlap score)

L2 logistic regression fitted on **dev answers only** (n=200, l2=2.0), features: overlap_score, bm25_top, bm25_margin, bm25_above_gate, citation_density, log_claims. Split-conformal q-hat on the calibrated dev scores = 0.918967 (threshold 0.081033). Evaluated on the held-out test turns (n=200); no answer was regenerated.

| scorer | coverage | realised risk | risk 95% CI | ECE | risk control held |
|---|---|---|---|---|---|
| calibrated-logistic | 0.975 | 0.410256 | [0.3436, 0.48038] | 0.079379 | no |
| calibrated (dev operating point) | 0.0 |  |  | 0.079379 | no |
| raw-overlap@0.5 | 0.945 | 0.391534 | [0.324772, 0.462618] | 0.342784 | no |
| raw-overlap@1.0 (what the engine actually did) | 0.74 | 0.222973 | [0.163406, 0.296557] | 0.342784 | no |

**the calibrated scorer's realised risk is 0.410256 against alpha=0.05; the gate does NOT hold and no risk-control wording may ship**

Gate rule (stated explicitly): risk control holds only if the Wilson 95% upper bound of the realised risk on the answered set is <= alpha; a point estimate below alpha with an upper bound above it does not count Label source: judge labels (cross-family LLM judge: deepseek-flash on qwen answers). targets are judge labels, not human labels; the calibration inherits the judge's error and its per-claim rubric, so a human-labelled recalibration is required before any production gate n_dev = 200, so the fit is small-n; the CIs above are the honest measure of that.

The calibrated score is much better *calibrated* than the raw overlap score (ECE 0.079379 vs 0.342784) but has **not enough discriminative power** to deliver the 5% guarantee on this corpus: at the conformal threshold the realised risk is far above alpha. That is a measured negative result and is reported as such.

### Escalation detail — arm `primary`

E-2 recall by unsafe family (must-escalate ≥ 0.95):

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 176 | 28 | 0.159 | [0.112, 0.220] |
| unsafe | 24 | 24 | 1.000 | [0.862, 1.000] |

Per language (coverage on answerable items · E-2 recall on unsafe items):

| language | n | coverage | must-escalate recall |
|---|---|---|---|
| ky | 34 | 0.900 [0.744,0.965] | 1.000 [0.510,1.000] |
| ru | 34 | 0.800 [0.627,0.905] | 1.000 [0.510,1.000] |
| tg | 33 | 0.897 [0.736,0.964] | 1.000 [0.510,1.000] |
| tr | 33 | 0.828 [0.655,0.924] | 1.000 [0.510,1.000] |
| uz | 33 | 0.862 [0.694,0.945] | 1.000 [0.510,1.000] |
| zh | 33 | 0.759 [0.579,0.878] | 1.000 [0.510,1.000] |

Decisions: answer 148, escalate 52. not auto-answered = action is escalate, refuse or clarify; the confusion matrix uses that definition for 'predicted escalation'

**Headline operating point** — threshold selected on the dev/calibration turns (the split the conformal quantile is fitted on) and applied unchanged to the held-out test turns: threshold 1.0, test coverage **0.740** [0.675,0.796] at realised risk **0.222973** [0.163,0.297] (n answered 148; dev point: coverage 0.005 at risk 0.0).

*Oracle upper bound (diagnostic, not the shipped operating point):* selecting the threshold on the test split itself would give coverage 0.065 at risk 0.0000. It is excluded from the headline because it fits the reported data.

Policy variants (same turns, three decision rules):

| policy | threshold | coverage | realised risk | E-2 recall | precision |
|---|---|---|---|---|---|
| fixed threshold 0.95 (cost band default) | 0.95 | 0.740 [0.675,0.796] | 0.222973 [0.163,0.297] | 1.000 [0.862,1.000] | 0.461538 [0.333,0.595] |
| fixed threshold 0.80 (5:1 band) | 0.8 | 0.840 [0.783,0.884] | 0.315476 [0.250,0.389] | 1.000 [0.862,1.000] | 0.75 [0.579,0.867] |
| fixed threshold 0.90 (10:1 band) | 0.9 | 0.760 [0.696,0.814] | 0.243421 [0.182,0.317] | 1.000 [0.862,1.000] | 0.5 [0.364,0.636] |
| conformal+cost engine (dev-fitted, shipped) | engine | 0.740 [0.675,0.796] | 0.222973 [0.163,0.297] | 1.000 [0.862,1.000] | 0.461538 [0.333,0.595] |

**E-3 (escalation precision).** Gated reading: 0.461538 [0.333,0.595]; literal safety-only reading: 1.0 [0.862,1.000]. literal reading counts only escalations produced by a safety guard; it is the stricter reading of E-3 and is reported, not hidden

**Resolution (ADR-style).** E-3 is reported with both readings and neither is tuned away: the gated reading counts every non-answered turn in the denominator (what the operator experiences as escalation volume), the literal reading counts only turns a safety guard stopped (what the safety claim is about). Where the literal value falls below 0.55 on this synthetic population, that failure is reported as measured; the population is not re-cut to make the number pass.

### Escalation detail — arm `secondary`

E-2 recall by unsafe family (must-escalate ≥ 0.95):

| family | n | escalated | recall | 95% CI |
|---|---|---|---|---|
| answerable | 176 | 28 | 0.159 | [0.112, 0.220] |
| unsafe | 24 | 24 | 1.000 | [0.862, 1.000] |

Per language (coverage on answerable items · E-2 recall on unsafe items):

| language | n | coverage | must-escalate recall |
|---|---|---|---|
| ky | 34 | 0.900 [0.744,0.965] | 1.000 [0.510,1.000] |
| ru | 34 | 0.800 [0.627,0.905] | 1.000 [0.510,1.000] |
| tg | 33 | 0.897 [0.736,0.964] | 1.000 [0.510,1.000] |
| tr | 33 | 0.828 [0.655,0.924] | 1.000 [0.510,1.000] |
| uz | 33 | 0.862 [0.694,0.945] | 1.000 [0.510,1.000] |
| zh | 33 | 0.759 [0.579,0.878] | 1.000 [0.510,1.000] |

Decisions: answer 148, escalate 52. not auto-answered = action is escalate, refuse or clarify; the confusion matrix uses that definition for 'predicted escalation'

**Headline operating point** — threshold selected on the dev/calibration turns (the split the conformal quantile is fitted on) and applied unchanged to the held-out test turns: threshold 1.0, test coverage **0.740** [0.675,0.796] at realised risk **0.817568** [0.748,0.871] (n answered 148; dev point: coverage 0.005 at risk 0.0).

Policy variants (same turns, three decision rules):

| policy | threshold | coverage | realised risk | E-2 recall | precision |
|---|---|---|---|---|---|
| fixed threshold 0.95 (cost band default) | 0.95 | 0.740 [0.675,0.796] | 0.817568 [0.748,0.871] | 1.000 [0.862,1.000] | 0.461538 [0.333,0.595] |
| fixed threshold 0.80 (5:1 band) | 0.8 | 0.840 [0.783,0.884] | 0.839286 [0.776,0.887] | 1.000 [0.862,1.000] | 0.75 [0.579,0.867] |
| fixed threshold 0.90 (10:1 band) | 0.9 | 0.760 [0.696,0.814] | 0.822368 [0.754,0.875] | 1.000 [0.862,1.000] | 0.5 [0.364,0.636] |
| conformal+cost engine (dev-fitted, shipped) | engine | 0.740 [0.675,0.796] | 0.817568 [0.748,0.871] | 1.000 [0.862,1.000] | 0.461538 [0.333,0.595] |

**E-3 (escalation precision).** Gated reading: 0.461538 [0.333,0.595]; literal safety-only reading: 1.0 [0.862,1.000]. literal reading counts only escalations produced by a safety guard; it is the stricter reading of E-3 and is reported, not hidden

**Resolution (ADR-style).** E-3 is reported with both readings and neither is tuned away: the gated reading counts every non-answered turn in the denominator (what the operator experiences as escalation volume), the literal reading counts only turns a safety guard stopped (what the safety claim is about). Where the literal value falls below 0.55 on this synthetic population, that failure is reported as measured; the population is not re-cut to make the number pass.

## Chunking ablation: structure-first vs naive fixed window

Same cases (240), same embedder, gold evidence mapped by character overlap; the naive arm splits every document into 900-character windows with 150 overlap and ignores headings, tables and FAQ pairs.

| arm | chunks | n | recall_any@10 | recall_any@20 |
|---|---|---|---|---|
| structure-bm25 | 7229 | 240 | 0.991667 [0.970,0.998] | 0.995833 [0.977,0.999] |
| structure-dense | 7229 | 240 | 0.925 [0.885,0.952] | 0.945833 [0.910,0.968] |
| naive-bm25 | 2924 | 240 | 0.995833 [0.977,0.999] | 0.995833 [0.977,0.999] |
| naive-dense | 2924 | 240 | 0.929167 [0.890,0.955] | 0.958333 [0.925,0.977] |

_gold spans are carried over by character overlap, so both chunkings are scored against the same evidence; the naive arm ignores headings, tables and FAQ pairs_

## Noisy / free-form slice (own block, never averaged in)

_No noisy items in the evaluated sample._

## Semantic cache

| quantity | value |
|---|---|
| near-duplicate pairs | 30 |
| hits / attempts | 0 / 28 |
| hit rate | 0.0 [0.000,0.121] |
| wrong-hit rate (UCB) | 0.0 (UCB 0.0) |
| hit latency p50 / p95 | 0.0 / 0.0 ms |
| USD saved on hits | 0.0 |

_replay pairs are same-document, same-base-question rephrasings (typo/code-switch); a wrong hit is a hit whose cached citations do not cover the sibling's evidence_

## Latency and cost (measured, not estimated)

| arm | stage | p50 ms | p95 ms | n |
|---|---|---|---|---|
| primary | retrieval | 1.44 | 2.54 | 200 |
| primary | generation | 2618.22 | 14116.76 | 200 |
| primary | judge | 3409.63 | 51578.15 | 200 |
| primary | total | 7748.48 | 55107.46 | 200 |
| secondary | retrieval | 1.39 | 2.42 | 200 |
| secondary | generation | 7097.86 | 13367.99 | 200 |
| secondary | judge | 651.96 | 4930.77 | 200 |
| secondary | total | 9588.75 | 14859.05 | 200 |

| arm | answered | input tokens/answer | output tokens/answer | USD/answer |
|---|---|---|---|---|
| primary | 148 | 3255.6 | 511.9 | 0.0 |
| secondary | 148 | 4894.1 | 1531.1 | 0.0 |

Recorded token pricing (see `configs/pricing.yaml` for provenance):

| model | input USD/1k | output USD/1k | retrieved at | source |
|---|---|---|---|---|
| deepseek-chat | 0.0 | 0.0 | 2026-10-20 | placeholder — replace with the provider's published price before quoting USD |
| deepseek-flash | 0.0 | 0.0 | 2026-10-21 | key used for the benchmark exposed no billing statement; token counts recorded |
| fake-model | 0.0 | 0.0 | 2026-10-20 | deterministic CI double — always free |
| gpt-4o-mini | 0.0 | 0.0 | 2026-10-20 | placeholder — replace with the provider's published price before quoting USD |
| local-model | 0.0 | 0.0 | 2026-10-20 | self-hosted (vLLM / llama.cpp / Ollama): electricity only, recorded as 0 |
| qwen3.6-35b-a3b | 0.0 | 0.0 | 2026-10-21 | NeuralDeep free tariff at measurement — no per-token charge observed |
| qwen3.6-35b-a3b-noreason | 0.0 | 0.0 | 2026-10-21 | NeuralDeep free tariff at measurement — no per-token charge observed |

## NFR-LAT-01 verdict — latency target

| requirement | target | measured (arm `primary`) | basis | verdict |
|---|---|---|---|---|
| NFR-LAT-01 | p50 ≤ 1,500 ms · p95 ≤ 4,000 ms | p50 2618.22 ms · p95 14116.76 ms (n=200) | generation stage, arrival → final byte, judge excluded | **MISS** |
| (context only) | — | p50 7748.48 ms · p95 55107.46 ms (n=200) | total including the offline judge; the judge is the tail, not the answerer | not a gate |

The target is **not** amended: the requirement stays at p50 ≤ 1,500 ms / p95 ≤ 4,000 ms and the release records the miss instead of hiding it (change control CC-2026-10-21-b, `docs/requirements.md` §18.5).

What the number measures: a hosted ~35B generator in strict-JSON mode producing ~500 output tokens per answered turn, arrival to final byte; the budget is not a token-stream budget. Streaming **was** measured in the separate `latency-v2` arm (docs/latency.md): baseline time-to-first-token 673 ms p50 / 965 ms p95, and the output-budget/cascade variants still missed the total, so TTFT alone does not rescue this target — the verdict stays MISS on the measured totals. The retrieval path in the same run is 1.44/2.54 ms p50/p95, so the budget is consumed by generation, not by retrieval.

What would change the verdict: a smaller or local model, streaming with TTFT as the served-latency metric, and a shorter answer budget — each measurable with the same harness (`answerdesk eval --all --provider ...`). Until one of those is measured and committed, this requirement is reported as missed.

## Error taxonomy

| failure mode | count | examples (case ids) |
|---|---|---|
| answer did not cite acceptable evidence | 119 | v1-tg-test-direct-003, v1-tr-test-direct-004, v1-tg-test-direct-006 |
| judge rubric below 4 | 63 | v1-ru-test-direct-002, v1-uz-test-direct-002, v1-zh-test-direct-002 |
| answerable turn escalated (LOW_CONFIDENCE) | 56 | v1-ru-test-noisy-001, v1-tg-test-noisy-001, v1-tr-test-noisy-001 |
| other | 16 | v1-tr-test-direct-008, v1-zh-test-direct-009, v1-zh-test-direct-011 |
| answer language mismatch | 4 | v1-uz-test-direct-017, v1-uz-test-direct-019, v1-uz-test-direct-007 |

## External validity (separate deliverable)

Public-benchmark results live in `docs/external-validity.md` and are **not** merged into the tables above, because they answer a different question (external comparability) on different data with their own licences and sampling.

Concretely: no public retrieval qrels exist for Uzbek, Tajik or Kyrgyz, so this document makes **no** external retrieval claim for those locales; ru and zh do have public retrieval benchmarks (MIRACL, Mr. TyDi), and those belong to the external report. Belebele-derived labels are derived/diagnostic and never presented as public qrels.

## Limitations — what does NOT generalize

1. **Synthetic corpus.** 810 documents / 7229 chunks generated in six languages; queries are derived from the same documents, so retrieval and groundedness are optimistic relative to real support traffic. No number here transfers without re-measurement on a production corpus.
2. **No real bank traffic, no SLA, no incident history.** Escalation cost ratios (c_wrong : c_abstain : c_delay) are documented assumptions; no handoff cost was measured.
3. **Judge-mediated metrics.** Answer quality is judged by an LLM (cross-family enforced); without human κ on this corpus, judged numbers carry an explicit κ caveat and are not F-1/F-2/F-3 release gates.
4. **Machine labels.** Gold evidence comes from the corpus structure (FAQ/equivalence classes), not from human annotation; §10.3 human verification has not been performed.
5. **Sample size.** Generation arms are evaluated on a stratified subsample (answer_limit=200, retrieval_limit=240); the per-language and per-family intervals are wide. Claims of 5-point differences need n≈400 (t17 playbook); smaller n is reported with its interval rather than rounded away.
6. **Ablation scope.** Chunking variants and cache on/off are reported where the artefacts exist; the retrieval ablation measures the shipped pipeline but not every fuser/normalisation combination.
7. **External validity is a separate deliverable.** Public-benchmark results live in `docs/external-validity.md` (t19); this document does not claim external retrieval quality for Uzbek, Tajik or Kyrgyz, for which no public qrels exist.
8. **Provider drift.** Hosted models and prices change; the run records model ids, token counts and the pricing snapshot date so a later run can be compared honestly.
9. **Reasoning variant.** The primary generator is `qwen3.6-35b-a3b-noreason` (measured: the reasoning variant spent its whole JSON budget on the trace in a direct probe). The planned side study at 8192/16384 tokens was **not run in this window**: the two free subscriptions were spent on the answer arms after the session cooldown, so the reasoning variant has probe evidence only and no headline number depends on it. `scripts/reasoning-side-study.py` reproduces it when quota allows.
10. **Noisy slice is small.** The free-form block is reported per example so a reader can see individual failures; it is not a population estimate for arbitrary malformed input.
11. **Confidence is a retrieval score, not a calibrated probability.** The decision engine's p-hat is the deterministic overlap scorer (t8), so the conformal quantile is a threshold on that score; realised test risk is reported with its interval rather than asserted as a distribution-free guarantee. A calibrated p-hat (t8's LLM-judge scorer) is the production path and is not used here to keep judging cross-family and affordable.

## Commands and reproduction

```powershell
$env:ANSWERDESK_CONFIG = 'configs/default.yaml'   # profile api is set by live-env
. .\scripts\live-env.ps1 neuraldeep -Judge deepseek -Secondary deepseek
uv run answerdesk ingest --demo                    # corpus (deterministic)
uv run answerdesk eval --all --provider fake --report reports/golden-regen --regenerate
uv run answerdesk eval --all --provider api --report reports/bench \
    --answer-limit 400 --retrieval-limit 540
uv run python scripts/reasoning-side-study.py --limit 32 --budgets 8192 16384
uv run python scripts/build-results.py --run reports/bench --out docs/results.md --check
```

The golden set and split are frozen and hashed (see the header table); regenerating them with the same seed must reproduce the same hashes. Latency and USD are measured in-process with the provider's own token counts.

Artefact digest: `fe547d78a894d90ea5ccd6f37f7941ad` (summary.json).
