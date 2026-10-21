# External validity

This document reports the **external** arm: public benchmarks, in their own harness, with their own licences. The internal frozen split remains the primary evidence (`docs/results.md`); nothing here is merged into those tables, and nothing here is a tuned number — the harness is fixed, the suites are public, and every run is reproducible from one command.

## Headline (what may and may not be claimed)

- **Comprehension:** retrieval-conditioned answering improves Belebele accuracy in 4 of 5 languages with intervals excluding zero (ru, uz, tg, ky); **zh shows no gain** and is reported as no claim.
- **Retrieval (native qrels):** on Mr. TyDi ru the hybrid arm beats BM25 with a paired bootstrap interval that excludes zero. This is the only language where a native retrieval-quality claim is made.
- **Retrieval (derived labels):** Belebele-derived retrieval is reported per language as *diagnostic only*; there are no public retrieval qrels for Uzbek, Tajik or Kyrgyz, so no retrieval-quality claim is made for them.
- **Judges:** no human labels exist in this checkout, so every judged metric in the external arm is labelled diagnostic; the external numbers below are judge-free by construction (option letters and qrels, not LLM judgments).

## Datasets, versions and licences

Curated suite: **answerdesk-bench 0.1.0** (schema 1.0, released 2026-10-21), manifest `self_sha256 = 65de8498b494b653b82be70bc854c7cb24f543fd911cfeda61705b522d356671`. It is committed to this repository, so the run is offline and byte-reproducible.

| licence file | licence |
|---|---|
| LICENSES/MIT.txt | MIT |
| belebele/ | CC-BY-SA-4.0 |
| domain/ | CC0-1.0 |
| mrtydi/ | Apache-2.0 |
| schema.json | MIT |

Upstream datasets are **not vendored**: Belebele is CC BY-SA 4.0 and its text never enters this repository or these reports — only item ids, aggregate metrics and counts leave the adapter. Mr. TyDi is Apache-2.0. The upstream cache lives in `.cache/answerdesk-eval/` (gitignored) and is never populated implicitly.

## Suites and per-language n

| suite | items | languages | licence | label status |
|---|---|---|---|---|
| belebele_qa | 750 | ru,uz,tg,ky,zh | CC-BY-SA-4.0 | native |
| belebele_retrieval | 750 | ru,uz,tg,ky,zh | CC-BY-SA-4.0 | derived/diagnostic |
| domain | 240 | ru,uz,tg,ky,zh,tr | CC0-1.0 | native |
| mrtydi_retrieval | 2369 | ru | Apache-2.0 | native qrels |

Every QA number below uses the **full test split** of the language (n = 120); the 30 per-language dev items are reserved for the judge-calibration template. The QA checkpoint contains 1200 answered items; 8 of them returned no parseable option letter and are counted as incorrect (never dropped).

## 1. Belebele-derived QA: closed-book vs retrieval-conditioned

Answerer: `deepseek-chat`. Judge-free: the model must answer with one option letter, which is compared with the labelled option. Context is the BM25 top-3 passage from the language's Belebele passage pool. This tests whether retrieval helps *comprehension*; it says nothing about retrieval quality.

| language | n | closed-book | RAG | delta (RAG − closed) | 95% CI | claim |
|---|---|---|---|---|---|---|
| ky | 120 | 0.458333 | 0.741667 | 0.283333 | [0.175, 0.391667] | delta quoted |
| ru | 120 | 0.583333 | 0.85 | 0.266667 | [0.175, 0.358333] | delta quoted |
| tg | 120 | 0.533333 | 0.791667 | 0.258333 | [0.158333, 0.358333] | delta quoted |
| uz | 120 | 0.541667 | 0.775 | 0.233333 | [0.133333, 0.333333] | delta quoted |
| zh | 120 | 0.7 | 0.691667 | -0.008333 | [-0.083333, 0.075] | no claim (interval crosses zero) |

Claimed languages: **ky, ru, tg, uz**. A delta is only quoted as a finding when its paired-bootstrap 95% interval excludes zero. Intervals that cross zero are reported as **no claim** and are listed explicitly.

## 2. Mr. TyDi ru: BM25 vs hybrid with hosted bge-m3

Native qrels (Apache-2.0), n = 994 test queries over 2428 passages. Fusion: {'alpha_bm25': 0.5, 'family': 'convex combination', 'normalisation': 'min-max'}. Judge-free: qrels are labels, not judgments.

| arm | nDCG@10 | 95% CI | recall@10 | 95% CI | MRR@10 |
|---|---|---|---|---|---|
| bm25 | 0.603617 | [0.578028, 0.629241] | 0.72837 | [0.699195, 0.755533] | 0.601402 |
| hybrid (0.5 bm25 + 0.5 bge-m3) | 0.747153 | [0.726172, 0.767994] | 0.8833 | [0.862148, 0.903421] | 0.728749 |

Paired bootstrap on the per-query difference (hybrid − BM25): nDCG@10 **+0.143536** [0.129003, 0.158084] (crosses zero: False); recall@10 **+0.15493** [0.132797, 0.179074] (crosses zero: False).

## 3. Belebele-derived retrieval (diagnostic only)

The qrels are derived from the reading-comprehension gold passage, not collected by a retrieval-labeling process. They are useful as a smoke test of the retrieval stack in five languages and are **never** a retrieval-quality claim — in particular not for uz/tg/ky, where no public qrels exist at all.

| language | n | bm25 nDCG@10 | hybrid nDCG@10 | delta | 95% CI | status |
|---|---|---|---|---|---|---|
| ky | 120 | 0.737558 | 0.827194 | 0.089637 | [0.056524, 0.122829] | diagnostic (derived labels) |
| ru | 120 | 0.724983 | 0.860306 | 0.135323 | [0.097471, 0.17497] | diagnostic (derived labels) |
| tg | 120 | 0.729744 | 0.839775 | 0.110032 | [0.077857, 0.149057] | diagnostic (derived labels) |
| uz | 120 | 0.681534 | 0.857894 | 0.17636 | [0.125321, 0.235669] | diagnostic (derived labels) |
| zh | 120 | 0.066397 | 0.969015 | 0.902618 | [0.856783, 0.943801] | diagnostic (derived labels) |

## 4. Judge calibration (gate for every judged metric)

Status: **pending-human-labels**. no human labels are present in this checkout; every judged metric in this arm is diagnostic until calibration-kappa.json exists with κ >= 0.6 per language

| language | items | sheet |
|---|---|---|
| ru | 150 | calibration-ru.csv |
| zh | 150 | calibration-zh.csv |
| uz | 150 | calibration-uz.csv |
| tg | 150 | calibration-tg.csv |
| ky | 150 | calibration-ky.csv |

The sheets are committed with empty label columns. Two annotators label each item blind to the judge, disagreements are adjudicated, and `answerdesk eval external --score-calibration <dir>` computes Cohen's κ per language and overall. **Until κ ≥ 0.6 exists for a language, no judged metric from that language may be quoted.** The numbers in this document are judge-free, so this gate does not weaken them; it prevents the *judged* internal metrics from being mistaken for externally validated ones.

## 5. Reproduction

```bash
# 1. list suites, licences and cache state (offline, no model calls)
uv run answerdesk eval external --list-suites

# 2. retrieval suites (needs the hosted embedder for the dense channel)
. ./scripts/live-env.ps1 neuraldeep
uv run answerdesk eval external --run --provider fake --embedder api --suites mrtydi_retrieval,belebele_retrieval,domain --report reports/external-p0

# 3. comprehension (one language per invocation; the QA checkpoint resumes)
. ./scripts/live-env.ps1 deepseek -Judge none
uv run answerdesk eval external --run --provider deepseek --answerer-model deepseek-chat --embedder fake --suites belebele_qa --languages ru --limit 150 --report reports/external-p0

# 4. score the human labels when they exist
uv run answerdesk eval external --score-calibration reports/external-p0/calibration

# 5. regenerate this document from the artefacts and check it is current
uv run python scripts/build-external.py --run reports/external-p0 --check
```

## 6. Provenance

```text
benchmark           answerdesk-bench 0.1.0
manifest_self_sha256 65de8498b494b653b82be70bc854c7cb24f543fd911cfeda61705b522d356671
dataset             answerdesk-bench (D:/Documents/deepseek-harness/default-workspace/answerdesk/data/bench/answerdesk-bench/v0.1)
files_sha256        29 files pinned in dataset-digest.json
run                 external-p0
generated_at        2026-10-21T11:13:59Z
hardware            Windows 11 / Python 3.12.10
qa_checkpoint_rows  1200 (parse failures: 8)
```

## 7. Non-claims

- no state-of-the-art claim of any kind
- no retrieval-quality claim for uz/tg/ky (derived labels only)
- no claim whose bootstrap interval crosses zero
- no calibrated-judge claim until the human-label κ file exists

In addition: these numbers come from one answerer model and one embedder, on the bench's fixed sampling, so they are not a leaderboard position and not a claim about any other model; and the comprehension delta is an *upper bound on the value of retrieval* for this prompt, not a product quality estimate.
