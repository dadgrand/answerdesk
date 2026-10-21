# Consistency audit

Does the system say the same thing in every language, under every surface form, and does the corpus agree with itself? This document reports the measured answers, including the ones that are negative. The internal benchmark (`docs/results.md`) measures quality; this audit measures **consistency**, which is a different property and can fail even when answers look plausible.

## What was measured

- Transcreated set: **63 facts × 6 locales** (378 instances) built from corpus `corpus-77d7d081813738a6` — each fact is an archetype + document index asked as the clean direct question in every locale.
- Generator `qwen3.6-35b-a3b-noreason`; judge `deepseek-flash` (cross-family). Answers: 378 clean + 252 surface-form rows.
- Storage: every row (question, answer, decision, cited documents, detected language, asserted values, judge score) is in `reports/consistency-p0/answers.jsonl`; the audit and invariance artefacts sit next to it.

## 1. Cross-language fact agreement (the headline finding)

For each language: how often the answer states the fact (judge ≥ 4 against the reference answer) and whether it cites the fact's gold document set.

| language | n | same fact | citation hit | value fidelity |
|---|---|---|---|---|
| ky | 63 | 0.269841 [0.175838, 0.390299] (n=63) | 0.333333 [0.229496, 0.456328] (n=63) | 0.333333 [0.229496, 0.456328] (n=63) |
| ru | 63 | 0.142857 [0.077019, 0.249746] (n=63) | 0.365079 [0.257154, 0.488513] (n=63) | 0.412698 [0.299585, 0.535847] (n=63) |
| tg | 63 | 0.333333 [0.229496, 0.456328] (n=63) | 0.285714 [0.189032, 0.407027] (n=63) | 0.444444 [0.328471, 0.566804] (n=63) |
| tr | 63 | 0.269841 [0.175838, 0.390299] (n=63) | 0.444444 [0.328471, 0.566804] (n=63) | 0.396825 [0.285319, 0.520191] (n=63) |
| uz | 63 | 0.238095 [0.149938, 0.356357] (n=63) | 0.206349 [0.124753, 0.321698] (n=63) | 0.428571 [0.313969, 0.551384] (n=63) |
| zh | 63 | 0.190476 [0.112462, 0.304068] (n=63) | 0.396825 [0.285319, 0.520191] (n=63) | 0.492063 [0.372668, 0.612371] (n=63) |

Pooled: same fact 0.240741 [0.200388, 0.28631] (n=378); citation overlap 0.338624 [0.292753, 0.387742] (n=378); value fidelity 0.417989 [0.369336, 0.468293] (n=378).

**This is a negative result and it is reported as one.** Fact-level agreement is low in every language, and citation overlap is low with it. The reason is visible in the next table: the clean question is not document-discriminating. One wording recurs in **~6.778 sibling documents** on average (0.952 of facts have at least one sibling, max 8), each with its own slot values, so a correct retriever may return a different document that answers the same question with different numbers. That does not excuse the number — a customer asking «Сколько стоит перевыпуск карты?» gets different amounts depending on which sibling is retrieved — but it does explain it, and it is filed as a deficiency below rather than hidden.

Cross-language pairs (both locales state the fact for the same fact id): ru-uz 0.047619, ky-ru 0.063492, ru-zh 0.063492, ru-tr 0.079365, uz-zh 0.095238, ky-uz 0.111111; pooled 0.120635.

## 2. Language discipline

Answer language against the query language, and the detected-language field against the query:

| language | n | answer language ok | detected ok | evidence locale ok |
|---|---|---|---|---|
| ky | 63 | 0.809524 [0.695932, 0.887538] (n=63) | 0.777778 [0.660877, 0.86275] (n=63) | 0.47619 [0.357821, 0.597296] (n=63) |
| ru | 63 | 0.904762 [0.807383, 0.955617] (n=63) | 0.857143 [0.750254, 0.922981] (n=63) | 0.555556 [0.433196, 0.671529] (n=63) |
| tg | 63 | 0.666667 [0.543672, 0.770504] (n=63) | 0.666667 [0.543672, 0.770504] (n=63) | 0.968254 [0.891435, 0.99125] (n=63) |
| tr | 63 | 0.952381 [0.86909, 0.983674] (n=63) | 0.952381 [0.86909, 0.983674] (n=63) | 0.587302 [0.464153, 0.700415] (n=63) |
| uz | 63 | 0.825397 [0.713786, 0.899606] (n=63) | 0.603175 [0.479809, 0.714681] (n=63) | 0.555556 [0.433196, 0.671529] (n=63) |
| zh | 63 | 0.857143 [0.750254, 0.922981] (n=63) | 0.857143 [0.750254, 0.922981] (n=63) | 0.968254 [0.891435, 0.99125] (n=63) |

`evidence locale ok` means every cited document id carries the query's locale. The pooled rate is **0.685**: the retriever does not filter by locale, so answers frequently cite documents written in another language of the corpus. That is the mechanical cause of low fact agreement below — the model reads a sibling locale's numbers — and it is filed as CNS-6.

## 3. Surface-form invariance

Measured on the 67 generated variant rows (RAG retriever, same generator and judge as above): does the variant keep the clean form's cited document set and the same decision?

- Evidence invariance: **0.19403 [0.117052, 0.304191] (n=67)**
- Decision invariance: **0.791045 [0.679282, 0.871243] (n=67)**

| variant | facts | same evidence | same decision |
|---|---|---|---|
| code_switch | 16 | 0.0625 [0.011119, 0.283287] (n=16) | 0.6875 [0.444044, 0.858354] (n=16) |
| telegraphic | 17 | 0.411765 [0.216111, 0.639946] (n=17) | 0.941176 [0.73018, 0.98954] (n=17) |
| translit | 17 | 0.058824 [0.01046, 0.26982] (n=17) | 0.588235 [0.360054, 0.783889] (n=17) |
| typo | 17 | 0.235294 [0.09555, 0.472618] (n=17) | 0.941176 [0.73018, 0.98954] (n=17) |

Clean-form answers for all six languages are in `answers.jsonl`; the variant arm runs on the locale configured with `--variant-locales` (ru in this run). **Caveat:** 185 variant rows hit provider 429s after the quota was exhausted and are excluded from the pair set, so the rates above are computed on the subset that completed (n = 67 pairs) and are therefore imprecise, not a population estimate.

Retrieval-level invariance is measured independently for **all six locales** on all facts (1512 pairs, BM25 top-5 document sets): **845/1512 = 0.559** keep the same evidence as the clean form.

| variant | pairs | same evidence |
|---|---|---|
| telegraphic | 378 | 273/378 = 0.722 |
| typo | 378 | 175/378 = 0.463 |
| translit | 378 | 189/378 = 0.5 |
| code_switch | 378 | 208/378 = 0.55 |

| language | pairs | same evidence |
|---|---|---|
| ky | 252 | 0.421 |
| ru | 252 | 0.583 |
| tg | 252 | 0.496 |
| tr | 252 | 0.782 |
| uz | 252 | 0.726 |
| zh | 252 | 0.345 |

These are eval-layer stress variants (function words dropped, one character swapped, the sentence romanised, another locale's term injected), harsher than the corpus's own noisy items; they measure resilience, not product-log quality.

## 4. Logical consistency of corpus v2.1

Audited **810 documents** (810 with tables, 4995 table rows, 135 with an as-of date). Findings by severity: {'high': 774}. `invariant_holds = False`; within-document blockers: 0.

**Cross-document contradictions were found** (severity high, owner ingest): the same named product carries different values for the same parameter in different documents of the same locale. Most affected parameters: `acil başvuru` (4), `acil havale` (3), `araştırma ve iade` (3), `asgari tutar` (3), `aylık ücret` (3), `başka bankaya havale` (2), `başvuru yanıtı` (4), `boshqa bankka o'tkazma` (4).

Findings per locale: ky 147, ru 141, tg 152, tr 141, uz 142, zh 51.

Example (verbatim ids only, no corpus text): `syn-ru-0033` and `syn-ru-0093` both describe the same product tier with different values for the same parameter row. This is a real documentation inconsistency: a reader who opens both documents gets two answers. It is filed to ingest with severity **high** and it is not fixed here — this task does not edit corpus or ingest code.

## 5. Honest limitations and filed deficiencies

| id | severity | finding | owner |
|---|---|---|---|
| CNS-1 | high | cross-document contradictions in corpus v2.1: 774 same-product/same-parameter conflicts across documents of one locale | ingest (data-engineer) |
| CNS-2 | high | clean questions are not document-discriminating (~6.778 siblings per fact), so fact-level answers differ depending on which sibling is retrieved | ingest (question design) |
| CNS-3 | medium | cross-language fact agreement 0.240741 and citation overlap 0.338624: multilingual answers do not agree on the same fact, partly explained by CNS-2 | eval + ingest |
| CNS-4 | medium | retrieval-level surface-form invariance 0.559: romanised, typo and code-switch queries often retrieve different evidence | retrieval (t23 owner) |
| CNS-5 | low | 1 judged rows failed at the judge and are counted as not-stating-the-fact, not dropped | eval |
| CNS-6 | high | retrieval has no locale filter: pooled evidence-locale rate 0.685, so answers cite other languages' documents | retrieval (t23 owner) |

Further limitations: the judge is a cross-family LLM, not a human; the transcreated set is one clean question per fact (the corpus's own noisy items are measured in the retrieval ablation, not here); the invariance variants are eval-layer stress forms; the audit's cross-document check only fires when the same product name and parameter key recur, so it is a lower bound on inconsistency.

## 6. Commands

```bash
# offline, deterministic (CI)
uv run pytest -q tests/unit/eval

# rebuild the transcreated set from corpus v2.1
uv run answerdesk eval consistency --build

# logical audit (0 blockers expected; high findings are reported)
uv run answerdesk eval consistency --audit --report reports/consistency-p0

# retrieval-level surface-form invariance, all locales, no model calls
uv run answerdesk eval consistency --retrieval --report reports/consistency-p0

# live answers (one language pair per invocation; the checkpoint resumes)
. ./scripts/live-env.ps1 neuraldeep -Judge deepseek
uv run answerdesk eval consistency --run --provider api --limit 50 --languages ru,uz --variant-locales "" --report reports/consistency-p0

# surface-form variants (ru)
uv run answerdesk eval consistency --run --provider api --limit 50 --languages ru --variant-locales ru --report reports/consistency-p0
```

## 7. Provenance

```text
corpus_version    corpus-77d7d081813738a6
facts_sha256      2d259e9b3f49b0a53229b1f9f3985b285a3527f4b356e507e4ee1735e4df1d42
generator_model   qwen3.6-35b-a3b-noreason
judge_model       deepseek-flash
rows              {'clean': 378, 'total': 630, 'variants': 252}
generated_at      2026-10-21T13:22:18Z
hardware          Windows 11 / Python 3.12.10
```
