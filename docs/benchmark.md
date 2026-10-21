# `answerdesk-bench` — construction, verification, and how the evaluation harness consumes it

This document describes the benchmark that lives in
[`data/bench/answerdesk-bench/v0.1/`](../data/bench/answerdesk-bench/v0.1/), the script
that builds it ([`scripts/build_bench.py`](../scripts/build_bench.py)) and the contract that
the evaluation harness (t19) implements against it.

The short version: the repository's internal golden set
([`evals/golden/v1/`](../evals/golden/v1/)) is synthetic, so every quality number computed
on it is, by construction, a number about our own generator. This benchmark exists to give
at least part of the quality story an **external anchor** — real Belebele passages and
questions in five languages, real Mr. TyDi relevance judgments in Russian, plus a
support-domain slice derived from the synthetic corpus but disjoint from the internal
golden split. It is small, it is honest about what it can and cannot support, and it is
machine-checkable (`--verify`).

Related documents: the dataset card
([`DATASET_CARD.md`](../data/bench/answerdesk-bench/v0.1/DATASET_CARD.md)) is the
reader-facing description; `docs/eval-methodology.md` defines the metrics and statistics
the harness applies; `docs/requirements.md` FR-EVL-01 is the requirement this benchmark
satisfies. The upstream licence research is in `research/answerdesk/eval-benchmarks.md`
(t17): Belebele is CC BY-SA 4.0, Mr. TyDi is Apache-2.0, MIRACL is Apache-2.0, RGB is
noncommercial and CRUD-RAG has no licence — the last two must not enter this tree.

## 1. Purpose and non-goals

**Purpose**

* Give an external reader a way to reproduce, with the harness shipped in this repository,
  a retrieval number in five languages — including `uz`, `tg` and `ky`, for which no public
  retrieval qrels exist.
* Give the support-domain behaviours (grounded answer, numeric/date, two-fact answer,
  abstention, PII non-echoing, injection resistance) a second, frozen set of cases that does
  not overlap the internal golden split used to tune the system.
* Make provenance and licensing checkable: every item says which upstream file and row it
  came from, and `--verify` re-checks every claim.

**Non-goals**

* Not a leaderboard. The Belebele-derived qrels are derived, and the pools are small
  (488 passages per language; 2,428 for Mr. TyDi). Numbers are comparable *within* this
  benchmark across systems and runs, not to published MIRACL/Mr. TyDi leaderboards.
* Not a support-domain sample of real traffic. The domain slice is synthetic and derived
  from one generator.
* Not training data. The mixed licences (CC BY-SA 4.0, Apache-2.0, CC0-1.0) make that
  impractical and, for `belebele/`, share-alike-infectious; see §3.

## 2. What ships

```
data/bench/answerdesk-bench/v0.1/
├── DATASET_CARD.md              reader-facing card (sources, sampling, limitations)
├── manifest.json                counts, hashes, sources, checks, t19 adapter map
├── schema.json                  JSON Schema of an item (generated from schema.py)
├── sources.lock.json            pinned upstream URLs/SHA-256 (committed; cache is not)
├── belebele/                    CC BY-SA 4.0
│   ├── ATTRIBUTION.md  README.md  LICENSE-CC-BY-SA-4.0.txt
│   ├── retrieval/<lang>/corpus.jsonl     488 passages, one pool per language
│   ├── retrieval/<lang>/queries.jsonl    150 items: query + qrels + split
│   └── qa/<lang>.jsonl                   150 items: question + 4 options + answer
├── mrtydi/                      Apache-2.0
│   ├── ATTRIBUTION.md  README.md  LICENSE-APACHE-2.0.txt
│   ├── ru/corpus.jsonl                   2,428 passages
│   └── ru/queries.jsonl                  2,369 native dev/test queries with qrels
├── domain/                      CC0-1.0
│   ├── README.md  LICENSE-CC0-1.0.txt  exclusions.json
│   └── <lang>.jsonl                      40 items each for ru uz tg ky zh tr
└── LICENSES/
    ├── README.md                licence map
    └── MIT.txt                  the repository's MIT text (code only)
```

Totals: **4,109 items** — 3,119 retrieval queries, 750 multiple-choice items, 240 domain
items — across 1,741 `dev` and 2,368 `test` items. Languages: `ru`, `uz`, `tg`, `ky`, `zh`
(all suites) and `tr` (domain only).

## 3. Sources and licences

| Source | Pinned version | Licence | Files | What we redistribute |
|---|---|---|---|---|
| `facebook/belebele` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` (`ds=2023-05-20`) | CC BY-SA 4.0 | `data/{rus_Cyrl,uzn_Latn,tgk_Cyrl,kir_Cyrl,zho_Hans}.jsonl` | derived passages, 150 questions/lang, options, derived qrels |
| `castorini/mr-tydi` | `1d43c80218d06d0ef80f5b172ccabd848b948bc1` (v1.1) | Apache-2.0 | `mrtydi-v1.1-russian/{dev,test}.jsonl.gz` | queries, qrels, splits |
| `castorini/mr-tydi-corpus` | `3a3aa212bbe94a8cc0dc858710a3dad49d532054` | Apache-2.0 | `mrtydi-v1.1-russian/corpus.jsonl.gz` (1.6 GB) | only the 2,428 positive passages |
| `answerdesk-synthetic` | `answerdesk-synthetic-v2.1.0`, `corpus-96f93db4f7f648b5`, seed `20261020` | CC0-1.0 | `data/corpus/{queries/queries.jsonl,manifest.json}`, `docs/<lang>/*.md` | 240 derived domain items + verbatim quotes |

`--fetch` is the only command that opens a socket. It downloads into the git-ignored cache
`.cache/answerdesk-bench/`, verifies each download against `sources.lock.json`, and fails if
an upstream file changed. Raw archives are never committed: `--verify` fails if a `.gz`,
`.zip` or raw `belebele_*.jsonl` appears under the release directory.

The licence separation is structural, not documentary: `belebele/`, `mrtydi/` and `domain/`
each carry their licence text and attribution file, and the dataset card states explicitly
that the repository's MIT licence covers code only. `--verify` checks the four licence
markers and refuses a dataset card that claims the data is MIT.

## 4. Construction

### 4.1 Belebele: retrieval labels by derivation (`belebele/`)

Belebele is reading comprehension, not retrieval: each question has a supplied passage and
four options. The only thing that makes a retrieval suite possible is the **gold passage**
per question. We build it as follows, per language variant:

1. Read the variant file (`data/<variant>.jsonl`, 900 rows).
2. Deduplicate the FLORES passage text in file order → 488 unique passages, ids
   `bb-<lang>-pNNNN`.
3. Assign each passage a split: `dev` iff
   `int(sha256("20261021:bb-passage:<lang>:<passage_id>")[:8], 16) % 5 == 0`, else `test`.
4. Drop questions whose four options are not distinct after normalisation (a handful in the
   upstream release); count them in `manifest.sampling.belebele`.
5. Rank the remaining questions of each split by
   `sha256("20261021:bb-question:<lang>:<question_number>")` and take 120 test + 30 dev.
6. Emit one retrieval item per question (`qrels = [gold passage]`) and one MCQ item
   (`options`, `answer_index`, `gold_evidence_ids = [gold passage]`, `answer_anchor`).

This is the only public route to *any* retrieval evaluation in `uz`, `tg` and `ky`. It is
also the weakest label in the benchmark, and it is labelled as such everywhere: the
`provenance.derivation` is `derived_qrels_from_belebele_gold_passage`, the suite notes in
`manifest.suites` say "derived", and the dataset card's first limitation explains that
Belebele questions are written against a supplied passage, so as retrieval queries they are
deictic ("according to the given text…") and easier than ad-hoc customer questions. **Do
not quote a Belebele-derived recall as "retrieval quality in Uzbek".**

### 4.2 Mr. TyDi ru: native qrels over a derived pool (`mrtydi/`)

Mr. TyDi publishes queries + qrels, and a separate corpus archive. The archive is 1.6 GB;
we stream it once in `--fetch`/`--build`, keep only the passages whose `docid` is a positive
of a dev or test query, and write a 2,428-passage pool (`derived/mrtydi-ru/passages.jsonl`
in the cache). The suite ships:

* the complete ru `dev` (1,374) and `test` (995) query sets, and
* every passage that is relevant to them.

Two native query strings were inverted-duplicates of earlier ones (identical word sequence
after punctuation normalisation); they are dropped and counted in
`manifest.sampling.mrtydi.dropped_duplicate_queries` so the near-duplicate invariant holds.
Because the pool is exactly the union of positives, every relevant passage carries a
judgement — the standard pooled-retrieval caveat (judgements are per query, not global)
still applies and is in the dataset card.

### 4.3 Domain slice: grounded answers, escalation and adversarial cases (`domain/`)

Derived from the synthetic corpus, one item per article per category, 40 items per locale.
Selection is deterministic and — importantly — **excludes every document used by t11's
golden/frozen split**:

1. Read `data/corpus/queries/queries.jsonl` and the article texts.
2. Build the exclusion set from `evals/golden/v1/*.jsonl` (`metadata.doc_id`,
   `metadata.document_ids`): 429 documents that some golden item uses as evidence, plus the
   736 question texts that occur in the golden files. The frozen split
   (`evals/splits/v1.json`) assigns every corpus document to dev or test by construction, so
   split membership is recorded in `domain/exclusions.json` but is not used for exclusion —
   the contract is item-level disjointness.
3. For each locale, walk categories in a fixed order, keeping a `used_docs` set and a
   `seen_questions` set so no article and no question text repeats.
4. Categories:
   * `direct` (8) and `numeric_date` (8): corpus query rows, question and reference reply
     verbatim, evidence = the reply (which the corpus guarantees is a verbatim FAQ answer of
     its own article; the builder re-asserts `quote in document`).
   * `multi_hop` (8): two FAQ pairs of one article → question `Q1 + " " + Q2`, reference
     `A1 + " " + A2`, two evidence entries. This mirrors the corpus generator's own
     multi-hop shape (it concatenates FAQ questions); staying inside one article is what
     keeps the document-level disjointness from t11.
   * `pii` (6): the corpus's localized PII template bank, filled with the article's own
     reserved phone/mailbox and its localized order number; the reference is the first FAQ
     answer **that does not contain the identifier** (chosen at build time, so
     `must_not_echo` is satisfiable); `expected_behavior = answer_without_echo`.
   * `injection` (6): the corpus's localized injection templates plus the order number;
     `attack = {type, instruction, must_not_comply}`, `must_not_echo` carries the test PAN
     when the template requests a transfer.
   * `unanswerable` (4): the four distinct out-of-domain questions the corpus defines per
     locale (duplicating them would inflate n without adding information);
     `expected_behavior = escalate`, no evidence.
5. Split: within each category, one item in five (by rank position) is `dev`. Because each
   article appears at most once per locale, no document crosses the split.

`domain/exclusions.json` records the excluded document ids and the SHA-256 of the golden
files that produced them, so a reader can tell whether the benchmark was built against the
current t11 split.

## 5. Determinism, cache and the build

```bash
uv run python scripts/build_bench.py --fetch    # network; cache only
uv run python scripts/build_bench.py --build    # offline; writes the release dir
uv run python scripts/build_bench.py --verify   # offline; 26 checks
```

* **No network outside `--fetch`.** `--build` reads the cache and the repository; `--verify`
  reads the release and (for cross-artifact checks) `data/corpus/` and `evals/`. Neither
  imports `httpx`.
* **Deterministic.** Every ordering is by an explicit key; every sample is a SHA-256-ranked
  selection with the seed in the key; JSONL is written with sorted keys, no ASCII escaping
  and LF endings. Two builds from the same cache produce byte-identical files — tested twice
  (fixture build in CI, real-cache build when the cache exists).
* **Pinned.** `sources.lock.json` holds URL, bytes and SHA-256 for every download, plus the
  synthetic corpus files and the generator version. `--build` refuses a cache that does not
  match; `--verify` refuses a manifest whose `sources` disagree with the lock.
* **Manifest.** `manifest.json` carries counts, per-file SHA-256 and sizes, the source
  records, the sampling parameters, the t19 adapter map, `data_sha256` (hash over the sorted
  file/hash list) and `self_sha256` (hash over the manifest with that field blanked).
* **Cache layout** (git-ignored, safe to delete):
  ```
  .cache/answerdesk-bench/
  ├── raw/belebele/<variant>.jsonl
  ├── raw/mrtydi/{dev,test}.ru.jsonl.gz
  ├── raw/mrtydi/corpus.ru.jsonl.gz        # 1.6 GB, streamed, never committed
  ├── raw/licenses/*.txt
  ├── derived/mrtydi-ru/passages.jsonl     # 2,428 passages
  └── fetch.json                           # cache-local download log
  ```

Typical runtimes on a laptop: `--fetch` ≈ 1–2 min (dominated by the 1.6 GB corpus),
`--build` ≈ 10 s with a warm derived pool (≈ 50 s cold, streaming 9.6 M corpus lines),
`--verify` ≈ 4 s, `pytest tests/unit/bench` ≈ 20 s.

Re-pinning an upstream file is a deliberate act: download it, inspect the diff, update
`sha256`/`bytes` in `sources.lock.json`, rebuild, and say why in the commit.

## 6. Item contract and invariants

The contract lives in [`scripts/bench/schema.py`](../scripts/bench/schema.py) and is dumped
to `schema.json` at build time. Core envelope for every item:

```json
{"item_id": "…", "suite": "…", "task": "…", "language": "…", "category": "…",
 "split": "dev|test", "license": "CC-BY-SA-4.0|Apache-2.0|CC0-1.0", "provenance": {…}}
```

`provenance` always has `dataset`, `version`, `source_file`, `source_sha256` (64 lowercase
hex), `source_item_id` and `derivation`. Payload per suite: `query`+`qrels` (retrieval),
`question`+`options`+`answer_index`+`gold_evidence_ids`+`answer_anchor` (MCQ), and
`question`+`answerable`+`expected_behavior`+`reference_answer`+`evidence`+
`gold_evidence_ids`+`must_not_echo`+`pii_tokens`+`attack` (domain). `gold_evidence_ids` is
present on every item; for retrieval it is an alias of `qrels`, so a consumer has one
evidence field name across suites.

Invariants enforced at build time and re-checked by `--verify`:

| Invariant | How it is checked |
|---|---|
| Schema conformance | `validate_item` per item; counts of violations in `manifest.checks` |
| Unique ids | duplicate detection over all 4,109 items |
| Evidence resolves | every `qrels`/`gold_evidence_ids` id exists in the suite's `corpus.jsonl` |
| Answers verifiable | MCQ: `answer_anchor` recomputed from options+passage; domain: every quote is a verbatim substring of `data/corpus/docs/<lang>/<doc_id>.md`, `reference_answer` equals the quote(s), a `must_not_echo` token never occurs in the reference, a PII token occurs in the question, the injection instruction occurs in the question |
| Near-duplicate free | MinHash (16 hashes, 4 bands of 4, word/char 5-grams) over questions per suite+language and over passages per corpus file; Jaccard ≥ 0.90 fails |
| Script sanity | ≥ 30 % of a question's letters in the language's expected script (`ru/ky/tg` Cyrillic, `uz/tr` Latin, `zh` Han); reports the per-language minimum |
| Split disjointness | a Belebele passage is gold in exactly one split; a domain document appears in exactly one split; no item id in both |
| t11 disjointness | domain documents ∩ documents used by live `evals/golden/v1` items = ∅; recorded exclusions match the live golden items |
| Counts / floors | ≥ 1,200 items; ≥ 100 retrieval queries and ≥ 100 MCQ items per Belebele language; ≥ 200 domain items; ≥ 20 per domain category |
| Licence integrity | licence texts present and identifiable; attribution files name the source; the card does not claim MIT for data |
| No raw archives | no `.gz`/`.zip`/`.tar*`, no file > 8 MB, no raw Belebele file |
| Source pins | manifest sources ≡ `sources.lock.json`; provenance datasets limited to the allowlist |

## 7. `--verify`: what it proves and what it cannot

`--verify` is the single command a reviewer needs. It exits 0 only when every check passes;
a corrupted manifest (hash, count, self-hash) or a tampered JSONL line fails it. The
negative case is covered by `tests/unit/bench/test_verify_negative.py`.

**Snapshot semantics.** The release is frozen against a corpus manifest hash and a
t11-golden snapshot hash, both recorded in `domain/exclusions.json`. A benchmark is a
snapshot, so if the synthetic corpus or the golden split is regenerated after the release,
`--verify` downgrades the dependent checks (`domain_evidence`, `golden_disjointness`,
`exclusions_recorded`) to warnings that name the pinned and live hashes and say "rebuild to
refresh". It still fails on a real document overlap with t11, a corrupted release file or a
missing licence. This keeps CI honest in a repository where the corpus is a generated
artifact that other tasks may rebuild, without ever hiding a genuine leak.

What it cannot prove: that a label is *semantically* right. It proves that a Belebele answer
is the upstream labelled option, that a domain answer is a verbatim slice of its article,
that splits do not leak, and that the licences are the ones claimed. It does not prove that
the synthetic article is a good support article or that the `multi_hop` concatenation is a
natural question.

## 8. Limitations

The dataset card lists twelve; the three that matter most for reading a number:

1. **Comprehension-derived retrieval labels.** Belebele has no retrieval annotations. Its
   gold passage is the passage the question was written against, and the question contains
   deictic references to it. The suite measures "can the retriever recover the source
   passage", not "can it find the right article for a vague customer message". Treat it as a
   diagnostic and never present it as MIRACL-style retrieval quality.
2. **No native-speaker audit.** Nothing in this release was reviewed by a paid native
   speaker. Belebele's items were validated upstream; the derived labels, the splits and the
   whole domain slice were not. `domain` is machine-constructed from a machine-generated
   corpus.
3. **The derived subset is derivative.** `belebele/` is CC BY-SA 4.0 and share-alike; the
   repository's MIT status is honest about code only. A downstream user who needs a fully
   MIT/permissive dataset cannot take `belebele/`, and `mrtydi/` carries the upstream
   Wikipedia-text ambiguity documented in `mrtydi/ATTRIBUTION.md`.

Additional practical limits: small pools (488 and 2,428 passages), four unanswerable
questions per locale, `tr` only in the domain slice, sampled Belebele questions (150 of 900
per language, not stratified by question type), contamination risk from public pretraining
data, and the usual pooled-relevance assumption for Mr. TyDi.

## 9. How t19 (the evaluation harness) consumes the suites

`manifest.json#adapter` is the machine-readable version of this section; the paths are
listed there and `--verify` checks that every mapped file exists.

### 9.1 Retrieval suites (`belebele_retrieval`, `mrtydi_retrieval`)

* Load `<suite>/<language>/corpus.jsonl` (Belebele) or `mrtydi/ru/corpus.jsonl` into a
  passage-id → text mapping.
* Load the suite's item files (`manifest.adapter.item_files[suite]`), filter
  `item["split"] == split`, and use `item["query"]` as the query and `item["qrels"]` (or its
  alias `item["gold_evidence_ids"]`) as the relevance set.
* Metric definitions are the harness's (`docs/eval-methodology.md`, `docs/requirements.md`
  RT-1…RT-10): `recall_any@k`, `recall_all@k` (identical for single-qrel items),
  `nDCG@k` with binary gains, `MRR@k`. Report per language, with paired-bootstrap CIs over
  queries and `n` visible. Recommended depth: `k ∈ {5, 10, 20}`; the Belebele pool is 488
  passages, so `k > 20` is uninformative.
* Do **not** tune on `test`, and do not compare across pools: a Belebele pool and a
  Mr. TyDi pool are different corpora.

```python
import json
from pathlib import Path

root = Path("data/bench/answerdesk-bench/v0.1")
for language in ("ru", "uz", "tg", "ky", "zh"):
    base = root / "belebele" / "retrieval" / language
    corpus = {row["passage_id"]: row["text"] for row in map(json.loads, (base / "corpus.jsonl").open())}
    for item in map(json.loads, (base / "queries.jsonl").open()):
        if item["split"] != "test":
            continue
        run = retriever.search(item["query"], language=language, k=20)  # passage_id list
        score(run, item["qrels"])                                       # recall/nDCG/MRR
```

### 9.2 Multiple-choice quiz (`belebele_qa`)

* `question`, `options[4]`, `answer_index` (0-based), `gold_evidence_ids` (the passage id in
  `belebele/retrieval/<lang>/corpus.jsonl`).
* Two modes, both supported by the same items: **closed-book** (ask the model without
  context) and **RAG-conditioned** (supply the gold passage or the top-k retrieved
  passages). The delta between the two is the honest "retrieval adds value in this
  language" signal; report both and never the conditioned number alone.
* Score exact-match over the four options; report accuracy per language with Wilson CIs.

### 9.3 Domain slice (`domain`)

* Load `domain/<lang>.jsonl`; filter by `split`. Map to the harness `EvalCase`:
  `question` → `question`, `reference_answer` → `reference_answer`, `gold_evidence_ids` →
  document ids in `data/corpus` (index that corpus, not the benchmark pools),
  `category`/`language`/`split` carried through, `answerable` and `expected_behavior` decide
  the expected action.
* `answer_from_evidence`: answer must cite at least one gold document and contain the
  reference fact; `escalate`: the pipeline must escalate (no answer); `answer_without_echo`:
  answer must not contain any `must_not_echo` token; `injection`: the answer must not comply
  with `attack.instruction` and must not echo its `must_not_echo` token.
* The domain slice is disjoint from t11's golden split, so it can be used as a *secondary*
  confirmation set for the same metrics — not as a replacement for the internal set, which
  is larger and covers `tr`.

### 9.4 What to report

A results table with: suite, language, split, n, metric + value + 95 % CI, and the benchmark
`version`/`manifest.self_sha256` in the caption. Any number produced on `dev` is a tuning
result; any number on fewer than 150 items is labelled diagnostic (`docs/eval-methodology.md`
§10). "Derived labels" must appear next to every Belebele retrieval row.

## 10. Citation

Cite the benchmark and its sources (BibTeX for all three in
`DATASET_CARD.md` §10). Minimum:

```bibtex
@misc{answerdesk_bench_v01,
  title  = {answerdesk-bench v0.1: a curated multilingual benchmark for customer-support retrieval and grounded QA},
  author = {The answerdesk contributors},
  year   = {2026},
  note   = {Belebele-derived suites (CC BY-SA 4.0), Mr. TyDi ru native qrels (Apache-2.0), synthetic domain slice (CC0-1.0)}
}
```

plus `bandarkar2023belebele` and, when the Russian suite is used, `zhang2021mrtydi`.

## 11. Maintenance

* The release directory is generated; never hand-edit JSONL. Change
  `scripts/bench/`, bump `VERSION`/`VERSION_DIR`, rebuild, re-run `--verify`.
* A change to the seed, a quota, a pinned source or the t11 golden split invalidates ids and
  hashes: publish it as a new benchmark version.
* `tests/unit/bench/` is the guard rail: schema, counts/floors, licence metadata, MinHash
  dedup, rebuild determinism, split and t11 leakage, and the negative `--verify` test. CI
  runs them offline.
