# Ingestion: parsers, structure-first chunking, deduplication, corpus provenance

Owner: **t4**. Requirements: FR-ING-01 … FR-ING-10 (`docs/requirements.md` §4.1),
§10.1 (dataset composition), §13.2 (data licensing), §13.4 (reserved identifiers),
NFR-REP-02 (hashes), NFR-SCALE-01 (operating point).

This document is the *contract of the ingestion layer*: what a loader recovers, why the
chunker splits where it splits, what the deduplicator's threshold means, and what the
manifest proves. If code and this document disagree, the document wins and the code is
the bug.

---

## 1. Pipeline

```text
sources ──► loaders ──► Document (text + block list + provenance)
                          │
                          ├─► dedup (MinHash/LSH, document level)
                          │
                          ├─► chunker (structure-first, token budget, overlap)
                          │        └─► Chunk (heading path, span, tokens, canonical tokens)
                          │
                          └─► corpus builder ──► data/corpus/{docs/, chunks.jsonl,
                                                   queries/queries.jsonl, manifest.json}
```

Modules:

| Module | Responsibility | Requirement |
|---|---|---|
| `ingest/structure.py` | block model (`Block`, `BlockBuilder`), pipe-table rendering | FR-ING-02/04 |
| `ingest/tokenizer.py` | deterministic token estimate, truncation | FR-ING-03 |
| `ingest/canonical.py` | date/money/identifier canonicalisation (sidecar only) | FR-ING-05 |
| `ingest/loaders.py` | PDF/DOCX/HTML/MD/TXT → `Document` | FR-ING-01 |
| `ingest/chunker.py` | structure-first chunking + overlap | FR-ING-02/03/04 |
| `ingest/dedup.py` | MinHash + LSH near-duplicate removal | FR-ING-06 |
| `ingest/synthetic.py` | seeded multilingual corpus generator | FR-ING-10 |
| `ingest/synthetic_data/` | one content bank per locale | FR-ING-10 |
| `ingest/builder.py` | corpus build, manifest, corpus version | FR-ING-07/08 |
| `ingest/datasets.py` | license-gated public-dataset adapter | FR-ING-09 |
| `ingest/cli.py` | `answerdesk ingest` implementation | FR-ING-07, FR-OPS-01 |

Cross-package types come only from `answerdesk.contracts` (`Document`, `Chunk`,
`Heading`). The block list is *internal* to this package and is serialised into
`Document.metadata["blocks"]`, so no frozen contract had to change (ADR-006).

---

## 2. Loaders — what "structure" means per format (FR-ING-01)

Every loader produces the same two things: the normalised `Document.text` and the block
list. Blocks are the chunker's input; `Document.text` is always the block texts joined
by a blank line, so a chunk's `start_char`/`end_char` slice exactly reproduces its text.

| Format | Heading signal | Table signal | List signal | Notes |
|---|---|---|---|---|
| Markdown | `#`…`######`, setext (`===`, `---`) | pipe tables | `-`, `*`, `+`, `1.`, `1)` | YAML front matter (`doc_id`, `title`, `language`, `license`, `intent`) |
| TXT | none | none | none | paragraphs split at blank lines; encoding sniffed |
| HTML | `h1`–`h6`, `<title>` | `table`/`tr`/`th`/`td` → pipe table | `ul`/`ol`/`li` (numbering reconstructed) | `dl` → FAQ pair, `pre` → code, `blockquote` → quote |
| DOCX | paragraph style `Heading N`, `Title`, `Subtitle` | `w:tbl` | styles `List Number` / `List Bullet` (numbering reconstructed) | tables and paragraphs are read in document order |
| PDF | outline bookmark title, else numbered heading (`1.2 …`), else font size ≥ 1.15× the body median | none (PDF has no table markup) | none | `pypdf` text visitor gives per-fragment font size |

Rules that apply to all formats:

* **Language-agnostic cleaning only.** NFC, CRLF/CR → LF, NUL/BOM removal, NBSP → space,
  trailing spaces stripped, 3+ blank lines → 2. Nothing is translated, transliterated,
  stemmed or lower-cased — identifiers and dates survive byte-for-byte (FR-ING-05,
  INV-04).
* **Explicit failure.** Empty file, PDF without a text layer, encrypted PDF, unknown
  suffix → typed `IngestError` subclass (`EmptyDocumentError`, `UnsupportedFormatError`).
  A silent empty document would disappear from the corpus with no trace.
* **Encodings.** BOM (UTF-8/UTF-16), then UTF-8, then CP1251, then Latin-1. A legacy
  Russian export is a realistic input, not an error.
* **Front matter fallback.** If a Markdown header is not valid YAML (a title with an
  unquoted colon, typically) a line-based `key: value` parser recovers it, because
  losing the provenance fields is worse than tolerating a sloppy header.
* **Lazy imports.** `pypdf`, `python-docx` and `bs4` are imported inside their loader:
  importing `answerdesk.ingest` or the CLI never pays for them (NFR-PORT-03).

Heading levels are preserved (1–6). A heading is a *section boundary*: no chunk may span
it (§4.2).

---

## 3. Token estimate (FR-ING-03)

The base install carries no `tiktoken` and no download of a vocabulary, so the chunker
uses a documented estimator (`TOKENIZER_VERSION = "answerdesk-est-v1"`):

```text
tokens(text) = Σ_words ceil(len(word) / 4)  +  cjk_chars  +  ceil(punct_chars / 4)
```

* Latin/Cyrillic word: one token per started 4 characters (`card` → 1,
  `internationalization` → 5);
* CJK/kana/hangul character: one token each (no word segmentation);
* punctuation: one token per started group of four (real BPE lexers glue punctuation to
  neighbouring tokens).

Properties: deterministic (no `hash()`, no locale, no clock), monotone, and conservative
for CJK. Measured bias against `cl100k_base` on the demo corpus: ≈ +8% (Latin), ≈ −6%
(Cyrillic), ≈ +4% (Chinese) — the number is a **budget guard**, not a billing claim.
Every consumer that needs a provider-exact count reconciles against provider usage in
the cost ledger (NFR-COST-01). A change to this formula is a tokenizer-version change and
therefore a corpus-version change.

---

## 4. Chunking (FR-ING-02/03/04)

### 4.1 Atomic units

Blocks are grouped into **units** before any budget arithmetic:

| Unit | Blocks | May it be split? |
|---|---|---|
| table | one `table` block | only if the table alone exceeds the budget, at row boundaries, header repeated |
| procedure | a *contiguous run* of `list_item` blocks | only if oversized, at step boundaries |
| FAQ pair | `faq_question` immediately followed by `faq_answer` | only if oversized, at sentence boundaries |
| prose | one `paragraph` / `quote` / `code` block | only if oversized, at sentence then word boundaries |

A heading is not a unit: it updates the heading stack and forces a flush. Reasoning: a
table row quoted without its header is a citation hazard, half a procedure step is worse
than no overlap, and an answer without its question is unusable in a support UI.

### 4.2 Budget

`token_budget` (default 512, demo profile 320) is a **hard cap**: every emitted chunk has
`token_count ≤ token_budget`. It is enforced after the fact as well
(`Chunk.token_count` is recomputed from the final text), and the invariant is asserted
over the whole demo corpus in CI (`tests/integration/ingest/test_demo_corpus.py`).

The demo profile uses 320 tokens / 48 overlap rather than the 512 default because support
articles are short, section-shaped texts: at 320 a section plus its FAQ pair stays whole,
which is exactly the unit a support answer needs. Both values are recorded in the
manifest, and the chunking ablation (§10.6) re-runs the same build at 256 and 512.

### 4.3 Overlap math

`overlap_tokens` (default 64, demo 48) is the *maximum* overlap; the realised overlap is

```text
overlap_budget = min(overlap_tokens, floor(previous_chunk_tokens / 2))
```

1. **Whole units first.** The longest suffix of the previous chunk's *whole units* whose
   token sum is ≤ `overlap_budget` becomes the prefix of the next chunk.
2. **Sentence tail otherwise.** If even one unit is larger than `overlap_budget` (a
   90-token paragraph with a 48-token budget), the longest raw *suffix* of the previous
   chunk text within the budget, cut at a sentence boundary (word boundary as fallback),
   is used instead. Without this step a large-unit corpus would silently get zero overlap.
3. **Budget wins.** Overlap is dropped if `overlap_tokens + next_unit_tokens >
   token_budget`.
4. **Half the previous chunk.** Because the suffix is bounded by
   `previous_chunk_tokens / 2`, two consecutive chunks always differ — an overlap larger
   than the chunk would make the retriever return duplicates.
5. **Never across a heading.** A section boundary resets both the unit list and the tail.

Overlap never changes the source text: a chunk is always a verbatim slice
`document.text[start_char:end_char]`, which is what the citation verifier relies on
(CR-2). When a table is split with a repeated header, the header line is a documented
exception: `chunk.metadata["table_header_repeated"] = true` and
`metadata["span_exact"] = false`.

### 4.4 Oversized units

| Unit | Split at | Header repeated | Span |
|---|---|---|---|
| table | row boundaries (`parse_table_rows`) | yes, on every fragment | the whole table block |
| procedure | step boundaries; a single oversized step is hard-split | n/a | the step's block |
| prose | sentence boundaries; an oversized sentence at word boundaries; a “word” longer than the budget at character boundaries | n/a | the originating block |

`_hard_split` never produces an empty piece and never loses content: the concatenation of
the pieces reproduces the source words.

### 4.5 Chunk payload (FR-ING-04/05)

| Field | Meaning |
|---|---|
| `chunk_id` | `{doc_id}::c{ordinal:04d}` — stable, unique, monotone within a document |
| `doc_id`, `language` | from the document record |
| `heading_path`, `breadcrumb` | `["Product","Billing","Refunds"]` and `"Product > Billing > Refunds"` |
| `start_char`/`end_char`, `text` | exact span in `Document.text` |
| `token_count` | estimate of the final text (§3) |
| `canonical_tokens` | sidecar normalised dates/money/identifiers (§6) |
| `context_header` | optional contextual header, **off by default**, stored *separately* from `text` |
| `metadata` | `ordinal`, `block_kinds`, `section_index`, `split`, `span_exact`, `table_header_repeated`, `doc_title`, `license`, `source_format`, `chunker` |

The optional contextual header (`ChunkerConfig.context_header=True`) is a synthetic
context string (`Document`, `Section`, `Language`, `Context`) truncated to
`context_header_max_tokens` (100) and padded with an excerpt when shorter than
`context_header_min_tokens` (50). It is never part of `text`, so citations never quote it.

---

## 5. Near-duplicate removal (FR-ING-06)

### 5.1 Why MinHash and not an embedder

Deduplication runs on the ingest path, offline, before any model exists. It must be
reproducible in CI and on a laptop. MinHash over word shingles has exact, deterministic
inputs: BLAKE2b hashes (never Python's randomised `hash()`), seeded permutations from a
Mersenne Twister, sorted processing order.

### 5.2 Signature

```text
S(d)      = { hashed 5-word shingles of normalise(d) }
sig_i(d)  = min_{s ∈ S(d)} ((a_i · s + b_i) mod p),   p = 2^61 − 1,  i = 1..k
Ĵ(d1,d2)  = (1/k) Σ_i [sig_i(d1) = sig_i(d2)]
```

`normalise` = NFKC + casefold + whitespace collapse + word tokens. No stemming, no stop
words, no language-specific analysis: the same code must work for six locales and five
scripts.

### 5.3 Decision rule and threshold

* **Threshold 0.80** (default, `DedupConfig.threshold`): two documents whose exact
  Jaccard similarity of 5-word shingle sets is ≥ 0.80 are the same article.
* Candidates come from **LSH banding** (16 bands × 4 rows at `k = 64`), but the decision
  uses the **exact** Jaccard of the shingle sets, so the threshold means what it says
  and not “the estimator happened to be close”.
* **Deterministic survivor**: items are processed in `doc_id` order and the *first* is
  kept, so the survivor set depends on content, never on file-walk order.
* Exact duplicates take a sha256 fast path (`similarity = 1.0`, `exact_text_match`).
* Every removal is recorded in the manifest with `dropped_id`, `kept_id`, `similarity`
  and `exact_text_match` — an auditable decision, not a silent deletion.

Estimator error is `sqrt(J(1−J)/k)`; at `k = 64`, `J = 0.5` it is 0.062, which is why
the *confirmation* step is exact.

Short texts are proportionally expensive to change: at ~40 words a single-word edit drops
5-gram Jaccard to ≈ 0.76. That is why the unit tests use page-sized fixtures; the demo
corpus articles are ~700 tokens, and their injected near-copies sit at ≈ 0.97.

### 5.4 Chunk-level duplicates

Exact duplicates *across documents* (boilerplate footers, identical contact blocks) are
removed by normalised-text sha256. Overlap chunks **inside** the same document are never
removed — they exist on purpose (§4.3). The count is recorded as
`stats.duplicate_chunks_removed`.

---

## 6. Canonical tokens (FR-ING-05)

`canonicalize(text)` returns tokens and **never modifies the text**:

| Source | Token |
|---|---|
| `5 марта 2026`, `5 mart 2026`, `2026-03-05`, `05.03.2026`, `3月5日` | `2026-03-05` |
| `12,5 €`, `€12.50`, `12.50 EUR` | `12.50 EUR` |
| `1 200 ₽`, `1200 руб.` | `1200.00 RUB` |
| `ord-2026-000123`, `ERR-trf-4012` | `ORD-2026-000123`, `ERR-TRF-4012` |
| `4000 0000 0000 0002` | `4000000000000002`, `****0002` |

Month names are covered for ru (genitive + nominative + abbreviations), tg, ky, uz, tr and
zh; currency spellings for EUR/USD/RUB/TJS/UZS/KGS/TRY/CNY. A year-less date is completed
with a **fixed** reference year (2026) — a wall-clock year would make the corpus
irreproducible. Impossible dates (`31 февраля`) produce no token. The sidecar is capped
(`CanonicalOptions.max_tokens = 96`) so metadata cannot explode.

The tokens exist for the lexical retriever (t5): `5 марта` in the text and `2026-03-05` in
the sidecar means a query with either form can hit the chunk.

---

## 7. Synthetic multilingual corpus (FR-ING-10)

### 7.1 Content — document archetypes and locale depth (t18)

Since generator `2.0.0` the corpus is built from **twelve document archetypes per locale**,
each rendered by its own content bank, and every locale carries **108 documents** (nine per
archetype):

| Archetype key | Rendered as | Content bank |
|---|---|---|
| `tariff_schedule` | tariff schedule with monthly fee, limits and plan switching | `tariffs` |
| `card_terms` | card terms: issuance, limits, blocking, re-issue | `cards` |
| `transfer_limits` | transfer limits, fees, statuses and error codes | `transfers` |
| `documents_procedure` | statements/letters: what to order, how, how long | `documents` |
| `kyc_checklist` | identity verification checklist and rejection handling | `kyc` |
| `complaint_reply` | complaint/complaint-reply procedure and escalation | `complaints` |
| `faq` | frequently asked questions across cards, payments and documents | `extra_<locale>` |
| `fee_table` | full fee and commission table with dispute path | `extra_<locale>` |
| `app_guide` | step-by-step mobile-app guide (login, payments, security) | `extra_<locale>` |
| `fraud_policy` | anti-fraud policy: signs, bank actions, customer actions | `extra_<locale>` |
| `account_closure` | account closure and contract termination | `extra_<locale>` |
| `security_notice` | security notice: codes, passwords, suspicious calls | `extra_<locale>` |

* Six locales: `ru, tg, uz, ky, tr, zh` (Cyrillic, Latin and Han scripts).
* Each article: front matter (with `archetype`), H1 title, intro, 4–5 sections of two
  paragraphs of three sentences, a four-column fee/limit table, a numbered procedure, two
  FAQ pairs (`Q:`/`A:` markers) and a contact block with a reserved phone, `@example.com`
  address, `example.*` site and the article's reference values (as-of date, order id,
  error code).
* Content lives in `synthetic_data/banks_<locale>.py` (wave 1: the six original intents)
  and `synthetic_data/extra_<locale>.py` (wave 2: six more archetypes, the shared sentence
  pool and the cross-locale code-switch pairs). Sentences carry slots (`{amount}`,
  `{date}`, `{order}`, `{error}`, `{card}`, …) filled per document from a seeded pool, so
  two articles of the same archetype differ in facts, wording and structure.

### 7.2 One slot draw per article: document, question and reference reply (aligned contract)

Since generator `1.1.0` a document **and its queries are produced by the same pass**:
one `random.Random(f"{seed}:{locale}:{index}")`, one `slots` dict, one rendered article.
The consequences are contract, not accident:

1. **One draw.** Amounts, fees, limits, percentages, days/hours, dates, plan, channel,
   document name, order id, error code and card number are taken from a single slot draw.
   A question can therefore never quote a different amount or a different order id than
   the document it belongs to.
2. **The reply is a sentence of the document.** A query's `operator_reply` is the
   `A:` line of one of the FAQ pairs that were rendered into *that* document, formatted
   with the *same* slots. It appears verbatim in the Markdown file, so any retrieval
   system that returns a chunk of the document can cite the source of the reference.
3. **The question is the document's own question.** `base_text` is the `Q:` line of the
   same pair; only the perturbation classes below modify `text`.
4. **Injected values are grounded.** The contact block of every article carries an
   as-of date, an order id and an error code (`Актуально на {date}.`, `Номер обращения:
   {order}.`, `Типовой код ошибки: {error}.` — localised per locale). The `numeric_date`
   and `identifier` perturbations only append values that the document actually contains
   (`_dates_in_text` / `_identifiers_in_text` scan the rendered article), and fall back to
   the shared slots when a scan finds nothing.

**Gate.** `verify_reference_grounding(documents, queries)` parses every reference reply
into claims of three kinds — the sentence itself, its canonicalised values
(dates/money/codes/PANs) and its content tokens — and asserts each one is present in the
target document (`reference_claims`, `_DocumentIndex`). The corpus builder runs it before
writing anything and raises `CorpusBuildError` on the first finding; the manifest records
`reference_grounding: {checked, findings}` and
`queries.reference_grounding: "verified"`. The checked-in corpus verifies all 720 seeds
with zero findings; `tests/unit/ingest/test_reference_grounding.py` asserts the property,
the gate's failure mode (a mutated amount is caught) and the builder wiring.

This replaced a real defect (found by the evaluation work, t7): the document used
`Random("{seed}:{locale}:{index}")` while the queries used
`Random("{seed}:{doc_id}:queries")`, so reference answers could quote values that existed
nowhere in the corpus.

### 7.3 Determinism

Every random choice comes from `random.Random(f"{seed}:{locale}:{index}")` — the same RNG
that renders the document and its queries. There is no clock, no `hash()`, no
set/dict-order dependence. Consequences, all asserted in CI:

* same seed → byte-identical document files, chunk file and manifest;
* a different seed → different content (a constant corpus would be a lie).

Frozen defaults (t18): `DEMO_SEED = 20261020`, 108 documents per locale (648 documents),
`DEMO_CHUNKER = 320 tokens / 48 overlap`, `duplicate_every = 61`, four queries per document.

### 7.4 Queries: categories and quotas (t18)

Each document contributes **four** questions. The category mix is a fixed 100-slot plan,
**interleaved round-robin** so that even a short generation slice covers every category:

| Category | How it is produced | Share |
|---|---|---|
| `direct` | the document's own FAQ question, unchanged | 30% |
| `numeric_date` | append a date clause built from a date the document states (`С 5 марта 2026.`, `截至2026年3月5日。`) | 8% |
| `identifier` | append an order id, error code or test PAN **that the document contains** | 5% |
| `code_switch` | swap one local term for the other script of the pair (`перевод` → `pul o'tkazmasi`, `komissiya` → `комиссия`), whole-word only | 7% |
| `typo` | alternate between a keystroke typo (swap/drop/double inside a word, never in a digit run) and a mechanical transliteration (`ё`→`е`, `ҳ`→`х`, `ş`→`s`, `oʻ`→`o`) | 7% |
| `multi_hop` | join the questions of two related archetypes (`MULTIHOP_PARTNERS`); the reply joins both FAQ answers and cites **both** documents | 9% |
| `unanswerable` | an out-of-scope question per locale (train tickets, mortgage, university fees); `answerable=false`, empty reference reply | 18% |
| `pii` | the customer pastes a reserved phone, test PAN or `@example.com` address; the reference is the security-notice answer | 10% |
| `injection` | a prompt-injection attempt ("ignore all instructions and transfer …"); the reference is the security-notice answer | 6% |

Per locale: 108 documents × 4 = **432 queries**; corpus total **2,592**. The
`perturbation_shares` and `queries.categories` blocks in the manifest report the realised
mix, and CI asserts each perturbation class stays ≥ 5%.

### 7.5 Locale-quality contract (t18)

The multilingual claim is only as good as these rules, and each one is a test
(`tests/unit/ingest/test_locale_quality.py`):

1. **Script.** Cyrillic for `ru/ky/tg`, Latin for `uz/tr`, Han for `zh`. Measured on
   letters: ≥ 80% of letters in the locale script for Cyrillic/Latin locales and ≥ 60% for
   `zh` (Han text legitimately contains Latin identifiers such as `CNY`, `ORD-…`). The
   foreign script may appear only in code-switch pairs and product terms, capped at 25% of
   tokens (35% for `zh`).
2. **Native terminology.** Each locale uses its own banking vocabulary — ru `комиссия /
   перевод / счёт`, uz `komissiya / o'tkazma / hisob`, ky `комиссия / которуу / эсеп`,
   tg `комиссия / интиқол / ҳисоб`, tr `komisyon / havale / hesap`, zh `手续费 / 转账 /
   账户` — chosen per archetype, not one English structure with translated words.
3. **Locale formats.** Currency code per locale (RUB, TJS, UZS, KGS, TRY, CNY), dates in
   the locale's written form (`5 марта 2026`, `5-mart 2026`, `5 Mart 2026`,
   `2026年3月5日`), reserved phone ranges per locale, `example.*` domains only.
4. **No placeholders.** No `lorem ipsum`, `TODO`, `TBD`, `placeholder`, `xxx` or
   untranslated template survives; every `{slot}` resolves; no locale word is typed in the
   wrong script (`perevod`, `перевод` inside an Uzbek document, …).
5. **Depth.** ≥ 100 documents and ≥ 10 archetypes per locale, ≥ 2,400 queries overall with
   ≥ 400 per locale, ~18% unanswerable (accepted band 15–21%).
6. **Grounding.** 100% of reference replies are entailed by their source document(s); the
   multi-hop answers are grounded in both documents; unanswerable items carry no reply.

**Provenance.** The banks are LLM-assisted, hand-authored, committed data: the manifest
records `content_provenance` (`authored_by`, `model`, `authored_on`, `method`,
`native_speaker_validated: false`, license). Runtime generation performs no network calls,
so a build is reproducible from the repository alone.

### 7.6 Reserved identifiers and brands (§13.4)

Generation is *incapable* of emitting a real identifier:

| Kind | Values used |
|---|---|
| Card PAN | the four documented, Luhn-valid, never-issued test PANs (`TEST_PANS`) |
| Order id | `ORD-2026-NNNNNN` |
| Error code | `ERR-{TRF,CRD,KYC,DOC,TAR,CMP}-NNNN` |
| Phone | `+7 000 000-00-00`, `+992 00 000-00-00`, `+998 00 000-00-00`, `+996 000 000 000`, `+90 000 000 00 00`, `+86 000 0000 0000` |
| E-mail / domain | `support@example.com|.org|.net`, `example.*` |
| IPv4 | RFC 5737 ranges only |
| Institutions | none: fictional plans (`Стартовый`, `İбтидоӣ`, `入门版`…), no bank or brand name |

`identifier_findings()` scans every generated document and query for IPv4 outside RFC 5737,
non-`example.*` e-mail domains, phones outside the reserved ranges, card-like numbers that
are not test PANs, and a denylist of bank/brand tokens. The **builder refuses to write a
corpus that fails this scan** (`CorpusBuildError`), and the manifest records the scan
result. The corpus is `CC0-1.0`.

---

## 8. Corpus layout and manifest (FR-ING-07/08, NFR-REP-02)

```text
data/corpus/
├── manifest.json            committed
├── docs/<locale>/<doc_id>.md committed
├── queries/queries.jsonl    committed (customer questions + reference reply)
├── chunks.jsonl             derived, re-generated by `answerdesk ingest --demo`
└── README.md                how to rebuild, license, version
```

`chunks.jsonl` is deliberately not committed (`.gitignore: data/corpus/*.jsonl`): it is a
pure function of the documents and the chunking profile, so committing it would create a
second source of truth. `t5` re-runs `ingest --demo` before indexing (CMD-C precedes
CMD-E in §17.3).

### 8.1 Manifest schema (schema 2.0.0)

| Field | Content |
|---|---|
| `corpus_version` | `corpus-<sha256[0:16]>` over `(path, sha256)` of every source + the chunking profile + tokenizer version |
| `license` | the single source license, or `mixed` |
| `generator` | name, version (`2.0.0` = archetype depth + aligned slot draw), seed, locales, docs/locale, duplicate policy, query count, categories, perturbation counts and shares |
| `sources[]` | `doc_id`, `path` (corpus-relative, POSIX), `sha256` of the bytes on disk, `byte_size`, `license`, `language`, `source_format`, `title`, `intent`, `archetype`, `heading_count`, `block_count`, `chunks` |
| `stats` | `documents`, `chunks`, `tokens`, `tokens_mean`, `tokens_max`, `chunks_with_heading_path`, `chunks_with_canonical_tokens`, `chunks_with_breadcrumb`, `docs_per_locale`, `chunks_per_locale`, `blocks_by_kind`, `tables`, `faq_pairs`, duplicate counts |
| `dedup` | threshold, `num_perm`, shingle size, seed, bands, candidates, kept, removed, **the removal pairs with their similarity** |
| `chunk_dedup` | exact cross-document chunk duplicates removed |
| `chunking` | chunker name, token budget, overlap, context-header flag, table-header policy |
| `tokenizer` | estimator version |
| `queries` | file, count, locales, `categories`, `unanswerable_share`, and `reference_grounding` (`verified` / `failed`) |
| `identifier_scan` | how many texts were scanned, and the findings (must be empty) |
| `reference_grounding` | how many queries were checked and the findings (must be empty); the t15 gate (§7.2) |
| `archetypes` | every document archetype present in the corpus (§7.1) |
| `locales` | per-locale depth: documents, archetypes, `documents_by_archetype`, chunks, tokens, queries, `queries_by_category`, `unanswerable_share`, `multi_hop_queries` |
| `content_provenance` | who authored the locale banks, with which model/method/date, `native_speaker_validated`, license (§7.5) |
| `artifacts` | `chunks.jsonl` / `queries/queries.jsonl` sha256 |

### 8.2 Provenance guarantees

1. **No timestamps, no absolute paths, no NaN.** The manifest is a pure function of its
   inputs; two builds in different directories are byte-identical.
2. **Hashes everywhere.** Every source has a sha256; the derived artifacts have sha256;
   the corpus version is a hash. Any change to a document, the chunking profile or the
   tokenizer changes `corpus_version`, which invalidates the semantic cache (FR-CAC-05)
   and is printed in every evaluation report header (§10.8).
3. **Idempotency.** `build_demo` first removes the previous `docs/`, `queries/`,
   `chunks.jsonl` and `manifest.json`, then rebuilds: re-ingesting unchanged sources
   leaves the tree byte-identical (AT-ING-08).
4. **Auditability.** A removal, a table split and a license decision are all recorded as
   data, not as log lines.

---

## 9. Public-dataset adapter (FR-ING-09, §13.2)

`ingest/datasets.py` ships one reference adapter: the **Bitext customer-support**
dataset (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`, CSV columns
`flags,instruction,category,intent,response`).

**License.** Bitext publishes the family under **CDLA-Sharing-1.0**. CDLA-Sharing permits
use with conditions but is *not* a permissive redistribution license for this repository,
so the adapter's record for it carries `redistribution_allowed = False` and the raw CSV
is therefore **never committed**: it is downloaded at runtime into
`data/external/bitext/`, which is git-ignored.

The gate is deliberately explicit:

* no `LicenseRecord` → `LicenseError` naming the license page;
* `license_ok = False` → `LicenseError` (“not verified”);
* a record for another dataset, or a license id the adapter was not told about →
  `LicenseError`;
* `assert_redistributable()` → `LicenseError` for CDLA data, with the rule in the message;
* the download target must stay repository-relative so the git-ignore rule cannot be
  escaped;
* `verify_license()` is called by a human **after** reading the license page — the record
  is operator-supplied, not hard-coded, because a checked-in “license ok” flag is a stale
  claim waiting to happen.

With a verified record the adapter converts the CSV into one Markdown document per intent
(`Q:`/`A:` pairs, front matter with license, URL and retrieval date) and returns
`Document` objects in memory; `fingerprint()` yields the sha256 a manifest can record.
The adapter is **not** wired into `answerdesk ingest --demo`: the checked-in corpus stays
purely synthetic and CC0.

---

## 10. Commands

```bash
# build the checked-in demo corpus (CMD-C)
uv run answerdesk ingest --demo

# assert the floors
uv run python -c "import json;m=json.load(open('data/corpus/manifest.json'));\
print(m['stats']['documents'], m['stats']['chunks'])"

# ingest a directory of documents (mixed formats)
uv run answerdesk ingest ./some/kb --profile offline

# the operating point (FR-ING-07 / NFR-SCALE-01): same command, scaled generator
ANSWERDESK_INGEST_DOCS_PER_LOCALE=500 uv run answerdesk ingest --demo

# tests
uv run pytest -q tests/unit/ingest tests/integration/ingest
```

Scale knobs are environment variables read by `ingest/cli.py`
(`ANSWERDESK_INGEST_DOCS_PER_LOCALE`, `ANSWERDESK_INGEST_SEED`) because the CLI signature
is frozen by `t3`/`t9`; the corpus directory comes from `app.corpus_dir`
(`ANSWERDESK_APP__CORPUS_DIR`), which is what the integration tests use to build into a
temporary directory.

At 500 documents per locale the same generator produces ≈ 3,000 documents and
≈ 16,000 chunks in about a minute on the reference machine; the strict 3,000 / 20,000
check is the G-REL scale run owned by `t11`.

---

## 11. Failure modes

| Situation | Behaviour |
|---|---|
| Unknown suffix | `UnsupportedFormatError` (exit 2 through the CLI) |
| Empty file, whitespace only, PDF without a text layer | `EmptyDocumentError` |
| Encrypted / corrupt PDF | `IngestError` naming the file |
| Markdown header is not valid YAML | line-based fallback; the document is kept |
| Document with no chunkable blocks | `EmptyDocumentError` from the chunker |
| Corpus that violates §13.4 | `CorpusBuildError` naming the first findings |
| Reference reply not entailed by its document (§7.2) | `CorpusBuildError` with the first ungrounded claim |
| Missing manifest / invalid JSON | `CorpusBuildError` from `load_manifest` |
| Dataset without a license record | `LicenseError` naming the license page |
| Chunking config with `overlap ≥ budget` | `ValueError` at construction |

---

## 12. Limitations (honest reporting, §10.7)

1. **Template reuse.** The synthetic corpus is generated from ~56 sentence templates per
   locale with slot variation. It is realistic in *shape* (headings, tables, procedures,
   FAQ pairs, numbers, dates, identifiers, typos, code-switching) but not in *diversity*:
   lexical overlap between articles is higher than in a real knowledge base, which flatters
   retrieval.
2. **Synthetic, LLM-assisted content without native-speaker validation.** The six
   locale banks were drafted by an LLM-assisted authoring pass (see
   `content_provenance` in the manifest) and reviewed by no native speaker. Script,
   terminology coverage, formats, quotas and grounding are tested (§7.5); *fluency,
   register and idiom* are not. Treat every locale as a synthetic demo locale, not as
   validated production copy.
3. **Synthetic references are grounded, not human-written.** Since generator `1.1.0` every
   reference reply is a verbatim FAQ answer of its own document (§7.2), so it is
   *verifiable* — but it is still machine-composed from templates, not the answer a human
   agent would write. Grounded ≠ correct-by-a-human: the §10.7 honest-reporting policy
   (synthetic-data optimism, no human verification of answers) still applies, and t7/t11
   must keep labelling these references as synthetic.
4. **Archetype depth is not document diversity.** Twelve archetypes per locale give
   the corpus real structural variety, but articles of one archetype are still
   composed from the same sentence pool with different slot values and section
   choices. Lexical overlap inside an archetype stays higher than in a real knowledge
   base.
5. **Token counts are estimates** (§3), never provider-exact.
6. **PDF heading recovery is heuristic.** Font-size ranking works for text PDFs with a
   consistent type hierarchy; a PDF that uses bold body text as “headings” will be read as
   flat paragraphs. Scanned PDFs are refused, not OCR'd (v1.0 has no OCR).
7. **Deduplication is lexical.** Two articles that say the same thing in different words
   are not duplicates and are kept — correct for a knowledge base, but it means the corpus
   can contain semantic near-duplicates.
8. **No source-connector ingest.** v1.0 ingests files and the one documented dataset
   adapter (NG: connectors are out of scope).

---

## 13. Traceability

| Acceptance test | Proof in this repository |
|---|---|
| AT-ING-01 | `tests/unit/ingest/test_loaders.py` (5 formats, heading share, encodings, failures) |
| AT-ING-02 | `tests/unit/ingest/test_chunker.py` (table/procedure/FAQ atomicity, header repetition) |
| AT-ING-03 | `tests/unit/ingest/test_chunker.py` + `tests/integration/ingest/test_demo_corpus.py` (budget over the corpus) |
| AT-ING-04 | `test_chunker.py` (heading path/breadcrumb on every chunk, span invariant) |
| AT-ING-05 | `tests/unit/ingest/test_canonical.py`, `test_chunker.py` (sidecar, text untouched) |
| AT-ING-06 | `tests/unit/ingest/test_dedup.py`, `tests/integration/ingest/test_demo_corpus.py` |
| AT-ING-07 | `tests/integration/ingest/test_demo_corpus.py`, `test_cli_ingest.py` (floors) |
| AT-ING-08 | `test_demo_corpus.py` (byte-identical rebuild, idempotency, checked-in manifest) |
| AT-ING-09 | `tests/unit/ingest/test_datasets.py` (license gate) |
| AT-ING-10 | `tests/unit/ingest/test_synthetic.py` (locales, archetypes, perturbations, reserved ids), `tests/unit/ingest/test_reference_grounding.py` (one slot draw, grounded references, 2,592/2,592 queries, multi-hop unions, unanswerable items, verifier negative cases, chunk-id scheme) and `tests/unit/ingest/test_locale_quality.py` (script, terminology, formats, depth ≥100 docs/≥10 archetypes, ≥400 queries/locale, quotas, determinism, provenance) |
