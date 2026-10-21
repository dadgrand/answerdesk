# Scale evidence — NFR-SCALE-01 (operating point)

Status: **MEASURED**. The operating point of the reference stack — **≥3,000 documents and
≥20,000 chunks** — was built from the same seeded generator that produces the demo corpus,
indexed, and measured end to end on the reference machine. All five NFR-SCALE-01 checks
pass with headroom.

| Check | Target (NFR-SCALE-01) | Measured | Pass |
|---|---|---|---|
| Documents | ≥ 3,000 | **3,000** | yes |
| Chunks | ≥ 20,000 | **26,068** | yes |
| Index build | ≤ 10 min | **7.6 – 7.8 s** | yes |
| Retrieval p95 | ≤ 500 ms | **11.8 – 12.2 ms** | yes |
| Peak RSS | ≤ 4 GB | **0.48 – 0.49 GB** | yes |

Measured numbers are the spread over three consecutive full runs of
`uv run python scripts/scale/measure.py --repeat 3`. The committed artifacts are
[`reports/scale/last-run.json`](../reports/scale/last-run.json) (machine-readable, includes
the exact command line, the full run list and the spread block) and
[`reports/scale/last-run.md`](../reports/scale/last-run.md) (the same run as a table).
The CI-sized probe is committed as
[`reports/scale/smoke-run.json`](../reports/scale/smoke-run.json): 72 docs / 684 chunks,
proving the pipeline shape in seconds without claiming the operating point.

## 1. How to reproduce

```bash
cd answerdesk

# CI-sized probe (~1 s, no network, no API key)
uv run python scripts/scale/measure.py --smoke

# operating point: 500 docs/locale × 6 locales = 3,000 documents
uv run python scripts/scale/measure.py --repeat 3

# larger / smaller points, different embedder backend
uv run python scripts/scale/measure.py --docs-per-locale 600 --queries 400 --repeat 2
uv run python scripts/scale/measure.py --embedder bge_m3   # needs the `local` extra
```

The script prints a one-line-per-phase summary and writes
`reports/scale/last-run.json` + `reports/scale/last-run.md` (or `smoke-run.*` in
`--smoke` mode). `--no-write` leaves the committed artifacts untouched. The operating point
exits non-zero if any NFR-SCALE-01 check fails, so a regression cannot pass silently.

## 2. What is measured

| Phase | What the number contains |
|---|---|
| **Ingest** | `build_demo` end to end: seeded generation of all six locales, the grounding gate (`verify_reference_grounding` over every query), the identifier/redistribution scan, deduplication, structure-first chunking, and writing the Markdown documents plus `chunks.jsonl`, `queries/queries.jsonl` and `manifest.json`. Wall clock, single process. |
| **Index build** | `build_indexes` from the corpus on disk: BM25 (inverted index over the multilingual tokenizer) **and** the dense index from the configured embedder, plus the chunk store and the index manifest. Warm page cache, cold in-process state, `force=True` (no reuse of a previous build). |
| **Retrieval** | 240 deterministically sampled corpus queries through `RetrievalPipeline.retrieve_with_timings` at `final_k=8`: query analysis, sparse channel, dense channel, fusion, dedupe. Percentiles are nearest-rank over the per-query wall clock. |
| **Index size** | Sum of file sizes of the index directory (`data/scale/index/<corpus_version>`), i.e. what actually has to be shipped or loaded from disk. |
| **Peak RSS** | Peak working set of the measuring process (`Psapi.GetProcessMemoryInfo` / `VmHWM`), covering ingest + index build + retrieval in one process. |

The run uses the corpus the project ships, not a toy copy: the scale corpus carries
`grounding findings: 0` and `identifier findings: 0` in its manifest, and it is produced by
the same `SyntheticConfig` code path, so a gate regression fails the scale run exactly as it
fails `answerdesk ingest --demo`.

## 3. Configuration of the recorded run

- Corpus: `corpus-651ca3bca4fbd48d` — 3,000 documents, 26,068 chunks, 6 locales
  (ru, tg, uz, ky, tr, zh × 500 documents), seed `20261020`, schema `2.1.0`,
  generator `2.1.0`, license CC0-1.0.
- Chunking: the demo profile (`320` tokens / `48` overlap), identical to the committed
  corpus so chunk counts are comparable per document.
- Index: BM25 (`k1=1.2`, `b=0.75`) + dense index over the **hashing** embedder
  (`dimension=64`), fuser from the offline profile, `final_k=8`.
- Storage: `data/scale/corpus` and `data/scale/index/...` — a clearly separated,
  git-ignored tree (`data/scale/.gitignore` ignores everything but itself). The demo corpus
  in `data/corpus` and its freeze hash are never touched by a scale run.

## 4. Hardware profile

| | |
|---|---|
| OS | Windows 11 (`10.0.26200`), x86-64 |
| CPUs | 28 logical cores |
| Python | CPython 3.12 |
| Storage | local SSD |
| Network | none used during the run (offline profile) |

The exact profile of the recorded run is embedded in `reports/scale/last-run.json` under
`hardware`, so a number can never be separated from the machine that produced it
(NFR-REP-01, R-13).

## 5. Numbers, with spread

Three consecutive full runs (`--repeat 3`), same machine, same seed:

| Metric | min | median | max |
|---|---|---|---|
| Ingest | 30.0 s | **30.5 s** | 30.5 s |
| Index build | 7.63 s | **7.70 s** | 7.77 s |
| Retrieval p50 | 2.76 ms | **3.06 ms** | 3.06 ms |
| Retrieval p95 | 11.85 ms | **12.04 ms** | 12.16 ms |
| Peak RSS | 0.479 GB | **0.486 GB** | 0.487 GB |

Index size is deterministic: **53.3 MiB** (55,913,339 bytes) on every run — it depends only
on the corpus and the embedder, which is why the spread table omits it.

Retrieval stage means over the 240 timed queries: sparse **1.8 ms**, dense **1.4 ms**,
fusion **0.23 ms**, dedupe **0.54 ms**, query analysis **0.09 ms**; the pipeline total is
≈4.0 ms mean and 11.8–12.2 ms p95. The observed max (~118 ms on one run) is a single cold
outlier — the first query of a process, which pays the import/allocation cost — and is
reported rather than trimmed.

## 6. Honest limitations

1. **The embedder is the offline hashing embedder, not bge-m3.** The recorded run measures
   the *pipeline* at the operating point (BM25 + a 64-dimensional dense arm), which is what
   CI and the offline profile use, and which keeps the run free of downloads and API keys.
   A real sentence-transformers model (`--embedder bge_m3`, `local` extra) is expected to
   dominate the index build: 26k chunks × bge-m3 on CPU is minutes, not seconds, and the
   10-minute budget in NFR-SCALE-01 was written with that path in mind. This run therefore
   does **not** prove the local-model index build stays under 10 minutes; it proves the
   ingestion, the BM25 build, the dense plumbing and the serving latency at scale. The
   benchmark that pins the real-model cost is the `local`-extra release run, not this one.
2. **Synthetic corpus scaling.** 3,000 documents come from the same generator and template
   pools as the 810-document demo corpus: more slot draws, the same vocabulary and
   structure. Vocabulary growth (a real 3k-doc knowledge base has far more distinct terms,
   longer postings lists and heavier BM25 memory) is therefore underestimated. Treat the
   index size and the sparse latency as a floor, not a ceiling.
3. **A single machine, a single process.** No concurrency, no sharding, no warm service
   cache. The retrieval numbers are single-threaded end-to-end pipeline calls, which is the
   comparable number for NFR-LAT-02 but not a service throughput measurement (that one
   lives in the service load run).
4. **Peak RSS is process-wide.** It includes the generator, the corpus in memory, both
   indexes and the Python interpreter, measured with the OS working-set API. It is not a
   breakdown per component, and shared pages may be counted once per process.
5. **Windows numbers.** The numbers were produced on Windows 11; Linux CI will differ
   (different allocator, different filesystem). The script is platform-portable and records
   the hardware profile on every run so a second machine can be compared honestly.

## 7. What this evidence does and does not claim

- It claims: at 3,000 documents / 26,068 chunks, the documented ingest command produces a
  gated corpus in ≈30 s, the index builds in ≈8 s and 53 MiB, and retrieval stays two
  orders of magnitude under the 500 ms p95 budget with 0.49 GB peak RSS.
- It does not claim: that bge-m3 or an API embedder keeps the same index-build time, that a
  production knowledge base has the same vocabulary statistics, or that a multi-tenant
  service sustains the same latency under load.

NFR-SCALE-01 is therefore marked **measured** for the offline reference configuration, with
the local-model index build explicitly deferred to the `local`-extra release run
(limitation 1). The demo corpus remains the fast CI artifact; the scale corpus is
regenerated on demand and never committed.
