# Scale run — operating-point

Command: `python.exe scripts/scale/measure.py --repeat 3`

## Verdict

**NFR-SCALE-01: MEASURED** — failed checks: none

| Check | Target | Measured | Pass |
|---|---|---|---|
| Documents | >= 3,000 | 3,000 | True |
| Chunks | >= 20,000 | 26,068 | True |
| Index build | <= 600 s | 7.7 s | True |
| Retrieval p95 | <= 500 ms | 11.8 ms | True |
| Peak RSS | <= 4.0 GB | 0.487 | True |

## Hardware

- Windows-11-10.0.26200-SP0 (AMD64, 28 CPUs)
- Python 3.12.10 (CPython)

## Configuration

- documents per locale: 500, locales: ru, tg, uz, ky, tr, zh, seed 20261020
- embedder: `hashing-64` (dim 64), fuser `rrf`, final_k 8
- chunking: {"chunker": "structure-first-v1", "token_budget": 320, "overlap_tokens": 48, "context_header": false, "repeat_table_header": true}

## Corpus

- corpus_version `corpus-651ca3bca4fbd48d` (schema 2.1.0), 3,000 documents, 26,068 chunks, 1,956,810 tokens, 12,000 queries, license CC0-1.0
- grounding findings: 0, identifier findings: 0
- path: `data\scale\corpus` (git-ignored)

## Numbers

- ingest: 30.5 s
- index build: 7.7 s (reported: {"bm25_build_ms": 5131.061, "dense_build_ms": 1647.868, "total_ms": 7271.474} ms)
- index size: 53.3 MiB (55,913,339 bytes)
- retrieval (240 queries): p50 2.8 ms, p95 11.8 ms, mean 3.6 ms, max 12.7 ms
- stage means: {"analysis_ms": 0.089, "dedupe_ms": 0.517, "dense_ms": 1.377, "fusion_ms": 0.217, "sparse_ms": 1.299, "total_ms": 3.499}
- peak RSS: 0.487 GB

## Spread over 3 runs

| Metric | min | median | max |
|---|---|---|---|
| ingest (s) | 30.009 | 30.453 | 30.477 |
| index build (s) | 7.633 | 7.701 | 7.768 |
| retrieval p50 (ms) | 2.759 | 3.055 | 3.059 |
| retrieval p95 (ms) | 11.846 | 12.044 | 12.160 |
| peak RSS (GB) | 0.479 | 0.486 | 0.487 |
