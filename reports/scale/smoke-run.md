# Scale run — smoke

Command: `python.exe scripts/scale/measure.py --smoke`

## Verdict

**NFR-SCALE-01: MISS** — failed checks: documents, chunks

| Check | Target | Measured | Pass |
|---|---|---|---|
| Documents | >= 3,000 | 72 | False |
| Chunks | >= 20,000 | 684 | False |
| Index build | <= 600 s | 0.2 s | True |
| Retrieval p95 | <= 500 ms | 2.2 ms | True |
| Peak RSS | <= 4.0 GB | 0.072 | True |

## Hardware

- Windows-11-10.0.26200-SP0 (AMD64, 28 CPUs)
- Python 3.12.10 (CPython)

## Configuration

- documents per locale: 12, locales: ru, tg, uz, ky, tr, zh, seed 20261020
- embedder: `hashing-64` (dim 64), fuser `rrf`, final_k 8
- chunking: {"chunker": "structure-first-v1", "token_budget": 320, "overlap_tokens": 48, "context_header": false, "repeat_table_header": true}

## Corpus

- corpus_version `corpus-c3d71a8626532ddd` (schema 2.1.0), 72 documents, 684 chunks, 50,065 tokens, 288 queries, license CC0-1.0
- grounding findings: 0, identifier findings: 0
- path: `data\scale\corpus` (git-ignored)

## Numbers

- ingest: 0.8 s
- index build: 0.2 s (reported: {"bm25_build_ms": 144.61, "dense_build_ms": 42.0, "total_ms": 208.863} ms)
- index size: 1.4 MiB (1,447,530 bytes)
- retrieval (24 queries): p50 1.7 ms, p95 2.2 ms, mean 1.9 ms, max 8.3 ms
- stage means: {"analysis_ms": 0.536, "dedupe_ms": 0.682, "dense_ms": 0.157, "fusion_ms": 0.214, "sparse_ms": 0.307, "total_ms": 1.895}
- peak RSS: 0.072 GB

