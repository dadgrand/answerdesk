# `data/bench` — curated external benchmarks

Generated artefacts for benchmarks that anchor the repository's quality claims in data the
project did not generate itself. Everything here is built by a script in `scripts/` and
checked by `scripts/*.py --verify`; nothing is hand-edited.

| Directory | Benchmark | Built by | Docs |
|---|---|---|---|
| `answerdesk-bench/v0.1/` | `answerdesk-bench` v0.1 — Belebele + Mr. TyDi + synthetic domain slice | [`scripts/build_bench.py`](../../scripts/build_bench.py) | [`docs/benchmark.md`](../../docs/benchmark.md), [`DATASET_CARD.md`](answerdesk-bench/v0.1/DATASET_CARD.md) |

## Licence warning

The repository is MIT; the benchmark data is not. `answerdesk-bench/v0.1/belebele/` is
CC BY-SA 4.0 (share-alike), `answerdesk-bench/v0.1/mrtydi/` is Apache-2.0, and
`answerdesk-bench/v0.1/domain/` is CC0-1.0. See
[`LICENSES/README.md`](answerdesk-bench/v0.1/LICENSES/README.md).

## Build cache

Upstream archives (including a 1.6 GB Mr. TyDi corpus) are downloaded to
`.cache/answerdesk-bench/`, which is git-ignored. `--build` and `--verify` never use the
network; only `--fetch` does.
