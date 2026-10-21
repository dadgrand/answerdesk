# Licence map — `answerdesk-bench` v0.1

The repository is MIT; **the data is not uniformly MIT**. This directory holds the licence
text that applies to each part of the benchmark.

| Path | Licence | Full text |
|---|---|---|
| `../belebele/` | CC BY-SA 4.0 (attribution + share-alike) | `../belebele/LICENSE-CC-BY-SA-4.0.txt` |
| `../mrtydi/` | Apache-2.0 | `../mrtydi/LICENSE-APACHE-2.0.txt` |
| `../domain/` | CC0-1.0 (public domain dedication) | `../domain/LICENSE-CC0-1.0.txt` |
| `schema.json`, `sources.lock.json`, `scripts/build_bench.py`, `scripts/bench/`, `tests/unit/bench/` | MIT | `MIT.txt` and the repository `LICENSE` |
| `DATASET_CARD.md`, `README.md` files | MIT (documentation) | `MIT.txt` |

## What this means in practice

* Reusing `belebele/` creates a derivative work of Belebele; it must stay CC BY-SA 4.0
  (attribution and share-alike notice: `../belebele/ATTRIBUTION.md`).
* Reusing `mrtydi/` requires the Apache-2.0 attribution and notice; its passages are
  Wikipedia text redistributed under Mr. TyDi's declared Apache-2.0 licence — see the
  residual-risk note in `../mrtydi/ATTRIBUTION.md`.
* Reusing `domain/` requires nothing (CC0-1.0), though attribution is appreciated.
* The code that builds and checks the benchmark is MIT and can be reused independently of
  the data.

`scripts/build_bench.py --verify` checks that these files exist, that each contains the
expected licence marker, and that the dataset card does not claim the data is MIT.
