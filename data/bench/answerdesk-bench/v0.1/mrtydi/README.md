# `mrtydi/` — Mr. TyDi ru, native qrels (Apache-2.0)

| Path | What |
|---|---|
| `ru/corpus.jsonl` | 2,428 passages (`passage_id` = `mrtydi-ru:<docid>`, `text`, `title`) |
| `ru/queries.jsonl` | 2,369 queries (native dev 1,374 / test 995) with `qrels` |

The pool is the union of the positive passages of the dev and test queries: every passage
that is relevant to a query in this suite is present, and every passage in the pool is a
judged positive of some query. It is small (≈2.4 k passages) and should be read as a
compact native-qrels sanity suite for Russian, not as a replacement for MIRACL-ru.

Licence: **Apache-2.0** — see `ATTRIBUTION.md` for attribution and the Wikipedia-text note.
The repository's MIT licence applies to code only.

## Loading

```python
import gzip, json
from pathlib import Path

root = Path("data/bench/answerdesk-bench/v0.1/mrtydi/ru")
corpus = {row["passage_id"]: row["text"] for row in map(json.loads, root.joinpath("corpus.jsonl").open())}
queries = [json.loads(line) for line in root.joinpath("queries.jsonl").open()]
test = [item for item in queries if item["split"] == "test"]
```

`docs/benchmark.md` §9 documents the adapter contract for the evaluation harness (t19).
