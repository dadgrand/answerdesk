# `belebele/` — Belebele-derived suites (CC BY-SA 4.0)

| Path | What |
|---|---|
| `retrieval/<lang>/corpus.jsonl` | the passage pool of one language: `passage_id`, `text`, `source_link`, `question_numbers`, provenance |
| `retrieval/<lang>/queries.jsonl` | 150 retrieval items per language (`query`, `qrels`, `split`) |
| `qa/<lang>.jsonl` | 150 multiple-choice items per language (`question`, `options`, `answer_index`, gold passage) |

Read `ATTRIBUTION.md` before reusing anything here: this directory is **CC BY-SA 4.0**,
not MIT, and it carries a share-alike obligation.

## How to read the retrieval suite

The `qrels` of an item are derived, not annotated: the gold passage is the passage the
comprehension question was written against. Belebele questions are phrased to be answered
*with the passage supplied* ("according to the given text…"), so they are deictic and less
specific than customer queries. Use this suite as a multilingual retrieval diagnostic, not
as a MIRACL-comparable benchmark. The dataset card (§8, limitation 1) says the same.

## Loading

```python
import json
from pathlib import Path

suite = Path("data/bench/answerdesk-bench/v0.1/belebele/retrieval/ru")
corpus = {row["passage_id"]: row["text"] for row in map(json.loads, suite.joinpath("corpus.jsonl").open())}
queries = [json.loads(line) for line in suite.joinpath("queries.jsonl").open()]
qrels = {item["item_id"]: item["qrels"] for item in queries}
```

`docs/benchmark.md` §9 documents the exact adapter contract for the evaluation harness (t19).
