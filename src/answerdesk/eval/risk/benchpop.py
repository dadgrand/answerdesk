"""The bench answer population: the product's own frozen test split (t11 turns).

This is the primary risk-control population. It has 200 dev and 200 test turns with
cross-family judge labels, so the split is the one the benchmark already froze — no fact can
leak across it. Retrieval is recomputed locally (BM25 over the frozen corpus, no network);
the cross-encoder feature needs one reranker call per turn.

Two decision-time features are unavailable here because t11 persisted only a 60-character
answer head: numeric consistency and the deterministic claim-support ratio. They are recorded
as ``unavailable`` and excluded from this population's model — the consistency population
still exercises them. That is stated in the artefact rather than papered over.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from answerdesk.eval.risk.features import (
    FeatureRow,
    citation_coverage,
    load_documents,
    overlap_score,
)

BENCH_TEST = Path("reports/bench/answer/primary/turns.csv")
BENCH_DEV = Path("reports/bench/_state/dev-turns.csv")
UNAVAILABLE = ("claim_support", "numeric_consistency")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(row: Mapping[str, str]) -> int:
    return 1 if row.get("answer_ok") == "True" else 0


def _bucket(row: Mapping[str, str]) -> str:
    return str(row.get("category") or row.get("corpus_category") or "unknown")


def load_bench_rows(
    *,
    test_path: Path = BENCH_TEST,
    dev_path: Path = BENCH_DEV,
    corpus: Path = Path("data/corpus"),
    reranker: Any | None = None,
    limit: int | None = None,
) -> list[FeatureRow]:
    """Feature rows for both frozen splits; ``split`` comes from the source files."""
    from answerdesk.contracts import Channel, Chunk, RetrievedPassage

    documents = load_documents(corpus)
    passages = list(documents.items())[:200]  # bounded pool so one index build stays cheap
    texts_by_doc = [text for _doc_id, text in passages]
    import bm25s

    tokenized = bm25s.tokenize(texts_by_doc, stopwords=None, show_progress=False)
    index = bm25s.BM25()
    index.index(tokenized, show_progress=False)

    def score_question(question: str) -> list[float]:
        _results, scores = index.retrieve(
            bm25s.tokenize([question], stopwords=None, show_progress=False),
            k=min(len(texts_by_doc), 20),
            show_progress=False,
        )
        return sorted((float(score) for score in scores[0]), reverse=True)

    rows: list[FeatureRow] = []
    for path, split in ((dev_path, "dev"), (test_path, "test")):
        if not path.is_file():
            continue
        records = _rows(path)
        if limit is not None:
            records = records[:limit]
        for record in records:
            question = record.get("question") or ""
            lexical = score_question(question)
            top = lexical[0] if lexical else 0.0
            second = lexical[1] if len(lexical) > 1 else 0.0
            margin = (top - second) / (abs(top) + 1e-6) if top else 0.0
            cited = [chunk for chunk in (record.get("cited") or "").split(";") if chunk]
            rerank_top = 0.0
            if reranker is not None and lexical:
                # rerank the heads of the BM25 ranking, capped so the call stays bounded
                heads = [(doc_id, text) for doc_id, text in passages[:8]]
                probe = [
                    RetrievedPassage(
                        chunk=Chunk(
                            chunk_id=f"{doc_id}::probe",
                            doc_id=doc_id,
                            text=text[:1200],
                            heading_path=[],
                            breadcrumb="",
                            start_char=0,
                            end_char=min(1200, len(text)),
                            token_count=min(200, len(text.split())),
                        ),
                        channel=Channel.BM25,
                        score=float(lexical[index] if index < len(lexical) else 0.0),
                        rank=index + 1,
                    )
                    for index, (doc_id, text) in enumerate(heads)
                ]
                ranked = reranker.rerank(question, probe, top_n=1)
                if ranked:
                    raw = float(getattr(ranked[0], "score", 0.0))
                    import math

                    rerank_top = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))
            claims = max(1.0, float(record.get("claims") or 1))
            features = {
                "rerank_top": rerank_top,
                "retrieval_margin": max(0.0, min(1.0, margin)),
                "citation_coverage": citation_coverage(cited),
                "overlap_score": overlap_score(question, texts_by_doc[0]) if texts_by_doc else 0.0,
                "bm25_top": top,
                "bm25_margin": margin,
                "bm25_above_gate": sum(1 for score in lexical if score > 0.0)
                / max(1, len(lexical)),
                "citation_density": len(cited) / claims,
                "log_claims": __import__("math").log1p(claims),
            }
            rows.append(
                FeatureRow(
                    sample_id=str(record.get("case_id") or f"{split}:{len(rows)}"),
                    fact_id=str(record.get("case_id") or ""),
                    locale=str(record.get("locale") or ""),
                    bucket=_bucket(record),
                    split=split,
                    features=features,
                    label=_label(record),
                )
            )
    return rows


def describe(rows: Sequence[FeatureRow]) -> dict[str, Any]:
    dev = [row for row in rows if row.split == "dev"]
    test = [row for row in rows if row.split == "test"]
    return {
        "population": "bench",
        "source": {
            "test": str(BENCH_TEST).replace("\\", "/"),
            "dev": str(BENCH_DEV).replace("\\", "/"),
        },
        "n": len(rows),
        "dev": len(dev),
        "test": len(test),
        "base_error_dev": round(1.0 - sum(row.label for row in dev) / max(1, len(dev)), 6),
        "base_error_test": round(1.0 - sum(row.label for row in test) / max(1, len(test)), 6),
        "unavailable_features": list(UNAVAILABLE),
        "note": (
            "numeric consistency and deterministic claim support need the full answer text, "
            "which t11 did not persist (60-character head only); they are excluded here and "
            "exercised on the consistency population instead"
        ),
    }


__all__ = ["BENCH_DEV", "BENCH_TEST", "UNAVAILABLE", "describe", "load_bench_rows"]
