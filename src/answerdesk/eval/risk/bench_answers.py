"""Persist full answer artefacts for the bench turns (risk-control v3, R-06).

The t11 benchmark stored only a 60-character answer head, which made two decision-time
features (claim support, numeric consistency) impossible to compute on the product population.
This module regenerates the bench dev and test answers with the shipped pipeline and the
cross-family judge, and writes one row per turn with everything the features need:

``case_id, split, locale, bucket, question, answer, claims[], citations[], cited_doc_ids[],
retrieved_doc_ids[], decision, reason_code, rubric, judge_model, generator_model, latency_ms,
input_tokens, output_tokens``.

The run is resumable (a checkpoint per split) and bounded: one generation and one judge call per
turn, no sampling beyond what the pipeline itself does. Retrieval is cached on disk, so a resumed
run pays only for the turns it has not answered yet.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from answerdesk.eval.corpus import load_corpus
from answerdesk.eval.live import build_embedder, build_index, build_reranker, retrieval_variants
from answerdesk.eval.risk.features import (
    citation_coverage,
    claim_support_ratio,
    fidelity,
    load_documents,
)

DEFAULT_OUT = Path("reports/risk-v3")
BENCH_DEV = Path("reports/bench/_state/dev-turns.csv")
BENCH_TEST = Path("reports/bench/answer/primary/turns.csv")


def _turns(path: Path, split: str) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = split
    return rows


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _append(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _judge(settings: Any) -> tuple[Any, str]:
    from answerdesk.eval.judge import RubricJudge
    from answerdesk.eval.live import judge_provider

    provider = judge_provider(settings)
    assert provider is not None
    model = settings.eval.judge_model or "judge"
    return RubricJudge(provider, judge_model=model), model


def _provider(settings: Any) -> tuple[Any, str]:
    from answerdesk.llm.factory import build_provider

    return build_provider(settings), settings.resolved_provider().model


def persist(
    *,
    out_dir: Path = DEFAULT_OUT,
    split: str,
    limit: int | None = None,
    settings: Any | None = None,
    corpus_dir: Path = Path("data/corpus"),
    context_k: int | None = None,
) -> dict[str, Any]:
    """Answer and judge one split of bench turns, checkpointing after every turn."""
    from answerdesk.config import load_settings
    from answerdesk.generate.pipeline import AnswerPipeline

    settings = settings or load_settings(profile=None)
    source = BENCH_DEV if split == "dev" else BENCH_TEST
    turns = _turns(source, split)
    if limit is not None:
        turns = turns[:limit]
    checkpoint = out_dir / f"bench-answers-{split}.jsonl"
    existing = _read(checkpoint)
    # a provider error is not an answer: drop those rows so a rerun retries them
    clean = [row for row in existing if not row.get("provider_error")]
    if len(clean) != len(existing):
        checkpoint.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean), encoding="utf-8"
        )
    done = {row["case_id"] for row in clean}
    pending = [row for row in turns if row["case_id"] not in done]
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    embedder = build_embedder(settings, Path("data/index"))
    index = build_index(settings, corpus, embedder, Path("data/index/bench"))
    variants = retrieval_variants(settings, index, reranker=build_reranker(settings))
    retriever = variants.get("cc@0.75") or next(iter(variants.values()))
    provider, generator_model = _provider(settings)
    pipeline = AnswerPipeline.from_settings(provider, settings)
    judge, judge_model = _judge(settings)
    documents = load_documents(corpus_dir)
    k = context_k or settings.generate.context_k_max
    produced: list[dict[str, Any]] = []
    for row in pending:
        question = row.get("question") or ""
        started = time.perf_counter()
        trace = retriever.retrieve(question, final_k=k)
        packed = list(getattr(trace, "packed", None) or trace.reranked or trace.fused or [])
        chunk_ids = [passage.chunk.chunk_id for passage in packed]
        try:
            result = pipeline.answer(question, packed)
        except Exception as exc:  # a provider failure is recorded, never invented
            produced.append(
                {
                    "case_id": row["case_id"],
                    "split": split,
                    "locale": row.get("locale", ""),
                    "bucket": row.get("category", "unknown"),
                    "question": question,
                    "answer": "",
                    "claims": [],
                    "citations": [],
                    "cited_doc_ids": [],
                    "retrieved_doc_ids": [],
                    "decision": "provider_error",
                    "reason_code": type(exc).__name__,
                    "rubric": None,
                    "generator_model": generator_model,
                    "judge_model": judge_model,
                    "provider_error": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
            )
            _append(checkpoint, produced[-1:])
            continue
        claims = []
        citations: list[str] = []
        for claim in result.claims:
            claim_citations = []
            for citation in claim.citation_ids:
                if isinstance(citation, int) and 0 <= citation < len(chunk_ids):
                    claim_citations.append(chunk_ids[citation])
                elif isinstance(citation, str) and citation in chunk_ids:
                    claim_citations.append(citation)
            citations.extend(claim_citations)
            claims.append({"text": str(claim.text), "citations": claim_citations})
        answer_text = " ".join(str(claim["text"]) for claim in claims)
        cited_docs = sorted({chunk_id.split("::", 1)[0] for chunk_id in citations})
        cited_text = "\n".join(documents.get(doc_id, "") for doc_id in cited_docs)
        rubric = None
        try:
            score = judge.grade(
                question=question,
                reference_answer=None,
                candidate_answer=answer_text,
                context=[cited_text[:4000]] if cited_text else [],
            )
            rubric = score.score
        except Exception:
            rubric = None
        produced.append(
            {
                "case_id": row["case_id"],
                "split": split,
                "locale": row.get("locale", ""),
                "bucket": row.get("category", "unknown"),
                "question": question,
                "answer": answer_text,
                "claims": claims,
                "citations": sorted(set(citations)),
                "cited_doc_ids": cited_docs,
                "retrieved_doc_ids": sorted({passage.chunk.doc_id for passage in packed}),
                "decision": result.signal.action.value,
                "reason_code": result.signal.reason_code.value,
                "rubric": rubric,
                "claim_support": round(claim_support_ratio(answer_text, cited_text), 6),
                "numeric_consistency": round(
                    float(str(fidelity(answer_text, cited_text)["fidelity"])), 6
                ),
                "citation_coverage": round(citation_coverage(cited_docs), 6),
                "generator_model": generator_model,
                "judge_model": judge_model,
                "provider_error": False,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        _append(checkpoint, produced[-1:])
    rows = _read(checkpoint)
    labelled = [row for row in rows if row.get("rubric") is not None]
    return {
        "split": split,
        "turns": len(turns),
        "persisted": len(rows),
        "labelled": len(labelled),
        "correct": sum(1 for row in labelled if float(row["rubric"]) >= 4),
        "provider_errors": sum(1 for row in rows if row.get("provider_error")),
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "generator_model": generator_model,
        "judge_model": judge_model,
    }


def rejudge_references(
    *,
    out_dir: Path = DEFAULT_OUT,
    split: str,
    settings: Any | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Re-grade persisted answers against the golden reference answer when one exists.

    The first judging pass had no reference (the persisted row carries only the question), so
    it scored context support. The golden split carries reference answers for most cases; using
    them makes the label a fact-correctness judgment, which is what the selective-answering
    policy is supposed to control.
    """
    from answerdesk.config import load_settings

    settings = settings or load_settings(profile=None)
    golden = Path("evals") / "golden" / "v1" / f"{split}.jsonl"
    references: dict[str, str] = {}
    if golden.is_file():
        for line in golden.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            reference = record.get("reference_answer") or record.get("metadata", {}).get(
                "reference_answer"
            )
            if reference and record.get("answerable"):
                references[str(record["case_id"])] = str(reference)
    checkpoint = out_dir / f"bench-answers-{split}.jsonl"
    rows = _read(checkpoint)
    judge, judge_model = _judge(settings)
    updated = 0
    for row in rows:
        if row.get("provider_error") or not row.get("answer"):
            continue
        reference = references.get(str(row["case_id"]))
        if not reference:
            continue
        if limit is not None and updated >= limit:
            break
        try:
            score = judge.grade(
                question=str(row.get("question") or ""),
                reference_answer=reference,
                candidate_answer=str(row.get("answer") or ""),
                context=[],
            )
            row["rubric"] = score.score
            row["label_source"] = "reference-answer judge"
            row["judge_model"] = judge_model
            updated += 1
        except Exception:
            continue
    checkpoint.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    labelled = [
        row for row in rows if row.get("rubric") is not None and not row.get("provider_error")
    ]
    return {
        "split": split,
        "with_reference": len(references),
        "rejudged": updated,
        "labelled": len(labelled),
        "correct": sum(1 for row in labelled if float(row["rubric"]) >= 4),
    }


__all__ = ["BENCH_DEV", "BENCH_TEST", "DEFAULT_OUT", "persist", "rejudge_references"]
