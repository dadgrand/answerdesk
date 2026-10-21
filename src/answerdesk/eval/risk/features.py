"""Decision-time answerability features for the risk-controlled policy (R-05 / t47).

Every feature here is computable **before** the answer is shown to a user and without any
label: the cross-encoder relevance of the best retrieved passage, the lexical retrieval margin,
the verifier's deterministic claim-support ratio, citation coverage, and the numeric/identifier
consistency of the answer against the passages it cites. Labels (cross-family judge) are only
used later, to fit and certify the threshold.

The self-consistency feature is dev-only by design: it needs k sampled drafts, so it is measured
on the dev split to quantify its value and is *not* available at test time (the deployable
feature set excludes it; the ablation reports the difference).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.eval.consistency.values import extract, fidelity

#: features available at decision time on the frozen test split
DECISION_FEATURES = (
    "rerank_top",
    "retrieval_margin",
    "claim_support",
    "citation_coverage",
    "numeric_consistency",
)
#: the six features of the t11 logistic, refit here as the comparison arm
LEGACY_FEATURES = (
    "overlap_score",
    "bm25_top",
    "bm25_margin",
    "bm25_above_gate",
    "citation_density",
    "log_claims",
)
#: dev-only feature (k=3 drafts); never used in the test-time model
SELF_CONSISTENCY = "self_consistency"

_TERM = re.compile(r"[\w’'\-]{2,}", re.UNICODE)


@dataclass
class FeatureRow:
    sample_id: str
    fact_id: str
    locale: str
    bucket: str
    split: str
    features: dict[str, float]
    label: int
    answer: str = ""
    cited: tuple[str, ...] = ()
    gold: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "fact_id": self.fact_id,
            "locale": self.locale,
            "bucket": self.bucket,
            "split": self.split,
            **{name: round(value, 6) for name, value in self.features.items()},
            "label": self.label,
        }


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TERM.finditer(text)}


def overlap_score(question: str, passage_text: str) -> float:
    """The t11 signal: share of question terms contained in the passage (saturates at 1.0)."""
    question_terms = _terms(question)
    if not question_terms:
        return 0.0
    return len(question_terms & _terms(passage_text)) / len(question_terms)


def bm25_scores(question: str, passages: Sequence[str]) -> list[float]:
    """Lexical scores for one question over a fixed passage list (deterministic, offline).

    Returns an empty list when the pool has no usable tokens: bm25s raises on an empty corpus,
    and a degenerate pool is a feature-availability problem rather than an exception.
    """
    import bm25s

    if not passages or not any(str(text).strip() for text in passages):
        return []
    index = bm25s.BM25()
    index.index(
        bm25s.tokenize(list(passages), stopwords=None, show_progress=False), show_progress=False
    )
    _results, scores = index.retrieve(
        bm25s.tokenize([question], stopwords=None, show_progress=False),
        k=min(len(passages), 20),
        show_progress=False,
    )
    ordered = sorted((float(score) for score in scores[0]), reverse=True)
    return ordered


def claim_support_ratio(answer: str, cited_text: str) -> float:
    """Deterministic verifier proxy: share of claim sentences whose values are supported.

    A claim is supported when every numeric/identifier value it asserts appears in the text it
    cites. This is the product verifier's question without its (label-correlated) judge.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        values = extract(sentence).all()
        if not values:
            supported += 1 if sentence else 0
            continue
        if not values - extract(cited_text).all():
            supported += 1
    return supported / len(sentences)


def citation_coverage(cited: Sequence[str]) -> float:
    """How much evidence the answer actually cites (0, 1/2, 1 for 0/1/2+ citations)."""
    unique = {citation for citation in cited if citation}
    return min(1.0, len(unique) / 2.0)


def build_row(
    *,
    row: Mapping[str, Any],
    split: str,
    documents: Mapping[str, str],
    reranker: Any | None = None,
    retrieval_passages: Sequence[str] | None = None,
    loader: Any | None = None,
) -> FeatureRow:
    """Assemble one feature row from a stored answer record plus recomputable evidence."""
    from answerdesk.contracts import Channel, Chunk, RetrievedPassage

    question = str(row.get("question") or "")
    answer = str(row.get("answer") or "")
    cited_docs = tuple(str(doc) for doc in (row.get("evidence_doc_ids") or ()))
    gold_docs = tuple(str(doc) for doc in (row.get("gold_doc_ids") or ()))
    cited_text = "\n".join(documents.get(doc, "") for doc in cited_docs)

    texts = list(retrieval_passages or [])
    if not texts:
        texts = [documents.get(doc, "") for doc in (row.get("retrieved_doc_ids") or ())]
    texts = [text for text in texts if text][:20]
    lexical = bm25_scores(question, texts) if texts else []
    top = lexical[0] if lexical else 0.0
    second = lexical[1] if len(lexical) > 1 else 0.0
    margin = (top - second) / (abs(top) + 1e-6) if top else 0.0
    above_gate = sum(1 for score in lexical if score > 0.0) / max(1, len(lexical))

    rerank_top = 0.0
    if reranker is not None and texts:
        passages = [
            RetrievedPassage(
                chunk=Chunk(
                    chunk_id=f"probe::{index}",
                    doc_id="probe",
                    text=text,
                    heading_path=[],
                    breadcrumb="",
                    start_char=0,
                    end_char=len(text),
                    token_count=len(text.split()),
                ),
                channel=Channel.BM25,
                score=float(score),
                rank=index + 1,
            )
            for index, (text, score) in enumerate(
                zip(texts, lexical or [0.0] * len(texts), strict=False)
            )
        ]
        reranked = reranker.rerank(question, passages, top_n=1)
        if reranked:
            raw = float(getattr(reranked[0], "score", 0.0))
            rerank_top = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))

    features = {
        "rerank_top": rerank_top,
        "retrieval_margin": max(0.0, min(1.0, margin)),
        "claim_support": claim_support_ratio(answer, cited_text) if answer else 0.0,
        "citation_coverage": citation_coverage(cited_docs),
        "numeric_consistency": (
            float(str(fidelity(answer, cited_text)["fidelity"])) if answer else 0.0
        ),
        "overlap_score": overlap_score(question, texts[0]) if texts else 0.0,
        "bm25_top": top,
        "bm25_margin": margin,
        "bm25_above_gate": above_gate,
        "citation_density": (len(cited_docs) / max(1.0, float(len(_terms(answer)) / 8 or 1))),
        "log_claims": math.log1p(max(1, len([s for s in re.split(r"[.!?]", answer) if s.strip()]))),
    }
    _ = loader
    rubric = row.get("rubric")
    label = 1 if rubric is not None and float(str(rubric)) >= 4 else 0
    return FeatureRow(
        sample_id=str(row.get("sample_id") or f"{row.get('fact_id')}::{row.get('locale')}"),
        fact_id=str(row.get("fact_id")),
        locale=str(row.get("locale")),
        bucket=str(row.get("archetype") or "unknown"),
        split=split,
        features=features,
        label=label,
        answer=answer,
        cited=cited_docs,
        gold=gold_docs,
    )


def load_documents(corpus_dir: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("docs/*/*.md")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def write_rows(path: Path, rows: Sequence[FeatureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({name for row in rows for name in row.features})
    header = ["sample_id", "fact_id", "locale", "bucket", "split", "label", *names]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row.sample_id,
                    row.fact_id,
                    row.locale,
                    row.bucket,
                    row.split,
                    str(row.label),
                    *[f"{row.features.get(name, 0.0):.6f}" for name in names],
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[FeatureRow]:
    import csv

    rows: list[FeatureRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            features = {
                name: float(value)
                for name, value in record.items()
                if name not in {"sample_id", "fact_id", "locale", "bucket", "split", "label"}
                and value not in (None, "")
            }
            rows.append(
                FeatureRow(
                    sample_id=record["sample_id"],
                    fact_id=record["fact_id"],
                    locale=record["locale"],
                    bucket=record["bucket"],
                    split=record["split"],
                    features=features,
                    label=int(record["label"]),
                )
            )
    return rows


def canonical_bytes(raw: bytes) -> bytes:
    """Normalise line endings to LF; every other byte is untouched (R-34/R-03).

    A digest over raw bytes is not reproducible: Git checks files out with CRLF on Windows
    and LF everywhere else, so the same committed table hashed two different ways and
    ``--check`` failed on a clean `git archive`. Hashing the canonical projection makes the
    digest a property of the *content*, which is what the provenance claim means.
    """
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def feature_digest(path: Path) -> str:
    """SHA-256 of a feature table over canonical LF bytes (line-ending independent)."""
    import hashlib

    return hashlib.sha256(canonical_bytes(path.read_bytes())).hexdigest()


__all__ = [
    "DECISION_FEATURES",
    "LEGACY_FEATURES",
    "SELF_CONSISTENCY",
    "FeatureRow",
    "bm25_scores",
    "build_row",
    "citation_coverage",
    "claim_support_ratio",
    "feature_digest",
    "load_documents",
    "overlap_score",
    "read_rows",
    "write_rows",
]
