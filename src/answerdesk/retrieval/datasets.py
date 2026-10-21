"""Retrieval cases derived from the corpus query set — a *dev harness*, not the golden set.

Provenance matters more than size here. The golden set (``golden/v1``) is owned by t7; what
this module builds is the retrieval-local evidence needed to fit fusion parameters and to
run the ablation table, with its labelling rule written down:

======================  ====================================================
field                   how it is derived
======================  ====================================================
``gold_chunk_ids``      every chunk of the query's source document. Coarse
                        (document-level) but *certain*: the synthetic query was
                        generated from that document, so a chunk from it is
                        never a wrong label.
``answer_chunk_ids``    the single chunk of the source document with the highest
                        lexical overlap with the generator's ``operator_reply``.
                        A heuristic — reported separately as the "strict" view,
                        never silently mixed with the coarse label.
``locale``              the generator's locale label (ground truth, not the
                        detector's guess), so language-ID accuracy can be
                        measured against it.
``category``            ``direct`` / ``typo`` / ``code_switch`` / ``numeric_date``
                        / ``identifier`` — used for the RT-8/RT-9 subsets.
======================  ====================================================

Both labels are stored, and every report says which one a number came from. That is the
same discipline requirements §10.1 applies to the golden set, applied one level down.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from answerdesk.contracts import Chunk, Split
from answerdesk.errors import CorpusMissingError
from answerdesk.index.text import tokenize
from answerdesk.retrieval.splits import SPLIT_SALT, SplitAssignment, assign

#: Query categories the corpus generator emits, and the ones the RT gates care about.
IDENTIFIER_CATEGORIES: frozenset[str] = frozenset({"identifier", "numeric_date"})
PERTURBED_CATEGORIES: frozenset[str] = frozenset({"typo", "code_switch"})


class RetrievalCase(BaseModel):
    """One retrieval evaluation item with both label granularities and its split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    category: str = "direct"
    intent: str = ""
    query: str = Field(min_length=1)
    queries_text: str = ""
    gold_document: str = Field(min_length=1)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    answer_chunk_ids: list[str] = Field(default_factory=list)
    perturbations: list[str] = Field(default_factory=list)
    split: Split | None = None

    @property
    def is_identifier(self) -> bool:
        return self.category in IDENTIFIER_CATEGORIES

    @property
    def is_perturbed(self) -> bool:
        return self.category in PERTURBED_CATEGORIES


class SplitBundle(BaseModel):
    """Cases grouped by split, plus the provenance a report header needs."""

    model_config = ConfigDict(extra="forbid")

    dev: list[RetrievalCase] = Field(default_factory=list)
    test: list[RetrievalCase] = Field(default_factory=list)
    salt: str = SPLIT_SALT

    @property
    def all(self) -> list[RetrievalCase]:
        return [*self.dev, *self.test]


def _strings(value: object) -> list[str]:
    """Coerce a JSON field that should be a list of strings (and may be null)."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _overlap(first: str, second: str) -> float:
    """Jaccard overlap of token sets — the heuristic behind ``answer_chunk_ids``."""
    left = set(tokenize(first, with_canonical=False))
    right = set(tokenize(second, with_canonical=False))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_cases(
    chunks: Sequence[Chunk],
    query_rows: Sequence[dict[str, object]],
) -> list[RetrievalCase]:
    """Turn raw query rows into cases, deriving gold chunks from the chunk store."""
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    cases: list[RetrievalCase] = []
    for row in query_rows:
        document = str(row.get("doc_id", ""))
        document_chunks = by_document.get(document, [])
        if not document_chunks:
            continue
        query = str(row.get("text") or row.get("base_text") or "").strip()
        if not query:
            continue
        reference = str(row.get("operator_reply") or "")
        best_position = 0
        best_score = -1.0
        for position, chunk in enumerate(document_chunks):
            score = _overlap(reference, chunk.text) if reference else 0.0
            if score > best_score:
                best_score, best_position = score, position
        best = document_chunks[best_position]
        cases.append(
            RetrievalCase(
                case_id=str(row.get("query_id") or f"{document}-q{len(cases)}"),
                locale=str(row.get("locale") or "und"),
                category=str(row.get("category") or "direct"),
                intent=str(row.get("intent") or ""),
                query=query,
                queries_text=str(row.get("base_text") or ""),
                gold_document=document,
                gold_chunk_ids=sorted(chunk.chunk_id for chunk in document_chunks),
                answer_chunk_ids=[best.chunk_id] if reference else [],
                perturbations=_strings(row.get("perturbations")),
            )
        )
    return cases


def load_query_rows(corpus_dir: str | Path) -> list[dict[str, object]]:
    """Read ``queries/queries.jsonl`` from a built corpus."""
    path = Path(corpus_dir) / "queries" / "queries.jsonl"
    if not path.is_file():
        raise CorpusMissingError(f"no query set at {path}; run `answerdesk ingest --demo` first")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def split_cases(
    cases: Sequence[RetrievalCase],
    *,
    salt: str = SPLIT_SALT,
) -> SplitBundle:
    """Assign every case to dev or test by its **document**, then attach the label."""
    assignment: SplitAssignment = assign(
        [(case.case_id, case.gold_document) for case in cases], salt=salt
    )
    assignment.assert_document_disjoint()
    dev_ids, test_ids = set(assignment.dev_ids), set(assignment.test_ids)
    bundle = SplitBundle(salt=salt)
    for case in cases:
        side = Split.DEV if case.case_id in dev_ids else Split.TEST
        if case.case_id not in dev_ids and case.case_id not in test_ids:  # pragma: no cover
            raise CorpusMissingError(f"case {case.case_id!r} was not assigned to a split")
        labelled = case.model_copy(update={"split": side})
        if side is Split.DEV:
            bundle.dev.append(labelled)
        else:
            bundle.test.append(labelled)
    return bundle


__all__ = [
    "IDENTIFIER_CATEGORIES",
    "PERTURBED_CATEGORIES",
    "RetrievalCase",
    "SplitBundle",
    "build_cases",
    "load_query_rows",
    "split_cases",
]
