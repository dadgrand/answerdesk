"""Typed reader for the curated ``answerdesk-bench`` v0.1 suite.

The repository carries the bench (committed by t20), so the external arm runs offline and
reproducibly. Belebele-derived text is CC BY-SA 4.0: it is read from the committed bench and
**never** copied into reports — artefacts carry ids, metrics and counts only.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from answerdesk.eval.external.datasets import DatasetRecord, ExternalDataError


@dataclass(frozen=True)
class QaItem:
    item_id: str
    language: str
    question: str
    options: tuple[str, ...]
    answer_index: int
    god_evidence_ids: tuple[str, ...]
    split: str
    licence: str
    suite: str

    @property
    def gold_passage_ids(self) -> tuple[str, ...]:
        return self.god_evidence_ids


@dataclass(frozen=True)
class RetrievalQuery:
    item_id: str
    language: str
    query: str
    qrels: dict[str, int]
    split: str
    licence: str
    suite: str


@dataclass(frozen=True)
class Passage:
    passage_id: str
    language: str
    text: str
    licence: str


def _qrels(raw: Any) -> dict[str, int]:
    """The bench stores qrels as a passage-id list; accept graded dicts too."""
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {str(entry): 1 for entry in raw}
    return {}


class Bench:
    """All suites of the curated benchmark, with lazy per-language loading."""

    def __init__(self, record: DatasetRecord) -> None:
        self.record = record
        self.root = record.root
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ExternalDataError("curated benchmark has no manifest.json")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    # -- discovery ---------------------------------------------------------------------
    @property
    def suites(self) -> dict[str, Any]:
        return dict(self.manifest.get("suites", {}))

    @property
    def languages(self) -> list[str]:
        return list(self.manifest.get("languages", []))

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unknown"))

    def licence_of(self, suite: str) -> str:
        return str(self.suites.get(suite, {}).get("license", "unknown"))

    def items_of(self, suite: str) -> int:
        return int(self.suites.get(suite, {}).get("items", 0))

    # -- readers -----------------------------------------------------------------------
    def _jsonl(self, relative: str) -> Iterator[dict[str, Any]]:
        path = self.root / relative
        if not path.is_file():
            raise ExternalDataError(f"bench file is missing: {relative}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def qa(self, language: str, *, split: str | None = "test") -> list[QaItem]:
        items = [
            QaItem(
                item_id=row["item_id"],
                language=row["language"],
                question=row["question"],
                options=tuple(row["options"]),
                answer_index=int(row["answer_index"]),
                god_evidence_ids=tuple(row.get("gold_evidence_ids", ())),
                split=row.get("split", ""),
                licence=row.get("license", ""),
                suite=row.get("suite", "belebele_qa"),
            )
            for row in self._jsonl(f"belebele/qa/{language}.jsonl")
        ]
        return [item for item in items if split is None or item.split == split]

    def retrieval_queries(
        self, language: str, *, split: str | None = "test"
    ) -> list[RetrievalQuery]:
        queries = [
            RetrievalQuery(
                item_id=row["item_id"],
                language=row["language"],
                query=row["query"],
                qrels=_qrels(row.get("qrels")),
                split=row.get("split", ""),
                licence=row.get("license", ""),
                suite=row.get("suite", "belebele_retrieval"),
            )
            for row in self._jsonl(f"belebele/retrieval/{language}/queries.jsonl")
        ]
        return [query for query in queries if split is None or query.split == split]

    def passages(self, language: str) -> list[Passage]:
        return [
            Passage(
                passage_id=row["passage_id"],
                language=row["language"],
                text=row["text"],
                licence=row.get("license", ""),
            )
            for row in self._jsonl(f"belebele/retrieval/{language}/corpus.jsonl")
        ]

    def mrtydi_queries(self, *, split: str | None = "test") -> list[RetrievalQuery]:
        queries = [
            RetrievalQuery(
                item_id=row["item_id"],
                language=row["language"],
                query=row["query"],
                qrels=_qrels(row.get("qrels")),
                split=row.get("split", ""),
                licence=row.get("license", ""),
                suite=row.get("suite", "mrtydi_retrieval"),
            )
            for row in self._jsonl("mrtydi/ru/queries.jsonl")
        ]
        return [query for query in queries if split is None or query.split == split]

    def mrtydi_passages(self) -> list[Passage]:
        return [
            Passage(
                passage_id=row["passage_id"],
                language=row.get("language", "ru"),
                text=row["text"],
                licence=row.get("license", ""),
            )
            for row in self._jsonl("mrtydi/ru/corpus.jsonl")
        ]

    def domain(self, language: str, *, split: str | None = "test") -> list[dict[str, Any]]:
        rows = list(self._jsonl(f"domain/{language}.jsonl"))
        return [row for row in rows if split is None or row.get("split") == split]

    def suite_counts(self) -> dict[str, dict[str, int]]:
        """Per-suite, per-language item counts straight from the manifest checks."""
        counts = self.manifest.get("counts", {})
        by_suite = counts.get("by_suite", {})
        by_language = counts.get("by_language", {})
        return {
            suite: {
                "items": int(by_suite.get(suite, 0)),
                **{language: int(by_language.get(language, 0)) for language in self.languages},
            }
            for suite in by_suite
        }


__all__ = ["Bench", "Passage", "QaItem", "RetrievalQuery"]
