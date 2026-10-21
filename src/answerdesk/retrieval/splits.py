"""Document-disjoint dev/test splitting with a hashed, auditable assignment (INV-09).

The split rule is deliberately boring and reproducible:

    bucket(doc_id) = int(sha256(salt + doc_id)[:8], 16) % 100
    bucket <  50   -> dev        (parameters are fitted here)
    bucket >= 50   -> test       (reporting only, never fitted on)

Hashing the **document**, not the query, is the important part: every query of a document
lands in the same split, so a dev query can never share its source article with a test
query. A random per-query split would leak exactly that (the model would be tuned on the
same article it is later scored against), which is the failure mode requirements §10.2
calls a hard-fail.

``salt`` is part of the provenance. Changing it re-rolls the split — which is allowed for
research, and recorded, so a reported number can always be traced to the split that
produced it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from answerdesk.contracts import Split
from answerdesk.retrieval.fit import SplitProvenance, split_hash

#: Salt of the shipped split. Frozen: it is part of every retrieval report header.
SPLIT_SALT = "answerdesk-retrieval-split-v1"

#: Fraction of documents that go to dev (the rest go to test).
DEV_PERCENT = 50


def document_bucket(doc_id: str, *, salt: str = SPLIT_SALT, buckets: int = 100) -> int:
    """Stable bucket in ``[0, buckets)`` for a document id (process-independent)."""
    if buckets < 1:
        raise ValueError("buckets must be >= 1")
    digest = hashlib.sha256(f"{salt}:{doc_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def split_for_document(
    doc_id: str,
    *,
    salt: str = SPLIT_SALT,
    dev_percent: int = DEV_PERCENT,
) -> Split:
    """``Split.DEV`` or ``Split.TEST`` for one document."""
    return Split.DEV if document_bucket(doc_id, salt=salt) < dev_percent else Split.TEST


@dataclass
class SplitAssignment:
    """The materialised split: item ids per side, plus reviewable provenance."""

    dev_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    dev_documents: list[str] = field(default_factory=list)
    test_documents: list[str] = field(default_factory=list)
    salt: str = SPLIT_SALT

    def provenance(self, name: Literal["dev", "test"]) -> SplitProvenance:
        """Provenance record for one side, hashing the full item list (not a name)."""
        if name == "dev":
            return SplitProvenance(
                name="dev",
                hash=split_hash(self.dev_ids),
                n_items=len(self.dev_ids),
                n_documents=len(self.dev_documents),
                salt=self.salt,
            )
        return SplitProvenance(
            name="test",
            hash=split_hash(self.test_ids),
            n_items=len(self.test_ids),
            n_documents=len(self.test_documents),
            salt=self.salt,
        )

    @property
    def dev_provenance(self) -> SplitProvenance:
        return self.provenance("dev")

    @property
    def test_provenance(self) -> SplitProvenance:
        return self.provenance("test")

    def assert_document_disjoint(self) -> None:
        """Raise when a document appears on both sides — the §10.2 hard-fail condition."""
        overlap = sorted(set(self.dev_documents) & set(self.test_documents))
        if overlap:
            raise ValueError(
                "document-level leakage between dev and test: "
                f"{len(overlap)} shared documents, first: {overlap[:5]}"
            )


def assign(items: Iterable[tuple[str, str]], *, salt: str = SPLIT_SALT) -> SplitAssignment:
    """Split ``(item_id, doc_id)`` pairs by document bucket, preserving input order."""
    assignment = SplitAssignment(salt=salt)
    seen_docs: dict[str, Split] = {}
    for item_id, doc_id in items:
        side = seen_docs.get(doc_id)
        if side is None:
            side = split_for_document(doc_id, salt=salt)
            seen_docs[doc_id] = side
        if side is Split.DEV:
            assignment.dev_ids.append(item_id)
        else:
            assignment.test_ids.append(item_id)
    assignment.dev_documents = sorted(doc for doc, side in seen_docs.items() if side is Split.DEV)
    assignment.test_documents = sorted(doc for doc, side in seen_docs.items() if side is Split.TEST)
    assignment.assert_document_disjoint()
    return assignment


def subset(
    items: Sequence[Any],
    ids: Sequence[str],
    id_of: Callable[[Any], str] | None = None,
) -> list[Any]:
    """Select the items whose id is in ``ids``, keeping the original order.

    ``id_of`` defaults to ``item.case_id``, which is the field every retrieval case has.
    """
    wanted = set(ids)
    getter: Callable[[Any], str] = id_of or (lambda item: str(item.case_id))
    return [item for item in items if getter(item) in wanted]


__all__ = [
    "DEV_PERCENT",
    "SPLIT_SALT",
    "SplitAssignment",
    "assign",
    "document_bucket",
    "split_for_document",
    "subset",
]
