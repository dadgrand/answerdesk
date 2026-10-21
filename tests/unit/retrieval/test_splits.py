"""Document-disjoint dev/test splitting (INV-09): determinism, leakage, proportions.

# AT-RET-05
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import Split
from answerdesk.retrieval.splits import (
    DEV_PERCENT,
    SPLIT_SALT,
    assign,
    document_bucket,
    split_for_document,
    subset,
)


def test_document_bucket_is_stable_and_bounded() -> None:
    for doc_id in ("kb-card", "kb-policy", "syn-ru-0001", ""):
        bucket = document_bucket(doc_id)
        assert document_bucket(doc_id) == bucket
        assert 0 <= bucket < 100


def test_salt_changes_the_assignment_for_at_least_one_document() -> None:
    documents = [f"doc-{index}" for index in range(40)]
    default = [split_for_document(doc) for doc in documents]
    rolled = [split_for_document(doc, salt="another-salt") for doc in documents]
    assert default != rolled


def test_split_is_roughly_balanced_for_the_default_percentage() -> None:
    documents = [f"doc-{index:04d}" for index in range(400)]
    dev = sum(1 for doc in documents if split_for_document(doc) is Split.DEV)
    share = dev / len(documents)
    assert abs(share - DEV_PERCENT / 100) < 0.10, share


def test_assignment_is_document_disjoint_and_complete() -> None:
    items = [
        (f"{doc}-q{index}", doc)
        for doc in ("doc-a", "doc-b", "doc-c", "doc-d")
        for index in range(5)
    ]
    assignment = assign(items)
    assert set(assignment.dev_ids) | set(assignment.test_ids) == {item[0] for item in items}
    assert not set(assignment.dev_ids) & set(assignment.test_ids)
    assignment.assert_document_disjoint()
    assert set(assignment.dev_documents).isdisjoint(assignment.test_documents)


def test_every_query_of_a_document_lands_on_one_side() -> None:
    items = [("q1", "same-doc"), ("q2", "same-doc"), ("q3", "same-doc")]
    assignment = assign(items)
    assert (
        ("q1" in assignment.dev_ids) == ("q2" in assignment.dev_ids) == ("q3" in assignment.dev_ids)
    )


def test_provenance_hashes_are_derived_from_the_item_ids() -> None:
    assignment = assign([(f"q{index}", f"doc-{index}") for index in range(30)])
    dev = assignment.dev_provenance
    test = assignment.test_provenance
    assert dev.name == "dev" and test.name == "test"
    assert dev.n_items == len(assignment.dev_ids)
    assert dev.n_documents == len(assignment.dev_documents)
    assert len(dev.hash) == 64
    assert dev.hash != test.hash
    assert dev.salt == SPLIT_SALT


def test_document_overlap_is_a_hard_error() -> None:
    assignment = assign([("q1", "doc-a"), ("q2", "doc-b")])
    assignment.dev_documents = ["doc-a"]
    assignment.test_documents = ["doc-a", "doc-b"]
    with pytest.raises(ValueError, match="document-level leakage"):
        assignment.assert_document_disjoint()


def test_subset_filters_by_id_and_keeps_order() -> None:
    class Item:
        def __init__(self, case_id: str) -> None:
            self.case_id = case_id

    items = [Item("a"), Item("b"), Item("c")]
    assert [item.case_id for item in subset(items, ["c", "a"])] == ["a", "c"]
    assert subset(items, []) == []
