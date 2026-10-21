"""Leakage control: splits are disjoint and the domain slice never touches t11's golden set."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from bench.build import golden_exclusions
from bench.config import CORPUS_DIR, GOLDEN_DIR, SPLITS_FILE
from bench.normalize import normalise_question


def _domain_documents(items: list[dict[str, Any]]) -> set[str]:
    return {
        str(doc_id)
        for item in items
        if item["suite"] == "domain"
        for doc_id in item.get("gold_evidence_ids", [])
    }


def test_no_item_id_lives_in_both_splits(items: list[dict[str, Any]]) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for item in items:
        by_split[item["split"]].add(item["item_id"])
    assert by_split["dev"] & by_split["test"] == set()


def test_a_belebele_passage_is_gold_evidence_in_one_split_only(items: list[dict[str, Any]]) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["suite"] == "belebele_qa":
            identifiers = item["gold_evidence_ids"]
        elif item["suite"] == "belebele_retrieval":
            identifiers = item["qrels"]
        else:
            continue
        for identifier in identifiers:
            splits[str(identifier)].add(item["split"])
    leaked = {passage: values for passage, values in splits.items() if len(values) > 1}
    assert leaked == {}


def test_domain_documents_do_not_cross_the_split(items: list[dict[str, Any]]) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["suite"] != "domain":
            continue
        for doc_id in item.get("gold_evidence_ids", []):
            by_split[item["split"]].add(str(doc_id))
    assert by_split["dev"] & by_split["test"] == set()


def test_domain_documents_are_disjoint_from_the_t11_golden_split(items: list[dict[str, Any]]) -> None:
    if not GOLDEN_DIR.is_dir():  # pragma: no cover - the repository ships the golden split
        pytest.skip("evals/golden/v1 is not present on this checkout")
    golden_documents, _, _ = golden_exclusions(GOLDEN_DIR, SPLITS_FILE)
    overlap = _domain_documents(items) & golden_documents
    assert overlap == set(), sorted(overlap)[:10]


def test_recorded_exclusions_cover_the_live_golden_split(bench_dir: Path) -> None:
    if not GOLDEN_DIR.is_dir():  # pragma: no cover
        pytest.skip("evals/golden/v1 is not present on this checkout")
    exclusions = json.loads((bench_dir / "domain" / "exclusions.json").read_text(encoding="utf-8"))
    recorded = {str(value) for value in exclusions["excluded_document_ids"]}
    golden_documents, golden_questions, golden_hashes = golden_exclusions(GOLDEN_DIR, SPLITS_FILE)
    assert golden_documents <= recorded, "the benchmark was built against an older golden split"
    assert len(golden_hashes) > 0
    assert exclusions["source_files"] == golden_hashes
    assert exclusions["excluded_questions"] == len(golden_questions)


def test_domain_questions_do_not_repeat_a_golden_question(items: list[dict[str, Any]]) -> None:
    if not GOLDEN_DIR.is_dir():  # pragma: no cover
        pytest.skip("evals/golden/v1 is not present on this checkout")
    _, golden_questions, _ = golden_exclusions(GOLDEN_DIR, SPLITS_FILE)
    domain_questions = {
        normalise_question(str(item["question"])) for item in items if item["suite"] == "domain"
    }
    assert domain_questions & golden_questions == set()


def test_domain_documents_exist_in_the_synthetic_corpus(items: list[dict[str, Any]]) -> None:
    if not (CORPUS_DIR / "docs").is_dir():  # pragma: no cover
        pytest.skip("data/corpus is not present on this checkout")
    documents = {path.stem for path in (CORPUS_DIR / "docs").rglob("*.md")}
    missing = _domain_documents(items) - documents
    assert missing == set(), sorted(missing)[:10]
