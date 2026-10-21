"""Counts and acceptance floors: the release must clear the numbers the task contract fixes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from bench.config import (
    MIN_DOMAIN_ITEMS,
    MIN_DOMAIN_ITEMS_PER_CATEGORY,
    MIN_QA_ITEMS_PER_LANG,
    MIN_RETRIEVAL_QUERIES_PER_LANG,
    MIN_TOTAL_ITEMS,
)
from bench.schema import DOMAIN_CATEGORIES

BELEBELE_LANGUAGES = ("ru", "uz", "tg", "ky", "zh")


def test_total_item_floor(items: list[dict[str, Any]]) -> None:
    assert len(items) >= MIN_TOTAL_ITEMS


def test_retrieval_and_qa_floors_per_language(items: list[dict[str, Any]]) -> None:
    retrieval = Counter(item["language"] for item in items if item["suite"] == "belebele_retrieval")
    qa = Counter(item["language"] for item in items if item["suite"] == "belebele_qa")
    for language in BELEBELE_LANGUAGES:
        assert retrieval[language] >= MIN_RETRIEVAL_QUERIES_PER_LANG, language
        assert qa[language] >= MIN_QA_ITEMS_PER_LANG, language
    assert "ru" in {item["language"] for item in items if item["suite"] == "mrtydi_retrieval"}


def test_domain_slice_covers_every_required_category(items: list[dict[str, Any]]) -> None:
    domain = [item for item in items if item["suite"] == "domain"]
    assert len(domain) >= MIN_DOMAIN_ITEMS
    categories = Counter(item["category"] for item in domain)
    for category in DOMAIN_CATEGORIES:
        assert categories[category] >= MIN_DOMAIN_ITEMS_PER_CATEGORY, category
    for required in ("unanswerable", "pii", "injection", "numeric_date", "multi_hop"):
        assert categories[required] > 0, required
    languages = {item["language"] for item in domain}
    assert {"ru", "uz", "tg", "ky", "zh", "tr"} <= languages


def test_counts_match_the_manifest(manifest: dict[str, Any], items: list[dict[str, Any]]) -> None:
    counts = manifest["counts"]
    assert counts["items"] == len(items)
    assert counts["by_suite"] == dict(sorted(Counter(item["suite"] for item in items).items()))
    assert counts["by_language"] == dict(
        sorted(Counter(item["language"] for item in items).items())
    )
    assert counts["by_split"] == dict(sorted(Counter(item["split"] for item in items).items()))
    assert counts["by_category"] == dict(
        sorted(Counter(item["category"] for item in items).items())
    )
    assert counts["retrieval_queries"] == (
        counts["by_suite"]["belebele_retrieval"] + counts["by_suite"]["mrtydi_retrieval"]
    )
    assert counts["qa_items"] == counts["by_suite"]["belebele_qa"]
    assert counts["domain_items"] == counts["by_suite"]["domain"]


def test_languages_cover_the_task_contract(manifest: dict[str, Any]) -> None:
    assert {"ru", "uz", "tg", "ky", "zh"} <= set(manifest["languages"])


def test_per_file_item_counts_match_the_manifest(manifest: dict[str, Any], bench_dir: Path) -> None:
    for row in manifest["files"]:
        path = bench_dir / row["path"]
        if not row["path"].endswith(".jsonl"):
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == row["items"], row["path"]


def test_splits_are_both_populated(items: list[dict[str, Any]]) -> None:
    splits = Counter(item["split"] for item in items)
    assert splits["dev"] > 0 and splits["test"] > splits["dev"]
