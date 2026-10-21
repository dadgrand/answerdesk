"""Near-duplicate detection: the tool works, and the committed release is clean."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from bench.checks import dedup_checks
from bench.config import MINHASH_BANDS, MINHASH_HASHES, MINHASH_THRESHOLD, SHINGLE_SIZE
from bench.normalize import near_duplicate_pairs, tokens


def _pairs(texts: list[str]) -> list[tuple[int, int, float]]:
    return near_duplicate_pairs(
        texts,
        threshold=MINHASH_THRESHOLD,
        num_hashes=MINHASH_HASHES,
        bands=MINHASH_BANDS,
        size=SHINGLE_SIZE,
    )


def test_identical_texts_are_duplicates() -> None:
    pairs = _pairs(["The refund window is fourteen days from approval.", "The refund window is fourteen days from approval."])
    assert pairs and pairs[0][0] == 0 and pairs[0][1] == 1
    assert pairs[0][2] == 1.0


def test_a_small_edit_in_a_long_text_is_caught() -> None:
    """A long text (a passage) with one changed token is still a near-duplicate at 0.90."""
    first = " ".join(f"token{index}" for index in range(220))
    second = first.replace("token37", "token37x")
    pairs = _pairs([first, second])
    assert pairs and pairs[0][2] >= MINHASH_THRESHOLD


def test_short_question_typos_are_below_the_shingle_threshold() -> None:
    """Document the sensitivity limit.

    Word 5-gram Jaccard on a ten-word question is dominated by the edit, so a one-word typo
    does not reach 0.90. Short duplicates are caught by the *exact* token-sequence test
    instead; MinHash guards the long text (passages) and near-identical questions.
    """
    short = [
        "The refund window is fourteen days from approval for all customers.",
        "The refund window is fourteen days from approval for every customer.",
    ]
    assert _pairs(short) == []


def test_unrelated_texts_are_not_duplicates() -> None:
    pairs = _pairs(
        [
            "The refund window is fourteen days from approval.",
            "Uzbekistan borders Kazakhstan, Kyrgyzstan, Tajikistan, Afghanistan and Turkmenistan.",
        ]
    )
    assert pairs == []


def test_cyrillic_and_han_texts_are_compared_correctly() -> None:
    cyrillic = _pairs(
        [
            "Срок возврата средств составляет четырнадцать дней с момента одобрения.",
            "Срок возврата средств составляет четырнадцать дней с момента одобрения.",
        ]
    )
    assert cyrillic and cyrillic[0][2] == 1.0
    han = _pairs(["退款将在批准后的十四天内处理完成。", "退款将在批准后的十四天内处理完成。"])
    assert han and han[0][2] == 1.0
    unrelated = _pairs(["退款将在批准后的十四天内处理完成。", "乌兹别克斯坦与哈萨克斯坦接壤。"])
    assert unrelated == []


def test_committed_items_have_no_near_duplicates(items, bench) -> None:
    report = dedup_checks(items, bench[2])
    assert report["ok"] is True
    assert report["duplicates"] == []
    assert report["duplicate_count"] == 0


def test_no_exact_duplicate_question_within_suite_and_language(items) -> None:
    grouped: dict[tuple[str, str], Counter[str]] = {}
    for item in items:
        text = item.get("question") or item.get("query", "")
        key = (item["suite"], item["language"])
        grouped.setdefault(key, Counter())[" ".join(tokens(text))] += 1
    duplicates = {
        key: [text for text, count in counter.items() if count > 1]
        for key, counter in grouped.items()
        if any(count > 1 for count in counter.values())
    }
    assert duplicates == {}


def test_passage_texts_are_unique_within_each_corpus_file(bench) -> None:
    passage_files = bench[2]
    assert passage_files, "expected at least one corpus file"
    for path, rows in passage_files.items():
        texts = [row["text"] for row in rows]
        assert len(texts) == len(set(texts)), path
        assert len({row["passage_id"] for row in rows}) == len(rows), path
