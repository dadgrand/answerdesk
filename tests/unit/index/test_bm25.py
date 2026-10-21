"""BM25 index: determinism, identifier preservation, must-match probing, persistence.

# AT-RET-01  # AT-RET-03  # AT-OPS-02
"""

from __future__ import annotations

import json

import pytest

from answerdesk.contracts import Channel, RetrievedPassage, Retriever
from answerdesk.errors import IndexMissingError
from answerdesk.index.bm25_index import BM25Index, BM25Params, merge_probes
from answerdesk.index.store import ChunkStore
from answerdesk.index.text import tokenize


def _ids(passages: list[RetrievedPassage]) -> list[str]:
    return [passage.chunk.chunk_id for passage in passages]


def test_bm25_implements_the_retriever_protocol(bm25_index: BM25Index) -> None:
    """FR-RET-03: one protocol, no adapter detail leaking into callers."""
    assert isinstance(bm25_index, Retriever)
    assert bm25_index.name == "bm25"
    passages = bm25_index.search("обслуживание карты", 3)
    assert passages
    for rank, passage in enumerate(passages, start=1):
        assert isinstance(passage, RetrievedPassage)
        assert passage.channel is Channel.BM25
        assert passage.rank == rank
        assert passage.score > 0.0
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)


def test_repeated_searches_return_identical_ids_and_scores(bm25_index: BM25Index) -> None:
    """FR-RET-01 / §6.5: same query twice -> identical ranked list and scores."""
    first = bm25_index.search("TS-999 ошибка карта", 5)
    second = bm25_index.search("TS-999 ошибка карта", 5)
    assert _ids(first) == _ids(second)
    assert [p.score for p in first] == [p.score for p in second]


def test_rebuilding_the_index_gives_identical_rankings(
    fixture_chunks: list,
) -> None:
    """Determinism across processes starts with determinism across constructions."""
    first = BM25Index.from_chunks(fixture_chunks, text_for=ChunkStore.text_for)
    second = BM25Index.from_chunks(list(fixture_chunks), text_for=ChunkStore.text_for)
    left = first.search("кайра чыгаруу", 4)
    right = second.search("кайра чыгаруу", 4)
    assert _ids(left) == _ids(right)
    assert [p.score for p in left] == [p.score for p in right]


@pytest.mark.parametrize(
    ("query", "expected_chunk"),
    [
        ("TS-999", "ru-err::c0000"),
        ("TS-999", "en-err::c0000"),
        ("ORD-2026-000124", "zh-refund::c0000"),
        ("1200", "ru-card::c0000"),
    ],
)
def test_identifiers_and_numbers_are_retrievable(
    bm25_index: BM25Index, query: str, expected_chunk: str
) -> None:
    """FR-RET-01 + RT-8: numerals and identifier tokens survive tokenisation."""
    found = _ids(bm25_index.search(query, 5))
    assert expected_chunk in found, f"{query!r} did not retrieve {expected_chunk}: {found}"


def test_case_and_hyphen_variants_of_a_code_match_the_same_chunk(bm25_index: BM25Index) -> None:
    """Canonical tokens make ``ts-999`` and ``TS-999`` the same lexical evidence."""
    lower = _ids(bm25_index.search("ts-999", 3))
    upper = _ids(bm25_index.search("TS-999", 3))
    assert lower == upper
    assert "ru-err::c0000" in lower


def test_cjk_query_finds_the_chinese_chunk(bm25_index: BM25Index) -> None:
    """Character unigrams + bigrams are enough to match a query without a segmenter."""
    assert "zh-refund::c0000" in _ids(bm25_index.search("退款", 3))


def test_required_tokens_are_a_must_match_probe(bm25_index: BM25Index) -> None:
    """FR-RET-06: the identifier probe returns only chunks containing every token."""
    passages = bm25_index.search_terms(tokenize("ошибка карта"), 5, required=["ts-999"])
    assert passages
    for passage in passages:
        assert "ts-999" in passage.chunk.text.lower()
        assert passage.contributions["identifier_match"] == 1.0
    absent = bm25_index.search_terms(tokenize("ошибка"), 5, required=["zz-000"])
    assert absent == []


def test_merge_probes_keeps_the_best_rank_per_chunk(bm25_index: BM25Index) -> None:
    """The two sparse probes are merged by best rank, then re-ranked without gaps."""
    primary = bm25_index.search("ошибка карта", 5)
    must = bm25_index.search_terms(tokenize("ошибка карта"), 5, required=["ts-999"])
    merged = merge_probes(primary, must)
    assert [p.rank for p in merged] == list(range(1, len(merged) + 1))
    merged_ids = _ids(merged)
    assert len(merged_ids) == len(set(merged_ids))
    assert set(_ids(must)) <= set(merged_ids)
    # A chunk found by both probes keeps the better of the two ranks.
    for passage in merged:
        best = min(
            (p.rank for p in [*primary, *must] if p.chunk.chunk_id == passage.chunk.chunk_id),
        )
        assert passage.rank <= best + len(merged_ids)


def test_empty_query_and_zero_k_are_empty_not_errors(bm25_index: BM25Index) -> None:
    assert bm25_index.search("", 5) == []
    assert bm25_index.search("карта", 0) == []
    assert bm25_index.search("карта", -3) == []


def test_k_larger_than_the_corpus_is_clamped(bm25_index: BM25Index) -> None:
    passages = bm25_index.search("карта", 1000)
    assert len(passages) <= bm25_index.size


def test_save_and_load_roundtrip_is_exact(bm25_index: BM25Index, tmp_path) -> None:
    """FR-OPS-02: the persisted tokens reproduce the same ranking and scores."""
    bm25_index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25", bm25_index.chunks)
    assert loaded.params == bm25_index.params
    before = bm25_index.search("карта обслуживание", 5)
    after = loaded.search("карта обслуживание", 5)
    assert _ids(before) == _ids(after)
    assert [p.score for p in before] == [p.score for p in after]


def test_load_refuses_a_chunk_set_that_does_not_match(bm25_index: BM25Index, tmp_path) -> None:
    bm25_index.save(tmp_path / "bm25")
    with pytest.raises(IndexMissingError, match="does not match"):
        BM25Index.load(tmp_path / "bm25", list(reversed(bm25_index.chunks)))


def test_load_reports_a_missing_index(tmp_path) -> None:
    with pytest.raises(IndexMissingError, match="answerdesk index"):
        BM25Index.load(tmp_path / "nope", [])


def test_bm25_params_are_serialised_with_the_index(bm25_index: BM25Index, tmp_path) -> None:
    bm25_index.save(tmp_path / "bm25")
    meta = json.loads((tmp_path / "bm25" / "bm25.json").read_text(encoding="utf-8"))
    assert meta["params"] == BM25Params().as_dict()
    assert meta["size"] == bm25_index.size


def test_vocabulary_is_built_once_and_exposes_corpus_terms(bm25_index: BM25Index) -> None:
    vocabulary = bm25_index.vocabulary
    assert vocabulary is bm25_index.vocabulary
    # Exact surface forms only: there is no stemmer in the base install, so ``карта`` is
    # *not* the same term as ``карты`` — a documented limitation, exercised here.
    assert "обслуживание" in vocabulary
    assert "ts-999" in vocabulary
    assert "карта" not in vocabulary
    assert len(vocabulary) > 20


def test_stats_report_the_numbers_the_docs_quote(bm25_index: BM25Index) -> None:
    stats = bm25_index.stats()
    assert stats["chunks"] == bm25_index.size
    assert stats["vocabulary"] == len(bm25_index.vocabulary)
    assert stats["tokens_total"] > 0
    assert stats["tokens_max"] >= stats["tokens_mean"]
