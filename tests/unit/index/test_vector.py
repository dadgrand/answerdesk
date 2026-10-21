"""Exact dense index: nearest-neighbour equivalence, determinism, persistence, protocol.

# AT-RET-02  # AT-RET-03
"""

from __future__ import annotations

import numpy as np
import pytest

from answerdesk.contracts import Channel, Retriever
from answerdesk.errors import IndexMissingError
from answerdesk.index.vector_index import (
    ExactVectorIndex,
    brute_force_top_k,
    l2_normalize,
)
from answerdesk.testing.doubles import HashingEmbedder


def _random_matrix(rng: np.random.Generator, rows: int, dimension: int) -> np.ndarray:
    return rng.normal(size=(rows, dimension)).astype(np.float32)


def test_exact_top_k_equals_the_brute_force_reference() -> None:
    """AT-RET-02 / §6.5: identical ids **and** identical order on a 200-vector fixture."""
    rng = np.random.default_rng(1337)
    matrix = _random_matrix(rng, rows=200, dimension=24)
    chunks = [_chunk(f"c{index:03d}") for index in range(200)]
    index = ExactVectorIndex.from_matrix(chunks, matrix, HashingEmbedder(dimension=24))

    for _ in range(5):
        query = rng.normal(size=24).astype(np.float32)
        fast = index.search_vector(query, 10, positive_only=False)
        reference = brute_force_top_k(matrix, query, 10)
        assert [position for position, _ in fast] == [position for position, _ in reference]
        for (_, left), (_, right) in zip(fast, reference, strict=True):
            assert left == pytest.approx(right, abs=1e-6)


def test_scores_descend_and_stay_within_the_cosine_range(
    vector_index: ExactVectorIndex,
) -> None:
    passages = vector_index.search(" kart yenileme ücreti", 5)
    assert passages
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)
    for passage in passages:
        assert -1.0 - 1e-6 <= passage.score <= 1.0 + 1e-6
        assert passage.channel is Channel.DENSE
        assert passage.contributions["cosine"] == pytest.approx(passage.score)


def test_implements_the_retriever_protocol(vector_index: ExactVectorIndex) -> None:
    """FR-RET-03: both channels satisfy the same protocol."""
    assert isinstance(vector_index, Retriever)
    assert vector_index.name == "dense"
    passages = vector_index.search("перевыпуск карты", 3)
    assert [p.rank for p in passages] == list(range(1, len(passages) + 1))


def test_search_is_deterministic_across_rebuilds(fixture_chunks: list) -> None:
    embedder = HashingEmbedder(dimension=32)
    first = ExactVectorIndex.from_chunks(fixture_chunks, embedder)
    second = ExactVectorIndex.from_chunks(list(fixture_chunks), embedder)
    left = first.search("Баровардани корт", 4)
    right = second.search("Баровардани корт", 4)
    assert [p.chunk.chunk_id for p in left] == [p.chunk.chunk_id for p in right]
    assert [p.score for p in left] == [p.score for p in right]


def test_zero_vectors_score_zero_instead_of_nan() -> None:
    """An empty embedding must not poison the ranking with ``nan``."""
    chunks = [_chunk("a"), _chunk("b")]
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = ExactVectorIndex.from_matrix(chunks, matrix, HashingEmbedder(dimension=2))
    passages = index.search_embedded([1.0, 0.0], 2)
    assert [p.chunk.chunk_id for p in passages] == ["b"]
    assert passages[0].score == pytest.approx(1.0)


def test_negative_cosines_are_dropped_by_default() -> None:
    chunks = [_chunk("positive"), _chunk("negative")]
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    index = ExactVectorIndex.from_matrix(chunks, matrix, HashingEmbedder(dimension=2))
    assert [p.chunk.chunk_id for p in index.search_embedded([1.0, 0.0], 2)] == ["positive"]
    both = index.search_vector([1.0, 0.0], 2, positive_only=False)
    assert len(both) == 2


def test_dimension_mismatch_between_embedder_and_vectors_is_refused() -> None:
    chunks = [_chunk("a")]
    with pytest.raises(IndexMissingError, match="dimension"):
        ExactVectorIndex.from_matrix(chunks, np.zeros((1, 4), dtype=np.float32), HashingEmbedder(8))


def test_save_and_load_roundtrip(vector_index: ExactVectorIndex, tmp_path) -> None:
    embedder = HashingEmbedder(dimension=32)
    vector_index.save(tmp_path / "dense")
    loaded = ExactVectorIndex.load(tmp_path / "dense", vector_index.chunks, embedder)
    before = vector_index.search("TR 250", 3)
    after = loaded.search("TR 250", 3)
    assert [p.chunk.chunk_id for p in before] == [p.chunk.chunk_id for p in after]
    assert [p.score for p in before] == pytest.approx([p.score for p in after], abs=1e-6)


def test_load_refuses_a_different_embedder_dimension(
    vector_index: ExactVectorIndex, tmp_path
) -> None:
    vector_index.save(tmp_path / "dense")
    with pytest.raises(IndexMissingError, match="dimensional"):
        ExactVectorIndex.load(tmp_path / "dense", vector_index.chunks, HashingEmbedder(64))


def test_load_reports_a_missing_index(tmp_path) -> None:
    with pytest.raises(IndexMissingError, match="answerdesk index"):
        ExactVectorIndex.load(tmp_path / "nope", [_chunk("a")], HashingEmbedder(8))


def test_vector_for_returns_normalised_rows(vector_index: ExactVectorIndex) -> None:
    vector = vector_index.vector_for("ru-err::c0000")
    assert vector is not None
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
    assert vector_index.vector_for("does-not-exist") is None


def test_l2_normalize_leaves_zero_rows_zero() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalised = l2_normalize(matrix)
    assert normalised[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalised[1].tolist() == [0.0, 0.0]
    assert np.isfinite(normalised).all()


def test_stats_report_the_memory_note(vector_index: ExactVectorIndex) -> None:
    stats = vector_index.stats()
    assert stats["chunks"] == vector_index.size
    assert stats["dimension"] == 32
    assert stats["bytes"] == vector_index.size * 32 * 4


def _chunk(chunk_id: str):
    from answerdesk.contracts import Chunk

    text = f"passage {chunk_id}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=2,
    )
