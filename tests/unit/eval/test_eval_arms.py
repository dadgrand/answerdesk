"""Retrieval-arm tests: hand-computed RRF, BM25 ordering, determinism (AT-RET-04).

# AT-RET-04  # AT-EVL-04  # AT-DET-02
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import Channel, RetrievedPassage
from answerdesk.eval.arms import BM25Arm, DenseHashArm, RrfArm, build_arms
from answerdesk.eval.corpus import Corpus


def test_bm25_finds_the_chunk_with_the_identifying_token(tiny_corpus: Corpus) -> None:
    arm = BM25Arm(tiny_corpus)
    passages = arm.search("Заблокировать карту", 5)
    assert passages
    assert passages[0].rank == 1
    assert passages[0].score > 0
    assert all(passage.chunk.doc_id for passage in passages)


def test_bm25_ranking_is_deterministic(tiny_corpus: Corpus) -> None:
    first = BM25Arm(tiny_corpus).search("карта блокировка", 10)
    second = BM25Arm(tiny_corpus).search("карта блокировка", 10)
    assert [p.chunk.chunk_id for p in first] == [p.chunk.chunk_id for p in second]
    assert [p.score for p in first] == [p.score for p in second]


def test_bm25_returns_nothing_for_an_empty_query(tiny_corpus: Corpus) -> None:
    assert BM25Arm(tiny_corpus).search("", 10) == []


def test_dense_arm_ranks_by_cosine_and_starts_at_one(tiny_corpus: Corpus) -> None:
    passages = DenseHashArm(tiny_corpus).search("блокировка карты", 5)
    assert passages
    assert [p.rank for p in passages] == list(range(1, len(passages) + 1))
    assert all(p.channel is Channel.DENSE for p in passages)
    assert passages[0].score >= passages[-1].score


class _StubArm:
    """Hand-built ranking so RRF can be checked against the formula, not against vibes."""

    def __init__(self, order: list[str], corpus: Corpus, channel: Channel) -> None:
        self._order = order
        self.corpus = corpus
        self.channel = channel
        self.name = "stub"

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                chunk=self.corpus.by_id[chunk_id],
                channel=self.channel,
                score=1.0,
                rank=rank,
            )
            for rank, chunk_id in enumerate(self._order[:k], start=1)
        ]


def test_rrf_matches_the_formula(tiny_corpus: Corpus) -> None:
    ids = list(tiny_corpus.by_id)
    first = [ids[0], ids[1], ids[2]]
    second = [ids[1], ids[0], ids[3]]
    arm = RrfArm(
        [
            _StubArm(first, tiny_corpus, Channel.BM25),
            _StubArm(second, tiny_corpus, Channel.DENSE),
        ],
        k=60,
        depth=50,
    )
    scores = arm.score_all("q")
    assert scores[ids[0]] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[ids[1]] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[ids[2]] == pytest.approx(1 / 63)
    assert scores[ids[3]] == pytest.approx(1 / 63)
    ranking = arm.search("q", 4)
    # ids[0] and ids[1] tie on score; the tie breaks on chunk id, deterministically
    assert [p.chunk.chunk_id for p in ranking] == sorted([ids[0], ids[1]]) + sorted(
        [ids[2], ids[3]]
    )


def test_rrf_needs_two_component_arms(tiny_corpus: Corpus) -> None:
    with pytest.raises(ValueError):
        RrfArm([_StubArm([], tiny_corpus, Channel.BM25)])


def test_build_arms_shares_components(tiny_corpus: Corpus) -> None:
    arms = build_arms(tiny_corpus, ("bm25", "hybrid-rrf"))
    assert set(arms) == {"bm25", "hybrid-rrf"}
    assert isinstance(arms["bm25"], BM25Arm)
    assert isinstance(arms["hybrid-rrf"], RrfArm)


def test_unknown_arm_is_a_configuration_error(tiny_corpus: Corpus) -> None:
    with pytest.raises(ValueError):
        build_arms(tiny_corpus, ("nope",))
