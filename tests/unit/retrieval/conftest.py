"""Shared fixtures for the retrieval tests: passages, channels and a tiny pipeline corpus.

Everything here is synthetic and offline on purpose — the retrieval maths must be provable
without a corpus, an embedder or a network (INV-08).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from answerdesk.contracts import Channel, Chunk, RetrievedPassage
from answerdesk.index.bm25_index import BM25Index
from answerdesk.index.store import ChunkStore
from answerdesk.index.vector_index import ExactVectorIndex
from answerdesk.retrieval.pipeline import PipelineConfig, RetrievalPipeline
from answerdesk.testing.doubles import HashingEmbedder

PassageFactory = Callable[..., RetrievedPassage]


@pytest.fixture()
def passage_factory() -> PassageFactory:
    """Build a passage from a chunk id, a score and a channel (rank is filled later)."""

    def _make(
        chunk_id: str,
        score: float = 1.0,
        channel: Channel = Channel.BM25,
        *,
        text: str | None = None,
        rank: int = 1,
    ) -> RetrievedPassage:
        body = text or f"passage body for {chunk_id} unordered words"
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            text=body,
            heading_path=["Test"],
            breadcrumb="Test",
            start_char=0,
            end_char=len(body),
            token_count=len(body.split()),
        )
        return RetrievedPassage(chunk=chunk, channel=channel, score=score, rank=rank)

    return _make


@pytest.fixture()
def sized_passages(passage_factory: PassageFactory) -> Callable[..., list[RetrievedPassage]]:
    """A channel result set with descending scores and sequential ranks."""

    def _make(
        prefix: str,
        scores: Sequence[float],
        channel: Channel = Channel.BM25,
    ) -> list[RetrievedPassage]:
        return [
            passage_factory(f"{prefix}{index}", score, channel, rank=index + 1)
            for index, score in enumerate(scores)
        ]

    return _make


@pytest.fixture()
def mini_corpus(make_document: Callable[..., object]) -> list[Chunk]:
    """A handful of real chunks (through the production chunker) for pipeline tests."""
    documents = [
        make_document(
            doc_id="kb-policy",
            title="Refund policy",
            text=(
                "# Refund policy\n\n"
                "Refunds are processed within 14 days of approval. "
                "The refund is returned to the original card.\n\n"
                "## Exceptions\n\n"
                "Orders paid by bank transfer take up to 30 days.\n"
            ),
        ),
        make_document(
            doc_id="kb-card",
            title="Card fees",
            text=(
                "# Card fees\n\n"
                "Servicing the Basic card costs 1200 RUB per year. "
                "Replacement takes 5 working days and costs 300 RUB.\n\n"
                "## Errors\n\n"
                "Error code TS-999 means the card is blocked until KYC completes.\n"
            ),
        ),
        make_document(
            doc_id="kb-transfer",
            title="Transfers",
            text=(
                "# Transfers\n\n"
                "International transfers arrive within 2 business days and cost 1% of the "
                "amount. Order ORD-2026-10009 can be tracked in the app.\n"
            ),
        ),
        make_document(
            doc_id="kb-zh",
            title="退款政策",
            text=("# 退款政策\n\n退款将在14天内处理。订单号 ORD-2026-000124。\n"),
        ),
        make_document(
            doc_id="kb-tg",
            title="Кортҳо",
            text=("# Кортҳо\n\nБаровардани корт 100 сомони аст ва 10 рӯз мегирад.\n"),
        ),
    ]
    return ChunkStore.from_documents(documents).chunks  # type: ignore[arg-type]


@pytest.fixture()
def mini_index(mini_corpus: Sequence[Chunk], hashing_embedder: HashingEmbedder) -> object:
    store = ChunkStore.from_chunks(mini_corpus, corpus_version="corpus-mini")
    from answerdesk.index.store import IndexBundle

    return IndexBundle.build(store, hashing_embedder)


@pytest.fixture()
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=48)


@pytest.fixture()
def bm25_index(mini_corpus: Sequence[Chunk]) -> BM25Index:
    return BM25Index.from_chunks(mini_corpus, text_for=ChunkStore.text_for)


@pytest.fixture()
def dense_index(
    mini_corpus: Sequence[Chunk], hashing_embedder: HashingEmbedder
) -> ExactVectorIndex:
    return ExactVectorIndex.from_chunks(mini_corpus, hashing_embedder, text_for=ChunkStore.text_for)


@pytest.fixture()
def pipeline(bm25_index: BM25Index, dense_index: ExactVectorIndex) -> RetrievalPipeline:
    return RetrievalPipeline(
        bm25=bm25_index,
        dense=dense_index,
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3),
    )
