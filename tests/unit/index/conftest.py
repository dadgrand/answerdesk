"""Shared fixtures for the index tests: a tiny multilingual chunk set and both indexes.

The chunk set is deliberately hand-written rather than generated: these tests are about
tokenisation, ranking determinism and exact-nearest-neighbour equivalence, so every fixture
string exists to prove one property (an identifier, a date, a CJK run, a table row).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from answerdesk.contracts import Chunk
from answerdesk.index.bm25_index import BM25Index
from answerdesk.index.store import ChunkStore
from answerdesk.index.vector_index import ExactVectorIndex
from answerdesk.testing.doubles import HashingEmbedder

#: (chunk_id, doc_id, breadcrumb, text) — the fixture corpus of every index test.
FIXTURE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ru-card::c0000",
        "ru-card",
        "Карты > Обслуживание",
        "Обслуживание карты «Базовая» стоит 1200 ₽ в год. Списание 5 марта 2026.",
    ),
    (
        "ru-err::c0000",
        "ru-err",
        "Поддержка > Ошибки",
        "Ошибка TS-999 означает блокировку карты. Позвоните в поддержку.",
    ),
    (
        "zh-refund::c0000",
        "zh-refund",
        "退款 > 处理",
        "退款将在14天内处理。订单号 ORD-2026-000124。",
    ),
    (
        "tr-renew::c0000",
        "tr-renew",
        "Kartlar > Yenileme",
        "Kart yenileme ücreti 250 TRY ve 10 gün içinde tamamlanır.",
    ),
    (
        "uz-card::c0000",
        "uz-card",
        "Kartalar > Qayta chiqarish",
        "Kartani qayta chiqarish 50 000 UZS turadi.",
    ),
    (
        "ky-card::c0000",
        "ky-card",
        "Карталар > Кайра чыгаруу",
        "Картаны кайра чыгаруу 300 сом турат жана 7 күн алат.",
    ),
    (
        "tg-card::c0000",
        "tg-card",
        "Кортҳо > Баровардан",
        "Баровардани корт 100 сомони аст ва 10 рӯз мегирад.",
    ),
    (
        "en-err::c0000",
        "en-err",
        "Support > Errors",
        "Error code TS-999 blocks the card until KYC is complete.",
    ),
    (
        "ru-table::c0000",
        "ru-table",
        "Тарифы > Таблица",
        "| Тариф | Цена |\n|---|---|\n| Базовая | 1200 ₽ |\n| Премиум | 3000 ₽ |",
    ),
    (
        "ru-card::c0001",
        "ru-card",
        "Карты > Обслуживание",
        "Перевыпуск карты занимает 5 рабочих дней и стоит 300 ₽.",
    ),
)


@pytest.fixture()
def fixture_chunks(make_chunk: Callable[..., Chunk]) -> list[Chunk]:
    """The fixture corpus as validated :class:`answerdesk.contracts.Chunk` objects."""
    chunks: list[Chunk] = []
    for chunk_id, doc_id, breadcrumb, text in FIXTURE_ROWS:
        heading_path = breadcrumb.split(" > ")
        chunks.append(
            make_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                heading_path=heading_path,
                breadcrumb=breadcrumb,
                end_char=len(text),
                token_count=len(text.split()),
            )
        )
    return chunks


@pytest.fixture()
def fixture_store(fixture_chunks: Sequence[Chunk]) -> ChunkStore:
    return ChunkStore.from_chunks(fixture_chunks, corpus_version="corpus-test")


@pytest.fixture()
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=32)


@pytest.fixture()
def bm25_index(fixture_chunks: Sequence[Chunk]) -> BM25Index:
    return BM25Index.from_chunks(fixture_chunks, text_for=ChunkStore.text_for)


@pytest.fixture()
def vector_index(
    fixture_chunks: Sequence[Chunk], hashing_embedder: HashingEmbedder
) -> ExactVectorIndex:
    return ExactVectorIndex.from_chunks(
        fixture_chunks, hashing_embedder, text_for=ChunkStore.text_for
    )
