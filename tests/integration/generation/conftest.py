"""Shared fixtures for the generation integration tests (offline, deterministic).

# AT-GEN-01  # AT-GEN-09
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest
from support import (
    CARD_FEE_CLAIM,
    CARD_FEE_QUOTE,
    CARD_FEE_TEXT,
    REFUND_TEXT,
    TRANSFER_TEXT,
    draft_json,
)

from answerdesk.config import Settings, load_settings
from answerdesk.contracts import (
    Channel,
    ChatProvider,
    Chunk,
    Document,
    Heading,
    RetrievedPassage,
)
from answerdesk.generate.packer import ContextPacker
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.prompts import PromptRegistry
from answerdesk.testing.doubles import FakeProvider


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient proxy variables must not leak into a hermetic test (INV-08)."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        if os.environ.get(name):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def make_doc() -> Callable[..., Document]:
    def _make(doc_id: str = "doc-card-fee", title: str = "Тарифы по карте «Вектор»") -> Document:
        return Document(
            doc_id=doc_id,
            title=title,
            source_path=f"{doc_id}.md",
            source_format="md",
            text=CARD_FEE_TEXT,
            language="ru",
            license="CC0-1.0",
            sha256="0" * 64,
            byte_size=len(CARD_FEE_TEXT.encode("utf-8")),
            headings=[Heading(level=2, text="Обслуживание", char_offset=0)],
        )

    return _make


@pytest.fixture()
def make_chunk() -> Callable[..., Chunk]:
    def _make(
        chunk_id: str = "chunk-card-fee",
        *,
        text: str = CARD_FEE_TEXT,
        doc_id: str = "doc-card-fee",
        title: str = "Тарифы по карте «Вектор»",
        heading: str = "Обслуживание",
        language: str = "ru",
        **overrides: Any,
    ) -> Chunk:
        payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "heading_path": [title, heading],
            "breadcrumb": f"{title} > {heading}",
            "start_char": 0,
            "end_char": len(text),
            "token_count": max(1, len(text.split())),
            "language": language,
            "metadata": {"doc_title": title},
        }
        payload.update(overrides)
        return Chunk.model_validate(payload)

    return _make


@pytest.fixture()
def make_passage(make_chunk: Callable[..., Chunk]) -> Callable[..., RetrievedPassage]:
    def _make(
        chunk: Chunk | None = None,
        *,
        score: float = 0.9,
        rank: int = 1,
        **chunk_kwargs: Any,
    ) -> RetrievedPassage:
        resolved = chunk if chunk is not None else make_chunk(**chunk_kwargs)
        return RetrievedPassage(chunk=resolved, channel=Channel.DENSE, score=score, rank=rank)

    return _make


@pytest.fixture()
def passages(make_passage: Callable[..., RetrievedPassage]) -> list[RetrievedPassage]:
    return [
        make_passage(chunk_id="chunk-card-fee", score=0.94, rank=1),
        make_passage(
            chunk_id="chunk-transfer",
            score=0.71,
            rank=2,
            text=TRANSFER_TEXT,
            title="Лимиты переводов",
            heading="Суточные лимиты",
            doc_id="doc-transfer",
        ),
        make_passage(
            chunk_id="chunk-refund",
            score=0.42,
            rank=3,
            text=REFUND_TEXT,
            title="Refund policy",
            heading="Refunds",
            doc_id="doc-refund",
            language="en",
        ),
    ]


@pytest.fixture()
def answer_draft() -> str:
    return draft_json([(CARD_FEE_CLAIM, "p1", CARD_FEE_QUOTE)])


@pytest.fixture()
def registry() -> PromptRegistry:
    return PromptRegistry.load()


@pytest.fixture()
def settings() -> Settings:
    return load_settings(profile="offline")


@pytest.fixture()
def pipeline_factory(registry: PromptRegistry, settings: Settings) -> Callable[..., AnswerPipeline]:
    def _make(
        provider: ChatProvider,
        *,
        packer: ContextPacker | None = None,
        repair_passes: int | None = None,
        language_notice_enabled: bool = True,
    ) -> AnswerPipeline:
        return AnswerPipeline(
            provider,
            registry=registry,
            packer=packer or ContextPacker.from_settings(settings),
            repair_passes=(
                settings.generate.repair_passes if repair_passes is None else repair_passes
            ),
            max_answer_tokens=settings.llm.max_tokens,
            language_notice_enabled=language_notice_enabled,
        )

    return _make


@pytest.fixture()
def fake_provider() -> FakeProvider:
    return FakeProvider(responses={})
