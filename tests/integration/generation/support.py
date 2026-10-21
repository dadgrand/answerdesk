"""Helpers shared by the generation integration tests (imported as ``support``).

Kept out of ``conftest.py`` so tests can import them by name without depending on pytest's
conftest loading order.

    # AT-GEN-01  # AT-GEN-09
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from answerdesk.contracts import ChatProvider, Completion, CompletionChunk, Message, ProviderConfig
from answerdesk.testing.doubles import default_fake_provider_config

CARD_FEE_TEXT = (
    "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. "
    "Плата списывается первого числа каждого месяца. "
    "Для карт, оформленных до 1 марта 2025 года, обслуживание бесплатное."
)
TRANSFER_TEXT = (
    "Суточный лимит переводов по карте «Вектор» — 150 000 рублей. "
    "Лимит на один перевод — 50 000 рублей."
)
REFUND_TEXT = (
    "Refunds are processed within 14 days of approval. "
    "The refund is sent to the original payment method."
)

CARD_FEE_CLAIM = "Обслуживание карты «Вектор» стоит 120 рублей в месяц."
CARD_FEE_QUOTE = "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц."
FEE_CHARGE_QUOTE = "Плата списывается первого числа каждого месяца."


def draft_json(
    claims: Sequence[tuple[str, str, str]],
    *,
    insufficient: bool = False,
    why: str = "",
) -> str:
    """Build a model draft: ``(claim text, passage label, quote)`` triples."""
    return json.dumps(
        {
            "insufficient_evidence": insufficient,
            "why": why,
            "claims": [
                {"text": text, "citations": [{"passage": label, "quote": quote}]}
                for text, label, quote in claims
            ],
        },
        ensure_ascii=False,
    )


class ScriptedProvider:
    """Test provider that returns queued replies in order (one per completion call)."""

    name = "scripted"

    def __init__(
        self,
        replies: Sequence[str | Completion],
        *,
        config: ProviderConfig | None = None,
    ) -> None:
        self.replies = list(replies)
        self.config = config or default_fake_provider_config()
        self.calls: list[list[Message]] = []
        self.schemas: list[dict[str, Any] | None] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Any = None,
    ) -> Completion:
        self.calls.append(list(messages))
        self.schemas.append(dict(response_schema) if response_schema is not None else None)
        if not self.replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Completion):
            return reply
        return Completion(
            text=reply,
            model=self.config.model,
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=len(reply.split()),
        )

    def stream(self, messages: Sequence[Message]) -> Sequence[CompletionChunk]:
        completion = self.complete(messages)
        return [
            CompletionChunk(delta=completion.text, done=False, model=completion.model),
            CompletionChunk(delta="", done=True, model=completion.model),
        ]

    @property
    def call_count(self) -> int:
        return len(self.calls)


def as_chat_provider(provider: ScriptedProvider) -> ChatProvider:
    """Type-level assertion helper: the double satisfies the frozen seam."""
    return provider


__all__ = [
    "CARD_FEE_CLAIM",
    "CARD_FEE_QUOTE",
    "CARD_FEE_TEXT",
    "FEE_CHARGE_QUOTE",
    "REFUND_TEXT",
    "TRANSFER_TEXT",
    "ScriptedProvider",
    "as_chat_provider",
    "draft_json",
]
