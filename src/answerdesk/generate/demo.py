"""The offline demo evidence pack and its deterministic model double.

``answerdesk ask --fake`` must work from a clean checkout with no corpus, no network and
no key (CMD-H, FR-GEN-09/10). The retrieval stack that would normally supply passages is
owned by t5; until the service wires it (t9/t14), the offline path answers from this
checked-in synthetic pack: six short support documents in the six locales, with a
hand-written model draft per question.

Everything here is synthetic and reserves no real identifiers (requirements §13.4):
product names are invented, prices are illustrative, and every price in a draft quotes its
passage verbatim — which is exactly what the citation validator checks.

``ask`` chooses a case by keyword overlap; a question that matches nothing yields an empty
passage list, so the pipeline escalates with ``NO_EVIDENCE`` instead of inventing an answer.

    # AT-GEN-09  # AT-GEN-10  # AT-OPS-03
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import Field

from answerdesk.contracts import Channel, Chunk, Contract, RetrievedPassage


class DemoCase(Contract):
    """One demo question: its evidence, its keyword triggers and the canned model draft."""

    case_id: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    question: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    chunk: Chunk
    score: float = 0.9
    draft: dict[str, object] = Field(default_factory=dict)


def _chunk(
    chunk_id: str,
    doc_id: str,
    title: str,
    heading: str,
    text: str,
    language: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        heading_path=[title, heading],
        breadcrumb=f"{title} > {heading}",
        start_char=0,
        end_char=len(text),
        token_count=max(1, len(text.split())),
        language=language,
        metadata={"doc_title": title},
    )


_CARD_FEE_RU = (
    "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. "
    "Плата списывается первого числа каждого месяца. "
    "Для карт, оформленных до 1 марта 2025 года, обслуживание бесплатное."
)

_DEMO_CHUNKS: list[tuple[str, str, str, str, str, str]] = [
    (
        "demo-card-fee-ru",
        "demo-card-fee",
        "Тарифы по карте «Вектор»",
        "Обслуживание",
        _CARD_FEE_RU,
        "ru",
    ),
    (
        "demo-transfer-limits-ru",
        "demo-transfer-limits",
        "Лимиты переводов",
        "Суточные лимиты",
        "Суточный лимит переводов по карте «Вектор» — 150 000 рублей. "
        "Лимит на один перевод — 50 000 рублей.",
        "ru",
    ),
    (
        "demo-refund-tg",
        "demo-refund-tg",
        "Бозгардонии пардохт",
        "Мӯҳлат",
        "Бозгардонии пардохт дар давоми 14 рӯзи корӣ анҷом дода мешавад.",
        "tg",
    ),
    (
        "demo-refund-uz",
        "demo-refund-uz",
        "To‘lovni qaytarish",
        "Muddat",
        "To‘lovni qaytarish 14 ish kuni ichida amalga oshiriladi.",
        "uz",
    ),
    (
        "demo-refund-ky",
        "demo-refund-ky",
        "Төлөмдү кайтаруу",
        "Мөөнөт",
        "Төлөмдү кайтаруу 14 жумуш күнүнүн ичинде жүргүзүлөт.",
        "ky",
    ),
    (
        "demo-refund-tr",
        "demo-refund-tr",
        "İade süreci",
        "Süre",
        "İade işlemi 14 iş günü içinde tamamlanır.",
        "tr",
    ),
    (
        "demo-refund-zh",
        "demo-refund-zh",
        "退款流程",
        "处理时间",
        "退款将在14个工作日内完成。",
        "zh",
    ),
    (
        "demo-kyc-en",
        "demo-kyc-en",
        "KYC requirements",
        "Documents",
        "Identity verification requires a passport and a selfie. "
        "Verification takes up to 2 business days.",
        "en",
    ),
]

_DEMO_CASES: list[DemoCase] = [
    DemoCase(
        case_id="card-fee-ru",
        locale="ru",
        question="Сколько стоит обслуживание карты?",
        keywords=["обслуживание", "карт", "стоит", "тариф", "fee", "card"],
        chunk=_chunk(*_DEMO_CHUNKS[0]),
        score=0.94,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Обслуживание дебетовой карты «Вектор» "
                            "стоит 120 рублей в месяц.",
                        }
                    ],
                },
                {
                    "text": "Плата списывается первого числа каждого месяца.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Плата списывается первого числа каждого месяца.",
                        }
                    ],
                },
            ],
        },
    ),
    DemoCase(
        case_id="transfer-limits-ru",
        locale="ru",
        question="Какой лимит перевода за сутки?",
        keywords=["лимит", "перевод", "сутки", "limit", "transfer"],
        chunk=_chunk(*_DEMO_CHUNKS[1]),
        score=0.91,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Суточный лимит переводов по карте «Вектор» — 150 000 рублей.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Суточный лимит переводов по карте «Вектор» — 150 000 рублей.",
                        }
                    ],
                }
            ],
        },
    ),
    DemoCase(
        case_id="refund-tg",
        locale="tg",
        question="Бозгардонии пардохт чӣ қадар вақт мегирад?",
        keywords=["бозгардонии", "пардохт", "refund", "мегирад"],
        chunk=_chunk(*_DEMO_CHUNKS[2]),
        score=0.88,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Бозгардонии пардохт дар давоми 14 рӯзи корӣ анҷом дода мешавад.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Бозгардонии пардохт дар давоми 14 рӯзи корӣ "
                            "анҷом дода мешавад.",
                        }
                    ],
                }
            ],
        },
    ),
    DemoCase(
        case_id="refund-uz",
        locale="uz",
        question="To‘lovni qaytarish qancha vaqt oladi?",
        keywords=["qaytarish", "to‘lovni", "refund", "vaqt"],
        chunk=_chunk(*_DEMO_CHUNKS[3]),
        score=0.88,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "To‘lovni qaytarish 14 ish kuni ichida amalga oshiriladi.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "To‘lovni qaytarish 14 ish kuni ichida amalga oshiriladi.",
                        }
                    ],
                }
            ],
        },
    ),
    DemoCase(
        case_id="refund-ky",
        locale="ky",
        question="Төлөмдү кайтаруу канча убакыт алат?",
        keywords=["кайтаруу", "төлөмдү", "refund", "убакыт"],
        chunk=_chunk(*_DEMO_CHUNKS[4]),
        score=0.88,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Төлөмдү кайтаруу 14 жумуш күнүнүн ичинде жүргүзүлөт.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Төлөмдү кайтаруу 14 жумуш күнүнүн ичинде жүргүзүлөт.",
                        }
                    ],
                }
            ],
        },
    ),
    DemoCase(
        case_id="refund-tr",
        locale="tr",
        question="İade ne kadar sürede tamamlanır?",
        keywords=["i̇ade", "iade", "süre", "refund", "kadar"],
        chunk=_chunk(*_DEMO_CHUNKS[5]),
        score=0.88,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "İade işlemi 14 iş günü içinde tamamlanır.",
                    "citations": [
                        {"passage": "p1", "quote": "İade işlemi 14 iş günü içinde tamamlanır."}
                    ],
                }
            ],
        },
    ),
    DemoCase(
        case_id="refund-zh",
        locale="zh",
        question="退款需要多长时间？",
        keywords=["退款", "时间", "refund"],
        chunk=_chunk(*_DEMO_CHUNKS[6]),
        score=0.88,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "退款将在14个工作日内完成。",
                    "citations": [{"passage": "p1", "quote": "退款将在14个工作日内完成。"}],
                }
            ],
        },
    ),
    DemoCase(
        case_id="kyc-en",
        locale="en",
        question="How long does identity verification take?",
        keywords=["identity", "verification", "kyc", "long", "document"],
        chunk=_chunk(*_DEMO_CHUNKS[7]),
        score=0.90,
        draft={
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Identity verification takes up to 2 business days.",
                    "citations": [
                        {"passage": "p1", "quote": "Verification takes up to 2 business days."}
                    ],
                },
                {
                    "text": "Identity verification requires a passport and a selfie.",
                    "citations": [
                        {
                            "passage": "p1",
                            "quote": "Identity verification requires a passport and a selfie.",
                        }
                    ],
                },
            ],
        },
    ),
]


def demo_cases() -> list[DemoCase]:
    """All demo cases (a copy: callers must not mutate the module state)."""
    return list(_DEMO_CASES)


def demo_corpus() -> list[Chunk]:
    """The whole synthetic demo corpus, for an operator who wants to see the evidence."""
    return [case.chunk for case in _DEMO_CASES]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE))


def match_case(question: str) -> DemoCase | None:
    """Pick the demo case with the highest keyword overlap (deterministic tie-break)."""
    asked = _words(question)
    if not asked:
        return None
    best: tuple[int, str, DemoCase] | None = None
    for case in _DEMO_CASES:
        keywords = {keyword.lower() for keyword in case.keywords}
        overlap = len(asked & keywords)
        # A single shared word is not evidence of intent; require the best score to be
        # strictly positive and stable across runs (tie-break on case_id).
        if overlap == 0:
            continue
        candidate = (-overlap, case.case_id, case)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best is not None else None


def demo_passages(question: str) -> list[RetrievedPassage]:
    """Passages for a demo question; empty when nothing matches (→ ``NO_EVIDENCE``)."""
    case = match_case(question)
    if case is None:
        return []
    return [
        RetrievedPassage(
            chunk=case.chunk,
            channel=Channel.DENSE,
            score=case.score,
            rank=1,
            stage="retrieval",
        )
    ]


def demo_draft_json(question: str) -> str | None:
    """The canned model reply for a demo question, as JSON text."""
    case = match_case(question)
    return None if case is None else json.dumps(case.draft, ensure_ascii=False)


def demo_responses() -> dict[str, str]:
    """``FakeProvider`` response map: question keyword → JSON draft.

    ``FakeProvider`` matches on a substring of the last user message, and the packed
    evidence (not the question alone) is in that message, so each key is a distinctive
    fragment of the *rendered prompt* for its case: the passage label plus the first
    words of the chunk. That keeps the double deterministic without teaching it anything
    about the pipeline internals.
    """
    responses: dict[str, str] = {}
    for case in _DEMO_CASES:
        fragment = " ".join(case.chunk.text.split())[:48]
        responses[fragment] = json.dumps(case.draft, ensure_ascii=False)
    return responses


def demo_questions() -> Sequence[str]:
    return [case.question for case in _DEMO_CASES]


__all__ = [
    "DemoCase",
    "demo_cases",
    "demo_corpus",
    "demo_draft_json",
    "demo_passages",
    "demo_questions",
    "demo_responses",
    "match_case",
]
