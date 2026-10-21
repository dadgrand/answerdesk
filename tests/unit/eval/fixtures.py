"""Tiny-corpus builders shared by the eval tests (imported by name, not via conftest)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from answerdesk.eval.datasets import GoldenConfig

LOCALES = ("ru", "tg", "uz", "ky", "tr", "zh")

#: One paragraph, one FAQ question and its answer per locale. The text only has to be
#: recognisable by the language detector and to share content words with the question —
#: these are harness fixtures, not a corpus.
DOC_TEMPLATE: dict[str, tuple[str, str, str]] = {
    "ru": (
        "Блокировка карты занимает 30 дней. Проверить статус можно по номеру ORD-2026-10001.",
        "Как заблокировать карту?",
        "Откройте приложение и заблокируйте карту, это занимает 30 дней.",
    ),
    "tg": (
        "Баста карта 30 рӯз мегирад. Ҳолатро аз рӯи рақами ORD-2026-10002 тафтиш кунед.",
        "Картаро чӣ тавр баста кунам?",
        "Барномаро кушоед ва картаро баста кунед, ин 30 рӯз мегирад.",
    ),
    "uz": (
        "Kartani bloklash 30 kun oladi. Holatni ORD-2026-10003 raqami bilan tekshiring.",
        "Kartani qanday bloklayman?",
        "Ilovani oching va kartani bloklang, bu 30 kun oladi.",
    ),
    "ky": (
        "Картаны блоктоо 30 күн алат. Абалды ORD-2026-10004 номери менен текшериңиз.",
        "Картаны кантип блоктойм?",
        "Колдонмодо картаны блоктоңуз, бул 30 күн алат.",
    ),
    "tr": (
        "Kart blokajı 30 gün sürer. Durumu ORD-2026-10005 numarasından kontrol edin.",
        "Kartımı nasıl bloke ederim?",
        "Uygulamayı açın ve kartı bloklayın, bu 30 gün sürer.",
    ),
    "zh": (
        "卡片封锁需要 30 天。请用 ORD-2026-10006 号码查看状态。",
        "如何封锁我的卡片？",
        "请在应用中封锁卡片，这需要 30 天。",
    ),
}


def _chunk(
    *,
    chunk_id: str,
    doc_id: str,
    text: str,
    locale: str,
    ordinal: int,
    block_kinds: Sequence[str],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "heading_path": [f"heading-{ordinal}"],
        "breadcrumb": f"heading-{ordinal}",
        "start_char": 0,
        "end_char": len(text),
        "token_count": len(text.split()),
        "language": locale,
        "context_header": None,
        "canonical_tokens": [],
        "metadata": {
            "block_kinds": list(block_kinds),
            "doc_title": f"{doc_id} title",
            "ordinal": ordinal,
            "license": "CC0-1.0",
            "source_format": "md",
            "span_exact": True,
            "split": False,
        },
    }


def write_tiny_corpus(root: Path, *, docs_per_locale: int = 6) -> Path:
    """Write a minimal but structurally faithful ``data/corpus`` tree."""
    corpus_dir = root / "corpus"
    (corpus_dir / "queries").mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for locale in LOCALES:
        paragraph, question, answer = DOC_TEMPLATE[locale]
        for index in range(docs_per_locale):
            doc_id = f"doc-{locale}-{index:02d}"
            base = index * 10
            chunks.append(
                _chunk(
                    chunk_id=f"{doc_id}::c0000",
                    doc_id=doc_id,
                    text=f"{paragraph} Запись {base}.",
                    locale=locale,
                    ordinal=0,
                    block_kinds=["paragraph"],
                )
            )
            chunks.append(
                _chunk(
                    chunk_id=f"{doc_id}::c0001",
                    doc_id=doc_id,
                    text=f"Q: {question} #{index}\n\nA: {answer}",
                    locale=locale,
                    ordinal=1,
                    block_kinds=["faq_question"],
                )
            )
            chunks.append(
                _chunk(
                    chunk_id=f"{doc_id}::c0002",
                    doc_id=doc_id,
                    text=f"Q: {question} {index}\n\nA: {answer} ({base + 5})",
                    locale=locale,
                    ordinal=2,
                    block_kinds=["faq_question"],
                )
            )
            queries.append(
                {
                    "query_id": f"{doc_id}-q0",
                    "locale": locale,
                    "intent": "cards",
                    "doc_id": doc_id,
                    "text": f"{question} #{index}",
                    "base_text": f"{question} #{index}",
                    "category": "direct",
                    "perturbations": [],
                    "operator_reply": answer,
                }
            )
            queries.append(
                {
                    "query_id": f"{doc_id}-q1",
                    "locale": locale,
                    "intent": "tariffs",
                    "doc_id": doc_id,
                    "text": f"{question} {index}",
                    "base_text": f"{question} {index}",
                    "category": "numeric_date",
                    "perturbations": ["numeric_date"],
                    "operator_reply": f"{answer} ({base + 5})",
                }
            )
    chunks_path = corpus_dir / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in chunks) + "\n",
        encoding="utf-8",
    )
    (corpus_dir / "queries" / "queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in queries) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "corpus_version": "corpus-tiny-test",
        "generator": {"name": "tiny-fixture", "seed": 7},
        "chunks": len(chunks),
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return corpus_dir


def tiny_config(**overrides: Any) -> GoldenConfig:
    """Generator config with reduced targets that a tiny corpus can satisfy."""
    base: dict[str, Any] = {
        "locales": list(LOCALES),
        "test_target_per_locale": 12,
        "dev_target_per_locale": 8,
        "injection_per_locale": 2,
        "pii_per_locale": 2,
        "near_miss_per_locale": 2,
        "claims_per_locale": 4,
        "seed": 424242,
    }
    base.update(overrides)
    return GoldenConfig(**base)
