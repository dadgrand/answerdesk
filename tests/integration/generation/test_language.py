"""Language contract: detect the six locales, answer in the question's language, notice when
the model does not comply (FR-GEN-06).

    # AT-GEN-06
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from support import REFUND_TEXT, ScriptedProvider, draft_json

from answerdesk.contracts import RetrievedPassage
from answerdesk.generate.language import (
    LANGUAGE_NOTICES,
    LOCALE_NAMES,
    SUPPORTED_LOCALES,
    detect_language,
    language_name,
    language_notice,
    languages_agree,
    normalise_apostrophes,
    text_language,
)
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.prompts import PromptRegistry

LOCALE_QUERIES = {
    "ru": "Сколько стоит обслуживание карты?",
    "tg": "Бозгардонии пардохт чӣ қадар вақт мегирад?",
    "uz": "To‘lovni qaytarish qancha vaqt oladi?",
    "ky": "Төлөмдү кайтаруу канча убакыт алат?",
    "tr": "İade ne kadar sürede tamamlanır?",
    "zh": "退款需要多长时间？",
    "en": "How long does identity verification take?",
}


@pytest.mark.parametrize(("locale", "query"), sorted(LOCALE_QUERIES.items()))
def test_detection_for_every_supported_locale(locale: str, query: str) -> None:
    """# AT-GEN-06 — the detector is deterministic and covers the six locales plus English."""
    assert detect_language(query) == locale
    assert detect_language(query) in SUPPORTED_LOCALES


def test_undecidable_text_does_not_claim_a_language() -> None:
    """A single word or a number string is not evidence of a language."""
    assert text_language("") is None
    assert text_language("   ") is None
    assert text_language("12345") is None
    assert detect_language("12345") == "en"  # documented fallback, never a crash


def test_apostrophe_variants_are_normalised() -> None:
    assert normalise_apostrophes("to\u2018lov") == "to'lov"
    assert normalise_apostrophes("o\u02bbzbek") == "o'zbek"
    assert detect_language("o'zbek tilida qanday?") == "uz"


def test_locale_names_and_notices_cover_every_locale() -> None:
    for locale in SUPPORTED_LOCALES:
        assert LOCALE_NAMES[locale]
        assert LANGUAGE_NOTICES[locale]
        assert language_name(locale) == LOCALE_NAMES[locale]
        assert language_notice(locale) == LANGUAGE_NOTICES[locale]
    assert language_name("xx") == "English"


def test_language_agreement_is_symmetric_on_undecidable_text() -> None:
    assert languages_agree("ru", "Обслуживание стоит 120 рублей.") is True
    assert languages_agree("ru", "Refunds are processed within 14 days.") is False
    assert languages_agree("ru", "42") is True


@pytest.mark.parametrize(("locale", "query"), sorted(LOCALE_QUERIES.items()))
def test_prompt_carries_the_locale(locale: str, query: str, registry: PromptRegistry) -> None:
    """# AT-GEN-06 — the prompt states the language and its name for every locale."""
    prompt = registry.get("grounded-answer")
    user = prompt.render_user(
        question=query,
        language=locale,
        language_name=language_name(locale),
        evidence="",
    )
    system = prompt.render_system(language=locale, language_name=language_name(locale))
    assert language_name(locale) in user
    assert f'"{locale}"' in system
    assert "language" in system.lower()


def test_answer_language_equals_query_language(
    pipeline_factory: Callable[..., AnswerPipeline],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-06 — a Russian question gets `language=ru`, not whatever the model chose."""
    passage = make_passage(
        text="Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
        doc_id="doc-fee",
        title="Тарифы",
    )
    draft = draft_json(
        [
            (
                "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
                "p1",
                "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц.",
            )
        ]
    )
    pipeline = pipeline_factory(ScriptedProvider([draft]))
    result = pipeline.answer("Сколько стоит обслуживание карты?", [passage])
    assert result.language == "ru"
    assert result.language_notice is False


def test_wrong_language_answer_gets_a_visible_notice(
    pipeline_factory: Callable[..., AnswerPipeline],
    make_passage: Callable[..., RetrievedPassage],
) -> None:
    """# AT-GEN-06 — a non-complying model produces a marked answer, not a silent one."""
    passage = make_passage(
        text=REFUND_TEXT,
        doc_id="doc-refund",
        title="Refund policy",
        heading="Refunds",
        language="en",
    )
    draft = draft_json(
        [
            (
                "Refunds are processed within 14 days of approval.",
                "p1",
                "Refunds are processed within 14 days of approval.",
            )
        ]
    )
    pipeline = pipeline_factory(ScriptedProvider([draft]))
    result = pipeline.answer("Сколько стоит возврат?", [passage])

    assert result.language == "ru"
    assert result.signal.is_answer is True
    assert result.language_notice is True
    assert language_notice("ru") in result.text
    assert result.text.index(language_notice("ru")) > result.text.index("Refunds")
