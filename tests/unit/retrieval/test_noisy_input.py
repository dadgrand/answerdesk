"""Noisy-input handling: segmentation, transliteration, bounded fuzzy matching, owner cases.

The three messages the owner actually typed are the specimen, and they are asserted verbatim:

    "как займ оформит"                                     telegraphic
    "деньги как получит"                                   telegraphic, mangled order
    "денги первод таджикистан сделал денги нет брату не дошли"   run-on, typos, two intents

    # AT-RET-06  # AT-RET-09
"""

from __future__ import annotations

import time

import pytest

from answerdesk.index.text import TermVocabulary, tokenize
from answerdesk.retrieval.noisy import (
    MAX_FUZZY_CANDIDATES,
    FuzzyMatcher,
    detect_noise_style,
    looks_transliterated,
    normalize_for_matching,
    segment_clauses,
    to_cyrillic,
    transliteration_candidates,
)
from answerdesk.retrieval.query_analysis import (
    QueryAnalysisConfig,
    QueryAnalyzer,
)

OWNER_RUN_ON = "денги первод таджикистан сделал денги нет брату не дошли"
OWNER_TELEGRAPHIC = ("как займ оформит", "деньги как получит")

#: A vocabulary that contains the words the owner's messages *mean*.
VOCAB = TermVocabulary(
    [
        tokenize("перевод перевода деньги займ кредит оформить оформит получить таджикистан"),
        tokenize("отслеживание статус розыск заявка возврат комиссия"),
    ]
)


# -- segmentation ------------------------------------------------------------------


def test_run_on_message_is_split_at_the_negation_restart() -> None:
    clauses = segment_clauses(OWNER_RUN_ON, locale="ru")
    assert len(clauses) == 2
    assert clauses[0].startswith("денги первод таджикистан")
    assert clauses[1].startswith("нет")  # the negation starts the second clause
    assert " ".join(clauses).split() == OWNER_RUN_ON.split()


def test_punctuation_is_the_first_boundary() -> None:
    assert segment_clauses("перевод не дошёл, брату не пришло", locale="ru") == [
        "перевод не дошёл",
        "брату не пришло",
    ]


def test_a_normally_punctuated_question_is_not_shredded() -> None:
    for text in (
        "перевод и его сроки",
        "Как оформить займ?",
        "перевод не дошёл",
        "сколько стоит обслуживание карты",
    ):
        assert segment_clauses(text, locale="ru") == [text]


def test_conjunction_boundary_requires_words_on_both_sides() -> None:
    assert segment_clauses("я отправил перевод а деньги не дошли", locale="ru") == [
        "я отправил перевод",
        "деньги не дошли",
    ]
    assert segment_clauses("деньги и перевод", locale="ru") == ["деньги и перевод"]


def test_long_mixed_intent_message_is_bounded_and_complete() -> None:
    text = "отправил перевод маме в таджикистан деньги не дошли а ещё хочу оформить займ на карту"
    clauses = segment_clauses(text, locale="ru", max_clauses=3)
    assert 1 < len(clauses) <= 3
    # Only clause conjunctions are dropped; every content word survives the split.
    assert set(text.split()) - {"а", "и", "но"} <= set(" ".join(clauses).split())


def test_empty_text_is_returned_unchanged() -> None:
    assert segment_clauses("   ", locale="ru") == ["   "]


# -- transliteration ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("latin", "cyrillic"),
    [
        ("zaym", "займ"),
        ("perevod", "перевод"),
        ("dengi", "денги"),
        ("tadzhikistan", "таджикистан"),
        ("kak", "как"),
    ],
)
def test_latin_terms_are_transliterated(latin: str, cyrillic: str) -> None:
    assert to_cyrillic(latin) == cyrillic


def test_transliteration_is_a_total_function() -> None:
    for text in ("", "123", "TS-999", "qwerty", "щ"):
        assert isinstance(to_cyrillic(text), str)


def test_transliteration_candidates_prefer_exact_vocabulary_hits() -> None:
    assert transliteration_candidates("perevod", vocabulary=VOCAB.terms) == ["перевод"]


def test_transliteration_falls_through_to_fuzzy_matching() -> None:
    matcher = FuzzyMatcher(VOCAB.terms)
    # "dengi" -> "денги" is not in the vocabulary; the bounded matcher resolves "деньги".
    candidates = transliteration_candidates("dengi", vocabulary=VOCAB.terms, matcher=matcher)
    assert candidates == ["деньги"]


def test_transliterated_detection_only_for_cyrillic_locales() -> None:
    assert looks_transliterated("zaym", locale="ru") is True
    assert looks_transliterated("zaym", locale="tr") is False
    assert looks_transliterated("карта", locale="ru") is False


# -- fuzzy matching ----------------------------------------------------------------


def test_fuzzy_matching_finds_the_typo_trigrams_miss() -> None:
    matcher = FuzzyMatcher(VOCAB.terms)
    assert matcher.match("денги")[0][0] == "деньги"
    assert matcher.match("первод")[0][0] == "перевод"


def test_fuzzy_matching_prefers_the_anchored_candidate() -> None:
    """``первод`` is one edit from ``перевод`` *and* from ``первой``; anchors decide."""
    matcher = FuzzyMatcher(set(VOCAB.terms) | {"первой"})
    ranked = [candidate for candidate, _d, _j in matcher.match("первод", limit=3)]
    assert ranked[0] == "перевод"


def test_fuzzy_matching_refuses_short_or_known_terms() -> None:
    matcher = FuzzyMatcher(VOCAB.terms)
    assert matcher.match("займ") == []  # already in the vocabulary
    assert matcher.match("день") == []  # too short to be a typo of anything


def test_fuzzy_matching_respects_the_limit_and_is_deterministic() -> None:
    matcher = FuzzyMatcher(VOCAB.terms)
    first = matcher.match("деньг", limit=1)
    second = matcher.match("деньг", limit=1)
    assert first == second
    assert len(first) <= 1


def test_fuzzy_matching_cost_is_bounded() -> None:
    """A telegraphic message must not cost milliseconds per term (§15.3 latency note)."""
    matcher = FuzzyMatcher(VOCAB.terms)
    terms = ["денги", "первод", "таджикистан", "сделал", "брату", "дошли", "нету", "денег"]
    started = time.perf_counter()
    for term in terms:
        matcher.match(term)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 5.0, f"fuzzy matching took {elapsed_ms:.2f} ms for {len(terms)} terms"
    assert MAX_FUZZY_CANDIDATES <= 32  # the pool bound is the reason the cost is bounded


# -- the analyser ------------------------------------------------------------------


def test_analyzer_segments_the_owner_run_on_message() -> None:
    analyzer = QueryAnalyzer(VOCAB)
    analysis = analyzer.analyze(OWNER_RUN_ON)
    assert analysis.is_run_on is True
    assert len(analysis.clauses) == 2
    assert analysis.language == "ru"


def test_analyzer_builds_a_content_only_view_for_telegraphic_input() -> None:
    analyzer = QueryAnalyzer(VOCAB)
    analysis = analyzer.analyze("деньги как получит")
    assert analysis.is_telegraphic is True
    assert "как" not in analysis.focus_terms
    assert "деньги" in analysis.focus_terms


def test_expansion_never_drops_the_original_tokens() -> None:
    analyzer = QueryAnalyzer(VOCAB)
    analysis = analyzer.analyze(OWNER_RUN_ON)
    # ``primary_terms`` de-duplicates (BM25 keeps repetition in ``base_terms``); the invariant
    # is that every distinct user token survives, in first-occurrence order.
    distinct_base = list(dict.fromkeys(analysis.base_terms))
    assert analysis.primary_terms[: len(distinct_base)] == distinct_base
    assert set(analysis.primary_terms) <= set(analysis.lexical_terms)
    assert set(analysis.expansion_terms) <= set(analysis.lexical_terms)
    assert all(term in analysis.lexical_terms for term in analysis.base_terms)


def test_typos_produce_expansion_terms_without_replacing_the_typo() -> None:
    analysis = QueryAnalyzer(VOCAB).analyze("денги первод не дошли")
    assert "денги" in analysis.primary_terms
    assert "деньги" in analysis.expansion_terms
    assert "перевод" in analysis.expansion_terms


def test_identifiers_in_a_noisy_message_stay_protected() -> None:
    analysis = QueryAnalyzer(VOCAB).analyze(
        "денги первод ORD-2026-10009 5 марта 2026 брату не дошли"
    )
    assert "ORD-2026-10009" in analysis.identifier_tokens
    assert "2026-03-05" in analysis.identifier_tokens
    for token in ("ORD-2026-10009", "2026-03-05"):
        assert token.lower() in analysis.primary_terms


def test_telegraphic_detection_needs_a_function_word() -> None:
    analyzer = QueryAnalyzer(VOCAB)
    assert analyzer.analyze("займ оформить").is_telegraphic is False
    assert analyzer.analyze("как займ оформит").is_telegraphic is True


def test_noise_profile_classifies_the_owner_styles() -> None:
    matcher = FuzzyMatcher(VOCAB.terms)
    assert (
        detect_noise_style(
            OWNER_RUN_ON,
            locale="ru",
            terms=tokenize(OWNER_RUN_ON),
            vocabulary=VOCAB.terms,
            matcher=matcher,
        ).style
        == "run_on"
    )
    assert (
        detect_noise_style(
            "zaym kak oformit",
            locale="ru",
            terms=tokenize("zaym kak oformit"),
            vocabulary=VOCAB.terms,
            matcher=matcher,
        ).style
        == "translit"
    )


def test_noise_profile_is_clean_for_a_normal_question() -> None:
    text = "как оформить возврат средств на карту"
    profile = detect_noise_style(text, locale="ru", terms=tokenize(text), vocabulary=VOCAB.terms)
    assert profile.style == "clean"
    assert normalize_for_matching(text) == text.lower()


def test_noisy_analysis_is_deterministic() -> None:
    analyzer = QueryAnalyzer(VOCAB)
    assert (
        analyzer.analyze(OWNER_RUN_ON).model_dump() == analyzer.analyze(OWNER_RUN_ON).model_dump()
    )


def test_stages_can_be_switched_off_individually() -> None:
    baseline = QueryAnalyzer(
        VOCAB,
        config=QueryAnalysisConfig(
            clause_segmentation=False,
            focus_probe=False,
            fuzzy_edit_distance=False,
            transliteration=False,
        ),
    )
    analysis = baseline.analyze(OWNER_RUN_ON)
    assert analysis.clauses == [analysis.normalized]
    assert analysis.focus_terms == []
    assert analysis.expansion_terms == []


@pytest.mark.parametrize("text", OWNER_TELEGRAPHIC)
def test_owner_telegraphic_cases_keep_a_retrievable_primary(text: str) -> None:
    analysis = QueryAnalyzer(VOCAB).analyze(text)
    assert analysis.primary_terms
    assert analysis.is_empty is False
    assert any(term in VOCAB for term in analysis.primary_terms)
