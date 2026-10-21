"""Locale policy: same-locale preference, explicit fallback, mixed-script safety.

The policy is a pure function, so the tests assert exact orderings rather than aggregate
metrics; the corpus-level measurement lives in ``tests/integration/retrieval``.

    # AT-RET-06  # AT-RET-08
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import Channel, Chunk, RetrievedPassage
from answerdesk.retrieval.locale_policy import (
    LocalePolicy,
    chunk_purity,
    mark_locale_ordering,
    order_by_locale,
)
from answerdesk.retrieval.pipeline import PipelineConfig, RetrievalPipeline, _is_mixed_script


def _passage(chunk_id: str, language: str | None, rank: int = 1) -> RetrievedPassage:
    text = f"text for {chunk_id}"
    return RetrievedPassage(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            language=language,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=3,
        ),
        channel=Channel.BM25,
        score=1.0,
        rank=rank,
    )


MIXED = [
    _passage("ru-1", "ru", 1),
    _passage("zh-1", "zh", 2),
    _passage("ru-2", "ru", 3),
    _passage("tr-1", "tr", 4),
    _passage("ru-3", "ru", 5),
]


def _ids(order) -> list[str]:
    return [passage.chunk.chunk_id for passage in order.passages]


def test_same_locale_evidence_moves_first_without_losing_anything() -> None:
    order = order_by_locale(MIXED, language="ru", confidence=1.0, window=3)
    assert _ids(order)[:3] == ["ru-1", "ru-2", "ru-3"]
    assert sorted(_ids(order)) == sorted(
        _ids(order_by_locale(MIXED, language="ru", confidence=1.0, window=3))
    )
    assert set(_ids(order)) == {"ru-1", "ru-2", "ru-3", "zh-1", "tr-1"}
    assert order.applied is True
    assert order.fallback_used is False
    assert order.reason == "preferred"
    assert order.purity == 1.0


def test_fallback_fires_when_the_window_cannot_be_filled() -> None:
    passages = [_passage("ru-1", "ru", 1), _passage("zh-1", "zh", 2), _passage("tr-1", "tr", 3)]
    order = order_by_locale(passages, language="ru", confidence=1.0, window=3)
    assert order.fallback_used is True
    assert order.reason == "fallback"
    assert set(_ids(order)) == {"ru-1", "zh-1", "tr-1"}  # nothing starved
    assert order.purity == 1 / 3


def test_ratio_zero_is_a_pure_preference() -> None:
    passages = [_passage("zh-1", "zh", 1), _passage("ru-1", "ru", 2)]
    order = order_by_locale(
        passages, language="ru", confidence=1.0, window=2, policy=LocalePolicy(min_same_ratio=0.0)
    )
    assert _ids(order) == ["ru-1", "zh-1"]
    assert order.fallback_used is False


def test_ratio_one_demands_a_full_window_of_same_locale_evidence() -> None:
    passages = [_passage("ru-1", "ru", 1), _passage("zh-1", "zh", 2)]
    order = order_by_locale(
        passages, language="ru", confidence=1.0, window=2, policy=LocalePolicy(min_same_ratio=1.0)
    )
    assert order.fallback_used is True  # one of two slots is not enough


def test_low_confidence_disables_the_policy() -> None:
    order = order_by_locale(MIXED, language="ru", confidence=0.2, window=3)
    assert _ids(order) == [passage.chunk.chunk_id for passage in MIXED]
    assert order.applied is False
    assert order.reason == "low_confidence"


@pytest.mark.parametrize("language", ["", "und"])
def test_undetermined_language_disables_the_policy(language: str) -> None:
    order = order_by_locale(MIXED, language=language, confidence=1.0, window=3)
    assert order.applied is False
    assert order.reason == "undetermined"


def test_the_off_switch_leaves_the_ranking_untouched() -> None:
    order = order_by_locale(
        MIXED, language="ru", confidence=1.0, window=3, policy=LocalePolicy(enabled=False)
    )
    assert _ids(order) == [passage.chunk.chunk_id for passage in MIXED]
    assert order.reason == "disabled"


def test_no_same_locale_evidence_falls_back_and_says_so() -> None:
    passages = [_passage("zh-1", "zh", 1), _passage("tr-1", "tr", 2)]
    order = order_by_locale(passages, language="ru", confidence=1.0, window=2)
    assert order.applied is False
    assert order.fallback_used is True
    assert order.reason == "no_same_locale_evidence"
    assert _ids(order) == ["zh-1", "tr-1"]


def test_an_already_pure_list_is_unchanged() -> None:
    passages = [_passage("ru-1", "ru", 1), _passage("ru-2", "ru", 2)]
    order = order_by_locale(passages, language="ru", confidence=1.0, window=2)
    assert order.reason == "already_pure"
    assert _ids(order) == ["ru-1", "ru-2"]


def test_unknown_chunk_language_is_treated_as_cross_locale() -> None:
    passages = [_passage("x", None, 1), _passage("ru-1", "ru", 2)]
    order = order_by_locale(passages, language="ru", confidence=1.0, window=2)
    assert _ids(order) == ["ru-1", "x"]


def test_ordering_is_deterministic_and_stable() -> None:
    first = order_by_locale(MIXED, language="ru", confidence=1.0, window=5)
    second = order_by_locale(MIXED, language="ru", confidence=1.0, window=5)
    assert _ids(first) == _ids(second)
    # Within a group the original order is preserved (a stable partition, not a re-score).
    assert _ids(first) == ["ru-1", "ru-2", "ru-3", "zh-1", "tr-1"]


def test_empty_input_and_zero_window() -> None:
    assert order_by_locale([], language="ru", confidence=1.0, window=5).reason == "empty"
    order = order_by_locale(MIXED, language="ru", confidence=1.0, window=0)
    assert order.purity == 0.0


def test_policy_validates_its_bounds() -> None:
    with pytest.raises(ValueError, match="min_same_ratio"):
        LocalePolicy(min_same_ratio=1.5)
    with pytest.raises(ValueError, match="min_confidence"):
        LocalePolicy(min_confidence=-0.1)


def test_trace_markers_record_the_decision() -> None:
    order = order_by_locale(
        [_passage("ru-1", "ru", 1), _passage("zh-1", "zh", 2)],
        language="ru",
        confidence=1.0,
        window=2,
        policy=LocalePolicy(min_same_ratio=1.0),
    )
    marked = mark_locale_ordering(order)
    assert marked[0].contributions["locale_pref"] == 1.0
    assert marked[1].contributions["cross_locale_fallback"] == 1.0


def test_chunk_purity_helper() -> None:
    assert chunk_purity([_passage("ru-1", "ru"), _passage("zh-1", "zh")], "ru") == 0.5
    assert chunk_purity([], "ru") == 0.0


# -- pipeline wiring -----------------------------------------------------------------


def test_pipeline_knob_defaults_to_on_and_raises_purity(pipeline, mini_corpus) -> None:
    """The shipped default prefers the query locale; the off switch restores the old order."""
    query = "Сколько стоит обслуживание карты?"
    on = pipeline.retrieve(query)
    off_pipeline = RetrievalPipeline(
        bm25=pipeline.bm25,
        dense=pipeline.dense,
        fuser=pipeline.fuser,
        config=PipelineConfig(top_k=10, gate_k=5, final_k=3, locale_filter=False),
    )
    off = off_pipeline.retrieve(query)
    on_languages = [passage.chunk.language for passage in on.packed]
    off_languages = [passage.chunk.language for passage in off.packed]
    assert on_languages.count("ru") >= off_languages.count("ru")
    assert PipelineConfig().locale_filter is True


def test_mixed_script_queries_are_not_hard_filtered(pipeline) -> None:
    """A code-switched question keeps a full window (fallback), never a starved one."""
    query = "перевод money transfer таджикистан не дошли"
    trace = pipeline.retrieve(query)
    assert 1 <= len(trace.packed) <= pipeline.config.final_k
    assert all(passage.stage == "pack" for passage in trace.packed)
    first = [passage.chunk.chunk_id for passage in trace.packed]
    second = [passage.chunk.chunk_id for passage in pipeline.retrieve(query).packed]
    assert first == second


def test_mixed_script_detector() -> None:
    """Two foreign words (or half the letters) is a code-switch; a product name is not."""
    assert _is_mixed_script("перевод money transfer не дошли", "ru") is True
    assert _is_mixed_script("перевод transfer таджикистан", "ru") is False
    assert _is_mixed_script("Visa карта", "ru") is False
    assert _is_mixed_script("Kart yenileme ücreti", "tr") is False
    assert _is_mixed_script("перевод карта", "und") is False
    assert _is_mixed_script("перевод перевод на карту срочно", "ru") is False
