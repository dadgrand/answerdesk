"""The optional LLM rewrite hook: off by default, cached, deterministic, identifier-guarded.

Driven by :class:`answerdesk.testing.doubles.FakeProvider`, so the whole file runs offline.

    # AT-RET-06  # AT-RET-09
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from answerdesk.config import ProviderConfig, load_settings
from answerdesk.contracts import Message
from answerdesk.retrieval.query_analysis import QueryAnalysisConfig, QueryAnalyzer
from answerdesk.retrieval.rewrite import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    CachingRewriter,
    LlmQueryRewriter,
    build_rewriter,
    validate_rewrite,
)
from answerdesk.testing.doubles import FakeProvider


class RecordingProvider(FakeProvider):
    """FakeProvider that records calls and can be told what to answer."""

    def __init__(self, reply: str | None = None, *, error: Exception | None = None) -> None:
        super().__init__(config=_fake_config())
        self.reply = reply
        self.error = error
        self.seen: list[Sequence[Message]] = []

    def complete(self, messages, *, response_schema=None):  # type: ignore[override]
        self.seen.append(list(messages))
        if self.error is not None:
            raise self.error
        if self.reply is not None:
            from answerdesk.contracts import Completion

            return Completion(text=self.reply, model="fake-model")
        return super().complete(messages, response_schema=response_schema)


def _fake_config() -> ProviderConfig:
    return ProviderConfig(
        preset="fake", base_url="memory://fake", model="fake-model", requires_api_key=False
    )


# -- validation --------------------------------------------------------------------


def test_a_faithful_rewrite_is_accepted() -> None:
    assert validate_rewrite("как оформить займ", "оформление займа условия") == (
        "оформление займа условия"
    )


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        "",
        "   ",
        "q" * 400,  # chatter, not a query
        "перевод на счёт 5555 5555 5555 4444",  # invents a card number
        "перевод ORD-2026-99999",  # invents an order id
        "перевод 2027-01-01",  # invents a date
    ],
)
def test_unsafe_rewrites_are_rejected(candidate: str | None) -> None:
    assert validate_rewrite("перевод не дошёл ORD-2026-11111", candidate) is None


def test_a_rewrite_may_not_drop_an_identifier() -> None:
    assert validate_rewrite("перевод ORD-2026-11111 не дошёл", "перевод не дошёл") is None


def test_a_rewrite_keeps_identifiers_verbatim() -> None:
    confirmed = validate_rewrite("перевод ORD-2026-11111", "статус перевода ORD-2026-11111")
    assert confirmed == "статус перевода ORD-2026-11111"


def test_quotes_and_prefixes_are_stripped() -> None:
    assert validate_rewrite("займ", '"оформить займ"') == "оформить займ"
    assert validate_rewrite("займ", "query: оформить займ") == "оформить займ"


# -- the hook ----------------------------------------------------------------------


def test_rewriter_sends_the_instruction_and_the_query() -> None:
    provider = RecordingProvider(reply="оформить займ условия")
    rewriter = LlmQueryRewriter(provider)
    assert rewriter.rewrite("как займ оформит") == "оформить займ условия"
    sent = provider.seen[0]
    assert sent[0].content == SYSTEM_PROMPT
    assert sent[1].content == "как займ оформит"
    assert rewriter.name.startswith("llm-rewrite:")
    assert PROMPT_VERSION in rewriter.name


def test_cached_rewriter_calls_the_model_once_per_distinct_query() -> None:
    provider = RecordingProvider(reply="оформить займ")
    cached = CachingRewriter(LlmQueryRewriter(provider))
    assert cached.rewrite("как займ оформит") == "оформить займ"
    assert cached.rewrite("как займ оформит") == "оформить займ"
    assert cached.rewrite("  как займ оформит  ") == "оформить займ"
    assert len(provider.seen) == 1
    assert (cached.calls, cached.hits) == (1, 2)


def test_cache_is_bounded_and_evicts_the_oldest_entry() -> None:
    provider = RecordingProvider(reply="статус перевода")
    cached = CachingRewriter(LlmQueryRewriter(provider), max_entries=2)
    for query in ("q1", "q2", "q3"):
        cached.rewrite(query)
    assert len(cached) == 2
    assert len(provider.seen) == 3


def test_build_rewriter_is_off_by_default() -> None:
    assert build_rewriter(RecordingProvider(reply="x")) is None
    assert build_rewriter(None, enabled=True) is None


def test_build_rewriter_wraps_in_a_cache_when_enabled() -> None:
    rewriter = build_rewriter(RecordingProvider(reply="оформить займ"), enabled=True)
    assert isinstance(rewriter, CachingRewriter)


# -- wiring into the analyser ------------------------------------------------------


def _analyzer(rewriter=None, *, rewrite: bool = True) -> QueryAnalyzer:
    return QueryAnalyzer(None, config=QueryAnalysisConfig(rewrite=rewrite), rewriter=rewriter)


def test_analyzer_uses_the_rewrite_as_an_extra_probe() -> None:
    provider = RecordingProvider(reply="оформить займ условия")
    analyzer = _analyzer(CachingRewriter(LlmQueryRewriter(provider)))
    analysis = analyzer.analyze("как займ оформит")
    assert analysis.rewrite == "оформить займ условия"
    assert "rewrite probe added" in analysis.notes


def test_analyzer_ignores_the_rewriter_when_the_config_says_off() -> None:
    provider = RecordingProvider(reply="оформить займ")
    analyzer = _analyzer(CachingRewriter(LlmQueryRewriter(provider)), rewrite=False)
    analysis = analyzer.analyze("как займ оформит")
    assert analysis.rewrite is None
    assert provider.seen == []


def test_analyzer_records_a_rejected_rewrite_instead_of_failing() -> None:
    provider = RecordingProvider(reply="перевод ORD-2026-99999")
    analyzer = _analyzer(CachingRewriter(LlmQueryRewriter(provider)))
    analysis = analyzer.analyze("перевод ORD-2026-11111")
    assert analysis.rewrite is None
    assert any("rewrite rejected" in note for note in analysis.notes)


def test_a_provider_failure_never_breaks_analysis() -> None:
    provider = RecordingProvider(error=RuntimeError("provider down"))
    analyzer = _analyzer(CachingRewriter(LlmQueryRewriter(provider)))
    analysis = analyzer.analyze("как займ оформит")
    assert analysis.rewrite is None
    assert any("rewrite unavailable" in note for note in analysis.notes)
    assert analysis.primary_terms  # retrieval still works


def test_rewrite_is_deterministic_for_the_same_provider() -> None:
    provider = RecordingProvider(reply="оформить займ")
    rewriter = CachingRewriter(LlmQueryRewriter(provider))
    first = _analyzer(rewriter).analyze("как займ оформит")
    second = _analyzer(rewriter).analyze("как займ оформит")
    assert first.rewrite == second.rewrite


def test_the_shipped_config_keeps_the_rewrite_off() -> None:
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[3] / "configs" / "retrieval" / "noisy.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["rewrite"]["enabled"] is False
    assert payload["pipeline"]["focus_probe"] is True
    assert payload["pipeline"]["clause_probes"] is False
    assert payload["pipeline"]["expansion_probe"] is False
    assert load_settings(profile="offline").embedding.backend == "hashing"
