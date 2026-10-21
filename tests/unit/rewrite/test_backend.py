"""The rewriter seam and backend selection: identity by default, graceful degradation.

# AT-GEN-13  # AT-GEN-16
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from answerdesk.rewrite import (
    DeterministicRewriter,
    IdentityRewriter,
    QueryRewriter,
    RewriteConfig,
    RewriteResult,
    TimedRewriter,
    build_rewriter,
    load_rewrite_config,
    rewrite_status,
)
from answerdesk.rewrite.errors import RewriteConfigError


class ExplodingRewriter:
    """Test double: a backend that always raises (the wrapper must catch it)."""

    name = "exploding"

    @property
    def available(self) -> bool:
        return True

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult:
        raise RuntimeError("model crashed")


def test_identity_is_the_default_and_the_fallback_shape() -> None:
    rewriter = build_rewriter(RewriteConfig())
    assert isinstance(rewriter, IdentityRewriter)
    assert isinstance(rewriter, QueryRewriter)
    result = rewriter.rewrite("zaym kak oformit", locale="ru")
    assert isinstance(result, RewriteResult)
    assert result.query == "zaym kak oformit"
    assert result.changed is False
    assert result.latency_ms == 0.0


def test_disabled_config_never_imports_a_model() -> None:
    status = rewrite_status(RewriteConfig(enabled=False, backend="local_lora"))
    assert status.active_backend == "identity"
    assert status.available is True
    assert status.fallback is False
    assert "disabled" in status.reason


def test_deterministic_backend_is_selected_and_wrapped() -> None:
    config = RewriteConfig(enabled=True, backend="deterministic")
    rewriter = build_rewriter(config)
    assert isinstance(rewriter, TimedRewriter)
    assert isinstance(rewriter.inner, DeterministicRewriter)
    result = rewriter.rewrite("подскажите пожалуйста сколько стоит карта", locale="ru")
    assert result.backend == "deterministic"
    assert "пожалуйста" not in result.query
    assert result.latency_ms >= 0.0


def test_deterministic_knobs_are_honoured() -> None:
    quiet = RewriteConfig(
        enabled=True,
        backend="deterministic",
        deterministic={"transliterate": False, "drop_fillers": False, "repair": False},
    )
    result = build_rewriter(quiet).rewrite("zaym kak oformit")
    assert result.query == "zaym kak oformit"
    loud = RewriteConfig(enabled=True, backend="deterministic")
    assert build_rewriter(loud).rewrite("zaym kak oformit").query == "займ как оформить"


def test_missing_adapter_degrades_to_identity_with_a_reason(tmp_path: Path) -> None:
    config = RewriteConfig(
        enabled=True,
        backend="local_lora",
        local_lora={"adapter_path": str(tmp_path / "nope")},
    )
    rewriter = build_rewriter(config)
    assert isinstance(rewriter, IdentityRewriter)
    status = rewrite_status(config)
    assert status.active_backend == "identity"
    assert status.fallback is True
    assert status.available is False
    assert "adapter" in status.reason
    assert status.adapter_path.endswith("nope")


def test_status_reports_the_configured_backend_verbatim(tmp_path: Path) -> None:
    config = RewriteConfig(
        enabled=True,
        backend="local_lora",
        local_lora={"adapter_path": str(tmp_path), "base_model": "Qwen/Qwen2.5-1.5B-Instruct"},
    )
    payload = rewrite_status(config).as_dict()
    assert payload["configured_backend"] == "local_lora"
    assert payload["base_model"].startswith("Qwen/")
    assert isinstance(payload["fallback"], bool)


def test_timed_rewriter_swallows_a_crashing_backend() -> None:
    """A rewriter must never break the request path: the original query is returned."""
    wrapped = TimedRewriter(ExplodingRewriter())
    result = wrapped.rewrite("сколько стоит карта")
    assert result.query == "сколько стоит карта"
    assert result.fallback is True
    assert result.available is False
    assert "RuntimeError" in result.notes


def test_timed_rewriter_enforces_a_latency_budget() -> None:
    ticks = iter([0.0, 5.0])
    wrapped = TimedRewriter(IdentityRewriter(), clock=lambda: next(ticks), timeout_ms=10.0)
    result = wrapped.rewrite("вопрос")
    assert result.fallback is True
    assert "budget" in result.notes
    assert result.query == "вопрос"


def test_rewriter_status_does_not_load_weights(tmp_path: Path) -> None:
    """Status is safe for /healthz: it must not import torch or touch the model."""
    config = RewriteConfig(
        enabled=True, backend="local_lora", local_lora={"adapter_path": str(tmp_path)}
    )
    status = rewrite_status(config)
    assert status.configured_backend == "local_lora"
    assert "train" in status.reason or "missing" in status.reason or not status.available


def test_config_is_strict_about_unknown_knobs(write_config: Callable[..., Path]) -> None:
    path = write_config("rewrite:\n  enabled: true\n  surprise: 1\n")
    with pytest.raises(RewriteConfigError, match="invalid"):
        load_rewrite_config(path)


def test_config_rejects_an_unknown_backend(write_config: Callable[..., Path]) -> None:
    path = write_config("rewrite:\n  enabled: true\n  backend: gpt5\n")
    with pytest.raises(RewriteConfigError):
        load_rewrite_config(path)


def test_config_accepts_the_repository_default() -> None:
    config = load_rewrite_config()
    assert config.enabled is False
    assert config.backend == "identity"
    assert config.local_lora.max_new_tokens == 64


def test_missing_config_file_means_disabled(tmp_path: Path) -> None:
    config = load_rewrite_config(tmp_path / "absent.yaml")
    assert config.enabled is False
    assert config.backend == "identity"


def test_bad_yaml_is_a_typed_error(write_config: Callable[..., Path]) -> None:
    path = write_config("rewrite: [unclosed\n")
    with pytest.raises(RewriteConfigError):
        load_rewrite_config(path)


def test_deterministic_rewriter_handles_degenerate_input() -> None:
    rewriter = DeterministicRewriter()
    assert rewriter.rewrite("").query == " "
    assert rewriter.rewrite("    ").query == " "
    assert rewriter.rewrite("  ??? ").query == "???"
    result = rewriter.rewrite("Уже чистый вопрос")
    assert result.changed is False
