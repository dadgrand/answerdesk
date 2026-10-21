"""Cost ledger and pricing table (NFR-COST-01, NFR-COST-02).

The ledger is the only place a dollar amount is computed. These tests pin the arithmetic,
the aggregation, the unpriced-model behaviour and — the point of NFR-COST-02 — that a price
change is a YAML edit, not a code edit.

    # AT-COST-01  # AT-COST-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.contracts import PricingEntry
from answerdesk.errors import ConfigError
from answerdesk.llm.cost import (
    CostLedger,
    CostReport,
    PricingTable,
    default_pricing_path,
    estimate_messages_tokens,
    estimate_tokens,
    percentile,
    summary_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def price_entry(
    model: str,
    input_usd: float,
    output_usd: float,
    *,
    retrieved_at: str = "2026-10-20",
) -> PricingEntry:
    return PricingEntry(
        model=model,
        input_usd_per_1k_tokens=input_usd,
        output_usd_per_1k_tokens=output_usd,
        retrieved_at=retrieved_at,
        source="unit test",
    )


# --------------------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------------------


def test_checked_in_pricing_table_loads_with_provenance() -> None:
    """# AT-COST-02 — the pinned table exists and every row states where it came from."""
    path = default_pricing_path()
    assert path is not None and path.is_file()
    table = PricingTable.from_yaml(path)
    assert "fake-model" in table.entries
    for entry in table.entries.values():
        assert entry.source
        assert entry.retrieved_at


def test_price_is_inputs_over_1000_times_the_rate() -> None:
    table = PricingTable.from_entries([price_entry("m", 0.5, 1.5)])
    quote = table.price("m", input_tokens=2000, output_tokens=1000)
    assert quote.usd == pytest.approx(0.5 * 2 + 1.5 * 1)
    assert quote.priced is True
    assert "unit test" in quote.pricing_source


def test_unknown_model_is_priced_zero_and_reported() -> None:
    """# AT-COST-01 — an unpriced model must be visible, not silently free."""
    table = PricingTable.from_entries([price_entry("known", 1.0, 1.0)])
    quote = table.price("mystery-model", input_tokens=10, output_tokens=10)
    assert quote.usd == 0.0
    assert quote.priced is False
    assert quote.pricing_source == "unknown"


def test_prefixed_model_names_reuse_the_underlying_row() -> None:
    table = PricingTable.from_entries([price_entry("gpt-4o-mini", 1.0, 2.0)])
    assert table.price("openai/gpt-4o-mini", input_tokens=1000, output_tokens=0).usd == 1.0


def test_invalid_pricing_yaml_is_an_actionable_config_error(tmp_path: Path) -> None:
    broken = tmp_path / "pricing.yaml"
    broken.write_text("pricing: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a list"):
        PricingTable.from_yaml(broken)
    missing_key = tmp_path / "pricing2.yaml"
    missing_key.write_text("something: else\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="pricing"):
        PricingTable.from_yaml(missing_key)


def test_a_price_change_is_a_config_edit(tmp_path: Path) -> None:
    """# AT-COST-02 — same code, different YAML, different USD."""
    table_file = tmp_path / "pricing.yaml"
    table_file.write_text(
        "pricing:\n"
        "  - model: m\n"
        "    input_usd_per_1k_tokens: 0.10\n"
        "    output_usd_per_1k_tokens: 0.40\n"
        "    retrieved_at: '2026-10-20'\n"
        "    source: 'test table'\n",
        encoding="utf-8",
    )
    ledger = CostLedger(pricing=PricingTable.from_yaml(table_file))
    ledger.record(preset="vllm", model="m", input_tokens=1000, output_tokens=500, latency_ms=12.0)
    assert ledger.records[0].usd == pytest.approx(0.10 + 0.20)


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------


def test_every_call_records_tokens_usd_latency_and_model(ledger: CostLedger) -> None:
    """# AT-COST-01 — the four fields NFR-COST-01 names, plus provenance."""
    record = ledger.record(
        preset="openai",
        model="gpt-4o-mini",
        input_tokens=120,
        output_tokens=40,
        latency_ms=321.5,
        prompt_version="grounded-answer-1.0.0",
        estimated_input_tokens=118,
    )
    assert record.input_tokens == 120
    assert record.output_tokens == 40
    assert record.latency_ms == pytest.approx(321.5)
    assert record.model == "gpt-4o-mini"
    assert record.prompt_version == "grounded-answer-1.0.0"
    assert record.usage_source == "provider"
    assert record.request_id == "unit-session-r0001"
    assert record.usd == pytest.approx(0.0)  # placeholder pricing: CI is free
    assert record.priced is True


def test_session_summary_aggregates_calls_models_and_latency(ledger: CostLedger) -> None:
    """# AT-COST-01 — per-session aggregation (t9/t7 read this)."""
    for latency, model, ok in (
        (100.0, "fake-model", True),
        (200.0, "fake-model", True),
        (400.0, "other-model", False),
    ):
        ledger.record(
            preset="fake",
            model=model,
            input_tokens=10,
            output_tokens=5,
            latency_ms=latency,
            ok=ok,
            error_code=None if ok else "provider_error",
            usage_source="provider" if ok else "none",
        )
    summary = ledger.summary()
    assert summary.session_id == "unit-session"
    assert summary.calls == 3
    assert summary.ok_calls == 2
    assert summary.failed_calls == 1
    assert summary.input_tokens == 30
    assert summary.output_tokens == 15
    assert summary.latency_p50_ms == pytest.approx(200.0)
    assert summary.latency_p95_ms == pytest.approx(400.0)
    assert [model.model for model in summary.by_model] == ["fake-model", "other-model"]
    assert summary.usage_sources == {"none": 1, "provider": 2}
    assert summary.unpriced_models == ["other-model"]
    assert summary.as_row()["calls"] == 3


def test_summary_reports_estimate_mismatch(ledger: CostLedger) -> None:
    """# AT-COST-01 — the local estimate is reconciled, and never used for billing."""
    ledger.record(
        preset="vllm",
        model="local-model",
        input_tokens=100,
        output_tokens=10,
        latency_ms=5.0,
        estimated_input_tokens=110,
    )
    summary = ledger.summary()
    assert summary.estimate_mismatch == pytest.approx(0.1)
    assert ledger.records[0].input_tokens == 100  # provider usage wins


def test_report_merges_sessions(ledger: CostLedger) -> None:
    other = CostLedger(session_id="second", pricing=ledger.pricing)
    ledger.record(
        preset="fake", model="fake-model", input_tokens=1, output_tokens=1, latency_ms=1.0
    )
    other.record(preset="fake", model="fake-model", input_tokens=2, output_tokens=2, latency_ms=2.0)
    report = CostReport.from_summaries([ledger.summary(), other.summary()])
    assert len(report.sessions) == 2
    assert report.calls == 2
    assert report.input_tokens == 3
    assert report.usd_total == pytest.approx(0.0)


def test_ledger_json_round_trip(ledger: CostLedger, tmp_path: Path) -> None:
    ledger.record(
        preset="fake", model="fake-model", input_tokens=4, output_tokens=2, latency_ms=3.0
    )
    path = ledger.write_json(tmp_path / "cost.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["records"][0]["model"] == "fake-model"
    summary = summary_from_mapping(payload["summary"])
    assert summary.calls == 1


def test_reset_clears_the_ledger(ledger: CostLedger) -> None:
    ledger.record(
        preset="fake", model="fake-model", input_tokens=1, output_tokens=1, latency_ms=1.0
    )
    ledger.reset()
    assert len(ledger) == 0
    assert ledger.summary().calls == 0


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def test_percentile_is_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 0.95) == 40.0
    assert percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile(values, 1.5)


def test_token_estimate_is_deterministic_and_counts_cjk() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == estimate_tokens("a" * 400)
    assert estimate_tokens("退款将在14个工作日内完成。") >= 10
    assert estimate_messages_tokens(["abcd", "efgh"]) == 2
