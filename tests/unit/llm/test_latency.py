"""Latency accounting: percentiles, TTFT capture and the artifact checker.

# AT-LAT-01  # AT-LAT-02
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.contracts import CompletionChunk
from answerdesk.llm.latency import (
    ArmLatency,
    ChunkTimer,
    LatencyLedger,
    LatencySample,
    LatencyStats,
    percentile,
    summarise_arm,
)
from answerdesk.llm.latency_report import (
    LatencyArtifact,
    QualityRow,
    RetrievalRow,
    check_artifact,
    derive_checks,
    load_artifact,
    render_markdown,
    run_offline,
)


def _chunk(delta: str, done: bool = False) -> CompletionChunk:
    return CompletionChunk(delta=delta, done=done, model="m")


def test_percentile_is_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.95) == 50.0
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile(values, 1.5)


def test_latency_stats_shape() -> None:
    stats = LatencyStats.of([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    assert stats.n == 10
    assert stats.p50_ms == 50.0
    assert stats.p95_ms == 100.0
    assert stats.mean_ms == pytest.approx(55.0)
    assert stats.min_ms == 10.0 and stats.max_ms == 100.0
    empty = LatencyStats.of([])
    assert empty.n == 0 and empty.p50_ms == 0.0


def test_chunk_timer_measures_ttft_on_the_first_non_empty_delta() -> None:
    """Keep-alives and empty deltas are not tokens and must not set TTFT."""
    ticks = iter([0.0, 0.5, 0.6, 1.0, 2.0])  # start, empty, first token, token, end
    timer = ChunkTimer(
        [
            _chunk(""),
            _chunk("hello"),
            _chunk(" world", done=True),
        ],
        clock=lambda: next(ticks),
    )
    chunks = list(timer)
    assert len(chunks) == 3
    assert timer.saw_first_token is True
    assert timer.ttft_ms == pytest.approx(600.0)  # 0.6 s after start, not 0.5
    assert timer.total_ms == pytest.approx(2000.0)
    assert timer.text == "hello world"
    assert timer.chunks == 3


def test_chunk_timer_reports_no_ttft_when_no_token_arrives() -> None:
    timer = ChunkTimer([_chunk("", done=True)], clock=lambda: 0.0)
    list(timer)
    assert timer.saw_first_token is False
    assert timer.ttft_ms is None


def test_ledger_groups_by_arm_and_reports_ttft_and_total() -> None:
    ledger = LatencyLedger()
    for index in range(4):
        ledger.add(
            LatencySample(
                arm="baseline",
                provider="p",
                model="m",
                streamed=True,
                ttft_ms=300.0 + index * 100,
                total_ms=3000.0 + index * 500,
                input_tokens=100,
                output_tokens=200,
                usd=0.002,
                chunks=10,
            )
        )
    ledger.add(
        LatencySample(
            arm="budget",
            provider="p",
            model="m",
            streamed=True,
            ttft_ms=200.0,
            total_ms=1200.0,
            output_tokens=50,
            usd=0.0005,
            chunks=5,
        )
    )
    arms = ledger.arms()
    assert set(arms) == {"baseline", "budget"}
    baseline = arms["baseline"]
    assert baseline.n == 4
    # nearest-rank: p50 of an even sample is the lower middle value, not an average
    assert baseline.ttft.p50_ms == pytest.approx(400.0)
    assert baseline.ttft.p95_ms == pytest.approx(600.0)
    assert baseline.total.p50_ms == pytest.approx(3500.0)  # nearest-rank lower middle
    assert baseline.output_tokens_mean == pytest.approx(200.0)
    rows = ledger.rows()
    assert {row["arm"] for row in rows} == {"baseline", "budget"}
    assert "ttft_p50_ms" in rows[0]


def test_summarise_arm_requires_samples() -> None:
    with pytest.raises(ValueError):
        summarise_arm([])


def test_checks_recompute_the_token_and_quality_claims() -> None:
    artifact = LatencyArtifact(
        measured_here=True,
        arms=[
            ArmLatency(
                arm="baseline",
                n=10,
                streamed=True,
                output_tokens_mean=200.0,
                ttft=LatencyStats(n=10, p50_ms=700.0, p95_ms=1500.0),
                total=LatencyStats(n=10, p50_ms=2400.0, p95_ms=4600.0),
            ),
            ArmLatency(
                arm="budget",
                n=10,
                streamed=True,
                output_tokens_mean=120.0,
                ttft=LatencyStats(n=10, p50_ms=600.0, p95_ms=1200.0),
                total=LatencyStats(n=10, p50_ms=1400.0, p95_ms=3200.0),
            ),
        ],
        quality=[
            QualityRow(arm="baseline", n=10, rubric_mean=4.2),
            QualityRow(arm="budget", n=10, rubric_mean=4.1),
        ],
        retrieval=RetrievalRow(n=10, p50_ms=1.4, p95_ms=2.5),
    )
    checks = derive_checks(artifact)
    assert checks.tokens_reduction_pct == pytest.approx(40.0)
    assert checks.token_target_met is True
    assert checks.quality_drop == pytest.approx(0.1)
    assert checks.quality_within_budget is True
    assert checks.retrieval_p95_ok is True
    assert check_artifact(artifact) == []


def test_checker_rejects_an_over_budget_quality_drop() -> None:
    artifact = LatencyArtifact(
        measured_here=True,
        arms=[
            ArmLatency(
                arm="baseline",
                n=5,
                streamed=True,
                output_tokens_mean=200.0,
                ttft=LatencyStats(n=5, p50_ms=700.0),
                total=LatencyStats(n=5, p50_ms=3000.0),
            ),
            ArmLatency(
                arm="budget",
                n=5,
                streamed=True,
                output_tokens_mean=100.0,
                ttft=LatencyStats(n=5, p50_ms=600.0),
                total=LatencyStats(n=5, p50_ms=1000.0),
            ),
        ],
        quality=[
            QualityRow(arm="baseline", n=5, rubric_mean=4.6),
            QualityRow(arm="budget", n=5, rubric_mean=4.0),
        ],
        retrieval=RetrievalRow(n=5, p50_ms=1.0, p95_ms=2.0),
    )
    problems = check_artifact(artifact)
    assert any("quality drop" in problem for problem in problems)


def test_checker_rejects_a_missing_ttft_for_a_streamed_arm() -> None:
    artifact = LatencyArtifact(
        measured_here=True,
        arms=[ArmLatency(arm="baseline", n=3, streamed=True)],
        retrieval=RetrievalRow(n=3, p50_ms=1.0, p95_ms=2.0),
    )
    problems = check_artifact(artifact)
    assert any("TTFT" in problem for problem in problems)


def test_checker_demands_the_honest_label_when_unmeasured() -> None:
    artifact = LatencyArtifact(
        measured_here=False,
        arms=[ArmLatency(arm="baseline", n=3), ArmLatency(arm="budget", n=3)],
        notes=["something else"],
    )
    problems = check_artifact(artifact)
    assert any("not measured here" in problem for problem in problems)
    artifact.notes.append("Not measured here: no endpoint")
    assert check_artifact(artifact) == []


def test_offline_run_is_labelled_and_checks_clean(tmp_path: Path) -> None:
    artifact = run_offline(repeats=2)
    assert artifact.measured_here is False
    assert {row.arm for row in artifact.arms} == {"baseline", "budget"}
    # the cap+claim-limit contract, not a model measurement
    assert derive_checks(artifact).token_target_met is True
    assert check_artifact(artifact) == []
    out = tmp_path / "latency-v2"
    from answerdesk.llm.latency_report import write_artifact

    written = write_artifact(artifact, out)
    assert {path.name for path in written} == {"latency.json", "latency.md"}
    loaded = load_artifact(out / "latency.json")
    assert loaded.measured_here is False
    markdown = render_markdown(artifact)
    assert "TTFT" in markdown and "retrieval" in markdown.lower()
    payload = json.loads((out / "latency.json").read_text(encoding="utf-8"))
    assert payload["checks"]["token_target_met"] is True
