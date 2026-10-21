"""Report tests: tables with n/CI, plots, ``latest`` pointer, honesty gates (AT-EVL-08/09).

# AT-EVL-08  # AT-EVL-09  # AT-EVL-07
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.contracts import Action, ReasonCode
from answerdesk.eval.errors import ReportError
from answerdesk.eval.judge import judge_metadata
from answerdesk.eval.models import MetricCell, RunHeader, RunResult, StatComparison, TurnRecord
from answerdesk.eval.report import (
    LIMITATIONS,
    REPLAY_BANNER,
    render_markdown,
    require_statistics,
    write_report,
)


def _header(*, arms: list[str]) -> RunHeader:
    return RunHeader(
        run_id="unit-run",
        schema_version="1.0.0",
        corpus_version="corpus-x",
        corpus_hash="a" * 64,
        golden_version="v1",
        golden_hash="b" * 64,
        split_version="v1",
        split_hash="c" * 64,
        seed=17,
        provider_mode="fake",
        replay=True,
        judge=judge_metadata(judge_model="fake-judge", prompt_version="none", rubric_version="r"),
        arms=list(arms),
        locales=["ru", "tr"],
        case_count=2,
        duration_s=1.5,
        hardware="test",
        notes=["REPLAY-MODE: doubles only"],
    )


def _turn(case_id: str, arm: str) -> TurnRecord:
    return TurnRecord(
        case_id=case_id,
        locale="ru",
        category="direct",
        arm=arm,
        answerable=True,
        expected_language="ru",
        language="ru",
        action=Action.ANSWER,
        reason_code=ReasonCode.ANSWERED,
        answered=True,
        escalated=False,
        ranked_chunk_ids=["c1"],
        gold_chunk_ids=["c1"],
        claims_total=1,
        claims_supported=1,
        groundedness=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        rubric_score=5.0,
        rubric_pass=True,
        answer_ok=True,
        confidence=0.9,
        latency_ms={"total": 12.0},
        usd=0.0001,
    )


def _result(*, arms: list[str] = ("bm25",), comparisons: bool = True) -> RunResult:
    cells = [
        MetricCell(
            name="recall_any@10",
            value=0.5,
            n=2,
            ci_low=0.1,
            ci_high=0.9,
            arm=arms[0],
            locale="macro",
        ),
        MetricCell(
            name="realised_risk",
            value=0.0,
            n=2,
            ci_low=0.0,
            ci_high=0.6,
            arm=arms[0],
        ),
        MetricCell(
            name="latency_total_p50_ms",
            value=12.0,
            n=2,
            ci_low=11.0,
            ci_high=13.0,
            method="none",
            arm=arms[0],
        ),
    ]
    comparison_list = []
    if comparisons and len(arms) > 1:
        comparison_list.append(
            StatComparison(
                metric="recall_any@10",
                arm_a=arms[0],
                arm_b=arms[1],
                n=2,
                mean_a=0.5,
                mean_b=0.0,
                diff=0.5,
                ci_low=0.1,
                ci_high=0.9,
                mcnemar_b=1,
                mcnemar_c=0,
                mcnemar_p=1.0,
                mcnemar_method="exact_binomial",
                significant=True,
            )
        )
    return RunResult(
        header=_header(arms=list(arms)),
        cells=cells,
        comparisons=comparison_list,
        turns=[_turn(turn_id, arms[0]) for turn_id in ("c1", "c2")],
        reliability={
            "n": 2,
            "curve": [{"coverage": 0.0, "risk": 0.0}, {"coverage": 1.0, "risk": 0.0}],
        },
        limitations=list(LIMITATIONS),
    )


def test_write_report_produces_every_artifact(tmp_path: Path) -> None:
    paths = write_report(
        _result(arms=("bm25", "rrf")),
        tmp_path / "reports" / "unit",
        run_id="unit",
        command="uv run answerdesk eval --synthetic --provider fake --report reports/unit",
        digest="d" * 64,
        latest_root=tmp_path / "reports",
    )
    assert paths.markdown.is_file()
    assert paths.metrics.is_file()
    assert paths.turns.is_file()
    assert paths.cells_csv.is_file()
    assert paths.plots, "the report must contain at least one plot (FR-EVL-08)"
    for plot in paths.plots:
        assert plot.suffix == ".png" and plot.stat().st_size > 0
    pointer = tmp_path / "reports" / "latest.json"
    assert pointer.is_file()
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert payload["run_id"] == "unit"
    assert payload["report"].endswith("report.md")
    assert paths.latest == pointer


def test_markdown_contains_header_banner_and_limitations(tmp_path: Path) -> None:
    paths = write_report(
        _result(arms=("bm25", "rrf")),
        tmp_path / "reports" / "unit",
        run_id="unit",
        command="cmd",
        digest="e" * 64,
        latest_root=tmp_path / "reports",
    )
    text = paths.markdown.read_text(encoding="utf-8")
    assert REPLAY_BANNER in text
    assert "## 1. Run header — REPLAY-MODE" in text
    assert "## 2. Retrieval — REPLAY-MODE" in text
    for field in (
        "corpus hash",
        "golden hash",
        "split hash",
        "judge model",
        "rubric version",
        "seed",
    ):
        assert field in text
    for limitation in LIMITATIONS:
        assert limitation.split(",")[0][:40] in text
    assert "paired bootstrap" in text
    assert "McNemar" in text
    assert "diagnostic" in text
    assert "e" * 64 in text


def test_markdown_tables_report_n_and_ci(tmp_path: Path) -> None:
    paths = write_report(
        _result(arms=("bm25", "rrf")),
        tmp_path / "reports" / "unit",
        run_id="unit",
        command="cmd",
        digest="f" * 64,
        latest_root=tmp_path / "reports",
    )
    text = paths.markdown.read_text(encoding="utf-8")
    metric_tables = [block for block in text.split("| metric |")[1:] if block]
    assert metric_tables
    for block in metric_tables:
        header_line = block.splitlines()[0]
        columns = [column.strip() for column in header_line.split("|")]
        assert "n" in columns
        assert any("CI" in column for column in columns), header_line


def test_render_markdown_mentions_single_arm_case() -> None:
    text = render_markdown(_result(arms=("bm25",)), run_id="unit", command="cmd", digest="0" * 64)
    assert "no arm-to-arm comparison" in text


def test_require_statistics_rejects_missing_comparisons() -> None:
    result = _result(arms=("bm25", "rrf"), comparisons=False)
    with pytest.raises(ReportError) as excinfo:
        require_statistics(result)
    assert "comparisons" in str(excinfo.value)


def test_require_statistics_rejects_a_missing_replay_note() -> None:
    result = _result(arms=("bm25",))
    broken = result.model_copy(update={"header": result.header.model_copy(update={"notes": []})})
    with pytest.raises(ReportError):
        require_statistics(broken)


def test_require_statistics_rejects_an_empty_ci() -> None:
    result = _result(arms=("bm25",))
    bad_cell = result.cells[0].model_copy(update={"n": 0})
    broken = result.model_copy(update={"cells": [bad_cell]})
    with pytest.raises(ReportError):
        require_statistics(broken)


def test_metric_cell_rejects_an_interval_that_excludes_its_value() -> None:
    with pytest.raises(ValueError):
        MetricCell(name="x", value=0.5, n=3, ci_low=0.6, ci_high=0.9)
