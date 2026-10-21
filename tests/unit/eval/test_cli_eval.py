"""CLI-level harness test: one offline run, frozen artifacts, stable digest (AT-EVL-08/10).

# AT-EVL-08  # AT-EVL-09  # AT-EVL-10  # AT-DET-02
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from fixtures import write_tiny_corpus

from answerdesk.config import Settings
from answerdesk.eval.cli import eval_command
from answerdesk.eval.errors import EvalError


def _run(
    settings: Settings,
    tmp_path: Path,
    *,
    regenerate: bool,
    report: Path,
    limit: int = 12,
) -> list[str]:
    lines: list[str] = []
    exit_code = eval_command(
        settings=settings,
        emit=lines.append,
        all_arms=False,
        retrieval_only=False,
        provider="fake",
        judge_mode="auto",
        report_dir=report,
        regenerate=regenerate,
        limit=limit,
        split="test",
        strict=False,
    )
    assert exit_code == 0
    return lines


def test_eval_command_regenerates_runs_and_writes_a_report(
    tmp_path: Path, settings_for: Callable[..., Settings]
) -> None:
    write_tiny_corpus(tmp_path)
    settings = settings_for()
    report = tmp_path / "reports" / "offline"
    lines = _run(settings, tmp_path, regenerate=True, report=report)
    assert (report / "report.md").is_file()
    assert (report / "metrics.json").is_file()
    assert (report / "turns.jsonl").is_file()
    assert list((report / "plots").glob("*.png"))
    assert (tmp_path / "reports" / "latest.json").is_file()
    text = "\n".join(lines)
    assert "recall_any@10" in text
    assert "deterministic digest" in text
    assert "judge fake-judge" in text
    metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["header"]["corpus_version"] == "corpus-tiny-test"
    assert metrics["header"]["replay"] is True
    assert metrics["cells"]


def test_eval_command_is_deterministic_across_runs(
    tmp_path: Path, settings_for: Callable[..., Settings]
) -> None:
    write_tiny_corpus(tmp_path)
    settings = settings_for()
    first = tmp_path / "reports" / "one"
    second = tmp_path / "reports" / "two"
    _run(settings, tmp_path, regenerate=True, report=first)
    _run(settings, tmp_path, regenerate=False, report=second)
    left = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    right = json.loads((second / "metrics.json").read_text(encoding="utf-8"))
    assert left["digest"] == right["digest"]

    # NFR-DET-02: identical after removing timing fields — latency cells are excluded.
    def stable(cells: list[dict[str, object]]) -> list[tuple[object, object]]:
        return [
            (cell["name"], cell["value"])
            for cell in cells
            if not str(cell["name"]).startswith("latency_")
        ]

    assert stable(left["cells"]) == stable(right["cells"])


def test_eval_command_fails_loudly_on_missing_artifacts(
    tmp_path: Path, settings_for: Callable[..., Settings]
) -> None:
    write_tiny_corpus(tmp_path)
    settings = settings_for()
    with pytest.raises(EvalError):
        eval_command(
            settings=settings,
            emit=lambda *_: None,
            provider="fake",
            report_dir=tmp_path / "reports" / "nope",
            regenerate=False,
            strict=False,
        )


def test_eval_command_rejects_llm_judge_without_a_real_provider(
    tmp_path: Path, settings_for: Callable[..., Settings]
) -> None:
    write_tiny_corpus(tmp_path)
    settings = settings_for()
    with pytest.raises(EvalError) as excinfo:
        eval_command(
            settings=settings,
            emit=lambda *_: None,
            provider="fake",
            judge_mode="llm",
            report_dir=tmp_path / "reports" / "nope",
            regenerate=True,
            limit=4,
            strict=False,
        )
    assert "--judge llm" in str(excinfo.value)


def test_retrieval_only_skips_the_judge(
    tmp_path: Path, settings_for: Callable[..., Settings]
) -> None:
    write_tiny_corpus(tmp_path)
    settings = settings_for()
    report = tmp_path / "reports" / "retrieval"
    lines: list[str] = []
    exit_code = eval_command(
        settings=settings,
        emit=lines.append,
        retrieval_only=True,
        provider="fake",
        report_dir=report,
        regenerate=True,
        limit=8,
        strict=False,
    )
    assert exit_code == 0
    text = "\n".join(lines)
    assert "recall_any@10" in text
    metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    names = {cell["name"] for cell in metrics["cells"]}
    assert "recall_any@10" in names
    assert "rubric_score" not in names
