"""The CLI contract of ``scripts/load.py``: smoke mode, JSON report, exit code.

The graded command is ``uv run python scripts/load.py --smoke``; this module proves the same
code path in-process (fast) and that the report contains everything the acceptance asks for.

    # AT-OPS-07
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

REQUIRED_REPORT_KEYS = {
    "generated_at",
    "mode",
    "machine",
    "service",
    "workload",
    "limitations",
    "levels",
    "leaks",
    "rate_limit",
    "assertions",
    "total_wall_s",
}


def test_smoke_cli_writes_a_json_report(load_harness, tmp_path, capsys) -> None:
    target = tmp_path / "load-smoke.json"
    exit_code = load_harness.main(["--smoke", "--quiet", "--json", str(target)])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out + captured.err
    assert "LOAD PASS" in captured.out

    report = json.loads(target.read_text(encoding="utf-8"))
    assert set(report) >= REQUIRED_REPORT_KEYS, sorted(REQUIRED_REPORT_KEYS - set(report))
    assert report["assertions"]["passed"] is True
    assert report["mode"] == "smoke"
    assert {level["level"]["concurrency"] for level in report["levels"]} == {1, 8}
    assert report["limitations"], "the report must carry its limitations"

    for level in report["levels"]:
        for phase in ("cold", "warm"):
            data = level[phase]
            assert {"latency_ms", "stages_ms", "cache", "throughput_rps", "status_counts"} <= set(
                data
            )
            assert data["latency_ms"]["p50"] is not None
            assert data["stages_ms"]


def test_custom_levels_and_request_count(load_harness, tmp_path, capsys) -> None:
    target = tmp_path / "load-custom.json"
    exit_code = load_harness.main(
        [
            "--levels",
            "2,4",
            "--requests",
            "8",
            "--waves",
            "2",
            "--quiet",
            "--json",
            str(target),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out + captured.err
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["mode"] == "custom"
    assert [level["level"]["concurrency"] for level in report["levels"]] == [2, 4]
    assert all(level["cold"]["requests"] == 8 for level in report["levels"])


def test_human_table_is_printed_by_default(load_harness, capsys) -> None:
    exit_code = load_harness.main(["--levels", "1", "--requests", "6", "--waves", "2"])
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out + captured.err
    assert "conc" in captured.out and "phase" in captured.out
    assert "LOAD PASS" in captured.out
    assert "rate limit:" in captured.out
