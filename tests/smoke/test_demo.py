"""AT-OPS-06 / CMD-U: ``answerdesk demo --smoke`` is a wired command, not a stub.

The command is the shortest honest path through the whole offline loop (corpus → index →
question → decision → service health), so this test pins three things a stub cannot fake: the
command exits 0, it writes a machine-readable summary, and every step in that summary reports
``ok``. The service step is patched out here because the suite's socket guard forbids every
connection including loopback; the real probe is exercised by the documented verify command
(``uv run answerdesk demo --smoke``) and by ``tests/integration/service``.

    # AT-OPS-06
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_demo_smoke_runs_the_offline_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    import answerdesk.cli as cli

    monkeypatch.setattr(
        cli,
        "_demo_service",
        lambda _settings: {"service": "skipped", "reason": "socket guard in the test suite"},
    )
    result = CliRunner().invoke(cli.app, ["demo", "--smoke", "--report", str(tmp_path / "demo")])

    assert result.exit_code == 0, result.output
    assert "demo: OK" in result.output
    summary = json.loads((tmp_path / "demo" / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "answerdesk-demo"
    assert summary["status"] == "ok"
    assert summary["generated_at"]
    steps = {step["step"]: step for step in summary["steps"]}
    assert set(steps) >= {
        "ingest --demo",
        "index --demo",
        "ask --fake",
        "calibrate --synthetic",
        "service healthz",
    }
    assert all(step["status"] == "ok" for step in steps.values())
    assert steps["ask --fake"]["citations"] >= 1
    assert steps["ask --fake"]["decision"]
    assert steps["service healthz"]["service"] == "skipped"


def test_demo_is_not_a_stub_in_the_help_surface() -> None:
    """The option exists and the command is documented — the t14 stub exited 2."""
    import re

    from typer.testing import CliRunner

    from answerdesk.cli import app

    # Force a wide, colourless canvas: rich truncates long option names at narrow widths
    # and wraps them in ANSI codes, which made this assertion CI-only flaky (t39 lesson).
    result = CliRunner().invoke(
        app,
        ["demo", "--help"],
        env={"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"},
        color=False,
    )
    assert result.exit_code == 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", result.output)
    assert "…" not in plain, "the help page was truncated before the assertions"
    assert "--smoke" in plain
    assert "--report" in plain
