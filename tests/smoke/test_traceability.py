"""The traceability matrix is enforced by a real script, not by a promise (NFR-REP-03 / CMD-T).

This replaces the earlier 12-id subset: the checker itself is executed, and two seeded
breakages prove that it actually fails when the matrix and the tree disagree.

    # AT-REP-03
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check-traceability.py"
REQUIREMENTS = REPO_ROOT / "docs" / "requirements.md"


def run_checker(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_checker_exists_and_the_matrix_is_green() -> None:
    """The release candidate must pass its own gate."""
    assert CHECKER.is_file(), "scripts/check-traceability.py is missing (CMD-T)"
    result = run_checker()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "every matrix path resolves" in result.stdout
    assert "121 rows" in result.stdout


def test_every_at_id_in_the_matrix_has_a_marker() -> None:
    """Parsed, not assumed: the JSON report enumerates the AT ids and their markers."""
    result = run_checker("--json")
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["rows"] == 121
    assert payload["at_ids"] == payload["marked_at_ids"]
    assert payload["missing_markers"] == []
    assert payload["missing_paths"] == []


def test_a_missing_marker_fails_the_gate(tmp_path: Path) -> None:
    """Seed the failure the script exists to catch: an AT id with no test marker."""
    (tmp_path / "docs").mkdir()
    shutil.copy(REQUIREMENTS, tmp_path / "docs" / "requirements.md")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    result = run_checker("--root", str(tmp_path))
    assert result.returncode == 1
    assert "have no marker" in result.stdout
    assert "AT-ING-01" in result.stdout


def test_a_broken_matrix_path_fails_the_gate(tmp_path: Path) -> None:
    """Seed the other failure: a matrix row pointing at a module that no longer exists."""
    (tmp_path / "docs").mkdir()
    text = REQUIREMENTS.read_text(encoding="utf-8")
    fake_at = "AT-" + "ZZZ-01"  # assembled at runtime: a literal marker here would be scanned
    seeded = f"| FR-ZZZ-01 | `policy/does_not_exist.py` | t0 | {fake_at} | CMD-A | G-CI |\n"
    text = text.replace("## 16. Traceability matrix", "## 16. Traceability matrix\n\n" + seeded, 1)
    (tmp_path / "docs" / "requirements.md").write_text(text, encoding="utf-8")
    result = run_checker("--root", str(tmp_path))
    assert result.returncode == 1
    assert "policy/does_not_exist.py" in result.stdout
    assert "do not exist" in result.stdout


def test_the_checker_reports_its_verdict() -> None:
    """A quiet gate must still be able to explain itself."""
    result = run_checker("--quiet")
    assert result.returncode == 0
    assert "OK" in result.stdout
