"""Traceability markers for the requirements this skeleton owns.

``docs/requirements.md`` §16 maps each requirement to an acceptance test id, and
``scripts/check-traceability.py`` (t12) verifies that every ``AT-`` id appears as a marker
in the repository. This test keeps the skeleton honest in the meantime: the requirements
assigned to t3 must have a marker somewhere in ``src`` or ``tests``.

    # AT-REP-03
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

OWNED_ATS = {
    "AT-OPS-01",  # CLI surface exists with --help for every subcommand
    "AT-OPS-03",  # provider presets are configuration
    "AT-OPS-04",  # offline/api/local profiles
    "AT-OPS-05",  # actionable config validation, secret redaction
    "AT-DET-01",  # offline, keyless, network-blocked test gate
    "AT-DET-03",  # fast gate + full gate
    "AT-PORT-01",  # one provider seam, local servers need no key
    "AT-PORT-02",  # Windows + Linux, no Docker
    "AT-PORT-03",  # heavy dependencies behind extras only
    "AT-SEC-03",  # no secrets in the repository
    "AT-QUAL-01",  # ruff + typing configured and clean
    "AT-DOC-01",  # documentation set exists and is accurate
}

MARKER = re.compile(r"#\s*(AT-[A-Z0-9]+-\d+)")


pytestmark = pytest.mark.smoke


def _collect_markers() -> set[str]:
    markers: set[str] = set()
    for directory in ("src", "tests", "scripts"):
        root = REPO_ROOT / directory
        if not root.exists():  # pragma: no cover - scripts arrive with t12/t14
            continue
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".ps1", ".yml", ".yaml", ".sh"} or not path.is_file():
                continue
            markers.update(MARKER.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return markers


def test_every_owned_requirement_has_a_traceability_marker() -> None:
    found = _collect_markers()
    missing = sorted(OWNED_ATS - found)
    assert not missing, f"evidence markers missing for {missing}"


def test_requirements_matrix_references_the_same_ids() -> None:
    """The AT ids above must exist in docs/requirements.md §16 (no invented ids)."""
    requirements = (REPO_ROOT / "docs" / "requirements.md").read_text(encoding="utf-8")
    for at_id in OWNED_ATS:
        assert at_id in requirements, f"{at_id} is not part of the frozen traceability matrix"
