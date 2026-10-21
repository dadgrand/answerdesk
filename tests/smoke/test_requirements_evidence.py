"""Evidence checks for the repository-level requirements that have no module of their own.

Every test here runs a real gate or reads a real artifact — none of them merely asserts that a
document mentions a number. They exist so the §16 matrix can point at evidence instead of at a
promise (review finding R-01).

    # AT-REP-01  # AT-REP-02  # AT-REP-04  # AT-DOC-02
    # AT-QUAL-02  # AT-QUAL-03  # AT-SCALE-01  # AT-SEC-05
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
RESULTS = REPO_ROOT / "docs" / "results.md"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


def load_script(name: str) -> ModuleType:
    """Import a hyphenated script file as a module (they are not importable by name)."""
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# NFR-REP-01 — every published number traces to an artifact
# --------------------------------------------------------------------------------------


def test_rep_01_number_checker_passes_on_the_release_candidate() -> None:
    """# AT-REP-01 — README quantities ⊆ generated docs/artifacts; results quantities ⊆ reports."""
    assert (SCRIPTS / "check-numbers.py").is_file(), "scripts/check-numbers.py is missing (CMD-NUM)"
    result = run_script("check-numbers.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "traces to the generated docs or the committed artifacts" in result.stdout


def test_rep_01_invented_number_fails_the_checker(tmp_path: Path) -> None:
    """# AT-REP-01 — the gate bites: an unbacked quantity in the README is caught."""
    (tmp_path / "docs").mkdir()
    results = RESULTS.read_text(encoding="utf-8")
    (tmp_path / "docs" / "results.md").write_text(results, encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    (tmp_path / "README.md").write_text(
        readme + "\n\nOur p99 latency was 987654 ms on every locale.\n", encoding="utf-8"
    )
    result = run_script("check-numbers.py", "--root", str(tmp_path))
    assert result.returncode == 1
    assert "987654" in result.stdout


# --------------------------------------------------------------------------------------
# NFR-REP-02 — provenance hashes are present and point at committed files
# --------------------------------------------------------------------------------------


def test_rep_02_provenance_block_is_complete() -> None:
    """# AT-REP-02 — the report header carries the hashes the run was produced from."""
    text = RESULTS.read_text(encoding="utf-8")
    start = text.index("## Provenance")
    block = text[start : text.index("## ", start + 3)]
    hashes = dict(re.findall(r"^(\w+)\s+([0-9a-f]{64})$", block, flags=re.MULTILINE))
    assert len(hashes) >= 5, hashes
    for required in ("corpus_hash", "golden_manifest", "split_manifest", "answerdesk_bench"):
        assert required in hashes, f"{required} missing from the provenance block"
    for value in hashes.values():
        assert len(value) == 64 and int(value, 16) >= 0
    for referenced in (
        "data/corpus/manifest.json",
        "evals/golden/v1/manifest.json",
        "evals/splits/v1.json",
        "configs/pricing.yaml",
    ):
        assert (REPO_ROOT / referenced).is_file(), referenced


# --------------------------------------------------------------------------------------
# NFR-REP-04 — offline licence audit
# --------------------------------------------------------------------------------------


def test_rep_04_licence_audit_passes_offline() -> None:
    """# AT-REP-04 — installed distributions are permissive and the repo posture is intact."""
    assert (SCRIPTS / "check-licenses.py").is_file(), (
        "scripts/check-licenses.py is missing (CMD-LIC)"
    )
    result = run_script("check-licenses.py", "--quiet")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "every installed distribution is permissive" in result.stdout


def test_rep_04_licence_policy_rejects_copyleft() -> None:
    """# AT-REP-04 — the classifier is not vacuous: copyleft and non-commercial are rejected."""
    module = load_script("check-licenses.py")
    for forbidden in (
        "GPL-3.0",
        "AGPL-3.0",
        "SSPL-1.0",
        "Business Source License 1.1",
        "Commons Clause",
        "Non-Commercial",
        "Proprietary",
    ):
        assert module.FORBIDDEN.search(forbidden), forbidden
        assert not module._allowed(forbidden), forbidden
    for allowed in ("MIT", "MIT License", "BSD-3-Clause", "Apache-2.0", "ISC", "MPL-2.0"):
        assert module._allowed(allowed), allowed


# --------------------------------------------------------------------------------------
# NFR-DOC-02 — the repository is English, with exactly one documented exception
# --------------------------------------------------------------------------------------


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text


def test_doc_02_english_documents_are_english() -> None:
    """# AT-DOC-02 — prose is English; Russian lives only in README.ru.md and quoted data."""
    cyrillic = re.compile(r"[\u0400-\u04FF]")
    for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md"):
        text = _strip_code((REPO_ROOT / name).read_text(encoding="utf-8"))
        match = cyrillic.search(text)
        if match is not None:
            context = text[max(0, match.start() - 60) : match.start() + 60]
            raise AssertionError(f"{name} has Cyrillic prose near: {context!r}")
    russian = (REPO_ROOT / "README.ru.md").read_text(encoding="utf-8")
    assert cyrillic.search(russian), "README.ru.md is meant to be the Russian counterpart"


# --------------------------------------------------------------------------------------
# NFR-QUAL-02 — the coverage gate exists and is wired
# --------------------------------------------------------------------------------------


def test_qual_02_coverage_gate_is_wired() -> None:
    """# AT-QUAL-02 — the four math packages carry a ≥70% gate in the offline script."""
    script = (SCRIPTS / "ci-offline.ps1").read_text(encoding="utf-8")
    assert "--cov-fail-under=70" in script
    for package in ("retrieval", "policy", "cache", "eval"):
        assert f"--cov=src/answerdesk/{package}" in script, package


# --------------------------------------------------------------------------------------
# NFR-QUAL-03 — the mutation audit is recorded
# --------------------------------------------------------------------------------------


def test_qual_03_mutation_audit_is_recorded() -> None:
    """# AT-QUAL-03 — the independent mutation spot-check is in the audit trail."""
    report = (REPO_ROOT / "docs" / "review-report.md").read_text(encoding="utf-8")
    start = report.index("## 2. Test quality")
    section = report[start : report.index("## 3.", start)]
    mutations = re.findall(r"^\|\s*M\d+\s*\|", section, flags=re.MULTILINE)
    assert len(mutations) >= 5, mutations
    assert "caught" in section.lower()
    assert "R-04" in report


# --------------------------------------------------------------------------------------
# NFR-SCALE-01 — the scale knob exists; the 3k run stays a labelled release step
# --------------------------------------------------------------------------------------


def test_scale_01_scale_knob_and_manifest_depth() -> None:
    """# AT-SCALE-01 — the corpus generator scales and records per-locale depth."""
    manifest = json.loads(
        (REPO_ROOT / "data" / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )
    per_locale = manifest["stats"]["docs_per_locale"]
    assert set(per_locale) == {"ru", "uz", "tg", "ky", "tr", "zh"}, per_locale
    assert min(per_locale.values()) >= 100
    for locale, payload in manifest["locales"].items():
        assert payload["documents"] == per_locale[locale], locale
    source = (REPO_ROOT / "src" / "answerdesk" / "ingest" / "synthetic.py").read_text(
        encoding="utf-8"
    )
    assert "docs_per_locale: int = 135" in source
    assert "docs_per_locale must be >= 1" in source
    ingestion_doc = (REPO_ROOT / "docs" / "ingestion.md").read_text(encoding="utf-8")
    assert "ANSWERDESK_INGEST_DOCS_PER_LOCALE" in ingestion_doc


# --------------------------------------------------------------------------------------
# NFR-SEC-05 — the half of the dependency audit that can run offline
# --------------------------------------------------------------------------------------


def test_sec_05_vulnerability_audit_is_documented_and_split() -> None:
    """# AT-SEC-05 — pip-audit is the networked step; the offline half is the licence audit."""
    repro = (REPO_ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")
    assert "pip-audit" in repro
    assert "check-licenses.py" in repro
    assert (SCRIPTS / "check-licenses.py").is_file()
