"""Integration: evidence-locale purity on the frozen corpus, measured the way the report is.

This runs the same measurement the operator command runs (``python -m
answerdesk.retrieval.locale_report``) but asserts in-test, so a regression cannot slip past
the unit suite. It needs the checked-in corpus and the committed consistency run; without
either it skips rather than failing (the rest of the suite must stay runnable).

    # AT-RET-06  # AT-RET-10
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.retrieval.locale_report import (
    DEFAULT_CONSISTENCY_RUN,
    INVARIANCE_TARGET,
    PURITY_TARGET,
    load_consistency_set,
    run_measurement,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS = REPO_ROOT / "data" / "corpus"
CONSISTENCY = REPO_ROOT / DEFAULT_CONSISTENCY_RUN

pytestmark = pytest.mark.integration


def _require_inputs() -> None:
    if not (CORPUS / "manifest.json").is_file():
        pytest.skip("no built corpus; run `answerdesk ingest --demo` first")
    if not (CONSISTENCY / "answers.jsonl").is_file():
        pytest.skip(f"no consistency run at {CONSISTENCY}")


@pytest.fixture(scope="module")
def report() -> dict:
    _require_inputs()
    return run_measurement(consistency_run=CONSISTENCY, corpus_dir=CORPUS, window=12)


def test_the_consistency_set_is_the_expected_size() -> None:
    _require_inputs()
    instances = load_consistency_set(CONSISTENCY)
    assert len(instances) == 378  # 63 facts × 6 locales
    assert {instance.locale for instance in instances} == {"ru", "uz", "ky", "tg", "tr", "zh"}


def test_evidence_locale_purity_reaches_the_target(report: dict) -> None:
    purity = report["purity"]
    assert purity["before"]["chunk_purity"] < 0.85, "baseline should still show the leak"
    assert purity["after"]["chunk_purity"] >= PURITY_TARGET, purity["after"]["chunk_purity"]
    assert purity["passed"] is True


def test_every_locale_clears_the_floor(report: dict) -> None:
    for locale, row in report["purity"]["after"]["per_locale"].items():
        assert row["chunk_purity"] >= 0.90, f"{locale}: {row['chunk_purity']}"


def test_the_policy_does_not_cost_evidence(report: dict) -> None:
    """Packed recall is a regression gate: improving is fine, losing gold evidence is not."""
    packed = report["packed_recall"]
    assert packed["delta"] >= -packed["tolerance"], packed
    guard = report["ablation_guard"]
    for arm, delta in guard["delta"].items():
        assert delta >= -guard["tolerance"], f"{arm}: {delta}"


def test_surface_form_invariance_is_exact(report: dict) -> None:
    invariance = report["invariance"]
    assert invariance["rate"] >= INVARIANCE_TARGET, invariance
    for variant, block in invariance["variants"].items():
        assert block["matched"] == invariance["instances"], variant


def test_measurement_is_deterministic(report: dict) -> None:
    assert report["determinism"]["digest_match"] is True
    assert report["passed"] is True


def test_committed_report_matches_the_measurement(report: dict) -> None:
    """The artifact under reports/locale-filter is the evidence; it must equal a fresh run."""
    path = REPO_ROOT / "reports" / "locale-filter" / "report.json"
    if not path.is_file():  # pragma: no cover - first run writes it
        pytest.skip("reports/locale-filter/report.json has not been written yet")
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed["purity"]["after"]["chunk_purity"] == report["purity"]["after"][
        "chunk_purity"
    ]
    assert committed["consistency_set"]["digest"] == report["consistency_set"]["digest"]
    assert committed["gates"] == report["gates"]
