"""Documentation and rubric-drift tests (AT-EVL-11, FR-EVL-11).

The methodology document is a requirement, not a nicety: CL-EVL-01 walks its sections and
checks the formulas against the code. This test keeps the cheap half of that honest — every
required subsection exists, and the rubric on disk is the rubric the judge runs.

    # AT-EVL-11
"""

from __future__ import annotations

from pathlib import Path

from answerdesk.eval.judge import JUDGE_PROMPT_VERSION, RUBRIC_TEXT, RUBRIC_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / "docs" / "eval-methodology.md"
RUBRIC = REPO_ROOT / "evals" / "judge" / "rubric_v1.md"
EVALS_README = REPO_ROOT / "evals" / "README.md"


def test_methodology_document_exists_and_covers_every_section() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = (
        "## 1. Commands and artifacts",
        "## 2. Golden set",
        "## 3. Splits and leakage",
        "## 4. Metrics",
        "## 5. The judge",
        "## 6. Known biases and limitations",
        "## 7. Labeling protocol",
        "## 8. Extending the harness",
    )
    for heading in required:
        assert heading in text, f"{heading!r} is missing from docs/eval-methodology.md"


def test_methodology_states_the_formulas() -> None:
    text = DOC.read_text(encoding="utf-8")
    for formula in ("recall_any@k", "recall_all@k", "nDCG@k", "MRR@k", "log2(i+1)", "AURC"):
        assert formula in text, f"formula {formula!r} is not documented"
    for phrase in ("Wilson", "paired bootstrap", "McNemar", "Cohen's κ", "equal-mass"):
        assert phrase in text, f"{phrase!r} is not documented"


def test_methodology_prints_the_honest_reporting_policy() -> None:
    text = DOC.read_text(encoding="utf-8")
    for item in (
        "Synthetic data optimism",
        "Small n",
        "Judge bias",
        "Conformal exchangeability",
        "Replay mode is labelled",
        "No claim of SOTA",
    ):
        assert item in text


def test_methodology_carries_its_traceability_marker() -> None:
    assert "# AT-EVL-11" in DOC.read_text(encoding="utf-8")


def _normalise(text: str) -> str:
    return " ".join(text.split())


def test_rubric_file_matches_the_code() -> None:
    """A rubric edit in one place only is silent metric drift (§10.5)."""
    text = _normalise(RUBRIC.read_text(encoding="utf-8"))
    for paragraph in RUBRIC_TEXT.split("\n\n"):
        normalised = _normalise(paragraph)
        if not normalised or normalised.startswith("#"):
            continue
        assert normalised in text, f"rubric paragraph drifted: {normalised[:80]!r}"
    assert RUBRIC_VERSION in text
    assert JUDGE_PROMPT_VERSION in text


def test_evals_readme_documents_the_artifacts() -> None:
    text = EVALS_README.read_text(encoding="utf-8")
    for path in (
        "golden/v1/test.jsonl",
        "golden/v1/dev.jsonl",
        "golden/v1/manifest.json",
        "splits/v1.json",
        "judge/rubric_v1.md",
        "adversarial/near_miss.jsonl",
    ):
        assert path in text
    assert "machine-derived" in text
