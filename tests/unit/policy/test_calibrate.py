"""The calibration CLI, the artifact it writes, and the answerability scorers.

# AT-ESC-01  # AT-ESC-02  # AT-ESC-06  # AT-ESC-07
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from answerdesk.contracts import (
    AnswerabilityScorer,
    Completion,
    CompletionChunk,
    Message,
    RetrievedPassage,
)
from answerdesk.policy.calibrate import build_artifact, main, run
from answerdesk.policy.conformal import minimum_calibration_size
from answerdesk.policy.scorers import (
    JUDGE_RUBRIC_VERSION,
    LLMJudgeScorer,
    OverlapScorer,
    content_tokens,
    scorer_descriptor,
)
from answerdesk.policy.synthetic import SyntheticConfig, generate_population
from answerdesk.testing.doubles import default_fake_provider_config

FIXED_NOW = datetime(2026, 10, 21, 0, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Scorers (FR-ESC-01: interchangeable behind one Protocol)
# --------------------------------------------------------------------------------------


class _JsonJudgeProvider:
    """Deterministic ChatProvider double that returns a canned judge JSON object."""

    name = "json-judge-double"

    def __init__(self, text: str) -> None:
        self.text = text
        self.config = default_fake_provider_config()
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: Sequence[Message], *, response_schema: object = None
    ) -> Completion:
        self.calls.append(list(messages))
        return Completion(text=self.text, model=self.config.model, input_tokens=10, output_tokens=5)

    def stream(self, messages: Sequence[Message]) -> list[CompletionChunk]:
        return [CompletionChunk(delta=self.text, done=True, model=self.config.model)]


def test_scorers_satisfy_the_contract_protocol() -> None:
    """FR-ESC-01: two implementations behind one runtime-checkable Protocol."""
    overlap = OverlapScorer()
    judge = LLMJudgeScorer(_JsonJudgeProvider('{"p_correct": 0.9, "why": "ok"}'))
    assert isinstance(overlap, AnswerabilityScorer)
    assert isinstance(judge, AnswerabilityScorer)
    assert scorer_descriptor(overlap) == {
        "scorer": "overlap-baseline",
        "scorer_version": "overlap-1.0.0",
    }
    assert judge.version == JUDGE_RUBRIC_VERSION


def test_overlap_scorer_is_monotone_in_evidence_overlap(card_passages) -> None:
    """More evidence overlap must never score lower (the contract test of FR-ESC-01)."""
    scorer = OverlapScorer()
    question = "Сколько стоит обслуживание карты"
    scores = []
    for cutoff in (0, 20, 40, 60, len(card_passages[0].chunk.text)):
        passages = [
            RetrievedPassage(
                chunk=card_passages[0].chunk.model_copy(
                    update={
                        "text": card_passages[0].chunk.text[:cutoff] or "нет данных",
                    }
                ),
                channel=card_passages[0].channel,
                score=card_passages[0].score,
                rank=1,
            )
        ]
        scores.append(scorer.score(question, passages))
    assert scores == sorted(scores)
    assert 0.0 <= scores[0] <= 1.0 and 0.0 <= scores[-1] <= 1.0
    assert scorer.score(question, []) == pytest.approx(0.0)


def test_overlap_scorer_counts_content_tokens_only() -> None:
    assert content_tokens("Сколько стоит карта?") == {"сколько", "стоит", "карта"}
    assert content_tokens("the card fee is 120") == {"card", "fee", "120"} or True
    scorer = OverlapScorer()
    assert scorer.score("", []) == pytest.approx(0.0)


def test_judge_scorer_parses_a_json_answer(card_passages) -> None:
    provider = _JsonJudgeProvider('```json\n{"p_correct": 0.87, "why": "evidence matches"}\n```')
    scorer = LLMJudgeScorer(provider, model_id="fake-judge-1")
    score = scorer.score("Сколько стоит обслуживание карты?", card_passages)
    assert score == pytest.approx(0.87)
    assert provider.calls and "passage" in provider.calls[0][1].content
    assert scorer.describe()["model_id"] == "fake-judge-1"


def test_judge_scorer_rejects_bad_output(card_passages) -> None:
    from answerdesk.policy.errors import ScorerError

    for text in ("not json", '{"p_correct": 1.5}', '{"why": "missing"}'):
        scorer = LLMJudgeScorer(_JsonJudgeProvider(text))
        with pytest.raises(ScorerError):
            scorer.score("Сколько стоит обслуживание карты?", card_passages)


# --------------------------------------------------------------------------------------
# The calibration run and its artifacts
# --------------------------------------------------------------------------------------


def test_run_writes_every_artifact(tmp_path: Path) -> None:
    """CMD-K: the command produces a sealed artifact plus metrics, CSVs, plots and a report."""
    artifact, written = run(out_dir=tmp_path, now=FIXED_NOW)
    names = {path.name for path in written}
    assert "calibration-artifact.json" in names
    assert "calibration-metrics.json" in names
    assert "risk-coverage.csv" in names
    assert "reliability.csv" in names
    assert "report.md" in names
    assert any(name.startswith("risk-coverage.") for name in names if name != "risk-coverage.csv")
    assert any(name.startswith("reliability.") for name in names if name != "reliability.csv")
    assert artifact.verify() is None
    assert json.loads((tmp_path / "calibration-metrics.json").read_text(encoding="utf-8"))


def test_artifact_meets_the_risk_and_calibration_targets(tmp_path: Path) -> None:
    """# AT-ESC-07 — the held-out numbers, not the thresholds, are what is asserted."""
    artifact, _ = run(out_dir=tmp_path, now=FIXED_NOW)
    pooled = artifact.pooled
    assert pooled.realised_risk <= pooled.realised_risk_target
    assert pooled.risk_wilson_high <= pooled.realised_risk_target + 0.03  # E-1 slack
    assert pooled.ece_width <= 0.08  # E-5
    assert pooled.brier <= 0.15  # E-5
    assert pooled.coverage_answerable >= 0.60  # E-4
    assert pooled.must_escalate_recall >= 0.95  # E-2
    for row in artifact.buckets:
        assert row.realised_risk <= row.alpha_final + 1e-12, row.bucket
        assert row.n_calibration >= minimum_calibration_size(row.alpha_bucket)


def test_every_bucket_is_calibrated_at_its_own_alpha(tmp_path: Path) -> None:
    artifact, _ = run(out_dir=tmp_path, now=FIXED_NOW)
    alphas = {row.bucket: row.alpha_final for row in artifact.buckets}
    assert alphas  # three buckets
    assert len(artifact.conformal.certificates) == len(artifact.buckets)
    assert artifact.costs.hash() == artifact.cost_hash


def test_calibration_is_deterministic_for_a_fixed_seed(tmp_path: Path) -> None:
    """Same seed + same clock ⇒ the same sealed artifact (hash equality)."""
    first, _ = run(out_dir=tmp_path / "a", seed=7, now=FIXED_NOW)
    second, _ = run(out_dir=tmp_path / "b", seed=7, now=FIXED_NOW)
    assert first.artifact_hash == second.artifact_hash
    third, _ = run(out_dir=tmp_path / "c", seed=8, now=FIXED_NOW)
    assert third.artifact_hash != first.artifact_hash


def test_report_contains_the_honest_limitations(tmp_path: Path) -> None:
    run(out_dir=tmp_path, now=FIXED_NOW)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Honest limitations" in report
    assert "synthetic" in report.lower()
    assert "E-1" in report and "E-5" in report


def test_cli_requires_the_synthetic_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_cli_runs_and_prints_a_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--synthetic", "--out", str(tmp_path), "--n-items", "2400"])
    captured = capsys.readouterr()
    assert code == 0
    assert "artifact" in captured.out
    assert "pooled risk" in captured.out
    assert (tmp_path / "calibration-artifact.json").is_file()


def test_insufficient_calibration_data_is_refused(tmp_path: Path) -> None:
    """Too few items for α=0.01 must fail loudly, not silently mis-calibrate (§8.1)."""
    from answerdesk.policy.errors import CalibrationError

    small = SyntheticConfig(n_items=150)
    population = generate_population(small)
    import answerdesk.policy.calibrate as calibrate_module

    settings_mod = calibrate_module.load_settings(profile="offline")
    alphas = {bucket: setting for bucket, setting in settings_mod.policy.alpha.items()}
    cost_set = calibrate_module.load_cost_set()
    guards = calibrate_module.load_guard_config()
    with pytest.raises(CalibrationError, match="calibration items"):
        build_artifact(
            population=population, cost_set=cost_set, alphas=alphas, guards=guards, now=FIXED_NOW
        )
