"""``python -m answerdesk.policy.calibrate`` — fit the artifact and write the evidence.

This is the command behind CMD-K and the acceptance for t8. It does five things, in this
order, and refuses to skip any of them:

1. generate the synthetic population (``--synthetic``; a live corpus would use the same code
   path in t11);
2. fit the split-conformal quantile per intent bucket at that bucket's α;
3. compose the cost-sensitive thresholds (conformal axis + cost bands + bucket α);
4. measure the held-out split: realised risk with a Wilson interval, coverage, ECE (both
   binning modes), Brier, AURC, must-escalate recall and escalation precision;
5. write a sealed artifact, a metrics JSON, CSVs, plots and a markdown report.

Exit codes: ``0`` success, ``2`` when the run cannot produce a trustworthy artifact (too
little calibration data for the requested α, invalid cost/guard config). A silent fallback
to "answer anyway" does not exist here — that is the point of the exercise.

    # AT-ESC-02  # AT-ESC-06  # AT-ESC-07
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from answerdesk.config import Settings, load_settings
from answerdesk.contracts import Action, Contract, IntentBucket
from answerdesk.policy.artifact import (
    ARTIFACT_VERSION,
    BucketMetrics,
    CalibrationArtifact,
    PooledMetrics,
)
from answerdesk.policy.conformal import (
    ConformalSet,
    fit_certificates,
    minimum_calibration_size,
)
from answerdesk.policy.costs import CostSet, load_cost_set
from answerdesk.policy.decision import DecisionEngine
from answerdesk.policy.errors import CalibrationError, CostModelError, GuardError
from answerdesk.policy.guards import GuardConfig, load_guard_config
from answerdesk.policy.metrics import (
    aurc,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
    wilson_interval,
)
from answerdesk.policy.plots import plot_reliability, plot_risk_coverage
from answerdesk.policy.synthetic import (
    POPULATION_VERSION,
    SyntheticConfig,
    SyntheticItem,
    SyntheticPopulation,
    generate_population,
    item_passages,
)
from answerdesk.policy.thresholds import ThresholdTable, build_threshold_table

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default synthetic-study configuration (a code constant, not a gated config key).
DEFAULT_CALIBRATION_CONFIG = Path("configs") / "policy" / "calibration.yaml"

#: Pooled risk ceiling from requirements §8.2 ("pooled must also satisfy ≤ 0.05").
POOLED_RISK_TARGET = 0.05

#: E-1 slack on the Wilson upper bound ("UCB ≤ α + 0.03").
RISK_UCB_SLACK = 0.03


def default_calibration_config_path() -> Path:
    return _REPO_ROOT / DEFAULT_CALIBRATION_CONFIG


def load_synthetic_config(path: str | Path | None = None) -> SyntheticConfig:
    file_path = Path(path) if path is not None else default_calibration_config_path()
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration config {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CalibrationError(f"calibration config {file_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise CalibrationError(f"calibration config {file_path} must be a mapping")
    population = raw.get("population", raw)
    if not isinstance(population, dict):
        raise CalibrationError(f"calibration config {file_path}: 'population' must be a mapping")
    try:
        return SyntheticConfig.model_validate(population)
    except Exception as exc:
        raise CalibrationError(f"calibration config {file_path} is invalid: {exc}") from exc


def _scores_by_bucket(items: Sequence[SyntheticItem]) -> dict[IntentBucket, list[float]]:
    grouped: dict[IntentBucket, list[float]] = {}
    for item in items:
        grouped.setdefault(item.bucket, []).append(item.risk_score)
    return grouped


def _alpha_map(settings: Settings) -> dict[IntentBucket, float]:
    return {bucket: float(settings.policy.alpha[bucket]) for bucket in IntentBucket}


def _evaluate(
    population: SyntheticPopulation,
    *,
    conformal: ConformalSet,
    thresholds: ThresholdTable,
    cost_set: CostSet,
    guards: GuardConfig,
) -> tuple[PooledMetrics, list[BucketMetrics], dict[str, object]]:
    """Run the real decision engine over the held-out split and measure E-1…E-6."""
    engine = DecisionEngine(
        thresholds=thresholds,
        conformal=conformal,
        cost_set=cost_set,
        guards=guards,
        artifact_hash="",
    )
    per_bucket: list[BucketMetrics] = []
    pooled_answered = 0
    pooled_errors = 0
    pooled_answerable = 0
    pooled_answered_answerable = 0
    escalated_bad = 0
    escalated = 0
    escalated_unsafe = 0
    unsafe_total = 0
    unsafe_caught = 0

    probs: list[float] = []
    labels: list[bool] = []

    for bucket, items in population.by_bucket(population.test).items():
        entry = thresholds.for_bucket(bucket)
        answered = 0
        errors = 0
        answerable = 0
        answered_answerable = 0
        for item in items:
            outcome = engine.evaluate(
                p_correct=item.p_hat,
                bucket=item.bucket,
                query=item.query,
                passages=item_passages(item),
                intent_label=item.bucket.value,
            )
            decision = outcome.decision
            probs.append(item.p_hat)
            labels.append(item.correct)
            if item.unsafe:
                unsafe_total += 1
                if decision.action is not Action.ANSWER:
                    unsafe_caught += 1
            if item.answerable:
                answerable += 1
            if decision.action is Action.ANSWER:
                answered += 1
                if item.answerable:
                    answered_answerable += 1
                if not item.correct:
                    errors += 1
            else:
                escalated += 1
                if item.unsafe:
                    escalated_unsafe += 1
                if item.unsafe or not item.correct:
                    escalated_bad += 1
        pooled_answered += answered
        pooled_errors += errors
        pooled_answerable += answerable
        pooled_answered_answerable += answered_answerable
        risk = errors / answered if answered else 0.0
        low, high = wilson_interval(errors, answered)
        per_bucket.append(
            BucketMetrics(
                bucket=bucket,
                n_calibration=len(population.by_bucket(population.calibration)[bucket]),
                n_test=len(items),
                alpha_bucket=entry.alpha_bucket,
                alpha_final=entry.alpha_final,
                q_hat=entry.q_hat,
                answer_min_probability=entry.answer_min_probability,
                coverage=answered / len(items) if items else 0.0,
                n_answered=answered,
                n_answerable=answerable,
                n_answered_answerable=answered_answerable,
                coverage_answerable=(answered_answerable / answerable if answerable else 0.0),
                n_errors=errors,
                realised_risk=round(risk, 10),
                risk_wilson_low=round(low, 10),
                risk_wilson_high=round(high, 10),
                ece_width=round(expected_calibration_error(probs_of(items), labels_of(items)), 10),
                ece_mass=round(
                    expected_calibration_error(probs_of(items), labels_of(items), mode="mass"),
                    10,
                ),
                brier=round(brier_score(probs_of(items), labels_of(items)), 10),
            )
        )

    total_test = len(probs)
    pooled_risk = pooled_errors / pooled_answered if pooled_answered else 0.0
    low, high = wilson_interval(pooled_errors, pooled_answered)
    curve = risk_coverage_curve(probs, labels)
    pooled = PooledMetrics(
        n_calibration=len(population.calibration),
        n_test=total_test,
        coverage=pooled_answered / total_test if total_test else 0.0,
        n_answered=pooled_answered,
        n_answerable=pooled_answerable,
        n_answered_answerable=pooled_answered_answerable,
        coverage_answerable=(
            pooled_answered_answerable / pooled_answerable if pooled_answerable else 0.0
        ),
        n_errors=pooled_errors,
        realised_risk=round(pooled_risk, 10),
        risk_wilson_low=round(low, 10),
        risk_wilson_high=round(high, 10),
        ece_width=round(expected_calibration_error(probs, labels), 10),
        ece_mass=round(expected_calibration_error(probs, labels, mode="mass"), 10),
        brier=round(brier_score(probs, labels), 10),
        aurc=round(aurc(curve), 10),
        must_escalate_recall=round(unsafe_caught / unsafe_total if unsafe_total else 1.0, 10),
        escalation_precision=round(escalated_bad / escalated if escalated else 0.0, 10),
        escalation_precision_safety=round(escalated_unsafe / escalated if escalated else 0.0, 10),
        realised_risk_target=POOLED_RISK_TARGET,
    )
    extras: dict[str, object] = {
        "risk_coverage": [
            {
                "coverage": round(point.coverage, 6),
                "risk": round(point.risk, 6),
                "threshold": round(point.threshold, 6),
                "n_selected": point.n_selected,
            }
            for point in curve
        ],
        "reliability": [
            {
                "index": row.index,
                "lower": round(row.lower, 6),
                "upper": round(row.upper, 6),
                "count": row.count,
                "mean_confidence": round(row.mean_confidence, 6),
                "accuracy": round(row.accuracy, 6),
            }
            for row in reliability_bins(probs, labels)
        ],
        "buckets_with_infeasible_conformal": [
            bucket.value for bucket in conformal.infeasible_buckets()
        ],
    }
    return pooled, per_bucket, extras


def probs_of(items: Sequence[SyntheticItem]) -> list[float]:
    return [item.p_hat for item in items]


def labels_of(items: Sequence[SyntheticItem]) -> list[bool]:
    return [item.correct for item in items]


def build_artifact(
    *,
    population: SyntheticPopulation,
    cost_set: CostSet,
    alphas: Mapping[IntentBucket, float],
    guards: GuardConfig,
    now: datetime | None = None,
) -> tuple[CalibrationArtifact, dict[str, object]]:
    """Fit conformal + thresholds, measure the held-out split, seal the artifact."""
    calibration_scores = _scores_by_bucket(population.calibration)
    for bucket, scores in calibration_scores.items():
        required = minimum_calibration_size(alphas[bucket])
        if len(scores) < required:
            raise CalibrationError(
                f"bucket {bucket.value!r} has {len(scores)} calibration items but alpha="
                f"{alphas[bucket]} needs at least {required} (§8.1)"
            )
    conformal = fit_certificates(
        calibration_scores,
        alphas,
        calibration_hash=population.calibration_hash,
        scorer_name="synthetic-population",
        scorer_version=POPULATION_VERSION,
    )
    thresholds = build_threshold_table(
        conformal_set=conformal,
        cost_set=cost_set,
        alpha_by_bucket=alphas,
        calibration_hash=population.calibration_hash,
    )
    pooled, bucket_metrics, extras = _evaluate(
        population,
        conformal=conformal,
        thresholds=thresholds,
        cost_set=cost_set,
        guards=guards,
    )
    created = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    artifact = CalibrationArtifact(
        artifact_version=ARTIFACT_VERSION,
        created_at=created,
        corpus_version=population.config.corpus_version,
        calibration_hash=population.calibration_hash,
        test_hash=population.test_hash,
        split=(
            f"synthetic-{population.config.calibration_fraction:.2f}/"
            f"{1 - population.config.calibration_fraction:.2f}"
        ),
        scorer_name="synthetic-population",
        scorer_version=POPULATION_VERSION,
        model_id="none (synthetic latent probability)",
        costs=cost_set,
        cost_hash=cost_set.hash(),
        conformal=conformal,
        thresholds=thresholds,
        pooled=pooled,
        buckets=bucket_metrics,
        notes=(
            "Synthetic study: proves the calibration/decision machinery, not a model's "
            "calibration (docs/escalation.md §7)."
        ),
    ).sealed()
    metrics: dict[str, object] = {
        "artifact_hash": artifact.artifact_hash,
        "threshold_ref": artifact.threshold_ref(),
        "population": population.summary(),
        "costs": cost_set.canonical(),
        "cost_hash": cost_set.hash(),
        "alphas": {bucket.value: alphas[bucket] for bucket in IntentBucket},
        "pooled": pooled.model_dump(mode="json"),
        "buckets": [row.model_dump(mode="json") for row in bucket_metrics],
        **extras,
    }
    return artifact, metrics


class GateCheck(Contract):
    """One release-gate check with its value, target and verdict."""

    check: str
    value: float
    target: float
    passed: bool


def _gate_rows(artifact: CalibrationArtifact) -> list[GateCheck]:
    """E-1…E-6 style checks with the explicit slack the requirements grant."""
    rows: list[GateCheck] = []
    for row in artifact.buckets:
        rows.append(
            GateCheck(
                check=f"E-1 risk<=alpha [{row.bucket.value}]",
                value=row.realised_risk,
                target=row.alpha_final,
                passed=row.realised_risk <= row.alpha_final,
            )
        )
        rows.append(
            GateCheck(
                check=(
                    f"E-1 risk UCB<=alpha+{RISK_UCB_SLACK} [{row.bucket.value}]"
                    + (" (n/a: 0 answered)" if row.n_answered == 0 else "")
                ),
                value=row.risk_wilson_high if row.n_answered else 0.0,
                target=row.alpha_final + RISK_UCB_SLACK,
                passed=(
                    row.n_answered == 0 or row.risk_wilson_high <= row.alpha_final + RISK_UCB_SLACK
                ),
            )
        )
    pooled = artifact.pooled
    rows.extend(
        [
            GateCheck(
                check="E-1 pooled risk<=0.05",
                value=pooled.realised_risk,
                target=POOLED_RISK_TARGET,
                passed=pooled.realised_risk <= POOLED_RISK_TARGET,
            ),
            GateCheck(
                check="E-2 must-escalate recall>=0.95",
                value=pooled.must_escalate_recall,
                target=0.95,
                passed=pooled.must_escalate_recall >= 0.95,
            ),
            GateCheck(
                check="E-3 escalation precision>=0.55 (unsafe or would-be-wrong)",
                value=pooled.escalation_precision,
                target=0.55,
                passed=pooled.escalation_precision >= 0.55,
            ),
            GateCheck(
                check="E-3b escalation precision (safety-only reading)",
                value=pooled.escalation_precision_safety,
                target=0.55,
                passed=pooled.escalation_precision_safety >= 0.55,
            ),
            GateCheck(
                check="E-4 coverage on answerable items>=0.60",
                value=pooled.coverage_answerable,
                target=0.60,
                passed=pooled.coverage_answerable >= 0.60,
            ),
            GateCheck(
                check="E-5 ECE(width)<=0.08",
                value=pooled.ece_width,
                target=0.08,
                passed=pooled.ece_width <= 0.08,
            ),
            GateCheck(
                check="E-5 Brier<=0.15",
                value=pooled.brier,
                target=0.15,
                passed=pooled.brier <= 0.15,
            ),
        ]
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _markdown_report(
    artifact: CalibrationArtifact,
    metrics: Mapping[str, object],
    gates: Sequence[GateCheck],
) -> str:
    pooled = artifact.pooled
    lines = [
        "# Escalation calibration report (synthetic)",
        "",
        f"* artifact: `{artifact.threshold_ref()}` (sha256 `{artifact.artifact_hash[:16]}…`)",
        f"* created: {artifact.created_at}",
        f"* scorer: `{artifact.scorer_name}` {artifact.scorer_version}",
        f"* calibration hash: `{artifact.calibration_hash[:16]}…`; "
        f"test hash: `{artifact.test_hash[:16]}…`",
        f"* costs: C_wrong={artifact.costs.c_wrong}, C_abstain={artifact.costs.c_abstain}, "
        f"C_delay={artifact.costs.c_delay} (cost hash `{artifact.cost_hash[:12]}…`)",
        f"* split: {artifact.split}; n_calibration={pooled.n_calibration}, n_test={pooled.n_test}",
        "",
        "## Pooled metrics",
        "",
        "| metric | value | target | pass |",
        "|---|---|---|---|",
    ]
    for gate in gates:
        passed = "yes" if gate.passed else "**NO**"
        lines.append(f"| {gate.check} | {gate.value:.4f} | {gate.target:.4f} | {passed} |")
    lines.extend(
        [
            f"| E-6 AURC (diagnostic) | {pooled.aurc:.4f} | — | — |",
            "",
            "## Per bucket",
            "",
            "| bucket | alpha | q_hat | answer_min_p | coverage(ans.) | risk | UCB | ECE | Brier |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for bucket_row in artifact.buckets:
        lines.append(
            f"| {bucket_row.bucket.value} | {bucket_row.alpha_final:.3f} | "
            f"{bucket_row.q_hat:.4f} | {bucket_row.answer_min_probability:.4f} | "
            f"{bucket_row.coverage_answerable:.3f} | {bucket_row.realised_risk:.4f} | "
            f"{bucket_row.risk_wilson_high:.4f} | {bucket_row.ece_width:.4f} | "
            f"{bucket_row.brier:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Honest limitations",
            "",
            "1. This is a **synthetic** study: `p̂` is a simulated calibrated scorer, so",
            "   the numbers prove the machinery (quantile, thresholds, guards, metrics), not",
            "   a model's calibration. t11 re-runs the same code on real scored traffic;",
            "   until then no quality claim is made.",
            "2. ECE depends on the binning rule; both equal-width and equal-mass values are in",
            "   `calibration-metrics.json`, and the reliability plot shows the bins.",
            "3. The conformal risk statement additionally needs a calibrated scorer; the",
            "   coverage guarantee (1 − α) is distribution-free only under exchangeability",
            "   (§8.6 monitoring).",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    artifact: CalibrationArtifact,
    metrics: Mapping[str, object],
    population: SyntheticPopulation,
    out_dir: str | Path,
) -> list[Path]:
    """Write the artifact, metrics, CSVs, plots and the markdown report."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(artifact.save(target / "calibration-artifact.json"))
    metrics_path = target / "calibration-metrics.json"
    metrics_path.write_text(
        json.dumps(dict(metrics), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(metrics_path)

    curve = metrics.get("risk_coverage", [])
    written.append(
        _write_csv(
            target / "risk-coverage.csv",
            curve if isinstance(curve, list) else [],
            ["coverage", "risk", "threshold", "n_selected"],
        )
    )
    reliability = metrics.get("reliability", [])
    written.append(
        _write_csv(
            target / "reliability.csv",
            reliability if isinstance(reliability, list) else [],
            ["index", "lower", "upper", "count", "mean_confidence", "accuracy"],
        )
    )
    pooled = artifact.pooled
    coverage = pooled.coverage
    from answerdesk.policy.metrics import RiskCoveragePoint

    points = [
        RiskCoveragePoint(
            coverage=float(row["coverage"]),
            risk=float(row["risk"]),
            threshold=float(row["threshold"]),
            n_selected=int(row["n_selected"]),
        )
        for row in (curve if isinstance(curve, list) else [])
    ]
    written.extend(
        plot_risk_coverage(
            points,
            target / "risk-coverage",
            alpha=artifact.pooled.realised_risk_target,
            coverage_at_operating_point=coverage,
        )
    )
    bins = reliability_bins(probs_of(population.test), labels_of(population.test))
    written.extend(plot_reliability(bins, target / "reliability", ece=pooled.ece_width))
    gates = _gate_rows(artifact)
    report_path = target / "report.md"
    report_path.write_text(_markdown_report(artifact, metrics, gates), encoding="utf-8")
    written.append(report_path)
    gates_path = target / "gate-checks.json"
    gates_path.write_text(
        json.dumps([gate.model_dump(mode="json") for gate in gates], indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(gates_path)
    return written


def run(
    *,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    costs_path: str | Path | None = None,
    guards_path: str | Path | None = None,
    seed: int | None = None,
    n_items: int | None = None,
    now: datetime | None = None,
) -> tuple[CalibrationArtifact, list[Path]]:
    """The whole calibration run, callable from tests without going through argparse."""
    population_config = load_synthetic_config(config_path)
    if seed is not None:
        population_config = population_config.model_copy(update={"seed": seed})
    if n_items is not None:
        population_config = population_config.model_copy(update={"n_items": n_items})
    population = generate_population(population_config)

    cost_set = load_cost_set(costs_path)
    guards = load_guard_config(guards_path)
    settings = load_settings(profile="offline")
    weights = _alpha_map(settings)
    artifact, metrics = build_artifact(
        population=population,
        cost_set=cost_set,
        alphas=weights,
        guards=guards,
        now=now,
    )
    written = write_outputs(
        artifact=artifact, metrics=metrics, population=population, out_dir=out_dir
    )
    return artifact, written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m answerdesk.policy.calibrate",
        description="Fit the conformal + cost-sensitive escalation artifact.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="calibrate on the synthetic population (the only source in v1.0 offline)",
    )
    parser.add_argument(
        "--out",
        default="reports/policy-smoke",
        help="output directory for the artifact, metrics and plots",
    )
    parser.add_argument(
        "--config", default=None, help="synthetic-study YAML (default configs/policy)"
    )
    parser.add_argument("--costs", default=None, help="cost matrix YAML")
    parser.add_argument("--guards", default=None, help="guard policy YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the generator seed")
    parser.add_argument("--n-items", type=int, default=None, help="override the population size")
    parser.add_argument("--quiet", action="store_true", help="only print the artifact path")
    args = parser.parse_args(argv)

    if not args.synthetic:
        parser.error(
            "--synthetic is required in v1.0: a live calibration set is produced by the "
            "evaluation harness (t7/t11), not by this command"
        )
    try:
        artifact, written = run(
            out_dir=args.out,
            config_path=args.config,
            costs_path=args.costs,
            guards_path=args.guards,
            seed=args.seed,
            n_items=args.n_items,
        )
    except (CalibrationError, CostModelError, GuardError) as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2

    if args.quiet:
        print(artifact.threshold_ref())
        return 0

    pooled = artifact.pooled
    print(f"artifact       {artifact.threshold_ref()} (sha256 {artifact.artifact_hash[:16]}…)")
    print(f"created        {artifact.created_at}")
    print(f"scorer         {artifact.scorer_name} {artifact.scorer_version}")
    print(f"split          {artifact.split}; n_cal={pooled.n_calibration} n_test={pooled.n_test}")
    print(
        f"pooled risk    {pooled.realised_risk:.4f} "
        f"[{pooled.risk_wilson_low:.4f}, {pooled.risk_wilson_high:.4f}] "
        f"(target ≤ {pooled.realised_risk_target})"
    )
    print(
        f"coverage       {pooled.coverage:.4f} of all turns / "
        f"{pooled.coverage_answerable:.4f} of answerable turns (E-4)"
    )
    print(f"ECE / Brier    {pooled.ece_width:.4f} / {pooled.brier:.4f}")
    print(f"AURC           {pooled.aurc:.4f}")
    print(
        f"E-2 / E-3      recall {pooled.must_escalate_recall:.4f} / "
        f"precision {pooled.escalation_precision:.4f}"
    )
    for row in artifact.buckets:
        print(
            f"  {row.bucket.value:<14} alpha={row.alpha_final:.3f} q_hat={row.q_hat:.4f} "
            f"answer>= {row.answer_min_probability:.4f} coverage={row.coverage:.3f} "
            f"risk={row.realised_risk:.4f} (UCB {row.risk_wilson_high:.4f})"
        )
    print(f"wrote {len(written)} files under {Path(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
