"""``python -m answerdesk.cache.calibrate`` — calibrate thresholds to the wrong-hit budget.

Flow (requirements §9.3, §9.5):

1. generate the adversarial near-miss replay set (``replay.py``);
2. measure pair similarities with the configured embedder (the CI hashing embedder by
   default, so the run is deterministic and offline);
3. choose, per cohort, the smallest threshold whose measured wrong-hit rate **and** Wilson
   upper bound fit the budget (``thresholds.calibrate_threshold``);
4. evaluate the resulting table on the same replay set and write the artifacts:
   ``cache-thresholds.json`` (what the service loads), ``cache-report.{json,md}`` (K-1…K-6
   numbers and precision/recall) and a hit-rate-vs-error plot (§9.6).

Exit codes: ``0`` success, ``2`` when no cohort could be calibrated (an unusable artifact is
worse than a loud failure).

    # AT-CAC-07  # AT-CAC-08
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from answerdesk.cache.errors import CacheArtifactError, CacheConfigError
from answerdesk.cache.replay import embed_replay_set, generate_replay_set
from answerdesk.cache.report import (
    CacheEvaluation,
    evaluate_cache,
    hit_rate_error_curve,
    write_report,
)
from answerdesk.cache.thresholds import (
    DEFAULT_EPSILON,
    ThresholdTable,
    build_threshold_table,
)
from answerdesk.contracts import Embedder
from answerdesk.policy.plots import plot_hit_rate_vs_error
from answerdesk.testing.doubles import HashingEmbedder


def calibrate(
    *,
    embedder: Embedder | None = None,
    seed: int = 20261020,
    pairs_per_cohort: int = 120,
    epsilon: float = DEFAULT_EPSILON,
    default_threshold: float = 0.93,
) -> tuple[ThresholdTable, CacheEvaluation, list[tuple[float, float]]]:
    """Calibrate + evaluate in one call (also used by the tests)."""
    resolved_embedder = embedder or HashingEmbedder(dimension=64)
    replay = generate_replay_set(seed=seed, pairs_per_cohort=pairs_per_cohort)
    embedded = embed_replay_set(replay, resolved_embedder)
    if not embedded:
        raise CacheArtifactError("replay set is empty; nothing to calibrate against")

    similarities: dict[str, list[float]] = {}
    wrong: dict[str, list[bool]] = {}
    compatible: dict[str, list[bool]] = {}
    exact: dict[str, list[bool]] = {}
    for item in embedded:
        similarities.setdefault(item.pair.cohort, []).append(item.similarity)
        wrong.setdefault(item.pair.cohort, []).append(item.pair.wrong_if_hit)
        compatible.setdefault(item.pair.cohort, []).append(item.entities_compatible)
        exact.setdefault(item.pair.cohort, []).append(item.exact_equivalent)

    table = build_threshold_table(
        similarities_by_cohort=similarities,
        wrong_by_cohort=wrong,
        compatible_by_cohort=compatible,
        exact_by_cohort=exact,
        replay_hash=replay.replay_hash,
        epsilon=epsilon,
        default_threshold=default_threshold,
    )
    evaluation = evaluate_cache(
        embedded, table=table, replay_hash=replay.replay_hash, epsilon=epsilon
    )
    curve = hit_rate_error_curve(embedded)
    return table, evaluation, curve


def run(
    *,
    out_dir: str | Path,
    seed: int = 20261020,
    pairs_per_cohort: int = 120,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[ThresholdTable, CacheEvaluation, list[Path]]:
    table, evaluation, curve = calibrate(
        seed=seed, pairs_per_cohort=pairs_per_cohort, epsilon=epsilon
    )
    written = write_report(evaluation=evaluation, table=table, out_dir=out_dir)
    written.extend(
        plot_hit_rate_vs_error(curve, Path(out_dir) / "hit-rate-vs-error", epsilon=epsilon)
    )
    return table, evaluation, written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m answerdesk.cache.calibrate",
        description="Calibrate semantic-cache thresholds to a wrong-hit budget.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use the synthetic near-miss replay set (the only source in v1.0 offline)",
    )
    parser.add_argument(
        "--out", default="reports/cache-smoke", help="output directory for thresholds and report"
    )
    parser.add_argument("--seed", type=int, default=20261020, help="replay-set seed")
    parser.add_argument(
        "--pairs-per-cohort", type=int, default=120, help="replay pairs generated per cohort"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="wrong-hit budget (requirements §9.5: 0.01)",
    )
    args = parser.parse_args(argv)
    if not args.synthetic:
        parser.error(
            "--synthetic is required in v1.0: the HITL near-miss replay set is produced by "
            "the evaluation harness (t7/t11), not by this command"
        )
    try:
        table, evaluation, written = run(
            out_dir=args.out,
            seed=args.seed,
            pairs_per_cohort=args.pairs_per_cohort,
            epsilon=args.epsilon,
        )
    except (CacheArtifactError, CacheConfigError) as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2

    summary = evaluation.summary_row()
    print(f"thresholds     {len(table.cohorts)} cohorts, epsilon={table.epsilon}")
    print(f"replay         {summary['n_pairs']} pairs ({evaluation.replay_hash[:16]}…)")
    print(
        f"hit rate       {summary['hit_rate']:.4f} "
        f"(informational {evaluation.informational_hit_rate:.4f}, floor 0.15)"
    )
    print(
        f"wrong hits     {summary['wrong_hits']} / {summary['hits']} "
        f"rate {summary['wrong_hit_rate']:.4f}, UCB {summary['wrong_hit_ucb']:.4f} "
        f"(budget {evaluation.epsilon})"
    )
    print(f"precision/recall {summary['precision']:.4f} / {summary['recall']:.4f}")
    for row in evaluation.cohorts:
        print(
            f"  {row.cohort:<34} tau={row.threshold:.3f} hit={row.hit_rate:.3f} "
            f"wrong={row.wrong_hits}/{row.hits} feasible={row.feasible}"
        )
    print(f"wrote {len(written)} files under {Path(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["calibrate", "main", "run"]
