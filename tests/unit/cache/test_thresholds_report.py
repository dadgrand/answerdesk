"""Threshold calibration to the wrong-hit budget, and the precision/recall report.

# AT-CAC-07  # AT-CAC-08
"""

from __future__ import annotations

from pathlib import Path

import pytest

from answerdesk.cache.replay import (
    embed_replay_set,
    generate_replay_set,
    similarities_by_cohort,
    wrong_by_cohort,
)
from answerdesk.cache.report import (
    HIT_RATE_FLOOR,
    evaluate_cache,
    hit_rate_error_curve,
    render_markdown,
    write_report,
)
from answerdesk.cache.thresholds import (
    DEFAULT_EPSILON,
    MIN_THRESHOLD,
    ThresholdTable,
    build_threshold_table,
    calibrate_threshold,
)
from answerdesk.contracts import Embedder

# --------------------------------------------------------------------------------------
# The sweep itself
# --------------------------------------------------------------------------------------


def test_smallest_feasible_threshold_wins() -> None:
    """Sims: 0.99 (same), 0.97 (wrong), 0.90 (same) → τ must exceed 0.97 but stay minimal."""
    row = calibrate_threshold(
        cohort="informational:cyrillic:len1",
        similarities=[0.99, 0.97, 0.90],
        wrong=[False, True, False],
        epsilon=0.01,
    )
    assert row.threshold == pytest.approx(0.99)
    assert row.wrong_hits == 0
    assert row.feasible is True


def test_entity_incompatible_pairs_do_not_constrain_the_threshold() -> None:
    """A wrong pair the G-2 guard would block is not a wrong hit."""
    row = calibrate_threshold(
        cohort="transactional:cyrillic:len1",
        similarities=[0.99, 0.97],
        wrong=[False, True],
        compatible=[True, False],
        epsilon=0.01,
    )
    # The sweep can only pick an observed similarity at or above the floor; the smallest
    # feasible one here is 0.97 (the incompatible wrong pair is dropped by G-2).
    assert row.threshold == pytest.approx(0.97)
    assert row.wrong_hits == 0


def test_exact_equivalents_count_as_hits_at_any_threshold() -> None:
    row = calibrate_threshold(
        cohort="informational:cyrillic:len1",
        similarities=[0.5, 0.4],
        wrong=[False, False],
        exact=[True, False],
        epsilon=0.01,
    )
    assert row.hits >= 1
    # No observed similarity clears the floor, so the default threshold is used and the hit
    # comes from the exact path (FR-CAC-01).
    assert row.threshold == pytest.approx(0.93)


def test_no_feasible_threshold_falls_back_to_no_semantic_hits() -> None:
    row = calibrate_threshold(
        cohort="informational:cyrillic:len1",
        similarities=[0.99, 0.99, 0.99],
        wrong=[True, True, True],
        epsilon=0.01,
    )
    assert row.feasible is False
    assert row.threshold == pytest.approx(1.0)
    assert row.hits == 0


def test_threshold_floor_is_enforced() -> None:
    row = calibrate_threshold(
        cohort="x",
        similarities=[0.85, 0.86],
        wrong=[False, False],
        epsilon=0.01,
    )
    assert row.threshold >= MIN_THRESHOLD


def test_table_round_trips_through_json(tmp_path: Path) -> None:
    table = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash="f" * 64,
        epsilon=DEFAULT_EPSILON,
        default_threshold=0.93,
    )
    path = table.save(tmp_path / "thresholds.json")
    loaded = ThresholdTable.load(path)
    assert loaded.replay_hash == table.replay_hash
    assert loaded.threshold_for("unknown-cohort") == pytest.approx(0.93)


# --------------------------------------------------------------------------------------
# The replay set and the report
# --------------------------------------------------------------------------------------


def test_replay_set_is_deterministic_and_adversarial() -> None:
    first = generate_replay_set(seed=3, pairs_per_cohort=12)
    second = generate_replay_set(seed=3, pairs_per_cohort=12)
    other = generate_replay_set(seed=4, pairs_per_cohort=12)
    assert first.replay_hash == second.replay_hash
    assert other.replay_hash != first.replay_hash
    families = {pair.family for pair in first.pairs}
    assert families == {"paraphrase", "near_miss", "distractor"}
    assert all(pair.wrong_if_hit == (pair.family != "paraphrase") for pair in first.pairs)
    assert len({pair.locale for pair in first.pairs}) == 6


def test_calibrated_table_meets_the_budget_on_the_replay_set(embedder: Embedder) -> None:
    """K-1: wrong-hit rate ≤ ε with the Wilson UCB inside 2ε (pooled, §9.3)."""
    replay = generate_replay_set(seed=5, pairs_per_cohort=120)
    embedded = embed_replay_set(replay, embedder)
    table = build_threshold_table(
        similarities_by_cohort=similarities_by_cohort(embedded),
        wrong_by_cohort=wrong_by_cohort(embedded),
        replay_hash=replay.replay_hash,
        epsilon=0.01,
    )
    evaluation = evaluate_cache(embedded, table=table, replay_hash=replay.replay_hash)
    assert evaluation.wrong_hit_rate <= evaluation.epsilon
    assert evaluation.wrong_hit_ucb <= 2 * evaluation.epsilon
    assert evaluation.passes_k1() is True
    assert evaluation.hits > 0
    assert 0.0 <= evaluation.precision <= 1.0
    assert 0.0 <= evaluation.recall <= 1.0


def test_informational_hit_rate_clears_the_floor(embedder: Embedder) -> None:
    """K-2 exists so K-1 cannot be satisfied by turning the cache off."""
    replay = generate_replay_set(seed=6, pairs_per_cohort=60)
    embedded = embed_replay_set(replay, embedder)
    table = build_threshold_table(
        similarities_by_cohort=similarities_by_cohort(embedded),
        wrong_by_cohort=wrong_by_cohort(embedded),
        replay_hash=replay.replay_hash,
    )
    evaluation = evaluate_cache(embedded, table=table, replay_hash=replay.replay_hash)
    assert evaluation.informational_hit_rate >= HIT_RATE_FLOOR


def test_wrong_hits_are_measured_not_assumed(embedder: Embedder) -> None:
    """A deliberately reckless table (τ=0) must produce wrong hits — the metric works."""
    replay = generate_replay_set(seed=7, pairs_per_cohort=30)
    embedded = embed_replay_set(replay, embedder)
    reckless = ThresholdTable(
        calibrated_at="2026-10-21T00:00:00+00:00",
        replay_hash=replay.replay_hash,
        epsilon=0.01,
        default_threshold=0.0,
    )
    evaluation = evaluate_cache(embedded, table=reckless, replay_hash=replay.replay_hash)
    assert evaluation.wrong_hits > 0
    assert evaluation.passes_k1() is False


def test_report_files_are_written(embedder: Embedder, tmp_path: Path) -> None:
    """# AT-CAC-08 — the precision/recall report is an artifact, not a printout."""
    replay = generate_replay_set(seed=8, pairs_per_cohort=40)
    embedded = embed_replay_set(replay, embedder)
    table = build_threshold_table(
        similarities_by_cohort=similarities_by_cohort(embedded),
        wrong_by_cohort=wrong_by_cohort(embedded),
        replay_hash=replay.replay_hash,
    )
    evaluation = evaluate_cache(embedded, table=table, replay_hash=replay.replay_hash)
    written = write_report(evaluation=evaluation, table=table, out_dir=tmp_path)
    names = {path.name for path in written}
    assert "cache-thresholds.json" in names
    assert "cache-report.json" in names
    assert "cache-report.md" in names
    markdown = (tmp_path / "cache-report.md").read_text(encoding="utf-8")
    assert "K-1" in markdown and "K-2" in markdown
    assert "precision" in markdown and "recall" in markdown
    assert render_markdown(evaluation, table=table) == markdown


def test_hit_rate_error_curve_is_monotone_in_error(embedder: Embedder) -> None:
    replay = generate_replay_set(seed=9, pairs_per_cohort=30)
    embedded = embed_replay_set(replay, embedder)
    curve = hit_rate_error_curve(embedded)
    assert curve
    errors = [point[0] for point in curve]
    hit_rates = [point[1] for point in curve]
    assert all(0.0 <= error <= 1.0 for error in errors)
    assert all(0.0 <= rate <= 1.0 for rate in hit_rates)
    # A safe operating point must exist: the floor hit rate with zero wrong hits.
    assert any(
        hit_rate >= HIT_RATE_FLOOR and error == 0.0
        for error, hit_rate in zip(errors, hit_rates, strict=True)
    )
