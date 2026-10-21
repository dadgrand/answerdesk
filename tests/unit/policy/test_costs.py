"""Cost model: derived bands, provenance, and no hard-coded thresholds.

# AT-ESC-03  # AT-ESC-06
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from answerdesk.config import load_settings
from answerdesk.policy.costs import (
    CostProvenance,
    CostSet,
    check_cost_drift,
    cost_set_from_settings,
    load_cost_set,
)
from answerdesk.policy.errors import CostModelError


def test_reference_costs_give_the_documented_bands() -> None:
    """§8.3 defaults: answer ≥ 0.95, clarify ∈ [0.90, 0.95), escalate < 0.90."""
    cost_set = load_cost_set()
    assert cost_set.c_wrong == pytest.approx(1.0)
    assert cost_set.c_abstain == pytest.approx(0.10)
    assert cost_set.c_delay == pytest.approx(0.05)
    assert cost_set.answer_min_probability == pytest.approx(0.95)
    assert cost_set.clarify_min_probability == pytest.approx(0.90)
    assert cost_set.alpha_from_costs == pytest.approx(0.10)
    assert cost_set.cost_ratio == pytest.approx(10.0)


def test_costs_are_configuration_not_constants(make_cost_set: Callable[..., CostSet]) -> None:
    """A different ratio moves the operating point exactly as the formula predicts."""
    strict = make_cost_set(c_wrong=1.0, c_abstain=0.10, c_delay=0.01)
    assert strict.answer_min_probability == pytest.approx(0.99)
    lax = make_cost_set(c_wrong=1.0, c_abstain=0.20, c_delay=0.10)
    assert lax.answer_min_probability == pytest.approx(0.90)
    assert lax.clarify_min_probability == pytest.approx(0.80)


def test_cost_set_hash_tracks_the_numbers(make_cost_set: Callable[..., CostSet]) -> None:
    first = make_cost_set()
    second = make_cost_set()
    assert first.hash() == second.hash()
    assert first.hash() != make_cost_set(c_wrong=2.0).hash()


def test_invalid_cost_ordering_is_rejected() -> None:
    with pytest.raises(ValueError, match="c_abstain must be <= c_wrong"):
        CostSet(
            c_wrong=1.0,
            c_abstain=2.0,
            c_delay=0.1,
            provenance=CostProvenance(source="s", owner="o", date="2026-10-21"),
        )
    with pytest.raises(ValueError, match="c_delay must be <= c_abstain"):
        CostSet(
            c_wrong=1.0,
            c_abstain=0.1,
            c_delay=0.2,
            provenance=CostProvenance(source="s", owner="o", date="2026-10-21"),
        )
    with pytest.raises(ValueError):
        CostSet(
            c_wrong=0.0,
            c_abstain=0.1,
            c_delay=0.1,
            provenance=CostProvenance(source="s", owner="o", date="2026-10-21"),
        )


def test_a_cost_without_provenance_is_invalid_config(tmp_path) -> None:
    """§8.3's rule, executable: no provenance ⇒ load fails with an explanation."""
    path = tmp_path / "costs.yaml"
    path.write_text("costs:\n  c_wrong: 1.0\n  c_abstain: 0.1\n  c_delay: 0.05\n", encoding="utf-8")
    with pytest.raises(CostModelError, match="provenance"):
        load_cost_set(path)


def test_cost_file_round_trip(tmp_path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(
        "costs:\n"
        "  c_wrong: 2.0\n"
        "  c_abstain: 0.2\n"
        "  c_delay: 0.05\n"
        "provenance:\n"
        "  source: pilot measurement\n"
        "  owner: ops\n"
        "  date: '2026-10-21'\n"
        "  note: measured during the pilot\n",
        encoding="utf-8",
    )
    cost_set = load_cost_set(path)
    assert cost_set.c_wrong == 2.0
    assert cost_set.answer_min_probability == pytest.approx(0.975)
    assert cost_set.provenance.source == "pilot measurement"


def test_settings_and_file_agree_today() -> None:
    """A drift check keeps `configs/default.yaml` and `configs/policy/costs.yaml` honest."""
    settings = load_settings(profile="offline")
    assert check_cost_drift(settings, load_cost_set()) == []
    from_settings = cost_set_from_settings(settings)
    assert from_settings.c_wrong == pytest.approx(1.0)
    assert "configs/default.yaml" in from_settings.provenance.source


def test_drift_is_detected(make_cost_set: Callable[..., CostSet]) -> None:
    settings = load_settings(profile="offline")
    drifted = make_cost_set(c_wrong=3.0)
    problems = check_cost_drift(settings, drifted)
    assert problems and "c_wrong" in problems[0]
