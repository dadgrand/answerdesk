"""Statistics tests against closed-form values (FR-EVL-07 / AT-EVL-07).

# AT-EVL-07  # AT-DET-02
"""

from __future__ import annotations

import math

import pytest

from answerdesk.eval.stats import (
    bootstrap_interval,
    cohen_kappa,
    mcnemar,
    paired_bootstrap,
    quantile,
    wilson_interval,
    z_for,
)


def test_z_value_is_the_textbook_quantile() -> None:
    assert z_for(0.95) == pytest.approx(1.959963984540054, abs=1e-12)
    assert z_for(0.90) == pytest.approx(1.6448536269514722, abs=1e-12)


def test_wilson_closed_form() -> None:
    interval = wilson_interval(15, 20)
    # p = 0.75, z = 1.96: centre = (0.75 + 0.096)/1.192; width from the standard formula
    z = 1.959963984540054
    p = 0.75
    denominator = 1 + z * z / 20
    centre = (p + z * z / (2 * 20)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / 20 + z * z / (4 * 20 * 20))
    assert interval.value == pytest.approx(0.75)
    assert interval.low == pytest.approx(max(0.0, centre - margin))
    assert interval.high == pytest.approx(min(1.0, centre + margin))


def test_wilson_edge_cases() -> None:
    """Wilson never collapses to zero width and never leaves [0, 1]."""
    zero = wilson_interval(0, 20)
    assert zero.low == 0.0
    assert 0.0 < zero.high < 1.0
    everything = wilson_interval(20, 20)
    assert everything.high == 1.0
    assert 0.0 < everything.low < 1.0
    empty = wilson_interval(0, 0)
    assert empty.n == 0 and empty.method == "none" and empty.low == 0.0 and empty.high == 1.0


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    first = bootstrap_interval(values, samples=500, seed=11)
    second = bootstrap_interval(values, samples=500, seed=11)
    assert (first.low, first.high) == (second.low, second.high)
    assert first.value == pytest.approx(0.625)
    assert first.low <= first.value <= first.high


def test_bootstrap_degenerate_cases() -> None:
    single = bootstrap_interval([0.4], samples=100)
    assert single.low == single.high == pytest.approx(0.4)
    constant = bootstrap_interval([0.3, 0.3, 0.3], samples=100)
    assert constant.width == pytest.approx(0.0)
    empty = bootstrap_interval([], samples=100)
    assert empty.n == 0


def test_paired_bootstrap_uses_the_pairing() -> None:
    a = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    b = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    difference = paired_bootstrap(a, b, samples=500, seed=5)
    assert difference.diff == pytest.approx(2 / 8)
    assert difference.n == 8
    assert difference.low <= difference.diff <= difference.high


def test_paired_bootstrap_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [1.0])


def test_paired_bootstrap_identical_arms_have_zero_width() -> None:
    values = [1.0, 0.0, 1.0, 1.0]
    difference = paired_bootstrap(values, values, samples=100)
    assert difference.diff == 0.0
    assert difference.low == difference.high == 0.0
    assert not difference.excludes_zero


def test_mcnemar_chi_square_closed_form() -> None:
    result = mcnemar(30, 10)
    # statistic = (|30-10| - 1)^2 / 40 = 361/40
    assert result.method == "chi2"
    assert result.statistic == pytest.approx(361 / 40)
    assert result.p_value == pytest.approx(math.erfc(math.sqrt((361 / 40) / 2)))
    assert result.p_value < 0.01
    assert result.significant


def test_mcnemar_exact_fallback_below_25_discordant() -> None:
    result = mcnemar(3, 1)
    assert result.method == "exact_binomial"
    # two-sided exact p for b=3, c=1 with n=4: 2 * P(X <= 1) = 2 * (1 + 4)/16
    assert result.p_value == pytest.approx(2 * (1 + 4) / 16)


def test_mcnemar_no_discordant_pairs() -> None:
    result = mcnemar(0, 0)
    assert result.method == "n/a"
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_symmetric_counts_are_not_significant() -> None:
    result = mcnemar(5, 5)
    assert result.p_value == pytest.approx(1.0)


def test_cohen_kappa_closed_form() -> None:
    # 10 items, agreement 8/10; both raters say yes 5 times and no 5 times
    first = [True] * 5 + [False] * 5
    second = [True] * 4 + [False] + [False] * 4 + [True]
    observed = 0.8
    expected = 0.5 * 0.5 + 0.5 * 0.5
    assert cohen_kappa(first, second) == pytest.approx((observed - expected) / (1 - expected))


def test_cohen_kappa_degenerate_and_perfect() -> None:
    assert cohen_kappa([True, True], [True, True]) == 0.0  # single class: undefined, not 1
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def test_quantile_linear_interpolation() -> None:
    assert quantile([10, 20, 30, 40, 50], 0.5) == pytest.approx(30.0)
    assert quantile([10, 20, 30, 40, 50], 0.95) == pytest.approx(48.0)
    assert quantile([], 0.5) == 0.0
