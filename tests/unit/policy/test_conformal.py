"""Split conformal: the quantile, its finite-sample correction, and the guarantee it buys.

# AT-ESC-02  # AT-ESC-06
"""

from __future__ import annotations

import math
import random

import pytest
from pydantic import ValidationError

from answerdesk.contracts import IntentBucket
from answerdesk.policy.conformal import (
    ConformalCertificate,
    ConformalSet,
    conformal_quantile,
    conformal_rank,
    fit_certificate,
    fit_certificates,
    minimum_calibration_size,
)
from answerdesk.policy.errors import CalibrationError


def test_rank_is_the_n_plus_one_correction() -> None:
    """``rank = ⌈(n+1)(1−α)⌉`` — hand-computed for the three shipped α tiers."""
    assert conformal_rank(19, 0.05) == 19  # ceil(20 * 0.95) = 19
    assert conformal_rank(20, 0.05) == 20  # ceil(21 * 0.95) = 20
    assert conformal_rank(99, 0.01) == 99  # ceil(100 * 0.99) = 99
    assert conformal_rank(100, 0.01) == 100
    assert conformal_rank(49, 0.02) == 49
    assert conformal_rank(50, 0.02) == 50


def test_minimum_calibration_size_matches_the_rank_algebra() -> None:
    """``ceil(1/α) − 1``: below this the quantile cannot be placed at all."""
    assert minimum_calibration_size(0.05) == 19
    assert minimum_calibration_size(0.02) == 49
    assert minimum_calibration_size(0.01) == 99
    for alpha, expected in ((0.05, 19), (0.02, 49), (0.01, 99)):
        assert conformal_rank(expected, alpha) <= expected
        assert conformal_rank(expected - 1, alpha) > expected - 1


def test_quantile_is_the_ranked_value() -> None:
    scores = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0]
    quantile = conformal_quantile(scores, 0.05)
    # n=10 -> rank = ceil(11 * 0.95) = 11 > 10 -> infeasible
    assert quantile.feasible is False
    assert quantile.q_hat == math.inf
    quantile = conformal_quantile(scores, 0.5)
    # n=10 -> rank = ceil(11 * 0.5) = 6 -> 6th smallest = 0.6
    assert quantile.feasible is True
    assert quantile.q_hat == pytest.approx(0.6)


def test_empty_and_invalid_inputs_are_rejected() -> None:
    with pytest.raises(CalibrationError):
        conformal_quantile([], 0.05)
    with pytest.raises(CalibrationError):
        conformal_quantile([0.5], 0.0)
    with pytest.raises(CalibrationError):
        conformal_quantile([0.5], 1.0)
    with pytest.raises(CalibrationError):
        conformal_quantile([1.5], 0.5)


def test_certificate_records_its_provenance() -> None:
    certificate = fit_certificate(
        [index / 100 for index in range(100)],
        bucket=IntentBucket.INFORMATIONAL,
        alpha=0.05,
        calibration_hash="a" * 64,
        scorer_name="overlap-baseline",
        scorer_version="overlap-1.0.0",
    )
    assert certificate.feasible is True
    assert certificate.coverage_lower_bound == pytest.approx(0.95)
    assert certificate.calibration_hash == "a" * 64
    assert "feasible" in certificate.description
    assert certificate.q_hat == pytest.approx(0.95)  # rank 96 of 100 values 0.00..0.99


def test_infeasible_certificate_never_claims_conformity() -> None:
    certificate = fit_certificate(
        [0.1, 0.2],
        bucket=IntentBucket.ACCOUNT_LEGAL,
        alpha=0.01,
        calibration_hash="b" * 64,
    )
    assert certificate.feasible is False
    conformal = ConformalSet(certificates=[certificate])
    assert conformal.conforming(0.0, IntentBucket.ACCOUNT_LEGAL) is False
    assert conformal.infeasible_buckets() == [IntentBucket.ACCOUNT_LEGAL]


def test_conformal_set_lookup() -> None:
    certificates = [
        fit_certificate(
            [index / 20 for index in range(20)],
            bucket=bucket,
            alpha=0.05,
            calibration_hash="c" * 64,
        )
        for bucket in (IntentBucket.INFORMATIONAL, IntentBucket.TRANSACTIONAL)
    ]
    conformal = ConformalSet(certificates=certificates)
    assert conformal.buckets() == [IntentBucket.INFORMATIONAL, IntentBucket.TRANSACTIONAL]
    threshold = conformal.for_bucket(IntentBucket.INFORMATIONAL).q_hat
    assert conformal.conforming(threshold, IntentBucket.INFORMATIONAL) is True
    assert conformal.conforming(threshold + 1e-9, IntentBucket.INFORMATIONAL) is False
    with pytest.raises(CalibrationError):
        conformal.for_bucket(IntentBucket.ACCOUNT_LEGAL)


def test_fit_certificates_skips_absent_buckets() -> None:
    scores = {
        IntentBucket.INFORMATIONAL: [index / 50 for index in range(50)],
        IntentBucket.ACCOUNT_LEGAL: [index / 200 for index in range(200)],
    }
    alphas = {
        IntentBucket.INFORMATIONAL: 0.05,
        IntentBucket.TRANSACTIONAL: 0.02,
        IntentBucket.ACCOUNT_LEGAL: 0.01,
    }
    conformal = fit_certificates(scores, alphas, calibration_hash="d" * 64)
    assert conformal.buckets() == [IntentBucket.INFORMATIONAL, IntentBucket.ACCOUNT_LEGAL]
    assert all(certificate.feasible for certificate in conformal.certificates)


def test_coverage_property_on_fresh_samples() -> None:
    """The split-conformal coverage claim, measured rather than asserted.

    Each of the 200 seeded trials draws its own calibration set of 199 nonconformity scores
    and one fresh test point, then asks whether the test point was covered
    (``s_new <= q̂``). The guarantee is ``P(cover) >= 1 − α``; with 200 trials and α = 0.05
    the realised coverage is a binomial draw around 0.95, so the test asserts the *pooled*
    nonconformity rate stays inside α plus binomial slack (and reports the exact numbers in
    the failure message). Asserting "≥ 95% of trials" per-trial would be a coin flip at this
    sample size — see docs/escalation.md §1.3.
    """
    alpha = 0.05
    rng = random.Random(2026)
    trials = 200
    covered = 0
    for _ in range(trials):
        calibration = [rng.random() for _ in range(199)]
        test_score = rng.random()
        quantile = conformal_quantile(calibration, alpha)
        if test_score <= quantile.q_hat:
            covered += 1
    nonconformity = 1.0 - covered / trials
    assert nonconformity <= alpha + 0.02, (
        f"pooled nonconformity {nonconformity:.4f} exceeds alpha={alpha} + 0.02 over "
        f"{trials} trials ({covered} covered)"
    )
    assert covered / trials >= 1.0 - alpha - 0.02


def test_certificate_is_a_contract_not_a_dict() -> None:
    certificate: ConformalCertificate = fit_certificate(
        [0.2] * 30, bucket=IntentBucket.TRANSACTIONAL, alpha=0.05, calibration_hash="e" * 64
    )
    payload = certificate.model_dump(mode="json")
    assert payload["bucket"] == "transactional"
    assert ConformalCertificate.model_validate(payload) == certificate
    with pytest.raises(ValidationError):
        ConformalCertificate.model_validate({**payload, "surprise": 1})
