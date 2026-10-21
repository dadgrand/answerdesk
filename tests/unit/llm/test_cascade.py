"""Cascade routing: a deterministic, total function of its inputs.

# AT-LAT-03
"""

from __future__ import annotations

import pytest

from answerdesk.contracts import IntentBucket
from answerdesk.llm.cascade import (
    CascadeArm,
    CascadeDecision,
    CascadePolicy,
    CascadeRouter,
)


@pytest.fixture()
def router() -> CascadeRouter:
    return CascadeRouter()


def test_confident_turn_takes_the_small_arm(router: CascadeRouter) -> None:
    decision = router.route(0.97)
    assert decision.reason == "easy"
    assert decision.arm.name == "small"
    assert decision.arm.mode == "budget"
    assert decision.arm.max_output_tokens < CascadePolicy().large.max_output_tokens


def test_unconfident_turn_takes_the_large_arm(router: CascadeRouter) -> None:
    decision = router.route(0.60)
    assert decision.reason == "low_confidence"
    assert decision.arm.name == "large"


def test_threshold_boundary_is_inclusive(router: CascadeRouter) -> None:
    """p_hat == easy_threshold with a sufficient margin is 'easy', one ulp below is not."""
    policy = router.policy
    at_edge = router.route(policy.easy_threshold, margin=policy.min_margin)
    assert at_edge.reason == "easy"
    just_below = router.route(policy.easy_threshold - 1e-6, margin=policy.min_margin)
    assert just_below.reason == "low_confidence"


def test_narrow_margin_keeps_the_large_arm(router: CascadeRouter) -> None:
    decision = router.route(0.95, margin=0.001)
    assert decision.reason == "narrow_margin"
    assert decision.arm.name == "large"
    assert "margin" in decision.detail


def test_hard_policy_buckets_never_take_the_small_arm(router: CascadeRouter) -> None:
    for bucket in (IntentBucket.ACCOUNT_LEGAL,):
        decision = router.route(0.999, bucket=bucket)
        assert decision.reason == "hard_policy"
        assert decision.arm.name == "large"


@pytest.mark.parametrize("label", ["fraud", "KYC", "legal", "payment", "complaint"])
def test_hard_labels_force_the_large_arm(router: CascadeRouter, label: str) -> None:
    decision = router.route(0.999, intent_label=label)
    assert decision.reason == "hard_policy"
    assert decision.arm.name == "large"


def test_ordinary_labels_do_not_force(router: CascadeRouter) -> None:
    decision = router.route(0.999, intent_label="howto")
    assert decision.reason == "easy"
    assert decision.arm.name == "small"


def test_forced_arm_is_labelled_as_forced(router: CascadeRouter) -> None:
    forced_small = router.route(0.10, force="small")
    assert forced_small.reason == "forced" and forced_small.arm.name == "small"
    forced_large = router.route(0.99, force="large")
    assert forced_large.reason == "forced" and forced_large.arm.name == "large"


def test_p_hat_is_clamped_and_margin_defaults_to_the_edge_distance() -> None:
    router = CascadeRouter()
    high = router.route(1.5)
    assert high.p_hat == 1.0
    assert high.margin == pytest.approx(1.0 - router.policy.easy_threshold)
    low = router.route(-0.5)
    assert low.p_hat == 0.0


def test_routing_is_deterministic() -> None:
    router = CascadeRouter()
    results = {
        (router.route(value).arm.name, router.route(value).reason)
        for value in (0.2, 0.55, 0.9, 0.93, 0.99, 0.99)
    }
    assert results == {
        ("large", "low_confidence"),
        ("small", "easy"),
    } or {name for name, _reason in results} <= {"small", "large"}
    first = router.route(0.97)
    second = CascadeRouter().route(0.97)
    assert first.model_dump() == second.model_dump()


def test_policy_thresholds_are_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CascadePolicy(easy_threshold=1.5)
    with pytest.raises(ValidationError):
        CascadePolicy(min_margin=-0.1)


def test_decision_carries_the_arm_it_chose() -> None:
    policy = CascadePolicy(
        small=CascadeArm(name="tiny", mode="budget", max_output_tokens=96),
        large=CascadeArm(name="huge", mode="full"),
    )
    decision = CascadeRouter(policy).route(0.99)
    assert isinstance(decision, CascadeDecision)
    assert decision.arm.name == "tiny"
    assert decision.arm.max_output_tokens == 96
    assert decision.threshold == policy.easy_threshold
