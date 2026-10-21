"""Retry policy and circuit breaker: the maths, without sleeping or connecting.

# AT-PORT-01
"""

from __future__ import annotations

import random

import pytest

from answerdesk.llm.circuit import CircuitBreaker, CircuitState
from answerdesk.llm.retry import (
    FATAL_STATUS,
    RETRYABLE_STATUS,
    RetryPolicy,
    Runner,
    parse_retry_after,
    retry_after_or_default,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------------------
# RetryPolicy
# --------------------------------------------------------------------------------------


def test_policy_rejects_impossible_parameters() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_s=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


def test_nominal_backoff_doubles_and_caps() -> None:
    """# AT-PORT-01 — delay = min(max, base * 2**attempt)."""
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0, jitter=0.0)
    assert [policy.nominal_delay(attempt) for attempt in range(5)] == [
        0.5,
        1.0,
        2.0,
        4.0,
        4.0,
    ]
    assert policy.delay_for(3) == 4.0  # jitter=0 means the nominal value


def test_jitter_is_symmetric_and_seeded() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=100.0, jitter=0.5)
    draws = [policy.delay_for(2, rng=random.Random(99)) for _ in range(20)]
    assert all(2.0 <= draw <= 6.0 for draw in draws)  # nominal 4.0, +/-50%
    again = [policy.delay_for(2, rng=random.Random(99)) for _ in range(20)]
    assert again == draws


def test_retryable_status_sets_are_disjoint_and_sensible() -> None:
    assert 503 in RETRYABLE_STATUS
    assert 429 in RETRYABLE_STATUS
    assert 401 in FATAL_STATUS
    assert 400 in FATAL_STATUS
    assert not (RETRYABLE_STATUS & FATAL_STATUS)


@pytest.mark.parametrize(
    ("attempt", "status", "expected"),
    [
        (0, 503, True),
        (2, 503, True),
        (3, 503, False),  # budget exhausted (max_retries=3)
        (0, 401, False),
        (0, 422, False),
        (0, None, True),  # a timeout/transport failure retries
    ],
)
def test_should_retry_matrix(attempt: int, status: int | None, expected: bool) -> None:
    policy = RetryPolicy(max_retries=3)
    assert policy.should_retry(attempt=attempt, status_code=status, timed_out=status is None) is (
        expected
    )


def test_parse_retry_after_accepts_only_delta_seconds() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(" 2.5 ") == 2.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
    assert parse_retry_after("-1") is None
    assert parse_retry_after("9999") == 30.0  # capped: CI has a 10-minute gate
    assert retry_after_or_default(None) == 1.0


def test_runner_skips_zero_sleeps() -> None:
    sleeps: list[float] = []
    runner = Runner(sleep=sleeps.append, rng=random.Random(0), clock=lambda: 0.0)
    runner.wait(0.0)
    runner.wait(0.25)
    assert sleeps == [0.25]


# --------------------------------------------------------------------------------------
# CircuitBreaker
# --------------------------------------------------------------------------------------


def test_breaker_opens_then_recovers() -> None:
    """# AT-PORT-01 — closed → open → half_open → closed, driven by an injected clock."""
    clock = Clock()
    breaker = CircuitBreaker(name="p", failure_threshold=2, recovery_timeout_s=10.0, clock=clock)
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow() is True

    breaker.record_success()  # a success resets the failure count
    assert breaker.failures == 0

    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False
    assert breaker.rejected == 1

    clock.advance(9.0)
    assert breaker.allow() is False
    clock.advance(1.0)
    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is False, "only one probe is admitted by default"

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 0


def test_breaker_half_open_probe_failure_reopens() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=1.0, clock=clock)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.advance(1.0)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.opened_at == 1.0


def test_breaker_snapshot_is_plain_data() -> None:
    breaker = CircuitBreaker(name="http:vllm", failure_threshold=1)
    breaker.record_failure()
    snapshot = breaker.snapshot()
    assert snapshot["name"] == "http:vllm"
    assert snapshot["state"] == "open"
    assert snapshot["failures"] == 1
    assert isinstance(snapshot["rejected"], int)


def test_breaker_rejects_impossible_thresholds() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout_s=-1.0)
