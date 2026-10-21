"""Fixtures for the provider-layer tests (no sockets, no keys, no sleeping).

# AT-PORT-01  # AT-COST-01
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import pytest
from mock_openai import MockOpenAI

from answerdesk.contracts import ProviderConfig
from answerdesk.llm.circuit import CircuitBreaker
from answerdesk.llm.cost import CostLedger, PricingTable
from answerdesk.llm.presets import provider_config
from answerdesk.llm.providers import HttpChatProvider
from answerdesk.llm.retry import RetryPolicy, Runner


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient proxy variables must not leak into a hermetic test (INV-08)."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        if os.environ.get(name):
            monkeypatch.delenv(name, raising=False)


class FakeClock:
    """Monotonic clock under test control, so latency and breaker maths are exact."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SleepRecorder:
    """Records requested sleeps and advances the fake clock instead of waiting."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.clock.advance(seconds)

    @property
    def total(self) -> float:
        return sum(self.sleeps)


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def sleeper(clock: FakeClock) -> SleepRecorder:
    return SleepRecorder(clock)


@pytest.fixture()
def runner(clock: FakeClock, sleeper: SleepRecorder) -> Runner:
    """Deterministic side effects: fixed clock, recorded sleeps, seeded jitter."""
    return Runner(sleep=sleeper, rng=random.Random(1234), clock=clock)


@pytest.fixture()
def mock_server() -> MockOpenAI:
    return MockOpenAI()


@pytest.fixture()
def pricing() -> PricingTable:
    return PricingTable.default()


@pytest.fixture()
def ledger(pricing: PricingTable) -> CostLedger:
    return CostLedger(session_id="unit-session", pricing=pricing)


@pytest.fixture()
def local_config() -> ProviderConfig:
    """A local-server config: localhost base_url, no key (NFR-PORT-01)."""
    return provider_config("vllm", timeout_s=2.0, max_retries=2)


@pytest.fixture()
def remote_config(monkeypatch: pytest.MonkeyPatch) -> ProviderConfig:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abcdef0123456789")
    return provider_config("openai", timeout_s=2.0, max_retries=2)


@pytest.fixture()
def http_provider_factory(
    mock_server: MockOpenAI,
    runner: Runner,
    ledger: CostLedger,
) -> Callable[..., HttpChatProvider]:
    """Build an :class:`HttpChatProvider` pointed at the mock server."""

    def _make(
        config: ProviderConfig | None = None,
        *,
        breaker: CircuitBreaker | None = None,
        policy: RetryPolicy | None = None,
        with_ledger: bool = True,
    ) -> HttpChatProvider:
        resolved = config or provider_config("vllm", timeout_s=2.0, max_retries=2)
        return HttpChatProvider(
            resolved,
            ledger=ledger if with_ledger else None,
            circuit=breaker,
            retry_policy=policy or RetryPolicy(max_retries=resolved.max_retries),
            runner=runner,
            transport=mock_server.transport(),
        )

    return _make
