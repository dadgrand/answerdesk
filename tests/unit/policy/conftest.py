"""Fixtures for the policy tests: a small sealed artifact, an engine, and text helpers.

The artifact is built once per session from a reduced synthetic population (still large
enough for the α=0.01 bucket's conformal rank), so decision tests exercise the real
thresholds rather than hand-written ones.

    # AT-ESC-02  # AT-ESC-06
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from answerdesk.config import load_settings
from answerdesk.contracts import Action, IntentBucket, ReasonCode, RetrievedPassage
from answerdesk.generate.pipeline import GenerationSignal
from answerdesk.policy.artifact import CalibrationArtifact
from answerdesk.policy.calibrate import build_artifact
from answerdesk.policy.costs import CostProvenance, CostSet, load_cost_set
from answerdesk.policy.decision import DecisionEngine
from answerdesk.policy.guards import GuardConfig, load_guard_config
from answerdesk.policy.synthetic import SyntheticConfig, generate_population

FIXED_NOW = datetime(2026, 10, 21, 0, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def alphas() -> dict[IntentBucket, float]:
    settings = load_settings(profile="offline")
    return {bucket: float(settings.policy.alpha[bucket]) for bucket in IntentBucket}


@pytest.fixture(scope="session")
def cost_set() -> CostSet:
    return load_cost_set()


@pytest.fixture(scope="session")
def guard_config() -> GuardConfig:
    return load_guard_config()


@pytest.fixture(scope="session")
def population_config() -> SyntheticConfig:
    return SyntheticConfig(n_items=2400, seed=20261020)


@pytest.fixture(scope="session")
def artifact(
    population_config: SyntheticConfig,
    cost_set: CostSet,
    alphas: dict[IntentBucket, float],
    guard_config: GuardConfig,
) -> CalibrationArtifact:
    population = generate_population(population_config)
    artifact, _metrics = build_artifact(
        population=population,
        cost_set=cost_set,
        alphas=alphas,
        guards=guard_config,
        now=FIXED_NOW,
    )
    return artifact


@pytest.fixture(scope="session")
def engine(artifact: CalibrationArtifact, guard_config: GuardConfig) -> DecisionEngine:
    return DecisionEngine.from_artifact(artifact, guards=guard_config, now=FIXED_NOW)


@pytest.fixture()
def make_cost_set() -> Callable[..., CostSet]:
    def _make(c_wrong: float = 1.0, c_abstain: float = 0.10, c_delay: float = 0.05) -> CostSet:
        return CostSet(
            c_wrong=c_wrong,
            c_abstain=c_abstain,
            c_delay=c_delay,
            provenance=CostProvenance(
                source="unit test", owner="t8", date="2026-10-21", note="fixture"
            ),
        )

    return _make


@pytest.fixture()
def signal_answer() -> GenerationSignal:
    return GenerationSignal(
        action=Action.ANSWER, reason_code=ReasonCode.ANSWERED, detail="verified"
    )


@pytest.fixture()
def signal_no_evidence() -> GenerationSignal:
    return GenerationSignal(
        action=Action.ESCALATE, reason_code=ReasonCode.NO_EVIDENCE, detail="empty pack"
    )


@pytest.fixture()
def card_passages() -> list[RetrievedPassage]:
    from answerdesk.contracts import Channel, Chunk

    text = (
        "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. "
        "Плата списывается первого числа каждого месяца."
    )
    chunk = Chunk(
        chunk_id="doc-fee::0",
        doc_id="doc-fee",
        text=text,
        heading_path=["Тарифы", "Обслуживание"],
        breadcrumb="Тарифы > Обслуживание",
        start_char=0,
        end_char=len(text),
        token_count=24,
        language="ru",
        metadata={"doc_title": "Тарифы по карте «Вектор»"},
    )
    return [RetrievedPassage(chunk=chunk, channel=Channel.DENSE, score=0.94, rank=1)]
