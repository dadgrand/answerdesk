"""Determinism: the same transcript produces a byte-identical answer object (FR-GEN-09).

Two doubles are under test here — ``FakeProvider`` (plumbing) and ``RecordedProvider``
(the real answer *shape*, captured once and replayed). Neither touches the network, which
is what makes the claim executable in CI.

    # AT-GEN-09
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import ScriptedProvider

from answerdesk.config import Settings
from answerdesk.contracts import Message, RetrievedPassage, Role
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.generate.prompts import PromptRegistry, content_digest
from answerdesk.llm.recorded import RecordedFixtureMiss, RecordedProvider, RecordingProvider
from answerdesk.testing.doubles import FakeProvider

QUESTION = "Сколько стоит обслуживание карты?"


def test_recorded_fixture_replays_a_byte_identical_answer(
    tmp_path: Path,
    settings: Settings,
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-09 — record once, replay twice, compare hashes."""
    recorder = RecordingProvider(ScriptedProvider([answer_draft]))
    recorded_run = AnswerPipeline.from_settings(recorder, settings).answer(QUESTION, passages)
    fixture = recorder.save(tmp_path / "answer.json")

    first = AnswerPipeline.from_settings(RecordedProvider.from_file(fixture), settings).answer(
        QUESTION, passages
    )
    second = AnswerPipeline.from_settings(RecordedProvider.from_file(fixture), settings).answer(
        QUESTION, passages
    )

    assert first.to_json() == second.to_json()
    assert first.claims == recorded_run.claims
    assert [c.quote for c in first.citations] == [c.quote for c in recorded_run.citations]
    assert first.signal == recorded_run.signal

    payload = json.loads(first.to_json())
    assert payload["prompt_version"] == "grounded-answer-1.0.0"
    assert payload["verifier_version"] == "lexical-verifier-1.0.0"


def test_fake_provider_is_deterministic_across_runs(
    settings: Settings,
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-09 — the CI double itself is a pure function of the prompt."""

    def run() -> str:
        provider = FakeProvider(responses={"Сколько стоит": answer_draft})
        return AnswerPipeline.from_settings(provider, settings).answer(QUESTION, passages).to_json()

    assert run() == run()


def test_recorded_fixture_does_not_embed_the_customer_question(
    tmp_path: Path,
) -> None:
    """A committed fixture must not leak prompt text: it stores digests, not prompts."""
    recorder = RecordingProvider(ScriptedProvider([json_draft()]))
    recorder.complete(
        [
            Message(role=Role.SYSTEM, content="system"),
            Message(role=Role.USER, content="Сколько стоит обслуживание карты?"),
        ],
        response_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )
    path = recorder.save(tmp_path / "answer.json")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    exchange = payload["exchanges"][0]
    assert len(exchange["request_digest"]) == 64
    assert exchange["schema_digest"]
    assert payload["model"]
    assert "Сколько стоит" not in text, "the committed fixture must not carry prompt text"
    assert len(exchange["request_preview"]) <= 160


def test_replay_reports_a_stale_fixture(
    tmp_path: Path,
    settings: Settings,
    passages: list[RetrievedPassage],
    answer_draft: str,
) -> None:
    """# AT-GEN-09 — a prompt change surfaces as a fixture miss, not as a wrong answer."""
    recorder = RecordingProvider(ScriptedProvider([answer_draft]))
    AnswerPipeline.from_settings(recorder, settings).answer(QUESTION, passages)
    fixture = recorder.save(tmp_path / "answer.json")

    registry = PromptRegistry.load()
    prompt = registry.get("grounded-answer")
    changed_system = prompt.system + "\nAlways answer in uppercase."
    changed = prompt.model_copy(
        update={
            "system": changed_system,
            "digest": content_digest(changed_system, prompt.user),
        }
    )
    tampered = PromptRegistry.from_prompts([changed, registry.get("repair-answer")])

    pipeline = AnswerPipeline.from_settings(RecordedProvider.from_file(fixture), settings)
    pipeline.registry = tampered
    with pytest.raises(RecordedFixtureMiss):
        pipeline.answer(QUESTION, passages)


def json_draft() -> str:
    from support import draft_json

    return draft_json([("x y z", "p1", "x y z")])
