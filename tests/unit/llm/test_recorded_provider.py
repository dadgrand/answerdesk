"""Recorded-fixture provider: capture once, replay forever, never touch the network.

# AT-GEN-09  # AT-PORT-01
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from answerdesk.contracts import ChatProvider, Message, Role
from answerdesk.llm.recorded import (
    RECORDED_MODEL,
    RecordedExchange,
    RecordedFixture,
    RecordedFixtureMiss,
    RecordedProvider,
    RecordingProvider,
    default_recorded_config,
    request_digest,
    schema_digest,
)
from answerdesk.testing.doubles import FakeProvider, default_fake_provider_config

MESSAGES = [
    Message(role=Role.SYSTEM, content="You are answerdesk."),
    Message(role=Role.USER, content="Refunds take how long?"),
]
SCHEMA = {"type": "object", "additionalProperties": False, "required": [], "properties": {}}


def test_recording_then_replaying_reproduces_the_completion(tmp_path: Path) -> None:
    """# AT-GEN-09 — the double is deterministic because the transcript is frozen."""
    recorder = RecordingProvider(FakeProvider(responses={"Refunds take": '{"claims": []}'}))
    live = recorder.complete(MESSAGES, response_schema=SCHEMA)
    path = recorder.save(tmp_path / "recorded.json")

    replayed = RecordedProvider.from_file(path)
    replay = replayed.complete(MESSAGES, response_schema=SCHEMA)
    assert replay.text == live.text
    assert replay.model == live.model
    assert replay.input_tokens == live.input_tokens
    assert replay.output_tokens == live.output_tokens
    assert replay.finish_reason == "stop"


def test_missing_exchange_is_a_loud_fixture_miss(tmp_path: Path) -> None:
    """# AT-GEN-09 — a stale fixture must fail, never replay a neighbouring answer."""
    recorder = RecordingProvider(FakeProvider())
    recorder.complete(MESSAGES)
    path = recorder.save(tmp_path / "recorded.json")
    replayed = RecordedProvider.from_file(path)
    replayed.complete(MESSAGES)  # consumes the positional fallback

    other = [Message(role=Role.USER, content="a question nobody recorded")]
    with pytest.raises(RecordedFixtureMiss) as excinfo:
        replayed.complete(other)
    assert "chars=" in excinfo.value.preview
    assert "nobody recorded" not in excinfo.value.preview, "the preview must not leak prompt text"


def test_positional_fallback_is_opt_in_for_hand_written_fixtures() -> None:
    """# AT-GEN-09 — a digest miss is a stale fixture; positional replay must be explicit."""
    fixture = RecordedFixture(
        model="hand-written",
        exchanges=[
            RecordedExchange(
                request_digest="0" * 64,
                request_preview="hand written",
                response={
                    "text": "canned",
                    "model": "hand-written",
                    "input_tokens": 1,
                    "output_tokens": 2,
                },
            )
        ],
    )
    strict = RecordedProvider(fixture)
    with pytest.raises(RecordedFixtureMiss):
        strict.complete(MESSAGES)

    fallback = RecordedProvider(fixture, allow_positional_fallback=True)
    completion = fallback.complete(MESSAGES)
    assert completion.text == "canned"
    with pytest.raises(RecordedFixtureMiss):
        fallback.complete(MESSAGES)


def test_digest_covers_messages_and_schema() -> None:
    """A prompt or schema change is a different request: the fixture cannot silently match."""
    base = request_digest(MESSAGES)
    assert base == request_digest(list(MESSAGES))
    changed = [MESSAGES[0], Message(role=Role.USER, content="different question")]
    assert request_digest(changed) != base
    assert request_digest(MESSAGES, SCHEMA) != base
    assert schema_digest(SCHEMA) == schema_digest(dict(SCHEMA))
    assert schema_digest(None) == ""


def test_streaming_a_recorded_answer_is_chunked_and_terminated() -> None:
    # AT-GEN-09 — the streaming path replays the same text, in words, then done.
    recorder = RecordingProvider(
        FakeProvider(responses={"Refunds take": "one two three four five"})
    )
    recorder.complete(MESSAGES)
    provider = RecordedProvider(recorder.to_fixture())
    stream = provider.stream(MESSAGES)
    chunks = stream.materialise()
    assert "".join(chunk.delta for chunk in chunks).split() == [
        "one",
        "two",
        "three",
        "four",
        "five",
    ]
    assert chunks[-1].done is True


def test_recorded_provider_is_a_chat_provider() -> None:
    # AT-PORT-01 — the double satisfies the same seam as the HTTP transport.
    provider = RecordedProvider(RecordedFixture(exchanges=[]))
    assert isinstance(provider, ChatProvider)
    assert provider.config.base_url == "memory://recorded"
    assert provider.config.requires_api_key is False


def test_fixture_json_is_stable_and_human_readable(tmp_path: Path) -> None:
    recorder = RecordingProvider(FakeProvider(responses={"Refunds take": "answer"}))
    recorder.complete(MESSAGES, response_schema=SCHEMA)
    first = recorder.save(tmp_path / "a.json").read_text(encoding="utf-8")
    second = recorder.save(tmp_path / "b.json").read_text(encoding="utf-8")
    assert first == second
    payload = json.loads(first)
    assert payload["version"] == "1.0.0"
    assert payload["exchanges"][0]["response"]["text"] == "answer"


def test_default_config_uses_the_recorded_model() -> None:
    config = default_recorded_config()
    assert config.model == RECORDED_MODEL
    assert config.preset == "recorded"
    assert default_fake_provider_config().preset == "fake"
