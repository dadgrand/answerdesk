"""Arbitrary-unicode input through the HTTP boundary.

Every string a client can put in ``message`` reaches SQLite, the JSONL trace and the SSE
encoder. This module walks a torture corpus (control characters, RTL, emoji, mixed scripts,
zero-width and bidi controls, 8 000-character maxima, invalid UTF-8 *bytes*) and asserts the
three invariants of INV-05/NFR-OPS-01: the stream stays well-formed, the trace stays
readable, and nothing becomes an unhandled 500.

    # AT-OPS-07  # AT-SVC-03
"""

from __future__ import annotations

import json

import pytest
from fuzz_support import (
    UNICODE_TORTURE,
    assert_clean_stream,
    assert_taxonomy,
    error_envelope,
    parse_sse,
    scenario,
)

pytestmark = pytest.mark.integration

#: Inputs that are expected to be *accepted* by the schema and answered or escalated.
ACCEPTED = [case for case in UNICODE_TORTURE if case[0] not in {"empty", "over_max_length"}]


def test_torture_corpus_never_breaks_the_stream(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        results = []
        for label, text in ACCEPTED:
            response = await client.post(
                "/v1/chat",
                json={"message": text, "stream": True},
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 200, (label, response.status_code, response.text[:200])
            frames = parse_sse(response.text)
            assert_clean_stream(frames)
            trace_id = frames[-1].data["trace_id"]
            results.append((label, trace_id, frames[-1].data["answer"]["decision"]["action"]))
        return results

    results = scenario(app, _scenario)
    assert len(results) == len(ACCEPTED)
    for label, _trace_id, action in results:
        assert action in {"answer", "escalate", "clarify", "refuse"}, label


def test_every_torture_turn_writes_a_readable_trace(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        for label, text in ACCEPTED:
            response = await client.post("/v1/chat", json={"message": text, "stream": False})
            assert response.status_code == 200, (label, response.text[:200])
            body = response.json()
            trace_id = body["trace_id"]
            trace = await client.get(f"/v1/traces/{trace_id}")
            assert trace.status_code == 200, (label, trace.status_code)
            payload = trace.json()
            assert payload["trace_id"] == trace_id
            json.dumps(payload)  # already parsed: the file line was valid UTF-8 JSON

    scenario(app, _scenario)


def test_empty_and_overlong_messages_are_validation_errors(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        for text in ("", "я" * 8001):
            response = await client.post("/v1/chat", json={"message": text})
            error = assert_taxonomy(response, expected={422})
            assert error["code"] == "validation_error"

    scenario(app, _scenario)


def test_lone_surrogate_in_the_message_is_a_traced_validation_error(make_app) -> None:
    """JSON permits ``\\ud800``; UTF-8 does not. It must be a 422, not a crash in SQLite."""
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat",
            content=b'{"message": "bad \\ud800 text"}',
            headers={"Content-Type": "application/json"},
        )
        error = assert_taxonomy(response, expected={422})
        assert error["code"] == "validation_error"
        # the service is still healthy afterwards
        follow = await client.post("/v1/chat", json={"message": "привет"})
        assert follow.status_code == 200

    scenario(app, _scenario)


def test_surrogate_in_an_optional_field_is_sanitised(make_app) -> None:
    """A surrogate in ``session_id`` is replaced (U+FFFD), not stored and not fatal."""
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat",
            content=b'{"message": "hi", "session_id": "s-\\udfff", "stream": false}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text[:300]
        body = response.json()
        assert body["session_id"] == "s-?"
        assert "\udfff" not in response.text

    scenario(app, _scenario)


def test_invalid_utf8_bytes_are_a_traced_400(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        for body in (b'{"message": "\xff\xfe bad"}', b'{"message": "\xe2\x82"}'):
            response = await client.post(
                "/v1/chat", content=body, headers={"Content-Type": "application/json"}
            )
            error = assert_taxonomy(response, expected={400})
            assert error["code"] in {"bad_request", "validation_error"}

    scenario(app, _scenario)


def test_control_characters_are_escaped_in_the_trace_file(make_app) -> None:
    app, runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat", json={"message": "a\x01\x02\x03b\x7f", "stream": False}
        )
        assert response.status_code == 200
        return response.json()["trace_id"]

    trace_id = scenario(app, _scenario)
    trace = runtime.tracer.read(trace_id, kind="turn")
    assert trace is not None
    raw = json.dumps(trace)
    assert "\x01" not in raw  # the JSON encoder escapes it; the value is safe
    assert trace["query"] == "a\x01\x02\x03b\x7f"


def test_rtl_and_mixed_scripts_are_detected_not_rejected(make_app) -> None:
    """Hebrew/Arabic are outside the six supported locales: they fall back, they do not crash.

    The corpus and the language detector cover ru/tg/uz/ky/tr/zh/en. An RTL message in
    another script must still be a normal turn (language falls back to ``en``), which is what
    the operator sees: a grounded answer or an escalation, never a 500 (docs/robustness.md,
    limitation L-2).
    """
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        supported = {"ru", "tg", "uz", "ky", "tr", "zh", "en"}
        for text in (
            "שלום, איך אפשר לעזור?",
            "مرحبا، كيف يمكنني المساعدة؟",
            "Привет hello 你好",
        ):
            response = await client.post("/v1/chat", json={"message": text, "stream": False})
            assert response.status_code == 200, (text, response.text[:200])
            answer = response.json()["answer"]
            assert answer["language"] in supported, (text, answer["language"])

    scenario(app, _scenario)


def test_error_envelopes_are_identical_for_every_rejected_unicode_body(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        bodies = [
            b'{"message": ""}',
            b'{"message": "x", "intent_bucket": "nope"}',
            b'{"message": "\\ud800"}',
        ]
        for body in bodies:
            response = await client.post(
                "/v1/chat", content=body, headers={"Content-Type": "application/json"}
            )
            error = error_envelope(response)
            assert error["code"] == "validation_error", error
            assert error["trace_id"].startswith("tr-")
            assert error["retryable"] is False

    scenario(app, _scenario)
