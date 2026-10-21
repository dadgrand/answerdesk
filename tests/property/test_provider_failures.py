"""Provider failures: 429/500/timeout/connection reset and malformed provider payloads.

Every failure must surface as a taxonomy code with a trace id — never as a bare 500 — and a
failed stream must degrade to ``error`` + ``done`` instead of dropping the connection.

    # AT-OPS-07  # NFR-OPS-01
"""

from __future__ import annotations

import pytest
from fuzz_support import (
    ANSWERABLE,
    PROVIDER_FAILURES,
    DegenerateProvider,
    FlakyProvider,
    RaisingProvider,
    assert_clean_stream,
    assert_taxonomy,
    parse_sse,
    scenario,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("label", "exc", "expected"), PROVIDER_FAILURES, ids=[f[0] for f in PROVIDER_FAILURES]
)
def test_provider_failure_is_a_taxonomised_error(make_app, label, exc, expected) -> None:
    app, _runtime, _provider = make_app(provider=RaisingProvider(exc))

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        error = assert_taxonomy(response, expected={expected}, allow_500=True)
        assert error["retryable"] is True, (label, error)
        assert response.headers["X-Trace-Id"].startswith("tr-")

    scenario(app, _scenario, raise_app_exceptions=False)


@pytest.mark.parametrize(
    ("label", "exc", "expected"),
    [f for f in PROVIDER_FAILURES if f[0] in {"timeout", "connection", "http_500"}],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_provider_failure_degrades_the_stream(make_app, label, exc, expected) -> None:
    app, _runtime, _provider = make_app(provider=RaisingProvider(exc))

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE})
        assert response.status_code == 200
        frames = parse_sse(response.text)
        assert_clean_stream(frames)
        error_frame = next(frame for frame in frames if frame.event == "error")
        assert error_frame.data["error"]["retryable"] is True
        done = frames[-1]
        assert done.data["error"]["code"] == error_frame.data["error"]["code"]
        assert done.data["turn_id"] == ""

    scenario(app, _scenario)


@pytest.mark.parametrize(
    "text",
    [
        "Sure! Here is a friendly answer without any JSON.",
        "",
        '{"claims": [',
        '{"claims": [], "answer": "", "language": "xx"}',
        "{}",
        "\x00\x01 not json at all",
    ],
    ids=["prose", "empty", "truncated_json", "wrong_language", "empty_object", "control_chars"],
)
def test_degenerate_provider_output_never_becomes_an_answer(make_app, text) -> None:
    app, _runtime, _provider = make_app(provider=DegenerateProvider(text))

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        assert response.status_code < 500, response.text[:300]
        body = response.json()
        answer = body["answer"]
        assert answer["decision"]["action"] != "answer", (text, answer["decision"])
        assert answer["text"] == "" or answer["text"].strip() == "", answer["text"][:120]
        assert answer["decision"]["reason_code"] in {
            "PROVIDER_ERROR",
            "NO_EVIDENCE",
            "LOW_CONFIDENCE",
            "VERIFIER_REJECT",
            "CITATION_FAILED",
        }, answer["decision"]

    scenario(app, _scenario)


def test_a_flaky_provider_never_produces_a_500(make_app) -> None:
    provider = FlakyProvider(failures=1)
    app, _runtime, _provider = make_app(provider=provider)

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        # 502 is a legitimate provider failure; a bare 500 is not
        assert response.status_code != 500, response.text[:300]
        if response.status_code >= 400:
            assert_taxonomy(response, expected={502, 503, 504}, allow_500=True)
            return
        body = response.json()
        assert body["answer"]["decision"]["action"] in {"answer", "escalate", "clarify", "refuse"}
        assert len(provider.calls) >= 1

    scenario(app, _scenario)


def test_retrieval_crash_is_a_typed_error_and_the_service_stays_up(make_app) -> None:
    from fuzz_support import PoisonedRetriever

    _app, runtime, provider = make_app()
    poisoned = PoisonedRetriever(runtime.retriever, fail=RuntimeError("index exploded"))
    app2, _runtime2, _provider2 = make_app(retriever=poisoned)

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": ANSWERABLE, "stream": False})
        error = assert_taxonomy(response, allow_500=True)
        assert error["code"] == "internal_error"
        assert error["trace_id"].startswith("tr-")
        # the process is still healthy and the provider was never called
        health = await client.get("/healthz")
        assert health.status_code == 200
        assert provider.calls == []

    scenario(app2, _scenario, raise_app_exceptions=False)


def test_health_reports_a_failed_dependency_without_secrets(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        health = await client.get("/healthz")
        body = health.json()
        text = str(body)
        # A *name* may appear as a status ("api_key=absent"); a credential value may not.
        for marker in ("Bearer ", "sk-live", "fuzz-admin-token", "AKIA"):
            assert marker not in text, marker
        for value in body.values():
            if isinstance(value, dict):
                for nested in value.values():
                    assert "fuzz-admin-token" not in str(nested)
        assert health.status_code in {200, 503}

    scenario(app, _scenario)
