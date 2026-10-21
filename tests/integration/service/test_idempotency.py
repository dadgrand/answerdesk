"""FR-SVC-02: an ``Idempotency-Key`` replay costs no model call and returns the same body.

    # AT-SVC-02
"""

from __future__ import annotations

import httpx

from support import ANSWERABLE_QUESTION, parse_sse

CHAT = "/v1/chat"


def test_same_key_replays_the_stored_response(make_client, fake_provider) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        headers = {"Idempotency-Key": "idem-0001"}
        body = {"message": ANSWERABLE_QUESTION, "stream": False}
        first = await client.post(CHAT, json=body, headers=headers)
        calls_after_first = len(fake_provider.calls)
        second = await client.post(CHAT, json=body, headers=headers)
        return {
            "first": first,
            "second": second,
            "calls_after_first": calls_after_first,
            "calls_after_second": len(fake_provider.calls),
        }

    outcome = make_client(scenario)
    first, second = outcome["first"], outcome["second"]
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json() == first.json()
    assert second.headers["x-idempotent-replay"] == "true"
    assert second.headers["x-trace-id"] == first.headers["x-trace-id"]
    assert outcome["calls_after_first"] == 1
    assert outcome["calls_after_second"] == 1, "a replay must not call the provider again"


def test_same_key_with_a_different_body_is_a_conflict(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        headers = {"Idempotency-Key": "idem-0002"}
        first = await client.post(CHAT, json={"message": "обслуживание карты"}, headers=headers)
        second = await client.post(CHAT, json={"message": "лимит перевода"}, headers=headers)
        return {"first": first.status_code, "second": second}

    outcome = make_client(scenario)
    second = outcome["second"]
    assert outcome["first"] == 200
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "idempotency_conflict"
    assert error["retryable"] is False


def test_streamed_then_replayed_request_returns_json(make_client, fake_provider) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> dict:
        headers = {"Idempotency-Key": "idem-0003"}
        body = {"message": ANSWERABLE_QUESTION}
        streamed = await client.post(CHAT, json=body, headers=headers)
        calls = len(fake_provider.calls)
        replay = await client.post(CHAT, json=body, headers=headers)
        return {"streamed": streamed, "replay": replay, "calls": calls,
                "calls_after": len(fake_provider.calls)}

    outcome = make_client(scenario)
    assert outcome["streamed"].headers["content-type"].startswith("text/event-stream")
    assert parse_sse(outcome["streamed"].text)[-1].event == "done"
    assert outcome["replay"].headers["content-type"].startswith("application/json")
    assert outcome["replay"].headers["x-idempotent-replay"] == "true"
    assert outcome["calls"] == 1
    assert outcome["calls_after"] == 1


def test_different_keys_are_different_turns(core_settings, service_settings, fake_provider) -> None:
    """Two keys must produce two turns; the cache is disabled so the model call is visible."""
    from support import run, lifespan_client

    from answerdesk.service.app import create_app
    from answerdesk.service.runtime import ServiceRuntime

    settings = core_settings.model_copy(deep=True)
    settings.cache.enabled = False
    runtime = ServiceRuntime.create(settings, service_settings, fake=True, provider=fake_provider)
    app = create_app(settings=settings, service_settings=service_settings, runtime=runtime)

    async def scenario() -> dict:
        async with lifespan_client(app) as client:
            body = {"message": ANSWERABLE_QUESTION, "stream": False}
            first = await client.post(CHAT, json=body, headers={"Idempotency-Key": "idem-a"})
            second = await client.post(CHAT, json=body, headers={"Idempotency-Key": "idem-b"})
            return {
                "first": first.json(),
                "second": second.json(),
                "calls": len(fake_provider.calls),
            }

    try:
        outcome = run(scenario())
    finally:
        runtime.close()
    assert outcome["first"]["trace_id"] != outcome["second"]["trace_id"]
    assert outcome["calls"] == 2
