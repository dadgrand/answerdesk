"""FR-SVC-03: one closed error table, one envelope, one status mapping.

The table-driven part is the point: a new code cannot be emitted without a status, a
retryability flag and an envelope that carries the trace id.

    # AT-SVC-03  # AT-SVC-06
"""

from __future__ import annotations

import httpx
import pytest

from answerdesk.errors import ConfigError
from answerdesk.llm.errors import ProviderHTTPError, ProviderTimeoutError
from answerdesk.service.errors import (
    ERROR_TAXONOMY,
    ServiceError,
    error_envelope,
    to_service_error,
)


def test_every_taxonomy_code_has_a_status_retryability_and_meaning() -> None:
    assert len(ERROR_TAXONOMY) >= 20
    for code, spec in ERROR_TAXONOMY.items():
        assert isinstance(code, str) and code
        assert 400 <= spec.status <= 599, code
        assert isinstance(spec.retryable, bool)
        assert spec.meaning, code


def test_error_envelope_is_complete_for_every_code() -> None:
    for code in ERROR_TAXONOMY:
        error = ServiceError(code, "boom")
        envelope = error.envelope("tr-123")
        assert envelope["code"] == code
        assert envelope["message"] == "boom"
        assert envelope["trace_id"] == "tr-123"
        assert envelope["retryable"] is ERROR_TAXONOMY[code].retryable
        assert set(error_envelope(error, "tr-123")) == {"error"}


def test_unknown_code_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown service error code"):
        ServiceError("made_up_code", "boom")


def test_module_errors_map_onto_the_taxonomy() -> None:
    assert to_service_error(ProviderTimeoutError("t")).code == "provider_timeout"
    assert to_service_error(ProviderTimeoutError("t")).http_status == 504
    http_error = ProviderHTTPError("500", status_code=500)
    mapped = to_service_error(http_error)
    assert mapped.code == "provider_error"
    assert mapped.retryable is True
    assert to_service_error(ConfigError("bad config")).code == "internal_error"
    assert to_service_error(RuntimeError("boom")).code == "internal_error"
    assert to_service_error(TimeoutError()).http_status == 504


def test_validation_error_is_422_with_a_detail(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> httpx.Response:
        return await client.post("/v1/chat", json={"message": "hi", "intent_bucket": "nonsense"})

    response = make_client(scenario)
    assert response.status_code == 422
    payload = response.json()["error"]
    assert payload["code"] == "validation_error"
    assert payload["retryable"] is False
    assert payload["detail"]
    assert payload["trace_id"]


def test_unknown_route_is_404_and_wrong_method_is_405(make_client) -> None:
    async def scenario(client: httpx.AsyncClient, runtime) -> tuple[int, dict, int]:
        missing = await client.get("/v1/does-not-exist")
        wrong = await client.get("/v1/chat")
        return missing.status_code, missing.json()["error"], wrong.status_code

    status, error, wrong_status = make_client(scenario)
    assert status == 404
    assert error["code"] == "not_found"
    assert wrong_status == 405


def test_oversized_body_is_413(core_settings, service_settings) -> None:
    from service_support import lifespan_client, run

    from answerdesk.service.app import create_app
    from answerdesk.service.config import ServiceSettings
    from answerdesk.service.runtime import ServiceRuntime

    settings = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        admin_token=service_settings.admin_token,
        max_body_bytes=1024,
    )
    runtime = ServiceRuntime.create(core_settings, settings, fake=True)
    app = create_app(settings=core_settings, service_settings=settings, runtime=runtime)

    async def scenario() -> httpx.Response:
        async with lifespan_client(app) as client:
            return await client.post("/v1/chat", json={"message": "x" * 5000})

    try:
        response = run(scenario())
    finally:
        runtime.close()
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert response.headers["x-trace-id"]


def test_rate_limited_response_carries_a_retry_hint(core_settings, service_settings) -> None:
    from service_support import lifespan_client, run

    from answerdesk.service.app import create_app
    from answerdesk.service.config import ServiceSettings
    from answerdesk.service.runtime import ServiceRuntime

    settings = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        admin_token=service_settings.admin_token,
        rate_limit_per_minute=2,
        rate_limit_burst=2,
    )
    runtime = ServiceRuntime.create(core_settings, settings, fake=True)
    app = create_app(settings=core_settings, service_settings=settings, runtime=runtime)

    async def scenario() -> list[httpx.Response]:
        responses = []
        async with lifespan_client(app) as client:
            for _ in range(4):
                responses.append(
                    await client.post(
                        "/v1/chat",
                        json={"message": "обслуживание карты", "stream": False},
                        headers={"X-Client-Id": "burst-client"},
                    )
                )
            # a different client is unaffected
            responses.append(
                await client.post(
                    "/v1/chat",
                    json={"message": "обслуживание карты", "stream": False},
                    headers={"X-Client-Id": "other-client"},
                )
            )
            # health is exempt
            responses.append(await client.get("/healthz"))
        return responses

    try:
        responses = run(scenario())
    finally:
        runtime.close()

    limited = [response for response in responses[:4] if response.status_code == 429]
    assert limited, [response.status_code for response in responses[:4]]
    error = limited[0].json()["error"]
    assert error["code"] == "rate_limited"
    assert error["retryable"] is True
    assert error["retry_after_s"] > 0
    assert int(limited[0].headers["retry-after"]) >= 1
    assert responses[4].status_code == 200, "the budget is per client"
    assert responses[5].status_code == 200, "/healthz is never rate limited"
