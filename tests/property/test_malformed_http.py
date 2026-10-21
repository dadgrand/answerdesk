"""Malformed HTTP requests: bodies, methods, routes, content types and headers.

The service may reject anything here, but it must reject it *in the taxonomy*: a code, a
message, a trace id, a retryable flag — and never a bare 500 with a stack trace.

    # AT-OPS-07  # AT-SVC-03
"""

from __future__ import annotations

import pytest
from fuzz_support import MALFORMED_BODIES, assert_taxonomy, parse_sse, scenario

pytestmark = pytest.mark.integration

TRACE_ID_PATTERN = r"^tr-[0-9a-f]{24}$"


def test_every_malformed_body_is_a_traced_4xx(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        for label, body in MALFORMED_BODIES:
            response = await client.post(
                "/v1/chat", content=body, headers={"Content-Type": "application/json"}
            )
            error = assert_taxonomy(response, allow_500=False)
            if response.status_code >= 400:
                assert error["code"] in {
                    "validation_error",
                    "bad_request",
                    "payload_too_large",
                    "unsupported_media_type",
                }, (label, error)

    scenario(app, _scenario)


def test_unknown_fields_and_bad_enums_are_rejected(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        cases = [
            {"message": "hi", "explain": True},
            {"message": "hi", "intent_bucket": "nonsense"},
            {"message": "hi", "intent_bucket": "transactional", "intent_label": "x" * 65},
            {"message": "hi", "stream": "maybe"},
        ]
        for payload in cases:
            response = await client.post("/v1/chat", json=payload)
            error = assert_taxonomy(response, expected={422})
            assert error["code"] == "validation_error"

    scenario(app, _scenario)


def test_content_types_are_either_accepted_or_rejected_with_a_code(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        body = b'{"message": "hi", "stream": false}'
        cases = {
            "text/plain": {"Content-Type": "text/plain"},
            "form": {"Content-Type": "application/x-www-form-urlencoded"},
            "missing": {},
            "json_charset": {"Content-Type": "application/json; charset=utf-8"},
        }
        for label, headers in cases.items():
            response = await client.post("/v1/chat", content=body, headers=headers)
            if label == "json_charset":
                assert response.status_code == 200, response.text[:200]
                continue
            assert_taxonomy(response)
            assert response.status_code < 500, (label, response.status_code)

    scenario(app, _scenario)


def test_wrong_methods_and_routes_are_404_or_405(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        for method in ("get", "put", "delete", "patch"):
            response = await getattr(client, method)("/v1/chat")
            error = assert_taxonomy(response, expected={405})
            assert error["code"] == "method_not_allowed"
        for path in ("/v1/nope", "/v1/traces/../../etc/passwd", "/v1/traces/" + "a" * 500):
            response = await client.get(path)
            error = assert_taxonomy(response, expected={404})
            assert error["code"] == "not_found"
        response = await client.get("/v1/traces/tr-ffffffffffffffffffffffff")
        assert_taxonomy(response, expected={404})

    scenario(app, _scenario)


def test_client_trace_ids_are_echoed_only_when_safe(make_app) -> None:
    """A header value flows into a response header and a JSONL line: it must be a token."""
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        import re

        good = "tr-0123456789abcdef01234567"
        response = await client.post(
            "/v1/chat", json={"message": "hi"}, headers={"X-Trace-Id": good}
        )
        assert response.status_code == 200
        assert response.headers["X-Trace-Id"] == good

        for hostile in ("bad id with spaces", "x" * 5000, "tr-\x01\x02", ""):
            response = await client.post(
                "/v1/chat", json={"message": "hi"}, headers={"X-Trace-Id": hostile}
            )
            assert response.status_code == 200
            echoed = response.headers["X-Trace-Id"]
            assert re.match(TRACE_ID_PATTERN, echoed), (hostile, echoed)

    scenario(app, _scenario)


def test_duplicate_and_unknown_headers_do_not_break_the_request(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat",
            json={"message": "hi", "stream": False},
            headers=[
                ("X-Trace-Id", "tr-aaaaaaaaaaaaaaaaaaaaaaaa"),
                ("X-Trace-Id", "tr-bbbbbbbbbbbbbbbbbbbbbbbb"),
                ("X-Unknown", "value"),
            ],
        )
        assert response.status_code == 200
        assert response.headers["X-Trace-Id"].startswith("tr-")

    scenario(app, _scenario)


def test_query_string_garbage_is_ignored(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post(
            "/v1/chat?limit=abc&unknown=%00", json={"message": "hi", "stream": False}
        )
        assert response.status_code == 200
        response = await client.get("/v1/traces/tr-0123456789abcdef01234567?kind=%00")
        assert_taxonomy(response, expected={404})

    scenario(app, _scenario)


def test_sse_response_is_well_formed_even_for_a_short_turn(make_app) -> None:
    app, _runtime, _provider = make_app()

    async def _scenario(client, _runtime):
        response = await client.post("/v1/chat", json={"message": "hi"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = parse_sse(response.text)
        assert frames[0].event == "retrieval"
        assert frames[-1].event == "done"

    scenario(app, _scenario)
