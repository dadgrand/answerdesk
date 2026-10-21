"""The OpenAPI document is exported, served and complete (FR-SVC-01/03, docs/api.md §7).

# AT-SVC-01  # AT-SVC-03  # AT-DOC-01
"""

from __future__ import annotations

import json
from pathlib import Path

from service_support import lifespan_client, run

from answerdesk.service.app import create_app, export_openapi, openapi_schema
from answerdesk.service.config import ServiceSettings

DOCUMENTED_PATHS = {
    "/v1/chat",
    "/v1/sessions",
    "/v1/sessions/{session_id}",
    "/v1/feedback",
    "/v1/escalations",
    "/v1/escalations/{ticket_id}",
    "/v1/traces/{trace_id}",
    "/v1/info",
    "/admin/ingest",
    "/admin/refresh",
    "/healthz",
    "/metrics",
}


def test_exported_schema_covers_every_documented_endpoint(tmp_path: Path) -> None:
    target = export_openapi(tmp_path / "openapi.json")
    assert target.is_file()
    schema = json.loads(target.read_text(encoding="utf-8"))
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "answerdesk service"
    assert schema["info"]["version"]
    assert set(schema["paths"]) == DOCUMENTED_PATHS, sorted(set(schema["paths"]) ^ DOCUMENTED_PATHS)
    for path, operations in schema["paths"].items():
        assert operations, path
        for method, operation in operations.items():
            assert operation.get("summary"), f"{method} {path} has no summary"
            assert operation.get("tags"), f"{method} {path} has no tag"


def test_chat_documents_the_streaming_response_and_error_envelope(tmp_path: Path) -> None:
    schema = json.loads(export_openapi(tmp_path / "openapi.json").read_text(encoding="utf-8"))
    chat = schema["paths"]["/v1/chat"]["post"]
    assert "200" in chat["responses"]
    assert "text/event-stream" in chat["responses"]["200"]["description"]
    for status in ("400", "401", "403", "404", "409", "413", "422", "429", "502", "503", "504"):
        assert status in chat["responses"], status
        assert chat["responses"][status]["content"]["application/json"]["schema"]
    assert "ErrorResponse" in schema["components"]["schemas"]
    error = schema["components"]["schemas"]["ErrorResponse"]["properties"]["error"]
    assert error

    chat_schema = schema["components"]["schemas"]["ChatRequest"]
    assert chat_schema["properties"]["message"]["maxLength"] == 8000
    assert "session_id" in chat_schema["properties"]


def test_admin_operations_document_the_bearer_header(tmp_path: Path) -> None:
    schema = json.loads(export_openapi(tmp_path / "openapi.json").read_text(encoding="utf-8"))
    for path in ("/admin/ingest", "/admin/refresh"):
        operation = schema["paths"][path]["post"]
        names = {parameter["name"] for parameter in operation.get("parameters", [])}
        assert "Authorization" in names, path


def test_schema_never_contains_a_secret(tmp_path: Path) -> None:
    settings = ServiceSettings(admin_token="super-secret-token-value")
    target = export_openapi(tmp_path / "openapi.json", service_settings=settings)
    text = target.read_text(encoding="utf-8")
    assert "super-secret-token-value" not in text
    assert "test-admin-token" not in text


def test_schema_is_served_when_docs_are_enabled(core_settings, service_settings) -> None:
    from answerdesk.service.runtime import ServiceRuntime

    runtime = ServiceRuntime.create(core_settings, service_settings, fake=True)
    app = create_app(settings=core_settings, service_settings=service_settings, runtime=runtime)

    async def scenario() -> tuple[int, dict, int]:
        async with lifespan_client(app) as client:
            served = await client.get("/openapi.json")
            docs = await client.get("/docs")
            return served.status_code, served.json(), docs.status_code

    try:
        status, schema, docs_status = run(scenario())
    finally:
        runtime.close()
    assert status == 200
    assert "/v1/chat" in schema["paths"]
    assert docs_status == 200


def test_docs_can_be_disabled(core_settings, service_settings) -> None:
    from answerdesk.service.runtime import ServiceRuntime

    service = ServiceSettings(
        db_path=service_settings.db_path,
        trace_dir=service_settings.trace_dir,
        docs_enabled=False,
    )
    runtime = ServiceRuntime.create(core_settings, service, fake=True)
    app = create_app(settings=core_settings, service_settings=service, runtime=runtime)

    async def scenario() -> tuple[int, int]:
        async with lifespan_client(app) as client:
            schema = await client.get("/openapi.json")
            docs = await client.get("/docs")
            return schema.status_code, docs.status_code

    try:
        schema_status, docs_status = run(scenario())
    finally:
        runtime.close()
    assert schema_status == 404
    assert docs_status == 404
    # The library schema is still exportable without the HTTP surface.
    assert openapi_schema(settings=core_settings, service_settings=service)["paths"]
