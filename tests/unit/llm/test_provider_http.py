"""Provider transport tests against the mock OpenAI-compatible server.

Every transport feature the contract names is exercised here: one HTTP client for six
presets, streaming SSE, structured output, timeouts, retries with jitter, the fallback
chain and the circuit breaker (NFR-PORT-01, FR-OPS-03).

    # AT-PORT-01  # AT-OPS-03  # AT-COST-01  # AT-ML-02
"""

from __future__ import annotations

import ast
import random
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from mock_openai import MockOpenAI, Reply

from answerdesk.contracts import Message, ProviderConfig, Role
from answerdesk.llm.circuit import CircuitBreaker, CircuitState
from answerdesk.llm.cost import CostLedger
from answerdesk.llm.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from answerdesk.llm.presets import provider_config
from answerdesk.llm.providers import HttpChatProvider, ProviderChain
from answerdesk.llm.retry import RetryPolicy

MESSAGES = [
    Message(role=Role.SYSTEM, content="You are answerdesk."),
    Message(role=Role.USER, content="Сколько стоит обслуживание карты?"),
]

PROVIDERS_MODULE = (
    Path(__file__).resolve().parents[3] / "src" / "answerdesk" / "llm" / "providers.py"
)


# --------------------------------------------------------------------------------------
# Request construction: one client, six presets, no branches
# --------------------------------------------------------------------------------------


def test_request_uses_the_chat_completions_endpoint_and_bearer_key(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    remote_config: ProviderConfig,
) -> None:
    """# AT-PORT-01 — one OpenAI-compatible wire format, key from config."""
    mock_server.queue_json("hello")
    provider = http_provider_factory(remote_config)
    completion = provider.complete(MESSAGES)

    assert mock_server.paths() == ["/v1/chat/completions"]
    headers = mock_server.last_headers()
    assert headers["authorization"] == "Bearer sk-test-abcdef0123456789"
    body = mock_server.last_body()
    assert body["model"] == remote_config.model
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert body["stream"] is False
    assert completion.text == "hello"
    assert completion.input_tokens == 12
    assert completion.output_tokens == 34


def test_local_server_needs_no_key_and_sends_no_authorization_header(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    local_config: ProviderConfig,
) -> None:
    """# AT-PORT-01 — local mode is keyless by construction."""
    mock_server.queue_json("локальный ответ")
    provider = http_provider_factory(local_config)
    provider.complete(MESSAGES)

    assert local_config.requires_api_key is False
    assert local_config.api_key is None
    assert "authorization" not in mock_server.last_headers()
    assert local_config.base_url.startswith("http://127.0.0.1")


@pytest.mark.parametrize(
    "preset", ["openai", "deepseek", "openrouter", "vllm", "llama_cpp", "ollama"]
)
def test_every_preset_speaks_the_same_wire_format(
    preset: str,
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """# AT-OPS-03 — a preset only changes defaults, never the code path."""
    for env_name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "VLLM_API_KEY"):
        monkeypatch.setenv(env_name, "sk-test-0123456789")
    config = provider_config(preset)
    mock_server.queue_json(f"reply from {preset}")
    provider = http_provider_factory(config)
    completion = provider.complete(MESSAGES)

    assert completion.text == f"reply from {preset}"
    expected_path = urlsplit(config.base_url).path.rstrip("/") + "/chat/completions"
    assert mock_server.paths() == [expected_path]
    assert mock_server.last_body()["model"] == config.model


def test_provider_module_has_no_preset_specific_branches() -> None:
    """# AT-PORT-01 — CL-PORT-01 as an executable check, not a promise.

    The transport must not compare against a preset name: presets are data. The AST check
    looks for a comparison, ``if`` test or match-case whose operand is a preset-name
    literal inside ``providers.py``.
    """
    preset_literals = {
        "openai",
        "openai_compatible",
        "deepseek",
        "openrouter",
        "vllm",
        "llama_cpp",
        "ollama",
        "fake",
    }
    tree = ast.parse(PROVIDERS_MODULE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in [node.left, *node.comparators]:
                if isinstance(comparator, ast.Constant) and comparator.value in preset_literals:
                    offenders.append(
                        f"line {comparator.lineno}: comparison with {comparator.value!r}"
                    )
        if (
            isinstance(node, ast.Match)
            and isinstance(node.subject, ast.Attribute)
            and node.subject.attr == "preset"
        ):
            offenders.append(f"line {node.lineno}: match on .preset")
    assert not offenders, f"provider-specific branching found: {offenders}"


def test_structured_output_sends_a_strict_json_schema(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-GEN-04 — the answer schema travels as ``response_format.json_schema``."""
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    }
    mock_server.queue_json('{"claims": []}')
    provider = http_provider_factory()
    provider.complete(MESSAGES, response_schema=schema)

    response_format = mock_server.last_body()["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema
    assert response_format["json_schema"]["strict"] is True


def test_json_mode_falls_back_to_json_object(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — ``llm.json_mode`` is a config edit, not a code edit."""
    mock_server.queue_json('{"ok": true}')
    config = provider_config("vllm", json_mode=True)
    provider = http_provider_factory(config)
    provider.complete(MESSAGES)
    assert mock_server.last_body()["response_format"] == {"type": "json_object"}


def test_no_response_format_when_neither_schema_nor_json_mode(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    mock_server.queue_json("plain")
    provider = http_provider_factory(provider_config("vllm", json_mode=False))
    provider.complete(MESSAGES)
    assert "response_format" not in mock_server.last_body()


def test_only_chat_completions_is_ever_called(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-ML-02 — no hidden translation endpoint (NFR-ML-02)."""
    mock_server.queue_json("ok")
    provider = http_provider_factory()
    provider.complete(MESSAGES)
    assert set(mock_server.paths()) == {"/v1/chat/completions"}


def test_tool_definitions_are_forwarded_verbatim(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — the OpenAI ``tools`` subset is forwarded for embedding callers.

    Grounded answering never sends tools (autonomous action is a v1.0 non-goal); the
    transport still exposes the capability without a preset-specific branch.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its synthetic id",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
    mock_server.queue_json("ok")
    provider = http_provider_factory()
    provider.complete(MESSAGES, tools=[tool], tool_choice="auto")

    body = mock_server.last_body()
    assert body["tools"] == [tool]
    assert body["tool_choice"] == "auto"
    assert "response_format" not in body


# --------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------


def test_streaming_yields_deltas_then_done(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — SSE parsing, incremental delivery, one terminal chunk."""
    mock_server.queue_sse(["Сколько ", "стоит ", "обслуживание?"])
    provider = http_provider_factory()
    stream = provider.stream(MESSAGES)

    first = next(iter(stream))
    assert first.delta == "Сколько "
    assert first.done is False

    materialised = stream.materialise()
    assert "".join(chunk.delta for chunk in materialised) == "Сколько стоит обслуживание?"
    assert materialised[-1].done is True
    assert sum(1 for chunk in materialised if chunk.done) == 1


def test_streaming_records_estimated_tokens_when_the_server_sends_no_usage(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    ledger: CostLedger,
) -> None:
    """# AT-COST-01 — a streamed call is still on the ledger, flagged as estimated."""
    mock_server.queue_sse(["one ", "two ", "three"])
    provider = http_provider_factory()
    provider.stream(MESSAGES).materialise()

    assert len(ledger) == 1
    record = ledger.records[0]
    assert record.ok is True
    assert record.usage_source == "estimated"
    assert record.input_tokens > 0
    assert record.estimated_input_tokens is not None


# --------------------------------------------------------------------------------------
# Failures, retries and timeouts
# --------------------------------------------------------------------------------------


def test_timeout_is_retried_then_raised_with_the_configured_budget(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    sleeper: Callable[[float], None],
) -> None:
    """# AT-PORT-01 — NFR-OPS-01 failure row 'provider timeout'."""
    mock_server.queue_timeout()
    mock_server.queue_timeout()
    mock_server.queue_timeout()
    provider = http_provider_factory(
        policy=RetryPolicy(max_retries=2, base_delay_s=0.5, jitter=0.0)
    )
    with pytest.raises(ProviderTimeoutError):
        provider.complete(MESSAGES)
    assert mock_server.call_count == 3  # 1 attempt + 2 retries


def test_retryable_status_is_retried_and_fatal_status_is_not(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — 503 is transient, 401 is a bad key: do not burn retries on it."""
    mock_server.queue_status(503)
    mock_server.queue_json("recovered")
    provider = http_provider_factory(policy=RetryPolicy(max_retries=2, jitter=0.0))
    assert provider.complete(MESSAGES).text == "recovered"
    assert mock_server.call_count == 2

    mock_server.replies.clear()
    mock_server.queue_status(401)
    with pytest.raises(ProviderHTTPError) as excinfo:
        provider.complete(MESSAGES)
    assert excinfo.value.status_code == 401
    assert excinfo.value.retryable is False
    assert mock_server.call_count == 3  # only the one new call


def test_retry_after_header_is_respected(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    sleeper: object,
) -> None:
    """# AT-PORT-01 — a server-requested cooldown wins over the backoff guess."""
    mock_server.queue_status(429, headers={"retry-after": "3"})
    mock_server.queue_json("after cooldown")
    provider = http_provider_factory(policy=RetryPolicy(max_retries=1, jitter=0.0))
    assert provider.complete(MESSAGES).text == "after cooldown"
    assert sleeper.sleeps == [3.0]  # type: ignore[attr-defined]


def test_connection_error_maps_to_provider_unreachable(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — an unreachable local server is a typed, retryable error."""
    mock_server.queue_transport_error()
    provider = http_provider_factory(policy=RetryPolicy(max_retries=0))
    with pytest.raises(ProviderConnectionError) as excinfo:
        provider.complete(MESSAGES)
    assert excinfo.value.retryable is True
    assert "http://127.0.0.1" in excinfo.value.message


def test_malformed_body_is_a_response_error(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    """# AT-PORT-01 — NFR-OPS-01 failure row 'malformed model JSON'."""
    mock_server.queue(Reply(status=200, text="this is not json"))
    provider = http_provider_factory()
    with pytest.raises(ProviderResponseError):
        provider.complete(MESSAGES)


def test_empty_choices_is_a_response_error(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
) -> None:
    mock_server.queue(Reply(status=200, payload={"choices": []}))
    provider = http_provider_factory()
    with pytest.raises(ProviderResponseError):
        provider.complete(MESSAGES)


def test_retry_policy_jitter_is_bounded_and_seeded() -> None:
    """# AT-PORT-01 — jitter stays inside [0.5x, 1.5x] and is reproducible."""
    policy = RetryPolicy(max_retries=3, base_delay_s=1.0, max_delay_s=8.0, jitter=0.5)
    delays = [policy.delay_for(attempt, rng=random.Random(7)) for attempt in range(3)]
    assert delays == [policy.delay_for(attempt, rng=random.Random(7)) for attempt in range(3)]
    for attempt, delay in enumerate(delays):
        nominal = policy.nominal_delay(attempt)
        assert 0.5 * nominal <= delay <= 1.5 * nominal


# --------------------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------------------


def test_circuit_opens_after_repeated_failures_and_fails_fast(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    clock: object,
) -> None:
    """# AT-PORT-01 — a dead provider stops costing a timeout per request."""
    breaker = CircuitBreaker(
        name="http:vllm", failure_threshold=2, recovery_timeout_s=10.0, clock=clock
    )  # type: ignore[arg-type]
    provider = http_provider_factory(breaker=breaker, policy=RetryPolicy(max_retries=0))
    mock_server.queue_status(500)
    with pytest.raises(ProviderHTTPError):
        provider.complete(MESSAGES)
    mock_server.queue_status(500)
    with pytest.raises(ProviderHTTPError):
        provider.complete(MESSAGES)
    assert breaker.state is CircuitState.OPEN
    calls_before = mock_server.call_count

    with pytest.raises(CircuitOpenError):
        provider.complete(MESSAGES)
    assert mock_server.call_count == calls_before, "an open circuit must not touch the network"

    clock.advance(10.0)  # type: ignore[attr-defined]
    mock_server.queue_json("back online")
    assert provider.complete(MESSAGES).text == "back online"
    assert breaker.state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_the_circuit(
    http_provider_factory: Callable[..., HttpChatProvider],
    mock_server: MockOpenAI,
    clock: object,
) -> None:
    breaker = CircuitBreaker(
        name="http:vllm", failure_threshold=1, recovery_timeout_s=5.0, clock=clock
    )  # type: ignore[arg-type]
    provider = http_provider_factory(breaker=breaker, policy=RetryPolicy(max_retries=0))
    mock_server.queue_status(503)
    with pytest.raises(ProviderHTTPError):
        provider.complete(MESSAGES)
    assert breaker.state is CircuitState.OPEN

    clock.advance(5.0)  # type: ignore[attr-defined]
    mock_server.queue_status(503)
    with pytest.raises(ProviderHTTPError):
        provider.complete(MESSAGES)
    assert breaker.state is CircuitState.OPEN, "a failed probe must not close the circuit"


# --------------------------------------------------------------------------------------
# Fallback chain
# --------------------------------------------------------------------------------------


def test_fallback_chain_answers_from_the_next_provider(
    mock_server: MockOpenAI,
    runner: object,
    ledger: CostLedger,
) -> None:
    """# AT-PORT-01 — the chain is itself a ChatProvider; callers cannot tell."""
    primary_mock = MockOpenAI()
    secondary_mock = MockOpenAI()
    primary_mock.queue_status(503)
    secondary_mock.queue_json("secondary answer")

    primary = HttpChatProvider(
        provider_config("openai_compatible", max_retries=0),
        transport=primary_mock.transport(),
        runner=runner,  # type: ignore[arg-type]
        ledger=ledger,
    )
    secondary = HttpChatProvider(
        provider_config("vllm", max_retries=0),
        transport=secondary_mock.transport(),
        runner=runner,  # type: ignore[arg-type]
        ledger=ledger,
    )
    chain = ProviderChain([primary, secondary])
    assert chain.name.startswith("chain[")
    assert chain.complete(MESSAGES).text == "secondary answer"
    assert len(ledger) == 2
    assert [record.ok for record in ledger.records] == [False, True]


def test_chain_exhaustion_reports_every_failure(
    mock_server: MockOpenAI,
    runner: object,
) -> None:
    first = MockOpenAI()
    second = MockOpenAI()
    first.queue_status(500)
    second.queue_timeout()
    chain = ProviderChain(
        [
            HttpChatProvider(
                provider_config("openai_compatible", max_retries=0),
                transport=first.transport(),
                runner=runner,  # type: ignore[arg-type]
            ),
            HttpChatProvider(
                provider_config("vllm", max_retries=0),
                transport=second.transport(),
                runner=runner,  # type: ignore[arg-type]
            ),
        ]
    )
    with pytest.raises(AllProvidersFailedError) as excinfo:
        chain.complete(MESSAGES)
    assert [name for name, _ in excinfo.value.failures] == [
        "http:openai_compatible",
        "http:vllm",
    ]
    assert excinfo.value.code == "provider_chain_exhausted"
    assert excinfo.value.http_status == 502
    assert excinfo.value.retryable is True


def test_chain_streaming_falls_back_before_the_first_delta(
    runner: object,
) -> None:
    first = MockOpenAI()
    second = MockOpenAI()
    first.queue_status(503)
    second.queue_sse(["fallback ", "stream"])
    chain = ProviderChain(
        [
            HttpChatProvider(
                provider_config("openai_compatible", max_retries=0),
                transport=first.transport(),
                runner=runner,  # type: ignore[arg-type]
            ),
            HttpChatProvider(
                provider_config("vllm", max_retries=0),
                transport=second.transport(),
                runner=runner,  # type: ignore[arg-type]
            ),
        ]
    )
    assert chain.stream(MESSAGES).text() == "fallback stream"


def test_chain_skips_a_provider_whose_breaker_is_open(runner: object) -> None:
    first = MockOpenAI()
    second = MockOpenAI()
    second.queue_json("from secondary")
    primary = HttpChatProvider(
        provider_config("openai_compatible", max_retries=0),
        transport=first.transport(),
        runner=runner,  # type: ignore[arg-type]
    )
    secondary = HttpChatProvider(
        provider_config("vllm", max_retries=0),
        transport=second.transport(),
        runner=runner,  # type: ignore[arg-type]
    )
    breaker = CircuitBreaker(
        name="http:openai_compatible", failure_threshold=1, recovery_timeout_s=60.0
    )
    breaker.record_failure()
    chain = ProviderChain([primary, secondary], breakers={"http:openai_compatible": breaker})
    assert chain.complete(MESSAGES).text == "from secondary"
    assert first.call_count == 0
    assert second.call_count == 1


# --------------------------------------------------------------------------------------
# Client ownership
# --------------------------------------------------------------------------------------


def test_provider_closes_only_its_own_client(mock_server: MockOpenAI) -> None:
    injected = httpx.Client(transport=mock_server.transport())
    provider = HttpChatProvider(provider_config("vllm"), client=injected)
    provider.close()
    assert injected.is_closed is False, "an injected client belongs to the caller"
    owned = HttpChatProvider(provider_config("vllm"), transport=mock_server.transport())
    owned.close()
    assert owned._client.is_closed is True


def test_base_url_must_be_http(local_config: ProviderConfig) -> None:
    broken = local_config.model_copy(update={"base_url": "memory://fake"})
    with pytest.raises(ValueError, match="base_url must be http"):
        HttpChatProvider(broken)
