"""The single OpenAI-compatible HTTP transport (ADR-003, NFR-PORT-01).

One client, one code path, zero provider-specific branches: presets only change
defaults in a :class:`~answerdesk.contracts.ProviderConfig` (``base_url``, ``model``,
key requirement). Everything the six presets need is in the chat-completions subset:

* streaming over Server-Sent Events (:meth:`HttpChatProvider.stream`);
* structured output — ``response_format`` ``json_schema`` when a schema is supplied,
  ``json_object`` when the config asks for JSON mode, nothing otherwise;
* a per-request timeout, exponential backoff with jitter, ``Retry-After`` support;
* a circuit breaker so a dead provider stops costing a timeout per request;
* a :class:`~answerdesk.llm.cost.CostLedger` record for **every** call (NFR-COST-01);
* :class:`ProviderChain` for the configured fallback order.

Invariants honoured here: INV-07 (this is the only module that constructs a model
HTTP client) and INV-08 (nothing in this module touches the network at import time or
without a caller asking for a completion).

    # AT-PORT-01  # AT-COST-01  # AT-OPS-03
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from answerdesk import PACKAGE_VERSION
from answerdesk.contracts import (
    ChatProvider,
    Completion,
    CompletionChunk,
    Message,
    ProviderConfig,
)
from answerdesk.errors import ConfigError
from answerdesk.llm.circuit import CircuitBreaker
from answerdesk.llm.cost import CostLedger, estimate_messages_tokens
from answerdesk.llm.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from answerdesk.llm.retry import RetryPolicy, Runner, parse_retry_after
from answerdesk.llm.stream import ChunkStream
from answerdesk.logging import get_logger

_LOGGER = get_logger(__name__)

#: The one endpoint every OpenAI-compatible server exposes.
CHAT_COMPLETIONS_PATH = "/chat/completions"

#: Finish reasons we keep verbatim; anything else becomes ``error`` (closed contract enum).
_FINISH_REASONS = frozenset({"stop", "length", "content_filter"})


def _error_message(exc: BaseException) -> str:
    """Human string for an exception, without leaking headers or keys."""
    if isinstance(exc, LLMError):
        return exc.message
    return str(exc)


def iter_sse_data(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse an SSE byte stream into JSON payloads.

    Deliberately tolerant: comment lines (``: ping``), blank keep-alives and the
    ``[DONE]`` sentinel are skipped, and a payload that is not JSON is ignored rather
    than crashing a stream that is already half-delivered.
    """
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        if not payload:
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            yield decoded


class HttpChatProvider:
    """One OpenAI-compatible transport for every preset (openai … ollama)."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        ledger: CostLedger | None = None,
        circuit: CircuitBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
        runner: Runner | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        schema_name: str = "answerdesk_answer",
        stage: str = "generation",
        prompt_version: str = "",
    ) -> None:
        if not config.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"provider base_url must be http(s), got {config.base_url!r} "
                "(the 'fake' preset is served by FakeProvider, not by HTTP)"
            )
        self.config = config
        self.name = f"http:{config.preset}"
        self.stage = stage
        self.prompt_version = prompt_version
        self.ledger = ledger
        self.circuit = circuit if circuit is not None else CircuitBreaker(name=self.name)
        self.retry_policy = retry_policy or RetryPolicy(max_retries=config.max_retries)
        self._runner = runner or Runner()
        self._schema_name = schema_name
        self._owns_client = client is None
        self._client = client or self._build_client(config, transport)

    # -- plumbing ----------------------------------------------------------------
    def _build_client(
        self, config: ProviderConfig, transport: httpx.BaseTransport | None
    ) -> httpx.Client:
        """Build the one client, tolerating an unusable ambient proxy configuration.

        ``trust_env=True`` is the right default (a corporate proxy must keep working), but
        on Windows ``urllib`` also reads the *registry* proxy, and a stale
        ``socks4://…`` entry there makes ``httpx`` raise before a single request is made.
        A model call to ``127.0.0.1`` must never be blocked by that, so an unusable proxy
        is ignored with a warning instead of turning into a startup crash. A genuinely
        broken explicit configuration still raises an actionable error.
        """
        timeout = httpx.Timeout(config.timeout_s)
        try:
            return httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)
        except (ValueError, ImportError) as exc:
            _LOGGER.warning(
                "ignoring unusable ambient proxy configuration for the model client",
                extra={"preset": config.preset, "reason": str(exc)},
            )
        try:
            return httpx.Client(
                timeout=timeout,
                transport=transport,
                follow_redirects=False,
                trust_env=False,
            )
        except (ValueError, ImportError) as exc:  # pragma: no cover - defensive
            raise ConfigError(
                f"cannot construct the HTTP client for provider {config.preset!r}: {exc}. "
                "httpx supports http(s) and socks5 proxies (socks5 needs the 'socksio' extra); "
                "fix or unset HTTP_PROXY / HTTPS_PROXY / ALL_PROXY."
            ) from exc

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith(CHAT_COMPLETIONS_PATH):
            return base
        return base + CHAT_COMPLETIONS_PATH

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "User-Agent": f"answerdesk/{PACKAGE_VERSION}",
        }
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"
        headers.update(self.config.headers)
        return headers

    def _body(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        response_schema: Mapping[str, Any] | None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role.value, "content": message.content} for message in messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        response_format = self._response_format(response_schema)
        if response_format is not None:
            body["response_format"] = response_format
        # Tool definitions are forwarded verbatim when a caller supplies them (the OpenAI
        # tools subset is common to every preset). Grounded answering never uses them: the
        # product's non-goal is autonomous action, so tools exist for embedding callers.
        if tools:
            body["tools"] = [dict(tool) for tool in tools]
            if tool_choice is not None:
                body["tool_choice"] = (
                    dict(tool_choice) if isinstance(tool_choice, Mapping) else tool_choice
                )
        return body

    def _response_format(self, response_schema: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Structured output: ``json_schema`` beats the coarser ``json_object`` mode."""
        if response_schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self._schema_name,
                    "schema": dict(response_schema),
                    "strict": True,
                },
            }
        if self.config.json_mode:
            return {"type": "json_object"}
        return None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpChatProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- guarded send -------------------------------------------------------------
    def _check_circuit(self) -> None:
        if not self.circuit.allow():
            raise CircuitOpenError(
                f"provider {self.config.preset!r} circuit is open after "
                f"{self.circuit.failures} failures; failing fast",
                detail=json.dumps(self.circuit.snapshot(), sort_keys=True),
            )

    def _record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        estimated_input_tokens: int | None,
        attempts: int,
        ok: bool,
        error_code: str | None = None,
        usage_source: str = "provider",
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.record(
            preset=self.config.preset,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            stage=self.stage,
            provider="http",
            prompt_version=self.prompt_version,
            estimated_input_tokens=estimated_input_tokens,
            usage_source=usage_source,  # type: ignore[arg-type]
            attempts=attempts,
            ok=ok,
            error_code=error_code,
        )

    def _fail(
        self,
        error: LLMError,
        *,
        attempts: int,
        latency_ms: float,
        estimated_input_tokens: int | None,
        model: str,
    ) -> None:
        self.circuit.record_failure()
        self._record(
            model=model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            estimated_input_tokens=estimated_input_tokens,
            attempts=attempts,
            ok=False,
            error_code=error.code,
            usage_source="none",
        )

    def _send(
        self, *, body: dict[str, Any], headers: Mapping[str, str], stream: bool
    ) -> tuple[httpx.Response, int]:
        """Send with bounded retries. Returns the 2xx response and the attempt count."""
        attempts = 0
        while True:
            attempts += 1
            error: LLMError
            retryable: bool
            wait_hint: float | None = None
            try:
                request = self._client.build_request(
                    "POST", self._endpoint(), json=body, headers=dict(headers)
                )
                response = self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                error = ProviderTimeoutError(
                    f"provider {self.config.preset!r} timed out after {self.config.timeout_s}s",
                    detail=str(exc),
                )
                retryable = self.retry_policy.should_retry(attempt=attempts - 1, timed_out=True)
            except httpx.TransportError as exc:
                error = ProviderConnectionError(
                    f"cannot reach provider {self.config.preset!r} at {self.config.base_url}: "
                    f"{exc.__class__.__name__}",
                    detail=str(exc),
                )
                retryable = self.retry_policy.should_retry(
                    attempt=attempts - 1, transport_error=True
                )
            else:
                if response.status_code < 400:
                    return response, attempts
                detail = self._safe_body(response)
                error = ProviderHTTPError(
                    f"provider {self.config.preset!r} returned HTTP {response.status_code}",
                    status_code=response.status_code,
                    retry_after_s=parse_retry_after(response.headers.get("retry-after")),
                    detail=detail,
                )
                wait_hint = error.retry_after_s
                retryable = self.retry_policy.should_retry(
                    attempt=attempts - 1, status_code=response.status_code
                )
                response.close()
            if not retryable:
                raise error
            wait = self.retry_policy.delay_for(attempts - 1, rng=self._runner.rng)
            if wait_hint is not None:
                wait = max(wait, wait_hint)
            self._runner.wait(wait)

    @staticmethod
    def _safe_body(response: httpx.Response) -> str:
        """A short, secret-free excerpt of an error body for the message/detail field."""
        try:
            if not response.is_stream_consumed:
                response.read()
            text = response.text
        except Exception:  # pragma: no cover - defensive: body could be unreadable
            return ""
        if not text:
            return ""
        compact = " ".join(text.split())
        return compact[:400]

    # -- completion ---------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> Completion:
        """One chat completion, retried, cost-accounted and circuit-broken."""
        self._check_circuit()
        estimated = estimate_messages_tokens(message.content for message in messages)
        body = self._body(
            messages,
            stream=False,
            response_schema=response_schema,
            tools=tools,
            tool_choice=tool_choice,
        )
        started = self._runner.clock()
        model = self.config.model
        try:
            response, attempts = self._send(
                body=body, headers=self._headers(accept="application/json"), stream=False
            )
            latency_ms = (self._runner.clock() - started) * 1000.0
            try:
                payload = response.json()
            except ValueError as exc:  # JSONDecodeError subclasses ValueError
                raise ProviderResponseError(
                    f"provider {self.config.preset!r} returned a non-JSON body",
                    detail=str(exc),
                ) from exc
            completion = self._completion_from(payload, latency_ms=latency_ms)
        except ProviderResponseError as exc:
            latency_ms = (self._runner.clock() - started) * 1000.0
            self._fail(
                exc,
                attempts=1,
                latency_ms=latency_ms,
                estimated_input_tokens=estimated,
                model=model,
            )
            raise
        except LLMError as exc:
            latency_ms = (self._runner.clock() - started) * 1000.0
            self._fail(
                exc,
                attempts=self.retry_policy.max_retries + 1,
                latency_ms=latency_ms,
                estimated_input_tokens=estimated,
                model=model,
            )
            raise
        self.circuit.record_success()
        self._record(
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=completion.latency_ms,
            estimated_input_tokens=estimated,
            attempts=attempts,
            ok=True,
            usage_source="provider",
        )
        return completion

    def _completion_from(self, payload: Any, *, latency_ms: float) -> Completion:
        if not isinstance(payload, dict):
            raise ProviderResponseError("provider response is not a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError("provider response contains no choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderResponseError("provider response message has no text content")
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        finish_reason = choice.get("finish_reason") or "stop"
        return Completion(
            text=content,
            model=str(payload.get("model") or self.config.model),
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=finish_reason if finish_reason in _FINISH_REASONS else "error",  # type: ignore[arg-type]
            latency_ms=round(latency_ms, 3),
        )

    # -- streaming ----------------------------------------------------------------
    #: The service-side seam may pass the schema so a streamed draft is JSON-constrained.
    accepts_stream_schema = True

    def stream(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> ChunkStream:
        """Streaming completion, exposed as a lazily materialised ``ChunkStream``.

        ``response_schema`` is forwarded as ``response_format`` (json_schema) when the provider is
        configured for JSON mode, so a *streamed* turn is constrained exactly like the
        non-streaming one. Without it the model answers in prose, the draft fails to parse, and a
        streamed answer could never be grounded — the t49 measurement hit precisely that.
        """
        return ChunkStream(self._stream_chunks(messages, response_schema=response_schema))

    def _stream_chunks(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Iterator[CompletionChunk]:
        """Yield deltas from an SSE response, then exactly one ``done=True`` chunk.

        Fallback semantics (see :class:`ProviderChain`): a failure *before* the first
        delta may be retried or handed to the next provider; a failure mid-stream is
        propagated, because the caller has already seen partial output.
        """
        self._check_circuit()
        estimated = estimate_messages_tokens(message.content for message in messages)
        body = self._body(messages, stream=True, response_schema=response_schema)
        started = self._runner.clock()
        model = self.config.model
        try:
            response, attempts = self._send(
                body=body, headers=self._headers(accept="text/event-stream"), stream=True
            )
        except LLMError as exc:
            latency_ms = (self._runner.clock() - started) * 1000.0
            self._fail(
                exc,
                attempts=self.retry_policy.max_retries + 1,
                latency_ms=latency_ms,
                estimated_input_tokens=estimated,
                model=model,
            )
            raise

        collected: list[str] = []
        usage: dict[str, Any] = {}
        try:
            for payload in iter_sse_data(response.iter_lines()):
                response_model = payload.get("model")
                if isinstance(response_model, str) and response_model:
                    model = response_model
                payload_usage = payload.get("usage")
                if isinstance(payload_usage, dict):
                    usage = payload_usage
                delta_text = _delta_text(payload)
                if delta_text:
                    collected.append(delta_text)
                    yield CompletionChunk(delta=delta_text, done=False, model=model)
        except httpx.HTTPError as exc:
            latency_ms = (self._runner.clock() - started) * 1000.0
            self._fail(
                ProviderConnectionError(f"stream from provider {self.config.preset!r} broke"),
                attempts=attempts,
                latency_ms=latency_ms,
                estimated_input_tokens=estimated,
                model=model,
            )
            raise ProviderConnectionError(
                f"stream from provider {self.config.preset!r} broke", detail=str(exc)
            ) from exc
        finally:
            response.close()
        latency_ms = (self._runner.clock() - started) * 1000.0
        self.circuit.record_success()
        streamed = "".join(collected)
        provider_tokens = {
            "input": _as_int(usage.get("prompt_tokens")),
            "output": _as_int(usage.get("completion_tokens")),
        }
        has_usage = bool(provider_tokens["input"] or provider_tokens["output"])
        self._record(
            model=model,
            input_tokens=provider_tokens["input"] if has_usage else estimated,
            output_tokens=provider_tokens["output"] if has_usage else len(streamed) // 4,
            latency_ms=latency_ms,
            estimated_input_tokens=estimated,
            attempts=attempts,
            ok=True,
            usage_source="provider" if has_usage else "estimated",
        )
        yield CompletionChunk(delta="", done=True, model=model)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _delta_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


class ProviderChain:
    """Ordered fallback across providers, each with its own circuit breaker.

    The chain is itself a :class:`~answerdesk.contracts.ChatProvider`, so callers never
    know whether one provider or five are configured (NFR-PORT-01).
    """

    def __init__(
        self,
        providers: Sequence[ChatProvider],
        *,
        name: str | None = None,
        breakers: Mapping[str, CircuitBreaker] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("ProviderChain needs at least one provider")
        self.providers = list(providers)
        self.name = name or "chain[" + ",".join(provider.name for provider in self.providers) + "]"
        self.config = self.providers[0].config
        self.breakers: dict[str, CircuitBreaker] = dict(breakers or {})

    def _breaker(self, provider: ChatProvider) -> CircuitBreaker | None:
        return self.breakers.get(provider.name)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        response_schema: Mapping[str, Any] | None = None,
    ) -> Completion:
        failures: list[tuple[str, str]] = []
        for provider in self.providers:
            breaker = self._breaker(provider)
            if breaker is not None and not breaker.allow():
                failures.append((provider.name, "circuit open"))
                continue
            try:
                completion = provider.complete(messages, response_schema=response_schema)
            except LLMError as exc:
                failures.append((provider.name, _error_message(exc)))
                continue
            if breaker is not None:
                breaker.record_success()
            return completion
        raise AllProvidersFailedError(
            f"all {len(self.providers)} providers failed", failures=failures
        )

    def stream(self, messages: Sequence[Message]) -> ChunkStream:
        return ChunkStream(self._stream_from_any(messages))

    def _stream_from_any(self, messages: Sequence[Message]) -> Iterator[CompletionChunk]:
        failures: list[tuple[str, str]] = []
        for provider in self.providers:
            breaker = self._breaker(provider)
            if breaker is not None and not breaker.allow():
                failures.append((provider.name, "circuit open"))
                continue
            try:
                iterator = iter(provider.stream(messages))
                first = next(iterator, None)
            except LLMError as exc:
                failures.append((provider.name, _error_message(exc)))
                continue
            if first is None:
                return
            if breaker is not None:
                breaker.record_success()
            yield first
            yield from iterator
            return
        raise AllProvidersFailedError(
            f"all {len(self.providers)} providers failed to stream", failures=failures
        )


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "HttpChatProvider",
    "ProviderChain",
    "iter_sse_data",
]
