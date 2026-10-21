"""Hosted reranking over a Cohere-style ``POST /rerank`` endpoint (FR-RET-07, ``local``-free).

Local cross-encoders are the quality reference, but they cost a multi-gigabyte download and
a CPU-bound forward pass per query. A hosted reranker gives the same *shape* of evidence —
a query-document relevance score per candidate, ordered by the server — over one HTTP call,
which is what makes a real RT-10 measurement possible on a laptop and in the live benchmark
(t11) without shipping torch.

Wire format (Cohere-compatible, as served by NeuralDeep's ``bge-reranker``)::

    POST {base_url}/rerank
    {"model": "bge-reranker", "query": "...", "documents": ["...", "..."], "top_n": 12}
    -> {"results": [{"index": 0, "relevance_score": 0.91}, ...]}

Contract decisions worth stating:

* **Index-mapped, not order-mapped.** Results are matched to passages by the server's
  ``index``, so a server that returns the top-n in score order (or any order) cannot shift
  the mapping between a score and a chunk. Duplicate or out-of-range indices are a
  :class:`ConfigError`.
* **Batched, merged, then truncated.** ``batch_size`` documents go per request; results are
  merged by global index and only then sorted by ``(-score, chunk_id)`` and cut to ``top_n``,
  so batching never changes the output order.
* **Missing results mean "not in the server's top-n", not score 0.** Those candidates are
  dropped, which is what a caller asking for ``top_n`` wants; a response with *no* results at
  all is still accepted (an empty ranking) but a malformed body is not.
* **Failure mapping mirrors the embedder.** Transport errors, 429 and 5xx are retried with
  exponential backoff plus jitter; 4xx, malformed JSON and shape violations raise
  :class:`answerdesk.errors.ConfigError` with the fix in the message. A rerank failure must
  never silently degrade to the un-reranked order, because the trace would then claim a
  stage that did not run.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from answerdesk.config import Settings
from answerdesk.contracts import RetrievedPassage
from answerdesk.errors import ConfigError
from answerdesk.index.embedders import require_api_key, resolve_api_key
from answerdesk.index.http import build_http_client
from answerdesk.rerank.cross_encoder import _renumber

#: Backoff base (seconds) and cap, identical to the embedder's policy.
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0


class ApiReranker:
    """Cross-encoder-as-a-service behind the :class:`answerdesk.contracts.Reranker` seam."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        batch_size: int = 16,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        name: str | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not base_url:
            raise ConfigError(
                "the hosted reranker needs a base URL: set retrieval.rerank.api_base_url "
                "(e.g. https://api.neuraldeep.ru/v1)"
            )
        if not model:
            raise ConfigError(
                "the hosted reranker needs a model name: set retrieval.rerank.api_model"
            )
        if batch_size < 1:
            raise ConfigError("rerank.batch_size must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or f"api-rerank:{model}"
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self._key = api_key
        self._client = client
        self._transport = transport
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random(0)
        self._requests = 0

    # -- introspection -------------------------------------------------------------
    @property
    def request_count(self) -> int:
        """Number of HTTP requests issued — the batching assertions read this."""
        return self._requests

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ApiReranker(model={self.model!r}, base_url={self.base_url!r})"

    # -- transport -----------------------------------------------------------------
    def _client_or_create(self) -> httpx.Client:
        """The client, built lazily through the proxy-tolerant factory (see index.http)."""
        if self._client is None:
            self._client = build_http_client(timeout=self.timeout_s, transport=self._transport)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key.get_secret_value()}"
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/rerank"
        attempt = 0
        last_error = ""
        while attempt <= self.max_retries:
            self._requests += 1
            try:
                response = self._client_or_create().post(url, json=payload, headers=self._headers())
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    break
                self._backoff(attempt)
                attempt += 1
                continue
            if response.status_code == 200:
                return self._parse(response, url)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if attempt == self.max_retries:
                    break
                self._backoff(attempt, retry_after=response.headers.get("Retry-After"))
                attempt += 1
                continue
            hint = self._hint(response.status_code)
            raise ConfigError(
                f"rerank endpoint {url} rejected the request: HTTP {response.status_code} "
                f"{response.reason_phrase}. {hint}",
                detail=response.text[:400],
            )
        raise ConfigError(
            f"rerank endpoint {url} failed after {self.max_retries + 1} attempt(s): "
            f"{last_error}. Check the endpoint, the network and rerank.api_max_retries."
        )

    def _backoff(self, attempt: int, *, retry_after: str | None = None) -> None:
        delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2**attempt))
        delay *= 1.0 + 0.25 * self._rng.random()
        if retry_after:
            # A malformed header is not fatal: fall back to the computed backoff.
            with contextlib.suppress(ValueError):
                delay = max(delay, min(_BACKOFF_CAP, float(retry_after)))
        self._sleep(delay)

    @staticmethod
    def _hint(status: int) -> str:
        if status in (401, 403):
            return (
                "The API key is missing, wrong or expired: check the environment variable "
                "named by rerank.api_key_env."
            )
        if status == 404:
            return (
                "The path is wrong: rerank.api_base_url must include the API prefix "
                "(e.g. https://api.neuraldeep.ru/v1) so that /rerank resolves."
            )
        if status == 400:
            return (
                "The request was rejected: check rerank.api_model against the models the "
                "endpoint serves (e.g. bge-reranker) and that documents are non-empty."
            )
        return "See docs/retrieval.md for hosted-API mode."

    @staticmethod
    def _parse(response: httpx.Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConfigError(
                f"rerank endpoint {url} returned non-JSON ({exc}). "
                "Is the base URL pointing at the API root?"
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigError(
                f"rerank endpoint {url} returned {type(payload).__name__}, not an object"
            )
        results = payload.get("results")
        if results is not None and not isinstance(results, list):
            raise ConfigError(f"rerank endpoint {url} returned a non-list 'results' field")
        return payload

    # -- reranking -----------------------------------------------------------------
    def rerank(
        self, query: str, passages: Sequence[RetrievedPassage], top_n: int
    ) -> list[RetrievedPassage]:
        """``Reranker`` protocol: score the candidates, sort, truncate, re-rank 1..n."""
        window = list(passages)
        if not window or top_n <= 0:
            return []
        scores: dict[int, float] = {}
        for start in range(0, len(window), self.batch_size):
            batch = window[start : start + self.batch_size]
            for local_index, score in self._score_batch(query, batch, top_n):
                scores[start + local_index] = score
        if not scores:
            # The server returned no scored documents: refuse rather than pretend the stage
            # ran and hand back the un-reranked order (the trace would be a lie).
            raise ConfigError(
                f"rerank endpoint {self.base_url}/rerank returned no scored documents for "
                f"{len(window)} candidates; check rerank.api_model and the request shape."
            )
        scored = [(scores[index], window[index]) for index in sorted(scores)]
        return _renumber(scored, top_n, score_key="rerank")

    def _score_batch(
        self, query: str, batch: Sequence[RetrievedPassage], top_n: int
    ) -> list[tuple[int, float]]:
        url = f"{self.base_url}/rerank"
        documents = [passage.chunk.text for passage in batch]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        body = self._post(payload)
        rows = body.get("results") or []
        scored: list[tuple[int, float]] = []
        seen: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ConfigError(f"rerank endpoint {url} returned a non-object result row")
            index = row.get("index")
            if not isinstance(index, int) or not 0 <= index < len(documents):
                raise ConfigError(
                    f"rerank endpoint {url} returned index {index!r} outside "
                    f"0..{len(documents) - 1}; the score-to-chunk mapping is broken"
                )
            if index in seen:
                raise ConfigError(f"rerank endpoint {url} returned duplicate index {index}")
            value = row.get("relevance_score", row.get("score"))
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"rerank endpoint {url} returned a non-numeric relevance score "
                    f"for index {index}"
                )
            seen.add(index)
            scored.append((index, float(value)))
        return scored


def api_reranker_from_settings(settings: Settings) -> ApiReranker:
    """Build the hosted reranker from configuration (``retrieval.rerank.backend='api'``)."""
    section = settings.retrieval.rerank
    base_url = section.api_base_url
    if not base_url:
        raise ConfigError(
            "retrieval.rerank.backend='api' needs retrieval.rerank.api_base_url "
            "(e.g. https://api.neuraldeep.ru/v1)."
        )
    model = section.api_model or section.model
    key, _source = resolve_api_key(section.api_key, section.api_key_env)
    require_api_key(base_url, key, section.api_key_env, f"rerank backend 'api' (model {model!r})")
    return ApiReranker(
        base_url=base_url,
        model=model,
        api_key=key,
        batch_size=section.batch_size,
        timeout_s=section.api_timeout_s,
        max_retries=section.api_max_retries,
    )


__all__ = ["ApiReranker", "api_reranker_from_settings"]
