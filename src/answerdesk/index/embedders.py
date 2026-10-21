"""Embedder selection: deterministic hashing in CI, hosted API or a local model elsewhere.

The retrieval stack never imports an embedding library itself; it receives an
:class:`answerdesk.contracts.Embedder`. This module is the one place that turns
configuration into an embedder:

======================================  =========================================
``embedding.backend``                    provider
======================================  =========================================
``hashing`` (default, CI, offline)      :class:`answerdesk.testing.doubles.HashingEmbedder`
``bge_m3`` / ``qwen3``                  sentence-transformers, ``local`` extra
``api``                                 hosted OpenAI-compatible ``/embeddings``
======================================  =========================================

Why the hashing embedder is allowed at all: it makes the *machinery* testable offline
(exact-NN equivalence, fusion ablations, determinism) at zero download cost. It is
explicitly **not** a quality claim — requirements §6.3 forbids its numbers from appearing
in ``results.md``.

Why a hosted embedder exists: measuring quality needs real multilingual vectors, and a
multi-gigabyte model download is a bad first-run experience on a laptop or in CI.
:class:`ApiEmbedder` speaks the OpenAI-compatible embeddings wire format
(``POST {base_url}/embeddings``), so NeuralDeep's ``bge-m3`` (1024-dim), OpenAI, vLLM,
llama.cpp and Ollama are the same code path with different defaults (NFR-PORT-01 applied
to retrieval).

Credential handling is part of the contract, not an implementation detail:

* the key is a :class:`pydantic.SecretStr` and never appears in a log line, an exception
  message, ``repr()`` or a ``model_dump()`` (INV-05);
* it is resolved as ``embedding.api_key`` first, then ``os.environ[embedding.api_key_env]``;
* a key is **required** for a non-loopback host and **optional** for a loopback one, so a
  local gateway keeps working while a public endpoint fails with a message that names the
  environment variable to set.

Failure mapping is deliberately narrow: transport errors, 429 and 5xx are retried with
exponential backoff plus jitter; everything else (401/403/404/400, malformed JSON, a
dimension mismatch) raises :class:`answerdesk.errors.ConfigError` with the fix in the
message. Retrieval never continues on a degraded vector space.

The local embedder loads its model on first use, not at construction, so ``answerdesk
index`` can report a configuration error before touching torch, and importing this module
stays free of heavy dependencies (NFR-PORT-03).
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import SecretStr

from answerdesk.config import Settings
from answerdesk.contracts import Embedder
from answerdesk.errors import ConfigError
from answerdesk.index.http import build_http_client
from answerdesk.testing.doubles import HashingEmbedder

#: Models the local backend knows how to load, and their vector dimension.
LOCAL_MODELS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large-instruct": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-8B": 4096,
}

#: Loopback hosts where an API key may legitimately be absent.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

#: Backoff base (seconds) and cap for the retry loop.
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0


def resolve_api_key(
    explicit: SecretStr | None, env_name: str | None
) -> tuple[SecretStr | None, str]:
    """Resolve a key from configuration or the environment; return ``(key, source)``.

    ``source`` is a safe-to-log description (``config``, ``env:NAME`` or ``none``): it says
    where the key came from, never what it is.
    """
    if explicit is not None:
        return explicit, "config"
    if env_name:
        raw = os.environ.get(env_name)
        if raw and raw.strip():
            return SecretStr(raw.strip()), f"env:{env_name}"
    return None, "none"


def is_loopback(base_url: str) -> bool:
    """True when the URL points at this machine (a key may then be omitted)."""
    host = urlsplit(base_url).hostname or ""
    return host.lower() in _LOOPBACK_HOSTS


def require_api_key(base_url: str, key: SecretStr | None, env_name: str, what: str) -> None:
    """Raise an actionable ConfigError when a remote endpoint has no key (INV-05)."""
    if key is not None or is_loopback(base_url):
        return
    raise ConfigError(
        f"{what} at {base_url} has no API key. Set {env_name} in the environment, or the "
        "matching ANSWERDESK_*__API_KEY setting. A loopback host may omit the key."
    )


class LocalSentenceTransformerEmbedder:
    """Sentence-transformers backend (lazy, ``local`` extra, FR-RET-02 local mode).

    ``name`` and ``dimension`` are available before the model is loaded, because both are
    part of the index fingerprint: two different embedders must never share an index.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        dimension: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.name = f"st:{model_name}"
        self.dimension = dimension or LOCAL_MODELS.get(model_name, 1024)
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:  # pragma: no cover - requires the local extra and model weights
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised by the docs, not CI
            raise ConfigError(
                f"embedding backend {self.model_name!r} needs the 'local' extra: "
                "run `uv sync --extra local`, or set ANSWERDESK_EMBEDDING__BACKEND=api for "
                "the hosted bge-m3 endpoint, or keep 'hashing' for the offline profile."
            ) from exc
        device = None if self.device == "auto" else self.device
        model = SentenceTransformer(self.model_name, device=device)
        declared = getattr(model, "get_sentence_embedding_dimension", None)
        if callable(declared):  # pragma: no cover - depends on the installed version
            resolved = int(declared())
            if resolved != self.dimension:
                raise ConfigError(
                    f"model {self.model_name!r} produces {resolved}-dimensional vectors but "
                    f"{self.dimension} was configured; fix embedding.dimension"
                )
        self._model = model
        return model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in np.asarray(vectors)]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed_documents([text])[0]


class ApiEmbedder:
    """Hosted embeddings over the OpenAI-compatible wire format (``POST /embeddings``).

    Deterministic with respect to the *client*: batches preserve input order, and each
    response is re-ordered by its own ``index`` field when the server returns one, so a
    server that answers out of order cannot scramble vectors relative to chunks. That
    misalignment would be invisible downstream and catastrophic for retrieval.

    ``dimension`` is discovered from the first successful response and pinned: every later
    batch is validated against it and against ``declared_dimension`` when the operator set
    ``embedding.api_dimension``. A mismatch raises :class:`ConfigError` — never a silent
    reshape, because the index fingerprint depends on this number.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        declared_dimension: int | None = None,
        batch_size: int = 32,
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
                "the hosted embedder needs a base URL: set embedding.api_base_url "
                "(e.g. https://api.neuraldeep.ru/v1)"
            )
        if not model:
            raise ConfigError("the hosted embedder needs a model name: set embedding.api_model")
        if batch_size < 1:
            raise ConfigError("embedding batch_size must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or f"api:{model}"
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self._key = api_key
        self._declared_dimension = declared_dimension
        #: Vector width: declared, or discovered from the first response (0 before). It is a
        #: plain attribute because the frozen ``Embedder`` protocol declares it as one.
        self.dimension = declared_dimension or 0
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
        return (
            f"ApiEmbedder(model={self.model!r}, base_url={self.base_url!r}, "
            f"dimension={self.dimension})"
        )

    # -- transport -----------------------------------------------------------------
    def _client_or_create(self) -> httpx.Client:
        """The client, built lazily through the proxy-tolerant factory (see index.http).

        ``trust_env`` stays on so a corporate proxy keeps working; an *unusable* ambient
        proxy (``socks4://…``, ``socks5://`` without ``socksio``) is ignored with a warning
        instead of raising ``ValueError`` at construction and killing live retrieval.
        """
        if self._client is None:
            self._client = build_http_client(timeout=self.timeout_s, transport=self._transport)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key is not None:
            headers["Authorization"] = f"Bearer {self._key.get_secret_value()}"
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries on 429/5xx/transport errors; ConfigError on everything else."""
        url = f"{self.base_url}/embeddings"
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
                f"embeddings endpoint {url} rejected the request: HTTP "
                f"{response.status_code} {response.reason_phrase}. {hint}",
                detail=response.text[:400],
            )
        raise ConfigError(
            f"embeddings endpoint {url} failed after {self.max_retries + 1} attempt(s): "
            f"{last_error}. Check the endpoint, the network and embedding.api_max_retries."
        )

    def _backoff(self, attempt: int, *, retry_after: str | None = None) -> None:
        delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2**attempt))
        # Jitter keeps a fleet of clients from retrying in lockstep; the RNG is seeded so
        # tests — and any reproducibility claim — see a deterministic sequence.
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
                "named by embedding.api_key_env."
            )
        if status == 404:
            return (
                "The path is wrong: embedding.api_base_url must include the API prefix "
                "(e.g. https://api.neuraldeep.ru/v1) so that /embeddings resolves."
            )
        if status == 400:
            return (
                "The request was rejected: check embedding.api_model against the models the "
                "endpoint serves (e.g. bge-m3) and that inputs are non-empty strings."
            )
        return "See docs/retrieval.md for hosted-API mode."

    @staticmethod
    def _parse(response: httpx.Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConfigError(
                f"embeddings endpoint {url} returned non-JSON ({exc}). "
                "Is the base URL pointing at the API root?"
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigError(
                f"embeddings endpoint {url} returned {type(payload).__name__}, not an object"
            )
        return payload

    # -- embedding -----------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(
                    "the hosted embedder refuses empty inputs: a zero vector would poison the "
                    "cosine index. Filter empty chunks before indexing."
                )
        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self._embed_batch(items[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload = self._post({"model": self.model, "input": list(batch)})
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ConfigError(
                f"embeddings endpoint {url} returned no 'data' array; the response is not an "
                "OpenAI-compatible embeddings payload."
            )
        if len(rows) != len(batch):
            raise ConfigError(
                f"embeddings endpoint {url} returned {len(rows)} vectors for {len(batch)} "
                "inputs; the index would be misaligned. Check embedding.batch_size against "
                "the endpoint's maximum batch size."
            )
        indexed: list[tuple[int, list[float]]] = []
        for position, row in enumerate(rows):
            if not isinstance(row, dict) or "embedding" not in row:
                raise ConfigError(f"embeddings endpoint {url} returned a row without 'embedding'")
            vector = row["embedding"]
            if not isinstance(vector, list) or not vector:
                raise ConfigError(f"embeddings endpoint {url} returned an empty vector")
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"embeddings endpoint {url} returned a non-numeric vector ({exc})"
                ) from exc
            index = row.get("index")
            placement = index if isinstance(index, int) else position
            indexed.append((placement, values))
        placements = [placement for placement, _ in indexed]
        if sorted(placements) == list(range(len(batch))):
            indexed.sort(key=lambda item: item[0])
        self._pin_dimension(len(indexed[0][1]), url)
        return [values for _, values in indexed]

    def _pin_dimension(self, observed: int, url: str) -> None:
        if self._declared_dimension is not None and observed != self._declared_dimension:
            raise ConfigError(
                f"embeddings endpoint {url} returned {observed}-dimensional vectors but "
                f"embedding.api_dimension declares {self._declared_dimension}. Set "
                "api_dimension to the model's real width (bge-m3 is 1024) or null to "
                "discover it automatically."
            )
        if self.dimension and self.dimension != observed:
            raise ConfigError(
                f"embeddings endpoint {url} changed dimensionality mid-run "
                f"({self.dimension} -> {observed}); the model or the endpoint changed "
                "underneath the index. Rebuild the index."
            )
        self.dimension = observed

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def api_embedder_from_settings(settings: Settings) -> ApiEmbedder:
    """Build the hosted embedder from configuration (``embedding.backend='api'``)."""
    section = settings.embedding
    base_url = section.api_base_url
    if not base_url:
        raise ConfigError(
            "embedding.backend='api' needs embedding.api_base_url "
            "(e.g. https://api.neuraldeep.ru/v1)."
        )
    model = section.api_model or section.model
    key, _source = resolve_api_key(section.api_key, section.api_key_env)
    require_api_key(
        base_url, key, section.api_key_env, f"embedding backend 'api' (model {model!r})"
    )
    return ApiEmbedder(
        base_url=base_url,
        model=model,
        api_key=key,
        declared_dimension=section.api_dimension,
        batch_size=section.batch_size,
        timeout_s=section.api_timeout_s,
        max_retries=section.api_max_retries,
    )


def default_embedder(settings: Settings) -> Embedder:
    """Resolve the embedder for the configured backend (never touches the network)."""
    backend = settings.embedding.backend
    if backend == "hashing":
        return HashingEmbedder(dimension=settings.embedding.dimension)
    if backend in ("bge_m3", "qwen3"):
        model = settings.embedding.model
        # `embedding.dimension` is the *hashing* width; a real model dictates its own.
        return LocalSentenceTransformerEmbedder(
            model,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
            dimension=LOCAL_MODELS.get(model),
        )
    return api_embedder_from_settings(settings)


__all__ = [
    "LOCAL_MODELS",
    "ApiEmbedder",
    "LocalSentenceTransformerEmbedder",
    "api_embedder_from_settings",
    "default_embedder",
    "is_loopback",
    "require_api_key",
    "resolve_api_key",
]
