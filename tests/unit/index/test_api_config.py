"""Hosted-API configuration surface and its effect on the index fingerprint.

Two contracts are checked here: the configuration layer (keys exist in ``config.py`` *and*
``configs/default.yaml``, unknown keys are still rejected, secrets stay ``SecretStr``) and
the index fingerprint (switching embedders must invalidate the index — FR-OPS-02).

    # AT-OPS-02  # AT-NFR-SEC-03  # AT-NFR-SEC-02
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from answerdesk.config import EmbeddingSettings, RerankSettings, load_settings
from answerdesk.errors import ConfigError
from answerdesk.index.bm25_index import BM25Params
from answerdesk.index.embedders import (
    ApiEmbedder,
    api_embedder_from_settings,
    default_embedder,
    resolve_api_key,
)
from answerdesk.index.store import (
    ChunkStore,
    EmbedderInfo,
    IndexBundle,
    compute_index_version,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SECRET = "sk-live-0123456789abcdef"


def _settings(**embedding: object):
    return load_settings(
        profile="api",
        overrides={
            "embedding": {
                "backend": "api",
                "api_base_url": "https://api.neuraldeep.ru/v1",
                "api_model": "bge-m3",
                "api_dimension": 1024,
                **embedding,
            }
        },
    )


def test_default_yaml_and_code_carry_the_same_hosted_keys() -> None:
    """Architect's rule: a key exists in ``config.py`` and ``configs/default.yaml`` together."""
    import yaml

    from answerdesk.config import Settings

    yaml_text = (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    payload = yaml.safe_load(yaml_text)
    for model, section in (
        (EmbeddingSettings, payload["embedding"]),
        (RerankSettings, payload["retrieval"]["rerank"]),
    ):
        code_keys = set(model.model_fields)
        yaml_keys = set(section)
        assert code_keys == yaml_keys, (
            f"{model.__name__} drift: only in code {sorted(code_keys - yaml_keys)}, "
            f"only in yaml {sorted(yaml_keys - code_keys)}"
        )
    assert "embedding" in set(Settings.model_fields)
    assert "api_base_url" in EmbeddingSettings.model_fields
    assert "api_key_env" in EmbeddingSettings.model_fields
    assert "backend" in RerankSettings.model_fields
    assert "api_base_url" in RerankSettings.model_fields


def test_hosted_keys_load_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_EMBEDDING__API_BASE_URL", "https://api.neuraldeep.ru/v1")
    monkeypatch.setenv("ANSWERDESK_EMBEDDING__API_MODEL", "bge-m3")
    monkeypatch.setenv("ANSWERDESK_EMBEDDING__API_KEY", SECRET)
    monkeypatch.setenv("ANSWERDESK_RETRIEVAL__RERANK__BACKEND", "api")
    settings = load_settings(profile="offline")

    assert settings.embedding.api_base_url == "https://api.neuraldeep.ru/v1"
    assert settings.embedding.api_model == "bge-m3"
    assert isinstance(settings.embedding.api_key, SecretStr)
    assert settings.retrieval.rerank.backend == "api"
    # A SecretStr never renders its value: not in the dump, not in the redacted view.
    assert SECRET not in json.dumps(settings.redacted())
    assert settings.redacted()["embedding"]["api_key"] == "**********"


def test_unknown_hosted_key_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_EMBEDDING__API_BOGUS", "1")
    with pytest.raises(ConfigError, match="api_bogus"):
        load_settings(profile="api")


def test_unknown_rerank_backend_is_rejected() -> None:
    with pytest.raises(ValidationError, match="backend"):
        RerankSettings(backend="colbert")


def test_default_embedder_builds_the_hosted_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", SECRET)
    embedder = default_embedder(_settings())
    assert isinstance(embedder, ApiEmbedder)
    assert embedder.name == "api:bge-m3"
    assert embedder.dimension == 1024  # declared via api_dimension
    assert SECRET not in repr(embedder)


def test_api_backend_needs_a_base_url() -> None:
    settings = _settings()
    settings.embedding.api_base_url = None
    with pytest.raises(ConfigError, match="api_base_url"):
        api_embedder_from_settings(settings)


def test_api_backend_needs_a_key_for_a_remote_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("ANSWERDESK_EMBEDDING__API_KEY", raising=False)
    with pytest.raises(ConfigError) as error:
        api_embedder_from_settings(_settings())
    message = str(error.value)
    assert "OPENAI_COMPATIBLE_API_KEY" in message
    assert SECRET not in message


def test_loopback_endpoint_may_run_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    settings = _settings(api_base_url="http://127.0.0.1:8000/v1")
    embedder = api_embedder_from_settings(settings)
    assert isinstance(embedder, ApiEmbedder)


def test_key_resolution_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_TEST_EMBED_KEY", "from-env")
    assert resolve_api_key(SecretStr("from-config"), "ANSWERDESK_TEST_EMBED_KEY") == (
        SecretStr("from-config"),
        "config",
    )
    key, source = resolve_api_key(None, "ANSWERDESK_TEST_EMBED_KEY")
    assert key is not None and key.get_secret_value() == "from-env"
    assert source == "env:ANSWERDESK_TEST_EMBED_KEY"
    assert resolve_api_key(None, "ANSWERDESK_MISSING_KEY") == (None, "none")
    assert resolve_api_key(None, None) == (None, "none")


def test_index_version_changes_when_the_embedder_identity_changes() -> None:
    """FR-OPS-02: a different embedder is a different index, not a quieter one."""
    base = dict(
        corpus_version="corpus-test",
        chunks_sha256="0" * 64,
        bm25_params=BM25Params(),
    )
    hashing = compute_index_version(**base, embedder=EmbedderInfo(name="hashing-64", dimension=64))
    hosted = compute_index_version(**base, embedder=EmbedderInfo(name="api:bge-m3", dimension=1024))
    wrong_width = compute_index_version(
        **base, embedder=EmbedderInfo(name="api:bge-m3", dimension=768)
    )
    assert len({hashing, hosted, wrong_width}) == 3


def test_bundle_fingerprint_uses_the_discovered_dimension(make_chunk) -> None:
    """A hosted embedder that discovers its width must not fingerprint as dimension 0."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [1.0, 0.0, 0.0, 0.5]}
                    for index, _text in enumerate(body["input"])
                ]
            },
        )

    chunks = [make_chunk(chunk_id=f"c{index}::0", text=f"passage {index}") for index in range(4)]
    store = ChunkStore.from_chunks(chunks)

    def build(width_response: list[float]) -> IndexBundle:
        def _handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": width_response}
                        for index, _text in enumerate(body["input"])
                    ]
                },
            )

        embedder = ApiEmbedder(
            base_url="http://127.0.0.1:8000/v1",
            model="bge-m3",
            batch_size=2,
            client=httpx.Client(transport=httpx.MockTransport(_handler)),
            sleep=lambda _seconds: None,
            rng=random.Random(0),
        )
        return IndexBundle.build(store, embedder)

    bundle = build([1.0, 0.0, 0.0, 0.5])
    assert bundle.manifest.embedder.dimension == 4
    assert bundle.manifest.embedder.name == "api:bge-m3"
    other = build([1.0, 0.0, 0.0, 0.5, 0.25, 0.1])
    assert other.manifest.embedder.dimension == 6
    assert other.manifest.index_version != bundle.manifest.index_version

    # The handler above is unused on purpose: each build() constructs its own transport.
    _ = handler
