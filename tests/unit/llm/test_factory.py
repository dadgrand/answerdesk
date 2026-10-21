"""Provider factory: settings in, provider (or chain) out, no key required locally.

# AT-OPS-03  # AT-OPS-04  # AT-PORT-01
"""

from __future__ import annotations

import pytest

from answerdesk.config import Settings, load_settings
from answerdesk.errors import ConfigError
from answerdesk.llm.factory import build_provider, fallback_configs, provider_sequence
from answerdesk.llm.providers import HttpChatProvider, ProviderChain
from answerdesk.testing.doubles import FakeProvider


def test_offline_profile_builds_the_deterministic_double() -> None:
    """# AT-OPS-04 — the default profile never touches the network."""
    provider = build_provider(load_settings(profile="offline"))
    assert isinstance(provider, FakeProvider)
    assert provider.config.preset == "fake"


def test_local_profile_builds_one_http_client_for_a_local_server() -> None:
    """# AT-PORT-01 — switching to a local server is a config edit, not a code edit."""
    settings = load_settings(profile="local", overrides={"llm": {"preset": "vllm"}})
    provider = build_provider(settings)
    assert isinstance(provider, HttpChatProvider)
    assert provider.config.base_url.startswith("http://127.0.0.1")
    assert provider.config.requires_api_key is False
    provider.close()


def test_force_fake_ignores_the_api_profile_and_its_key_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """# AT-OPS-03 — `ask --fake` must work with no key at all."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(profile="api")
    provider = build_provider(settings, force_fake=True)
    assert isinstance(provider, FakeProvider)
    assert provider.config.preset == "fake"


def test_fallback_presets_become_a_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """# AT-PORT-01 — the fallback order is configuration, and the chain is a provider."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(
        profile="local",
        overrides={"llm": {"preset": "ollama", "fallback_presets": ["vllm"]}},
    )
    provider = build_provider(settings)
    assert isinstance(provider, ProviderChain)
    assert [p.config.preset for p in provider.providers] == ["ollama", "vllm"]


def test_unkeyed_remote_fallback_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fallback whose key is unset is simply unavailable — logged, not an error."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = load_settings(
        profile="local",
        overrides={"llm": {"preset": "vllm", "fallback_presets": ["deepseek"]}},
    )
    assert fallback_configs(settings) == []
    provider = build_provider(settings)
    assert isinstance(provider, HttpChatProvider)
    provider.close()


def test_fallback_with_a_key_is_kept_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-fallback")
    settings = load_settings(
        profile="local",
        overrides={
            "llm": {"preset": "vllm", "fallback_presets": ["deepseek", "vllm"]},
        },
    )
    configs = fallback_configs(settings)
    assert [config.preset for config in configs] == ["deepseek", "vllm"]
    provider = build_provider(settings)
    assert isinstance(provider, ProviderChain)
    assert [p.config.preset for p in provider.providers] == ["vllm", "deepseek"]


def test_provider_sequence_puts_the_primary_first() -> None:
    settings = Settings()
    sequence = provider_sequence(settings)
    assert sequence[0].preset == "fake"


def test_api_profile_without_a_key_still_fails_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """# AT-OPS-05 — the factory surfaces the config error rather than a mystery 401."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_provider(load_settings(profile="api"))
