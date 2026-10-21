"""Preset table tests: names are data, defaults are data, keys come from the environment.

# AT-OPS-03  # AT-PORT-01
"""

from __future__ import annotations

import pytest

from answerdesk.config import REQUIRED_PRESETS
from answerdesk.errors import ConfigError
from answerdesk.llm.presets import (
    ALIASES,
    LOCAL_PRESETS,
    PRESETS,
    REMOTE_PRESETS,
    is_local,
    normalise_preset_name,
    preset_names,
    provider_config,
    resolve_api_key,
    resolve_preset,
)

EXPECTED_KEYS = {
    "openai",
    "openai_compatible",
    "deepseek",
    "openrouter",
    "vllm",
    "llama_cpp",
    "ollama",
    "fake",
}


def test_presets_cover_every_named_provider() -> None:
    """# AT-OPS-03 — the six presets the requirements name, plus openai and the fake."""
    assert set(preset_names()) == EXPECTED_KEYS
    assert set(REQUIRED_PRESETS) <= set(PRESETS)
    for name in REQUIRED_PRESETS:
        preset = PRESETS[name]
        assert preset.base_url.startswith(("http://", "https://"))
        assert preset.model


def test_local_presets_need_no_key_and_remote_presets_do() -> None:
    """# AT-PORT-01 — keyless local mode is a property of the table, not a special case."""
    for name in LOCAL_PRESETS:
        assert PRESETS[name].requires_api_key is False
        assert is_local(name) is True
        assert provider_config(name, env={}).api_key is None
    for name in REMOTE_PRESETS:
        assert PRESETS[name].requires_api_key is True
        assert PRESETS[name].api_key_env
        assert is_local(name) is False
    assert PRESETS["fake"].requires_api_key is False
    assert PRESETS["fake"].base_url == "memory://fake"


def test_local_presets_point_at_localhost() -> None:
    """# AT-PORT-01 — "local server" means 127.0.0.1, not a vendor endpoint."""
    for name in ("vllm", "llama_cpp", "ollama", "openai_compatible"):
        assert PRESETS[name].base_url.startswith("http://127.0.0.1")


def test_aliases_are_accepted_on_input_only() -> None:
    """# AT-OPS-03 — `llama.cpp` (docs/requirements spelling) resolves to `llama_cpp`."""
    assert normalise_preset_name("llama.cpp") == "llama_cpp"
    assert normalise_preset_name("Openai-Compatible") == "openai_compatible"
    assert resolve_preset("llama.cpp").name == "llama_cpp"
    assert resolve_preset("llama-cpp").name == "llama_cpp"
    assert ALIASES["openai-compatible"] == "openai_compatible"


def test_unknown_preset_names_the_known_ones() -> None:
    with pytest.raises(ConfigError) as excinfo:
        resolve_preset("nope")
    assert "unknown provider preset" in excinfo.value.message
    assert "openai" in excinfo.value.message


def test_provider_config_fills_defaults_from_the_table() -> None:
    """# AT-OPS-03 — base_url and model are defaults, never code branches."""
    config = provider_config("deepseek", env={"DEEPSEEK_API_KEY": "sk-deepseek-1"})
    assert config.preset == "deepseek"
    assert config.base_url == PRESETS["deepseek"].base_url
    assert config.model == PRESETS["deepseek"].model
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "sk-deepseek-1"
    assert config.requires_api_key is True


def test_explicit_values_override_preset_defaults() -> None:
    config = provider_config(
        "vllm", base_url="http://127.0.0.1:9999/v1", model="my-model", timeout_s=5.0
    )
    assert config.base_url == "http://127.0.0.1:9999/v1"
    assert config.model == "my-model"
    assert config.timeout_s == 5.0


def test_local_preset_ignores_the_environment_key() -> None:
    """A local server may accept a key, but it is never *required* and never invented."""
    preset = PRESETS["llama_cpp"]
    assert resolve_api_key(preset, env={"VLLM_API_KEY": "sk-ignored"}) is None
    config = provider_config("llama_cpp", env={"VLLM_API_KEY": "sk-ignored"})
    assert config.api_key is None


def test_remote_preset_without_key_gives_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError) as excinfo:
        provider_config("openai")
    message = excinfo.value.message
    assert "OPENAI_API_KEY" in message
    assert "vllm" in message  # names the keyless alternative


def test_resolved_provider_redacts_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-value")
    config = provider_config("openrouter")
    redacted = config.redacted()
    assert redacted["api_key"] == "***"
    assert "sk-or-secret-value" not in str(redacted)


def test_fake_preset_is_keyless_and_never_http() -> None:
    config = provider_config("fake")
    assert config.requires_api_key is False
    assert config.base_url == "memory://fake"
    assert config.model == "fake-model"


def test_registry_scope_boundaries() -> None:
    """Local and remote sets are disjoint and together cover the non-fake presets."""
    assert set(LOCAL_PRESETS) | set(REMOTE_PRESETS) | {"fake"} == EXPECTED_KEYS
    assert not (set(LOCAL_PRESETS) & set(REMOTE_PRESETS))
