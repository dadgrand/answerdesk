"""Configuration tests: presets, profiles, precedence and redaction.

Maps to FR-OPS-03 (provider presets by configuration only), FR-OPS-04 (profiles),
FR-OPS-05 (actionable validation, no secret leakage) and NFR-PORT-01.

    # AT-OPS-03  # AT-OPS-04  # AT-OPS-05
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from answerdesk.config import (
    PROVIDER_PRESETS,
    REQUIRED_PRESETS,
    Settings,
    apply_profile,
    default_config_path,
    load_settings,
)
from answerdesk.contracts import IntentBucket
from answerdesk.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_required_provider_presets_exist() -> None:
    """The six presets named by FR-OPS-03, plus openai and the CI double."""
    for name in REQUIRED_PRESETS:
        assert name in PROVIDER_PRESETS, name
        preset = PROVIDER_PRESETS[name]
        assert preset.base_url.startswith(("http://", "https://", "memory://"))
        assert preset.model
    for local in ("vllm", "llama_cpp", "ollama", "fake", "openai_compatible"):
        assert PROVIDER_PRESETS[local].requires_api_key is False


def test_defaults_are_the_frozen_offline_profile() -> None:
    settings = Settings()
    assert settings.profile == "offline"
    assert settings.llm.preset == "fake"
    assert settings.embedding.backend == "hashing"
    assert settings.retrieval.rerank.enabled is False
    assert settings.retrieval.fuser == "rrf"
    assert settings.retrieval.rrf_k == 60


def test_policy_defaults_match_the_requirements() -> None:
    """Appendix C: alpha tiers 0.05 / 0.02 / 0.01 and costs 1.00 / 0.10 / 0.05."""
    policy = Settings().policy
    assert policy.alpha[IntentBucket.INFORMATIONAL] == pytest.approx(0.05)
    assert policy.alpha[IntentBucket.TRANSACTIONAL] == pytest.approx(0.02)
    assert policy.alpha[IntentBucket.ACCOUNT_LEGAL] == pytest.approx(0.01)
    assert policy.costs.c_wrong == pytest.approx(1.0)
    assert policy.costs.c_abstain == pytest.approx(0.10)
    assert policy.costs.c_delay == pytest.approx(0.05)
    # cost-derived bands of requirements §8.3
    assert policy.costs.answer_threshold == pytest.approx(0.95)
    assert policy.costs.escalate_threshold == pytest.approx(0.90)
    # alpha_final = min(alpha_bucket, c_abstain / c_wrong)
    assert policy.effective_alpha(IntentBucket.INFORMATIONAL) == pytest.approx(0.05)
    assert policy.effective_alpha(IntentBucket.TRANSACTIONAL) == pytest.approx(0.02)


def test_cache_and_eval_defaults() -> None:
    settings = Settings()
    assert settings.cache.enabled is True
    assert settings.cache.epsilon == pytest.approx(0.01)
    assert settings.cache.ttl_class["escalation"] == 600
    assert "refund" in settings.cache.high_risk_intents
    assert settings.eval.seed == 1337
    assert settings.eval.bootstrap_samples == 1000


def test_yaml_defaults_match_code_defaults() -> None:
    """configs/default.yaml must not drift away from the code defaults."""
    path = default_config_path()
    assert path is not None and path.is_file(), "configs/default.yaml is missing"
    loaded = load_settings(path)
    code = Settings()
    assert loaded.profile == code.profile
    assert loaded.retrieval.rrf_k == code.retrieval.rrf_k
    assert loaded.retrieval.fuser == code.retrieval.fuser
    assert loaded.generate.context_k_max == code.generate.context_k_max
    assert loaded.policy.alpha == code.policy.alpha
    assert loaded.cache.ttl_class == code.cache.ttl_class


def test_profiles_switch_behaviour_without_code_changes() -> None:
    """FR-OPS-04: offline / api / local are pure configuration."""
    offline = load_settings(profile="offline")
    assert offline.llm.preset == "fake"
    assert offline.retrieval.rerank.enabled is False

    api = load_settings(profile="api")
    assert api.llm.preset == "openai"
    assert api.eval.provider == "api"

    local = load_settings(profile="local")
    assert local.llm.preset == "ollama"
    assert local.retrieval.rerank.enabled is True
    assert local.embedding.backend == "bge_m3"


def test_profile_files_load() -> None:
    for name, expected_preset in (
        ("offline", "fake"),
        ("api", "openai"),
        ("local", "ollama"),
    ):
        path = REPO_ROOT / "configs" / "profiles" / f"{name}.yaml"
        settings = load_settings(path)
        assert settings.profile == name
        assert settings.llm.preset == expected_preset


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_RETRIEVAL__RRF_K", "88")
    assert Settings().retrieval.rrf_k == 88
    monkeypatch.setenv("ANSWERDESK_LLM__PRESET", "ollama")
    assert Settings().llm.preset == "ollama"


def test_explicit_overrides_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANSWERDESK_RETRIEVAL__RRF_K", "88")
    settings = load_settings(overrides={"retrieval": {"rrf_k": 99}})
    assert settings.retrieval.rrf_k == 99


def test_local_server_needs_no_api_key() -> None:
    """NFR-PORT-01: a local vLLM/llama.cpp/Ollama server is reachable with zero keys."""
    settings = load_settings(profile="local", overrides={"llm": {"preset": "vllm"}})
    provider = settings.resolved_provider()
    assert provider.requires_api_key is False
    assert provider.api_key is None
    assert provider.base_url.startswith("http://127.0.0.1")


def test_api_profile_without_key_gives_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FR-OPS-05: the error names the variable the operator has to set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANSWERDESK_LLM__API_KEY", raising=False)
    settings = load_settings(profile="api")
    with pytest.raises(ConfigError) as excinfo:
        settings.resolved_provider()
    message = excinfo.value.message
    assert "OPENAI_API_KEY" in message
    assert "offline" in message  # tells the operator how to proceed without a key


def test_api_key_from_environment_is_used_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-test-0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    settings = load_settings(profile="api")
    provider = settings.resolved_provider()
    assert provider.api_key is not None
    assert provider.api_key.get_secret_value() == secret
    assert provider.redacted()["api_key"] == "***"

    dumped = json.dumps(settings.redacted())
    assert secret not in dumped
    # The key lives in the environment, not in the settings object: the dump records
    # "no key configured here" rather than echoing what it found.
    assert settings.redacted()["llm"]["api_key"] is None


def test_offline_profile_rejects_a_remote_preset() -> None:
    with pytest.raises(ConfigError, match="offline"):
        load_settings(profile="offline", overrides={"llm": {"preset": "deepseek"}})


def test_profile_preset_coherence_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="real provider preset"):
        Settings(profile="api").resolved_provider()
    with pytest.raises(ConfigError, match="offline"):
        Settings(profile="offline", llm={"preset": "openai"}).resolved_provider()


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(Exception) as excinfo:
        Settings(llm={"preset": "not-a-provider"})
    assert "unknown provider preset" in str(excinfo.value)


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("profile: offline\nsurprise: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level keys"):
        load_settings(path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad2.yaml"
    path.write_text("retrieval:\n  surprise: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_settings(path)


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(retrieval={"rrf_k": 0})
    with pytest.raises(ValidationError):
        Settings(generate={"context_k_min": 12, "context_k_max": 8})
    with pytest.raises(ValidationError):
        Settings(policy={"alpha": {IntentBucket.INFORMATIONAL: 1.5}})


def test_resolve_path_uses_the_repository_root() -> None:
    settings = Settings()
    resolved = settings.resolve_path("data/corpus")
    assert resolved.is_absolute()
    assert resolved.parts[-2:] == ("data", "corpus")


def test_apply_profile_is_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Settings()
    applied = apply_profile(original, "local")
    assert applied.llm.preset == "ollama"
    assert original.llm.preset == "fake"
    assert original.profile == "offline"
