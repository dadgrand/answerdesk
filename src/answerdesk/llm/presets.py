"""Provider presets: the only place that knows provider *names* exist.

A preset is data — a default ``base_url``, a default model and the name of the
environment variable that holds the key. Nothing here (and nothing in
``providers.py``) branches on a preset name; the single transport in
:mod:`answerdesk.llm.providers` consumes the resulting
:class:`answerdesk.contracts.ProviderConfig` (requirements NFR-PORT-01, FR-OPS-03;
ADR-003).

``PRESETS`` is the public surface named by the requirements (CMD-I prints its keys).
``llama.cpp`` and ``openai-compatible`` are accepted as *input aliases* — operator
muscle memory differs from Python identifier rules — and are normalised to the
canonical keys ``llama_cpp`` / ``openai_compatible``.

    # AT-OPS-03  # AT-PORT-01
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import SecretStr

from answerdesk.config import PROVIDER_PRESETS, ProviderPreset
from answerdesk.contracts import ProviderConfig
from answerdesk.errors import ConfigError

#: Canonical preset table (a shallow copy: values are frozen models).
PRESETS: dict[str, ProviderPreset] = dict(PROVIDER_PRESETS)

#: Input aliases, normalised to a canonical key before any lookup.
ALIASES: dict[str, str] = {
    "llama.cpp": "llama_cpp",
    "llamacpp": "llama_cpp",
    "llama-cpp": "llama_cpp",
    "openai-compatible": "openai_compatible",
    "openai_compatible_server": "openai_compatible",
    "oai_compatible": "openai_compatible",
}

#: Presets that speak a remote API and therefore need a key (FR-OPS-03).
REMOTE_PRESETS: tuple[str, ...] = ("openai", "deepseek", "openrouter")

#: Presets that are local server processes: no key, localhost by default.
LOCAL_PRESETS: tuple[str, ...] = ("openai_compatible", "vllm", "llama_cpp", "ollama")

#: The CI double. Never reaches the network; never needs a key (ADR-004).
FAKE_PRESET = "fake"


def normalise_preset_name(name: str) -> str:
    """Map an input alias (``llama.cpp``) onto a canonical preset key (``llama_cpp``)."""
    cleaned = name.strip().lower()
    return ALIASES.get(cleaned, cleaned)


def preset_names() -> list[str]:
    """Canonical preset keys, sorted — the exact output of ``CMD-I``."""
    return sorted(PRESETS)


def resolve_preset(name: str) -> ProviderPreset:
    """Look up a preset, accepting aliases and raising an actionable error otherwise."""
    key = normalise_preset_name(name)
    preset = PRESETS.get(key)
    if preset is None:
        known = ", ".join(preset_names())
        raise ConfigError(
            f"unknown provider preset {name!r}; known presets: {known} "
            f"(aliases: {', '.join(sorted(ALIASES))})"
        )
    return preset


def resolve_api_key(
    preset: ProviderPreset,
    *,
    env: Mapping[str, str] | None = None,
    explicit: SecretStr | None = None,
) -> SecretStr | None:
    """Key precedence: explicit value > environment > none.

    Local presets return ``None`` and never look at the environment, which is what
    makes "point it at a local model with zero keys" true (NFR-PORT-01).
    """
    if explicit is not None:
        return explicit
    if not preset.api_key_env:
        return None
    value = (env if env is not None else os.environ).get(preset.api_key_env)
    return SecretStr(value) if value else None


def provider_config(
    name: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: SecretStr | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = 60.0,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    headers: Mapping[str, str] | None = None,
) -> ProviderConfig:
    """Build a :class:`ProviderConfig` for a preset, filling defaults from the table.

    This is the one function the CLI, the fallback chain and the tests share, so a
    preset can never grow a second meaning (CL-PORT-01).
    """
    preset = resolve_preset(name)
    resolved_key = resolve_api_key(preset, env=env, explicit=api_key)
    if preset.requires_api_key and resolved_key is None:
        raise ConfigError(
            f"provider preset {preset.name!r} requires an API key: set "
            f"{preset.api_key_env} in the environment (or ANSWERDESK_LLM__API_KEY), "
            f"or switch to a local preset ({', '.join(LOCAL_PRESETS)}) / profile 'offline'."
        )
    return ProviderConfig(
        preset=preset.name,
        base_url=base_url or preset.base_url,
        model=model or preset.model,
        api_key=resolved_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        headers=dict(headers or {}),
        requires_api_key=preset.requires_api_key,
    )


def is_local(preset_name: str) -> bool:
    """True for the presets that are expected to answer on localhost (docs/tests)."""
    return normalise_preset_name(preset_name) in LOCAL_PRESETS


__all__ = [
    "ALIASES",
    "FAKE_PRESET",
    "LOCAL_PRESETS",
    "PRESETS",
    "REMOTE_PRESETS",
    "is_local",
    "normalise_preset_name",
    "preset_names",
    "provider_config",
    "resolve_api_key",
    "resolve_preset",
]
