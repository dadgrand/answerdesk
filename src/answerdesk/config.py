"""Configuration: provider presets, profiles, YAML + environment loading.

Design (ADR-008):

* one :class:`Settings` object, pydantic-settings, environment prefix ``ANSWERDESK_``
  and nested delimiter ``__`` (e.g. ``ANSWERDESK_LLM__PRESET=ollama``);
* YAML files under ``configs/`` provide defaults, the environment overrides them at
  leaf granularity, explicit constructor arguments override everything;
* three profiles — ``offline`` (deterministic doubles, no network), ``api``
  (OpenAI-compatible API), ``local`` (vLLM / llama.cpp / Ollama);
* provider selection never touches code: presets differ only in defaults
  (NFR-PORT-01, FR-OPS-03). The API key is a ``SecretStr`` and is redacted in every
  dump, log line and error message (INV-05).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from answerdesk.contracts import IntentBucket, ProviderConfig
from answerdesk.errors import ConfigError

Profile = Literal["offline", "api", "local"]

# src/answerdesk/config.py -> parents[0]=answerdesk, [1]=src, [2]=repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ProviderPreset(BaseModel):
    """Defaults for one OpenAI-compatible provider. Keys come from the environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    base_url: str
    model: str
    api_key_env: str | None
    requires_api_key: bool


#: Every preset speaks the same chat-completions wire format (ADR-003).
PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        requires_api_key=True,
    ),
    "openai_compatible": ProviderPreset(
        name="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
        model="default",
        api_key_env="OPENAI_COMPATIBLE_API_KEY",
        requires_api_key=False,
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        requires_api_key=True,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        api_key_env="OPENROUTER_API_KEY",
        requires_api_key=True,
    ),
    "vllm": ProviderPreset(
        name="vllm",
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen/Qwen3-8B",
        api_key_env="VLLM_API_KEY",
        requires_api_key=False,
    ),
    "llama_cpp": ProviderPreset(
        name="llama_cpp",
        base_url="http://127.0.0.1:8080/v1",
        model="local-model",
        api_key_env=None,
        requires_api_key=False,
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        api_key_env=None,
        requires_api_key=False,
    ),
    "fake": ProviderPreset(
        name="fake",
        base_url="memory://fake",
        model="fake-model",
        api_key_env=None,
        requires_api_key=False,
    ),
}

#: Presets the requirements name explicitly (FR-OPS-03); ``fake`` is CI-only.
REQUIRED_PRESETS: tuple[str, ...] = (
    "openai_compatible",
    "deepseek",
    "openrouter",
    "vllm",
    "llama_cpp",
    "ollama",
)


class Section(BaseModel):
    """Base for config sections: strict about unknown keys, relaxed about extras."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AppSettings(Section):
    name: str = "answerdesk"
    data_dir: str = "data"
    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    reports_dir: str = "reports"
    log_dir: str = "logs"


class LLMSettings(Section):
    preset: str = "fake"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    json_mode: bool = True
    fallback_presets: list[str] = Field(default_factory=list)
    pricing_table: str = "configs/pricing.yaml"

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, value: str) -> str:
        if value not in PROVIDER_PRESETS:
            known = ", ".join(sorted(PROVIDER_PRESETS))
            raise ValueError(f"unknown provider preset {value!r}; known presets: {known}")
        return value

    @field_validator("fallback_presets")
    @classmethod
    def _known_fallbacks(cls, values: list[str]) -> list[str]:
        unknown = [v for v in values if v not in PROVIDER_PRESETS]
        if unknown:
            raise ValueError(f"unknown fallback presets: {unknown}")
        return values


class EmbeddingSettings(Section):
    backend: Literal["hashing", "bge_m3", "qwen3", "api"] = "hashing"
    model: str = "BAAI/bge-m3"
    #: Width of the *hashing* embedder. A real model (local or hosted) declares its own.
    dimension: int = Field(default=64, gt=0)
    batch_size: int = Field(default=32, gt=0)
    device: Literal["cpu", "cuda", "auto"] = "auto"
    # -- hosted API backend (backend="api", NFR-PORT-01 for retrieval) --------------
    #: OpenAI-compatible base URL, e.g. ``https://api.neuraldeep.ru/v1``.
    api_base_url: str | None = None
    #: Model name sent in the request body, e.g. ``bge-m3``.
    api_model: str | None = None
    #: Declared vector width. ``None`` means "discover from the first response and pin it".
    api_dimension: int | None = Field(default=None, gt=0)
    api_key: SecretStr | None = None
    #: Environment variable that may hold the key instead of ``api_key``.
    api_key_env: str = "OPENAI_COMPATIBLE_API_KEY"
    api_timeout_s: float = Field(default=30.0, gt=0)
    api_max_retries: int = Field(default=3, ge=0, le=10)


class RerankSettings(Section):
    enabled: bool = False
    #: ``cross_encoder`` = local sentence-transformers model (``local`` extra),
    #: ``lexical`` = the deterministic offline stand-in, ``api`` = hosted ``/rerank``.
    backend: Literal["cross_encoder", "lexical", "api"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_in: int = Field(default=50, gt=0)
    top_out: int = Field(default=12, gt=0)
    batch_size: int = Field(default=16, gt=0)
    # -- hosted API backend (backend="api") ----------------------------------------
    api_base_url: str | None = None
    api_model: str | None = None
    api_key: SecretStr | None = None
    api_key_env: str = "OPENAI_COMPATIBLE_API_KEY"
    api_timeout_s: float = Field(default=30.0, gt=0)
    api_max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("top_out")
    @classmethod
    def _top_out_le_top_in(cls, value: int, info: Any) -> int:
        top_in = info.data.get("top_in")
        if isinstance(top_in, int) and value > top_in:
            raise ValueError("rerank.top_out must be <= rerank.top_in")
        return value


class RetrievalSettings(Section):
    fuser: Literal["rrf", "cc"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    cc_alpha: float | None = Field(default=None, gt=0.0, lt=1.0)
    top_k: int = Field(default=50, gt=0)
    gate_k: int = Field(default=20, gt=0)
    dedupe_cosine: float = Field(default=0.95, gt=0.0, le=1.0)
    mmr_lambda: float | None = Field(default=None, gt=0.0, le=1.0)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


class GenerateSettings(Section):
    context_k_min: int = Field(default=8, gt=0)
    context_k_max: int = Field(default=12, gt=0)
    token_budget: int = Field(default=8000, gt=0)
    repair_passes: int = Field(default=1, ge=0, le=1)
    verifier: Literal["lexical", "judge", "nli"] = "lexical"

    @field_validator("context_k_max")
    @classmethod
    def _k_range(cls, value: int, info: Any) -> int:
        k_min = info.data.get("context_k_min")
        if isinstance(k_min, int) and value < k_min:
            raise ValueError("generate.context_k_max must be >= generate.context_k_min")
        return value


class CostSettings(Section):
    c_wrong: float = Field(default=1.0, gt=0.0)
    c_abstain: float = Field(default=0.10, ge=0.0)
    c_delay: float = Field(default=0.05, ge=0.0)

    @property
    def answer_threshold(self) -> float:
        """``1 - c_delay / c_wrong`` — the lower bound of the answer band (requirements §8.3)."""
        return 1.0 - (self.c_delay / self.c_wrong)

    @property
    def escalate_threshold(self) -> float:
        """``1 - c_abstain / c_wrong`` — below this the turn must be escalated."""
        return 1.0 - (self.c_abstain / self.c_wrong)


class PolicySettings(Section):
    alpha: dict[IntentBucket, float] = Field(
        default_factory=lambda: {
            IntentBucket.INFORMATIONAL: 0.05,
            IntentBucket.TRANSACTIONAL: 0.02,
            IntentBucket.ACCOUNT_LEGAL: 0.01,
        }
    )
    costs: CostSettings = Field(default_factory=CostSettings)
    calibration_artifact: str | None = None

    @field_validator("alpha")
    @classmethod
    def _alpha_range(cls, value: dict[IntentBucket, float]) -> dict[IntentBucket, float]:
        for bucket, alpha in value.items():
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha for {bucket} must be in (0, 1), got {alpha}")
        return value

    def effective_alpha(self, bucket: IntentBucket) -> float:
        """``min(alpha_bucket, c_abstain / c_wrong)`` — the stricter wins (requirements §8.2)."""
        return min(self.alpha[bucket], self.costs.c_abstain / self.costs.c_wrong)


class CacheSettings(Section):
    enabled: bool = True
    epsilon: float = Field(default=0.01, gt=0.0, lt=1.0)
    similarity_threshold: float = Field(default=0.93, gt=0.0, le=1.0)
    ttl_class: dict[str, int] = Field(
        default_factory=lambda: {"policy": 3600, "howto": 604800, "escalation": 600}
    )
    high_risk_intents: list[str] = Field(
        default_factory=lambda: ["account", "refund", "legal", "payment"]
    )


class EvalSettings(Section):
    """Evaluation knobs, including the *judge* provider (requirements §10.5).

    §10.5 forbids a model family from grading its own output in a reported metric, so the
    judge is a first-class, separately configured provider. ``judge_preset = None`` keeps
    the legacy behaviour (judge runs on the generator provider) and the harness then labels
    every judged number as diagnostic unless the two families differ.
    """

    seed: int = 1337
    split: str = "evals/splits/v1.json"
    judge_model: str = "fake-judge"
    provider: Literal["fake", "api"] = "fake"
    bootstrap_samples: int = Field(default=1000, gt=0)
    # --- judge provider override (cross-family judging) ---------------------------
    judge_preset: str | None = None
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None
    judge_api_key_env: str | None = None
    judge_timeout_s: float = Field(default=180.0, gt=0)
    judge_max_tokens: int = Field(default=512, gt=0)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    #: Declared family label ("qwen", "deepseek", …); derived from the model name if null.
    judge_family: str | None = None
    #: Extra seconds between live provider calls (free tiers rate-limit bursts).
    min_call_interval_s: float = Field(default=0.0, ge=0.0)
    # --- secondary generator arm (the comparison model of the benchmark) ----------
    secondary_preset: str | None = "deepseek"
    secondary_model: str | None = None
    secondary_base_url: str | None = None
    secondary_api_key: SecretStr | None = None
    secondary_api_key_env: str | None = "DEEPSEEK_API_KEY"
    secondary_timeout_s: float = Field(default=180.0, gt=0)
    secondary_max_tokens: int = Field(default=1024, gt=0)


class LoggingSettings(Section):
    level: str = "INFO"
    json_output: bool = True


def _env_leaf_overrides(instance: BaseModel) -> dict[str, Any]:
    """Values the environment/.env actually set, as a nested dict of leaves only."""
    out: dict[str, Any] = {}
    for name in type(instance).model_fields:
        if name not in instance.model_fields_set:
            continue
        value = getattr(instance, name)
        if isinstance(value, BaseModel):
            nested = value.model_dump(exclude_unset=True)
            if nested:
                out[name] = nested
        else:
            out[name] = value
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    unknown = sorted(set(raw) - set(Settings.model_fields))
    if unknown:
        known = ", ".join(sorted(Settings.model_fields))
        raise ConfigError(
            f"config file {path} has unknown top-level keys {unknown}; known keys: {known}"
        )
    return raw


def default_config_path() -> Path | None:
    """``ANSWERDESK_CONFIG`` wins; otherwise the repository's ``configs/default.yaml``."""
    env = os.environ.get("ANSWERDESK_CONFIG")
    if env:
        return Path(env)
    candidate = _REPO_ROOT / "configs" / "default.yaml"
    return candidate if candidate.is_file() else None


class Settings(BaseSettings):
    """Root settings object. Construct directly or via :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="ANSWERDESK_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    profile: Profile = "offline"
    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generate: GenerateSettings = Field(default_factory=GenerateSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # -- path helpers -------------------------------------------------------------
    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a config path against the repository root (or CWD when installed)."""
        path = Path(value)
        if path.is_absolute():
            return path
        root = _REPO_ROOT if (_REPO_ROOT / "pyproject.toml").is_file() else Path.cwd()
        return root / path

    # -- provider resolution ------------------------------------------------------
    def resolved_provider(self) -> ProviderConfig:
        """Turn the LLM settings into a concrete :class:`ProviderConfig`.

        Raises :class:`ConfigError` with an actionable message when a remote preset has
        no key — the failure the operator will actually hit (FR-OPS-05).
        """
        preset = PROVIDER_PRESETS[self.llm.preset]
        if self.profile == "offline" and preset.name != "fake":
            raise ConfigError(
                f"profile 'offline' must use the deterministic 'fake' provider, got "
                f"{preset.name!r}. Use --profile api/local for a real model."
            )
        if self.profile in ("api", "local") and preset.name == "fake":
            raise ConfigError(
                f"profile {self.profile!r} needs a real provider preset, got 'fake'. "
                "Pick a preset (e.g. ANSWERDESK_LLM__PRESET=openai) or use --profile offline."
            )
        api_key = self.llm.api_key
        if api_key is None and preset.api_key_env:
            env_value = os.environ.get(preset.api_key_env)
            if env_value:
                api_key = SecretStr(env_value)
        if preset.requires_api_key and api_key is None:
            raise ConfigError(
                f"provider preset '{preset.name}' requires an API key. Set "
                f"{preset.api_key_env} in the environment, or "
                f"ANSWERDESK_LLM__API_KEY, or switch to profile 'offline' / 'local'."
            )
        return ProviderConfig(
            preset=preset.name,
            base_url=self.llm.base_url or preset.base_url,
            model=self.llm.model or preset.model,
            api_key=api_key,
            timeout_s=self.llm.timeout_s,
            max_retries=self.llm.max_retries,
            temperature=self.llm.temperature,
            max_tokens=self.llm.max_tokens,
            json_mode=self.llm.json_mode,
            requires_api_key=preset.requires_api_key,
        )

    # -- redacted dump ------------------------------------------------------------
    def redacted(self) -> dict[str, Any]:
        """Full settings as plain data with every secret replaced (safe to print)."""
        data = self.model_dump(mode="json")
        llm = data.get("llm", {})
        llm["api_key"] = "***" if self.llm.api_key is not None else None
        return data


def apply_profile(settings: Settings, profile: Profile | None = None) -> Settings:
    """Apply profile defaults without clobbering an explicitly chosen preset.

    * ``offline`` — deterministic doubles, no network, rerank off (CI default)
    * ``api``     — API provider (``openai`` unless another preset is configured)
    * ``local``   — local OpenAI-compatible server (``ollama`` unless configured),
                    local embedder/reranker enabled
    """
    resolved = settings.model_copy(deep=True)
    chosen: Profile = profile or resolved.profile
    resolved.profile = chosen

    if chosen == "offline":
        if resolved.llm.preset != "fake":
            # An explicit non-fake preset in an offline profile is a configuration
            # contradiction: offline means no network.
            raise ConfigError(
                f"profile 'offline' cannot use provider preset {resolved.llm.preset!r}; "
                "use profile 'api' or 'local', or set ANSWERDESK_LLM__PRESET=fake"
            )
        resolved.retrieval.rerank.enabled = False
        resolved.embedding.backend = "hashing"
        resolved.eval.provider = "fake"
    elif chosen == "api":
        if resolved.llm.preset == "fake":
            resolved.llm.preset = "openai"
        resolved.eval.provider = "api"
    elif chosen == "local":
        if resolved.llm.preset == "fake":
            resolved.llm.preset = "ollama"
        resolved.retrieval.rerank.enabled = True
        if resolved.embedding.backend == "hashing":
            resolved.embedding.backend = "bge_m3"
    return resolved


def load_settings(
    config_path: str | Path | None = None,
    profile: Profile | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings with precedence: explicit ``overrides`` > environment > YAML > defaults.

    ``config_path`` defaults to :func:`default_config_path`. Both API keys and
    ``ANSWERDESK_*`` variables work without editing YAML (FR-OPS-03).
    """
    yaml_data: dict[str, Any] = {}
    path = Path(config_path) if config_path is not None else default_config_path()
    if path is not None:
        if not path.is_file() and config_path is not None:
            raise ConfigError(f"config file not found: {path}")
        if path.is_file():
            yaml_data = _read_config_file(path)

    try:
        env_layer = _env_leaf_overrides(Settings())
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in the environment: {exc}") from exc
    merged = _deep_merge(yaml_data, env_layer)
    if overrides:
        merged = _deep_merge(merged, overrides)
    if profile is not None:
        merged["profile"] = profile

    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
    return apply_profile(settings)


__all__ = [
    "PROVIDER_PRESETS",
    "REQUIRED_PRESETS",
    "AppSettings",
    "CacheSettings",
    "CostSettings",
    "EmbeddingSettings",
    "EvalSettings",
    "GenerateSettings",
    "LLMSettings",
    "LoggingSettings",
    "PolicySettings",
    "Profile",
    "ProviderPreset",
    "RerankSettings",
    "RetrievalSettings",
    "Settings",
    "apply_profile",
    "default_config_path",
    "load_settings",
]
