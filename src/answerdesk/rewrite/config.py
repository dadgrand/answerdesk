"""Config for the optional query-rewriter backend (off by default).

`configs/rewrite/default.yaml` is the only place a deployment turns the rewriter on. The
loader validates the knob set strictly (`extra="forbid"`), so a typo is a startup error rather
than a silently ignored setting, and a missing file falls back to the built-in defaults —
which are *disabled*.

    # AT-GEN-13  # AT-GEN-16
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from answerdesk.rewrite.errors import RewriteConfigError

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default config file relative to the repository root (a code constant, no new gated key).
DEFAULT_CONFIG_PATH = Path("configs") / "rewrite" / "default.yaml"

BackendName = Literal["identity", "deterministic", "local_lora"]


class DeterministicOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transliterate: bool = True
    drop_fillers: bool = True
    repair: bool = True


class LocalLoRAOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str = "data/training/adapter"
    device: str = "auto"
    max_new_tokens: int = Field(default=64, ge=8, le=256)
    timeout_ms: float = Field(default=2000.0, gt=0.0)


class RewriteConfig(BaseModel):
    """The whole rewriter configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    backend: BackendName = "identity"
    deterministic: DeterministicOptions = Field(default_factory=DeterministicOptions)
    local_lora: LocalLoRAOptions = Field(default_factory=LocalLoRAOptions)

    def is_local_lora(self) -> bool:
        return self.backend == "local_lora"


def default_config_path() -> Path:
    return _REPO_ROOT / DEFAULT_CONFIG_PATH


def load_rewrite_config(path: str | Path | None = None) -> RewriteConfig:
    """Load and validate the rewriter config; a missing file means "disabled"."""
    file_path = Path(path) if path is not None else default_config_path()
    if not file_path.is_file():
        return RewriteConfig()
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RewriteConfigError(f"cannot read rewrite config {file_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RewriteConfigError(f"rewrite config {file_path} is not valid YAML: {exc}") from exc
    if raw is None:
        return RewriteConfig()
    if not isinstance(raw, dict):
        raise RewriteConfigError(f"rewrite config {file_path} must be a mapping")
    payload = raw.get("rewrite", raw)
    if not isinstance(payload, dict):
        raise RewriteConfigError(f"rewrite config {file_path}: 'rewrite' must be a mapping")
    try:
        return RewriteConfig.model_validate(payload)
    except Exception as exc:
        raise RewriteConfigError(f"rewrite config {file_path} is invalid: {exc}") from exc


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "BackendName",
    "DeterministicOptions",
    "LocalLoRAOptions",
    "RewriteConfig",
    "default_config_path",
    "load_rewrite_config",
]
