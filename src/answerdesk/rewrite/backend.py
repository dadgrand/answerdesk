"""The backend factory: config in, a working rewriter out, never an exception.

Rules, in order:

1. ``enabled: false`` (the default) → :class:`IdentityRewriter`; nothing is imported, nothing
   is loaded, latency is zero.
2. ``backend: deterministic`` → :class:`DeterministicRewriter` (zero dependencies).
3. ``backend: local_lora`` → :class:`LocalLoRARewriter` **if** the `train` extra and the
   adapter directory are present; otherwise a warning is logged and identity is used, with the
   reason surfaced in :func:`rewrite_status` so the console can show it.
4. Every backend is wrapped in :class:`TimedRewriter`, so a raising rewriter, a broken adapter
   or a blown latency budget degrades to the original query instead of failing a request.

    # AT-GEN-13  # AT-GEN-16
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from answerdesk.logging import get_logger
from answerdesk.rewrite.base import IdentityRewriter, QueryRewriter, TimedRewriter
from answerdesk.rewrite.config import RewriteConfig, load_rewrite_config
from answerdesk.rewrite.deterministic import DeterministicRewriter
from answerdesk.rewrite.local_lora import LocalLoRARewriter

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RewriterStatus:
    """What the service/console needs to say about the rewriter (no secrets, no imports)."""

    enabled: bool
    configured_backend: str
    active_backend: str
    available: bool
    fallback: bool
    reason: str = ""
    adapter_path: str = ""
    base_model: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "configured_backend": self.configured_backend,
            "active_backend": self.active_backend,
            "available": self.available,
            "fallback": self.fallback,
            "reason": self.reason,
            "adapter_path": self.adapter_path,
            "base_model": self.base_model,
        }


def build_rewriter(
    config: RewriteConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> QueryRewriter:
    """Build the configured rewriter, degrading to identity when anything is missing."""
    resolved = config if config is not None else load_rewrite_config(config_path)
    if not resolved.enabled or resolved.backend == "identity":
        return IdentityRewriter()

    if resolved.backend == "deterministic":
        options = resolved.deterministic
        inner: QueryRewriter = DeterministicRewriter(
            transliterate=options.transliterate,
            drop_fillers=options.drop_fillers,
            repair=options.repair,
        )
        return TimedRewriter(inner)

    local = LocalLoRARewriter(
        adapter_path=resolved.local_lora.adapter_path,
        base_model=resolved.local_lora.base_model,
        device=resolved.local_lora.device,
        max_new_tokens=resolved.local_lora.max_new_tokens,
        timeout_ms=resolved.local_lora.timeout_ms,
    )
    missing = local.missing
    if missing:
        _LOGGER.warning(
            "local query rewriter unavailable; falling back to identity",
            extra={"missing": missing, "adapter_path": str(local.adapter_path)},
        )
        return IdentityRewriter()
    return TimedRewriter(local, timeout_ms=resolved.local_lora.timeout_ms)


def rewrite_status(
    config: RewriteConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> RewriterStatus:
    """Describe the rewriter without loading any weights (safe for /healthz and the console)."""
    resolved = config if config is not None else load_rewrite_config(config_path)
    if not resolved.enabled or resolved.backend == "identity":
        return RewriterStatus(
            enabled=resolved.enabled,
            configured_backend=resolved.backend,
            active_backend="identity",
            available=True,
            fallback=False,
            reason="disabled by configuration" if not resolved.enabled else "identity backend",
        )
    if resolved.backend == "deterministic":
        return RewriterStatus(
            enabled=True,
            configured_backend=resolved.backend,
            active_backend="deterministic",
            available=True,
            fallback=False,
            reason="dependency-free normaliser",
        )
    local = LocalLoRARewriter(
        adapter_path=resolved.local_lora.adapter_path,
        base_model=resolved.local_lora.base_model,
        device=resolved.local_lora.device,
        max_new_tokens=resolved.local_lora.max_new_tokens,
        timeout_ms=resolved.local_lora.timeout_ms,
    )
    missing = local.missing
    return RewriterStatus(
        enabled=True,
        configured_backend=resolved.backend,
        active_backend="identity" if missing else "local_lora",
        available=not missing,
        fallback=bool(missing),
        reason=(
            f"missing: {', '.join(missing)}; install with `uv sync --extra train`"
            if missing
            else "adapter present"
        ),
        adapter_path=str(local.adapter_path),
        base_model=local.base_model,
    )


__all__ = ["RewriterStatus", "build_rewriter", "rewrite_status"]
