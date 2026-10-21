"""Errors for the semantic-cache layer.

A cache failure is never fatal to a turn: a guard rejection demotes a hit to a miss
(requirements §9.2). The exceptions exist for configuration and artifact problems, which
must fail loudly at startup rather than degrade silently.

    # AT-CAC-05
"""

from __future__ import annotations

from answerdesk.errors import AnswerdeskError


class CacheError(AnswerdeskError):
    """Base class for semantic-cache failures."""

    code = "cache_error"


class CacheConfigError(CacheError):
    """Invalid cache configuration (thresholds, TTL classes, guard settings)."""

    code = "cache_config_error"


class CacheArtifactError(CacheError):
    """A calibrated threshold table is missing, corrupt, or bound to another corpus."""

    code = "cache_artifact_error"


class CacheMissError(CacheError):
    """A lookup that the caller required to hit (used by tests and the replay harness)."""

    code = "cache_miss"


__all__ = [
    "CacheArtifactError",
    "CacheConfigError",
    "CacheError",
    "CacheMissError",
]
