"""Typed errors for the grounded-generation layer.

Kept separate from :mod:`answerdesk.errors` (t3's frozen taxonomy) because these are
module-internal failure modes that the pipeline *handles*; only the ones that must
reach the API envelope are re-raised as :class:`answerdesk.errors.ProviderError`.

    # AT-GEN-04  # AT-GEN-07
"""

from __future__ import annotations

from answerdesk.errors import AnswerdeskError


class GenerationError(AnswerdeskError):
    """Base class for generation failures."""

    code = "generation_error"


class PromptRegistryError(GenerationError):
    """The prompt registry is inconsistent: missing prompt, bad digest, unknown variable."""

    code = "prompt_registry_error"


class PromptRenderError(PromptRegistryError):
    """A template referenced a variable the caller did not supply."""

    code = "prompt_render_error"


class PromptVersionError(PromptRegistryError):
    """A prompt body changed without a version bump — refused (FR-GEN-04)."""

    code = "prompt_version_error"


class MalformedModelOutput(GenerationError):
    """The model did not return a draft that matches the answer schema (NFR-OPS-01)."""

    code = "malformed_model_output"
    retryable = True


class EvidenceError(GenerationError):
    """The packed context cannot support an answer (empty pack, no usable passage)."""

    code = "evidence_error"


__all__ = [
    "EvidenceError",
    "GenerationError",
    "MalformedModelOutput",
    "PromptRegistryError",
    "PromptRenderError",
    "PromptVersionError",
]
