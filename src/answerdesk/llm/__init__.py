"""Universal provider layer: one OpenAI-compatible client, one cost ledger.

Owning task: **t6**. Public surface for the rest of the stack (t7 evaluation, t8 policy,
t9 service):

======================  ==========================================================
:class:`HttpChatProvider`  the only model HTTP client in the repository (INV-07)
:class:`ProviderChain`     ordered fallback across providers, per-provider breakers
``PRESETS``                the preset table (aliases normalised, keys never in code)
:class:`CostLedger`        tokens/USD/latency for every call (NFR-COST-01)
:class:`RecordedProvider`  replay a captured transcript offline (AT-GEN-09)
``build_provider``         settings → provider, with no provider-specific branches
======================  ==========================================================

Requirements: NFR-PORT-01, NFR-COST-01, NFR-COST-02, FR-OPS-03, FR-GEN-09.

Every type that crosses this module's boundary comes from :mod:`answerdesk.contracts`.
"""

from __future__ import annotations

from answerdesk.llm.circuit import CircuitBreaker, CircuitState
from answerdesk.llm.cost import (
    CostLedger,
    CostRecord,
    CostReport,
    CostSummary,
    ModelCost,
    PriceQuote,
    PricingTable,
    default_pricing_path,
    estimate_tokens,
    percentile,
)
from answerdesk.llm.errors import (
    AllProvidersFailedError,
    CircuitOpenError,
    LLMError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from answerdesk.llm.factory import build_provider, fallback_configs, provider_sequence
from answerdesk.llm.presets import (
    ALIASES,
    LOCAL_PRESETS,
    PRESETS,
    REMOTE_PRESETS,
    is_local,
    normalise_preset_name,
    preset_names,
    provider_config,
    resolve_preset,
)
from answerdesk.llm.providers import CHAT_COMPLETIONS_PATH, HttpChatProvider, ProviderChain
from answerdesk.llm.recorded import (
    RecordedFixture,
    RecordedFixtureMiss,
    RecordedProvider,
    RecordingProvider,
    request_digest,
)
from answerdesk.llm.retry import RetryPolicy, Runner, parse_retry_after

__all__ = [
    "ALIASES",
    "CHAT_COMPLETIONS_PATH",
    "LOCAL_PRESETS",
    "PRESETS",
    "REMOTE_PRESETS",
    "AllProvidersFailedError",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CostLedger",
    "CostRecord",
    "CostReport",
    "CostSummary",
    "HttpChatProvider",
    "LLMError",
    "ModelCost",
    "PriceQuote",
    "PricingTable",
    "ProviderChain",
    "ProviderConnectionError",
    "ProviderHTTPError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "RecordedFixture",
    "RecordedFixtureMiss",
    "RecordedProvider",
    "RecordingProvider",
    "RetryPolicy",
    "Runner",
    "build_provider",
    "default_pricing_path",
    "estimate_tokens",
    "fallback_configs",
    "is_local",
    "normalise_preset_name",
    "parse_retry_after",
    "percentile",
    "preset_names",
    "provider_config",
    "provider_sequence",
    "request_digest",
    "resolve_preset",
]
