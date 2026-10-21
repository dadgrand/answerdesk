"""Provider construction: settings in, one :class:`ChatProvider` out.

The factory is the only place that turns configuration into an object, and it is
deliberately boring: pick the primary preset, append the configured fallbacks, wrap
them in a :class:`~answerdesk.llm.providers.ProviderChain` when there is more than one.
Every preset goes through the *same* transport; the sole exception is the CI double
``fake``, which exists so that offline CI can run the pipeline without a socket
(ADR-004). That exception is a test double, not a provider-specific code path
(CL-PORT-01).

    # AT-OPS-03  # AT-PORT-01
"""

from __future__ import annotations

from collections.abc import Sequence

from answerdesk.config import Settings
from answerdesk.contracts import ChatProvider, ProviderConfig
from answerdesk.errors import ConfigError
from answerdesk.llm.circuit import CircuitBreaker
from answerdesk.llm.cost import CostLedger
from answerdesk.llm.presets import FAKE_PRESET, provider_config
from answerdesk.llm.providers import HttpChatProvider, ProviderChain
from answerdesk.llm.retry import RetryPolicy, Runner
from answerdesk.logging import get_logger

_LOGGER = get_logger(__name__)


def _fake_provider(config: ProviderConfig) -> ChatProvider:
    """The deterministic CI double, imported lazily to keep the production import graph clean."""
    from answerdesk.testing.doubles import FakeProvider

    return FakeProvider(config)


def _from_config(
    config: ProviderConfig,
    *,
    ledger: CostLedger | None,
    stage: str,
    prompt_version: str,
    runner: Runner | None,
    transport: object | None,
) -> ChatProvider:
    if config.preset == FAKE_PRESET:
        return _fake_provider(config)
    return HttpChatProvider(
        config,
        ledger=ledger,
        circuit=CircuitBreaker(name=f"http:{config.preset}"),
        retry_policy=RetryPolicy(max_retries=config.max_retries),
        runner=runner,
        transport=transport,  # type: ignore[arg-type]
        stage=stage,
        prompt_version=prompt_version,
    )


def fallback_configs(settings: Settings) -> list[ProviderConfig]:
    """Resolve the configured fallback presets, skipping any that cannot be keyed.

    A fallback without a key is not an error: the operator asked for "try deepseek if
    openai fails", and an unset ``DEEPSEEK_API_KEY`` means that branch is simply not
    available. The skip is logged (without the key) so it is visible in the trace.
    """
    configs: list[ProviderConfig] = []
    for name in settings.llm.fallback_presets:
        try:
            config = provider_config(
                name,
                timeout_s=settings.llm.timeout_s,
                max_retries=settings.llm.max_retries,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                json_mode=settings.llm.json_mode,
            )
        except ConfigError as exc:
            _LOGGER.warning(
                "fallback provider skipped",
                extra={"preset": name, "reason": exc.message},
            )
            continue
        configs.append(config)
    return configs


def build_provider(
    settings: Settings,
    *,
    force_fake: bool = False,
    ledger: CostLedger | None = None,
    stage: str = "generation",
    prompt_version: str = "",
    runner: Runner | None = None,
    transport: object | None = None,
) -> ChatProvider:
    """Build the configured provider (or a fallback chain) from :class:`Settings`.

    ``force_fake=True`` is what ``answerdesk ask --fake`` uses: it ignores the API key
    requirement and returns the deterministic double, so the offline path never touches
    the network even when the profile says ``api``.
    """
    if force_fake:
        return _fake_provider(provider_config(FAKE_PRESET))

    primary = settings.resolved_provider()
    providers: list[ChatProvider] = [
        _from_config(
            primary,
            ledger=ledger,
            stage=stage,
            prompt_version=prompt_version,
            runner=runner,
            transport=transport,
        )
    ]
    seen = {primary.preset}
    for config in fallback_configs(settings):
        if config.preset in seen:
            continue
        seen.add(config.preset)
        providers.append(
            _from_config(
                config,
                ledger=ledger,
                stage=stage,
                prompt_version=prompt_version,
                runner=runner,
                transport=transport,
            )
        )
    if len(providers) == 1:
        return providers[0]
    return ProviderChain(providers)


def provider_sequence(settings: Settings) -> Sequence[ProviderConfig]:
    """Primary + fallback configs, in attempt order — for diagnostics and the console."""
    return [settings.resolved_provider(), *fallback_configs(settings)]


__all__ = ["build_provider", "fallback_configs", "provider_sequence"]
