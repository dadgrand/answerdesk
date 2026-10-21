"""The one HTTP client factory the hosted retrieval adapters use — proxy-tolerant by design.

``httpx.Client`` defaults to ``trust_env=True``, and that default is *right*: a corporate
proxy configured through ``HTTP_PROXY``/``HTTPS_PROXY``/``ALL_PROXY`` must keep working, and
on Windows ``urllib`` also reads the proxy from the registry, so users who never set a
variable can still legitimately be behind one.

The failure mode this module exists for: an ambient proxy whose scheme httpx cannot use —
this machine exposes ``socks4://127.0.0.1:10808``, and ``socks5://`` without the optional
``socksio`` package behaves the same way — makes ``httpx.Client(...)`` raise ``ValueError:
Unknown scheme for proxy URL`` **at construction time**, before a single request. A hosted
embedding call then dies with a stack trace that says nothing about proxies, and retrieval
cannot start at all.

The policy, mirroring the chat provider layer (:mod:`answerdesk.llm.providers`) so the whole
stack behaves one way:

1. try the client exactly as configured, ``trust_env`` untouched — a valid proxy keeps
   working and is never silently bypassed;
2. if httpx rejects the ambient proxy configuration (``ValueError``/``ImportError``), log a
   warning that names the reason and build the client with ``trust_env=False``, i.e. ignore
   the unusable ambient proxy and go direct;
3. if even that fails, raise :class:`answerdesk.errors.ConfigError` naming the variables to
   fix — never a bare ``ValueError`` from three frames deep.

Explicitly passing a ``transport`` (a ``httpx.MockTransport`` in tests, or a custom transport
in a deployment) is orthogonal: the transport is used either way, so proxy tolerance can be
tested without a socket.
"""

from __future__ import annotations

import logging

import httpx

from answerdesk.errors import ConfigError

_LOGGER = logging.getLogger(__name__)

#: Variables an operator has to look at when a proxy configuration is the problem.
PROXY_ENV_VARS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def build_http_client(
    *,
    timeout: float,
    transport: httpx.BaseTransport | None = None,
    follow_redirects: bool = False,
) -> httpx.Client:
    """Build a client, tolerating an unusable *ambient* proxy configuration.

    Returns a client on the first successful attempt. On an unusable ambient proxy the
    client is rebuilt with ``trust_env=False`` (direct connection) and a warning is logged;
    a genuinely broken explicit configuration raises :class:`ConfigError`.
    """
    timeout_config = httpx.Timeout(timeout)
    try:
        return httpx.Client(
            timeout=timeout_config, transport=transport, follow_redirects=follow_redirects
        )
    except (ValueError, ImportError) as exc:
        _LOGGER.warning(
            "ignoring unusable ambient proxy configuration for the hosted retrieval client",
            extra={"reason": str(exc), "proxy_env": list(PROXY_ENV_VARS)},
        )
    try:
        return httpx.Client(
            timeout=timeout_config,
            transport=transport,
            follow_redirects=follow_redirects,
            trust_env=False,
        )
    except (ValueError, ImportError) as exc:  # pragma: no cover - defensive
        raise ConfigError(
            f"cannot construct the HTTP client for the hosted retrieval endpoint: {exc}. "
            "httpx supports http(s) proxies (socks5 needs the optional 'socksio' package); "
            "fix or unset HTTP_PROXY / HTTPS_PROXY / ALL_PROXY.",
            detail=", ".join(PROXY_ENV_VARS),
        ) from exc


__all__ = ["PROXY_ENV_VARS", "build_http_client"]
