"""Optional LLM query rewrite: a cached, identifier-guarded hook that is off by default.

Why it is optional and off: a rewrite costs a model call on the hot path, and the query
analyzer already handles the mechanical noise (segmentation, transliteration, bounded fuzzy
matching) without one. The hook exists for the class that rules cannot reach — a message whose
*vocabulary* is right but whose *intent* is implicit — and it is measured, not assumed.

Three properties are non-negotiable, and each has a test:

1. **Deterministic and cached.** ``CachingRewriter`` memoises on ``(rewriter name, normalized
   query)``, so replaying the same goldens costs one call per distinct query and never
   depends on the model's mood twice. ``LlmQueryRewriter`` sends ``temperature=0`` through the
   provider config either way.
2. **Identifiers cannot be invented.** ``validate_rewrite`` rejects any candidate that
   contains a digit-bearing token or a date/money/order-code form which the *input* did not
   contain, and also rejects an empty result, a length blow-up, and a rewrite that drops an
   input identifier. A rejected rewrite returns ``None`` and the analyzer proceeds with the
   mechanical expansion only.
3. **Additive.** The rewrite is an *extra probe*, never a replacement: the raw query keeps
   its own arm in the sparse channel (``FR-RET-06``).

The provider is injected (``ChatProvider``), which is what makes the offline tests possible:
they pass :class:`answerdesk.testing.doubles.FakeProvider` and check the cache, the guard and
the determinism without a socket.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from answerdesk.contracts import ChatProvider, Message, Role
from answerdesk.index.text import identifier_tokens, normalize_text

#: Bumped when the instruction changes — part of the rewriter's name and cache key.
PROMPT_VERSION = "query-rewrite-v1"

#: The instruction. Deliberately narrow: reformulate, never answer, never invent.
SYSTEM_PROMPT = (
    "You rewrite a customer-support message into a short search query for a knowledge base.\n"
    "Rules:\n"
    "1. Keep the user's language; do not translate.\n"
    "2. Keep every number, date, order id, card fragment and error code verbatim.\n"
    "3. Do not add facts, names, amounts or identifiers that are not in the message.\n"
    "4. Remove greetings, politeness and filler; keep the intent words.\n"
    "5. Output the query only, one line, at most 25 words."
)

#: Rewrites longer than this are almost certainly model chatter, not a query.
MAX_REWRITE_CHARS = 240

#: A rewrite that grows the query by more than this factor is rejected.
MAX_GROWTH = 3.0

_QUOTES = "\"'«»”“`"


@runtime_checkable
class QueryRewriter(Protocol):
    """The seam: something that can reformulate a query, or decline (``None``)."""

    name: str

    def rewrite(self, query: str) -> str | None: ...


def validate_rewrite(raw: str, candidate: str | None) -> str | None:
    """Return a safe rewrite, or ``None`` when the candidate must be rejected.

    Rejection reasons, all of which are checked in tests: empty output, model chatter
    (too long), a rewrite that lost an identifier the user typed, and any *new* digit-bearing
    token or date/money form — the ``CR-2``-style rule applied to queries: a rewrite may
    reformulate, it may not invent referents.
    """
    if candidate is None:
        return None
    cleaned = candidate.strip().strip(_QUOTES).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^\s*(query|rewrite)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    if not cleaned or len(cleaned) > MAX_REWRITE_CHARS:
        return None
    raw_normalized = normalize_text(raw)
    if len(cleaned) > max(len(raw_normalized) * MAX_GROWTH, 40):
        return None

    input_identifiers = {token.lower() for token in identifier_tokens(raw_normalized)}
    output_identifiers = {token.lower() for token in identifier_tokens(cleaned)}
    if output_identifiers - input_identifiers:
        return None  # invented number / date / code
    if input_identifiers - output_identifiers:
        return None  # dropped an identifier the user typed
    if cleaned == raw_normalized:
        return None
    return cleaned


class LlmQueryRewriter:
    """Rewrite through an injected :class:`answerdesk.contracts.ChatProvider`."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        prompt_version: str = PROMPT_VERSION,
        max_words: int = 25,
    ) -> None:
        self.provider = provider
        self.prompt_version = prompt_version
        self.max_words = max_words
        self.name = f"llm-rewrite:{prompt_version}:{getattr(provider, 'name', 'provider')}"

    def rewrite(self, query: str) -> str | None:
        messages: Sequence[Message] = (
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=Role.USER, content=query),
        )
        completion = self.provider.complete(messages)
        return validate_rewrite(query, completion.text)


class CachingRewriter:
    """Memoise an inner rewriter on ``(name, normalized query)`` with a bounded cache."""

    def __init__(self, inner: QueryRewriter, *, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.inner = inner
        self.max_entries = max_entries
        self.name = f"cached:{inner.name}"
        self._cache: OrderedDict[str, str | None] = OrderedDict()
        self.calls = 0
        self.hits = 0

    def __len__(self) -> int:
        return len(self._cache)

    def rewrite(self, query: str) -> str | None:
        key = normalize_text(query).lower()
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.calls += 1
        value = self.inner.rewrite(query)
        self._cache[key] = value
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return value

    def clear(self) -> None:
        self._cache.clear()
        self.calls = 0
        self.hits = 0


def build_rewriter(
    provider: ChatProvider | None,
    *,
    enabled: bool = False,
    prompt_version: str = PROMPT_VERSION,
    max_entries: int = 1024,
) -> QueryRewriter | None:
    """Return a cached rewriter when the operator enabled one, else ``None`` (default)."""
    if not enabled or provider is None:
        return None
    return CachingRewriter(
        LlmQueryRewriter(provider, prompt_version=prompt_version), max_entries=max_entries
    )


__all__ = [
    "MAX_REWRITE_CHARS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "CachingRewriter",
    "LlmQueryRewriter",
    "QueryRewriter",
    "build_rewriter",
    "validate_rewrite",
]
