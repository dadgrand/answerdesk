"""The offline generator double: a ``FakeProvider`` that emits schema-valid drafts.

``answerdesk.testing.doubles.FakeProvider`` returns "a deterministic reply to: …", which
the generation pipeline correctly treats as malformed JSON and escalates. That is the right
behaviour for the pipeline's own tests, but an evaluation needs a *generator* that actually
answers, otherwise every answer-quality metric is measured on an empty set.

This provider therefore subclasses ``FakeProvider`` (so ``isinstance(provider, FakeProvider)``
is true, the offline profile is honest, and CI keeps running on doubles) and overrides the
one documented extension point, ``_reply``. Its policy is stated once, here, because the
report must be able to describe it:

1. parse the ``<passage id="pN">`` blocks the packer rendered;
2. read the question from the prompt text outside those blocks;
3. if no passage shares a content word with the question, **refuse**
   (``insufficient_evidence``) — a generator with no evidence must not answer;
4. if the question carries an exact identifier/number that no passage contains, **refuse**.
   This mirrors the documented failure mode of absent records and is what makes the
   unanswerable family measurable offline;
5. otherwise emit one claim per strongly matching passage, quoting the first sentence of
   that passage *verbatim*, which is exactly the citation contract the verifier enforces.

Consequences, stated so nobody mistakes them for quality: in replay mode groundedness is
high by construction and citation precision is high by construction; the numbers validate
the *machinery*, they do not describe a model (``REPLAY-MODE``, §10.7 item 9).

    # AT-EVL-05  # AT-EVL-08  # AT-GEN-09
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from answerdesk.contracts import Message, ProviderConfig
from answerdesk.eval.corpus import anchor_tokens, content_words
from answerdesk.testing.doubles import FakeProvider, default_fake_provider_config

# The header is ``<passage id="pN" doc="..." heading="acme > faq">``: a heading may itself
# contain ``>``, so the tag end is found as the first ``">`` (quote then angle bracket),
# which no attribute value can contain.
_PASSAGE = re.compile(r'<passage id="(?P<label>p\d+)"[\s\S]*?">\s*(?P<body>[\s\S]*?)</passage>')
_CUSTOMER_QUESTION = re.compile(
    r"<customer_question>(?P<question>.*?)</customer_question>", re.DOTALL
)
_SENTENCE = re.compile(r"[^.!?。！？]+[.!?。！？]?")
#: A claim shorter than this is not worth citing; quoting it verbatim would look silly.
MIN_SENTENCE_CHARS = 12
#: How much weaker a second passage may be and still earn its own claim.
SECOND_PASSAGE_RATIO = 0.6


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE.finditer(text) if match.group(0).strip()]


def _first_usable_sentence(body: str) -> str | None:
    """First citable sentence, preferring an FAQ answer over the FAQ question it answers."""
    answer = body.split("A:", 1)[1] if "A:" in body else body
    for sentence in _sentences(answer):
        if len(sentence) >= MIN_SENTENCE_CHARS:
            return sentence
    stripped = answer.strip()
    return stripped if len(stripped) >= MIN_SENTENCE_CHARS else None


def _question_text(messages: Sequence[Message]) -> str:
    """The turn text: the delimited customer question, or the message minus the evidence."""
    last_user = next(
        (message.content for message in reversed(messages) if message.role == "user"), ""
    )
    delimited = _CUSTOMER_QUESTION.search(last_user)
    if delimited is not None:
        return delimited.group("question")
    return _PASSAGE.sub(" ", last_user)


def _overlap(question_words: set[str], body: str) -> float:
    body_words = content_words(body)
    if not question_words or not body_words:
        return 0.0
    shared = float(len(question_words & body_words))
    return shared / math.sqrt(float(len(question_words)))


class EvidenceAnchoredProvider(FakeProvider):
    """Deterministic, offline, citation-shaped generator (a ``FakeProvider`` subclass)."""

    name = "fake-evidence"

    def __init__(self, config: ProviderConfig | None = None, *, max_claims: int = 2) -> None:
        super().__init__(config or default_fake_provider_config())
        self.max_claims = max(1, max_claims)
        self.refusals: list[str] = []

    # -- FakeProvider extension point ---------------------------------------------
    def _reply(self, messages: Sequence[Message]) -> str:
        last_user = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        passages = [
            (match.group("label"), match.group("body").strip())
            for match in _PASSAGE.finditer(last_user)
        ]
        question = _question_text(messages)
        if not passages:
            return self._refuse("no evidence was packed for this turn")
        anchors = anchor_tokens(question)
        evidence_text = "\n".join(body for _, body in passages)
        missing = [anchor for anchor in anchors if anchor not in _normalise(evidence_text)]
        if missing:
            return self._refuse(
                "the requested record is not in the knowledge base: " + ", ".join(missing[:3])
            )
        question_words = content_words(question)
        scored = sorted(
            ((_overlap(question_words, body), label, body) for label, body in passages),
            key=lambda item: (-round(item[0], 12), item[1]),
        )
        if not scored or scored[0][0] <= 0.0:
            return self._refuse("no passage shares a content word with the question")
        claims: list[dict[str, object]] = []
        for score, label, body in scored[: self.max_claims]:
            if claims and score < SECOND_PASSAGE_RATIO * scored[0][0]:
                break
            sentence = _first_usable_sentence(body)
            if sentence is None:
                continue
            claims.append(
                {
                    "text": sentence,
                    "citations": [{"passage": label, "quote": sentence}],
                }
            )
        if not claims:
            return self._refuse("no citable sentence was available in the evidence")
        return json.dumps(
            {"insufficient_evidence": False, "why": "", "claims": claims},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _refuse(self, why: str) -> str:
        self.refusals.append(why)
        return json.dumps(
            {"insufficient_evidence": True, "why": why, "claims": []},
            ensure_ascii=False,
            sort_keys=True,
        )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


__all__ = ["EvidenceAnchoredProvider"]
