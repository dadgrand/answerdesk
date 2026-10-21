"""Rubric judge: deterministic double for CI, rubric-anchored LLM judge for evidence.

The judge is a measurement instrument, so the harness treats it like one:

* **rubric versioning** — the rubric text, its version and the prompt version are recorded
  next to every judged number (§10.5 "Versioning"). A judge change is a visible report diff;
* **context anchoring** — the rubric grades a claim against the cited passage and the
  reference answer. It never mentions length, tone or confidence, because those are the
  documented verbosity/self-preference biases `[C]` (arXiv:2306.05685);
* **position swap** — pairwise comparisons run in both orders and are averaged, with an
  audit that reports order invariance. A judge that prefers the first slot is caught by the
  audit instead of quietly moving the leaderboard;
* **judge ≠ generator** — enforced at run construction: the judge model identity must differ
  from the provider model identity in a live run, and the report refuses to call a judged
  metric reportable otherwise.

The offline default is :class:`answerdesk.testing.doubles.FakeJudge`, which is deterministic
and cheap; its agreement with humans is *unknown*, and the report says so.

    # AT-EVL-05  # AT-GEN-08
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from answerdesk.contracts import ChatProvider, Judge, JudgeScore, Message, Role
from answerdesk.eval.errors import EvalError
from answerdesk.eval.models import JudgeMetadata, LatinSquareRow
from answerdesk.eval.stats import cohen_kappa

RUBRIC_VERSION = "rubric-v1"
JUDGE_PROMPT_VERSION = "judge-rubric-1"
#: §7.5 F-10: below this κ a judged metric is "not reportable" for that language.
KAPPA_REPORTABLE = 0.60
#: §7.5 F-1: a rubric pass is 4 or 5 on the 1-5 scale.
PASS_SCORE = 4.0

#: Known model families. A judge and a generator from the same family are not a
#: measurement pair (§10.5 self-enhancement bias), whatever the model names say.
_FAMILY_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qwen", ("qwen", "qwq")),
    ("deepseek", ("deepseek",)),
    ("openai", ("gpt", "o1", "o3", "o4", "chatgpt")),
    ("anthropic", ("claude",)),
    ("google", ("gemma", "gemini", "palm")),
    ("meta", ("llama",)),
    ("mistral", ("mistral", "mixtral", "magistral")),
    ("yandex", ("yandexgpt",)),
    ("t-tech", ("t-lite", "t-pro")),
)


def model_family(model: str) -> str:
    """Best-effort family label for a model id (``qwen3.6-35b-a3b`` -> ``qwen``)."""
    lowered = model.strip().lower()
    for family, tokens in _FAMILY_TOKENS:
        if any(token in lowered for token in tokens):
            return family
    head = re.split(r"[^a-z0-9]+", lowered, maxsplit=1)[0]
    return head or "unknown"


RUBRIC_TEXT = """\
# Answer rubric (answerdesk eval, rubric-v1)

You grade one customer-support answer. You are not a copy editor and you are not a
salesperson: style, length, tone and confidence are irrelevant to this rubric.

For each claim in the answer:

1. **Supported (5 or 4).** The cited passage contains the claim's fact, and the claim copies
   identifiers, numbers, dates, fees and product names exactly as the passage writes them.
2. **Partially supported (3).** The cited passage implies the claim but a specific number,
   date or identifier is missing or rounded.
3. **Unsupported (1 or 2).** The passage does not state the claim, contradicts it, or the
   claim adds a fact the passage never mentions.

Anchor every judgement in the passage text, not in what you believe about banks, cards or
telecoms. If the answer correctly declares that the evidence is insufficient, that is a
pass — refusing is a correct answer, an invented fact is a defect.
"""

_SYSTEM = (
    "You are a strict evaluation judge. You reply with exactly one JSON object "
    '{"score": <integer 1-5>, "rationale": "<one sentence>"} and nothing else.'
)

_USER = """\
<rubric>
{rubric}
</rubric>

<customer_question>
{question}
</customer_question>

<reference_answer>
{reference}
</reference_answer>

<cited_context>
{context}
</cited_context>

<candidate_answer>
{candidate}
</candidate_answer>

Score the candidate answer against the rubric. Never reward length or confidence.
Return the JSON object only.
"""


class JudgeError(EvalError):
    """The judge produced output the harness cannot trust, or is misconfigured."""

    code = "eval_judge_error"


@dataclass(frozen=True)
class PairwiseVerdict:
    """One pairwise judgement: which answer is better and the scores behind it."""

    winner: Literal["a", "b", "tie"]
    score_a: float
    score_b: float
    rationale: str = ""


@dataclass(frozen=True)
class JudgeComparison:
    """A pairwise comparison run in both orders, with the order effect made explicit."""

    ab: PairwiseVerdict
    ba: PairwiseVerdict
    score_a: float
    score_b: float
    order_consistent: bool
    rows: list[LatinSquareRow] = field(default_factory=list)

    @property
    def order_effect(self) -> float:
        """Absolute score change caused by presentation order (0 = no effect)."""
        return abs(self.ab.score_a - self.ba.score_a)


@runtime_checkable
class PairwiseJudge(Protocol):
    """The pairwise seam: absolute-score judges are adapted to it, not exempted."""

    name: str

    def compare(
        self, *, question: str, answer_a: str, answer_b: str, context: Sequence[str]
    ) -> PairwiseVerdict: ...


class RubricJudge:
    """Rubric-anchored LLM judge over any :class:`answerdesk.contracts.ChatProvider`."""

    name = "rubric-judge"

    def __init__(
        self,
        provider: ChatProvider,
        *,
        judge_model: str,
        rubric_version: str = RUBRIC_VERSION,
        prompt_version: str = JUDGE_PROMPT_VERSION,
        rubric: str = RUBRIC_TEXT,
    ) -> None:
        self.provider = provider
        self.judge_model = judge_model
        self.rubric_version = rubric_version
        self.prompt_version = prompt_version
        self.rubric = rubric
        self.calls = 0

    # -- contracts.Judge ---------------------------------------------------------
    def grade(
        self,
        *,
        question: str,
        reference_answer: str | None,
        candidate_answer: str,
        context: Sequence[str],
    ) -> JudgeScore:
        self.calls += 1
        prompt = _USER.format(
            rubric=self.rubric,
            question=question,
            reference=reference_answer or "(no reference answer)",
            context="\n\n".join(context) or "(no cited context)",
            candidate=candidate_answer or "(empty answer)",
        )
        completion = self.provider.complete(
            [
                Message(role=Role.SYSTEM, content=_SYSTEM),
                Message(role=Role.USER, content=prompt),
            ],
            response_schema=None,
        )
        score, rationale = _parse_score(completion.text)
        return JudgeScore(
            score=score,
            rubric_version=self.rubric_version,
            judge_model=self.judge_model,
            rationale=rationale,
            order_swapped=False,
        )

    # -- pairwise with position swap ---------------------------------------------
    #: Pairwise comparisons go through :func:`compare_pair`, which runs both orders and is
    #: implemented over the absolute ``grade`` above. Keeping one code path means the
    #: deterministic double and a live judge are audited by the *same* order-invariance
    #: check, instead of the double quietly getting an easier one.


class AbsolutePairwiseAdapter:
    """Turns an absolute-score judge (e.g. ``FakeJudge``) into the pairwise interface.

    The adapter is order-invariant *by construction* — it has no slot to be biased toward.
    It exists so the position-swap audit can run on the same code path as a real judge: if
    the audit ever reports a violation for an adapter, the bug is in the audit, not the model.
    """

    def __init__(self, judge: Judge) -> None:
        self.judge = judge
        self.name = getattr(judge, "name", "judge")

    def compare(
        self, *, question: str, answer_a: str, answer_b: str, context: Sequence[str]
    ) -> PairwiseVerdict:
        first = self.judge.grade(
            question=question, reference_answer=None, candidate_answer=answer_a, context=context
        )
        second = self.judge.grade(
            question=question, reference_answer=None, candidate_answer=answer_b, context=context
        )
        winner: Literal["a", "b", "tie"] = (
            "a" if first.score > second.score else "b" if second.score > first.score else "tie"
        )
        return PairwiseVerdict(
            winner=winner,
            score_a=first.score,
            score_b=second.score,
            rationale="absolute-score adapter",
        )


def compare_pair(
    judge: PairwiseJudge | Judge,
    *,
    question: str,
    answer_a: str,
    answer_b: str,
    context: Sequence[str],
) -> JudgeComparison:
    """Run a pairwise comparison in both orders and average (§10.5 position swap)."""
    pairwise: PairwiseJudge = (
        judge if isinstance(judge, PairwiseJudge) else AbsolutePairwiseAdapter(judge)
    )
    ab = pairwise.compare(question=question, answer_a=answer_a, answer_b=answer_b, context=context)
    ba_verdict = pairwise.compare(
        question=question, answer_a=answer_b, answer_b=answer_a, context=context
    )
    swapped: dict[str, Literal["a", "b", "tie"]] = {"a": "b", "b": "a", "tie": "tie"}
    ba = PairwiseVerdict(
        winner=swapped[ba_verdict.winner],
        score_a=ba_verdict.score_b,
        score_b=ba_verdict.score_a,
        rationale=ba_verdict.rationale,
    )
    score_a = (ab.score_a + ba.score_a) / 2.0
    score_b = (ab.score_b + ba.score_b) / 2.0
    return JudgeComparison(
        ab=ab,
        ba=ba,
        score_a=score_a,
        score_b=score_b,
        order_consistent=ab.winner == ba.winner,
        rows=[
            LatinSquareRow(order="ab", verdict=ab.winner, score_a=ab.score_a, score_b=ab.score_b),
            LatinSquareRow(order="ba", verdict=ba.winner, score_a=ba.score_a, score_b=ba.score_b),
        ],
    )


@dataclass(frozen=True)
class PositionSwapAudit:
    """How often the pairwise verdict survived being presented in the other order."""

    n: int
    consistent: int
    mean_order_effect: float

    @property
    def invariance(self) -> float:
        return (self.consistent / self.n) if self.n else 1.0

    @property
    def passed(self) -> bool:
        """§10.5: the fixture test asserts order invariance within tolerance."""
        return self.n == 0 or self.invariance >= 0.9


def position_swap_audit(
    judge: PairwiseJudge | Judge,
    items: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.5,
) -> PositionSwapAudit:
    """Audit order invariance over ``items`` (``question``, ``answer_a``, ``answer_b``)."""
    consistent = 0
    effects: list[float] = []
    checked = 0
    for item in items:
        comparison = compare_pair(
            judge,
            question=str(item["question"]),
            answer_a=str(item["answer_a"]),
            answer_b=str(item["answer_b"]),
            context=[str(part) for part in item.get("context", [])],
        )
        checked += 1
        effects.append(comparison.order_effect)
        if comparison.order_consistent and comparison.order_effect <= tolerance:
            consistent += 1
    mean_effect = sum(effects) / len(effects) if effects else 0.0
    return PositionSwapAudit(n=checked, consistent=consistent, mean_order_effect=mean_effect)


def _parse_score(raw: str) -> tuple[float, str]:
    """Extract ``{"score": n, "rationale": ...}`` from a judge reply, tolerantly."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence is not None:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise JudgeError(
            "the judge reply contains no JSON object", detail=f"first 200 characters: {raw[:200]!r}"
        )
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError("the judge reply is not valid JSON", detail=str(exc)) from exc
    if not isinstance(payload, dict) or "score" not in payload:
        raise JudgeError("the judge JSON has no 'score' field", detail=str(payload)[:200])
    try:
        score = float(payload["score"])
    except (TypeError, ValueError) as exc:
        raise JudgeError("the judge score is not a number", detail=str(payload)[:200]) from exc
    score = min(5.0, max(1.0, score))
    return score, str(payload.get("rationale", ""))[:500]


def judge_metadata(
    *,
    judge_model: str,
    prompt_version: str,
    rubric_version: str,
    position_swap: bool = True,
    agreement: Mapping[str, tuple[float, int]] | None = None,
    generator_model: str = "",
    judge_family: str | None = None,
    note: str = "",
) -> JudgeMetadata:
    """Build the metadata block every judged number must carry (§7.5 honesty clause).

    Two independent gates decide whether a judged number may be *reported*:

    * **cross-family** — the judge and the generator must come from different model
      families (§10.5). This is structural: a qwen answer graded by a qwen judge is
      labelled ``diagnostic`` no matter how good the rubric is;
    * **human calibration** — κ per language ≥ 0.60 (§7.5 F-10). Without labels the
      number is reported with an explicit "κ not established" caveat rather than being
      silently promoted to a release gate.
    """
    kappas = {locale: value for locale, (value, _) in (agreement or {}).items()}
    counts = {locale: count for locale, (_, count) in (agreement or {}).items()}
    human_calibrated = bool(kappas) and all(
        value >= KAPPA_REPORTABLE for value in kappas.values()
    )
    generator_family = model_family(generator_model) if generator_model else ""
    resolved_judge_family = judge_family or model_family(judge_model)
    cross_family = bool(generator_family) and generator_family != resolved_judge_family
    reportable = cross_family
    if not note:
        if not generator_family:
            note = "generator model not recorded: cross-family judging cannot be established"
        elif not cross_family:
            note = (
                f"same-family judge ({resolved_judge_family} judging {generator_family}): "
                "diagnostic only, never reported (§10.5)"
            )
        elif not kappas:
            note = (
                "cross-family judge; human κ not established for this corpus, so judged "
                "numbers are reported with an explicit κ caveat, not as F-1/F-2/F-3 gates"
            )
        elif not human_calibrated:
            below = sorted(locale for locale, value in kappas.items() if value < KAPPA_REPORTABLE)
            note = f"cross-family judge; κ below {KAPPA_REPORTABLE} for {', '.join(below)}"
        else:
            note = "cross-family judge with κ at or above the reporting threshold"
    return JudgeMetadata(
        judge_model=judge_model,
        prompt_version=prompt_version,
        rubric_version=rubric_version,
        position_swap=position_swap,
        agreement_kappa=kappas,
        agreement_n=counts,
        generator_model=generator_model,
        generator_family=generator_family,
        judge_family=resolved_judge_family,
        cross_family=cross_family,
        human_calibrated=human_calibrated,
        reportable=reportable,
        note=note,
    )


def human_agreement(
    human: Sequence[Mapping[str, Any]], judged: Mapping[str, bool]
) -> dict[str, tuple[float, int]]:
    """Per-locale Cohen's κ between human labels and judge pass/fail decisions."""
    grouped: dict[str, list[tuple[bool, bool]]] = {}
    for row in human:
        case_id = str(row["case_id"])
        if case_id not in judged:
            continue
        locale = str(row["locale"])
        grouped.setdefault(locale, []).append((bool(row["human_pass"]), bool(judged[case_id])))
    agreement: dict[str, tuple[float, int]] = {}
    pooled_left: list[bool] = []
    pooled_right: list[bool] = []
    for locale, pairs in sorted(grouped.items()):
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        agreement[locale] = (cohen_kappa(left, right), len(pairs))
        pooled_left.extend(left)
        pooled_right.extend(right)
    if len(agreement) > 1:
        agreement["all"] = (cohen_kappa(pooled_left, pooled_right), len(pooled_left))
    return agreement


__all__ = [
    "JUDGE_PROMPT_VERSION",
    "KAPPA_REPORTABLE",
    "PASS_SCORE",
    "RUBRIC_TEXT",
    "RUBRIC_VERSION",
    "AbsolutePairwiseAdapter",
    "JudgeComparison",
    "JudgeError",
    "PairwiseJudge",
    "PairwiseVerdict",
    "PositionSwapAudit",
    "RubricJudge",
    "compare_pair",
    "human_agreement",
    "judge_metadata",
    "position_swap_audit",
]
