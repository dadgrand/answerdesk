"""Typed artifacts of the evaluation harness: manifests, turns, cells, headers.

The contract models (:mod:`answerdesk.contracts`) define what crosses a module seam.
These models define what an *evaluation run* writes to disk, and they are closed for the
same reason: a report field nobody declared is a silent claim.

Everything here is deterministic given the seed: no timestamps inside hashed manifests, no
dict ordering that depends on ``hash()``.

    # AT-EVL-01  # AT-EVL-02  # AT-EVL-07  # AT-EVL-09
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from answerdesk.contracts import (
    Action,
    CacheStatus,
    Contract,
    EvalCategory,
    IntentBucket,
    ReasonCode,
    Split,
)


class FileDigest(Contract):
    """sha256 of one artifact plus how many records it holds."""

    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    items: int = Field(ge=0)


class CompositionCheck(Contract):
    """One §10.1 quota, with target, actual and verdict — printed, never assumed."""

    name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    actual: float | int
    passed: bool


class SplitManifest(Contract):
    """Frozen document-level split (FR-EVL-03, §10.2).

    The split unit is the **document**: a test item may not derive from a document whose
    chunks feed the dev split, otherwise paraphrase leakage inflates every number.
    """

    version: str = Field(min_length=1)
    seed: int
    corpus_version: str = Field(min_length=1)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    dev_doc_ids: list[str] = Field(default_factory=list)
    test_doc_ids: list[str] = Field(default_factory=list)
    sha256: str = Field(default="0" * 64, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _disjoint(self) -> SplitManifest:
        overlap = set(self.dev_doc_ids) & set(self.test_doc_ids)
        if overlap:
            raise ValueError(f"dev and test document sets overlap: {sorted(overlap)[:5]}")
        return self

    @property
    def payload(self) -> dict[str, Any]:
        """Everything the self-hash covers (the hash field itself is excluded)."""
        return {
            "version": self.version,
            "seed": self.seed,
            "corpus_version": self.corpus_version,
            "test_ratio": self.test_ratio,
            "dev_doc_ids": self.dev_doc_ids,
            "test_doc_ids": self.test_doc_ids,
        }


class GoldenManifest(Contract):
    """Provenance and composition of one golden-set version (FR-EVL-02, §10.1)."""

    version: str = Field(min_length=1)
    seed: int
    corpus_version: str = Field(min_length=1)
    corpus_hash: str = Field(min_length=64, max_length=64)
    split_hash: str = Field(min_length=64, max_length=64)
    generator: str = Field(min_length=1)
    generator_config: dict[str, Any] = Field(default_factory=dict)
    files: list[FileDigest] = Field(default_factory=list)
    composition: list[CompositionCheck] = Field(default_factory=list)
    by_locale: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    by_split: dict[str, int] = Field(default_factory=dict)
    multilingual_cells: dict[str, int] = Field(default_factory=dict)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.composition)


class LatinSquareRow(Contract):
    """One judge invocation of a pairwise item, so position bias is auditable."""

    order: Literal["ab", "ba"]
    verdict: str
    score_a: float = Field(ge=0.0, le=5.0)
    score_b: float = Field(ge=0.0, le=5.0)


class JudgeMetadata(Contract):
    """Identity of the judge — a judged number without this is a defect (§7.5)."""

    judge_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    position_swap: bool = True
    agreement_kappa: dict[str, float] = Field(default_factory=dict)
    agreement_n: dict[str, int] = Field(default_factory=dict)
    #: Model that produced the answers being judged, and both families (§10.5).
    generator_model: str = ""
    generator_family: str = ""
    judge_family: str = ""
    cross_family: bool = False
    #: Human κ established for the judge on this corpus/language set?
    human_calibrated: bool = False
    reportable: bool = False
    note: str = ""


class TurnRecord(Contract):
    """What happened on one golden case for one arm — the raw material of every metric."""

    case_id: str = Field(min_length=1)
    locale: str = Field(min_length=2)
    category: EvalCategory
    intent_bucket: IntentBucket = IntentBucket.INFORMATIONAL
    split: Split = Split.TEST
    arm: str = Field(min_length=1)

    answerable: bool
    must_escalate: bool = False
    expected_language: str = Field(min_length=2)

    # -- decision -------------------------------------------------------------
    action: Action
    reason_code: ReasonCode
    answered: bool
    escalated: bool

    # -- retrieval ------------------------------------------------------------
    ranked_chunk_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    acceptable_chunk_ids: list[str] = Field(default_factory=list)
    required_citation_chunk_ids: list[str] = Field(default_factory=list)
    evidence_cited: bool = False

    # -- answer quality -------------------------------------------------------
    claims_total: int = Field(default=0, ge=0)
    claims_supported: int = Field(default=0, ge=0)
    groundedness: float | None = Field(default=None, ge=0.0, le=1.0)
    groundedness_judged: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    phantom_citations: int = Field(default=0, ge=0)
    unsupported_displayed_claims: int = Field(default=0, ge=0)
    rubric_score: float | None = Field(default=None, ge=1.0, le=5.0)
    rubric_pass: bool | None = None
    #: Diagnostic only: the same rubric run against the reference answer. A gap between this
    #: and ``rubric_score`` means the reference is not entailed by the evidence.
    reference_rubric_score: float | None = Field(default=None, ge=1.0, le=5.0)
    language: str = Field(min_length=2)
    language_match: bool = True
    answer_ok: bool = False
    #: Uncalibrated confidence proxy (packed evidence score); 0.0 for a refusal. The
    #: conformal calibrator that turns this into ``p̂`` is t8's; see calibration.py.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # -- operations -----------------------------------------------------------
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usd: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_status: CacheStatus = CacheStatus.MISS
    error: str | None = None

    @model_validator(mode="after")
    def _decision_is_consistent(self) -> TurnRecord:
        if self.answered == self.escalated:
            raise ValueError("a turn is either answered or escalated, never both")
        if self.answered and self.action is not Action.ANSWER:
            raise ValueError("answered=True requires action=answer")
        return self


class MetricCell(Contract):
    """One reported number: value, n, 95% CI, and how the interval was computed.

    FR-EVL-07: the report may not print a metric without ``n`` and a CI — a rule the
    report generator enforces, not the reader.
    """

    name: str = Field(min_length=1)
    value: float
    n: int = Field(ge=0)
    ci_low: float
    ci_high: float
    method: Literal["wilson", "bootstrap", "none"] = "wilson"
    arm: str = ""
    locale: str = ""
    category: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _interval_ordered(self) -> MetricCell:
        if self.ci_low > self.value or self.ci_high < self.value:
            raise ValueError(
                f"metric {self.name!r}: CI [{self.ci_low}, {self.ci_high}] does not contain "
                f"the point estimate {self.value}"
            )
        return self


class TargetCheck(Contract):
    """A §6.4/§7.5/§8.5/§9.5 target compared against a measured cell.

    ``gated`` is false whenever the run cannot speak to the gate (replay mode, fake
    judge, machine labels): the report prints the number and says why it is diagnostic.
    """

    target_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    cell_name: str = Field(min_length=1)
    value: float | None = None
    passed: bool | None = None
    gated: bool = False
    note: str = ""


class RunHeader(Contract):
    """First page of every report: what produced these numbers (FR-EVL-09, §10.8)."""

    run_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    corpus_version: str = ""
    corpus_hash: str = ""
    golden_version: str = ""
    golden_hash: str = ""
    split_version: str = ""
    split_hash: str = ""
    seed: int = 0
    provider_mode: Literal["fake", "api", "local"] = "fake"
    replay: bool = True
    judge: JudgeMetadata
    arms: list[str] = Field(default_factory=list)
    locales: list[str] = Field(default_factory=list)
    case_count: int = Field(default=0, ge=0)
    duration_s: float = Field(default=0.0, ge=0.0)
    hardware: str = ""
    pricing_id: str = ""
    pricing_date: str = ""
    notes: list[str] = Field(default_factory=list)


class StatComparison(Contract):
    """A paired comparison between two arms, with the tests FR-EVL-07/§6.3 demands."""

    metric: str = Field(min_length=1)
    arm_a: str = Field(min_length=1)
    arm_b: str = Field(min_length=1)
    n: int = Field(ge=0)
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    bootstrap: int = 1000
    mcnemar_b: int = Field(default=0, ge=0)
    mcnemar_c: int = Field(default=0, ge=0)
    mcnemar_statistic: float = 0.0
    mcnemar_p: float = 1.0
    mcnemar_method: Literal["chi2", "exact_binomial", "n/a"] = "n/a"
    significant: bool = False
    note: str = ""


class RunResult(Contract):
    """Everything a report needs, in one serialisable object."""

    header: RunHeader
    cells: list[MetricCell] = Field(default_factory=list)
    comparisons: list[StatComparison] = Field(default_factory=list)
    targets: list[TargetCheck] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)
    reliability: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    def cells_named(self, name: str) -> list[MetricCell]:
        return [cell for cell in self.cells if cell.name == name]

    def cell(self, name: str, *, arm: str = "", locale: str = "") -> MetricCell | None:
        for cell in self.cells:
            if cell.name == name and cell.arm == arm and cell.locale == locale:
                return cell
        return None


class DatasetBundle(Contract):
    """A loaded golden set: cases plus the manifests that make it auditable."""

    cases: list[Any] = Field(default_factory=list)  # list[EvalCase], kept loose for typing
    golden: GoldenManifest | None = None
    split: SplitManifest | None = None
    corpus_hash: str = ""
    notes: list[str] = Field(default_factory=list)

    def by_split(self, split: Split) -> list[Any]:
        return [case for case in self.cases if case.split is split]


def category_counts(cases: list[Any]) -> dict[str, int]:
    """Counts by category name, stable order — used by manifests and reports."""
    counts: dict[str, int] = {}
    for case in cases:
        key = getattr(case.category, "value", str(case.category))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def locale_counts(cases: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.locale] = counts.get(case.locale, 0) + 1
    return dict(sorted(counts.items()))


def split_counts(cases: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        key = getattr(case.split, "value", str(case.split))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def merge_counts(*counts: Mapping[str, int]) -> dict[str, int]:
    total: dict[str, int] = {}
    for block in counts:
        for key, value in block.items():
            total[key] = total.get(key, 0) + value
    return dict(sorted(total.items()))


__all__ = [
    "CompositionCheck",
    "DatasetBundle",
    "FileDigest",
    "GoldenManifest",
    "JudgeMetadata",
    "LatinSquareRow",
    "MetricCell",
    "RunHeader",
    "RunResult",
    "SplitManifest",
    "StatComparison",
    "TargetCheck",
    "TurnRecord",
    "category_counts",
    "locale_counts",
    "merge_counts",
    "split_counts",
]
