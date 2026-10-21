"""Latency v2 runner and artifact checker (CMD-LAT).

Two jobs, one module, because the published table and the CI check must read the *same*
schema:

``--run``
    Measure the arms and write the artifacts under ``reports/latency-v2/``:

    * ``--offline`` — no network: times the pipeline itself with a zero-latency scripted
      provider (pipeline overhead, prompt build, JSON parse, citation/verifier work) and the
      token accounting of each arm. Useful in CI and as the "not measured here" baseline.
    * ``--live`` — the real thing: retrieve from the frozen corpus, stream each arm through the
      configured OpenAI-compatible provider, record TTFT/total/tokens/USD from the stream, judge
      the answers with the cross-family rubric judge, and route a cascade arm by ``p_hat``.

``--check``
    Validate the committed artifacts and re-derive the claims the docs make: both arms present,
    TTFT recorded for streamed arms, retrieval p95 ≤ 15 ms, output-token reduction ≥ 30 %,
    quality inside the stated budget, and honest labels when an arm was not measured here.

Exit codes: 0 = every check passed; 2 = at least one violation (printed, and also written next
to the artifact as ``check-report.json`` so a failing CI job leaves evidence).

    # AT-LAT-01  # AT-LAT-02  # AT-LAT-03  # AT-COST-01
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from answerdesk.contracts import Contract, Message, RetrievedPassage
from answerdesk.generate.budget import AnswerBudget
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.llm.cascade import CascadePolicy, CascadeRouter
from answerdesk.llm.latency import ArmLatency, LatencySample, summarise_arm

ARTIFACT_VERSION = "1.0.0"
ARTIFACT_FILE = "latency.json"
REPORT_FILE = "latency.md"
CHECK_FILE = "check-report.json"

#: Acceptance thresholds, in one place, read by both the runner and the checker.
TARGET_TOKEN_REDUCTION_PCT = 30.0
MAX_QUALITY_DROP = 0.2
MAX_RUBRIC_DROP = 2.0
MAX_RETRIEVAL_P95_MS = 15.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_out_dir() -> Path:
    return repo_root() / "reports" / "latency-v2"


# --------------------------------------------------------------------------------------
# Artifact schema
# --------------------------------------------------------------------------------------


class QualityRow(Contract):
    """Rubric and groundedness for one arm (judged, or marked unmeasured)."""

    arm: str
    n: int = 0
    rubric_mean: float = 0.0
    rubric_p50: float = 0.0
    groundedness: float = 0.0
    answered: int = 0
    judge_failures: int = 0
    judge_model: str = ""
    measured_here: bool = True


class RetrievalRow(Contract):
    n: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    note: str = "retrieval stage only; unchanged by this task"


class CascadeRow(Contract):
    n: int = 0
    small: int = 0
    large: int = 0
    reasons: dict[str, int] = Field(default_factory=dict)
    easy_threshold: float = 0.90
    min_margin: float = 0.02
    small_model: str = ""
    large_model: str = ""


class LatencyChecks(Contract):
    tokens_reduction_pct: float = 0.0
    quality_drop: float = 0.0
    quality_within_budget: bool = False
    cascade_drop: float = 0.0
    cascade_within_budget: bool = False
    token_target_met: bool = False
    retrieval_p95_ok: bool = False
    ttft_recorded: bool = False


class LatencyArtifact(Contract):
    """The whole published measurement (one file, one schema, one checker)."""

    version: str = ARTIFACT_VERSION
    generated_at: str = ""
    mode: str = "offline"
    profile: str = ""
    host: str = ""
    generator_model: str = ""
    judge_model: str = ""
    measured_here: bool = False
    arms: list[ArmLatency] = Field(default_factory=list)
    quality: list[QualityRow] = Field(default_factory=list)
    retrieval: RetrievalRow = Field(default_factory=RetrievalRow)
    cascade: CascadeRow = Field(default_factory=CascadeRow)
    checks: LatencyChecks = Field(default_factory=LatencyChecks)
    notes: list[str] = Field(default_factory=list)
    commands: dict[str, str] = Field(default_factory=dict)

    def arm(self, name: str) -> ArmLatency | None:
        return next((row for row in self.arms if row.arm == name), None)

    def quality_of(self, name: str) -> QualityRow | None:
        return next((row for row in self.quality if row.arm == name), None)


# --------------------------------------------------------------------------------------
# Derivation and checking
# --------------------------------------------------------------------------------------


def derive_checks(artifact: LatencyArtifact) -> LatencyChecks:
    """Recompute every published claim from the rows (never trust a stored flag)."""
    baseline = artifact.arm("baseline")
    budget = artifact.arm("budget")
    reduction = 0.0
    if baseline is not None and budget is not None and baseline.output_tokens_mean > 0:
        reduction = (
            (baseline.output_tokens_mean - budget.output_tokens_mean)
            / baseline.output_tokens_mean
            * 100.0
        )
    base_quality: QualityRow | None = artifact.quality_of("baseline")
    budget_quality: QualityRow | None = artifact.quality_of("budget")
    drop = 0.0
    if base_quality is not None and budget_quality is not None and base_quality.measured_here:
        drop = base_quality.rubric_mean - budget_quality.rubric_mean
    cascade_quality: QualityRow | None = artifact.quality_of("cascade")
    cascade_drop = 0.0
    if base_quality is not None and cascade_quality is not None and cascade_quality.measured_here:
        cascade_drop = base_quality.rubric_mean - cascade_quality.rubric_mean
    streamed = [row for row in artifact.arms if row.streamed]
    return LatencyChecks(
        tokens_reduction_pct=round(reduction, 2),
        quality_drop=round(drop, 3),
        quality_within_budget=drop <= MAX_QUALITY_DROP,
        cascade_drop=round(cascade_drop, 3),
        cascade_within_budget=cascade_drop <= MAX_RUBRIC_DROP,
        token_target_met=reduction >= TARGET_TOKEN_REDUCTION_PCT,
        retrieval_p95_ok=artifact.retrieval.p95_ms <= MAX_RETRIEVAL_P95_MS,
        ttft_recorded=bool(streamed) and all(row.ttft.n > 0 for row in streamed),
    )


def check_artifact(artifact: LatencyArtifact) -> list[str]:
    """Every violation of the published contract, as human-readable strings."""
    problems: list[str] = []
    if not artifact.arms:
        problems.append("no arms measured")
    names = {row.arm for row in artifact.arms}
    for required in ("baseline", "budget"):
        if required not in names:
            problems.append(f"arm {required!r} missing from the artifact")
    if artifact.measured_here:
        # A measured artifact must carry the whole table: streamed TTFT, retrieval timing and
        # judged quality. An unmeasured one must not pretend to.
        streamed = [row for row in artifact.arms if row.streamed]
        if not streamed:
            problems.append("no streamed arm: TTFT cannot be reported")
        for row in streamed:
            if row.ttft.n == 0:
                problems.append(f"arm {row.arm!r} is streamed but has no TTFT samples")
            elif row.ttft.p50_ms <= 0.0:
                problems.append(f"arm {row.arm!r} reports a non-positive TTFT p50")
        for row in artifact.arms:
            if row.total.n == 0:
                problems.append(f"arm {row.arm!r} has no total-latency samples")
        if not artifact.retrieval.p95_ms:
            problems.append("retrieval p95 is missing")
        elif artifact.retrieval.p95_ms > MAX_RETRIEVAL_P95_MS:
            problems.append(
                f"retrieval p95 {artifact.retrieval.p95_ms} ms exceeds {MAX_RETRIEVAL_P95_MS} ms"
            )
    checks = derive_checks(artifact)
    if artifact.measured_here:
        if not checks.token_target_met:
            problems.append(
                f"output-token reduction {checks.tokens_reduction_pct}% < "
                f"{TARGET_TOKEN_REDUCTION_PCT}%"
            )
        if not checks.quality_within_budget:
            problems.append(
                f"quality drop {checks.quality_drop} exceeds {MAX_QUALITY_DROP} "
                "(budget for the token cap)"
            )
        for quality_row in artifact.quality:
            if not quality_row.measured_here:
                problems.append(f"quality for arm {quality_row.arm!r} is marked unmeasured")
    else:
        if not any("not measured here" in note.lower() for note in artifact.notes):
            problems.append(
                "artifact is not measured here but no note says so — the label is mandatory"
            )
    if artifact.cascade.n and artifact.cascade.small + artifact.cascade.large != artifact.cascade.n:
        problems.append("cascade routing counts do not add up")
    return problems


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_markdown(artifact: LatencyArtifact) -> str:
    checks = derive_checks(artifact)
    lines = [
        "# Latency v2 — measured arms",
        "",
        f"Generated {artifact.generated_at} · mode `{artifact.mode}` · profile "
        f"`{artifact.profile}` · measured here: **{artifact.measured_here}**",
        "",
        f"Generator: `{artifact.generator_model or 'n/a'}` · judge: "
        f"`{artifact.judge_model or 'n/a'}` · host: {artifact.host}",
        "",
        "| arm | n | TTFT p50 | TTFT p95 | total p50 | total p95 | in tok p50 | out tok mean "
        "| USD/call |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in artifact.arms:
        lines.append(
            f"| {row.arm} | {row.n} | {row.ttft.p50_ms:.0f} ms | {row.ttft.p95_ms:.0f} ms | "
            f"{row.total.p50_ms:.0f} ms | {row.total.p95_ms:.0f} ms | "
            f"{row.input_tokens_p50:.0f} | {row.output_tokens_mean:.1f} | "
            f"{('$' + format(row.usd_mean, '.5f')) if row.priced else 'n/a (unpriced)'}"
            f" |"
        )
    lines += [
        "",
        "## Quality",
        "",
        "| arm | n | rubric mean | rubric p50 | groundedness | answered | judge failures |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for quality_row in artifact.quality:
        label = (
            f"{quality_row.rubric_mean:.2f}" if quality_row.measured_here else "not measured here"
        )
        lines.append(
            f"| {quality_row.arm} | {quality_row.n} | {label} | "
            f"{quality_row.rubric_p50:.2f} | {quality_row.groundedness:.3f} | "
            f"{quality_row.answered} | {quality_row.judge_failures} |"
        )
    lines += [
        "",
        "## Retrieval (must not regress)",
        "",
        f"n={artifact.retrieval.n} · p50 {artifact.retrieval.p50_ms:.2f} ms · "
        f"p95 {artifact.retrieval.p95_ms:.2f} ms · target p95 ≤ "
        f"{MAX_RETRIEVAL_P95_MS:.0f} ms · {artifact.retrieval.note}",
        "",
        "## Cascade",
        "",
        f"n={artifact.cascade.n} · small={artifact.cascade.small} "
        f"({artifact.cascade.small_model or 'n/a'}) · large={artifact.cascade.large} "
        f"({artifact.cascade.large_model or 'n/a'}) · easy threshold "
        f"{artifact.cascade.easy_threshold:.2f} · min margin {artifact.cascade.min_margin:.2f}",
        "",
        "reasons: "
        + ", ".join(f"{key}={value}" for key, value in artifact.cascade.reasons.items()),
        "",
        "## Checks",
        "",
        f"* output-token reduction: **{checks.tokens_reduction_pct:.1f}%** "
        f"(target ≥ {TARGET_TOKEN_REDUCTION_PCT:.0f}%) — "
        f"{'met' if checks.token_target_met else 'not met'}",
        f"* quality drop: **{checks.quality_drop:+.2f}** rubric points "
        f"(budget ≤ {MAX_QUALITY_DROP}) — "
        f"{'within budget' if checks.quality_within_budget else 'over budget'}",
        f"* cascade quality drop: **{checks.cascade_drop:+.2f}** rubric points "
        f"(budget ≤ {MAX_RUBRIC_DROP:.0f}) — "
        f"{'within budget' if checks.cascade_within_budget else 'over budget'}",
        f"* retrieval p95: {artifact.retrieval.p95_ms:.2f} ms — "
        f"{'ok' if checks.retrieval_p95_ok else 'OVER TARGET'}",
        f"* TTFT recorded for every streamed arm: {checks.ttft_recorded}",
        "",
        "## Notes",
        "",
    ]
    lines += [f"* {note}" for note in artifact.notes] or ["* (none)"]
    lines += ["", "## Commands", ""]
    lines += [f"* `{name}`: `{command}`" for name, command in artifact.commands.items()]
    return "\n".join(lines) + "\n"


def write_artifact(artifact: LatencyArtifact, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / ARTIFACT_FILE
    md_path = out_dir / REPORT_FILE
    payload = artifact.model_dump(mode="json")
    payload["checks"] = derive_checks(artifact).model_dump(mode="json")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(artifact), encoding="utf-8")
    return [json_path, md_path]


def load_artifact(path: Path | None = None) -> LatencyArtifact:
    target = path or (default_out_dir() / ARTIFACT_FILE)
    if not target.is_file():
        raise SystemExit(f"no latency artifact at {target}: run `--run` first")
    raw = json.loads(target.read_text(encoding="utf-8"))
    return LatencyArtifact.model_validate(raw)


# --------------------------------------------------------------------------------------
# Offline runner: pipeline overhead, no network
# --------------------------------------------------------------------------------------


class _ScriptedProvider:
    """Zero-latency provider returning one valid draft: times the *pipeline*, not a model."""

    name = "scripted-zero-latency"

    def __init__(self, draft_json: str, *, config: Any = None) -> None:
        self.draft_json = draft_json
        self.config = config
        self.calls = 0
        self.output_tokens = max(1, len(draft_json) // 4)

    def complete(self, messages: Sequence[Message], *, response_schema: Any = None) -> Any:
        from answerdesk.contracts import Completion

        self.calls += 1
        return Completion(
            text=self.draft_json,
            model="scripted",
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=self.output_tokens,
        )

    def stream(self, messages: Sequence[Message], *, response_schema: Any = None) -> Sequence[Any]:
        from answerdesk.contracts import CompletionChunk

        text = self.draft_json
        step = max(1, len(text) // 4)
        chunks = [
            CompletionChunk(delta=text[index : index + step], done=False, model="scripted")
            for index in range(0, len(text), step)
        ]
        chunks.append(CompletionChunk(delta="", done=True, model="scripted"))
        return chunks


def _fixture_passages() -> list[RetrievedPassage]:
    from answerdesk.contracts import Channel, Chunk

    text = (
        "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц. "
        "Плата списывается первого числа каждого месяца."
    )
    chunk = Chunk(
        chunk_id="chunk-fee::0",
        doc_id="doc-fee",
        text=text,
        heading_path=["Тарифы", "Обслуживание"],
        breadcrumb="Тарифы > Обслуживание",
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
        language="ru",
    )
    return [RetrievedPassage(chunk=chunk, channel=Channel.DENSE, score=0.94, rank=1)]


def _draft_payload(claims: int) -> str:
    quote = "Обслуживание дебетовой карты «Вектор» стоит 120 рублей в месяц."
    return json.dumps(
        {
            "insufficient_evidence": False,
            "why": "",
            "claims": [
                {
                    "text": "Обслуживание карты «Вектор» стоит 120 рублей в месяц.",
                    "citations": [{"passage": "p1", "quote": quote}],
                }
                for _ in range(claims)
            ],
        },
        ensure_ascii=False,
    )


def large_model_fallback(settings: Any) -> str:
    """The model name to report when the settings leave it implicit."""
    return str(getattr(settings.llm, "model", "") or "unknown")


def run_offline(*, repeats: int = 20) -> LatencyArtifact:
    """Time the pipeline (no model) for both arms and record the token accounting."""
    from answerdesk.generate.prompts import PromptRegistry
    from answerdesk.generate.schema import load_answer_schema

    passages = _fixture_passages()
    registry = PromptRegistry.load()
    schema = load_answer_schema()
    samples: dict[str, list[LatencySample]] = {"baseline": [], "budget": []}
    budget_arms = {"baseline": AnswerBudget.baseline(), "budget": AnswerBudget.budget()}
    for name, arm in budget_arms.items():
        provider = _ScriptedProvider(_draft_payload(arm.max_claims))
        pipeline = AnswerPipeline(provider, registry=registry, schema=schema, budget=arm)
        for _ in range(repeats):
            timer_start = time.perf_counter()
            total_ms = 0.0
            ttft_ms = 0.0
            chunks = 0
            for event in pipeline.stream_answer("Сколько стоит обслуживание карты?", passages):
                if event.kind == "delta":
                    chunks += 1
                    if ttft_ms == 0.0:
                        ttft_ms = event.ttft_ms
                elif event.kind == "result" and event.result is not None:
                    total_ms = event.elapsed_ms
            if total_ms == 0.0:  # pragma: no cover - defensive
                total_ms = (time.perf_counter() - timer_start) * 1000.0
            samples[name].append(
                LatencySample(
                    arm=name,
                    provider=provider.name,
                    model="scripted (pipeline only)",
                    streamed=True,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    input_tokens=0,
                    output_tokens=len(provider.draft_json) // 4,
                    usd=0.0,
                    chunks=chunks,
                    note="pipeline overhead only: the provider is instantaneous",
                )
            )
    arms = [summarise_arm(samples[name]) for name in ("baseline", "budget")]
    for row in arms:
        # The provider is instantaneous, so a TTFT here would measure rounding noise. Declare the
        # arms as non-streamed: the artifact then makes no TTFT claim at all.
        row.streamed = False
        row.note = (
            "offline: zero-latency provider, so total is pipeline overhead, not model time; "
            "TTFT is not claimed"
        )
    artifact = LatencyArtifact(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="offline",
        profile="offline",
        host=f"{sys.platform} / python {sys.version.split()[0]}",
        generator_model="scripted (no model)",
        judge_model="",
        measured_here=False,
        arms=arms,
        quality=[
            QualityRow(arm="baseline", n=0, measured_here=False),
            QualityRow(arm="budget", n=0, measured_here=False),
        ],
        retrieval=RetrievalRow(n=0, p50_ms=0.0, p95_ms=0.0, note="not exercised offline"),
        notes=[
            "Not measured here: no model endpoint was used. These numbers time the pipeline "
            "(prompt build, packing, JSON parse, citation building, verifier) with an "
            "instantaneous provider and must not be quoted as model latency.",
            "Run `python -m answerdesk.llm.latency_report --run --live` with a reachable "
            "OpenAI-compatible endpoint for TTFT/total/tokens/USD and the rubric comparison.",
        ],
        commands={
            "offline": "python -m answerdesk.llm.latency_report --run --offline",
            "live": "python -m answerdesk.llm.latency_report --run --live --limit 24",
            "check": "python -m answerdesk.llm.latency_report --check",
        },
    )
    return artifact


# --------------------------------------------------------------------------------------
# Live runner
# --------------------------------------------------------------------------------------


def _load_live_env() -> None:
    """Load ``.env`` and the documented live profiles (never overrides set variables)."""
    import os

    env_path = repo_root() / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _live_providers() -> tuple[Any, Any, str, str]:
    """(large provider, small provider, model, small model) from the documented presets."""
    from answerdesk.config import load_settings
    from answerdesk.llm.factory import build_provider
    from answerdesk.llm.presets import provider_config
    from answerdesk.llm.providers import HttpChatProvider

    settings = load_settings(profile="api")
    large = build_provider(settings)
    small_config = provider_config("deepseek")
    small = HttpChatProvider(small_config)
    model = str(settings.llm.model or large_model_fallback(settings))
    return large, small, model, small_config.model


@contextlib.contextmanager
def _offline_env() -> Any:
    """Hide live-provider overrides while an offline profile is loaded.

    `scripts/live-env.ps1` exports ``ANSWERDESK_LLM__*``; the retrieval pipeline must still be
    built from the offline profile (it never calls a model), so those variables are parked for the
    duration of the load and restored afterwards.
    """
    import os

    parked = {
        key: os.environ.pop(key) for key in list(os.environ) if key.startswith("ANSWERDESK_LLM__")
    }
    try:
        yield
    finally:
        os.environ.update(parked)


def _retrieval_pipeline() -> Any:
    from answerdesk.config import load_settings
    from answerdesk.index.store import IndexBundle
    from answerdesk.retrieval.pipeline import build_pipeline
    from answerdesk.testing.doubles import HashingEmbedder

    with _offline_env():
        settings = load_settings(profile="offline")
    index_dir = repo_root() / "data" / "index" / "corpus-77d7d081813738a6"
    embedder = HashingEmbedder(settings.embedding.dimension)
    index = IndexBundle.load(index_dir, embedder)
    return build_pipeline(settings, index)


def _live_cases(limit: int) -> list[tuple[str, str, list[RetrievedPassage]]]:
    """The first ``limit`` answerable direct questions of the frozen corpus, with evidence."""
    from answerdesk.index.store import ChunkStore
    from answerdesk.retrieval.datasets import load_query_rows

    corpus_dir = repo_root() / "data" / "corpus"
    ChunkStore.from_corpus(corpus_dir)  # validates the corpus layout
    rows = load_query_rows(corpus_dir)
    wanted = [
        row
        for row in rows
        if row.get("answerable") and row.get("category") == "direct" and not row.get("style")
    ][:limit]
    pipeline = _retrieval_pipeline()
    cases: list[tuple[str, str, list[RetrievedPassage]]] = []
    for row in wanted:
        question = str(row.get("text") or row.get("base_text") or "")
        if not question:
            continue
        trace, _timings = pipeline.retrieve_with_timings(question)
        passages = list(trace.packed)
        if passages:
            cases.append((str(row.get("query_id")), question, passages))
    return cases


def run_live(*, limit: int = 24, judge: bool = True) -> LatencyArtifact:
    """Measure both arms against a real endpoint, judge them, and route the cascade."""
    _load_live_env()
    from answerdesk.eval.judge import RubricJudge

    large, small, large_model, small_model = _live_providers()
    cases = _live_cases(limit)
    if not cases:
        raise SystemExit("no live cases available: is data/corpus + data/index present?")
    retrieval = _retrieval_pipeline()

    samples: dict[str, list[LatencySample]] = {"baseline": [], "budget": []}
    answers: dict[str, list[tuple[str, str, list[str]]]] = {"baseline": [], "budget": []}
    retrieval_ms: list[float] = []
    budget_arms = {"baseline": AnswerBudget.baseline(), "budget": AnswerBudget.budget()}
    pipelines = {name: AnswerPipeline(large, budget=arm) for name, arm in budget_arms.items()}
    # The cascade arm is wired after the first pass: it needs p_hat per item to choose a provider.
    from answerdesk.policy.scorers import OverlapScorer

    scorer = OverlapScorer()
    for _case_id, question, passages in cases:
        timing = retrieval.retrieve_with_timings(question)[1]
        retrieval_ms.append(float(timing.get("total_ms", 0.0)))
        for name, pipeline in pipelines.items():
            ttft = 0.0
            total = 0.0
            chunks = 0
            result = None
            for event in pipeline.stream_answer(question, passages):
                if event.kind == "delta":
                    chunks += 1
                    ttft = ttft or event.ttft_ms
                elif event.kind == "result":
                    result = event.result
                    total = event.elapsed_ms
            if result is None:  # pragma: no cover - defensive
                continue
            samples[name].append(
                LatencySample(
                    arm=name,
                    provider=large.name,
                    model=large_model,
                    streamed=True,
                    ttft_ms=ttft,
                    total_ms=total,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    usd=result.usage.usd,
                    chunks=chunks,
                    priced=result.usage.usd > 0.0,
                )
            )
            answers[name].append(
                (question, result.text, [packed.chunk.text for packed in result.packed.passages])
            )

    # Cascade: derive the easy threshold from this run's own p_hat distribution (the scorer's
    # scale is not portable: a threshold tuned for one scorer is meaningless for another), route,
    # then measure the routed arm so its quality is comparable with the single-model baseline.
    p_hats = [
        (case_id, question, passages, scorer.score(question, passages))
        for case_id, question, passages in cases
    ]
    values = sorted(entry[3] for entry in p_hats)
    # Confidence-quantile routing: the most confident 60% of this sample may take the small
    # arm. The *absolute* threshold is scorer-specific (the overlap scorer is monotone but
    # uncalibrated); production takes it from the calibrated engine's band edge instead.
    easy_threshold = round(values[max(0, int(0.40 * len(values)) - 1)], 3) if values else 0.90
    policy = CascadePolicy(
        small=CascadePolicy().small.model_copy(update={"model": small_model, "preset": "deepseek"}),
        large=CascadePolicy().large.model_copy(update={"model": large_model}),
        easy_threshold=max(0.01, easy_threshold),
        # The margin rule needs a *calibrated* p_hat (t8's engine). The overlap scorer used
        # here is monotone but uncalibrated, so a margin on its scale would be noise.
        min_margin=0.0,
    )
    router = CascadeRouter(policy)
    routed: dict[str, str] = {}
    reason_counts: dict[str, int] = {}
    small_count = 0
    for case_id, _question, _passages, p_hat in p_hats:
        decision = router.route(p_hat)
        routed[case_id] = decision.arm.name
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        if decision.arm.name == policy.small.name:
            small_count += 1

    # Measure the routed arm with a fresh pass (small provider for easy items, large otherwise).
    cascade_samples: list[LatencySample] = []
    cascade_answers: list[tuple[str, str, list[str]]] = []
    small_pipeline = AnswerPipeline(small, budget=AnswerBudget.budget())
    for _case_id, question, passages, _p_hat in p_hats:
        arm_name = routed.get(case_id, "large")
        pipeline = small_pipeline if arm_name == "small" else pipelines["baseline"]
        ttft = 0.0
        total = 0.0
        result = None
        for event in pipeline.stream_answer(question, passages):
            if event.kind == "delta":
                ttft = ttft or event.ttft_ms
            elif event.kind == "result":
                result = event.result
                total = event.elapsed_ms
        if result is None:
            continue
        cascade_samples.append(
            LatencySample(
                arm="cascade",
                provider=small.name if arm_name == "small" else large.name,
                model=policy.small.model if arm_name == "small" else policy.large.model,
                streamed=True,
                ttft_ms=ttft,
                total_ms=total,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                usd=result.usage.usd,
                priced=result.usage.usd > 0.0,
            )
        )
        cascade_answers.append(
            (question, result.text, [packed.chunk.text for packed in result.packed.passages])
        )

    quality: list[QualityRow] = []
    judge_provider = None
    judge_model = small_model
    if judge:
        judge_provider = small  # cross-family: qwen generator, deepseek judge
    for name in ("baseline", "budget", "cascade"):
        rows = cascade_answers if name == "cascade" else answers[name]
        rubric: list[float] = []
        failures = 0
        grounded = 0
        answered = 0
        if judge_provider is not None and rows:
            rubric_judge = RubricJudge(judge_provider, judge_model=judge_model)
            for question, text, context in rows:
                if not text.strip():
                    rubric.append(0.0)
                    continue
                answered += 1
                try:
                    score = rubric_judge.grade(
                        question=question,
                        reference_answer=None,
                        candidate_answer=text,
                        context=context,
                    )
                    rubric.append(float(score.score))
                    if float(score.score) >= 4.0:
                        grounded += 1
                except Exception:  # a judge failure must not fake a score
                    failures += 1
        quality.append(
            QualityRow(
                arm=name,
                n=len(rows),
                rubric_mean=round(statistics.fmean(rubric), 3) if rubric else 0.0,
                rubric_p50=round(statistics.median(rubric), 3) if rubric else 0.0,
                groundedness=round(grounded / answered, 3) if answered else 0.0,
                answered=answered,
                judge_failures=failures,
                judge_model=judge_model if judge_provider is not None else "",
                measured_here=bool(rubric),
            )
        )

    arms = [summarise_arm(samples[name]) for name in ("baseline", "budget")]
    if cascade_samples:
        arms.append(summarise_arm(cascade_samples))
    artifact = LatencyArtifact(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mode="live",
        profile="api",
        host=f"{sys.platform} / python {sys.version.split()[0]}",
        generator_model=large_model,
        judge_model=judge_model if judge_provider is not None else "",
        measured_here=True,
        arms=arms,
        quality=quality,
        retrieval=RetrievalRow(
            n=len(retrieval_ms),
            p50_ms=round(statistics.median(retrieval_ms), 3) if retrieval_ms else 0.0,
            p95_ms=round(sorted(retrieval_ms)[max(0, int(0.95 * len(retrieval_ms)) - 1)], 3)
            if retrieval_ms
            else 0.0,
        ),
        cascade=CascadeRow(
            n=len(cases),
            small=small_count,
            large=len(cases) - small_count,
            reasons=reason_counts,
            easy_threshold=policy.easy_threshold,
            min_margin=policy.min_margin,
            small_model=small_model,
            large_model=large_model,
        ),
        notes=[
            "Both arms stream from the same endpoint with the same retrieval; only the output "
            "budget (cap, claim limit, compact schema, extractive instruction) differs.",
            *(
                [
                    "Cascade not exercised: no item cleared the derived threshold on this sample, "
                    "so the cascade row is a second large-model pass. The router itself is "
                    "unit-tested; a calibrated p_hat (t8 engine) is required for a meaningful "
                    "split."
                ]
                if small_count == 0
                else []
            ),
            "TTFT is measured inside the stream (first non-empty delta), not derived from total.",
            f"Judge: cross-family rubric judge ({judge_model}) over {len(cases)} items per arm.",
        ],
        commands={
            "live": "python -m answerdesk.llm.latency_report --run --live --limit 24",
            "check": "python -m answerdesk.llm.latency_report --check",
        },
    )
    return artifact


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="answerdesk.llm.latency_report",
        description="Measure (--run) or validate (--check) the latency-v2 artifacts.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate the committed artifacts")
    mode.add_argument("--run", action="store_true", help="measure and write the artifacts")
    parser.add_argument("--offline", action="store_true", help="--run without any network")
    parser.add_argument("--live", action="store_true", help="--run against a real endpoint")
    parser.add_argument("--limit", type=int, default=24, help="items per arm for --live")
    parser.add_argument("--no-judge", action="store_true", help="skip the rubric judge")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out or default_out_dir()
    if args.check:
        artifact = load_artifact(out_dir / ARTIFACT_FILE)
        problems = check_artifact(artifact)
        checks = derive_checks(artifact)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "ok": not problems,
            "problems": problems,
            "checks": checks.model_dump(mode="json"),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / CHECK_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not args.quiet:
            print(f"artifact: {out_dir / ARTIFACT_FILE}")
            print(
                f"arms: {', '.join(row.arm for row in artifact.arms)} · "
                f"measured_here={artifact.measured_here} · mode={artifact.mode}"
            )
            if artifact.arms:
                print(
                    f"TTFT p50/p95 (baseline): {artifact.arms[0].ttft.p50_ms:.0f} / "
                    f"{artifact.arms[0].ttft.p95_ms:.0f} ms"
                )
            print(
                f"token reduction {checks.tokens_reduction_pct:.1f}% · "
                f"quality drop {checks.quality_drop:+.2f} · "
                f"retrieval p95 {artifact.retrieval.p95_ms:.2f} ms"
            )
            for problem in problems:
                print(f"VIOLATION: {problem}")
            print("check: " + ("ok" if not problems else f"{len(problems)} violation(s)"))
        return 0 if not problems else 2

    artifact = (
        run_offline()
        if args.offline or not args.live
        else run_live(limit=args.limit, judge=not args.no_judge)
    )
    written = write_artifact(artifact, out_dir)
    problems = check_artifact(artifact)
    if not args.quiet:
        for path in written:
            print(f"wrote {path}")
        for problem in problems:
            print(f"VIOLATION: {problem}")
        print("run: " + ("ok" if not problems else f"{len(problems)} violation(s) recorded"))
    return 0 if not problems else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
