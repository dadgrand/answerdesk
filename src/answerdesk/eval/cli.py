"""``answerdesk eval``: the command behind the reproducibility contract (§10.8, CMD-Q).

The command is a thin shell; the work lives in the modules it calls:

    corpus → golden set (frozen, validated) → arms → runner → metrics → report

Exit codes: ``0`` when the run completed and the report was written, ``2`` for a data or
reporting failure (missing corpus, missing labels, split leakage, a report that would print a
number without an interval). A *target* failing is not an error here: gates are evaluated by
the evidence task (t11), and a harness that refused to run because the system is not good
enough yet would never produce the baseline that shows it.

    # AT-EVL-08  # AT-EVL-09  # AT-EVL-10
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from answerdesk.config import Settings
from answerdesk.contracts import EvalCase, Judge, Split
from answerdesk.eval.arms import DEFAULT_ARMS, arm_specs, build_arms
from answerdesk.eval.corpus import load_corpus
from answerdesk.eval.datasets import GoldenConfig, generate_golden, load_golden, write_golden
from answerdesk.eval.errors import EvalError
from answerdesk.eval.judge import (
    AbsolutePairwiseAdapter,
    human_agreement,
    position_swap_audit,
)
from answerdesk.eval.models import RunResult, TurnRecord
from answerdesk.eval.report import LIMITATIONS, comparison_table, target_table, write_report
from answerdesk.eval.runner import (
    EvalRunner,
    RunnerConfig,
    build_header,
    build_judge,
    build_provider,
    build_result,
    deterministic_digest,
)
from answerdesk.generate.pipeline import AnswerPipeline

#: Default golden version — frozen at ``evals/golden/v1`` (§10.2).
GOLDEN_VERSION = "v1"
#: Default seed of the generator; changing it changes the split and therefore the frozen hash.
GOLDEN_SEED = 20261021
JudgeMode = Literal["auto", "fake", "llm", "none"]


class Emit(Protocol):
    """``typer.echo`` and a test double both satisfy this."""

    def __call__(self, message: str = "", /) -> None: ...


def eval_command(
    *,
    settings: Settings,
    emit: Emit,
    all_arms: bool = False,
    retrieval_only: bool = False,
    provider: str = "fake",
    judge_mode: str = "auto",
    report_dir: Path | None = None,
    regenerate: bool = False,
    arms: Sequence[str] | None = None,
    limit: int | None = None,
    split: str = "test",
    seed: int = GOLDEN_SEED,
    version: str = GOLDEN_VERSION,
    strict: bool = True,
    answer_limit: int | None = None,
    retrieval_limit: int | None = None,
    calibration: Path | None = None,
    force_index: bool = False,
    replay: bool = False,
    secondary: bool = True,
) -> int:
    """Run the evaluation end to end, write a report, print a summary; returns the exit code."""
    # The golden root is derived from the configured split path so a test (or an operator
    # profile) can relocate the whole ``evals/`` tree without a new config key.
    split_path = settings.resolve_path(settings.eval.split)
    evals_root = split_path.parent.parent
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    config = GoldenConfig(version=version, seed=seed)

    if regenerate:
        generated = generate_golden(corpus, config, strict=strict)
        manifest = write_golden(generated, corpus, evals_root)
        passed = sum(1 for check in manifest.composition if check.passed)
        emit(
            f"generated golden set {version}: {len(generated.cases)} cases, "
            f"composition checks {passed}/{len(manifest.composition)} passed"
        )

    if all_arms:
        return _run_benchmark(
            settings=settings,
            emit=emit,
            version=version,
            seed=seed,
            report_dir=report_dir,
            answer_limit=answer_limit,
            retrieval_limit=retrieval_limit,
            calibration=calibration,
            force_index=force_index,
            replay=replay or provider == "fake",
            secondary=secondary,
            retrieval_only=retrieval_only,
        )

    cases, manifest, split_manifest = load_golden(
        evals_root, corpus=corpus, version=version, strict=strict
    )
    selected = _select_split(cases, split)
    emit(
        f"golden {version}: {len(selected)} {split} cases over "
        f"{len({case.locale for case in selected})} locales"
    )

    arm_names = _arm_names(arms=arms, all_arms=all_arms)
    resolved_judge = _judge_mode(judge_mode, provider, retrieval_only)
    run_config = RunnerConfig(
        arms=arm_names,
        ks=(10, 20),
        seed=seed,
        bootstrap=settings.eval.bootstrap_samples,
        provider_mode=_provider_mode(provider, settings),
        judge_mode=resolved_judge,
        limit=limit,
    )
    chat_provider = build_provider(settings, mode=run_config.provider_mode)
    judge, judge_meta = build_judge(
        settings,
        mode="llm" if resolved_judge == "llm" else "fake",
        provider=chat_provider,
    )
    notes = _judge_notes(judge, judge_meta, selected, resolved_judge, evals_root)
    pipeline = AnswerPipeline.from_settings(chat_provider, settings)
    runner = EvalRunner(
        corpus=corpus,
        arms=build_arms(corpus, run_config.arms),
        pipeline=pipeline,
        judge=judge,
        config=run_config,
    )
    outcome = runner.run(selected)
    run_id = _run_id(report_dir, version, run_config)
    header = build_header(
        run_id=run_id,
        corpus=corpus,
        golden_hash=_files_digest(manifest.files),
        golden_version=version,
        split_version=split_manifest.version,
        split_hash=split_manifest.sha256,
        seed=run_config.seed,
        config=run_config,
        judge=judge_meta,
        arms=arm_specs(run_config.arms),
        case_count=len(selected),
        duration_s=outcome.duration_s,
        settings=settings,
        extra_notes=[*outcome.notes, *notes],
    )
    result = build_result(
        header=header,
        outcome=outcome,
        config=run_config,
        judged=judge_meta.reportable,
        limitations=list(LIMITATIONS),
    )
    digest = deterministic_digest(result)
    directory = _report_directory(settings, report_dir, run_id)
    command = _reproduction_command(provider, report_dir, version, split)
    paths = write_report(
        result,
        directory,
        run_id=run_id,
        command=command,
        digest=digest,
        latest_root=settings.resolve_path(settings.app.reports_dir),
        extra_notes=notes,
    )
    _print_summary(emit, result, paths.markdown, digest, len(selected))
    return 0


def _run_benchmark(
    *,
    settings: Settings,
    emit: Emit,
    version: str,
    seed: int,
    report_dir: Path | None,
    answer_limit: int | None,
    retrieval_limit: int | None,
    calibration: Path | None,
    force_index: bool,
    replay: bool,
    secondary: bool,
    retrieval_only: bool,
) -> int:
    """The full evidence suite behind ``answerdesk eval --all`` (FR-EVL-10)."""
    from answerdesk.eval.bench import BenchConfig, run_bench

    split_path = settings.resolve_path(settings.eval.split)
    evals_root = split_path.parent.parent
    out_dir = (
        Path(report_dir).resolve()
        if report_dir is not None
        else (settings.resolve_path(settings.app.reports_dir) / "bench").resolve()
    )
    generator_arms: tuple[str, ...] = ()
    if not retrieval_only:
        generator_arms = ("primary", "secondary") if secondary else ("primary",)
    config = BenchConfig(
        run_id=out_dir.name,
        golden_version=version,
        seed=seed,
        bootstrap=settings.eval.bootstrap_samples,
        answer_limit=answer_limit,
        retrieval_limit=retrieval_limit,
        calibration_path=str(calibration) if calibration is not None else None,
        force_index=force_index,
        replay=replay,
        generator_arms=generator_arms,
    )
    summary = run_bench(settings, out_dir=out_dir, config=config, evals_root=evals_root)
    emit(
        f"bench {summary['run_id']}: corpus {summary['corpus_version']} "
        f"({summary['documents']} docs / {summary['chunks']} chunks), "
        f"{summary['test_cases']} test cases, {summary['duration_s']} s"
    )
    for arm, values in summary.get("retrieval", {}).items():
        cell = values.get("recall_any@10")
        if cell:
            emit(
                f"retrieval {arm}: recall_any@10 {cell['value']:.3f} "
                f"[{cell['ci_low']:.3f},{cell['ci_high']:.3f}] n={cell['n']}"
            )
    for arm, models in summary.get("models", {}).items():
        emit(
            f"arm {arm}: generator {models['generator']} judge {models['judge']} "
            f"cross-family {'yes' if models['cross_family'] else 'no'} "
            f"reportable {'yes' if models['reportable'] else 'no'}"
        )
    if replay:
        emit("REPLAY-MODE: no API key detected or forced; numbers are diagnostics")
    # run_bench redirects a replay run that would overwrite live evidence, so report the
    # directory that actually holds this run's artefacts.
    emit(f"artifacts: {summary.get('run_dir', out_dir)}")
    return 0


def _arm_names(*, arms: Sequence[str] | None, all_arms: bool) -> tuple[str, ...]:
    if arms:
        return tuple(arms)
    if all_arms:
        return tuple(DEFAULT_ARMS)
    return ("hybrid-rrf", "bm25", "dense-hash")


def _provider_mode(provider: str, settings: Settings) -> Literal["fake", "api", "local"]:
    if provider == "fake":
        return "fake"
    if provider == "api":
        return "local" if settings.profile == "local" else "api"
    raise EvalError(f"unknown provider mode {provider!r}; use fake|api")


def _judge_mode(
    judge_mode: str, provider: str, retrieval_only: bool
) -> Literal["fake", "llm", "none"]:
    if retrieval_only:
        return "none"
    if judge_mode == "auto":
        return "fake" if provider == "fake" else "llm"
    if judge_mode == "llm" and provider == "fake":
        raise EvalError(
            "--judge llm needs a real provider: a deterministic double cannot grade. "
            "Use --judge fake (the default) or a real --provider with --profile api/local.",
        )
    if judge_mode in ("fake", "llm"):
        return judge_mode  # type: ignore[return-value]
    raise EvalError(f"unknown judge mode {judge_mode!r}; use auto|fake|llm")


def _select_split(cases: Sequence[EvalCase], split: str) -> list[EvalCase]:
    if split == "all":
        return list(cases)
    try:
        target = Split(split)
    except ValueError as exc:
        raise EvalError(f"unknown split {split!r}; use dev|test|all") from exc
    selected = [case for case in cases if case.split is target]
    if not selected:
        raise EvalError(f"the golden set has no {split} cases")
    return selected


def _files_digest(files: Sequence[object]) -> str:
    """Stable identity of the frozen artifact set: sha256 over ``path:sha256`` lines.

    Using one file's digest would make the header ambiguous (which file?), and hashing the
    loaded cases would change with ``--split``. The artifact set is the right unit.
    """
    import hashlib

    lines = sorted(f"{getattr(item, 'path', '')}:{getattr(item, 'sha256', '')}" for item in files)
    if not lines:
        return ""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _judge_notes(
    judge: Judge,
    metadata: object,
    cases: Sequence[EvalCase],
    mode: Literal["fake", "llm", "none"],
    evals_root: Path,
) -> list[str]:
    notes: list[str] = []
    if mode == "none":
        notes.append("judge disabled (--retrieval-only): judged metrics are absent by design")
        return notes
    audit_items = _audit_items(cases)
    if audit_items:
        audit = position_swap_audit(AbsolutePairwiseAdapter(judge), audit_items)
        notes.append(
            f"position-swap audit: {audit.consistent}/{audit.n} comparisons kept their verdict "
            f"across both orders (invariance {audit.invariance:.2f}, "
            f"mean order effect {audit.mean_order_effect:.3f})"
        )
    human = _human_calibration(evals_root, cases)
    if human:
        agreement = human_agreement(human, _judge_passes(judge, cases))
        rendered = ", ".join(
            f"{locale} κ={value[0]:.3f} (n={value[1]})" for locale, value in agreement.items()
        )
        notes.append(f"human agreement: {rendered}")
    else:
        notes.append(
            "human agreement: no calibration labels on disk (evals/judge/human_v1.jsonl); "
            "judged metrics stay not-reportable (§7.5 F-10)"
        )
    return notes


def _audit_items(cases: Sequence[EvalCase]) -> list[dict[str, object]]:
    """Pairwise audit items built from reference answers: full vs truncated.

    A real pairwise audit needs two candidate answers. In a replay run the honest cheap
    substitute is the reference answer against a deliberately weakened copy of itself: an
    order-sensitive judge still shows up, and no extra model call is invented.
    """
    items: list[dict[str, object]] = []
    for case in cases:
        if not case.reference_answer:
            continue
        reference = case.reference_answer
        items.append(
            {
                "question": case.question,
                "answer_a": reference,
                "answer_b": reference[: max(20, len(reference) // 2)],
                "context": [reference],
            }
        )
        if len(items) >= 8:
            break
    return items


def _human_calibration(evals_root: Path, cases: Sequence[EvalCase]) -> list[dict[str, object]]:
    """Optional human labels for κ (§10.5); absent by default, and the report says so."""
    path = evals_root / "judge" / "human_v1.jsonl"
    if not path.is_file():
        return []
    known = {case.case_id for case in cases}
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("case_id") in known:
            rows.append(payload)
    return rows


def _judge_passes(judge: Judge, cases: Sequence[EvalCase]) -> dict[str, bool]:
    passes: dict[str, bool] = {}
    for case in cases:
        if not case.reference_answer:
            continue
        score = judge.grade(
            question=case.question,
            reference_answer=case.reference_answer,
            candidate_answer=case.reference_answer,
            context=[case.reference_answer],
        )
        passes[case.case_id] = score.score >= 4.0
    return passes


def _run_id(report_dir: Path | None, version: str, config: RunnerConfig) -> str:
    if report_dir is not None:
        return Path(report_dir).name or f"eval-{version}"
    return f"eval-{version}-{'-'.join(config.arms)}"


def _report_directory(settings: Settings, report_dir: Path | None, run_id: str) -> Path:
    if report_dir is not None:
        return Path(report_dir).resolve()
    return (settings.resolve_path(settings.app.reports_dir) / run_id).resolve()


def _reproduction_command(provider: str, report_dir: Path | None, version: str, split: str) -> str:
    target = str(report_dir) if report_dir is not None else f"reports/eval-{version}"
    extra = "" if split == "test" else f" --split {split}"
    return f"uv run answerdesk eval --synthetic --provider {provider} --report {target}{extra}"


def _print_summary(emit: Emit, result: RunResult, markdown: Path, digest: str, cases: int) -> None:
    header = result.header
    emit(
        f"run {header.run_id}: {cases} cases x {len(header.arms)} arms in {header.duration_s:.1f}s"
    )
    emit(
        f"corpus {header.corpus_hash[:16]} golden {header.golden_hash[:16]} "
        f"split {header.split_hash[:16]} seed {header.seed}"
    )
    emit(
        f"judge {header.judge.judge_model} (prompt {header.judge.prompt_version}, "
        f"rubric {header.judge.rubric_version}); reportable: "
        f"{'yes' if header.judge.reportable else 'no'}"
    )
    for cell in result.cells:
        if cell.locale == "macro" and cell.name in {"recall_any@10", "recall_any@20"}:
            emit(
                f"{cell.name} [{cell.arm}] = {cell.value:.3f} "
                f"[{cell.ci_low:.3f}, {cell.ci_high:.3f}] n={cell.n}"
            )
    for name in ("realised_risk", "coverage", "must_escalate_recall", "citation_recall"):
        named = next(
            (item for item in result.cells if item.name == name and item.arm == header.arms[0]),
            None,
        )
        if named is not None and named.n > 0:
            emit(f"{name} [{named.arm}] = {named.value:.3f} n={named.n}")
    if result.comparisons:
        emit("")
        emit(comparison_table(result.comparisons))
    if result.targets:
        emit("")
        emit(target_table(result.targets))
    emit("")
    emit(f"deterministic digest sha256:{digest}")
    emit(f"report: {markdown}")


def summarize_turns(turns: Sequence[TurnRecord]) -> dict[str, int]:
    """Small diagnostic used by tests: how many turns of each outcome."""
    summary: dict[str, int] = {"total": len(turns), "answered": 0, "correct": 0, "escalated": 0}
    for turn in turns:
        summary["answered"] += 1 if turn.answered else 0
        summary["escalated"] += 1 if turn.escalated else 0
        summary["correct"] += 1 if turn.answer_ok else 0
    return summary


__all__ = ["GOLDEN_SEED", "GOLDEN_VERSION", "Emit", "eval_command", "summarize_turns"]
