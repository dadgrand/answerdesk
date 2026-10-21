"""Report generation: markdown tables, plots, the ``latest`` pointer, and honesty checks.

A report is the evidence; prose claims downstream of it are only as good as the table. Two
rules are therefore enforced in code, not in a reviewer's memory:

* **no number without n and a 95% CI** (FR-EVL-07). Every table cell is a
  :class:`~answerdesk.eval.models.MetricCell`, which cannot be constructed without an
  interval that contains its point estimate;
* **no gate claim the run cannot support** (FR-EVL-09, §10.7 item 9). A replay run is
  labelled in a banner *and* per table, every target is marked ``diagnostic`` unless the
  run used a real provider and a reportable judge, and the honest-limitations policy is
  pasted into the report itself rather than linked to.

Files written into ``reports/<run>/``::

    report.md            the report (tables + numbers + limitations)
    metrics.json         every cell, comparison and target, machine-readable
    turns.jsonl          one line per (case, arm) — the raw material for an audit
    cells.csv            the same cells as CSV, for a spreadsheet
    plots/*.png          retrieval, calibration, risk–coverage, latency

plus ``reports/latest.json`` (and a ``latest`` symlink when the filesystem allows it) so a
reader always knows which run is current.

    # AT-EVL-08  # AT-EVL-09  # AT-EVL-07
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.eval.errors import ReportError
from answerdesk.eval.models import MetricCell, RunResult, StatComparison, TargetCheck

REPLAY_BANNER = (
    "> **REPLAY-MODE.** This run used deterministic doubles (FakeProvider / FakeJudge) and "
    "machine-labelled synthetic data. It validates the harness, not the system: no number "
    "here may be presented as a live-model or production result (§10.7 item 9, NFR-REP-01)."
)

#: §10.7, verbatim in every report (the policy is normative, not advisory).
LIMITATIONS: tuple[str, ...] = (
    "**Synthetic data optimism.** Queries derive from the corpus and are paraphrased from it; "
    "recall and groundedness are therefore optimistic relative to real support traffic. No "
    "number here transfers to a production corpus without re-measurement.",
    "**Small n.** A few hundred test items give roughly ±3 points on the macro retrieval "
    "metric and much wider intervals on per-locale and subset metrics. Every cell carries n "
    "and a CI; a 'win' that does not clear the CI is reported as noise.",
    "**Judge bias.** Judged metrics are LLM-mediated; low-resource languages are likely worse. "
    "κ per language is published; where κ < 0.60 the metric is marked 'not reportable' rather "
    "than softened with adjectives.",
    "**Benchmark ≠ support domain.** Public leaderboard values are vendor-curated snapshots; "
    "an embedder choice is justified only by our own ablation.",
    "**Conformal exchangeability.** The risk guarantee holds under exchangeability. v1.0 has "
    "no real traffic, so drift is simulated, not observed.",
    "**Environment dependence.** Latency and cost numbers are for the recorded machine and "
    "the pinned pricing table; they are not portable claims.",
    "**No production evidence.** No SLA, no incident history, no live traffic, no measured "
    "handoff cost.",
    "**Cache results are replay-based.** The semantic cache belongs to t8; this run measures "
    "an exact-normalised-query probe, not a production hit rate.",
    "**Replay mode is labelled.** Any number produced without a live model/provider carries "
    "REPLAY-MODE and may never be presented as a live result.",
    "**No claim of SOTA.** The repository claims reproducibility, calibration artifacts and "
    "citation verification — not state-of-the-art retrieval or generation.",
)


@dataclass
class ReportPaths:
    """Where the report artifacts landed."""

    directory: Path
    markdown: Path
    metrics: Path
    turns: Path
    cells_csv: Path
    plots: list[Path] = field(default_factory=list)
    latest: Path | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            "directory": str(self.directory),
            "markdown": str(self.markdown),
            "metrics": str(self.metrics),
            "turns": str(self.turns),
            "cells_csv": str(self.cells_csv),
            "plots": ",".join(str(path) for path in self.plots),
            "latest": str(self.latest) if self.latest is not None else "",
        }


# --------------------------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------------------------


def _cell(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _metric_cell_row(cell: MetricCell) -> list[str]:
    return [
        cell.name,
        _cell(cell.value),
        f"[{_cell(cell.ci_low)}, {_cell(cell.ci_high)}]",
        str(cell.n),
        cell.method,
        cell.note or "",
    ]


def _group(cells: Sequence[MetricCell], arm: str) -> dict[str, list[MetricCell]]:
    grouped: dict[str, list[MetricCell]] = {}
    for cell in cells:
        if cell.arm != arm:
            continue
        grouped.setdefault(cell.name, []).append(cell)
    return grouped


def _sort_locales(cells: Sequence[MetricCell]) -> list[MetricCell]:
    order = {"macro": 0}
    return sorted(cells, key=lambda cell: (order.get(cell.locale, 1), cell.locale))


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def render_markdown(
    result: RunResult,
    *,
    run_id: str,
    command: str,
    digest: str,
    extra_notes: Sequence[str] = (),
) -> str:
    """Render the full report; the layout mirrors §10.8's artifact list."""
    header = result.header
    lines: list[str] = [
        f"# answerdesk evaluation report — `{run_id}`",
        "",
        REPLAY_BANNER
        if header.replay
        else "> **LIVE-COMPONENT RUN.** Not a production trial; see limitations.",
        "",
        "## 1. Run header",
        "",
        _table(
            ("field", "value"),
            [
                ("run id", f"`{header.run_id}`"),
                ("schema / contracts", header.schema_version),
                ("corpus version", f"`{header.corpus_version}`"),
                ("corpus hash (sha256)", f"`{header.corpus_hash}`"),
                ("golden version", header.golden_version),
                ("golden hash (sha256)", f"`{header.golden_hash}`"),
                ("split version", header.split_version),
                ("split hash (sha256)", f"`{header.split_hash}`"),
                ("seed", str(header.seed)),
                ("provider mode", f"{header.provider_mode}{' (replay)' if header.replay else ''}"),
                ("judge model", header.judge.judge_model),
                ("judge prompt version", header.judge.prompt_version),
                ("judge rubric version", header.judge.rubric_version),
                ("judge position swap", "on" if header.judge.position_swap else "off"),
                ("judge κ reportable", "yes" if header.judge.reportable else "no"),
                ("arms", ", ".join(header.arms)),
                ("locales", ", ".join(header.locales)),
                ("cases (test split)", str(header.case_count)),
                ("wall clock", f"{header.duration_s:.1f} s"),
                ("hardware", header.hardware),
                ("pricing table", header.pricing_id or "n/a"),
                ("deterministic digest", f"`{digest}`"),
            ],
        ),
        "",
        "Every cell below carries `n` and a 95% interval. `method` says how the interval was "
        "computed: `wilson` for a proportion, `bootstrap` for a mean (B resamples, §10.4), "
        "`none` for a distribution summary such as p95 latency.",
        "",
        f"Judge note: {header.judge.note or 'n/a'}.",
        "",
    ]
    if extra_notes:
        lines.extend(["Run notes:", "", *[f"* {note}" for note in extra_notes], ""])

    if header.replay:
        lines.extend(
            [
                "Every section below is marked **REPLAY-MODE**: a number produced by "
                "deterministic doubles is a diagnostic, never a live-model result.",
                "",
            ]
        )
    lines.extend(_retrieval_section(result))
    lines.extend(_answer_section(result))
    lines.extend(_escalation_section(result))
    lines.extend(_operations_section(result))
    lines.extend(_statistics_section(result))
    lines.extend(_targets_section(result))
    lines.extend(_limitations_section())
    lines.extend(
        [
            "## 9. Reproduction",
            "",
            "```bash",
            command,
            "```",
            "",
            "Same seed + same split + same corpus ⇒ identical `metrics.json` after removing "
            "timing fields (NFR-DET-02). The deterministic digest above is computed over "
            "exactly that projection.",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if header.replay:
        # FR-EVL-09: a replay run is labelled per section, not only in the banner.
        rendered = _mark_replay_sections(rendered)
    return rendered


def _mark_replay_sections(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and "REPLAY-MODE" not in line:
            out.append(f"{line} — REPLAY-MODE")
        else:
            out.append(line)
    return "\n".join(out)


def _retrieval_section(result: RunResult) -> list[str]:
    header = result.header
    lines = ["## 2. Retrieval", ""]
    cells = [
        cell
        for cell in result.cells
        if "@" in cell.name
        and cell.name.split("@")[0]
        in {
            "recall_any",
            "recall_all",
            "nDCG",
            "MRR",
        }
    ]
    if not cells:
        return [*lines, "_No retrieval rows: the selection carried no retrieval labels._", ""]
    for arm in header.arms:
        grouped = _group(cells, arm)
        if not grouped:
            continue
        lines.append(f"### 2.1 Arm `{arm}` — macro over locales")
        lines.append("")
        rows = []
        for name in sorted(grouped):
            macro = next((cell for cell in grouped[name] if cell.locale == "macro"), None)
            if macro is None:
                continue
            rows.append(
                [
                    name,
                    _cell(macro.value),
                    f"[{_cell(macro.ci_low)}, {_cell(macro.ci_high)}]",
                    str(macro.n),
                    macro.method,
                ]
            )
        lines.append(_table(("metric", "value", "95% CI", "n", "method"), rows))
        lines.append("")
        lines.append(f"### 2.2 Arm `{arm}` — per locale (§6.3 step 4, NFR-ML-01)")
        lines.append("")
        rows = []
        for name in sorted(grouped):
            per_locale = [cell for cell in _sort_locales(grouped[name]) if cell.locale != "macro"]
            if not per_locale:
                continue
            for cell in per_locale:
                rows.append(
                    [
                        name,
                        cell.locale or "(all)",
                        _cell(cell.value),
                        f"[{_cell(cell.ci_low)}, {_cell(cell.ci_high)}]",
                        str(cell.n),
                    ]
                )
        lines.append(_table(("metric", "locale", "value", "95% CI", "n"), rows))
        lines.append("")
    return lines


def _answer_section(result: RunResult) -> list[str]:
    header = result.header
    names = {
        "groundedness_verifier",
        "groundedness_judged",
        "groundedness_claim_pooled",
        "citation_precision",
        "citation_recall",
        "rubric_score",
        "rubric_pass_rate",
        "phantom_citations",
        "unsupported_displayed_claims",
        "language_agreement",
        "answer_accuracy_by_category",
    }
    lines = ["## 3. Answer quality", ""]
    rows = []
    for arm in header.arms:
        for cell in result.cells:
            if cell.arm != arm or cell.name not in names:
                continue
            label = f"{arm} / {cell.name}" + (f" [{cell.category}]" if cell.category else "")
            rows.append(
                [
                    label,
                    _cell(cell.value),
                    f"[{_cell(cell.ci_low)}, {_cell(cell.ci_high)}]",
                    str(cell.n),
                    cell.method,
                ]
            )
    if not rows:
        return [*lines, "_No answered turns in this selection._", ""]
    lines.append(_table(("arm / metric", "value", "95% CI", "n", "method"), rows))
    lines.append("")
    lines.append(
        "`phantom_citations` and `unsupported_displayed_claims` are structural counts "
        "(F-4/F-5): they are reported as a rate over answered turns and must be 0."
    )
    lines.append("")
    return lines


def _escalation_section(result: RunResult) -> list[str]:
    header = result.header
    names = {
        "realised_risk",
        "coverage",
        "must_escalate_recall",
        "escalation_precision",
        "realised_risk_by_bucket",
        "coverage_by_bucket",
    }
    lines = ["## 4. Escalation and calibration", ""]
    rows = []
    for arm in header.arms:
        for cell in result.cells:
            if cell.arm != arm or cell.name not in names:
                continue
            label = f"{arm} / {cell.name}" + (f" [{cell.category}]" if cell.category else "")
            rows.append(
                [
                    label,
                    _cell(cell.value),
                    f"[{_cell(cell.ci_low)}, {_cell(cell.ci_high)}]",
                    str(cell.n),
                    cell.note or "",
                ]
            )
    if rows:
        lines.append(_table(("arm / metric", "value", "95% CI", "n", "note"), rows))
        lines.append("")
    reliability = result.reliability
    if reliability:
        lines.append("### 4.1 Reliability (E-5/E-6 diagnostic)")
        lines.append("")
        lines.append(
            _table(
                ("quantity", "value"),
                [
                    ("n", str(reliability.get("n", 0))),
                    ("ECE equal-width", _cell(reliability.get("ece_equal_width"))),
                    ("ECE equal-mass", _cell(reliability.get("ece_equal_mass"))),
                    ("Brier", _cell(reliability.get("brier"))),
                    ("AURC", _cell(reliability.get("aurc"))),
                    ("mean confidence", _cell(reliability.get("mean_confidence"))),
                    ("accuracy", _cell(reliability.get("accuracy"))),
                    ("coverage at target risk", _cell(reliability.get("coverage_at_target"))),
                    ("risk at target", _cell(reliability.get("risk_at_target"))),
                ],
            )
        )
        lines.append("")
        lines.append(
            "`p̂` here is the packed evidence score, not a conformal calibrator (t8 owns "
            "conformal prediction), so E-5 is a diagnostic and no risk guarantee is claimed."
        )
        lines.append("")
    return lines


def _operations_section(result: RunResult) -> list[str]:
    header = result.header
    prefixes = ("latency_", "usd_per_answer", "tokens_", "cache_hit_rate")
    lines = ["## 5. Operations", ""]
    rows = []
    for arm in header.arms:
        for cell in result.cells:
            if cell.arm != arm or not cell.name.startswith(prefixes):
                continue
            rows.append(
                [
                    f"{arm} / {cell.name}",
                    _cell(cell.value, 4),
                    f"[{_cell(cell.ci_low, 4)}, {_cell(cell.ci_high, 4)}]",
                    str(cell.n),
                    cell.method,
                ]
            )
    if not rows:
        return [*lines, "_No operational rows._", ""]
    lines.append(_table(("arm / metric", "value", "bracket", "n", "method"), rows))
    lines.append("")
    lines.append(
        "Latency is measured in-process around the pipeline call, so it is a regression "
        "guard (NFR-LAT-03) and not a server-side latency claim (NFR-LAT-01). The bracket "
        "for a percentile is the p(q-5)..p(q+5) spread, not a confidence interval."
    )
    lines.append("")
    return lines


def _statistics_section(result: RunResult) -> list[str]:
    lines = ["## 6. Statistics: paired bootstrap and McNemar", ""]
    if not result.comparisons:
        return [
            *lines,
            "_A single arm was requested; no arm-to-arm comparison is possible. Run "
            "`--all` or the default multi-arm configuration to compare arms._",
            "",
        ]
    rows = []
    for comparison in result.comparisons:
        rows.append(
            [
                comparison.metric,
                f"{comparison.arm_a} − {comparison.arm_b}",
                str(comparison.n),
                _cell(comparison.mean_a),
                _cell(comparison.mean_b),
                _cell(comparison.diff),
                f"[{_cell(comparison.ci_low)}, {_cell(comparison.ci_high)}]",
                f"b={comparison.mcnemar_b}, c={comparison.mcnemar_c}",
                f"{_cell(comparison.mcnemar_p, 4)} ({comparison.mcnemar_method})",
                "yes" if comparison.significant else "no",
            ]
        )
    lines.append(
        _table(
            (
                "metric",
                "arms",
                "n",
                "mean A",
                "mean B",
                "Δ",
                "Δ 95% CI (paired bootstrap)",
                "McNemar",
                "p (method)",
                "CI excludes 0",
            ),
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Paired bootstrap resamples **queries**, not arms (B = "
        f"{result.comparisons[0].bootstrap}); McNemar uses the χ² approximation with the "
        "exact-binomial fallback when b + c < 25. A difference whose interval contains 0 is "
        "noise and is reported as such (§6.3 step 5)."
    )
    lines.append("")
    return lines


def _targets_section(result: RunResult) -> list[str]:
    lines = ["## 7. Targets", ""]
    if not result.targets:
        return [*lines, "_No targets evaluated._", ""]
    rows = []
    for target in result.targets:
        if target.passed is None:
            verdict = "n/a"
        elif target.gated:
            verdict = "pass" if target.passed else "FAIL"
        else:
            verdict = "diagnostic"
        rows.append(
            [
                target.target_id,
                target.description,
                target.threshold,
                _cell(target.value),
                verdict,
                target.note or "",
            ]
        )
    lines.append(_table(("id", "target", "threshold", "measured", "verdict", "note"), rows))
    lines.append("")
    lines.append(
        "`diagnostic` means the run cannot speak to the gate: a replay-mode provider, a "
        "judge without human calibration labels, or a layer owned by another task. The number "
        "is printed anyway — hiding it would be the dishonesty §10.7 forbids."
    )
    lines.append("")
    return lines


def _limitations_section() -> list[str]:
    lines = ["## 8. Limitations (normative, §10.7)", ""]
    lines.extend(f"{index}. {item}" for index, item in enumerate(LIMITATIONS, start=1))
    lines.append("")
    return lines


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_retrieval(result: RunResult, directory: Path) -> Path | None:
    plottable = [
        cell
        for cell in result.cells
        if cell.name.startswith("recall_any@") and cell.locale == "macro" and cell.n > 0
    ]
    if not plottable:
        return None
    plt = _pyplot()
    names = sorted({cell.name for cell in plottable}, key=lambda name: int(name.split("@")[1]))
    arms = sorted({cell.arm for cell in plottable})
    values = {
        (arm, name): next(
            (cell.value for cell in plottable if cell.arm == arm and cell.name == name), 0.0
        )
        for arm in arms
        for name in names
    }
    width = 0.8 / max(1, len(arms))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, arm in enumerate(arms):
        positions = [position + index * width for position in range(len(names))]
        axis.bar(
            positions,
            [values[(arm, name)] for name in names],
            width=width,
            label=arm,
        )
    axis.set_xticks([position + width * (len(arms) - 1) / 2 for position in range(len(names))])
    axis.set_xticklabels(names)
    axis.set_ylabel("macro recall_any")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Retrieval recall_any@k by arm (macro over locales)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    target = directory / "plots" / "retrieval_recall.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return target


def _plot_reliability(result: RunResult, directory: Path) -> Path | None:
    reliability = result.reliability
    bins = reliability.get("bins_equal_width") or []
    if not bins:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    axis.plot(
        [entry["mean_confidence"] for entry in bins],
        [entry["accuracy"] for entry in bins],
        marker="o",
        label="equal-width bins",
    )
    mass = reliability.get("bins_equal_mass") or []
    if mass:
        axis.plot(
            [entry["mean_confidence"] for entry in mass],
            [entry["accuracy"] for entry in mass],
            marker="s",
            linestyle=":",
            label="equal-mass bins",
        )
    axis.set_xlabel("mean confidence (evidence score)")
    axis.set_ylabel("accuracy")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Reliability diagram (E-5 diagnostic)")
    axis.legend()
    axis.grid(alpha=0.3)
    target = directory / "plots" / "calibration_reliability.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return target


def _plot_risk_coverage(result: RunResult, directory: Path) -> Path | None:
    curve = result.reliability.get("curve") or []
    if len(curve) < 2:
        return None
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(
        [point["coverage"] for point in curve],
        [point["risk"] for point in curve],
        label="risk–coverage (evidence score)",
    )
    axis.set_xlabel("coverage (fraction auto-answered)")
    axis.set_ylabel("realised risk")
    axis.set_title("Risk–coverage curve (E-6 diagnostic)")
    axis.set_xlim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    target = directory / "plots" / "risk_coverage.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return target


def _plot_latency(result: RunResult, directory: Path) -> Path | None:
    rows = [
        cell
        for cell in result.cells
        if cell.name.startswith("latency_") and cell.name.endswith(("_p50_ms", "_p95_ms"))
    ]
    if not rows:
        return None
    plt = _pyplot()
    arms = sorted({cell.arm for cell in rows})
    stages = sorted(
        {
            cell.name.removeprefix("latency_").removesuffix("_p50_ms").removesuffix("_p95_ms")
            for cell in rows
        }
    )
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, suffix in enumerate(("p50", "p95")):
        values = []
        for stage in stages:
            values.append(
                next(
                    (
                        cell.value
                        for cell in rows
                        if cell.name == f"latency_{stage}_{suffix}_ms" and cell.arm == arms[0]
                    ),
                    0.0,
                )
            )
        axis.bar(
            [position + index * width for position in range(len(stages))],
            values,
            width=width,
            label=f"{suffix} ({arms[0]})",
        )
    axis.set_xticks([position + width / 2 for position in range(len(stages))])
    axis.set_xticklabels(stages)
    axis.set_ylabel("milliseconds")
    axis.set_title(f"Stage latency in-process ({arms[0]})")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    target = directory / "plots" / "latency.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return target


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def require_statistics(result: RunResult) -> None:
    """Fail loudly when a report would print numbers without the statistics behind them."""
    if len(result.header.arms) > 1 and not result.comparisons:
        raise ReportError(
            "a multi-arm run produced no comparisons: the report would have no paired "
            "bootstrap or McNemar output (FR-EVL-07)",
        )
    for cell in result.cells:
        if cell.method != "none" and (cell.n == 0 or cell.ci_low > cell.ci_high):
            raise ReportError(
                f"metric {cell.name!r} for arm {cell.arm!r} has n={cell.n} and interval "
                f"[{cell.ci_low}, {cell.ci_high}]: refusing to print it (FR-EVL-07)",
            )
    if result.header.replay and not any("REPLAY" in note.upper() for note in result.header.notes):
        raise ReportError("a replay run must carry a REPLAY note in the header (§10.7 item 9)")


def write_report(
    result: RunResult,
    directory: str | Path,
    *,
    run_id: str,
    command: str,
    digest: str,
    latest_root: str | Path | None = None,
    extra_notes: Sequence[str] = (),
) -> ReportPaths:
    """Write markdown, JSON, CSV, turns and plots; maintain the stable ``latest`` pointer."""
    require_statistics(result)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    markdown = target / "report.md"
    markdown.write_text(
        render_markdown(
            result, run_id=run_id, command=command, digest=digest, extra_notes=extra_notes
        ),
        encoding="utf-8",
    )
    metrics = target / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "header": result.header.model_dump(mode="json"),
                "cells": [cell.model_dump(mode="json") for cell in result.cells],
                "comparisons": [item.model_dump(mode="json") for item in result.comparisons],
                "targets": [item.model_dump(mode="json") for item in result.targets],
                "reliability": result.reliability,
                "limitations": list(result.limitations),
                "digest": digest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    turns = target / "turns.jsonl"
    turns.write_text(
        "\n".join(
            json.dumps(turn.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for turn in result.turns
        )
        + "\n",
        encoding="utf-8",
    )
    cells_csv = target / "cells.csv"
    with cells_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "value",
                "ci_low",
                "ci_high",
                "n",
                "method",
                "arm",
                "locale",
                "category",
                "note",
            ]
        )
        for cell in result.cells:
            writer.writerow(
                [
                    cell.name,
                    cell.value,
                    cell.ci_low,
                    cell.ci_high,
                    cell.n,
                    cell.method,
                    cell.arm,
                    cell.locale,
                    cell.category,
                    cell.note,
                ]
            )
    plots = [
        path
        for path in (
            _plot_retrieval(result, target),
            _plot_reliability(result, target),
            _plot_risk_coverage(result, target),
            _plot_latency(result, target),
        )
        if path is not None
    ]
    latest = _write_latest_pointer(
        directory=target,
        run_id=run_id,
        latest_root=Path(latest_root) if latest_root is not None else target.parent,
    )
    return ReportPaths(
        directory=target,
        markdown=markdown,
        metrics=metrics,
        turns=turns,
        cells_csv=cells_csv,
        plots=plots,
        latest=latest,
    )


def _write_latest_pointer(*, directory: Path, run_id: str, latest_root: Path) -> Path | None:
    """Write ``latest.json`` and, when the OS allows it, a ``latest`` symlink to the run."""
    latest_root = latest_root.resolve()
    directory = directory.resolve()
    latest_root.mkdir(parents=True, exist_ok=True)
    pointer = latest_root / "latest.json"
    try:
        report = (directory / "report.md").relative_to(latest_root).as_posix()
    except ValueError:  # the run directory lives outside the reports root
        report = str(directory / "report.md")
    payload = {"run_id": run_id, "directory": directory.name, "report": report}
    pointer.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    link = latest_root / "latest"
    try:
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.exists():
            return pointer
        link.symlink_to(directory.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        return pointer
    return pointer


def comparison_table(comparisons: Sequence[StatComparison]) -> str:
    """Compact comparison table (used by tests and by the CLI's stdout summary)."""
    return _table(
        ("metric", "arm A", "arm B", "n", "Δ", "95% CI", "p"),
        [
            [
                item.metric,
                item.arm_a,
                item.arm_b,
                str(item.n),
                _cell(item.diff),
                f"[{_cell(item.ci_low)}, {_cell(item.ci_high)}]",
                _cell(item.mcnemar_p, 4),
            ]
            for item in comparisons
        ],
    )


def target_table(targets: Sequence[TargetCheck]) -> str:
    return _table(
        ("id", "target", "threshold", "measured", "verdict"),
        [
            [
                item.target_id,
                item.description,
                item.threshold,
                _cell(item.value),
                "n/a"
                if item.passed is None
                else ("pass" if item.passed else "FAIL")
                if item.gated
                else "diagnostic",
            ]
            for item in targets
        ],
    )


__all__ = [
    "LIMITATIONS",
    "REPLAY_BANNER",
    "ReportPaths",
    "comparison_table",
    "render_markdown",
    "require_statistics",
    "target_table",
    "write_report",
]
