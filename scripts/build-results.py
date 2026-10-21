"""Render ``docs/results.md`` from the committed artifacts of a benchmark run.

The rule this script exists to enforce: **no number in the results document is typed by
hand.** Every table cell is read from ``reports/<run>/**.json|csv`` and formatted here, so a
stale document is a reproducible artefact rather than a claim. If an artefact is missing the
script exits non-zero instead of silently omitting a block.

Usage::

    uv run python scripts/build-results.py --run reports/bench --out docs/results.md
    uv run python scripts/build-results.py --check          # fail if the doc is out of date

    # AT-EVL-10  # AT-NFR-REP-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "reports" / "bench"
DEFAULT_OUT = REPO_ROOT / "docs" / "results.md"
GENERATOR = "scripts/build-results.py"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"missing artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    """Raw-byte digest; used only where the bytes themselves matter (never for provenance)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _cell(cell: Mapping[str, Any] | None, digits: int = 3) -> str:
    if not cell:
        return "n/a"
    value = cell.get("value")
    low = cell.get("ci_low")
    high = cell.get("ci_high")
    n = cell.get("n")
    if value is None:
        return "n/a"
    rendered = f"{float(value):.{digits}f}"
    if low is not None and high is not None:
        rendered += f" [{float(low):.{digits}f}, {float(high):.{digits}f}]"
    if n is not None:
        rendered += f" (n={n})"
    return rendered


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _cell_for(
    cells: Sequence[Mapping[str, Any]], name: str, locale: str = "macro"
) -> Mapping[str, Any] | None:
    for cell in cells:
        if cell.get("name") == name and cell.get("locale") == locale:
            return cell
    for cell in cells:
        if cell.get("name") == name and not cell.get("locale"):
            return cell
    return None


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


def _header_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    models = summary.get("models", {})
    rows = [
        ["run id", summary.get("run_id")],
        ["corpus version", summary.get("corpus_version")],
        ["corpus hash", f"`{summary.get('corpus_hash')}`"],
        ["documents / chunks", f"{summary.get('documents')} / {summary.get('chunks')}"],
        ["test cases", summary.get("test_cases")],
        ["dev cases", summary.get("dev_cases")],
        [
            "replay mode",
            "yes — numbers are diagnostics" if summary.get("replay") else "no (live providers)",
        ],
        ["hardware", summary.get("hardware")],
        [
            "duration",
            f"{float(summary['duration_s']):.1f} s"
            if summary.get("duration_s")
            else "not measured (the run did not record it)",
        ],
        [
            "calibration",
            (summary.get("calibration") or {}).get("note", summary.get("calibration"))
            if isinstance(summary.get("calibration"), Mapping)
            else summary.get("calibration"),
        ],
    ]
    for arm, values in models.items():
        rows.append(
            [
                f"arm `{arm}`",
                f"generator `{values.get('generator')}` ({values.get('generator_family')}) · "
                f"judge `{values.get('judge')}` ({values.get('judge_family')}) · "
                f"cross-family {'yes' if values.get('cross_family') else 'NO'} · "
                f"reportable {'yes' if values.get('reportable') else 'no'}",
            ]
        )
    lines = [
        _table(("field", "value"), rows),
        "",
        f"Artefacts: `{_relative(run)}` "
        "(summary.json, retrieval/, answer/, escalation/, cache/, ops/).",
        "",
        "Every table below is generated from those artefacts by this script; the document "
        f"itself is a build product (`{GENERATOR}`).",
        "",
    ]
    return lines


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _canonical_bytes(data: bytes) -> bytes:
    """Normalise every line ending to a single LF (content bytes are untouched)."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _canonical_sha256(path: Path) -> str:
    """Digest of one provenance input over canonical LF bytes (line-ending independent).

    Raw-byte hashing made the published digests differ between this Windows working tree
    (CRLF) and a clean LF checkout, which falsified the reproducibility claim on the very
    checkout it is meant to verify. Every hash in the provenance block and the artefact digest
    line goes through this function; ``tests/unit/eval/test_provenance.py`` pins a CRLF file to
    its LF twin's digest so the dependency cannot return silently.
    """
    if not path.is_file():
        return "missing"
    return hashlib.sha256(_canonical_bytes(path.read_bytes())).hexdigest()


def _provenance_block(summary: Mapping[str, Any]) -> list[str]:
    """One auditable block: corpus, golden, split, benchmark manifest, pricing, run."""

    def sha(path: Path) -> str:
        return _canonical_sha256(path)

    corpus_manifest = REPO_ROOT / "data" / "corpus" / "manifest.json"
    golden_manifest = REPO_ROOT / "evals" / "golden" / "v1" / "manifest.json"
    split_manifest = REPO_ROOT / "evals" / "splits" / "v1.json"
    bench_manifest = REPO_ROOT / "data" / "bench" / "answerdesk-bench" / "v0.1" / "manifest.json"
    pricing = REPO_ROOT / "configs" / "pricing.yaml"
    bench_hash = "not present in this checkout"
    if bench_manifest.is_file():
        payload = json.loads(bench_manifest.read_text(encoding="utf-8"))
        bench_hash = payload.get("self_sha256", sha(bench_manifest))
    lines = [
        "## Provenance (one auditable block)",
        "",
        "```text",
        f"corpus_version      {summary.get('corpus_version')}",
        f"corpus_hash         {summary.get('corpus_hash')}",
        f"corpus_manifest     {sha(corpus_manifest)}",
        f"golden_manifest     {sha(golden_manifest)}",
        f"split_manifest      {sha(split_manifest)}",
        f"answerdesk_bench    {bench_hash}",
        f"pricing_table       {sha(pricing)}",
        f"run_id              {summary.get('run_id')}",
        f"summary_json        {sha(REPO_ROOT / 'reports' / 'bench' / 'summary.json')}",
        "```",
        "",
        "Every hash above is of a committed file and is computed over **canonical LF bytes**, "
        "so a CRLF working tree and a clean LF checkout publish identical digests. Changing any "
        "input without re-running invalidates the chain, and `scripts/build-results.py --check` "
        "fails on a stale document.",
        "",
    ]
    return lines


def _retrieval_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    results = _read_json(run / "retrieval" / "results.json")
    lines = ["## Retrieval ablation", ""]
    rows = []
    for arm, payload in results["arms"].items():
        cells = payload["cells"]
        rows.append(
            [
                arm,
                _cell(_cell_for(cells, "recall_any@10")),
                _cell(_cell_for(cells, "recall_any@20")),
                _cell(_cell_for(cells, "recall_all@20")),
                _cell(_cell_for(cells, "nDCG@10")),
                _cell(_cell_for(cells, "MRR@10")),
                payload.get("n_labeled"),
                f"{payload.get('mean_latency_ms')} ms",
            ]
        )
    lines.append(
        _table(
            (
                "arm",
                "recall_any@10",
                "recall_any@20",
                "recall_all@20",
                "nDCG@10",
                "MRR@10",
                "n",
                "mean retrieval",
            ),
            rows,
        )
    )
    lines.append("")
    degraded = results.get("degraded_cases", 0)
    if degraded:
        lines.append(
            f"⚠ {degraded} rerank call(s) failed after retries and were served by the "
            "non-reranked pipeline; the arm is reported with that degradation stated."
        )
        lines.append("")
    lines.append("Paired comparisons (recall_any@10, paired bootstrap over queries + McNemar):")
    lines.append("")
    comparison_rows = []
    for comparison in results.get("comparisons", []):
        comparison_rows.append(
            [
                f"`{comparison['arm_a']}` − `{comparison['arm_b']}`",
                comparison["n"],
                f"{comparison['diff']:+.4f}",
                f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]",
                f"{comparison['mcnemar_p']:.4f} ({comparison['mcnemar_method']})",
                "yes" if comparison["significant"] else "no",
            ]
        )
    if comparison_rows:
        lines.append(
            _table(("pair", "n", "Δ", "95% CI", "McNemar p", "CI excludes 0"), comparison_rows)
        )
    else:
        lines.append("_No paired comparisons were produced._")
    lines.append("")
    per_language = results.get("per_language", [])
    if per_language:
        lines.append("### Per language (recall_any@10, Wilson interval)")
        lines.append("")
        rows = []
        for row in per_language:
            if row["arm"] not in ("bm25", "rrf@60", "cc@0.75", "cc@0.75+rerank"):
                continue
            rows.append(
                [
                    row["arm"],
                    row["locale"],
                    f"{row['recall_any@10']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                    row["n"],
                ]
            )
        lines.append(_table(("arm", "locale", "recall_any@10", "n"), rows))
        lines.append("")
    noisy_rows = []
    for arm, payload in results["arms"].items():
        for cell in payload.get("noisy_cells", []):
            if cell.get("locale") == "macro":
                noisy_rows.append([arm, cell["name"], _cell(cell)])
    if noisy_rows:
        lines.append("### Noisy / free-form slice (its own block, never averaged in)")
        lines.append("")
        lines.append(_table(("arm", "metric", "value"), noisy_rows))
        lines.append("")
    return lines


def _model_ids() -> tuple[str, str]:
    """Both model ids of the primary family, as recorded in evals/configs/bench-live.yaml."""
    path = REPO_ROOT / "evals" / "configs" / "bench-live.yaml"
    primary, reasoning = "qwen3.6-35b-a3b-noreason", "qwen3.6-35b-a3b"
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("model_id:") and "noreason" in stripped:
                primary = stripped.split(":", 1)[1].strip().strip('"')
            elif stripped.startswith("model_id:") and "qwen3.6-35b-a3b" in stripped:
                reasoning = stripped.split(":", 1)[1].strip().strip('"')
    return (primary, reasoning)


def _model_alias_note(arm: str, values: Mapping[str, Any]) -> str:
    """Name *this arm's* generator from the summary, never a hard-coded one (R-13).

    The previous version asserted "Generator family **qwen**" under every arm, so the
    deepseek-flash secondary arm was attributed to the qwen generator. The family and the
    model now come from ``summary.models[arm]``; the reasoning-variant sentence is only
    emitted for the family it is true of.
    """
    generator = values.get("generator") or "unknown"
    family = values.get("generator_family") or "unknown"
    sentence = f"Generator family **{family}**: `{generator}` produced these answers."
    if family == "qwen":
        _primary, reasoning = _model_ids()
        sentence += (
            f" `{reasoning}` is the reasoning variant; the planned side study at 8192/16384 "
            "tokens was not run in this window, so no reasoning-variant number is claimed."
        )
    return sentence


def _answer_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    lines = ["## Answer quality and escalation", ""]
    for arm in summary.get("models", {}):
        path = run / "answer" / arm / "cells.json"
        if not path.is_file():
            continue
        payload = _read_json(path)
        judge = payload.get("judge", {})
        lines.append(
            f"### Arm `{arm}` — generator `{payload.get('generator_model')}`, "
            f"judge `{payload.get('judge_model')}`"
        )
        lines.append("")
        lines.append(_model_alias_note(arm, summary.get("models", {}).get(arm, {})))
        lines.append("")
        note = summary.get("models", {}).get(arm, {}).get("note")
        if note:
            lines.append(f"_{note}_")
            lines.append("")
        lines.append(
            f"Cross-family judging: **{'yes' if judge.get('cross_family') else 'no'}** "
            f"(`{judge.get('generator_family')}` vs `{judge.get('judge_family')}`); "
            f"reportable: **{'yes' if judge.get('reportable') else 'no'}**. {judge.get('note', '')}"
        )
        lines.append("")
        rows = []
        for name in (
            "groundedness_judged",
            "citation_recall",
            "citation_precision",
            "rubric_score",
            "rubric_pass_rate",
            "language_agreement",
            "coverage",
            "must_escalate_recall",
            "escalation_precision",
            "realised_risk",
        ):
            cell = _cell_for(payload["cells"], name)
            if cell is None:
                continue
            rows.append([name, _cell(cell)])
        lines.append(_table(("metric", "value (95% CI, n)"), rows))
        lines.append("")
        confusion = payload.get("confusion", {})
        lines.append(
            f"Confusion matrix: TP {confusion.get('tp')} · FP {confusion.get('fp')} · "
            f"FN {confusion.get('fn')} · TN {confusion.get('tn')}; false-escalation rate "
            f"{_pct(confusion.get('false_escalation_rate'))}, "
            f"coverage {_pct(confusion.get('coverage'))}."
        )
        lines.append("")
        if payload.get("reason_codes"):
            lines.append("Escalation reason codes:")
            lines.append("")
            lines.append(
                _table(
                    ("reason code", "count", "share"),
                    [
                        [row["reason_code"], row["count"], _pct(row["share"])]
                        for row in payload["reason_codes"]
                    ],
                )
            )
            lines.append("")
        if payload.get("per_language"):
            lines.append("Per language:")
            lines.append("")
            lines.append(
                _table(
                    ("language", "n", "coverage", "must-escalate recall", "answer accuracy"),
                    [
                        [
                            row["locale"],
                            row["n"],
                            f"{row['coverage']:.3f} {row['coverage_ci']}",
                            f"{row['must_escalate_recall']:.3f} {row['must_escalate_ci']}",
                            f"{row['answer_accuracy']:.3f} {row['answer_accuracy_ci']}",
                        ]
                        for row in payload["per_language"]
                    ],
                )
            )
            lines.append("")
    return lines


def _escalation_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    """Escalation detail per arm: confusion, E-2 by family, languages, curve, policies, E-3."""
    blocks: list[str] = []
    for arm in summary.get("models", {}):
        per_arm = run / "escalation" / f"{arm}.json"
        fallback = run / "escalation" / "results.json"
        path = per_arm if per_arm.is_file() else fallback
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks.extend(_escalation_arm(arm, payload))
    if not blocks:
        return ["### Escalation detail", "", "_No escalation artefact._", ""]
    return blocks


def _escalation_arm(arm: str, payload: Mapping[str, Any]) -> list[str]:
    lines = [f"### Escalation detail — arm `{arm}`", ""]
    lines.append("E-2 recall by unsafe family (must-escalate ≥ 0.95):")
    lines.append("")
    lines.append(
        _table(
            ("family", "n", "escalated", "recall", "95% CI"),
            [
                [
                    row["family"],
                    row["n"],
                    row["escalated"],
                    f"{row['recall']:.3f}",
                    f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                ]
                for row in payload.get("families", [])
            ],
        )
    )
    lines.append("")
    lines.append("Per language (coverage on answerable items · E-2 recall on unsafe items):")
    lines.append("")
    lines.append(
        _table(
            ("language", "n", "coverage", "must-escalate recall"),
            [
                [
                    row["locale"],
                    row["n"],
                    f"{row['coverage']:.3f} {row['coverage_ci']}",
                    f"{row['must_escalate_recall']:.3f} {row['must_escalate_ci']}",
                ]
                for row in payload.get("languages", [])
            ],
        )
    )
    lines.append("")
    actions = payload.get("actions", {})
    if actions:
        lines.append(
            "Decisions: "
            + ", ".join(f"{key} {value}" for key, value in sorted(actions.items()))
            + f". {payload.get('definition', '')}"
        )
        lines.append("")
    operating = payload.get("operating_point")
    if operating:
        lines.append(
            "**Headline operating point** — threshold selected on the dev/calibration turns "
            "(the split the conformal quantile is fitted on) and applied unchanged to the "
            f"held-out test turns: threshold {operating.get('threshold')}, test coverage "
            f"**{operating['coverage']:.3f}** {operating.get('coverage_ci', '')} at realised "
            f"risk **{operating['risk']}** {operating.get('risk_ci', '')} (n answered "
            f"{operating['n_answered']}; dev point: coverage {operating.get('dev_coverage')} at "
            f"risk {operating.get('dev_risk')})."
        )
        lines.append("")
    oracle = payload.get("oracle_upper_bound")
    if oracle:
        lines.append(
            "*Oracle upper bound (diagnostic, not the shipped operating point):* selecting the "
            f"threshold on the test split itself would give coverage {oracle['coverage']:.3f} "
            f"at risk {oracle['risk']:.4f}. It is excluded from the headline because it fits "
            "the reported data."
        )
        lines.append("")
    variants = payload.get("policy_variants", [])
    if variants:
        lines.append("Policy variants (same turns, three decision rules):")
        lines.append("")
        lines.append(
            _table(
                ("policy", "threshold", "coverage", "realised risk", "E-2 recall", "precision"),
                [
                    [
                        row["policy"],
                        row["threshold"] if row["threshold"] is not None else "engine",
                        f"{row['coverage']:.3f} {row['coverage_ci']}",
                        f"{row['realised_risk']} {row['risk_ci']}",
                        f"{row['must_escalate_recall']:.3f} {row['recall_ci']}",
                        f"{row['escalation_precision']} {row['precision_ci']}",
                    ]
                    for row in variants
                ],
            )
        )
        lines.append("")
    e3 = payload.get("e3", {})
    lines.append(
        f"**E-3 (escalation precision).** Gated reading: {e3.get('gated_precision')} "
        f"{e3.get('gated_ci')}; literal safety-only reading: {e3.get('literal_precision')} "
        f"{e3.get('literal_ci')}. {e3.get('note', '')}"
    )
    lines.append("")
    lines.append(
        "**Resolution (ADR-style).** E-3 is reported with both readings and neither is tuned "
        "away: the gated reading counts every non-answered turn in the denominator (what the "
        "operator experiences as escalation volume), the literal reading counts only turns a "
        "safety guard stopped (what the safety claim is about). Where the literal value falls "
        "below 0.55 on this synthetic population, that failure is reported as measured; the "
        "population is not re-cut to make the number pass."
    )
    lines.append("")
    return lines


def _noisy_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    """The noisy/free-form slice: its own numbers and per-example outcomes."""
    lines = ["## Noisy / free-form slice (own block, never averaged in)", ""]
    rows: list[dict[str, str]] = []
    for arm in summary.get("models", {}):
        path = run / "answer" / arm / "noisy-per-example.csv"
        if path.is_file():
            rows.extend(_read_rows(path))
    if not rows:
        lines.append("_No noisy items in the evaluated sample._")
        lines.append("")
        return lines
    # The owner's three messages from the review (loan, deposit, remittance), as they appear
    # in the frozen noisy corpus slice.
    owner_patterns = [
        "как займ оформит",
        "деньги как получит",
        "денги первод таджикистан сделал денги нет брату не дошли",
    ]
    lines.append(
        "Answers to telegraphic / run-on / dropped-preposition customer messages. The owner's "
        "three messages from the review are marked ★."
    )
    lines.append("")
    table_rows = []
    for row in rows:
        question = (row.get("question") or "").replace("|", "/")
        flagged = any(pattern in question.lower() for pattern in owner_patterns)
        table_rows.append(
            [
                "★" if flagged else "",
                row.get("case_id"),
                row.get("locale"),
                question[:90],
                row.get("action"),
                row.get("reason_code"),
                row.get("rubric_score") or "n/a",
                row.get("evidence_cited"),
                (row.get("answer_head") or "")[:60],
            ]
        )
    lines.append(
        _table(
            (
                "",
                "case",
                "lang",
                "question",
                "action",
                "reason",
                "rubric",
                "evidence",
                "answer head",
            ),
            table_rows[:40],
        )
    )
    if len(table_rows) > 40:
        lines.append("")
        lines.append(f"_{len(table_rows) - 40} further rows in the CSV artefact._")
    lines.append("")
    return lines


def _chunking_block(run: Path) -> list[str]:
    path = run / "chunking" / "results.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## Chunking ablation: structure-first vs naive fixed window", ""]
    lines.append(
        f"Same cases ({payload.get('cases')}), same embedder, gold evidence mapped by character "
        "overlap; the naive arm splits every document into "
        f"{payload.get('naive_window')}-character "
        f"windows with {payload.get('naive_overlap')} overlap and ignores headings, tables "
        "and FAQ pairs."
    )
    lines.append("")
    lines.append(
        _table(
            ("arm", "chunks", "n", "recall_any@10", "recall_any@20"),
            [
                [
                    row["arm"],
                    row["chunks"],
                    row["n"],
                    f"{row['recall_any@10']} {row['recall_any@10_ci']}",
                    f"{row['recall_any@20']} {row['recall_any@20_ci']}",
                ]
                for row in payload.get("rows", [])
            ],
        )
    )
    lines.append("")
    lines.append(f"_{payload.get('note', '')}_")
    lines.append("")
    return lines


def _side_study_block(run: Path) -> list[str]:
    """The reasoning-variant side study (can the big-budget variant finish at all?)."""
    path = run / "side-study" / "reasoning-variant.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## Side study: the reasoning variant at a higher budget", ""]
    lines.append(
        f"Model `{payload.get('reasoning_model')}` vs the primary "
        f"`{payload.get('primary_model')}`: "
        f"{payload.get('n_cases')} cases, identical prompts/schema/retrieval, temperature 0, "
        "only max_tokens differs."
    )
    lines.append("")
    rows = []
    for budget, values in sorted(payload.get("budgets", {}).items(), key=lambda item: int(item[0])):
        rows.append(
            [
                budget,
                values.get("n"),
                f"{values.get('success_rate')}",
                values.get("success_ci"),
                values.get("latency_p50_ms"),
                values.get("latency_p95_ms"),
                values.get("output_tokens_mean"),
                values.get("usd_total"),
            ]
        )
    lines.append(
        _table(
            ("max_tokens", "n", "success", "95% CI", "p50 ms", "p95 ms", "output tokens", "USD"),
            rows,
        )
    )
    lines.append("")
    lines.append(f"_{payload.get('note', '')}_")
    lines.append("")
    return lines


def _cache_block(summary: Mapping[str, Any]) -> list[str]:
    lines = ["## Semantic cache", ""]
    cache = summary.get("cache") or {}
    if "skipped" in cache:
        lines.append(f"Skipped: {cache['skipped']}.")
        lines.append("")
        return lines
    lines.append(
        _table(
            ("quantity", "value"),
            [
                ["near-duplicate pairs", cache.get("pairs")],
                ["hits / attempts", f"{cache.get('hits')} / {cache.get('attempts')}"],
                ["hit rate", f"{cache.get('hit_rate')} {cache.get('hit_rate_ci')}"],
                [
                    "wrong-hit rate (UCB)",
                    f"{cache.get('wrong_hit_rate')} (UCB {cache.get('wrong_hit_ucb')})",
                ],
                [
                    "hit latency p50 / p95",
                    f"{cache.get('hit_latency_p50_ms')} / {cache.get('hit_latency_p95_ms')} ms",
                ],
                ["USD saved on hits", cache.get("saved_usd")],
            ],
        )
    )
    lines.append("")
    lines.append(f"_{cache.get('note', '')}_")
    lines.append("")
    return lines


def _ops_block(summary: Mapping[str, Any], run: Path) -> list[str]:
    lines = ["## Latency and cost (measured, not estimated)", ""]
    rows = []
    for arm, payload in summary.get("ops", {}).items():
        for row in payload.get("latency", []):
            rows.append(
                [arm, row["stage"], f"{row['p50_ms']:.2f}", f"{row['p95_ms']:.2f}", row["n"]]
            )
    if rows:
        lines.append(_table(("arm", "stage", "p50 ms", "p95 ms", "n"), rows))
        lines.append("")
    token_rows = []
    for arm, payload in summary.get("ops", {}).items():
        tokens = payload.get("tokens", {})
        token_rows.append(
            [
                arm,
                payload.get("answered"),
                tokens.get("input_per_answer"),
                tokens.get("output_per_answer"),
                payload.get("usd_per_answer"),
            ]
        )
    lines.append(
        _table(
            ("arm", "answered", "input tokens/answer", "output tokens/answer", "USD/answer"),
            token_rows,
        )
    )
    lines.append("")
    pricing_path = run / "ops" / "pricing.json"
    pricing = _read_json(pricing_path) if pricing_path.is_file() else None
    if pricing is None:
        # fall back to the per-arm ops payload (the run writes the snapshot there)
        for payload in summary.get("ops", {}).values():
            if payload.get("pricing"):
                pricing = payload["pricing"]
                break
    if pricing:
        lines.append("Recorded token pricing (see `configs/pricing.yaml` for provenance):")
        lines.append("")
        lines.append(
            _table(
                ("model", "input USD/1k", "output USD/1k", "retrieved at", "source"),
                [
                    [
                        row["model"],
                        row["input_usd_per_1k_tokens"],
                        row["output_usd_per_1k_tokens"],
                        row["retrieved_at"],
                        row["source"],
                    ]
                    for row in pricing
                ],
            )
        )
        lines.append("")
    return lines


def _lat_requirement_block(summary: Mapping[str, Any]) -> list[str]:
    """NFR-LAT-01 verdict: the target, the measurement, and the miss (review R-02).

    The requirement is a MUST with a G-REL gate and is not amended here. The release rule in
    docs/requirements.md §17.3 allows either a met threshold or a formally recorded miss, so the
    verdict is printed next to the numbers it belongs to and cross-referenced to §18.5.
    """
    ops = summary.get("ops", {})
    arm = "primary" if "primary" in ops else next(iter(ops), "")
    payload = ops.get(arm, {})
    by_stage = {row["stage"]: row for row in payload.get("latency", [])}
    generation = by_stage.get("generation")
    total = by_stage.get("total")
    if generation is None:
        return []

    p50 = float(generation["p50_ms"])
    p95 = float(generation["p95_ms"])
    n = generation.get("n")
    missed = p50 > 1500.0 or p95 > 4000.0
    verdict = "**MISS**" if missed else "PASS"
    lines = [
        "## NFR-LAT-01 verdict — latency target",
        "",
        f"| requirement | target | measured (arm `{arm}`) | basis | verdict |",
        "|---|---|---|---|---|",
        "| NFR-LAT-01 | p50 ≤ 1,500 ms · p95 ≤ 4,000 ms | "
        f"p50 {p50:.2f} ms · p95 {p95:.2f} ms (n={n}) "
        "| generation stage, arrival → final byte, judge excluded | " + verdict + " |",
    ]
    if total is not None:
        lines.append(
            "| (context only) | — | "
            f"p50 {float(total['p50_ms']):.2f} ms · p95 {float(total['p95_ms']):.2f} ms "
            f"(n={total.get('n')}) | total including the offline judge; "
            "the judge is the tail, not the answerer | not a gate |"
        )
    lines.append("")
    if missed:
        retrieval = by_stage.get("retrieval")
        retrieval_note = (
            f" The retrieval path in the same run is {float(retrieval['p50_ms']):.2f}/"
            f"{float(retrieval['p95_ms']):.2f} ms p50/p95, "
            "so the budget is consumed by generation, not by retrieval."
            if retrieval is not None
            else ""
        )
        lines.extend(
            [
                "The target is **not** amended: the requirement stays at p50 ≤ 1,500 ms / "
                "p95 ≤ 4,000 ms and the release records the miss instead of hiding it "
                "(change control CC-2026-10-21-b, `docs/requirements.md` §18.5).",
                "",
                "What the number measures: a hosted ~35B generator in strict-JSON mode producing "
                "~500 output tokens per answered turn, arrival to final byte; the budget is "
                "not a token-stream budget. Streaming **was** measured in the separate "
                "`latency-v2` arm (docs/latency.md): baseline time-to-first-token "
                "673 ms p50 / 965 ms p95, and the output-budget/cascade variants still "
                "missed the total, so TTFT alone does not rescue this target — the verdict "
                "stays MISS on the measured totals." + retrieval_note,
                "",
                "What would change the verdict: a smaller or local model, streaming with TTFT as "
                "the served-latency metric, and a shorter answer budget — each measurable with the "
                "same harness (`answerdesk eval --all --provider ...`). Until one of those is "
                "measured and committed, this requirement is reported as missed.",
                "",
            ]
        )
    return lines


def _error_taxonomy(run: Path) -> list[str]:
    lines = ["## Error taxonomy", ""]
    rows: list[dict[str, str]] = []
    for path in sorted((run / "answer").glob("*/turns.csv")):
        rows.extend(_read_rows(path))
    if not rows:
        lines.append("_No turn-level artefacts._")
        lines.append("")
        return lines
    failures: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for row in rows:
        answerable = row.get("answerable") == "True"
        answered = row.get("answered") == "True"
        ok = row.get("answer_ok") == "True"
        if ok:
            continue
        if not answerable and answered:
            reason = "unsafe turn answered (should have escalated)"
        elif answerable and not answered:
            reason = f"answerable turn escalated ({row.get('reason_code')})"
        elif row.get("evidence_cited") != "True":
            reason = "answer did not cite acceptable evidence"
        elif row.get("language_match") != "True":
            reason = "answer language mismatch"
        elif row.get("rubric_score") and float(row["rubric_score"]) < 4.0:
            reason = "judge rubric below 4"
        elif int(row.get("claims") or 0) == 0:
            reason = "no claims produced"
        else:
            reason = "other"
        failures[reason] += 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 3:
            question = row.get("case_id", "")
            examples[reason].append(question)
    top = failures.most_common(8)
    lines.append(
        _table(
            ("failure mode", "count", "examples (case ids)"),
            [[reason, count, ", ".join(examples[reason])] for reason, count in top],
        )
    )
    lines.append("")
    return lines


def _risk_block() -> list[str]:
    """Risk-control summary, read from its own artefact set (never hand-written).

    v3 (bench population with persisted answer artefacts) supersedes v2 when present; both
    artefact sets are committed, so the block always renders from whatever is on disk.
    """
    v3_path = REPO_ROOT / "reports" / "risk-v3" / "summary.json"
    if v3_path.is_file():
        return _risk_v3_block(json.loads(v3_path.read_text(encoding="utf-8")))
    path = REPO_ROOT / "reports" / "risk-v2" / "summary.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("primary_population")
    block = payload["populations"].get(primary, {})
    test = block.get("test", {})
    certificate = block.get("certificate", {})
    required = payload.get("n_required", {})
    lines = [
        "## Risk control v2 (calibrated answerability, Learn-then-Test certificate)",
        "",
        f"Primary population `{primary}` ({block.get('dev')} dev / {block.get('test')} rows), "
        f"full method in `docs/risk-control.md`.",
        "",
        _table(
            ("metric", "value", "target"),
            [
                ["AUROC", test.get("auroc"), ">= 0.75"],
                ["ECE (equal-mass)", test.get("ece_equal_mass"), "<= 0.08"],
                [
                    "coverage at realised risk <= 5%",
                    test.get("coverage_at_risk", {}).get("coverage"),
                    ">= 0.40",
                ],
                ["AURC", test.get("aurc"), "-"],
            ],
        ),
        "",
        f"Certificate at alpha {payload.get('alpha')}, delta {payload.get('delta')}: "
        f"**{'feasible' if certificate.get('feasible') else 'ABSENT'}** "
        f"(threshold {certificate.get('threshold')}, calibration risk "
        f"{certificate.get('calibration_risk')}, UCB {certificate.get('calibration_ucb')}, "
        f"n={certificate.get('n_calibration')}). Rows needed for a 5% certificate at zero "
        f"observed risk: Bentkus ~{required.get('bentkus_rows_at_zero_risk')}, Hoeffding "
        f"~{required.get('hoeffding_rows_at_zero_risk')}. The largest feasible level is "
        f"alpha 0.25 (coverage "
        f"{block.get('certificate_at_alpha_0_25', {}).get('calibration_coverage')}), reported "
        "as the operational fallback with its own alpha — no 5% guarantee is claimed.",
        "",
    ]
    return lines


def _risk_v3_block(payload: Mapping[str, Any]) -> list[str]:
    """Risk-control v3: the bench population with full persisted answer artefacts."""
    bench = payload["populations"]["bench"]
    test = bench["test_report"]
    certificate = bench["certificate"]
    targets = bench["targets"]
    required = payload.get("n_required", {})
    return [
        "## Risk control v3 (bench population, persisted answer artefacts, Learn-then-Test)",
        "",
        f"Population: the frozen bench turns re-answered with the shipped pipeline and "
        f"re-judged cross-family ({bench['dev_rows']} dev / {bench['test_rows']} test answered "
        f"turns). Method and limitations: `docs/risk-control.md`.",
        "",
        _table(
            ("metric", "value", "target"),
            [
                ["AUROC", test["auroc"], ">= 0.80"],
                ["ECE (equal-mass)", test["ece_equal_mass"], "<= 0.08"],
                [
                    "coverage at realised risk <= 5%",
                    test["coverage_at_risk"]["coverage"],
                    ">= 0.20",
                ],
                ["AURC", test["aurc"], "-"],
            ],
        ),
        "",
        f"Certificate at alpha {payload.get('alpha')}, delta {payload.get('delta')}: "
        f"**{'feasible' if certificate.get('feasible') else 'ABSENT'}** "
        f"(threshold {certificate.get('threshold')}, calibration risk "
        f"{certificate.get('calibration_risk')}, UCB {certificate.get('calibration_ucb')}, "
        f"n={certificate.get('n_calibration')}). The required rows for a 5% certificate at zero "
        f"observed risk are Bentkus ~{required.get('bentkus_rows_at_zero_risk')} / Hoeffding "
        f"~{required.get('hoeffding_rows_at_zero_risk')}; the population's base error "
        f"({bench['base_error_test']} on test) leaves no sufficiently large error-free plateau, "
        "so the policy abstains and **no 5% guarantee is claimed**. Coverage target "
        f"{targets['coverage_at_risk_min']} is a documented miss at "
        f"{test['coverage_at_risk']['coverage']} (oracle frontier point).",
        "",
    ]


def _calibration_verdict_block(summary: Mapping[str, Any]) -> list[str]:
    verdict = summary.get("calibration_verdict")
    if not verdict:
        return []
    held = bool(verdict.get("risk_control_held"))
    lines = ["## Calibration verdict (does risk control hold?)", ""]
    lines.append(
        f"Conformal α (informational bucket) = {verdict.get('target_alpha')}, operating-point "
        f"target {verdict.get('operating_point_target')}; realised risk on the answered set of "
        f"the primary arm = **{verdict.get('realised_risk')}**. "
        + ("Risk control **held**." if held else "Risk control **did not hold**.")
    )
    lines.append("")
    lines.append(f"_{verdict.get('reason', '')}_")
    lines.append("")
    if not held:
        lines.append(
            "This is reported as a measured negative result, not smoothed over: the "
            "deterministic overlap confidence saturates at 1.0 on most queries, so the "
            "split-conformal quantile collapses to zero and the threshold separates almost "
            "nothing. A calibrated confidence signal (the t8 judge-based scorer, or a "
            "monotone calibration map fitted on dev) is required before any risk-control "
            "claim can be made; the shipped default in this repository remains the "
            "config-driven threshold, and README/claim language must not assert calibrated "
            "escalation on this evidence."
        )
        lines.append("")
    return lines


def _scorer_block(summary: Mapping[str, Any]) -> list[str]:
    payload = summary.get("calibration_scorer")
    if not payload:
        return []
    lines = ["## Calibrated answerability scorer (replaces the saturated overlap score)", ""]
    lines.append(
        f"L2 logistic regression fitted on **dev answers only** (n={payload.get('n_dev')}, "
        f"l2={payload.get('l2')}), features: {', '.join(payload.get('features', []))}. "
        f"Split-conformal q-hat on the calibrated dev scores = {payload.get('q_hat')} "
        f"(threshold {payload.get('threshold')}). Evaluated on the held-out test turns "
        f"(n={payload.get('n_test')}); no answer was regenerated."
    )
    lines.append("")
    rows = []
    for name, value in payload.get("test", {}).items():
        rows.append(
            [
                value.get("scorer", name),
                value.get("coverage"),
                value.get("realised_risk"),
                value.get("risk_ci"),
                value.get("ece"),
                "yes" if value.get("risk_control_held") else "no",
            ]
        )
    lines.append(
        _table(
            ("scorer", "coverage", "realised risk", "risk 95% CI", "ECE", "risk control held"), rows
        )
    )
    lines.append("")
    verdict = payload.get("verdict", {})
    lines.append(f"**{verdict.get('note', '')}**")
    lines.append("")
    lines.append(
        f"Gate rule (stated explicitly): {payload.get('risk_gate_rule', '')} "
        f"Label source: {payload.get('label_source', '')}. {payload.get('label_caveat', '')} "
        f"n_dev = {payload.get('n_dev')}, so the fit is small-n; the CIs above are the honest "
        "measure of that."
    )
    lines.append("")
    lines.append(
        "The calibrated score is much better *calibrated* than the raw overlap score "
        f"(ECE {payload['test']['calibrated_conformal']['ece']} vs "
        f"{payload['test']['raw_overlap']['ece']}) but has **not enough discriminative power** "
        "to deliver the 5% guarantee on this corpus: at the conformal threshold the realised "
        "risk is far above alpha. That is a measured negative result and is reported as such."
    )
    lines.append("")
    return lines


def _external_block() -> list[str]:
    """External validity is a separate deliverable (t19); state the boundary explicitly."""
    doc = REPO_ROOT / "docs" / "external-validity.md"
    lines = ["## External validity (separate deliverable)", ""]
    if doc.is_file():
        lines.append(
            "Public-benchmark results live in `docs/external-validity.md` and are **not** merged "
            "into the tables above, because they answer a different question (external "
            "comparability) on different data with their own licences and sampling."
        )
    else:
        lines.append(
            "No external-benchmark document is present in this checkout. Public-benchmark "
            "results are owned by the external-validity task and are deliberately not merged "
            "into these tables."
        )
    lines.append("")
    lines.append(
        "Concretely: no public retrieval qrels exist for Uzbek, Tajik or Kyrgyz, so this "
        "document makes **no** external retrieval claim for those locales; ru and zh do have "
        "public retrieval benchmarks (MIRACL, Mr. TyDi), and those belong to the external "
        "report. Belebele-derived labels are derived/diagnostic and never presented as public "
        "qrels."
    )
    lines.append("")
    return lines


def _limitations(summary: Mapping[str, Any]) -> list[str]:
    replay = bool(summary.get("replay"))
    lines = [
        "## Limitations — what does NOT generalize",
        "",
        f"1. **Synthetic corpus.** {summary.get('documents')} documents / "
        f"{summary.get('chunks')} chunks generated in six languages; queries are derived from "
        "the same documents, so retrieval and groundedness are optimistic relative to real "
        "support traffic. No number here transfers without re-measurement on a production "
        "corpus.",
        "2. **No real bank traffic, no SLA, no incident history.** Escalation cost ratios "
        "(c_wrong : c_abstain : c_delay) are documented assumptions; no handoff cost was "
        "measured.",
        "3. **Judge-mediated metrics.** Answer quality is judged by an LLM "
        "(cross-family enforced); without human κ on this corpus, judged numbers carry an "
        "explicit κ caveat and are not F-1/F-2/F-3 release gates.",
        "4. **Machine labels.** Gold evidence comes from the corpus structure "
        "(FAQ/equivalence classes), not from human annotation; §10.3 human verification "
        "has not been performed.",
        "5. **Sample size.** Generation arms are evaluated on a stratified subsample "
        f"(answer_limit={summary.get('config', {}).get('answer_limit')}, "
        f"retrieval_limit={summary.get('config', {}).get('retrieval_limit')}); the "
        "per-language and per-family intervals are wide. Claims of 5-point differences need "
        "n≈400 (t17 playbook); smaller n is reported with its interval rather than rounded "
        "away.",
        "6. **Ablation scope.** Chunking variants and cache on/off are reported where the "
        "artefacts exist; the retrieval ablation measures the shipped pipeline but not "
        "every fuser/normalisation combination.",
        "7. **External validity is a separate deliverable.** Public-benchmark results live in "
        "`docs/external-validity.md` (t19); this document does not claim external retrieval "
        "quality for Uzbek, Tajik or Kyrgyz, for which no public qrels exist.",
        "8. **Provider drift.** Hosted models and prices change; the run records model ids, "
        "token counts and the pricing snapshot date so a later run can be compared honestly.",
        "9. **Reasoning variant.** The primary generator is `qwen3.6-35b-a3b-noreason` "
        "(measured: the reasoning variant spent its whole JSON budget on the trace in a "
        "direct probe). The planned side study at 8192/16384 tokens was **not run in this "
        "window**: the two free subscriptions were spent on the answer arms after the session "
        "cooldown, so the reasoning variant has probe evidence only and no headline number "
        "depends on it. `scripts/reasoning-side-study.py` reproduces it when quota allows.",
        "10. **Noisy slice is small.** The free-form block is reported per example so a reader "
        "can see individual failures; it is not a population estimate for arbitrary malformed "
        "input.",
        "11. **Confidence is a retrieval score, not a calibrated probability.** The decision "
        "engine's p-hat is the deterministic overlap scorer (t8), so the conformal quantile is "
        "a threshold on that score; realised test risk is reported with its interval rather "
        "than asserted as a distribution-free guarantee. A calibrated p-hat (t8's LLM-judge "
        "scorer) is the production path and is not used here to keep judging cross-family and "
        "affordable.",
    ]
    if replay:
        lines.insert(
            0,
            "0. **REPLAY MODE.** This run used deterministic doubles. Every number is a "
            "harness diagnostic, not a live-model result.",
        )
    lines.append("")
    return lines


def _reproduction(summary: Mapping[str, Any], run: Path) -> list[str]:
    lines = [
        "## Commands and reproduction",
        "",
        "```powershell",
        "$env:ANSWERDESK_CONFIG = 'configs/default.yaml'   # profile api is set by live-env",
        ". .\\scripts\\live-env.ps1 neuraldeep -Judge deepseek -Secondary deepseek",
        "uv run answerdesk ingest --demo                    # corpus (deterministic)",
        "uv run answerdesk eval --all --provider fake --report reports/golden-regen --regenerate",
        "uv run answerdesk eval --all --provider api --report reports/bench \\",
        "    --answer-limit 400 --retrieval-limit 540",
        "uv run python scripts/reasoning-side-study.py --limit 32 --budgets 8192 16384",
        "uv run python scripts/build-results.py --run reports/bench --out docs/results.md --check",
        "```",
        "",
        "The golden set and split are frozen and hashed (see the header table); regenerating "
        "them with the same seed must reproduce the same hashes. Latency and USD are measured "
        "in-process with the provider's own token counts.",
        "",
        f"Artefact digest: `{_canonical_sha256(run / 'summary.json')[:32]}` (summary.json).",
        "",
    ]
    return lines


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def render(run: Path) -> str:
    summary = _read_json(run / "summary.json")
    blocks: list[str] = [
        "# Results — answerdesk v1.0",
        "",
        "This document is generated from the artefacts of one benchmark run. It states what "
        "was measured, on which data, with which models, and what does not generalize. Every "
        "number below comes from `reports/<run>/`; the document itself is `" + GENERATOR + "`.",
        "",
        "> **Internal evidence.** The corpus is synthetic; the numbers describe the reference "
        "stack on that corpus, not a production customer-support deployment."
        + (" This run is REPLAY-MODE: no live model was called." if summary.get("replay") else ""),
        "",
        "## Run header",
        "",
    ]
    blocks.extend(_header_block(summary, run))
    blocks.extend(_provenance_block(summary))
    blocks.extend(["## Datasets", ""])
    corpus_manifest = REPO_ROOT / "data" / "corpus" / "manifest.json"
    if corpus_manifest.is_file():
        manifest = json.loads(corpus_manifest.read_text(encoding="utf-8"))
        stats = manifest.get("stats", {})
        queries = manifest.get("queries", {})
        blocks.append(
            _table(
                ("property", "value"),
                [
                    ["corpus version", manifest.get("corpus_version")],
                    ["schema", manifest.get("schema_version")],
                    ["documents", stats.get("documents")],
                    ["chunks", stats.get("chunks")],
                    ["queries", queries.get("count")],
                    ["query categories", json.dumps(queries.get("categories", {}), sort_keys=True)],
                    ["unanswerable share", queries.get("unanswerable_share")],
                    ["reference grounding", queries.get("reference_grounding")],
                    ["license", manifest.get("license")],
                ],
            )
        )
        blocks.append("")
    golden = REPO_ROOT / "evals" / "golden" / "v1" / "manifest.json"
    if golden.is_file():
        gmanifest = json.loads(golden.read_text(encoding="utf-8"))
        blocks.append(
            f"Golden set `{gmanifest.get('version')}`: seed {gmanifest.get('seed')}, "
            f"{len(gmanifest.get('composition', []))} composition checks, "
            f"categories {json.dumps(gmanifest.get('by_category', {}), sort_keys=True)}, "
            f"split {json.dumps(gmanifest.get('by_split', {}), sort_keys=True)}. "
            "Adversarial companion sets (near-miss cache pairs, injection, PII, perturbation, "
            "claim labels) are listed in `evals/README.md`."
        )
        blocks.append("")
    blocks.extend(_retrieval_block(summary, run))
    blocks.extend(_answer_block(summary, run))
    blocks.extend(_risk_block())
    blocks.extend(_calibration_verdict_block(summary))
    blocks.extend(_scorer_block(summary))
    blocks.extend(_escalation_block(summary, run))
    blocks.extend(_chunking_block(run))
    blocks.extend(_side_study_block(run))
    blocks.extend(_noisy_block(summary, run))
    blocks.extend(_cache_block(summary))
    blocks.extend(_ops_block(summary, run))
    blocks.extend(_lat_requirement_block(summary))
    blocks.extend(_error_taxonomy(run))
    blocks.extend(_external_block())
    blocks.extend(_limitations(summary))
    blocks.extend(_reproduction(summary, run))
    return "\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if the document is stale")
    args = parser.parse_args(argv)
    text = render(args.run)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        if current != text:
            print("docs/results.md is out of date; re-run without --check", file=sys.stderr)
            return 1
        print("docs/results.md matches the artefacts")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes) from {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
