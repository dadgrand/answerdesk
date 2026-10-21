"""Render docs/consistency.md from the committed consistency artefacts.

Every number is read from ``reports/<run>/summary.json``, ``logical-audit.json``,
``retrieval-invariance.json`` and the answers checkpoint, so the document cannot drift from the
evidence and ``--check`` fails when it does.

    uv run python scripts/build-consistency.py --run reports/consistency-p0 \
        --out docs/consistency.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "reports" / "consistency-p0"
DEFAULT_OUT = REPO_ROOT / "docs" / "consistency.md"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(block: dict | None) -> str:
    if not block:
        return "n/a"
    return (
        f"{block.get('rate')} [{block.get('ci_low')}, {block.get('ci_high')}] (n={block.get('n')})"
    )


def render(run: Path) -> str:
    summary = _read(run / "summary.json")
    audit = _read(run / "logical-audit.json") if (run / "logical-audit.json").is_file() else {}
    retrieval = (
        _read(run / "retrieval-invariance.json")
        if (run / "retrieval-invariance.json").is_file()
        else {}
    )
    per_language = summary.get("fact_agreement", {}).get("per_language", {})
    ambiguity = summary.get("ambiguity", {})
    surface = summary.get("surface_invariance", {})
    per_variant = surface.get("per_variant", {})
    language = summary.get("language_discipline", {})

    lines: list[str] = []
    lines.append("# Consistency audit")
    lines.append("")
    lines.append(
        "Does the system say the same thing in every language, under every surface form, and "
        "does the corpus agree with itself? This document reports the measured answers, "
        "including the ones that are negative. The internal benchmark (`docs/results.md`) "
        "measures quality; this audit measures **consistency**, which is a different property "
        "and can fail even when answers look plausible."
    )
    lines.append("")
    lines.append("## What was measured")
    lines.append("")
    lines.append(
        f"- Transcreated set: **{summary.get('set', {}).get('facts')} facts × "
        f"{len(summary.get('set', {}).get('locales', []))} locales** "
        f"({summary.get('set', {}).get('instances')} instances) built from corpus "
        f"`{summary.get('corpus_version')}` — each fact is an archetype + document index asked "
        "as the clean direct question in every locale."
    )
    lines.append(
        f"- Generator `{summary.get('generator_model')}`; judge `{summary.get('judge_model')}` "
        "(cross-family). Answers: "
        f"{summary.get('rows', {}).get('clean')} clean + {summary.get('rows', {}).get('variants')} "
        "surface-form rows."
    )
    lines.append(
        "- Storage: every row (question, answer, decision, cited documents, detected language, "
        "asserted values, judge score) is in `reports/consistency-p0/answers.jsonl`; the audit "
        "and invariance artefacts sit next to it."
    )
    lines.append("")
    lines.append("## 1. Cross-language fact agreement (the headline finding)")
    lines.append("")
    lines.append(
        "For each language: how often the answer states the fact (judge ≥ 4 against the "
        "reference answer) and whether it cites the fact's gold document set."
    )
    lines.append("")
    table = []
    for locale, block in sorted(per_language.items()):
        table.append(
            [
                locale,
                block["answers"],
                _fmt(block["same_fact"]),
                _fmt(block["citation_hit"]),
                _fmt(block["fidelity_exact"]),
            ]
        )
    lines.append(_table(("language", "n", "same fact", "citation hit", "value fidelity"), table))
    lines.append("")
    lines.append(
        f"Pooled: same fact {_fmt(summary['fact_agreement']['pooled'])}; citation overlap "
        f"{_fmt(summary['citation_overlap']['pooled'])}; value fidelity "
        f"{_fmt(summary['value_fidelity']['pooled'])}."
    )
    lines.append("")
    lines.append(
        "**This is a negative result and it is reported as one.** Fact-level agreement is low "
        "in every language, and citation overlap is low with it. The reason is visible in the "
        "next table: the clean question is not document-discriminating. One wording recurs in "
        f"**~{ambiguity.get('mean_siblings')} sibling documents** on average "
        f"({ambiguity.get('share_with_siblings')} of facts have at least one sibling, max "
        f"{ambiguity.get('max_siblings')}), each with its own slot values, so a correct "
        "retriever may return a different document that answers the same question with "
        "different numbers. That does not excuse the number — a customer asking «Сколько стоит "
        "перевыпуск карты?» gets different amounts depending on which sibling is retrieved — but "
        "it does explain it, and it is filed as a deficiency below rather than hidden."
    )
    lines.append("")
    cross = summary["fact_agreement"].get("cross_language", {})
    if cross.get("pairs"):
        worst = sorted(cross["pairs"].items(), key=lambda item: item[1]["rate"])[:6]
        lines.append(
            "Cross-language pairs (both locales state the fact for the same fact id): "
            + ", ".join(f"{name} {block['rate']}" for name, block in worst)
            + f"; pooled {cross.get('pooled', {}).get('rate')}."
        )
        lines.append("")
    lines.append("## 2. Language discipline")
    lines.append("")
    lines.append(
        "Answer language against the query language, and the detected-language field against "
        "the query:"
    )
    lines.append("")
    table = []
    for locale, block in sorted(language.items()):
        table.append(
            [
                locale,
                block["answers"],
                _fmt(block["discipline"]),
                _fmt(block["detected"]),
                _fmt(block["leakage"]),
            ]
        )
    lines.append(
        _table(("language", "n", "answer language ok", "detected ok", "evidence locale ok"), table)
    )
    lines.append("")
    pooled_evidence = None
    if language:
        flags = [block["leakage"]["successes"] for block in language.values()]
        totals = [block["leakage"]["n"] for block in language.values()]
        if sum(totals):
            pooled_evidence = round(sum(flags) / sum(totals), 3)
    lines.append(
        "`evidence locale ok` means every cited document id carries the query's locale. The "
        f"pooled rate is **{pooled_evidence}**: the retriever does not filter by locale, so "
        "answers frequently cite documents written in another language of the corpus. That is "
        "the mechanical cause of low fact agreement below — the model reads a sibling locale's "
        "numbers — and it is filed as CNS-6."
    )
    lines.append("")
    lines.append("## 3. Surface-form invariance")
    lines.append("")
    if surface.get("pairs"):
        lines.append(
            f"Measured on the {surface['pairs']} generated variant rows (RAG retriever, same "
            "generator and judge as above): does the variant keep the clean form's cited "
            "document set and the same decision?"
        )
        lines.append("")
        lines.append(f"- Evidence invariance: **{_fmt(surface['evidence_invariance'])}**")
        lines.append(f"- Decision invariance: **{_fmt(surface['decision_invariance'])}**")
        lines.append("")
        table = []
        for kind, block in sorted(per_variant.items()):
            table.append(
                [
                    kind,
                    block["facts"],
                    _fmt(block["evidence"]),
                    _fmt(block["decision"]),
                ]
            )
        lines.append(_table(("variant", "facts", "same evidence", "same decision"), table))
        lines.append("")
        provider_errors = summary.get("provider_error_rows", 0)
        lines.append(
            "Clean-form answers for all six languages are in `answers.jsonl`; the variant arm "
            "runs on the locale configured with `--variant-locales` (ru in this run). "
            f"**Caveat:** {provider_errors} variant rows hit provider 429s after the quota was "
            "exhausted and are excluded from the pair set, so the rates above are computed on "
            "the subset that completed (n = "
            f"{surface['pairs']} pairs) and are therefore imprecise, not a population estimate."
        )
    else:
        lines.append(
            "**NOT MEASURED in this run.** `summary.json` has `surface_invariance.pairs = 0`: "
            "the variant arm did not execute, so no rate is reported (a zero-sample CI would "
            "look like a measurement and is not one)."
        )
    lines.append("")
    retrieval_invariance_rate = 0.0
    if retrieval.get("rows"):
        _rows = retrieval.get("rows", [])
        retrieval_invariance_rate = round(
            sum(1 for row in _rows if row["same_evidence"]) / max(1, len(_rows)), 3
        )
        pairs = retrieval["rows"]
        successes = sum(1 for row in pairs if row["same_evidence"])
        order = ("telegraphic", "typo", "translit", "code_switch")
        lines.append(
            "Retrieval-level invariance is measured independently for **all six locales** on "
            f"all facts ({len(pairs)} pairs, BM25 top-5 document sets): "
            f"**{successes}/{len(pairs)} = {round(successes / len(pairs), 3)}** keep the same "
            "evidence as the clean form."
        )
        lines.append("")
        table = []
        for kind in order:
            subset = [row for row in pairs if row["variant"] == kind]
            hits = sum(1 for row in subset if row["same_evidence"])
            table.append(
                [
                    kind,
                    len(subset),
                    f"{hits}/{len(subset)} = {round(hits / len(subset), 3) if subset else 0}",
                ]
            )
        lines.append(_table(("variant", "pairs", "same evidence"), table))
        lines.append("")
        table = []
        for locale in sorted({row["locale"] for row in pairs}):
            subset = [row for row in pairs if row["locale"] == locale]
            hits = sum(1 for row in subset if row["same_evidence"])
            table.append([locale, len(subset), f"{round(hits / len(subset), 3) if subset else 0}"])
        lines.append(_table(("language", "pairs", "same evidence"), table))
        lines.append("")
        lines.append(
            "These are eval-layer stress variants (function words dropped, one character "
            "swapped, the sentence romanised, another locale's term injected), harsher than the "
            "corpus's own noisy items; they measure resilience, not product-log quality."
        )
        lines.append("")
    lines.append("## 4. Logical consistency of corpus v2.1")
    lines.append("")
    lines.append(
        f"Audited **{audit.get('checked_documents')} documents** "
        f"({audit.get('coverage', {}).get('with_tables')} with tables, "
        f"{audit.get('coverage', {}).get('table_rows')} table rows, "
        f"{audit.get('coverage', {}).get('with_as_of')} with an as-of date). Findings by "
        f"severity: {audit.get('by_severity')}. `invariant_holds = "
        f"{audit.get('invariant_holds')}`; within-document blockers: "
        f"{audit.get('by_severity', {}).get('blocker', 0)}."
    )
    lines.append("")
    if audit.get("conflict_parameters"):
        top = list(audit["conflict_parameters"].items())[:8]
        lines.append(
            "**Cross-document contradictions were found** (severity high, owner ingest): the "
            "same named product carries different values for the same parameter in different "
            "documents of the same locale. Most affected parameters: "
            + ", ".join(f"`{name}` ({count})" for name, count in top)
            + "."
        )
        lines.append("")
        per_locale = {
            locale: block["findings"] for locale, block in audit.get("locales", {}).items()
        }
        lines.append(
            "Findings per locale: "
            + ", ".join(f"{locale} {count}" for locale, count in sorted(per_locale.items()))
            + "."
        )
        lines.append("")
        lines.append(
            "Example (verbatim ids only, no corpus text): `syn-ru-0033` and `syn-ru-0093` both "
            "describe the same product tier with different values for the same parameter row. "
            "This is a real documentation inconsistency: a reader who opens both documents gets "
            "two answers. It is filed to ingest with severity **high** and it is not fixed here "
            "— this task does not edit corpus or ingest code."
        )
        lines.append("")
    lines.append("## 5. Honest limitations and filed deficiencies")
    lines.append("")
    lines.append(
        "| id | severity | finding | owner |\n|---|---|---|---|\n"
        f"| CNS-1 | high | cross-document contradictions in corpus v2.1: "
        f"{audit.get('by_severity', {}).get('high', 0)} same-product/same-parameter conflicts "
        "across documents of one locale | ingest (data-engineer) |\n"
        "| CNS-2 | high | clean questions are not document-discriminating "
        f"(~{ambiguity.get('mean_siblings')} siblings per fact), so fact-level answers differ "
        "depending on which sibling is retrieved | ingest (question design) |\n"
        "| CNS-3 | medium | cross-language fact agreement "
        f"{summary['fact_agreement']['pooled'].get('rate')} and citation overlap "
        f"{summary['citation_overlap']['pooled'].get('rate')}: multilingual answers do not "
        "agree on the same fact, partly explained by CNS-2 | eval + ingest |\n"
        "| CNS-4 | medium | retrieval-level surface-form invariance "
        f"{retrieval_invariance_rate}: "
        "romanised, typo and code-switch queries often retrieve different evidence | "
        "retrieval (t23 owner) |\n"
        f"| CNS-5 | low | {summary.get('judge_failures')} judged rows failed at the judge and are "
        "counted as not-stating-the-fact, not dropped | eval |\n"
        f"| CNS-6 | high | retrieval has no locale filter: pooled evidence-locale rate "
        f"{pooled_evidence}, so answers cite other languages' documents | retrieval (t23 owner) |"
    )
    lines.append("")
    lines.append(
        "Further limitations: the judge is a cross-family LLM, not a human; the transcreated "
        "set is one clean question per fact (the corpus's own noisy items are measured in the "
        "retrieval ablation, not here); the invariance variants are eval-layer stress forms; "
        "the audit's cross-document check only fires when the same product name and parameter "
        "key recur, so it is a lower bound on inconsistency."
    )
    lines.append("")
    lines.append("## 6. Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("# offline, deterministic (CI)")
    lines.append("uv run pytest -q tests/unit/eval")
    lines.append("")
    lines.append("# rebuild the transcreated set from corpus v2.1")
    lines.append("uv run answerdesk eval consistency --build")
    lines.append("")
    lines.append("# logical audit (0 blockers expected; high findings are reported)")
    lines.append("uv run answerdesk eval consistency --audit --report reports/consistency-p0")
    lines.append("")
    lines.append("# retrieval-level surface-form invariance, all locales, no model calls")
    lines.append("uv run answerdesk eval consistency --retrieval --report reports/consistency-p0")
    lines.append("")
    lines.append("# live answers (one language pair per invocation; the checkpoint resumes)")
    lines.append(". ./scripts/live-env.ps1 neuraldeep -Judge deepseek")
    lines.append(
        "uv run answerdesk eval consistency --run --provider api --limit 50 --languages ru,uz "
        '--variant-locales "" --report reports/consistency-p0'
    )
    lines.append("")
    lines.append("# surface-form variants (ru)")
    lines.append(
        "uv run answerdesk eval consistency --run --provider api --limit 50 --languages ru "
        "--variant-locales ru --report reports/consistency-p0"
    )
    lines.append("```")
    lines.append("")
    lines.append("## 7. Provenance")
    lines.append("")
    lines.append("```text")
    lines.append(f"corpus_version    {summary.get('corpus_version')}")
    lines.append(f"facts_sha256      {summary.get('set', {}).get('facts_sha256')}")
    lines.append(f"generator_model   {summary.get('generator_model')}")
    lines.append(f"judge_model       {summary.get('judge_model')}")
    lines.append(f"rows              {summary.get('rows')}")
    lines.append(f"generated_at      {summary.get('generated_at')}")
    lines.append(f"hardware          {summary.get('hardware')}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


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
            print(f"{args.out} is stale; regenerate it", file=sys.stderr)
            return 1
        print(f"{args.out} matches the artefacts")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text.encode())} bytes) from {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
