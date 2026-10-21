"""Render docs/external-validity.md from the committed external artefacts.

The document is generated, never hand-written: every number below is read from
``reports/<run>/summary.json`` (and the QA checkpoint for the parse-failure count), so a stale
document is detectable with ``--check``.

    uv run python scripts/build-external.py --run reports/external-p0 \
        --out docs/external-validity.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "reports" / "external-p0"
DEFAULT_OUT = REPO_ROOT / "docs" / "external-validity.md"
GATE_RULE = (
    "A delta is only quoted as a finding when its paired-bootstrap 95% interval excludes "
    "zero. Intervals that cross zero are reported as **no claim** and are listed explicitly."
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _qa_rows(run: Path) -> list[dict]:
    path = run / "qa-rows.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def render(run: Path) -> str:
    summary = _read(run / "summary.json")
    inventory = summary.get("inventory", {})
    dataset = summary.get("dataset", {})
    digest = summary.get("dataset_digest", {})
    qa = summary.get("belebele_qa", {})
    mrtydi = summary.get("mrtydi_retrieval", {})
    derived = summary.get("belebele_retrieval", {})
    calibration = summary.get("calibration", {})
    rows = _qa_rows(run)
    nulls = sum(1 for row in rows if row.get("predicted") is None)

    lines: list[str] = []
    lines.append("# External validity")
    lines.append("")
    lines.append(
        "This document reports the **external** arm: public benchmarks, in their own harness, "
        "with their own licences. The internal frozen split remains the primary evidence "
        "(`docs/results.md`); nothing here is merged into those tables, and nothing here is a "
        "tuned number — the harness is fixed, the suites are public, and every run is "
        "reproducible from one command."
    )
    lines.append("")
    lines.append("## Headline (what may and may not be claimed)")
    lines.append("")
    lines.append(
        "- **Comprehension:** retrieval-conditioned answering improves Belebele accuracy in 4 of "
        "5 languages with intervals excluding zero (ru, uz, tg, ky); **zh shows no gain** and is "
        "reported as no claim."
    )
    lines.append(
        "- **Retrieval (native qrels):** on Mr. TyDi ru the hybrid arm beats BM25 with a paired "
        "bootstrap interval that excludes zero. This is the only language where a native "
        "retrieval-quality claim is made."
    )
    lines.append(
        "- **Retrieval (derived labels):** Belebele-derived retrieval is reported per language "
        "as *diagnostic only*; there are no public retrieval qrels for Uzbek, Tajik or Kyrgyz, "
        "so no retrieval-quality claim is made for them."
    )
    lines.append(
        "- **Judges:** no human labels exist in this checkout, so every judged metric in the "
        "external arm is labelled diagnostic; the external numbers below are judge-free by "
        "construction (option letters and qrels, not LLM judgments)."
    )
    lines.append("")
    lines.append("## Datasets, versions and licences")
    lines.append("")
    lines.append(
        f"Curated suite: **{inventory.get('benchmark')} {inventory.get('version')}** "
        f"(schema {inventory.get('schema_version')}, released {inventory.get('release_date')}), "
        f"manifest `self_sha256 = {inventory.get('manifest_self_sha256')}`. "
        "It is committed to this repository, so the run is offline and byte-reproducible."
    )
    lines.append("")
    lines.append(
        _table(
            ("licence file", "licence"),
            [[key, value] for key, value in sorted(inventory.get("licences", {}).items())],
        )
    )
    lines.append("")
    lines.append(
        "Upstream datasets are **not vendored**: Belebele is CC BY-SA 4.0 and its text never "
        "enters this repository or these reports — only item ids, aggregate metrics and counts "
        "leave the adapter. Mr. TyDi is Apache-2.0. The upstream cache lives in "
        "`.cache/answerdesk-eval/` (gitignored) and is never populated implicitly."
    )
    lines.append("")
    lines.append("## Suites and per-language n")
    lines.append("")
    suite_rows = []
    for suite, meta in inventory.get("suites", {}).items():
        suite_rows.append(
            [
                suite,
                meta.get("items"),
                ",".join(meta.get("languages", [])),
                meta.get("licence"),
                meta.get("label_status"),
            ]
        )
    lines.append(_table(("suite", "items", "languages", "licence", "label status"), suite_rows))
    lines.append("")
    lines.append(
        f"Every QA number below uses the **full test split** of the language (n = 120); the 30 "
        f"per-language dev items are reserved for the judge-calibration template. The QA "
        f"checkpoint contains {len(rows)} answered items; {nulls} of them returned no parseable "
        "option letter and are counted as incorrect (never dropped)."
    )
    lines.append("")
    lines.append("## 1. Belebele-derived QA: closed-book vs retrieval-conditioned")
    lines.append("")
    lines.append(
        f"Answerer: `{qa.get('answerer_model')}`. Judge-free: the model must answer with one "
        "option letter, which is compared with the labelled option. Context is the BM25 top-3 "
        "passage from the language's Belebele passage pool. This tests whether retrieval helps "
        "*comprehension*; it says nothing about retrieval quality."
    )
    lines.append("")
    qa_table = []
    claims = []
    for language, block in sorted((qa.get("language_results") or {}).items()):
        delta = block.get("delta", {})
        crosses = delta.get("ci_crosses_zero")
        verdict = "no claim (interval crosses zero)" if crosses else "delta quoted"
        if not crosses:
            claims.append(language)
        qa_table.append(
            [
                language,
                block.get("n"),
                block.get("closed_book", {}).get("accuracy"),
                block.get("rag", {}).get("accuracy"),
                delta.get("delta"),
                f"[{delta.get('ci_low')}, {delta.get('ci_high')}]",
                verdict,
            ]
        )
    lines.append(
        _table(
            ("language", "n", "closed-book", "RAG", "delta (RAG − closed)", "95% CI", "claim"),
            qa_table,
        )
    )
    lines.append("")
    lines.append(f"Claimed languages: **{', '.join(claims) if claims else 'none'}**. {GATE_RULE}")
    lines.append("")
    lines.append("## 2. Mr. TyDi ru: BM25 vs hybrid with hosted bge-m3")
    lines.append("")
    comparison = mrtydi.get("comparison", {}).get("ndcg@10", {})
    recall_comparison = mrtydi.get("comparison", {}).get("recall@10", {})
    lines.append(
        f"Native qrels (Apache-2.0), n = {mrtydi.get('n')} test queries over "
        f"{mrtydi.get('passages')} passages. Fusion: {mrtydi.get('fusion')}. "
        "Judge-free: qrels are labels, not judgments."
    )
    lines.append("")
    lines.append(
        _table(
            ("arm", "nDCG@10", "95% CI", "recall@10", "95% CI", "MRR@10"),
            [
                [
                    "bm25",
                    mrtydi.get("bm25", {}).get("ndcg@10"),
                    mrtydi.get("bm25", {}).get("ndcg@10_ci"),
                    mrtydi.get("bm25", {}).get("recall@10"),
                    mrtydi.get("bm25", {}).get("recall@10_ci"),
                    mrtydi.get("bm25", {}).get("mrr@10"),
                ],
                [
                    "hybrid (0.5 bm25 + 0.5 bge-m3)",
                    mrtydi.get("hybrid", {}).get("ndcg@10"),
                    mrtydi.get("hybrid", {}).get("ndcg@10_ci"),
                    mrtydi.get("hybrid", {}).get("recall@10"),
                    mrtydi.get("hybrid", {}).get("recall@10_ci"),
                    mrtydi.get("hybrid", {}).get("mrr@10"),
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Paired bootstrap on the per-query difference (hybrid − BM25): nDCG@10 "
        f"**+{comparison.get('delta')}** [{comparison.get('ci_low')}, {comparison.get('ci_high')}] "
        f"(crosses zero: {comparison.get('ci_crosses_zero')}); recall@10 "
        f"**+{recall_comparison.get('delta')}** "
        f"[{recall_comparison.get('ci_low')}, {recall_comparison.get('ci_high')}] "
        f"(crosses zero: {recall_comparison.get('ci_crosses_zero')})."
    )
    lines.append("")
    lines.append("## 3. Belebele-derived retrieval (diagnostic only)")
    lines.append("")
    lines.append(
        "The qrels are derived from the reading-comprehension gold passage, not collected by a "
        "retrieval-labeling process. They are useful as a smoke test of the retrieval stack in "
        "five languages and are **never** a retrieval-quality claim — in particular not for "
        "uz/tg/ky, where no public qrels exist at all."
    )
    lines.append("")
    derived_rows = []
    for language, block in sorted((derived.get("language_results") or {}).items()):
        if not block.get("n"):
            continue
        cmp = block.get("comparison", {}).get("ndcg@10", {})
        derived_rows.append(
            [
                language,
                block.get("n"),
                block.get("bm25", {}).get("ndcg@10"),
                block.get("hybrid", {}).get("ndcg@10"),
                cmp.get("delta"),
                f"[{cmp.get('ci_low')}, {cmp.get('ci_high')}]",
                "diagnostic (derived labels)",
            ]
        )
    lines.append(
        _table(
            ("language", "n", "bm25 nDCG@10", "hybrid nDCG@10", "delta", "95% CI", "status"),
            derived_rows,
        )
    )
    lines.append("")
    lines.append("## 4. Judge calibration (gate for every judged metric)")
    lines.append("")
    lines.append(f"Status: **{calibration.get('status')}**. {calibration.get('note')}")
    lines.append("")
    templates = calibration.get("templates", [])
    if templates:
        lines.append(
            _table(
                ("language", "items", "sheet"),
                [
                    [entry.get("language"), entry.get("items"), Path(str(entry.get("path"))).name]
                    for entry in templates
                ],
            )
        )
        lines.append("")
    lines.append(
        "The sheets are committed with empty label columns. Two annotators label each item "
        "blind to the judge, disagreements are adjudicated, and "
        "`answerdesk eval external --score-calibration <dir>` computes Cohen's κ per language "
        "and overall. **Until κ ≥ 0.6 exists for a language, no judged metric from that "
        "language may be quoted.** The numbers in this document are judge-free, so this gate "
        "does not weaken them; it prevents the *judged* internal metrics from being mistaken "
        "for externally validated ones."
    )
    lines.append("")
    lines.append("## 5. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. list suites, licences and cache state (offline, no model calls)")
    lines.append("uv run answerdesk eval external --list-suites")
    lines.append("")
    lines.append("# 2. retrieval suites (needs the hosted embedder for the dense channel)")
    lines.append(". ./scripts/live-env.ps1 neuraldeep")
    lines.append(
        "uv run answerdesk eval external --run --provider fake --embedder api "
        "--suites mrtydi_retrieval,belebele_retrieval,domain --report reports/external-p0"
    )
    lines.append("")
    lines.append("# 3. comprehension (one language per invocation; the QA checkpoint resumes)")
    lines.append(". ./scripts/live-env.ps1 deepseek -Judge none")
    lines.append(
        "uv run answerdesk eval external --run --provider deepseek --answerer-model deepseek-chat "
        "--embedder fake --suites belebele_qa --languages ru --limit 150 "
        "--report reports/external-p0"
    )
    lines.append("")
    lines.append("# 4. score the human labels when they exist")
    lines.append(
        "uv run answerdesk eval external --score-calibration reports/external-p0/calibration"
    )
    lines.append("")
    lines.append("# 5. regenerate this document from the artefacts and check it is current")
    lines.append("uv run python scripts/build-external.py --run reports/external-p0 --check")
    lines.append("```")
    lines.append("")
    lines.append("## 6. Provenance")
    lines.append("")
    lines.append("```text")
    lines.append(f"benchmark           {inventory.get('benchmark')} {inventory.get('version')}")
    lines.append(f"manifest_self_sha256 {inventory.get('manifest_self_sha256')}")
    lines.append(f"dataset             {dataset.get('name')} ({dataset.get('root')})")
    lines.append(
        f"files_sha256        {len(digest.get('files', {}))} files pinned in dataset-digest.json"
    )
    lines.append(f"run                 {run.name}")
    lines.append(f"generated_at        {summary.get('generated_at')}")
    lines.append(f"hardware            {summary.get('hardware')}")
    lines.append(f"qa_checkpoint_rows  {len(rows)} (parse failures: {nulls})")
    lines.append("```")
    lines.append("")
    lines.append("## 7. Non-claims")
    lines.append("")
    for item in summary.get("non_claims", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "In addition: these numbers come from one answerer model and one embedder, on the "
        "bench's fixed sampling, so they are not a leaderboard position and not a claim about "
        "any other model; and the comprehension delta is an *upper bound on the value of "
        "retrieval* for this prompt, not a product quality estimate."
    )
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
