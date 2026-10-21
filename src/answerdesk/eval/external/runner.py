"""External-validity runner: suites, artefacts and the honest labels.

One command produces everything the external document quotes::

    uv run answerdesk eval external --run --report reports/external-p0

The runner refuses to invent numbers. Every artefact states which suite, language, split, n
and licence produced it; the Belebele-derived retrieval labels are marked derived; the
judge-calibration gate marks judged metrics diagnostic until a κ file exists.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.config import Settings, load_settings
from answerdesk.eval.external.bench import Bench
from answerdesk.eval.external.datasets import (
    DatasetRecord,
    load_curated,
)
from answerdesk.eval.external.qa import delta, run_condition, summarise
from answerdesk.eval.external.retrieval import (
    DenseRanker,
    HybridRanker,
    LexicalRanker,
    compare,
    evaluate_scores,
    score_queries,
)

SUITE_ORDER = ("belebele_qa", "belebele_retrieval", "mrtydi_retrieval", "domain")
BELEBELE_LANGUAGES = ("ru", "zh", "uz", "tg", "ky")
#: retrieval-quality claims are only admissible where public qrels exist
NATIVE_QRELS_LANGUAGES = ("ru",)
DERIVED_LABEL_CAVEAT = (
    "Belebele-derived retrieval qrels are derived from the reading-comprehension gold "
    "passage, not collected by a retrieval-labeling process; they are diagnostic for "
    "uz/tg/ky (no public qrels exist in any language for these) and are never a "
    "retrieval-quality claim."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ExternalConfig:
    run_id: str = "external-p0"
    report_dir: Path = Path("reports/external-p0")
    suites: tuple[str, ...] = SUITE_ORDER
    languages: tuple[str, ...] = BELEBELE_LANGUAGES
    limit: int | None = 150
    samples: int = 1000
    seed: int = 0
    provider: str = "fake"
    answerer_model: str | None = None
    embedder: str = "auto"
    top_k: int = 3
    calibrate_items: int = 150
    replay: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def inventory(bench: Bench, config: ExternalConfig) -> dict[str, Any]:
    """What this run may claim, per suite and language, with licence and label status."""
    suites: dict[str, Any] = {}
    for suite, meta in bench.suites.items():
        languages = list(meta.get("languages", []))
        counts = {language: None for language in languages}
        suites[suite] = {
            "items": int(meta.get("items", 0)),
            "languages": languages,
            "licence": meta.get("license", "unknown"),
            "task": meta.get("task", ""),
            "notes": meta.get("notes", ""),
            "paths": meta.get("paths", []),
            "label_status": (
                "derived/diagnostic"
                if suite == "belebele_retrieval"
                else ("native qrels" if suite == "mrtydi_retrieval" else "native")
            ),
            "counts_by_language": counts,
        }
    return {
        "benchmark": bench.manifest.get("benchmark"),
        "version": bench.version,
        "schema_version": bench.manifest.get("schema_version"),
        "release_date": bench.manifest.get("release_date"),
        "manifest_self_sha256": bench.manifest.get("self_sha256"),
        "licences": bench.manifest.get("license_map", {}),
        "suites": suites,
        "derived_label_caveat": DERIVED_LABEL_CAVEAT,
        "non_claims": [
            "no state-of-the-art claim of any kind",
            "no retrieval-quality claim for uz/tg/ky (derived labels only)",
            "no claim whose bootstrap interval crosses zero",
            "no calibrated-judge claim until the human-label κ file exists",
        ],
        "requested": {
            "suites": list(config.suites),
            "languages": list(config.languages),
            "limit": config.limit,
        },
    }


def _provider(settings: Settings, config: ExternalConfig) -> tuple[Any, str]:
    from answerdesk.llm.factory import build_provider

    if config.replay or config.provider == "fake":
        from answerdesk.testing.doubles import FakeProvider

        return FakeProvider(), "fake-model"
    if config.provider == "deepseek":
        import os

        from pydantic import SecretStr

        key = os.environ.get("DEEPSEEK_API_KEY")
        clone = settings.model_copy(deep=True)
        clone.llm.preset = "deepseek"
        clone.llm.base_url = "https://api.deepseek.com/v1"
        clone.llm.model = config.answerer_model or "deepseek-flash"
        clone.llm.api_key = SecretStr(key) if key else clone.llm.api_key
        clone.llm.max_tokens = 64
        clone.llm.temperature = 0.0
        # deepseek rejects response_format=json_object unless the prompt contains "json";
        # the multiple-choice answer is plain text, so JSON mode is switched off here.
        clone.llm.json_mode = False
        return build_provider(clone), clone.llm.model
    return build_provider(settings), settings.resolved_provider().model


def _embedder(settings: Settings, config: ExternalConfig) -> Any:
    if config.embedder == "fake" or config.replay:
        from answerdesk.testing.doubles import HashingEmbedder

        return HashingEmbedder()
    from answerdesk.index.embedders import api_embedder_from_settings

    return api_embedder_from_settings(settings)


def run_qa(
    *,
    bench: Bench,
    settings: Settings,
    config: ExternalConfig,
    languages: Sequence[str],
    provider: Any,
    model: str,
    progress: Any | None = None,
) -> dict[str, Any]:
    out_dir = config.report_dir
    checkpoint = out_dir / "qa-rows.jsonl"
    per_language: dict[str, Any] = {}
    # languages already answered in earlier chunks are summarised too, never re-generated
    answered_languages = set()
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answered_languages.add(json.loads(line)["language"])
    for language in sorted(set(languages) | answered_languages):
        items = bench.qa(language, split="test")
        if config.limit is not None:
            items = items[: config.limit]
        if not items:
            per_language[language] = {"n": 0, "status": "no items in the test split"}
            continue
        passages = bench.passages(language)
        ranker = LexicalRanker(passages)
        if language in languages:
            for condition in ("closed", "rag"):
                run_condition(
                    provider=provider,
                    items=items,
                    condition=condition,
                    ranker=ranker,
                    out_path=checkpoint,
                    top_k=config.top_k,
                    model=model,
                    progress=progress,
                )
        rows = [
            json.loads(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if row["language"] == language]
        closed = summarise(rows, "closed")
        rag = summarise(rows, "rag")
        per_language[language] = {
            "n": closed.n,
            "closed_book": {
                "correct": closed.correct,
                "accuracy": round(closed.accuracy, 6),
                "ci": [round(closed.ci[0], 6), round(closed.ci[1], 6)],
            },
            "rag": {
                "correct": rag.correct,
                "accuracy": round(rag.accuracy, 6),
                "ci": [round(rag.ci[0], 6), round(rag.ci[1], 6)],
            },
            "delta": delta(rows, samples=config.samples, seed=config.seed),
            "claim_status": (
                "native comprehension claim admissible"
                if language in NATIVE_QRELS_LANGUAGES or language == "zh"
                else "comprehension claim admissible; retrieval labels derived/diagnostic"
            ),
        }
    return {
        "suite": "belebele_qa",
        "task": "multiple_choice (judge-free: option letter is scored against answer_index)",
        "answerer_model": model,
        "language_results": per_language,
    }


def run_belebele_retrieval(
    *,
    bench: Bench,
    settings: Settings,
    config: ExternalConfig,
    languages: Sequence[str],
    embedder: Any,
) -> dict[str, Any]:
    """Derived-qrels retrieval: BM25 vs hybrid bge-m3, explicitly diagnostic."""
    per_language: dict[str, Any] = {}
    for language in languages:
        queries = bench.retrieval_queries(language, split="test")
        if config.limit is not None:
            queries = queries[: config.limit]
        passages = bench.passages(language)
        if not queries or not passages:
            per_language[language] = {"n": 0, "status": "empty"}
            continue
        lexical = LexicalRanker(passages)
        dense = DenseRanker(passages, embedder)
        hybrid = HybridRanker(passages, lexical, dense, alpha=0.5)
        passage_ids = [passage.passage_id for passage in passages]
        bm25 = evaluate_scores(
            arm="bm25",
            scores=score_queries(lexical, queries),
            queries=queries,
            passage_ids=passage_ids,
            samples=config.samples,
            seed=config.seed,
        )
        hybrid_report = evaluate_scores(
            arm="hybrid",
            scores=score_queries(hybrid, queries),
            queries=queries,
            passage_ids=passage_ids,
            samples=config.samples,
            seed=config.seed,
        )
        per_language[language] = {
            "n": len(queries),
            "passages": len(passages),
            "bm25": {
                "ndcg@10": round(bm25.ndcg10, 6),
                "ndcg@10_ci": [round(bm25.ndcg10_ci[0], 6), round(bm25.ndcg10_ci[1], 6)],
                "recall@10": round(bm25.recall10, 6),
                "recall@10_ci": [round(bm25.recall10_ci[0], 6), round(bm25.recall10_ci[1], 6)],
            },
            "hybrid": {
                "ndcg@10": round(hybrid_report.ndcg10, 6),
                "ndcg@10_ci": [
                    round(hybrid_report.ndcg10_ci[0], 6),
                    round(hybrid_report.ndcg10_ci[1], 6),
                ],
                "recall@10": round(hybrid_report.recall10, 6),
                "recall@10_ci": [
                    round(hybrid_report.recall10_ci[0], 6),
                    round(hybrid_report.recall10_ci[1], 6),
                ],
            },
            "comparison": compare(bm25, hybrid_report, samples=config.samples, seed=config.seed),
            "label_status": "derived/diagnostic",
        }
    return {
        "suite": "belebele_retrieval",
        "task": "retrieval with derived qrels (reading-comprehension gold passage)",
        "label_status": "derived/diagnostic",
        "caveat": DERIVED_LABEL_CAVEAT,
        "language_results": per_language,
    }


def run_mrtydi(
    *,
    bench: Bench,
    config: ExternalConfig,
    embedder: Any,
) -> dict[str, Any]:
    """Native Russian qrels: BM25 against hybrid with hosted bge-m3."""
    queries = bench.mrtydi_queries(split="test")
    if config.limit is not None:
        queries = queries[: config.limit]
    passages = bench.mrtydi_passages()
    lexical = LexicalRanker(passages)
    dense = DenseRanker(passages, embedder)
    hybrid = HybridRanker(passages, lexical, dense, alpha=0.5)
    passage_ids = [passage.passage_id for passage in passages]
    bm25 = evaluate_scores(
        arm="bm25",
        scores=score_queries(lexical, queries),
        queries=queries,
        passage_ids=passage_ids,
        samples=config.samples,
        seed=config.seed,
    )
    hybrid_report = evaluate_scores(
        arm="hybrid",
        scores=score_queries(hybrid, queries),
        queries=queries,
        passage_ids=passage_ids,
        samples=config.samples,
        seed=config.seed,
    )
    comparison = compare(bm25, hybrid_report, samples=config.samples, seed=config.seed)
    # per-query detail lives in CSVs, not in the summary: keeps the summary readable and
    # diffable, and gives a reviewer the raw rows behind every aggregate
    for name, report in (
        ("mrtydi-bm25-per-query.csv", bm25),
        ("mrtydi-hybrid-per-query.csv", hybrid_report),
    ):
        path = config.report_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            import csv as _csv

            writer = _csv.DictWriter(
                handle, fieldnames=["item_id", "language", "ndcg@10", "recall@10", "mrr@10"]
            )
            writer.writeheader()
            for row in report.per_query:
                writer.writerow(
                    {
                        key: row[key]
                        for key in ("item_id", "language", "ndcg@10", "recall@10", "mrr@10")
                    }
                )
    return {
        "suite": "mrtydi_retrieval",
        "task": "retrieval with native qrels (Mr. TyDi ru, Apache-2.0)",
        "label_status": "native qrels",
        "n": len(queries),
        "passages": len(passages),
        "split": "test",
        "fusion": {"alpha_bm25": 0.5, "normalisation": "min-max", "family": "convex combination"},
        "bm25": _retrieval_payload(bm25),
        "hybrid": _retrieval_payload(hybrid_report),
        "comparison": comparison,
        "per_query_files": ["mrtydi-bm25-per-query.csv", "mrtydi-hybrid-per-query.csv"],
    }


def _retrieval_payload(report: Any) -> dict[str, Any]:
    return {
        "ndcg@10": round(report.ndcg10, 6),
        "ndcg@10_ci": [round(report.ndcg10_ci[0], 6), round(report.ndcg10_ci[1], 6)],
        "recall@10": round(report.recall10, 6),
        "recall@10_ci": [round(report.recall10_ci[0], 6), round(report.recall10_ci[1], 6)],
        "mrr@10": round(report.mrr10, 6),
        "n": report.n,
    }


def build_calibration_templates(
    *, bench: Bench, config: ExternalConfig, languages: Sequence[str]
) -> dict[str, Any]:
    from answerdesk.eval.external.calibration import build_template, write_protocol

    out_dir = config.report_dir / "calibration"
    templates = []
    for language in languages:
        items = [row for row in bench.qa(language, split=None)]
        rows = [
            {"item_id": item.item_id, "question": item.question, "answer_text": ""}
            for item in items
        ]
        templates.append(
            build_template(
                language=language,
                items=rows,
                out_dir=out_dir,
                limit=config.calibrate_items,
            ).to_dict()
        )
    protocol = write_protocol(out_dir, languages=languages, limit=config.calibrate_items)
    return {
        "templates": templates,
        "protocol": str(protocol),
        "status": "pending-human-labels",
        "note": (
            "no human labels are present in this checkout; every judged metric in this arm is "
            "diagnostic until calibration-kappa.json exists with κ >= 0.6 per language"
        ),
    }


def run_external(
    config: ExternalConfig,
    *,
    settings: Settings | None = None,
    bench: Bench | None = None,
    provider: Any | None = None,
    embedder: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    settings = settings or load_settings(profile=None)
    record: DatasetRecord = load_curated(verify=True)
    bench = bench or Bench(record)
    out_dir = config.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": (
            f"{platform.system()} {platform.release()} / Python {platform.python_version()}"
        ),
        "dataset": record.to_dict(),
        "inventory": inventory(bench, config),
        "suites_requested": list(config.suites),
        "languages": list(config.languages),
        "replay": config.replay,
        "judge_gate": {
            "status": "diagnostic-only",
            "reason": "no human labels in this checkout; judged metrics are diagnostic",
        },
    }
    answerer_model = "fake-model"
    if "belebele_qa" in config.suites:
        provider = provider or _provider(settings, config)[0]
        answerer_model = config.answerer_model or (
            _provider(settings, config)[1]
            if config.provider != "fake" and not config.replay
            else "fake-model"
        )
        summary["belebele_qa"] = run_qa(
            bench=bench,
            settings=settings,
            config=config,
            languages=[lang for lang in config.languages if lang in BELEBELE_LANGUAGES],
            provider=provider,
            model=answerer_model,
            progress=progress,
        )
    if "mrtydi_retrieval" in config.suites:
        embedder = embedder or _embedder(settings, config)
        summary["mrtydi_retrieval"] = run_mrtydi(bench=bench, config=config, embedder=embedder)
    if "belebele_retrieval" in config.suites:
        embedder = embedder or _embedder(settings, config)
        summary["belebele_retrieval"] = run_belebele_retrieval(
            bench=bench,
            settings=settings,
            config=config,
            languages=[lang for lang in config.languages if lang in BELEBELE_LANGUAGES],
            embedder=embedder,
        )
    if True:  # the domain slice is always reported as delegated to the internal harness
        summary["domain"] = {
            "suite": "domain",
            "task": "grounded QA on the synthetic corpus (CC0-1.0)",
            "status": "delegated to the internal benchmark (reports/bench)",
            "note": (
                "the domain slice reuses the internal harness and its frozen split; external "
                "numbers come from Belebele and Mr. TyDi only"
            ),
        }
    previous_path = out_dir / "summary.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        for suite in ("mrtydi_retrieval", "belebele_retrieval", "belebele_qa", "domain"):
            if suite not in config.suites and suite in previous:
                summary[suite] = previous[suite]
        if "belebele_qa" in config.suites:
            merged = dict(previous.get("belebele_qa", {}).get("language_results", {}))
            merged.update(summary.get("belebele_qa", {}).get("language_results", {}))
            summary.setdefault("belebele_qa", {})["language_results"] = merged
    summary["calibration"] = build_calibration_templates(
        bench=bench, config=config, languages=list(config.languages)
    )
    summary["dataset_digest"] = {
        "manifest_sha256": record.manifest_sha256,
        "files": {entry.path: entry.sha256 for entry in record.files},
        "digest": _sha256(
            out_dir / "qa-rows.jsonl" if (out_dir / "qa-rows.jsonl").is_file() else Path(__file__)
        ),
    }
    summary["duration_s"] = round(time.perf_counter() - started, 3)
    summary["non_claims"] = summary["inventory"]["non_claims"]
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "dataset-digest.json", summary["dataset_digest"])
    (out_dir / "report.md").write_text(render_external_report(summary), encoding="utf-8")
    return summary


def render_external_report(summary: dict[str, Any]) -> str:
    """Human-readable artefacts summary; ids and metrics only, never dataset text."""
    lines = [
        f"# External validity run `{summary.get('run_id')}`",
        "",
        f"Hardware: {summary.get('hardware')}",
        "",
        f"Curated benchmark: {summary['inventory'].get('benchmark')} "
        f"{summary['inventory'].get('version')} "
        f"(manifest `{summary['inventory'].get('manifest_self_sha256', '')[:16]}…`)",
        "",
    ]
    qa = summary.get("belebele_qa")
    if qa:
        lines += [
            "## Belebele-derived QA (judge-free, option-letter scoring)",
            "",
            f"Answerer: `{qa.get('answerer_model')}`",
            "",
            "| language | n | closed-book | RAG | delta | 95% CI | crosses zero |",
            "|---|---|---|---|---|---|---|",
        ]
        for language, row in qa["language_results"].items():
            delta_row = row.get("delta", {})
            lines.append(
                f"| {language} | {row.get('n')} | {row.get('closed_book', {}).get('accuracy')} | "
                f"{row.get('rag', {}).get('accuracy')} | {delta_row.get('delta')} | "
                f"[{delta_row.get('ci_low')}, {delta_row.get('ci_high')}] | "
                f"{'yes' if delta_row.get('ci_crosses_zero') else 'no'} |"
            )
        lines.append("")
    mrtydi = summary.get("mrtydi_retrieval")
    if mrtydi:
        comparison = mrtydi.get("comparison", {})
        lines += [
            "## Mr. TyDi ru (native qrels)",
            "",
            f"n = {mrtydi.get('n')} queries, {mrtydi.get('passages')} passages, "
            f"fusion {mrtydi.get('fusion')}",
            "",
            "| arm | nDCG@10 | recall@10 | MRR@10 |",
            "|---|---|---|---|",
            f"| bm25 | {mrtydi['bm25']['ndcg@10']} | {mrtydi['bm25']['recall@10']} | "
            f"{mrtydi['bm25']['mrr@10']} |",
            f"| hybrid | {mrtydi['hybrid']['ndcg@10']} | {mrtydi['hybrid']['recall@10']} | "
            f"{mrtydi['hybrid']['mrr@10']} |",
            "",
            f"Paired bootstrap nDCG@10 delta: {comparison.get('ndcg@10', {}).get('delta')} "
            f"[{comparison.get('ndcg@10', {}).get('ci_low')}, "
            f"{comparison.get('ndcg@10', {}).get('ci_high')}] "
            f"(crosses zero: {comparison.get('ndcg@10', {}).get('ci_crosses_zero')})",
            "",
        ]
    derived = summary.get("belebele_retrieval")
    if derived:
        lines += ["## Belebele-derived retrieval (diagnostic only)", ""]
        for language, row in derived["language_results"].items():
            if not row.get("n"):
                continue
            comparison = row.get("comparison", {}).get("ndcg@10", {})
            lines.append(
                f"- {language}: n={row['n']}, bm25 nDCG@10 {row['bm25']['ndcg@10']}, "
                f"hybrid {row['hybrid']['ndcg@10']}, delta {comparison.get('delta')} "
                f"[{comparison.get('ci_low')}, {comparison.get('ci_high')}], "
                f"crosses zero {comparison.get('ci_crosses_zero')} — **derived labels, "
                "diagnostic**"
            )
        lines.append("")
    calibration = summary.get("calibration", {})
    lines += [
        "## Judge calibration",
        "",
        f"Status: {calibration.get('status')}",
        "",
        str(calibration.get("note", "")),
        "",
        "## Non-claims",
        "",
    ]
    lines += [f"- {item}" for item in summary.get("non_claims", [])]
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BELEBELE_LANGUAGES",
    "DERIVED_LABEL_CAVEAT",
    "SUITE_ORDER",
    "ExternalConfig",
    "inventory",
    "render_external_report",
    "run_external",
    "run_mrtydi",
    "run_qa",
]
