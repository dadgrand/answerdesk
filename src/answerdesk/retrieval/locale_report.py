"""Deterministic measurement of the locale policy (CNS-6) — ``python -m ... --check``.

What it measures, on the **consistency query set** (the same 378 clean instances the
consistency run used: 63 facts × 6 locales, each locale answering from its own document):

* **evidence-locale purity** — share of the packed context written in the query's locale,
  pooled over instances and chunks, per locale, and the share of instances whose context is
  entirely (or ≥90%) same-locale. Before = the policy off, after = the policy on.
* **surface-form invariance** — a query's upper-cased/spaced/punctuated variant must produce
  the identical packed context, per instance.
* **ablation non-regression** — ``recall_any@20`` for the BM25, dense and hybrid arms on the
  corpus's clean ``direct`` queries, policy off vs on; the contract allows |Δ| ≤ 0.02.
* **determinism** — the whole "on" measurement is run twice and the packed-context digest must
  match.

Everything is retrieval-level: no model call, no network, no randomness (``--seed`` exists for
the record only). ``--check`` exits 0 only when every gate holds, which is what the task
contract verifies; the artifacts it writes (``report.json``, ``report.md``) are the committed
before/after evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerdesk.config import load_settings
from answerdesk.errors import CorpusMissingError
from answerdesk.index.embedders import default_embedder
from answerdesk.index.store import ChunkStore, IndexBundle
from answerdesk.retrieval.datasets import CLEAN_CATEGORIES, build_cases, load_query_rows
from answerdesk.retrieval.fusion import FusionParams
from answerdesk.retrieval.locale_policy import LocalePolicy
from answerdesk.retrieval.pipeline import PipelineConfig, RetrievalPipeline

#: The consistency run whose clean instances define the query set.
DEFAULT_CONSISTENCY_RUN = Path("reports/consistency-p0")
#: Where this report is written (committed evidence).
DEFAULT_OUTPUT_DIR = Path("reports/locale-filter")
#: Packed context size used for the purity measurement (the shipped default).
DEFAULT_WINDOW = 12
#: Contract floor: pooled purity after the policy.
PURITY_TARGET = 0.90
#: Contract floor: invariance must be exact.
INVARIANCE_TARGET = 1.0
#: Contract floor: allowed ablation regression, absolute.
ABLATION_TOLERANCE = 0.02


@dataclass(frozen=True)
class ConsistencyInstance:
    """One clean consistency query: what was asked, in which locale, with which gold docs."""

    instance_id: str
    locale: str
    fact_id: str
    archetype: str
    question: str
    gold_doc_ids: tuple[str, ...]


def load_consistency_set(path: Path | str) -> list[ConsistencyInstance]:
    """Read the clean instances of a consistency run (``answers.jsonl``)."""
    source = Path(path)
    answers = source / "answers.jsonl" if source.is_dir() else source
    if not answers.is_file():
        raise CorpusMissingError(
            f"no consistency answers at {answers}; pass --consistency <run dir> "
            "(reports/consistency-p0 is the committed run)"
        )
    instances: list[ConsistencyInstance] = []
    seen: set[tuple[str, str]] = set()
    for line in answers.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("variant") != "clean":
            continue
        locale = str(row.get("locale") or "")
        fact_id = str(row.get("fact_id") or "")
        if (locale, fact_id) in seen:
            continue
        seen.add((locale, fact_id))
        instances.append(
            ConsistencyInstance(
                instance_id=f"{fact_id}:{locale}",
                locale=locale,
                fact_id=fact_id,
                archetype=str(row.get("archetype") or ""),
                question=str(row.get("question") or ""),
                gold_doc_ids=tuple(str(value) for value in row.get("gold_doc_ids") or ()),
            )
        )
    instances.sort(key=lambda item: (item.locale, item.fact_id))
    return instances


def _set_digest(instances: Sequence[ConsistencyInstance]) -> str:
    digest = hashlib.sha256()
    for instance in instances:
        digest.update(f"{instance.instance_id}\x00{instance.question}\n".encode())
    return digest.hexdigest()


def _surface_variants(question: str) -> dict[str, str]:
    """Meaning-preserving surface variants of the same question.

    Case folding is *language-sensitive*: Turkish upper-casing maps ``ı`` → ``I`` and ``i`` →
    ``I``, so a naive ``.upper()`` silently changes the word (``kartın`` → ``kartin``) and that
    is a different query, not a surface variant. Lower-casing is safe for every shipped locale
    here (the corpus questions are already lower-case except for the first word), so the
    case variant lowers the whole question; the punctuation variant keeps the original case and
    only changes spacing and trailing punctuation.
    """
    lowered = question.lower()
    return {
        "case": f"  {lowered}  ?",
        "punctuation": f"\t{question}  !",
    }


def _packed_digest(packed_per_instance: dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for instance_id in sorted(packed_per_instance):
        digest.update(instance_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update("|".join(packed_per_instance[instance_id]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pipeline(
    bundle: IndexBundle,
    config: PipelineConfig,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        bm25=bundle.bm25,
        dense=bundle.dense,
        fuser=FusionParams(fuser="rrf", rrf_k=60).build(),
        config=config,
    )


def _measure_condition(
    pipeline: RetrievalPipeline,
    instances: Sequence[ConsistencyInstance],
    *,
    window: int,
) -> dict[str, Any]:
    """Run the pipeline for every instance and aggregate purity + recall."""
    per_locale: dict[str, dict[str, float]] = defaultdict(
        lambda: {"chunks": 0.0, "same": 0.0, "instances": 0.0, "pure": 0.0, "pure90": 0.0}
    )
    packed_per_instance: dict[str, list[str]] = {}
    total_chunks = 0
    same_chunks = 0
    pure_instances = 0
    pure90_instances = 0
    recall20 = 0
    recall10 = 0
    for instance in instances:
        trace = pipeline.retrieve(instance.question, final_k=window)
        packed = trace.packed
        packed_per_instance[instance.instance_id] = [passage.chunk.chunk_id for passage in packed]
        languages = [passage.chunk.language or "" for passage in packed]
        same = sum(1 for language in languages if language == instance.locale)
        total_chunks += len(languages)
        same_chunks += same
        row = per_locale[instance.locale]
        row["chunks"] += len(languages)
        row["same"] += same
        row["instances"] += 1
        if packed and same == len(languages):
            pure_instances += 1
        if packed and same / len(languages) >= 0.9:
            pure90_instances += 1
        row["pure"] += 1.0 if packed and same == len(languages) else 0.0
        row["pure90"] += 1.0 if packed and same / len(languages) >= 0.9 else 0.0

        gold = set(instance.gold_doc_ids)
        fused20 = {passage.chunk.doc_id for passage in trace.fused[:20]}
        fused10 = {passage.chunk.doc_id for passage in trace.fused[:10]}
        recall20 += 1 if gold & fused20 else 0
        recall10 += 1 if gold & fused10 else 0

    n = len(instances)
    return {
        "chunk_purity": round(same_chunks / total_chunks, 4) if total_chunks else 0.0,
        "instance_pure": round(pure_instances / n, 4) if n else 0.0,
        "instance_pure_90": round(pure90_instances / n, 4) if n else 0.0,
        "recall_any_at_20": round(recall20 / n, 4) if n else 0.0,
        "recall_any_at_10": round(recall10 / n, 4) if n else 0.0,
        "instances": n,
        "chunks": total_chunks,
        "per_locale": {
            locale: {
                "instances": int(values["instances"]),
                "chunk_purity": round(values["same"] / values["chunks"], 4)
                if values["chunks"]
                else 0.0,
                "instance_pure": round(values["pure"] / values["instances"], 4)
                if values["instances"]
                else 0.0,
                "instance_pure_90": round(values["pure90"] / values["instances"], 4)
                if values["instances"]
                else 0.0,
            }
            for locale, values in sorted(per_locale.items())
        },
        "packed_digest": _packed_digest(packed_per_instance),
        "packed": packed_per_instance,
    }


def _measure_invariance(
    pipeline: RetrievalPipeline,
    instances: Sequence[ConsistencyInstance],
    clean_packed: dict[str, list[str]],
    *,
    window: int,
) -> dict[str, Any]:
    """Every instance asked in a different surface form must pack the same chunks."""
    per_variant: dict[str, dict[str, Any]] = {}
    for variant_name in ("case", "punctuation"):
        matched = 0
        mismatches: list[str] = []
        for instance in instances:
            trace = pipeline.retrieve(
                _surface_variants(instance.question)[variant_name], final_k=window
            )
            variant_ids = [passage.chunk.chunk_id for passage in trace.packed]
            if variant_ids == clean_packed.get(instance.instance_id):
                matched += 1
            else:
                mismatches.append(instance.instance_id)
        per_variant[variant_name] = {
            "matched": matched,
            "mismatches": mismatches[:10],
        }
    n = len(instances)
    total_matched = sum(block["matched"] for block in per_variant.values())
    total = n * len(per_variant)
    return {
        "instances": n,
        "variants": per_variant,
        "matched": total_matched,
        "total": total,
        "rate": round(total_matched / total, 4) if total else 0.0,
    }


def _packed_recall(
    pipeline: RetrievalPipeline,
    instances: Sequence[ConsistencyInstance],
    *,
    window: int,
) -> float:
    """Share of instances whose packed window contains a gold document."""
    hits = 0
    for instance in instances:
        trace = pipeline.retrieve(instance.question, final_k=window)
        if {passage.chunk.doc_id for passage in trace.packed} & set(instance.gold_doc_ids):
            hits += 1
    return round(hits / len(instances), 4) if instances else 0.0


def _measure_ablation_guard(
    bundle: IndexBundle,
    cases: Sequence[Any],
    *,
    window: int,
    top_k: int,
) -> dict[str, Any]:
    """Recall@20 per arm on clean direct questions, locale policy off vs on."""
    from answerdesk.retrieval import ablation as ablation_module

    selected = [case for case in cases if case.category in CLEAN_CATEGORIES][:600]
    gold = [case.gold_chunk_ids for case in selected]
    arms = (
        ablation_module.ArmSpec(name="bm25", fuser="none", channels=("bm25",)),
        ablation_module.ArmSpec(name="dense", fuser="none", channels=("dense",)),
        ablation_module.ArmSpec(name="hybrid-rrf", fuser="rrf"),
    )
    results: dict[str, dict[str, float]] = {}
    for condition, flag in (("off", False), ("on", True)):
        config = PipelineConfig(top_k=top_k, gate_k=20, final_k=window, locale_filter=flag)
        pipeline = _pipeline(bundle, config)
        cache = ablation_module._candidate_sets(pipeline, selected, top_k=top_k)
        for arm in arms:
            rankings = [
                ablation_module._ranked_ids(
                    arm,
                    candidates=cache[case.case_id],
                    params=FusionParams(fuser="rrf", rrf_k=60),
                    config=config,
                    reranker=ablation_module.LexicalOverlapReranker(),
                    query=case.query,
                )
                for case in selected
            ]
            metrics = ablation_module.evaluate_cases(rankings, gold, 20)
            results.setdefault(condition, {})[arm.name] = round(metrics.recall_any, 4)
    deltas = {arm: round(results["on"][arm] - results["off"][arm], 4) for arm in results["off"]}
    return {
        "n": len(selected),
        "off": results["off"],
        "on": results["on"],
        "delta": deltas,
        "tolerance": ABLATION_TOLERANCE,
        "passed": all(value >= -ABLATION_TOLERANCE for value in deltas.values()),
    }


def run_measurement(
    *,
    corpus_dir: Path | str | None = None,
    consistency_run: Path | str = DEFAULT_CONSISTENCY_RUN,
    window: int = DEFAULT_WINDOW,
    seed: int = 1337,
) -> dict[str, Any]:
    """Produce the whole locale-filter report as a JSON-serialisable dict."""
    started = time.perf_counter()
    settings = load_settings()
    corpus = (
        Path(corpus_dir)
        if corpus_dir is not None
        else settings.resolve_path(settings.app.corpus_dir)
    )
    instances = load_consistency_set(consistency_run)
    if not instances:
        raise CorpusMissingError(f"the consistency set at {consistency_run} is empty")
    store = ChunkStore.from_corpus(corpus)
    bundle = IndexBundle.build(store, default_embedder(settings))
    cases = build_cases(store.chunks, load_query_rows(corpus))

    off_config = PipelineConfig(top_k=50, gate_k=20, final_k=window, locale_filter=False)
    on_config = PipelineConfig(top_k=50, gate_k=20, final_k=window, locale_filter=True)

    before = _measure_condition(_pipeline(bundle, off_config), instances, window=window)
    after = _measure_condition(_pipeline(bundle, on_config), instances, window=window)
    packed_recall_before = _packed_recall(_pipeline(bundle, off_config), instances, window=window)
    packed_recall_after = _packed_recall(_pipeline(bundle, on_config), instances, window=window)
    repeat = _measure_condition(_pipeline(bundle, on_config), instances, window=window)

    invariance = _measure_invariance(
        _pipeline(bundle, on_config), instances, after["packed"], window=window
    )
    guard = _measure_ablation_guard(bundle, cases, window=window, top_k=50)

    def _drop_packed(block: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in block.items() if key != "packed"}

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "corpus_version": store.corpus_version,
        "corpus_chunk_hash": store.fingerprint(),
        "consistency_run": Path(consistency_run).as_posix(),
        "consistency_set": {
            "instances": len(instances),
            "locales": sorted({instance.locale for instance in instances}),
            "digest": _set_digest(instances),
        },
        "window": window,
        "policy": {
            "enabled_by_default": True,
            "min_same_ratio": on_config.locale_min_same_ratio,
            "min_confidence": on_config.locale_min_confidence,
            "description": "same-locale first, cross-locale fallback when the window cannot be "
            "filled from the query's locale",
        },
        "purity": {
            "target": PURITY_TARGET,
            "before": _drop_packed(before),
            "after": _drop_packed(after),
            "delta": round(after["chunk_purity"] - before["chunk_purity"], 4),
            "passed": after["chunk_purity"] >= PURITY_TARGET,
        },
        "packed_recall": {
            "before": packed_recall_before,
            "after": packed_recall_after,
            "delta": round(packed_recall_after - packed_recall_before, 4),
            "tolerance": ABLATION_TOLERANCE,
            # A regression gate, not a symmetry gate: improving is fine, losing evidence is not.
            "passed": (packed_recall_after - packed_recall_before) >= -ABLATION_TOLERANCE,
        },
        "invariance": invariance,
        "ablation_guard": guard,
        "determinism": {
            "digest_match": after["packed_digest"] == repeat["packed_digest"],
            "digest": after["packed_digest"],
        },
        "seed": seed,
        "timings_ms": {"total": round((time.perf_counter() - started) * 1000.0, 3)},
    }
    gates = {
        "purity": report["purity"]["passed"],
        "invariance": invariance["rate"] >= INVARIANCE_TARGET,
        "ablation_guard": guard["passed"],
        "packed_recall": report["packed_recall"]["passed"],
        "determinism": report["determinism"]["digest_match"],
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """The committed human-readable evidence: before/after, per locale, gates."""
    purity = report["purity"]
    before = purity["before"]
    after = purity["after"]
    lines = [
        "# Locale-aware retrieval (CNS-6): before / after",
        "",
        f"- corpus: `{report['corpus_version']}` (chunk hash `{report['corpus_chunk_hash'][:16]}`)",
        f"- query set: consistency run `{report['consistency_run']}` — "
        f"{report['consistency_set']['instances']} clean instances, "
        f"{len(report['consistency_set']['locales'])} locales, "
        f"digest `{report['consistency_set']['digest'][:16]}`",
        f"- packed context window: {report['window']} chunks",
        f"- policy: same-locale first, cross-locale fallback when the window cannot be filled "
        f"(min same-locale share {report['policy']['min_same_ratio']}, min language confidence "
        f"{report['policy']['min_confidence']})",
        "",
        "## Evidence-locale purity",
        "",
        "| metric | before (policy off) | after (policy on) | Δ |",
        "|---|---|---|---|",
        f"| chunk purity (pooled) | {before['chunk_purity']:.3f} | {after['chunk_purity']:.3f} "
        f"| {purity['delta']:+.3f} |",
        f"| instances fully same-locale | {before['instance_pure']:.3f} "
        f"| {after['instance_pure']:.3f} "
        f"| {after['instance_pure'] - before['instance_pure']:+.3f} |",
        f"| instances ≥90% same-locale | {before['instance_pure_90']:.3f} "
        f"| {after['instance_pure_90']:.3f} "
        f"| {after['instance_pure_90'] - before['instance_pure_90']:+.3f} |",
        f"| `recall_any@20` (doc-level) | {before['recall_any_at_20']:.3f} "
        f"| {after['recall_any_at_20']:.3f} "
        f"| {after['recall_any_at_20'] - before['recall_any_at_20']:+.3f} |",
        f"| `recall_any@10` (doc-level) | {before['recall_any_at_10']:.3f} "
        f"| {after['recall_any_at_10']:.3f} "
        f"| {after['recall_any_at_10'] - before['recall_any_at_10']:+.3f} |",
        "",
        f"Target: chunk purity ≥ **{purity['target']:.2f}** — "
        f"**{'PASS' if purity['passed'] else 'FAIL'}**.",
        "",
        "## Per locale",
        "",
        "| locale | instances | chunk purity before | chunk purity after | fully pure before "
        "| fully pure after |",
        "|---|---|---|---|---|---|",
    ]
    for locale, after_row in after["per_locale"].items():
        before_row = before["per_locale"].get(locale, {})
        lines.append(
            f"| {locale} | {after_row['instances']} | {before_row.get('chunk_purity', 0):.3f} "
            f"| {after_row['chunk_purity']:.3f} | {before_row.get('instance_pure', 0):.3f} "
            f"| {after_row['instance_pure']:.3f} |"
        )
    invariance = report["invariance"]
    guard = report["ablation_guard"]
    lines.extend(
        [
            "",
            "## Guards",
            "",
            f"- surface-form invariance: {invariance['matched']}/{invariance['instances']} "
            f"(rate {invariance['rate']:.3f}) — same packed context for a casings/spacing/"
            f"punctuation variant",
            "- ablation non-regression (`recall_any@20` on clean `direct` questions, "
            f"n={guard['n']}):",
            "",
            "| arm | policy off | policy on | Δ |",
            "|---|---|---|---|",
        ]
    )
    for arm, value in guard["off"].items():
        lines.append(
            f"| {arm} | {value:.3f} | {guard['on'][arm]:.3f} | {guard['delta'][arm]:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"Tolerance: each arm must not drop more than {guard['tolerance']:.2f} — "
            f"**{'PASS' if guard['passed'] else 'FAIL'}**.",
            f"- determinism: packed-context digest identical across two runs — "
            f"**{'PASS' if report['determinism']['digest_match'] else 'FAIL'}**",
            "",
            "## Gates",
            "",
        ]
    )
    for gate, ok in report["gates"].items():
        lines.append(f"- {gate}: {'PASS' if ok else 'FAIL'}")
    lines.append("")
    lines.append(f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**")
    return "\n".join(lines) + "\n"


def write_report(
    report: dict[str, Any], out_dir: Path | str = DEFAULT_OUTPUT_DIR
) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` (the committed evidence)."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    markdown_path = target / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """``--check`` measures, writes the report and exits non-zero on any failed gate."""
    parser = argparse.ArgumentParser(
        prog="python -m answerdesk.retrieval.locale_report",
        description="Measure evidence-locale purity before/after the locale policy (CNS-6).",
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero on a failed gate")
    parser.add_argument("--corpus", type=Path, default=None, help="corpus directory")
    parser.add_argument(
        "--consistency",
        type=Path,
        default=DEFAULT_CONSISTENCY_RUN,
        help="consistency run directory (its clean instances define the query set)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="artifact directory")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="packed context size")
    parser.add_argument("--seed", type=int, default=1337, help="recorded seed (no randomness used)")
    args = parser.parse_args(argv)

    report = run_measurement(
        corpus_dir=args.corpus,
        consistency_run=args.consistency,
        window=args.window,
        seed=args.seed,
    )
    json_path, markdown_path = write_report(report, args.out)
    purity = report["purity"]
    print(
        f"locale purity: {purity['before']['chunk_purity']:.3f} -> "
        f"{purity['after']['chunk_purity']:.3f} "
        f"(target {purity['target']:.2f}, {'PASS' if purity['passed'] else 'FAIL'})"
    )
    for gate, ok in report["gates"].items():
        print(f"  {gate}: {'PASS' if ok else 'FAIL'}")
    print(f"artifacts: {json_path} , {markdown_path}")
    if args.check and not report["passed"]:
        print("locale filter check failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())


__all__ = [
    "ABLATION_TOLERANCE",
    "DEFAULT_CONSISTENCY_RUN",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_WINDOW",
    "INVARIANCE_TARGET",
    "PURITY_TARGET",
    "ConsistencyInstance",
    "load_consistency_set",
    "main",
    "render_markdown",
    "run_measurement",
    "write_report",
]

# The policy is imported for its defaults in the report metadata; keeping the import explicit
# documents that the report measures exactly the shipped policy.
_ = LocalePolicy
