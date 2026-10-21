"""Runner for the consistency audit: build the set, audit the corpus, measure, report.

Two execution modes, both reproducible from one command:

* ``--replay`` — deterministic doubles (FakeProvider / FakeJudge), no network, used by CI;
* live — qwen3.6-35b-a3b answers with the cross-family deepseek-flash judge; both model ids
  are recorded in every artefact.

Artefacts land in ``reports/consistency-<run>/``: the transcreated set, per-row answers, the
retrieval-level invariance table, the corpus audit and a rendered report.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.config import Settings, load_settings
from answerdesk.eval.consistency import agreement as metrics
from answerdesk.eval.consistency.facts import LOCALES, build_groups, load_set, write_set
from answerdesk.eval.consistency.logical import audit_corpus
from answerdesk.eval.consistency.values import fidelity, language_discipline
from answerdesk.eval.consistency.variants import variants_for

DEFAULT_RUN = "consistency-p0"
SET_DIR = Path("evals") / "consistency" / "v1"


@dataclass
class ConsistencyConfig:
    run_id: str = DEFAULT_RUN
    report_dir: Path = Path("reports") / DEFAULT_RUN
    corpus_dir: Path = Path("data/corpus")
    set_dir: Path = SET_DIR
    locales: tuple[str, ...] = LOCALES
    facts_limit: int | None = None
    answer_limit: int | None = None
    variant_locales: tuple[str, ...] = ("ru",)
    provider: str = "fake"
    replay: bool = False
    samples: int = 1000
    seed: int = 0
    judge: bool = True
    retriever: str = "cc@0.75"
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def corpus_version(corpus_dir: Path) -> str:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("corpus_version", "unknown"))


def build_set(config: ConsistencyConfig) -> dict[str, Any]:
    groups = build_groups(config.corpus_dir, limit=config.facts_limit, locales=config.locales)
    manifest = write_set(groups, config.set_dir, corpus_version=corpus_version(config.corpus_dir))
    _write_json(config.report_dir / "set-manifest.json", manifest)
    return manifest


def _corpus_documents(corpus_dir: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for path in sorted(corpus_dir.glob("docs/*/*.md")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    return documents


def _provider(settings: Settings, config: ConsistencyConfig) -> tuple[Any, str]:
    from answerdesk.llm.factory import build_provider

    if config.replay or config.provider == "fake":
        from answerdesk.testing.doubles import FakeProvider

        return FakeProvider(), "fake-model"
    return build_provider(settings), settings.resolved_provider().model


def _judge(settings: Settings, config: ConsistencyConfig) -> tuple[Any, str]:
    if config.replay or not config.judge:
        from answerdesk.testing.doubles import FakeJudge

        return FakeJudge(), "fake-judge"
    from answerdesk.eval.judge import RubricJudge
    from answerdesk.eval.live import judge_provider

    provider = judge_provider(settings)
    assert provider is not None
    model = settings.eval.judge_model or "judge"
    return RubricJudge(provider, judge_model=model), model


def _answer(
    *, pipeline: Any, question: str, gold: Sequence[str], retriever: Any, k: int
) -> dict[str, Any]:
    trace = retriever.retrieve(question, final_k=k)
    packed = list(getattr(trace, "packed", None) or trace.reranked or trace.fused or [])
    result = pipeline.answer(question, packed)
    # claim citations are indices into the packed passage list
    chunk_ids = [passage.chunk.chunk_id for passage in packed]
    citations: set[str] = set()
    for claim in result.claims:
        for citation in claim.citation_ids:
            if isinstance(citation, int) and 0 <= citation < len(chunk_ids):
                citations.add(chunk_ids[citation])
            elif isinstance(citation, str) and citation in chunk_ids:
                citations.add(citation)
    doc_ids = sorted({chunk_id.split("::", 1)[0] for chunk_id in citations})
    return {
        "decision": result.signal.action.value,
        "reason_code": result.signal.reason_code.value,
        "answer": " ".join(claim.text for claim in result.claims),
        "citations": sorted(citations),
        "evidence_doc_ids": doc_ids,
        "retrieved_doc_ids": sorted({passage.chunk.doc_id for passage in packed}),
        "claims": len(result.claims),
        "input_tokens": int(result.usage.input_tokens),
        "output_tokens": int(result.usage.output_tokens),
        "usd": float(result.usage.usd),
    }


def run_answers(config: ConsistencyConfig, *, settings: Settings | None = None) -> dict[str, Any]:
    """Clean-form answers for every fact and locale, plus the four non-clean variants for the
    configured locales; resumable through the answers checkpoint."""
    from answerdesk.eval.corpus import load_corpus
    from answerdesk.eval.judge import judge_metadata
    from answerdesk.eval.live import build_embedder, build_index, build_reranker, retrieval_variants
    from answerdesk.generate.pipeline import AnswerPipeline

    settings = settings or load_settings(profile=None)
    facts = load_set(config.set_dir / "facts.jsonl")
    if config.answer_limit:
        facts = facts[: config.answer_limit]
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    checkpoint = config.report_dir / "answers.jsonl"
    done = {(row["fact_id"], row["locale"], row["variant"]) for row in _read_rows(checkpoint)}
    provider, generator_model = _provider(settings, config)
    judge, judge_model = _judge(settings, config)
    pipeline = AnswerPipeline.from_settings(provider, settings)
    if config.replay:
        variants = {"bm25": _fake_retriever(corpus), "cc@0.75": _fake_retriever(corpus)}
    else:
        embedder = build_embedder(settings, Path("data/index"))
        index = build_index(settings, corpus, embedder, Path("data/index/bench"))
        variants = retrieval_variants(settings, index, reranker=build_reranker(settings))
    retriever = variants.get(config.retriever) or next(iter(variants.values()))
    documents = _corpus_documents(config.corpus_dir)
    judge_meta = judge_metadata(
        judge_model=judge_model,
        prompt_version="judge-rubric-1",
        rubric_version="rubric-v1",
        generator_model=generator_model,
        judge_family=settings.eval.judge_family,
    )
    rows: list[dict[str, Any]] = []
    for fact in facts:
        for locale, instance in sorted(fact["instances"].items()):
            if locale not in config.locales:
                continue
            forms = [("clean", instance["question"])]
            if locale in config.variant_locales:
                forms = [
                    (variant.kind, variant.text)
                    for variant in variants_for(locale, instance["question"], seed=config.seed)
                ]
            for kind, question in forms:
                key = (fact["fact_id"], locale, kind)
                if key in done:
                    continue
                try:
                    measured = _answer(
                        pipeline=pipeline,
                        question=question,
                        gold=instance["gold_doc_ids"],
                        retriever=retriever,
                        k=settings.generate.context_k_max,
                    )
                except Exception as exc:  # a provider failure is recorded, not invented
                    row = {
                        "fact_id": fact["fact_id"],
                        "archetype": fact["archetype"],
                        "locale": locale,
                        "variant": kind,
                        "question": question,
                        "decision": "provider_error",
                        "reason_code": type(exc).__name__,
                        "answer": "",
                        "citations": [],
                        "evidence_doc_ids": [],
                        "retrieved_doc_ids": [],
                        "gold_doc_ids": instance["gold_doc_ids"],
                        "language_expected": locale,
                        "language_detected": "und",
                        "language_ok": False,
                        "language_detected_ok": False,
                        "evidence_locale_ok": False,
                        "citation_hit": False,
                        "citation_precision": 0.0,
                        "citation_recall": 0.0,
                        "fidelity": 0.0,
                        "fidelity_vs_cited": 0.0,
                        "values_asserted": 0,
                        "values_unsupported": [],
                        "values_unsupported_vs_cited": [],
                        "same_fact": False,
                        "rubric": None,
                        "judge_failed": False,
                        "provider_error": True,
                        "generator_model": generator_model,
                        "judge_model": judge_model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "usd": 0.0,
                    }
                    rows.append(row)
                    _append_rows(checkpoint, [row])
                    continue
                gold_text = " ".join(
                    documents.get(doc_id, "") for doc_id in instance["gold_doc_ids"]
                )
                cited_text = " ".join(
                    documents.get(doc_id, "") for doc_id in measured["evidence_doc_ids"]
                )
                check = fidelity(measured["answer"], gold_text)
                check_cited = fidelity(measured["answer"], cited_text)
                language = language_discipline(
                    query_locale=locale,
                    answer=measured["answer"],
                    answer_language_setting=locale,
                )
                cite = metrics.citation_overlap(
                    cited=measured["evidence_doc_ids"], gold=instance["gold_doc_ids"]
                )
                same_fact = False
                rubric = None
                # only the clean form is judged: the variant forms are scored by decision and
                # evidence invariance, which are deterministic and need no judge
                if kind == "clean" and judge is not None and measured["answer"].strip():
                    try:
                        score = judge.grade(
                            question=instance["question"],
                            reference_answer=instance["reference_answer"],
                            candidate_answer=measured["answer"],
                            context=[gold_text[:4000]] if gold_text else [],
                        )
                        rubric = score.score
                        same_fact = bool(score.score >= 4)
                    except Exception:  # a judge failure is recorded, never invented
                        rubric = None
                row = {
                    "fact_id": fact["fact_id"],
                    "archetype": fact["archetype"],
                    "locale": locale,
                    "variant": kind,
                    "question": question,
                    "decision": measured["decision"],
                    "reason_code": measured["reason_code"],
                    "answer": measured["answer"],
                    "citations": measured["citations"],
                    "evidence_doc_ids": measured["evidence_doc_ids"],
                    "retrieved_doc_ids": measured["retrieved_doc_ids"],
                    "gold_doc_ids": instance["gold_doc_ids"],
                    "language_expected": language["expected"],
                    "language_detected": language["detected"],
                    "language_ok": bool(language["ok"]),
                    "language_detected_ok": bool(language["detected"] == locale),
                    "evidence_locale_ok": all(
                        doc_id.split("-")[1] == locale
                        for doc_id in measured["evidence_doc_ids"]
                        if doc_id.startswith("syn-")
                    ),
                    "citation_hit": bool(cite["hit"]),
                    "citation_precision": cite["precision"],
                    "citation_recall": cite["recall"],
                    "fidelity": check["fidelity"],
                    "fidelity_vs_cited": check_cited["fidelity"],
                    "values_unsupported_vs_cited": check_cited["unsupported"],
                    "values_asserted": check["asserted_count"],
                    "values_unsupported": check["unsupported"],
                    "same_fact": same_fact,
                    "rubric": rubric,
                    "judge_failed": (
                        kind == "clean" and rubric is None and bool(measured["answer"].strip())
                    ),
                    "generator_model": generator_model,
                    "judge_model": judge_model,
                    "input_tokens": measured["input_tokens"],
                    "output_tokens": measured["output_tokens"],
                    "usd": measured["usd"],
                }
                rows.append(row)
                _append_rows(checkpoint, [row])
    summary = summarise(
        config,
        rows=_read_rows(checkpoint),
        generator_model=generator_model,
        judge_model=judge_model,
        judge_meta=judge_meta.model_dump(mode="json"),
    )
    return summary


class _fake_retriever:
    """Deterministic lexical retriever for offline runs (corpus chunks, no embeddings)."""

    def __init__(self, corpus: Any) -> None:
        self.corpus = corpus
        self.chunks = list(corpus.chunks)

    def retrieve(self, question: str, final_k: int = 12) -> Any:
        from dataclasses import dataclass, field

        from answerdesk.contracts import Channel, RetrievedPassage

        @dataclass
        class Trace:
            packed: list[Any] = field(default_factory=list)

        tokens = {token.lower() for token in question.split() if len(token) > 2}
        scored = []
        for chunk in self.chunks:
            overlap = len(tokens & {token.lower() for token in chunk.text.split()})
            if overlap:
                scored.append((overlap, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        packed = [
            RetrievedPassage(chunk=chunk, channel=Channel.BM25, score=float(score), rank=rank)
            for rank, (score, chunk) in enumerate(scored[:final_k], start=1)
        ]
        return Trace(packed=packed)


def summarise(
    config: ConsistencyConfig,
    *,
    rows: Sequence[dict[str, Any]],
    generator_model: str,
    judge_model: str,
    judge_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean = [row for row in rows if row["variant"] == "clean"]
    variant_rows = [row for row in rows if row["variant"] != "clean"]
    per_language: dict[str, Any] = {}
    for locale in sorted({str(row["locale"]) for row in clean}):
        subset = [row for row in clean if row["locale"] == locale]
        per_language[locale] = {
            "answers": len(subset),
            "same_fact": metrics.rate([bool(row["same_fact"]) for row in subset]).to_dict(),
            "citation_hit": metrics.rate([bool(row["citation_hit"]) for row in subset]).to_dict(),
            "citation_precision_mean": round(
                sum(float(row["citation_precision"]) for row in subset) / len(subset), 6
            ),
            "fidelity_exact": metrics.rate(
                [float(row["fidelity"]) == 1.0 for row in subset]
            ).to_dict(),
            "decisions": _decision_counts(subset),
        }
    summary = {
        "run_id": config.run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": (
            f"{platform.system()} {platform.release()} / Python {platform.python_version()}"
        ),
        "replay": config.replay,
        "corpus_version": corpus_version(config.corpus_dir),
        "generator_model": generator_model,
        "judge_model": judge_model,
        "judge": judge_meta or {},
        "set": json.loads((config.set_dir / "manifest.json").read_text(encoding="utf-8"))
        if (config.set_dir / "manifest.json").is_file()
        else {},
        "rows": {"total": len(rows), "clean": len(clean), "variants": len(variant_rows)},
        "fact_agreement": {
            "per_language": per_language,
            "pooled": metrics.rate([bool(row["same_fact"]) for row in clean]).to_dict(),
            "cross_language": metrics.cross_language_pairs(clean),
        },
        "citation_overlap": {
            "per_language": {
                locale: block["citation_hit"] for locale, block in per_language.items()
            },
            "pooled": metrics.rate([bool(row["citation_hit"]) for row in clean]).to_dict(),
        },
        "language_discipline": metrics.language_discipline(clean),
        "value_fidelity": {
            "pooled": metrics.rate([float(row["fidelity"]) == 1.0 for row in clean]).to_dict(),
            "per_language": {
                locale: block["fidelity_exact"] for locale, block in per_language.items()
            },
        },
        "provider_error_rows": sum(1 for row in rows if row.get("provider_error")),
        "surface_invariance": metrics.invariance(
            variant_rows + [row for row in clean if row["locale"] in config.variant_locales]
        ),
        "variants_per_locale": {
            locale: len([row for row in variant_rows if row["locale"] == locale])
            for locale in sorted({str(row["locale"]) for row in variant_rows})
        },
        "judge_failures": sum(1 for row in rows if row["judge_failed"]),
    }
    summary["ambiguity"] = measure_ambiguity(config)
    _write_json(config.report_dir / "summary.json", summary)
    (config.report_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def _decision_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["decision"])] = counts.get(str(row["decision"]), 0) + 1
    return dict(sorted(counts.items()))


def run_retrieval_invariance(config: ConsistencyConfig) -> dict[str, Any]:
    """Evidence invariance for every locale and fact, from lexical retrieval only.

    No model calls: BM25 over corpus v2.1 ranks the clean question and its four surface forms,
    and the top-5 evidence set is compared. This is the broad, cheap half of the invariance
    measurement (n = facts x locales per form); the generated answers add the decision half on
    the configured locales.
    """
    from answerdesk.eval.consistency.variants import variants_for
    from answerdesk.eval.corpus import load_corpus
    from answerdesk.eval.external.bench import Passage
    from answerdesk.eval.external.retrieval import LexicalRanker

    corpus = load_corpus(config.corpus_dir)
    passages = [
        Passage(
            passage_id=chunk.chunk_id,
            language=chunk.language or "",
            text=chunk.text,
            licence="CC0-1.0",
        )
        for chunk in corpus.chunks
    ]
    ranker = LexicalRanker(passages)
    facts = load_set(config.set_dir / "facts.jsonl")
    if config.answer_limit:
        facts = facts[: config.answer_limit]
    rows: list[dict[str, Any]] = []
    for fact in facts:
        for locale, instance in sorted(fact["instances"].items()):
            clean = ranker.rank(instance["question"], k=5)
            clean_docs = {passage_id.split("::", 1)[0] for passage_id in clean.ranked}
            for variant in variants_for(locale, instance["question"], seed=config.seed):
                if variant.kind == "clean":
                    continue
                ranked = ranker.rank(variant.text, k=5)
                docs = {passage_id.split("::", 1)[0] for passage_id in ranked.ranked}
                rows.append(
                    {
                        "fact_id": fact["fact_id"],
                        "locale": locale,
                        "variant": variant.kind,
                        "same_evidence": docs == clean_docs,
                        "clean_docs": sorted(clean_docs),
                        "variant_docs": sorted(docs),
                    }
                )
    _write_json(config.report_dir / "retrieval-invariance.json", {"rows": rows})
    return {"pairs": len(rows), "rows": rows}


def measure_ambiguity(config: ConsistencyConfig) -> dict[str, Any]:
    """How many documents answer the same clean question?

    A direct question whose wording recurs across documents of one archetype is not
    document-discriminating: a correct retriever may legitimately return a sibling document
    with a different slot value. This is measured so the agreement numbers are read correctly.
    """
    import re as _re

    facts = load_set(config.set_dir / "facts.jsonl")
    if config.answer_limit:
        facts = facts[: config.answer_limit]
    documents: dict[str, str] = {}
    for path in sorted(config.corpus_dir.glob("docs/*/*.md")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    siblings: list[dict[str, Any]] = []
    for fact in facts:
        for locale, instance in sorted(fact["instances"].items()):
            if locale not in config.locales:
                continue
            needle = _re.escape(instance["question"].strip().lower())
            matches = [
                doc_id
                for doc_id, text in documents.items()
                if doc_id.startswith(f"syn-{locale}-") and _re.search(needle, text.lower())
            ]
            siblings.append(
                {
                    "fact_id": fact["fact_id"],
                    "locale": locale,
                    "gold_doc_ids": instance["gold_doc_ids"],
                    "documents_with_question": sorted(matches),
                    "siblings": max(0, len(matches) - 1),
                }
            )
    counts = [row["siblings"] for row in siblings]
    return {
        "facts": len(siblings),
        "mean_siblings": round(sum(counts) / len(counts), 3) if counts else 0.0,
        "share_with_siblings": round(sum(1 for count in counts if count > 0) / len(counts), 3)
        if counts
        else 0.0,
        "max_siblings": max(counts) if counts else 0,
        "rows": siblings,
        "definition": (
            "siblings = same-locale documents whose text contains the clean question verbatim"
        ),
    }


def run_audit(config: ConsistencyConfig) -> dict[str, Any]:
    audit = audit_corpus(config.corpus_dir, locales=config.locales)
    _write_json(config.report_dir / "logical-audit.json", audit)
    return audit


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Consistency audit run `{summary.get('run_id')}`",
        "",
        f"Generator: `{summary.get('generator_model')}` · judge: `{summary.get('judge_model')}` "
        f"· replay: {summary.get('replay')} · corpus `{summary.get('corpus_version')}`",
        "",
        "## Fact agreement and citation overlap (clean form)",
        "",
        "| language | n | same fact | citation hit | value fidelity |",
        "|---|---|---|---|---|",
    ]
    per_language = summary["fact_agreement"]["per_language"]
    for locale, block in sorted(per_language.items()):
        lines.append(
            f"| {locale} | {block['answers']} | {block['same_fact']['rate']} "
            f"[{block['same_fact']['ci_low']}, {block['same_fact']['ci_high']}] | "
            f"{block['citation_hit']['rate']} "
            f"[{block['citation_hit']['ci_low']}, {block['citation_hit']['ci_high']}] | "
            f"{block['fidelity_exact']['rate']} |"
        )
    lines.append("")
    surface = summary.get("surface_invariance", {})
    lines += [
        "## Surface-form invariance",
        "",
        f"pairs: {surface.get('pairs')}; evidence {surface.get('evidence_invariance', {})}",
        f"decision {surface.get('decision_invariance', {})}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ConsistencyConfig",
    "build_set",
    "render_report",
    "run_answers",
    "run_audit",
    "summarise",
]
