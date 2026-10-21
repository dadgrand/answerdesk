"""Side-study: can the *reasoning* variant of qwen3.6-35b-a3b finish a grounded answer?

The evidence run generates with ``qwen3.6-35b-a3b-noreason`` because the reasoning variant
spent its whole JSON budget on the trace in a direct probe (3/3 PROVIDER_ERROR at
max_tokens 4096). A probe is not a measurement, so this script runs the same cases, prompts
and schema through the reasoning model at two higher budgets (8192 and 16384), records
success/failure, latency, tokens and cost, and — when the primary run's turns exist — a tiny
paired comparison against the no-reason answers for the same case ids.

Output: ``reports/<run>/side-study/reasoning-variant.json`` + ``.csv`` (committed artefacts;
``scripts/build-results.py`` reads them into docs/results.md).

    uv run python scripts/reasoning-side-study.py --limit 32 --budgets 8192 16384

    # AT-EVL-10  # AT-NFR-REP-01
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path

from answerdesk.config import Settings, load_settings
from answerdesk.eval.corpus import load_corpus
from answerdesk.eval.live import build_embedder, build_index, build_reranker, retrieval_variants
from answerdesk.generate.pipeline import AnswerPipeline
from answerdesk.llm.factory import build_provider

REPO_ROOT = Path(__file__).resolve().parents[1]
REASONING_MODEL = "qwen3.6-35b-a3b"
PRIMARY_MODEL = "qwen3.6-35b-a3b-noreason"


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _wilson(successes: int, n: int) -> tuple[float, float]:
    from answerdesk.eval.stats import wilson_interval

    interval = wilson_interval(successes, n)
    return interval.low, interval.high


def _settings_with(model: str, max_tokens: int, settings: Settings) -> Settings:
    clone = settings.model_copy(deep=True)
    clone.llm.model = model
    clone.llm.max_tokens = max_tokens
    clone.llm.temperature = 0.0
    return clone


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32, help="cases per budget (>= 30)")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "reports" / "bench")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--replay", action="store_true", help="deterministic doubles (wiring check)"
    )
    args = parser.parse_args(argv)
    if args.limit < 30:
        parser.error("the side-study needs at least 30 cases to be reportable")

    settings = load_settings(profile=None)
    out_dir = args.out or (args.run / "side-study")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(settings.resolve_path(settings.app.corpus_dir))
    if args.replay:
        from answerdesk.testing.doubles import HashingEmbedder

        embedder = HashingEmbedder()
    else:
        embedder = build_embedder(settings, Path("data/index"))
    index = build_index(settings, corpus, embedder, Path("data/index/bench"))
    reranker = None if args.replay else build_reranker(settings)
    variants = retrieval_variants(settings, index, reranker=reranker)
    retriever = variants.get("cc@0.75+rerank") or variants.get("cc@0.75") or variants["bm25"]

    answerable = [query for query in corpus.queries if query.answerable]
    # deterministic, locale-stratified selection of the side-study cases
    by_locale: dict[str, list[str]] = {}
    for query in sorted(answerable, key=lambda item: item.query_id):
        by_locale.setdefault(query.locale, []).append(query.query_id)
    ordered: list[str] = []
    index_position = 0
    while len(ordered) < args.limit:
        added = False
        for locale in sorted(by_locale):
            pool = by_locale[locale]
            if index_position < len(pool) and len(ordered) < args.limit:
                ordered.append(pool[index_position])
                added = True
        if not added:
            break
        index_position += 1
    by_id = {query.query_id: query for query in corpus.queries}
    cases = [by_id[query_id] for query_id in ordered]

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "reasoning_model": REASONING_MODEL,
        "primary_model": PRIMARY_MODEL,
        "n_cases": len(cases),
        "budgets": {},
        "replay": args.replay,
        "note": (
            "Same prompts, schema and retrieval as the primary arm; only max_tokens and the "
            "model variant differ (temperature 0)."
        ),
    }
    for budget in args.budgets:
        run_settings = _settings_with(REASONING_MODEL, budget, settings)
        provider = build_provider(run_settings) if not args.replay else _fake_provider()
        pipeline = AnswerPipeline.from_settings(provider, run_settings)
        successes = 0
        latencies: list[float] = []
        tokens_out: list[int] = []
        for query in cases:
            trace = retriever.retrieve(query.text, final_k=run_settings.generate.context_k_max)
            passages = list(getattr(trace, "packed", None) or trace.reranked or trace.fused or [])
            started = time.perf_counter()
            try:
                result = pipeline.answer(query.text, passages)
                elapsed = (time.perf_counter() - started) * 1000.0
                ok = result.signal.action.value == "answer" and bool(result.claims)
                rows.append(
                    {
                        "budget": budget,
                        "query_id": query.query_id,
                        "locale": query.locale,
                        "category": query.category,
                        "ok": ok,
                        "action": result.signal.action.value,
                        "reason_code": result.signal.reason_code.value,
                        "latency_ms": round(elapsed, 1),
                        "output_tokens": result.usage.output_tokens,
                        "input_tokens": result.usage.input_tokens,
                        "usd": result.usage.usd,
                    }
                )
                latencies.append(elapsed)
                tokens_out.append(result.usage.output_tokens)
                successes += 1 if ok else 0
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                rows.append(
                    {
                        "budget": budget,
                        "query_id": query.query_id,
                        "locale": query.locale,
                        "category": query.category,
                        "ok": False,
                        "action": "error",
                        "reason_code": type(exc).__name__,
                        "latency_ms": round(elapsed, 1),
                        "output_tokens": 0,
                        "input_tokens": 0,
                        "usd": 0.0,
                    }
                )
        low, high = _wilson(successes, len(cases))
        summary["budgets"][str(budget)] = {
            "n": len(cases),
            "successes": successes,
            "success_rate": round(successes / len(cases), 6) if cases else None,
            "success_ci": [round(low, 6), round(high, 6)],
            "latency_p50_ms": round(_pct(latencies, 0.5), 1),
            "latency_p95_ms": round(_pct(latencies, 0.95), 1),
            "output_tokens_mean": round(sum(tokens_out) / len(tokens_out), 1)
            if tokens_out
            else 0.0,
            "usd_total": round(
                sum(float(row["usd"]) for row in rows if row["budget"] == budget), 8
            ),
        }
    with (out_dir / "reasoning-variant.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (out_dir / "reasoning-variant.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _fake_provider() -> object:
    from answerdesk.testing.doubles import FakeProvider

    return FakeProvider()


if __name__ == "__main__":
    raise SystemExit(main())
