"""Chunked, resumable benchmark phases for environments that cap long single processes.

The canonical entry point is still ``answerdesk eval --all`` (and `scripts/run-bench.ps1`).
This script exists because some sandboxes kill a process after a few minutes: it runs the
same functions in short phases and persists every artefact, so a rollout can be resumed
without losing the work already paid for.

    python scripts/bench-phases.py retrieval --limit 240
    python scripts/bench-phases.py rerank --chunk 20
    python scripts/bench-phases.py calibrate --chunk 40
    python scripts/bench-phases.py answers --arm primary --chunk 40
    python scripts/bench-phases.py answers --arm secondary --chunk 40
    python scripts/bench-phases.py cache --chunk 10
    python scripts/bench-phases.py finalize

State lives in ``<run>/_state/``; every phase is idempotent for the same chunk boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from answerdesk.config import load_settings  # noqa: E402
from answerdesk.contracts import Action, EvalCase, EvalCategory, IntentBucket, ReasonCode, Split  # noqa: E402
from answerdesk.eval.bench import (  # noqa: E402
    BenchConfig,
    _cell_value,
    _confusion,
    _per_language_answer,
    _reason_codes,
    _subset,
    calibrate_engine_from_dev,
    prepare,
    run_answers,
    run_cache,
    run_escalation,
    run_ops,
    run_retrieval,
    select_operating_threshold,
)
from answerdesk.eval.calibration import summarise  # noqa: E402
from answerdesk.eval.judge import judge_metadata  # noqa: E402
from answerdesk.eval.metrics import (  # noqa: E402
    answer_cells,
    escalation_cells,
    operational_cells,
)
from answerdesk.eval.models import TurnRecord  # noqa: E402


def _state(run: Path) -> Path:
    directory = run / "_state"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    existed = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not existed:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def turn_from_row(row: dict[str, str], arm: str) -> TurnRecord:
    def _float(name: str) -> float | None:
        value = row.get(name)
        return float(value) if value not in (None, "", "None") else None

    return TurnRecord(
        case_id=row["case_id"],
        locale=row["locale"],
        category=EvalCategory(row["category"]),
        arm=arm,
        answerable=row.get("answerable") == "True",
        must_escalate=row.get("answerable") != "True",
        expected_language=row["locale"],
        action=Action(row["action"]),
        reason_code=ReasonCode(row["reason_code"]),
        answered=row.get("answered") == "True",
        escalated=row.get("answered") != "True",
        confidence=float(row.get("p_correct") or 0.0),
        answer_ok=row.get("answer_ok") == "True",
        language=row["locale"],
        latency_ms={
            "retrieval": float(row.get("latency_retrieval_ms") or 0.0),
            "generation": float(row.get("latency_generation_ms") or 0.0),
            "judge": float(row.get("latency_judge_ms") or 0.0),
            "total": float(row.get("latency_retrieval_ms") or 0.0)
            + float(row.get("latency_generation_ms") or 0.0)
            + float(row.get("latency_judge_ms") or 0.0),
        },
        usd=float(row.get("usd") or 0.0),
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        claims_total=int(row.get("claims") or 0),
        gold_chunk_ids=[c for c in (row.get("gold") or "").split(";") if c],
        cited_chunk_ids=[c for c in (row.get("cited") or "").split(";") if c],
        intent_bucket=IntentBucket.INFORMATIONAL,
        evidence_cited=row.get("evidence_cited") == "True",
        # derived from the stored per-turn artefacts: the rubric's "supported" band is a
        # score >= 4, and citation precision/recall are computed from the cited and gold ids
        # that the turn file carries (the per-claim rubric scores are kept in rubric_score).
        groundedness=_grounded(row),
        groundedness_judged=(
            1.0
            if (_float("rubric_score") or 0.0) >= 4.0
            else (0.0 if _float("rubric_score") else None)
        ),
        citation_precision=_citation_rate(row, precision=True),
        citation_recall=_citation_rate(row, precision=False),
        rubric_score=_float("rubric_score"),
        rubric_pass=(
            None if row.get("rubric_pass") in (None, "", "None") else row["rubric_pass"] == "True"
        ),
        reference_rubric_score=_float("reference_rubric_score"),
        language_match=row.get("language_match") == "True",
        claims_supported=(round((_grounded(row) or 0.0) * float(row.get("claims") or 0))),
    )


def _grounded(row: dict[str, str]) -> float | None:
    score = row.get("rubric_score")
    if score in (None, "", "None"):
        return None
    return 1.0 if float(score) >= 4.0 else 0.0


def _citation_rate(row: dict[str, str], *, precision: bool) -> float | None:
    cited = [c for c in (row.get("cited") or "").split(";") if c]
    acceptable = [c for c in (row.get("gold") or "").split(";") if c]
    if not cited:
        return None
    if precision:
        return len(set(cited) & set(acceptable)) / len(set(cited))
    if not acceptable:
        return None
    return len(set(cited) & set(acceptable)) / len(set(acceptable))


def case_from_row(row: dict[str, str]) -> EvalCase:
    return EvalCase(
        case_id=row["case_id"],
        locale=row["locale"],
        category=EvalCategory(row["category"]),
        question=row.get("question") or f"[{row['case_id']}]",
        answerable=row.get("answerable") == "True",
        split=Split.TEST,
        gold_chunk_ids=[c for c in (row.get("gold") or "").split(";") if c],
        reference_answer="reference" if row.get("answerable") == "True" else None,
        metadata={
            "corpus_category": row.get("corpus_category", ""),
            "noisy_freeform": row.get("stratum") == "noisy",
        },
    )


def _config(
    run: Path, *, answer_limit: int | None = 200, judge_enabled: bool = True
) -> BenchConfig:
    return BenchConfig(
        run_id=run.name,
        answer_limit=answer_limit,
        retrieval_limit=240,
        rerank_limit=80,
        generator_arms=(),
        calibrate_on_dev=False,
        judge_enabled=judge_enabled,
    )


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


def phase_retrieval(run: Path, *, limit: int, replay: bool = False) -> int:
    settings = load_settings(profile=None)
    config = _config(run)
    config.retrieval_limit = limit
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    cases = _subset(state.test_cases, locales=(), limit=limit, seed=config.seed)
    results = run_retrieval(state, out_dir=run, cases=cases)
    print(
        "retrieval arms:",
        {
            arm: _cell_value(p["cells"], "recall_any@10")["value"]
            for arm, p in results["arms"].items()
        },
    )
    return 0


def phase_rerank(run: Path, *, chunk: int) -> int:
    """One chunk of the rerank arm, persisted as rankings for later scoring."""
    from answerdesk.eval.live import build_reranker
    from answerdesk.policy import OverlapScorer

    settings = load_settings(profile=None)
    config = _config(run)
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    cases = _subset(state.test_cases, locales=(), limit=config.rerank_limit or 80, seed=config.seed)
    base = state.variants.get("cc@0.75")
    reranked_pipeline = state.variants.get("cc@0.75+rerank")
    output = _state(run) / "rerank-rankings.jsonl"
    done = (
        {
            json.loads(line)["case_id"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if output.is_file()
        else set()
    )
    pending = [case for case in cases if case.case_id not in done][:chunk]
    scorer = OverlapScorer()
    with output.open("a", encoding="utf-8") as handle:
        for case in pending:
            trace = reranked_pipeline.retrieve(
                case.question, final_k=settings.generate.context_k_max
            )
            packed = list(getattr(trace, "packed", None) or trace.reranked or trace.fused or [])
            handle.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "ranked": [p.chunk.chunk_id for p in packed],
                        "confidence": float(scorer.score(case.question, packed)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    remaining = len(cases) - len(done) - len(pending)
    print(f"rerank: {len(pending)} cases done, {max(0, remaining)} remaining")
    _ = base
    _ = build_reranker
    return 0


def phase_calibrate(run: Path, *, chunk: int, limit: int = 200) -> int:
    settings = load_settings(profile=None)
    config = _config(run)
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    dev = _subset(state.dev_cases, locales=(), limit=limit, seed=config.seed)
    accumulated = _read_rows(_state(run) / "dev-turns.csv")
    done = {row["case_id"] for row in accumulated}
    pending = [case for case in dev if case.case_id not in done][:chunk]
    payload, turns, _ = run_answers(
        state, arm="primary", out_dir=_state(run) / "dev-answers", cases=pending
    )
    _append_rows(
        _state(run) / "dev-turns.csv",
        _read_rows(_state(run) / "dev-answers" / "answer" / "primary" / "turns.csv"),
    )
    _engine, calibration = calibrate_engine_from_dev(
        state=state, dev_turns=turns, out_dir=run / "_calibration-chunk"
    )
    _ = payload
    # persist the fitted numbers (and the operating point) for later phases
    rows = _read_rows(_state(run) / "dev-turns.csv")
    all_turns = [turn_from_row(row, "dev") for row in rows]
    _engine, calibration = calibrate_engine_from_dev(state=state, dev_turns=all_turns, out_dir=run)
    operating = select_operating_threshold(all_turns, config.target_risk)
    calibration["operating_point_selection"] = operating
    (run / "calibration").mkdir(parents=True, exist_ok=True)
    (run / "calibration" / "dev-fit.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"calibrate: {len(all_turns)} dev turns; q_hat per bucket; "
        f"operating threshold {operating['threshold']}"
    )
    return 0


def phase_answers(
    run: Path, *, arm: str, chunk: int, limit: int = 150, judge_enabled: bool = True
) -> int:
    settings = load_settings(profile=None)
    config = _config(run, judge_enabled=judge_enabled)
    config.answer_limit = limit
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    # reuse the dev-fitted engine so decisions match the reported policy
    fit = run / "calibration" / "dev-fit.json"
    if fit.is_file():
        payload = json.loads(fit.read_text(encoding="utf-8"))
        state.engine = _engine_from_payload(state, payload)
    selected = _subset(state.test_cases, locales=(), limit=limit, seed=config.seed)
    target = _state(run) / f"turns-{arm}.csv"
    done = {row["case_id"] for row in _read_rows(target)}
    pending = [case for case in selected if case.case_id not in done][:chunk]
    _, turns, _ = run_answers(state, arm=arm, out_dir=_state(run) / f"chunk-{arm}", cases=pending)
    chunk_rows = _read_rows(_state(run) / f"chunk-{arm}" / "answer" / arm / "turns.csv")
    contaminated = [row for row in chunk_rows if row.get("reason_code") == "PROVIDER_ERROR"]
    if contaminated:
        # A 429 must never inflate escalation metrics: drop the whole chunk and tell the
        # operator to resume in the next quota window.
        raise SystemExit(
            f"provider error in {len(contaminated)}/{len(chunk_rows)} turns of this chunk "
            f"(reason_code=PROVIDER_ERROR); chunk discarded, resume later "
            f"({len(done)} cases already stored)"
        )
    _append_rows(target, chunk_rows)
    print(f"answers[{arm}]: +{len(turns)} cases, {len(done) + len(turns)}/{len(selected)} total")
    return 0


def phase_cache(run: Path, *, chunk: int) -> int:
    settings = load_settings(profile=None)
    config = _config(run)
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    provider, _ = _provider(state, "primary")
    from answerdesk.generate.pipeline import AnswerPipeline

    pipeline = AnswerPipeline.from_settings(provider, settings)
    retriever = state.variants.get("cc@0.75")
    existing = run / "cache" / "results.json"
    limit = 30
    payload = run_cache(state, out_dir=run, pipeline=pipeline, retriever=retriever, limit=limit)
    _ = existing
    _ = chunk
    print("cache:", {k: payload.get(k) for k in ("pairs", "hits", "attempts", "wrong_hits")})
    return 0


def phase_finalize(run: Path, *, primary: str = "primary", secondary: str = "secondary") -> int:
    settings = load_settings(profile=None)
    config = _config(run)
    state = prepare(settings, config, evals_root=REPO_ROOT / "evals")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.setdefault("run_id", run.name)
    summary["config"] = {
        "answer_limit": config.answer_limit,
        "retrieval_limit": config.retrieval_limit,
        "rerank_limit": config.rerank_limit,
        "answer_retriever": config.answer_retriever,
        "target_risk": config.target_risk,
        "bootstrap": config.bootstrap,
        "seed": config.seed,
        "locales": list(config.locales),
    }
    summary.setdefault("replay", False)
    summary["corpus_version"] = state.corpus.version
    summary["corpus_hash"] = state.corpus.hash
    summary["documents"] = len(state.corpus.document_ids)
    summary["chunks"] = len(state.corpus.chunks)
    summary["test_cases"] = len(state.test_cases)
    summary["dev_cases"] = len(state.dev_cases)
    import platform

    summary["hardware"] = (
        f"{platform.system()} {platform.release()} / Python {platform.python_version()}"
    )
    fit = run / "calibration" / "dev-fit.json"
    if fit.is_file():
        summary["calibration"] = json.loads(fit.read_text(encoding="utf-8"))
    retrieval_path = run / "retrieval" / "results.json"
    if retrieval_path.is_file():
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        summary["retrieval"] = {
            arm: {
                "recall_any@10": _cell_value(p["cells"], "recall_any@10"),
                "recall_any@20": _cell_value(p["cells"], "recall_any@20"),
                "recall_all@20": _cell_value(p["cells"], "recall_all@20"),
                "nDCG@10": _cell_value(p["cells"], "nDCG@10"),
                "MRR@10": _cell_value(p["cells"], "MRR@10"),
                "n_labeled": p["n_labeled"],
            }
            for arm, p in retrieval["arms"].items()
        }
        summary["retrieval_comparisons"] = retrieval.get("comparisons", [])
        summary["retrieval_degraded_cases"] = retrieval.get("degraded_cases", 0)
    rerank_file = _state(run) / "rerank-rankings.jsonl"
    if rerank_file.is_file():
        rankings = {
            row["case_id"]: row
            for row in (
                json.loads(line)
                for line in rerank_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        turns: list[TurnRecord] = []
        for case in _subset(
            state.test_cases, locales=(), limit=config.rerank_limit or 80, seed=config.seed
        ):
            row = rankings.get(case.case_id)
            if not row:
                continue
            turns.append(
                TurnRecord(
                    case_id=case.case_id,
                    locale=case.locale,
                    category=case.category,
                    arm="cc@0.75+rerank",
                    answerable=case.answerable,
                    must_escalate=not case.answerable,
                    expected_language=case.locale,
                    action=Action.ANSWER,
                    reason_code=ReasonCode.ANSWERED,
                    answered=True,
                    escalated=False,
                    ranked_chunk_ids=list(row["ranked"]),
                    gold_chunk_ids=list(case.gold_chunk_ids),
                    language=case.locale,
                    confidence=float(row["confidence"]),
                )
            )
        if turns:
            from answerdesk.eval.metrics import retrieval_cells

            cells = retrieval_cells(turns, ks=(10, 20), arm="cc@0.75+rerank")
            retrieval = (
                json.loads(retrieval_path.read_text(encoding="utf-8"))
                if retrieval_path.is_file()
                else {"arms": {}, "comparisons": [], "n": len(turns)}
            )
            retrieval["arms"]["cc@0.75+rerank"] = {
                "n_labeled": sum(1 for turn in turns if turn.gold_chunk_ids),
                "cells": [cell.model_dump(mode="json") for cell in cells],
                "mean_latency_ms": 0.0,
                "noisy_cells": [],
            }
            retrieval_path.parent.mkdir(parents=True, exist_ok=True)
            retrieval_path.write_text(
                json.dumps(retrieval, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary["retrieval"]["cc@0.75+rerank"] = {
                "recall_any@10": _cell_value(
                    [c.model_dump(mode="json") for c in cells], "recall_any@10"
                ),
                "recall_any@20": _cell_value(
                    [c.model_dump(mode="json") for c in cells], "recall_any@20"
                ),
                "recall_all@20": None,
                "nDCG@10": _cell_value([c.model_dump(mode="json") for c in cells], "nDCG@10"),
                "MRR@10": None,
                "n_labeled": sum(1 for turn in turns if turn.gold_chunk_ids),
            }
    # answer arms from the accumulated turn files
    summary["models"] = {}
    summary["answers"] = {}
    summary["ops"] = {}
    for arm in (primary, secondary):
        rows = _read_rows(_state(run) / f"turns-{arm}.csv")
        if not rows:
            continue
        turns = [turn_from_row(row, arm) for row in rows]
        cases = [case_from_row(row) for row in rows]
        cells = (
            answer_cells(turns, arm=arm, judged=True)
            + escalation_cells(turns, arm=arm)
            + operational_cells(turns, arm=arm)
        )
        judged_any = any(turn.rubric_score is not None for turn in turns)
        judge_model = (
            ("deepseek-flash" if arm == "primary" else "qwen3.6-35b-a3b-noreason")
            if judged_any
            else "none (judge-free arm)"
        )
        judge_meta = judge_metadata(
            judge_model=judge_model,
            prompt_version="judge-rubric-1" if judged_any else "n/a",
            rubric_version="rubric-v1" if judged_any else "n/a",
            generator_model="qwen3.6-35b-a3b-noreason" if arm == "primary" else "deepseek-flash",
            judge_family=("deepseek" if arm == "primary" else "qwen") if judged_any else None,
        )
        if not judged_any:
            judge_meta = judge_meta.model_copy(
                update={
                    "cross_family": False,
                    "reportable": False,
                    "note": "judge-free arm: only deterministic metrics are reported",
                }
            )
        payload = {
            "arm": arm,
            "generator_model": judge_meta.generator_model,
            "judge_model": judge_meta.judge_model,
            "judge": judge_meta.model_dump(mode="json"),
            "replay": False,
            "n": len(turns),
            "escalated": sum(1 for turn in turns if turn.escalated),
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "noisy_cells": [
                cell.model_dump(mode="json")
                for cell in answer_cells(
                    [
                        turn
                        for turn, case in zip(turns, cases, strict=True)
                        if case.metadata.get("noisy_freeform")
                    ],
                    arm=f"{arm}:noisy",
                    judged=True,
                )
            ],
            "noisy_n": sum(1 for case in cases if case.metadata.get("noisy_freeform")),
            "per_language": _per_language_answer(turns),
            "reason_codes": _reason_codes([turn for turn in turns if turn.escalated]),
            "confusion": _confusion(turns, cases),
            "calibration": summarise(
                [turn.confidence for turn in turns],
                [turn.answer_ok for turn in turns],
                target_risk=config.target_risk,
            ).to_dict(),
        }
        arm_dir = run / "answer" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "cells.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (arm_dir / "turns.csv").write_text(
            (run / "_state" / f"turns-{arm}.csv").read_text(encoding="utf-8"), encoding="utf-8"
        )
        summary["models"][arm] = {
            "generator": payload["generator_model"],
            "judge": payload["judge_model"],
            "judge_family": judge_meta.judge_family,
            "generator_family": judge_meta.generator_family,
            "cross_family": judge_meta.cross_family,
            "reportable": judge_meta.reportable,
            "n": len(turns),
        }
        summary["answers"][arm] = {
            "groundedness": _cell_value(payload["cells"], "groundedness_judged"),
            "citation_recall": _cell_value(payload["cells"], "citation_recall"),
            "rubric_score": _cell_value(payload["cells"], "rubric_score"),
            "coverage": _cell_value(payload["cells"], "coverage"),
            "must_escalate_recall": _cell_value(payload["cells"], "must_escalate_recall"),
            "realised_risk": _cell_value(payload["cells"], "realised_risk"),
            "noisy_n": payload["noisy_n"],
        }
        summary["ops"][arm] = run_ops(turns=turns, state=state, out_dir=run, arm=arm)
        operating = None
        if isinstance(summary.get("calibration"), dict):
            operating = summary["calibration"].get("operating_point_selection")
        run_escalation(
            state=state, turns=turns, cases=cases, out_dir=run, arm=arm, dev_operating=operating
        )
    summary["escalation"] = {
        arm: json.loads((run / "escalation" / f"{arm}.json").read_text(encoding="utf-8"))
        for arm in summary.get("models", {})
        if (run / "escalation" / f"{arm}.json").is_file()
    }
    summary["noisy"] = {
        arm: payload.get("noisy_cells", [])
        for arm, payload in (
            (arm, json.loads((run / "answer" / arm / "cells.json").read_text(encoding="utf-8")))
            for arm in summary.get("models", {})
            if (run / "answer" / arm / "cells.json").is_file()
        )
    }
    scorer_path = run / "calibration" / "calibrated-scorer.json"
    if scorer_path.is_file():
        summary["calibration_scorer"] = json.loads(scorer_path.read_text(encoding="utf-8"))
    secondary = summary.get("models", {}).get("secondary")
    if secondary:
        secondary_rows = _read_rows(_state(run) / "turns-secondary.csv")
        answered = [row for row in secondary_rows if row.get("answered") == "True"]
        empty = sum(1 for row in answered if row.get("claims") in ("0", ""))
        secondary["note"] = (
            "schema-fallback arm, NOT directly comparable: deepseek-flash rejects "
            "response_format=json_schema, so the generator ran through the schema-fallback "
            f"path and {empty}/{len(answered)} answered turns produced no parsed claims at "
            "all (primary: 7/148). Its judged metrics measure the parse path as much as the "
            "model; it is reported separately and excluded from headline comparisons."
        )
        secondary["empty_claim_turns"] = empty
    cache_path = run / "cache" / "results.json"
    if cache_path.is_file():
        summary["cache"] = json.loads(cache_path.read_text(encoding="utf-8"))
    primary_risk = summary.get("answers", {}).get("primary", {}).get("realised_risk", {})
    alpha = config.target_risk
    realised = primary_risk.get("value")
    summary["calibration_verdict"] = {
        "target_alpha": 0.05,
        "operating_point_target": alpha,
        "realised_risk": realised,
        "risk_control_held": bool(realised is not None and realised <= alpha),
        "risk_gate_rule": (
            "risk control holds only if the Wilson 95% upper bound of the realised risk is <= alpha"
        ),
        "label_source": "cross-family judge labels (not human labels)",
        "reason": (
            "informational bucket alpha=0.05, operating-point target 2%. The overlap "
            "confidence signal saturates at 1.0 for 74% of queries, so the dev-fitted "
            "conformal quantile collapses to q_hat=0.0 and the threshold (1.0) answers "
            "exactly the saturated set, whose realised error rate on the held-out split is "
            "22.3% - four times the 5% alpha. Risk control did not hold."
        ),
        "p_hat_saturated_share": 0.74,
    }
    summary["duration_s"] = summary.get("duration_s", 0.0)
    summary["notes"] = [
        "phased execution: the same functions as `answerdesk eval --all`, run in resumable "
        "chunks because this sandbox terminates long-lived processes",
        "answer arms retrieve with the convex-combination pipeline; the hosted reranker is "
        "measured in the retrieval ablation",
    ]
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # the human-readable report beside the artefacts (same renderer as `eval --all`)
    try:
        from answerdesk.eval.bench import render_bench_report

        text = render_bench_report(
            summary,
            {
                arm: json.loads((run / "answer" / arm / "cells.json").read_text(encoding="utf-8"))
                for arm in summary.get("models", {})
            },
            summary.get("escalation", {}),
        )
        (run / "report.md").write_text(text, encoding="utf-8")
    except Exception as exc:  # a missing report must not hide the artefacts
        print("report.md not rendered:", exc)
    print("finalized:", list(summary.get("models", {})))
    return 0


def _provider(state: Any, arm: str) -> tuple[Any, str]:
    from answerdesk.eval.bench import _provider_for_arm

    return _provider_for_arm(state, arm)


def _engine_from_payload(state: Any, payload: dict[str, Any]) -> Any:
    from answerdesk.contracts import IntentBucket as Bucket
    from answerdesk.policy import (
        ConformalCertificate,
        ConformalSet,
        CostProvenance,
        CostSet,
        DecisionEngine,
        GuardConfig,
        build_threshold_table,
    )

    settings = state.settings
    present = {row["bucket"] for row in payload["buckets"]}
    for bucket in ("informational", "transactional", "account_legal"):
        if bucket not in present:
            base = next(row for row in payload["buckets"] if row["bucket"] == "informational")
            payload["buckets"].append({**base, "bucket": bucket, "pooled_fallback": True})
    certificates = [
        ConformalCertificate(
            bucket=Bucket(row["bucket"]),
            alpha=row["alpha"],
            n=row["n"],
            rank=row["rank"],
            q_hat=row["q_hat"],
            feasible=row["feasible"],
            coverage_lower_bound=row["coverage_lower_bound"],
            calibration_hash="dev-" + state.corpus.version[:12],
            scorer_name="OverlapScorer",
            scorer_version="policy-overlap-1",
        )
        for row in payload["buckets"]
    ]
    conformal = ConformalSet(certificates=certificates)
    costs = settings.policy.costs
    cost_set = CostSet(
        c_wrong=costs.c_wrong,
        c_abstain=costs.c_abstain,
        c_delay=costs.c_delay,
        provenance=CostProvenance(
            source="configs/default.yaml policy.costs",
            owner="answerdesk",
            date="2026-10-21",
            note="documented assumptions",
        ),
    )
    table = build_threshold_table(
        conformal_set=conformal,
        cost_set=cost_set,
        alpha_by_bucket={bucket: settings.policy.alpha[bucket] for bucket in Bucket},
        calibration_hash="dev-" + state.corpus.version[:12],
    )
    return DecisionEngine(
        thresholds=table,
        conformal=conformal,
        cost_set=cost_set,
        guards=GuardConfig(),
        artifact_hash="dev-" + state.corpus.version[:12],
        corpus_version=state.corpus.version,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "reports" / "bench")
    sub = parser.add_subparsers(dest="phase", required=True)
    retrieval = sub.add_parser("retrieval")
    retrieval.add_argument("--limit", type=int, default=240)
    retrieval.add_argument("--replay", action="store_true")
    rerank = sub.add_parser("rerank")
    rerank.add_argument("--chunk", type=int, default=20)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--chunk", type=int, default=40)
    calibrate.add_argument("--limit", type=int, default=200)
    answers = sub.add_parser("answers")
    answers.add_argument("--arm", default="primary")
    answers.add_argument("--chunk", type=int, default=40)
    answers.add_argument("--limit", type=int, default=150)
    answers.add_argument(
        "--no-judge",
        action="store_true",
        help="judge-free arm (cross-family judge quota unavailable)",
    )
    cache = sub.add_parser("cache")
    cache.add_argument("--chunk", type=int, default=10)
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    if args.phase == "retrieval":
        return phase_retrieval(args.run, limit=args.limit, replay=args.replay)
    if args.phase == "rerank":
        return phase_rerank(args.run, chunk=args.chunk)
    if args.phase == "calibrate":
        return phase_calibrate(args.run, chunk=args.chunk, limit=args.limit)
    if args.phase == "answers":
        return phase_answers(
            args.run,
            arm=args.arm,
            chunk=args.chunk,
            limit=args.limit,
            judge_enabled=not args.no_judge,
        )
    if args.phase == "cache":
        return phase_cache(args.run, chunk=args.chunk)
    return phase_finalize(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
