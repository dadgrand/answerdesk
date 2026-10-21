"""Dev-only self-consistency measurement (k=3 sampled drafts).

Self-consistency needs extra sampling, so it is measured on the **dev split only** and reported
as an ablation: if it improves discrimination there, the production path could buy it with
k extra generations; it is never used in the test-time model, which must run on one generation.

Drafts are generated at a non-zero temperature with the same provider as the answers, and
agreement is the mean pairwise token-set Jaccard between drafts of one question. The run is
bounded (``--limit`` questions x k calls) and recorded with its model id and cost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from answerdesk.eval.risk.features import SELF_CONSISTENCY, read_rows

DRAFTS = 3
TEMPERATURE = 0.7


def _jaccard(left: str, right: str) -> float:
    left_terms = {term.lower() for term in left.split() if len(term) > 2}
    right_terms = {term.lower() for term in right.split() if len(term) > 2}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _provider(settings: Any) -> tuple[Any, str]:
    from answerdesk.llm.factory import build_provider

    clone = settings.model_copy(deep=True)
    clone.llm.temperature = TEMPERATURE
    clone.llm.max_tokens = 256
    clone.llm.json_mode = False
    return build_provider(clone), clone.llm.model


def build_self_consistency(
    *, out_dir: Path, limit: int = 25, drafts: int = DRAFTS, settings: Any | None = None
) -> dict[str, Any]:
    from answerdesk.config import load_settings
    from answerdesk.contracts import Message, Role

    settings = settings or load_settings(profile=None)
    rows = [row for row in read_rows(out_dir / "features.csv") if row.split == "dev"]
    questions: dict[str, str] = {}
    for row in rows:
        questions.setdefault(row.fact_id, row.sample_id)
    selected = sorted(questions)[:limit]
    provider, model = _provider(settings)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fact_id in selected:
        sample_id = questions[fact_id]
        prompt = (
            "Ответь на вопрос поддержки кратко, одним-двумя предложениями, только по фактам. "
            f"Вопрос: {sample_id.split('::')[0]}"
        )
        outputs: list[str] = []
        for _ in range(drafts):
            completion = provider.complete(
                [
                    Message(role=Role.SYSTEM, content="Ты помощник поддержки. Отвечай кратко."),
                    Message(role=Role.USER, content=prompt),
                ],
                response_schema=None,
            )
            outputs.append(getattr(completion, "text", "") or "")
        pairwise = [
            _jaccard(outputs[i], outputs[j])
            for i in range(len(outputs))
            for j in range(i + 1, len(outputs))
        ]
        agreement = sum(pairwise) / len(pairwise) if pairwise else 0.0
        results.append({"fact_id": fact_id, "samples": outputs, "agreement": round(agreement, 6)})
    payload = {
        "kind": SELF_CONSISTENCY,
        "split": "dev",
        "drafts": drafts,
        "temperature": TEMPERATURE,
        "model": model,
        "questions": len(results),
        "calls": len(results) * drafts,
        "duration_s": round(time.perf_counter() - started, 3),
        "mean_agreement": round(sum(row["agreement"] for row in results) / len(results), 6)
        if results
        else 0.0,
        "rows": results,
        "note": (
            "dev-only feature: the test-time model uses a single generation, so this measures "
            "how much discrimination k=3 sampling would buy, not a deployable feature"
        ),
    }
    (out_dir / "self-consistency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def attach_to_summary(out_dir: Path) -> dict[str, Any] | None:
    """Compute the dev AUROC with and without the feature, if the measurement exists."""
    path = out_dir / "self-consistency.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in read_rows(out_dir / "features.csv") if row.split == "dev"]
    by_fact = {row["fact_id"]: row["agreement"] for row in payload.get("rows", [])}
    usable = [row for row in rows if row.fact_id in by_fact]
    if len(usable) < 10 or len({row.label for row in usable}) < 2:
        payload["ablation"] = {"status": "not enough dev rows with drafts to compare"}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return dict(payload)
    from answerdesk.eval.risk import metrics
    from answerdesk.eval.risk.features import DECISION_FEATURES
    from answerdesk.eval.risk.model import fit, fit_platt

    base_features = tuple(DECISION_FEATURES)
    labels = [row.label for row in usable]
    base = fit([row.features for row in usable], labels, features=base_features, l2=2.0, steps=2000)
    augmented_rows = [{**row.features, SELF_CONSISTENCY: by_fact[row.fact_id]} for row in usable]
    augmented = fit(
        augmented_rows, labels, features=(*base_features, SELF_CONSISTENCY), l2=2.0, steps=2000
    )
    base_scores = fit_platt(base.predict([row.features for row in usable]), labels).apply(
        base.predict([row.features for row in usable])
    )
    aug_scores = fit_platt(augmented.predict(augmented_rows), labels).apply(
        augmented.predict(augmented_rows)
    )
    payload["ablation"] = {
        "n_dev_rows": len(usable),
        "dev_auroc_without": round(metrics.auroc(base_scores, labels), 6),
        "dev_auroc_with": round(metrics.auroc(aug_scores, labels), 6),
        "status": "measured on dev only",
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dict(payload)


__all__ = ["DRAFTS", "attach_to_summary", "build_self_consistency"]
