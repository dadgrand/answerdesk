"""Judge-calibration protocol: a blind human-label template plus κ scoring.

The reference stack judges answers with an LLM. That is only admissible evidence when the
judge itself is calibrated against humans. This module builds the artefact that makes that
possible — a per-language template of 150 items — and scores it when the labels come back.
Until then every judged metric in the external arm is labelled **diagnostic**; the tooling
must not be able to publish a self-graded claim silently.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerdesk.eval.stats import cohen_kappa

ITEMS_PER_LANGUAGE = 150
TEMPLATE_COLUMNS = (
    "item_id",
    "language",
    "item_ref",
    "question",
    "answer_text",
    "judge_label",
    "human_label_a",
    "human_label_b",
    "adjudicated_label",
    "notes",
)
LABEL_VALUES = ("supported", "unsupported", "unclear")


@dataclass
class CalibrationTemplate:
    language: str
    path: Path
    items: int

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "path": str(self.path), "items": self.items}


def _sample(items: Sequence[dict[str, Any]], language: str, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda row: str(row.get("item_id", "")))
    return ranked[:limit]


def build_template(
    *,
    language: str,
    items: Sequence[dict[str, Any]],
    out_dir: Path,
    limit: int = ITEMS_PER_LANGUAGE,
) -> CalibrationTemplate:
    """Write a blank labelling sheet; the judge column is filled only if a judge ran."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration-{language}.csv"
    selected = _sample(items, language, limit)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "item_id": row.get("item_id", ""),
                    "language": language,
                    # the question text stays in the licensed benchmark (attributed, committed);
                    # reports must not carry CC BY-SA text, so the sheet references the item
                    "item_ref": (f"data/bench/answerdesk-bench/v0.1/belebele/qa/{language}.jsonl"),
                    "question": "",
                    "answer_text": (row.get("answer_text") or "")[:800],
                    "judge_label": row.get("judge_label", ""),
                    "human_label_a": "",
                    "human_label_b": "",
                    "adjudicated_label": "",
                    "notes": "",
                }
            )
    return CalibrationTemplate(language=language, path=path, items=len(selected))


def write_protocol(out_dir: Path, *, languages: Sequence[str], limit: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "PROTOCOL.md"
    path.write_text(
        "# Judge-calibration protocol (external arm)\n\n"
        f"Template size: {limit} items per language, sampled deterministically by item id\n"
        "from the\n"
        f"test split of the suite, for: {', '.join(languages)}.\n\n"
        "0. Read each item's question and evidence from the licensed benchmark file named in\n"
        "   `item_ref` (the sheet carries no CC BY-SA text).\n"
        "1. Two annotators label every item independently as `supported`, `unsupported` or\n"
        "   `unclear`, seeing the question, the retrieved evidence and the answer — **not** the\n"
        "   judge's label, which is hidden until both sheets are returned.\n"
        "2. Disagreements are adjudicated to `adjudicated_label`; the adjudicated sheet is the\n"
        "   reference. A second-pass agreement rate below 0.7 triggers a rubric clarification\n"
        "   round before any κ is reported.\n"
        "3. `answerdesk eval external --score-calibration <dir>` computes Cohen's κ between\n"
        "   the judge (binary: supported vs the rest) and the adjudicated labels, per language\n"
        "   and overall, and writes `calibration-kappa.json`.\n"
        "4. **Until that file exists with κ ≥ 0.6 in every language, every judged metric in the\n"
        "   external report is labelled diagnostic and must not be quoted as a quality claim.**\n"
        "   The external arm's primary numbers (multiple-choice accuracy, retrieval nDCG/recall)\n"
        "   are judge-free by construction, so they are unaffected by this gate.\n",
        encoding="utf-8",
    )
    return path


def _label_binary(value: str) -> bool | None:
    normalised = (value or "").strip().lower()
    if normalised in {"supported", "yes", "true", "1", "pass"}:
        return True
    if normalised in {"unsupported", "no", "false", "0", "fail", "unclear"}:
        return False
    return None


def score_sheets(paths: Iterable[Path]) -> dict[str, Any]:
    """Cohen's κ per language and overall, from filled calibration sheets."""
    per_language: dict[str, dict[str, Any]] = {}
    pooled_judge: list[bool] = []
    pooled_human: list[bool] = []
    missing = 0
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        language = rows[0]["language"] if rows else path.stem
        judge: list[bool] = []
        human: list[bool] = []
        for row in rows:
            judge_label = _label_binary(row.get("judge_label", ""))
            human_label = _label_binary(row.get("adjudicated_label", "")) or _label_binary(
                row.get("human_label_a", "")
            )
            if judge_label is None or human_label is None:
                missing += 1
                continue
            judge.append(judge_label)
            human.append(human_label)
        if len(judge) >= 2:
            kappa = cohen_kappa(judge, human)
            per_language[language] = {
                "items": len(judge),
                "kappa": round(float(kappa), 6),
                "agreement": round(
                    sum(1 for a, b in zip(judge, human, strict=True) if a == b) / len(judge), 6
                ),
            }
            pooled_judge.extend(judge)
            pooled_human.extend(human)
        else:
            per_language[language] = {"items": len(judge), "kappa": None, "agreement": None}
    overall = None
    if len(pooled_judge) >= 2:
        overall = round(float(cohen_kappa(pooled_judge, pooled_human)), 6)
    return {
        "per_language": per_language,
        "overall_kappa": overall,
        "items_scored": len(pooled_judge),
        "items_missing_labels": missing,
        "status": "calibrated" if overall is not None else "pending-human-labels",
    }


def write_kappa(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / "calibration-kappa.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ITEMS_PER_LANGUAGE",
    "TEMPLATE_COLUMNS",
    "CalibrationTemplate",
    "build_template",
    "score_sheets",
    "write_kappa",
    "write_protocol",
]
