"""Belebele-derived multiple-choice QA: closed-book against RAG-conditioned answering.

Both conditions ask the *same* four-option question in the item's language and are scored
**judge-free**: the model must answer with an option letter and it is counted correct only
when the letter matches ``answer_index``. That keeps the external arm reproducible without an
LLM judge — and therefore without a self-graded claim.

RAG conditioning feeds the lexical channel's top-k passages from the language's Belebele pool.
It tests whether retrieval *helps comprehension*; it says nothing about retrieval quality on
this data (the labels are derived, see the dataset card).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from answerdesk.eval.external.bench import Passage, QaItem
from answerdesk.eval.external.retrieval import LexicalRanker
from answerdesk.eval.stats import paired_bootstrap, wilson_interval

LETTERS = ("A", "B", "C", "D", "E", "F")

_SYSTEM = (
    "You are answering a multiple-choice reading-comprehension question. Reply with exactly "
    "one letter and nothing else."
)

_LETTER = re.compile(r"\b([A-F])\b")


def build_prompt(item: QaItem, condition: str, passages: Sequence[Passage]) -> list[dict[str, str]]:
    options = "\n".join(f"{LETTERS[index]}) {text}" for index, text in enumerate(item.options))
    parts = [f"Вопрос / Question:\n{item.question}", f"Варианты / Options:\n{options}"]
    if condition == "rag":
        context = "\n\n".join(passage.text for passage in passages)
        parts.insert(0, f"Контекст / Context:\n{context}")
        parts.append(
            "Ответь одной буквой варианта, опираясь только на контекст. / "
            "Answer with one option letter, using only the context."
        )
    else:
        parts.append("Ответь одной буквой варианта. / Answer with one option letter.")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_letter(text: str) -> str | None:
    match = _LETTER.search(text.strip().upper())
    return match.group(1) if match else None


@dataclass
class QaRow:
    item_id: str
    language: str
    condition: str
    predicted: str | None
    gold: str
    correct: bool
    latency_ms: float
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "item_id": self.item_id,
            "language": self.language,
            "condition": self.condition,
            "predicted": self.predicted,
            "gold": self.gold,
            "correct": self.correct,
            "latency_ms": round(self.latency_ms, 3),
        }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_condition(
    *,
    provider: Any,
    items: Sequence[QaItem],
    condition: str,
    ranker: LexicalRanker | None,
    out_path: Path,
    top_k: int = 3,
    max_tokens: int = 8,
    model: str = "",
    progress: Any | None = None,
) -> list[QaRow]:
    """Answer every item in one condition, checkpointing after each call (resumable)."""
    done = {row["item_id"] for row in _read_rows(out_path) if row.get("condition") == condition}
    pending = [item for item in items if item.item_id not in done]
    produced: list[QaRow] = []
    for position, item in enumerate(pending, start=1):
        passages: list[Passage] = []
        if condition == "rag" and ranker is not None:
            ranked = ranker.rank(item.question, k=top_k)
            by_id = {passage.passage_id: passage for passage in ranker.passages}
            passages = [by_id[passage_id] for passage_id in ranked.ranked if passage_id in by_id]
        messages = build_prompt(item, condition, passages)
        started = time.perf_counter()
        completion = provider.complete([_message(role) for role in messages], response_schema=None)
        latency_ms = (time.perf_counter() - started) * 1000.0
        letter = parse_letter(getattr(completion, "text", "") or "")
        gold = LETTERS[item.answer_index]
        row = QaRow(
            item_id=item.item_id,
            language=item.language,
            condition=condition,
            predicted=letter,
            gold=gold,
            correct=bool(letter == gold),
            latency_ms=latency_ms,
            model=model,
        )
        produced.append(row)
        _append_rows(out_path, [row.to_dict()])
        if progress is not None:
            progress(position, len(pending), row)
    return produced


def _message(role: dict[str, str]) -> Any:
    from answerdesk.contracts import Message, Role

    return Message(role=Role(role["role"]), content=role["content"])


@dataclass
class ConditionAccuracy:
    condition: str
    n: int
    correct: int
    accuracy: float
    ci: tuple[float, float]


def summarise(rows: Sequence[dict[str, Any]], condition: str) -> ConditionAccuracy:
    subset = [row for row in rows if row["condition"] == condition]
    correct = sum(1 for row in subset if row["correct"])
    interval = wilson_interval(correct, len(subset)) if subset else None
    return ConditionAccuracy(
        condition=condition,
        n=len(subset),
        correct=correct,
        accuracy=(correct / len(subset)) if subset else 0.0,
        ci=(interval.low, interval.high) if interval else (0.0, 1.0),
    )


def delta(rows: Sequence[dict[str, Any]], *, samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Paired bootstrap on per-item correctness (RAG minus closed-book)."""
    by_item: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], {})[row["condition"]] = bool(row["correct"])
    pairs = [
        (values["closed"], values["rag"])
        for values in by_item.values()
        if "closed" in values and "rag" in values
    ]
    if not pairs:
        return {"n": 0, "delta": None, "ci_low": None, "ci_high": None, "ci_crosses_zero": True}
    base = [1.0 if pair[0] else 0.0 for pair in pairs]
    cond = [1.0 if pair[1] else 0.0 for pair in pairs]
    result = paired_bootstrap(cond, base, samples=samples, seed=seed)
    return {
        "n": int(result.n),
        "closed_book_accuracy": round(float(result.mean_b), 6),
        "rag_accuracy": round(float(result.mean_a), 6),
        "delta": round(float(result.diff), 6),
        "ci_low": round(float(result.low), 6),
        "ci_high": round(float(result.high), 6),
        "ci_crosses_zero": bool(result.low <= 0.0 <= result.high),
        "samples": samples,
        "seed": seed,
    }


__all__ = [
    "ConditionAccuracy",
    "QaRow",
    "build_prompt",
    "delta",
    "parse_letter",
    "run_condition",
    "summarise",
]
