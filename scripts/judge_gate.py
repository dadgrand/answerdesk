"""Judge gate: offline human-κ scoring and the mechanical calibration gate.

The repository judges answers with an LLM. A judged metric may only be quoted as evidence
once the judge has been shown to agree with humans on that corpus. This module implements the
*mechanical* half of that rule:

* **Ingest** the per-language labelling sheets produced by
  ``answerdesk eval external --run`` (``calibration-<lang>.csv``, 150 items per language).
* **Validate** the sheet schema strictly — a malformed sheet must block the gate, never
  silently reduce the sample.
* **Score** three quantities: pairwise agreement between the two annotators, Cohen's κ
  between the judge and the adjudicated reference, and the pooled versions of both, each with
  a deterministic bootstrap confidence interval.
* **Decide** one of ``pending`` / ``blocked`` / ``open`` from the policy file. When the
  decision is ``open`` (and the input is not a dry-run fixture) the judged metrics may be
  labelled *calibrated*; otherwise they stay *diagnostic*. The flip needs no code change: it
  happens when real labels clear the thresholds in ``evals/judge/gate-policy.json``.

Nothing here fabricates labels. Blank sheets produce ``pending``; the dry-run fixtures under
``tests/unit/eval/fixtures/judge-gate/`` carry :data:`SYNTHETIC_MARKER` in every row, are
marked ``synthetic: true`` in the artefact and can never be ``publishable``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "evals" / "judge" / "gate-policy.json"
DEFAULT_SHEETS_DIR = REPO_ROOT / "reports" / "external-p0" / "calibration"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "judge-gate"
FIXTURES_DIR = REPO_ROOT / "tests" / "unit" / "eval" / "fixtures" / "judge-gate"

ARTEFACT_SCHEMA_VERSION = "1"
SYNTHETIC_MARKER = "SYNTHETIC-DRY-RUN"
GATE_FILE = "gate.json"

_TRUE_LABELS = frozenset({"supported", "yes", "true", "1", "pass"})
_FALSE_LABELS = frozenset({"unsupported", "no", "false", "0", "fail", "unclear"})
_BLANK = frozenset({"", "none", "null", "nan", "-"})


class GateError(RuntimeError):
    """A configuration or sheet set the gate cannot score."""

    def __init__(self, message: str, *, errors: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.errors = list(errors)


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class Policy:
    """Gate parameters. Lives in ``evals/judge/gate-policy.json`` so thresholds are data."""

    languages: tuple[str, ...]
    items_per_language: int
    labels: tuple[str, ...]
    required_columns: tuple[str, ...]
    judge_kappa_floor: float
    pairwise_agreement_floor: float
    bootstrap_samples: int
    bootstrap_seed: int
    confidence: float
    synthetic_marker: str = SYNTHETIC_MARKER

    @classmethod
    def load(cls, path: Path | None = None) -> Policy:
        policy_path = Path(path) if path is not None else POLICY_PATH
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise GateError(f"gate policy not found: {policy_path}") from error
        except json.JSONDecodeError as error:
            raise GateError(f"gate policy is not valid JSON: {policy_path}: {error}") from error
        try:
            return cls(
                languages=tuple(str(value) for value in payload["languages"]),
                items_per_language=int(payload["items_per_language"]),
                labels=tuple(str(value) for value in payload["labels"]),
                required_columns=tuple(str(value) for value in payload["required_columns"]),
                judge_kappa_floor=float(payload["thresholds"]["judge_kappa"]),
                pairwise_agreement_floor=float(payload["thresholds"]["pairwise_agreement"]),
                bootstrap_samples=int(payload["ci"]["samples"]),
                bootstrap_seed=int(payload["ci"]["seed"]),
                confidence=float(payload["ci"]["confidence"]),
                synthetic_marker=str(payload.get("synthetic_marker", SYNTHETIC_MARKER)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GateError(f"gate policy is incomplete: {policy_path}: {error!r}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "languages": list(self.languages),
            "items_per_language": self.items_per_language,
            "labels": list(self.labels),
            "required_columns": list(self.required_columns),
            "thresholds": {
                "judge_kappa": self.judge_kappa_floor,
                "pairwise_agreement": self.pairwise_agreement_floor,
            },
            "ci": {
                "method": "bootstrap-percentile",
                "samples": self.bootstrap_samples,
                "seed": self.bootstrap_seed,
                "confidence": self.confidence,
            },
            "synthetic_marker": self.synthetic_marker,
        }


# --------------------------------------------------------------------------- statistics


def cohen_kappa(first: Sequence[Any], second: Sequence[Any]) -> float | None:
    """Cohen's κ for any two equal-length label sequences.

    Returns ``None`` when κ is undefined (fewer than two items, or both raters used a single
    label so expected agreement is 1). Matches :func:`answerdesk.eval.stats.cohen_kappa` on
    binary inputs by construction.
    """
    if len(first) != len(second):
        raise ValueError("labellings must have the same length")
    n = len(first)
    if n < 2:
        return None
    observed = sum(1 for left, right in zip(first, second, strict=True) if left == right) / n
    left_counts: dict[Any, int] = {}
    right_counts: dict[Any, int] = {}
    for left, right in zip(first, second, strict=True):
        left_counts[left] = left_counts.get(left, 0) + 1
        right_counts[right] = right_counts.get(right, 0) + 1
    expected = sum(
        (left_counts.get(label, 0) / n) * (right_counts.get(label, 0) / n)
        for label in set(left_counts) | set(right_counts)
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def bootstrap_kappa(
    first: Sequence[Any],
    second: Sequence[Any],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for κ; deterministic for a given (seed, inputs)."""
    n = len(first)
    if n < 3 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        picks = rng.integers(0, n, size=n)
        left = [first[int(index)] for index in picks]
        right = [second[int(index)] for index in picks]
        kappa = cohen_kappa(left, right)
        if kappa is not None:
            values.append(kappa)
    if len(values) < max(20, samples // 10):
        return None
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(values, alpha))
    high = float(np.quantile(values, 1.0 - alpha))
    return (round(low, 6), round(high, 6))


def _binary(value: str) -> bool | None:
    normalised = (value or "").strip().lower()
    if normalised in _TRUE_LABELS:
        return True
    if normalised in _FALSE_LABELS:
        return False
    return None


def _category(value: str, labels: Sequence[str]) -> str | None:
    normalised = (value or "").strip().lower()
    if normalised in _BLANK:
        return None
    if normalised in {label.lower() for label in labels}:
        return normalised
    return None


# --------------------------------------------------------------------------- sheets


@dataclass
class Sheet:
    """One parsed, validated labelling sheet."""

    path: Path
    language: str
    rows: list[dict[str, str]]
    synthetic: bool
    sha256: str

    def relative_path(self) -> str:
        try:
            return self.path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:  # pragma: no cover - fixture outside the repo
            return self.path.as_posix()


def _is_synthetic(rows: Sequence[Mapping[str, str]], name: str, marker: str) -> bool:
    if marker and any(marker in str(row.get("notes", "")) for row in rows):
        return True
    lowered = name.lower()
    return "dry-run" in lowered or lowered.startswith("dryrun") or "synthetic" in lowered


def load_sheet(path: Path, policy: Policy) -> Sheet:
    """Read and validate one sheet. Structural problems raise :class:`GateError`."""
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise GateError(f"cannot read sheet {path}: {error}") from error
    missing = [column for column in policy.required_columns if column not in columns]
    if missing:
        errors.append(f"{path.name}: missing columns {missing}")
    if errors:
        raise GateError(f"invalid sheet {path}", errors=errors)

    language = ""
    for row in rows:
        value = str(row.get("language", "")).strip().lower()
        if value and not language:
            language = value
    if not language:
        language = path.stem.replace("calibration-", "").strip().lower()
    if language not in policy.languages:
        errors.append(f"{path.name}: language {language!r} not in policy {list(policy.languages)}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for index, row in enumerate(rows, start=2):
        item_id = str(row.get("item_id", "")).strip()
        if not item_id:
            errors.append(f"{path.name}:{index}: empty item_id")
        elif item_id in seen:
            duplicates.append(item_id)
        seen.add(item_id)
        for column in ("judge_label", "human_label_a", "human_label_b", "adjudicated_label"):
            value = str(row.get(column, "")).strip()
            if value and _category(value, policy.labels) is None:
                errors.append(
                    f"{path.name}:{index}: {column}={value!r} is not one of {list(policy.labels)}"
                )
    if duplicates:
        errors.append(f"{path.name}: duplicate item_id values {sorted(set(duplicates))[:5]}")
    if not rows:
        errors.append(f"{path.name}: no data rows")
    if errors:
        raise GateError(f"invalid sheet {path}", errors=errors)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Sheet(
        path=path,
        language=language,
        rows=rows,
        synthetic=_is_synthetic(rows, path.name, policy.synthetic_marker),
        sha256=digest,
    )


def discover_sheets(directory: Path, policy: Policy, *, pattern: str = "*.csv") -> list[Sheet]:
    """Load every sheet in ``directory``; raises :class:`GateError` with all problems."""
    if not directory.is_dir():
        raise GateError(f"sheet directory does not exist: {directory}")
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise GateError(f"no {pattern} sheets in {directory}")
    sheets: list[Sheet] = []
    errors: list[str] = []
    for path in paths:
        try:
            sheets.append(load_sheet(path, policy))
        except GateError as error:
            errors.extend(error.errors or [str(error)])
    if errors:
        raise GateError("sheet set failed validation", errors=errors)
    return sheets


# --------------------------------------------------------------------------- scoring


@dataclass
class LanguageScore:
    language: str
    items: int = 0
    labelled: int = 0
    judge_kappa: float | None = None
    judge_kappa_ci: tuple[float, float] | None = None
    pairwise_kappa: float | None = None
    pairwise_kappa_ci: tuple[float, float] | None = None
    pairwise_agreement: float | None = None
    disagreements: int = 0
    unresolved: int = 0
    status: str = "blank"
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "labelled": self.labelled,
            "judge_kappa": self.judge_kappa,
            "judge_kappa_ci": list(self.judge_kappa_ci) if self.judge_kappa_ci else None,
            "pairwise_kappa": self.pairwise_kappa,
            "pairwise_kappa_ci": list(self.pairwise_kappa_ci) if self.pairwise_kappa_ci else None,
            "pairwise_agreement": self.pairwise_agreement,
            "disagreements": self.disagreements,
            "unresolved": self.unresolved,
            "status": self.status,
            "problems": self.problems,
        }


def disagreement_rows(sheet: Sheet, policy: Policy) -> list[dict[str, str]]:
    """Annotator A ≠ B rows, annotated for adjudication."""
    rows: list[dict[str, str]] = []
    for row in sheet.rows:
        left = _category(str(row.get("human_label_a", "")), policy.labels)
        right = _category(str(row.get("human_label_b", "")), policy.labels)
        if left is None or right is None or left == right:
            continue
        adjudicated = _category(str(row.get("adjudicated_label", "")), policy.labels)
        rows.append(
            {
                "item_id": str(row.get("item_id", "")),
                "language": sheet.language,
                "judge_label": str(row.get("judge_label", "")),
                "human_label_a": str(row.get("human_label_a", "")),
                "human_label_b": str(row.get("human_label_b", "")),
                "adjudicated_label": str(row.get("adjudicated_label", "")),
                "notes": str(row.get("notes", "")),
                "reason": "adjudicated" if adjudicated else "unresolved",
            }
        )
    return rows


def _reference(row: Mapping[str, str], policy: Policy) -> str | None:
    adjudicated = _category(str(row.get("adjudicated_label", "")), policy.labels)
    if adjudicated is not None:
        return adjudicated
    left = _category(str(row.get("human_label_a", "")), policy.labels)
    right = _category(str(row.get("human_label_b", "")), policy.labels)
    if left is not None and left == right:
        return left
    return None


def score_language(sheet: Sheet, policy: Policy) -> LanguageScore:
    """Score one language sheet: pairwise and judge-vs-reference κ with bootstrap CIs."""
    score = LanguageScore(language=sheet.language, items=len(sheet.rows))
    judge_values: list[bool] = []
    reference_values: list[bool] = []
    a_values: list[str] = []
    b_values: list[str] = []
    labelled_rows = 0
    unlabelled_rows = 0

    for row in sheet.rows:
        judge = _category(str(row.get("judge_label", "")), policy.labels)
        left = _category(str(row.get("human_label_a", "")), policy.labels)
        right = _category(str(row.get("human_label_b", "")), policy.labels)
        if judge is None and left is None and right is None:
            unlabelled_rows += 1
            continue
        labelled_rows += 1
        if left is None or right is None:
            score.problems.append(f"{row.get('item_id', '?')}: annotator labels incomplete")
            continue
        a_values.append(left)
        b_values.append(right)
        reference = _reference(row, policy)
        judge_binary = _binary(judge or "")
        reference_binary = _binary(reference or "")
        if reference is None:
            score.unresolved += 1
        elif judge_binary is not None and reference_binary is not None:
            judge_values.append(judge_binary)
            reference_values.append(reference_binary)

    score.labelled = labelled_rows
    if labelled_rows == 0:
        score.status = "blank"
        return score
    if unlabelled_rows:
        score.problems.append(f"{unlabelled_rows} rows have no labels at all")
    score.status = "scored" if not score.problems and not score.unresolved else "incomplete"

    if len(a_values) >= 2:
        score.pairwise_kappa = _round(cohen_kappa(a_values, b_values))
        score.pairwise_kappa_ci = bootstrap_kappa(
            a_values,
            b_values,
            samples=policy.bootstrap_samples,
            seed=policy.bootstrap_seed,
            confidence=policy.confidence,
        )
        score.pairwise_agreement = round(
            sum(1 for left, right in zip(a_values, b_values, strict=True) if left == right)
            / len(a_values),
            6,
        )
    if len(judge_values) >= 2:
        score.judge_kappa = _round(cohen_kappa(judge_values, reference_values))
        score.judge_kappa_ci = bootstrap_kappa(
            judge_values,
            reference_values,
            samples=policy.bootstrap_samples,
            seed=policy.bootstrap_seed + 1,
            confidence=policy.confidence,
        )
    score.disagreements = len(disagreement_rows(sheet, policy))
    return score


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


# --------------------------------------------------------------------------- gate


def evaluate(
    directory: Path,
    policy: Policy,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Score a sheet set and decide the gate. Never raises for content problems — a bad set
    produces ``status=blocked`` with reasons, so the caller always gets an artefact."""
    artefact: dict[str, Any] = {
        "schema_version": ARTEFACT_SCHEMA_VERSION,
        "source": source or _relative(directory),
        "policy": policy.to_dict(),
        "inputs": [],
        "per_language": {},
        "pooled": {
            "items": 0,
            "labelled": 0,
            "judge_kappa": None,
            "judge_kappa_ci": None,
            "pairwise_kappa": None,
            "pairwise_kappa_ci": None,
            "pairwise_agreement": None,
        },
        "disagreements": 0,
        "synthetic": False,
        "reasons": [],
    }
    try:
        sheets = discover_sheets(directory, policy)
        artefact["inputs"] = [
            {"path": sheet.relative_path(), "sha256": sheet.sha256} for sheet in sheets
        ]
        artefact["synthetic"] = any(sheet.synthetic for sheet in sheets)
        scores = {sheet.language: score_language(sheet, policy) for sheet in sheets}
        disagreements = [row for sheet in sheets for row in disagreement_rows(sheet, policy)]
    except GateError as error:
        artefact["status"] = "blocked"
        artefact["judged_metrics_label"] = "diagnostic"
        artefact["publishable"] = False
        artefact["reasons"] = error.errors or [str(error)]
        artefact["disagreement_rows"] = []
        return artefact

    missing_languages = [language for language in policy.languages if language not in scores]
    extra_languages = sorted(set(scores) - set(policy.languages))
    counts = [score.items for score in scores.values()]
    wrong_counts = [
        f"{language}: {score.items} rows (policy {policy.items_per_language})"
        for language, score in sorted(scores.items())
        if score.items != policy.items_per_language
    ]
    if not wrong_counts and counts and min(counts) != max(counts):
        wrong_counts.append(f"sheet sizes differ: {sorted(counts)}")

    reasons: list[str] = []
    if missing_languages:
        reasons.append(f"no sheet for required languages: {missing_languages}")
    if extra_languages:
        reasons.append(f"sheets for languages outside the policy: {extra_languages}")
    reasons.extend(f"row count mismatch — {item}" for item in wrong_counts)
    for language, score in sorted(scores.items()):
        reasons.extend(f"{language}: {problem}" for problem in score.problems)

    statuses = [score.status for score in scores.values()]
    if all(status == "blank" for status in statuses) and not missing_languages and not wrong_counts:
        status = "pending"
    elif any(status in {"blank", "incomplete"} for status in statuses) or missing_languages:
        status = "blocked"
    else:
        status = "open"

    for language, score in sorted(scores.items()):
        if score.status == "blank":
            reasons.append(f"{language}: no labels in the sheet")
        elif score.status == "incomplete":
            reasons.append(f"{language}: labels incomplete")
        if score.status != "scored":
            continue
        if score.unresolved:
            reasons.append(
                f"{language}: {score.unresolved} annotator disagreements lack adjudication"
            )
        if (
            score.pairwise_agreement is not None
            and score.pairwise_agreement < policy.pairwise_agreement_floor
        ):
            reasons.append(
                f"{language}: pairwise agreement {score.pairwise_agreement:.3f} < {policy.pairwise_agreement_floor}"
            )
        if score.judge_kappa is None:
            reasons.append(f"{language}: judge–human κ undefined (constant labels?)")
        elif score.judge_kappa < policy.judge_kappa_floor:
            reasons.append(
                f"{language}: judge–human κ {score.judge_kappa:.3f} < {policy.judge_kappa_floor}"
            )
    if status == "open" and reasons:
        status = "blocked"

    pooled = _pooled_metrics(sheets, policy)
    artefact["per_language"] = {
        language: score.to_dict() for language, score in sorted(scores.items())
    }
    artefact["pooled"] = pooled
    artefact["disagreements"] = len(disagreements)
    artefact["disagreement_rows"] = disagreements
    artefact["status"] = status
    artefact["judged_metrics_label"] = "calibrated" if status == "open" else "diagnostic"
    artefact["publishable"] = bool(status == "open" and not artefact["synthetic"])
    if status == "pending":
        reasons = [f"human labels absent for: {', '.join(sorted(scores))}"]
    artefact["reasons"] = reasons
    artefact["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return artefact


def _pooled_metrics(sheets: Sequence[Sheet], policy: Policy) -> dict[str, Any]:
    judge: list[bool] = []
    reference: list[bool] = []
    left_labels: list[str] = []
    right_labels: list[str] = []
    items = 0
    labelled = 0
    for sheet in sheets:
        items += len(sheet.rows)
        for row in sheet.rows:
            judge_label = _category(str(row.get("judge_label", "")), policy.labels)
            left = _category(str(row.get("human_label_a", "")), policy.labels)
            right = _category(str(row.get("human_label_b", "")), policy.labels)
            if judge_label is None and left is None and right is None:
                continue
            labelled += 1
            if left is None or right is None:
                continue
            left_labels.append(left)
            right_labels.append(right)
            reference_label = _reference(row, policy)
            judge_binary = _binary(judge_label or "")
            reference_binary = _binary(reference_label or "")
            if judge_binary is not None and reference_binary is not None:
                judge.append(judge_binary)
                reference.append(reference_binary)
    pooled: dict[str, Any] = {
        "items": items,
        "labelled": labelled,
        "judge_kappa": _round(cohen_kappa(judge, reference)) if len(judge) >= 2 else None,
        "judge_kappa_ci": bootstrap_kappa(
            judge,
            reference,
            samples=policy.bootstrap_samples,
            seed=policy.bootstrap_seed + 2,
            confidence=policy.confidence,
        )
        if len(judge) >= 3
        else None,
        "pairwise_kappa": _round(cohen_kappa(left_labels, right_labels))
        if len(left_labels) >= 2
        else None,
        "pairwise_kappa_ci": bootstrap_kappa(
            left_labels,
            right_labels,
            samples=policy.bootstrap_samples,
            seed=policy.bootstrap_seed + 3,
            confidence=policy.confidence,
        )
        if len(left_labels) >= 3
        else None,
        "pairwise_agreement": (
            round(
                sum(
                    1
                    for left, right in zip(left_labels, right_labels, strict=True)
                    if left == right
                )
                / len(left_labels),
                6,
            )
            if left_labels
            else None
        ),
    }
    return pooled


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:  # pragma: no cover - fixture outside the repo
        return path.as_posix()


# --------------------------------------------------------------------------- artefacts


def write_artefacts(artefact: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write ``gate.json``, ``gate.md`` and ``disagreements.csv`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(artefact)
    payload.setdefault("generated_at", datetime.now(UTC).isoformat(timespec="seconds"))
    gate_payload = {key: value for key, value in payload.items() if key != "disagreement_rows"}
    gate_path = out_dir / GATE_FILE
    gate_path.write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path = out_dir / "gate.md"
    markdown_path.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    disagreements_path = out_dir / "disagreements.csv"
    _write_disagreements(payload, disagreements_path)
    return {"gate": gate_path, "markdown": markdown_path, "disagreements": disagreements_path}


def _write_disagreements(artefact: Mapping[str, Any], path: Path) -> None:
    columns = (
        "item_id",
        "language",
        "judge_label",
        "human_label_a",
        "human_label_b",
        "adjudicated_label",
        "notes",
        "reason",
    )
    rows = artefact.get("disagreement_rows", [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def render_markdown(artefact: Mapping[str, Any]) -> str:
    """Human summary of the gate artefact."""
    status = str(artefact.get("status", "blocked"))
    label = str(artefact.get("judged_metrics_label", "diagnostic"))
    lines = [
        "# Judge calibration gate",
        "",
        f"- **Status:** `{status}`",
        f"- **Judged metrics:** `{label}`",
        f"- **Publishable:** `{str(bool(artefact.get('publishable', False))).lower()}`",
        f"- **Synthetic input:** `{str(bool(artefact.get('synthetic', False))).lower()}`",
        f"- **Source:** `{artefact.get('source', '')}`",
    ]
    if artefact.get("generated_at"):
        lines.append(f"- **Generated:** {artefact['generated_at']}")
    lines.extend(
        [
            "",
            "## Per language",
            "",
            "| language | rows | labelled | judge κ [95% CI] | pairwise agreement | pairwise κ | disagreements | status |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for language, block in sorted((artefact.get("per_language") or {}).items()):
        judge_ci = block.get("judge_kappa_ci") or [None, None]
        judge = block.get("judge_kappa")
        judge_text = (
            "—"
            if judge is None
            else f"{judge:.3f} [{judge_ci[0]:.3f}, {judge_ci[1]:.3f}]"
            if judge_ci[0] is not None
            else f"{judge:.3f}"
        )
        agreement = block.get("pairwise_agreement")
        agreement_text = "—" if agreement is None else f"{agreement:.3f}"
        pairwise = block.get("pairwise_kappa")
        pairwise_text = "—" if pairwise is None else f"{pairwise:.3f}"
        lines.append(
            f"| {language} | {block.get('items', 0)} | {block.get('labelled', 0)} | {judge_text} | "
            f"{agreement_text} | {pairwise_text} | {block.get('disagreements', 0)} | {block.get('status', '')} |"
        )
    pooled = artefact.get("pooled") or {}
    lines.extend(
        [
            "",
            "## Pooled",
            "",
            f"- rows: {pooled.get('items', 0)} (labelled {pooled.get('labelled', 0)})",
            f"- judge κ: {pooled.get('judge_kappa')} {pooled.get('judge_kappa_ci')}",
            f"- pairwise agreement: {pooled.get('pairwise_agreement')}",
            f"- pairwise κ: {pooled.get('pairwise_kappa')} {pooled.get('pairwise_kappa_ci')}",
        ]
    )
    reasons = artefact.get("reasons") or []
    if reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "`judged_metrics_label` is `calibrated` only when status is `open`; a missing,",
            "corrupt or stale gate is treated as `diagnostic` by `load_judged_label` (fail-closed).",
            "Synthetic dry-run fixtures can reach `open` to prove the mechanics but are never",
            "`publishable`: real human labels are required for a reportable claim.",
            "",
        ]
    )
    return "\n".join(lines)


def load_judged_label(path: Path | None = None) -> str:
    """Consumer entry point: ``"calibrated"`` or ``"diagnostic"``; fail-closed."""
    gate_path = Path(path) if path is not None else DEFAULT_OUT_DIR / GATE_FILE
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "diagnostic"
    if str(payload.get("status")) != "open":
        return "diagnostic"
    if bool(payload.get("synthetic", False)):
        return "diagnostic"
    return "calibrated" if bool(payload.get("publishable", False)) else "diagnostic"


# --------------------------------------------------------------------------- self-check


def check() -> dict[str, Any]:
    """Offline self-check used by ``score-judge-calibration.py --check``.

    Proves, without network and without touching committed artefacts: the committed sheets are
    pending; the committed gate agrees; the dry-run fixtures flip the gate when labels are
    good and block it when they are not; malformed sheets are rejected; fixtures are marked
    synthetic and contain no real human data.
    """
    results: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    policy = Policy.load()
    committed = evaluate(DEFAULT_SHEETS_DIR, policy)
    record(
        "committed_sheets_pending",
        committed["status"] == "pending" and committed["judged_metrics_label"] == "diagnostic",
        f"status={committed['status']} label={committed['judged_metrics_label']}",
    )
    gate_path = DEFAULT_OUT_DIR / GATE_FILE
    if gate_path.is_file():
        stored = json.loads(gate_path.read_text(encoding="utf-8"))
        same = _comparable(stored) == _comparable(committed)
        record(
            "committed_gate_matches",
            same,
            f"stored={stored.get('status')} recomputed={committed['status']}",
        )
    else:
        record("committed_gate_matches", False, f"{gate_path} is missing")

    fixture_policy = Policy.load(FIXTURES_DIR / "policy.json")
    high = evaluate(FIXTURES_DIR / "high", fixture_policy)
    record(
        "dry_run_high_opens",
        high["status"] == "open"
        and high["judged_metrics_label"] == "calibrated"
        and high["synthetic"],
        f"status={high['status']} kappa={high['pooled'].get('judge_kappa')}",
    )
    record(
        "dry_run_high_not_publishable",
        high["publishable"] is False,
        f"publishable={high['publishable']}",
    )
    again = evaluate(FIXTURES_DIR / "high", fixture_policy)
    record(
        "dry_run_deterministic",
        _comparable(high) == _comparable(again),
        "two evaluations agree apart from generated_at",
    )
    low = evaluate(FIXTURES_DIR / "low", fixture_policy)
    record(
        "dry_run_low_blocks",
        low["status"] == "blocked",
        f"status={low['status']} reasons={len(low['reasons'])}",
    )
    partial = evaluate(FIXTURES_DIR / "partial", fixture_policy)
    record(
        "dry_run_partial_blocks",
        partial["status"] == "blocked",
        f"status={partial['status']}",
    )
    record(
        "fixtures_marked_synthetic",
        _fixtures_are_marked(fixture_policy),
        "every dry-run row and filename carries the synthetic marker",
    )
    record(
        "fail_closed_without_gate",
        load_judged_label(FIXTURES_DIR / "does-not-exist.json") == "diagnostic",
        "missing gate → diagnostic",
    )
    record(
        "invalid_sheet_rejected",
        _invalid_sheet_rejected(fixture_policy),
        "bad label / duplicate id / wrong columns block the gate",
    )
    record(
        "no_human_labels_committed",
        _no_human_labels(DEFAULT_SHEETS_DIR, policy),
        "committed external sheets have empty human columns",
    )
    return {
        "ok": all(result["ok"] for result in results),
        "policy": policy.to_dict(),
        "checks": results,
    }


def _comparable(artefact: Mapping[str, Any]) -> dict[str, Any]:
    skip = {"generated_at", "disagreement_rows"}
    return {key: value for key, value in artefact.items() if key not in skip}


def _fixtures_are_marked(policy: Policy) -> bool:
    for path in sorted(FIXTURES_DIR.rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(
            policy.synthetic_marker not in str(row.get("notes", "")) for row in rows
        ):
            return False
    return True


def _invalid_sheet_rejected(policy: Policy) -> bool:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibration-ru.csv"
        path.write_text(
            "item_id,language,item_ref,question,answer_text,judge_label,human_label_a,human_label_b,adjudicated_label,notes\n"
            "dup,ru,ref,,,,,,,SYNTHETIC-DRY-RUN\n"
            "dup,ru,ref,,,,,,,SYNTHETIC-DRY-RUN\n",
            encoding="utf-8",
        )
        artefact = evaluate(Path(directory), policy)
        return artefact["status"] == "blocked" and any(
            "duplicate" in reason for reason in artefact["reasons"]
        )


def _no_human_labels(directory: Path, policy: Policy) -> bool:
    if not directory.is_dir():
        return True
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for column in ("human_label_a", "human_label_b", "adjudicated_label"):
                    if str(row.get(column, "")).strip():
                        return False
    return True


def collect_disagreements(directory: Path, policy: Policy) -> list[dict[str, str]]:
    """All annotator-disagreement rows for a sheet directory (for adjudication)."""
    sheets = discover_sheets(directory, policy)
    return [row for sheet in sheets for row in disagreement_rows(sheet, policy)]


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_SHEETS_DIR",
    "FIXTURES_DIR",
    "GATE_FILE",
    "GateError",
    "LanguageScore",
    "Policy",
    "SYNTHETIC_MARKER",
    "bootstrap_kappa",
    "check",
    "cohen_kappa",
    "collect_disagreements",
    "disagreement_rows",
    "evaluate",
    "load_judged_label",
    "load_sheet",
    "render_markdown",
    "score_language",
    "write_artefacts",
]
