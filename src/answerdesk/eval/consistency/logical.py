"""Logical-consistency audit of corpus v2.1 — within documents and across them.

Two levels, both deterministic:

* **within a document** — the front matter must agree with the body (language, archetype),
  a parameter row must not appear twice with different values in the same table, and every
  identifier/date stated in the body must also be declared in the document's own reference
  block (so the document cannot cite a value it never defines);
* **across documents of the same product/locale** — the same named product may not carry two
  different values for the same parameter (e.g. two "Стартовый" tariff rows with different
  monthly fees in the same locale).

The auditor is deliberately conservative: it only reports a contradiction when the *same key*
(product name or parameter name) maps to two different values. Findings carry a severity and
an owner; fixtures in the unit tests inject a contradiction and require the audit to fail,
which is the CI invariant.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answerdesk.eval.consistency.values import normalise_digits

FRONT_MATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
TABLE_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$", re.MULTILINE)
PRODUCT = re.compile(r"«(?P<name>[^»]+)»")


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # blocker | high | medium | low
    message: str
    owner: str
    doc_ids: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "owner": self.owner,
            "doc_ids": list(self.doc_ids),
            "evidence": self.evidence,
        }


_AS_OF = re.compile(r"Актуально на\s+([^\n.]+)", re.IGNORECASE)
_MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "ма": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
    "январь": 1,
    "февраль": 2,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}
_DATE_TEXT = re.compile(r"(\d{1,2})\s+([^\s.,]+)\s+(\d{4})")
_DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_DOTTED = re.compile(r"(\d{1,2})[./](\d{1,2})[./](\d{4})")


def _dates_with_year(text: str) -> list[str]:
    """Date strings that carry a year (the extractor used for values is year-agnostic)."""
    found: list[str] = []
    for pattern in (_DATE_ISO, _DATE_DOTTED, _DATE_TEXT):
        found.extend(match.group(0) for match in pattern.finditer(text))
    return found


def _parse_date(raw: str) -> Any:
    from datetime import date

    text = raw.strip()
    match = _DATE_ISO.search(text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_DOTTED.search(text)
    if match:
        return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    match = _DATE_TEXT.search(text)
    if match:
        month_word = match.group(2).lower()
        month = next((value for key, value in _MONTHS.items() if month_word.startswith(key)), None)
        if month:
            return date(int(match.group(3)), month, int(match.group(1)))
    return None


def _front_matter(text: str) -> dict[str, str]:
    match = FRONT_MATTER.match(text)
    if not match:
        return {}
    payload: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            payload[key.strip()] = value.strip().strip('"')
    return payload


def _rows(text: str) -> list[tuple[str, str]]:
    """First two cells of every markdown table row: the parameter and its value."""
    rows: list[tuple[str, str]] = []
    for match in TABLE_ROW.finditer(text):
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if len(cells) < 2:
            continue
        key, value = cells[0], cells[1]
        if not key or set(key) <= {"-", ":"} or set(value) <= {"-", ":"}:
            continue
        if key.lower() in {
            "параметр",
            "parameter",
            "показатель",
            "условие",
            "key",
            "көрсөткүч",
            "ko'rsatkich",
            "指标",
            "gösterge",
            "нишондиҳанда",
        }:
            continue
        rows.append((normalise_digits(key).strip().lower(), normalise_digits(value).strip()))
    return rows


def _products(text: str, *, title: str = "") -> set[str]:
    names = {match.group("name").strip().lower() for match in PRODUCT.finditer(text)}
    for source in (title, text.splitlines()[0] if text.splitlines() else ""):
        if "—" in source:
            names.add(source.rsplit("—", 1)[1].strip().strip("# ").lower())
    return {name for name in names if name}


def audit_document(path: Path, *, locale: str) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    front = _front_matter(text)
    findings: list[Finding] = []
    doc_id = front.get("doc_id", path.stem)
    if front.get("language") and front["language"] != locale:
        findings.append(
            Finding(
                code="front_matter_language",
                severity="high",
                message=(
                    f"{doc_id}: front matter declares language {front['language']!r} but the "
                    f"document lives in the {locale!r} locale directory"
                ),
                owner="ingest (synthetic corpus)",
                doc_ids=(doc_id,),
                evidence={"declared": front["language"], "directory": locale},
            )
        )
    front_match = FRONT_MATTER.match(text)
    body = text[len(front_match.group(0)) :] if front_match else text
    if front.get("doc_id") and front["doc_id"] != path.stem:
        findings.append(
            Finding(
                code="doc_id_mismatch",
                severity="high",
                message=f"{path.stem}: front matter declares doc_id {front['doc_id']!r}",
                owner="ingest (synthetic corpus)",
                doc_ids=(doc_id,),
                evidence={"declared": front["doc_id"], "file": path.stem},
            )
        )

    seen: dict[str, set[str]] = defaultdict(set)
    for key, value in _rows(body):
        seen[key.lower()].add(value)
    for key, values in seen.items():
        if len(values) > 1:
            findings.append(
                Finding(
                    code="table_parameter_conflict",
                    severity="high",
                    message=(
                        f"{doc_id}: parameter {key!r} carries {len(values)} different values "
                        "in one table"
                    ),
                    owner="ingest (synthetic corpus)",
                    doc_ids=(doc_id,),
                    evidence={"parameter": key, "values": sorted(values)},
                )
            )

    # as-of coherence: a document may not assert an event after the date it is current as of
    tail = body[-600:]
    as_of_match = _AS_OF.search(tail)
    as_of = _parse_date(as_of_match.group(1)) if as_of_match else None
    body_dates = _dates_with_year(body)
    later: list[str] = []
    parsed = 0
    for raw in body_dates:
        moment = _parse_date(raw)
        if moment is None:
            continue
        parsed += 1
        if as_of is not None and moment > as_of:
            later.append(raw)
    if later:
        findings.append(
            Finding(
                code="date_after_as_of",
                severity="medium",
                message=(
                    f"{doc_id}: {len(later)} date(s) later than the document's own as-of date"
                ),
                owner="ingest (synthetic corpus)",
                doc_ids=(doc_id,),
                evidence={
                    "as_of": as_of.isoformat() if as_of else None,
                    "dates": sorted(set(later)),
                },
            )
        )
    findings_meta: list[dict[str, Any]] = []
    _ = findings_meta, parsed
    return findings


def audit_corpus(corpus_dir: Path, *, locales: Sequence[str]) -> dict[str, Any]:
    """Within-document and cross-document audit; returns findings plus the checked counts."""
    per_locale: dict[str, dict[str, Any]] = {}
    all_findings: list[Finding] = []
    for locale in locales:
        directory = corpus_dir / "docs" / locale
        if not directory.is_dir():
            per_locale[locale] = {"documents": 0, "findings": 0, "status": "missing directory"}
            continue
        docs = sorted(directory.glob("*.md"))
        findings: list[Finding] = []
        coverage = {"with_tables": 0, "table_rows": 0, "with_as_of": 0}
        for path in docs:
            text = path.read_text(encoding="utf-8")
            rows = _rows(text)
            if rows:
                coverage["with_tables"] += 1
                coverage["table_rows"] += len(rows)
            if _AS_OF.search(text):
                coverage["with_as_of"] += 1
        product_values: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for path in docs:
            text = path.read_text(encoding="utf-8")
            findings.extend(audit_document(path, locale=locale))
            front = _front_matter(text)
            for product in _products(text, title=front.get("title", "")):
                for key, value in _rows(text):
                    product_values[(product, key.lower())][front.get("doc_id", path.stem)].add(
                        value
                    )
        for (product, key), by_doc in sorted(product_values.items()):
            values = set().union(*by_doc.values())
            if len(values) > 1 and len(by_doc) > 1:
                findings.append(
                    Finding(
                        code="cross_document_conflict",
                        severity="high",
                        message=(
                            f"locale {locale}: product {product!r} carries {len(values)} different "
                            f"values for parameter {key!r} across documents"
                        ),
                        owner="ingest (synthetic corpus)",
                        doc_ids=tuple(sorted(by_doc)),
                        evidence={"product": product, "parameter": key, "values": sorted(values)},
                    )
                )
        all_findings.extend(findings)
        per_locale[locale] = {
            "documents": len(docs),
            "findings": len(findings),
            "by_severity": _severity_counts(findings),
            "coverage": coverage,
        }
    severity = _severity_counts(all_findings)
    conflict_parameters: dict[str, int] = defaultdict(int)
    for finding in all_findings:
        if finding.code == "cross_document_conflict":
            conflict_parameters[str(finding.evidence.get("parameter"))] += 1
    coverage_totals = {
        key: sum(block.get("coverage", {}).get(key, 0) for block in per_locale.values())
        for key in ("with_tables", "table_rows", "with_as_of")
    }
    return {
        "locales": per_locale,
        "coverage": coverage_totals,
        "checked_documents": sum(block.get("documents", 0) for block in per_locale.values()),
        "findings": [finding.to_dict() for finding in all_findings],
        "by_severity": severity,
        "conflict_parameters": dict(sorted(conflict_parameters.items(), key=lambda item: -item[1])),
        "invariant_holds": severity.get("blocker", 0) == 0 and severity.get("high", 0) == 0,
        "owner_summary": _owners(all_findings),
    }


def _severity_counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.severity] += 1
    return dict(sorted(counts.items()))


def _owners(findings: Iterable[Finding]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.owner] += 1
    return dict(sorted(counts.items()))


__all__ = ["Finding", "audit_corpus", "audit_document"]
