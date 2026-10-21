#!/usr/bin/env python
"""Check that every number published in the docs is backed by an artifact (NFR-REP-01).

Two claims, two sources:

* **Level A — README.** Every quantity in ``README.md`` must appear in the generated
  ``docs/results.md`` or ``docs/external-validity.md``. The README may quote the evidence,
  never invent it.
* **Level B — results.** Every quantity in ``docs/results.md`` must appear in the committed
  artifacts under ``reports/`` (or be a requirement threshold copied from
  ``docs/requirements.md``, or a documented exception).

Fenced code blocks are ignored: they contain commands and captured program output, whose
numbers are proven by executing the command, not by an artifact. Hashes, ISO dates, version
strings, section references and ``AT-/FR-/NFR-`` ids are not quantities and are skipped.

Usage
-----
    uv run python scripts/check-numbers.py             # summary, exit 0/1
    uv run python scripts/check-numbers.py --verbose   # list every checked number
    uv run python scripts/check-numbers.py --json

Exit codes: 0 = every checked number is traceable; 1 = at least one is not; 2 = inputs missing.

Documented exceptions live in ``EXCEPTIONS`` below and are printed on every run, so an
exception can never hide silently. Add one only with a reason that names the artifact or the
command that proves the number instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

README = Path("README.md")
RESULTS = Path("docs") / "results.md"
EXTERNAL = Path("docs") / "external-validity.md"
REQUIREMENTS = Path("docs") / "requirements.md"
REPORTS = Path("reports")
ARTIFACT_SUFFIXES = (".json", ".csv", ".md", ".txt", ".log")

_NUMBER = re.compile(r"(?<![\w.§#/-])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_HASH = re.compile(r"\b[0-9a-f]{16,}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_VERSION = re.compile(r"\bv?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b")
_SECTION = re.compile(r"§\s*\d+(?:\.\d+)*")
_ID = re.compile(r"\b(?:AT|FR|NFR|INV|RT|CMD|CL|ADR|D|R|NG|Q|E|F|K)-[A-Z0-9]+-?\d+\b")


@dataclass(frozen=True)
class Exception_:
    """A number the checker is allowed to skip, with the reason it is still honest."""

    pattern: str
    level: str  # "A" (README), "B" (results), or "AB"
    reason: str


#: Print on every run. Each entry must name what actually proves the number.
EXCEPTIONS: tuple[Exception_, ...] = (
    Exception_(
        r"^95$", "AB", "the conventional confidence level (95% CI); a definition, not a measurement"
    ),
    Exception_(r"^1\.0\.0$", "AB", "the contracts schema version in docs/requirements.md"),
    Exception_(r"^0\.1\.0$", "AB", "the package version (pyproject.toml)"),
    Exception_(r"^1\d{3}$", "B", "a four-digit year inside a model id or date string"),
    Exception_(
        r"^5173$", "A", "the Vite dev-server port printed by `npm run dev` (web/vite.config.ts)"
    ),
    Exception_(
        r"^8192$",
        "B",
        "token-budget parameter of `scripts/reasoning-side-study.py`, not a measurement",
    ),
    Exception_(
        r"^16384$",
        "B",
        "token-budget parameter of `scripts/reasoning-side-study.py`, not a measurement",
    ),
    Exception_(
        r"^(1|2|3|4|5|6|7|8|9|10|11|12)$",
        "B",
        "small counts used as table row/column labels (not claims)",
    ),
)


@dataclass
class Claim:
    number: str
    canonical: str
    source: str
    line: int


@dataclass
class Report:
    checked: int = 0
    unverified: list[Claim] = field(default_factory=list)
    exceptions_used: dict[str, int] = field(default_factory=dict)
    level_a_checked: int = 0
    level_b_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.unverified


def canonical(token: str) -> str:
    """Normalise a number token: drop thousands separators and a trailing percent sign."""
    value = token.replace(",", "").replace("%", "")
    if value.startswith("+"):
        value = value[1:]
    try:
        number = float(value)
    except ValueError:  # pragma: no cover - the regex only matches numbers
        return value
    if number == int(number) and "." not in value and "e" not in value.lower():
        return str(int(number))
    return repr(number)


def variants(canonical_value: str) -> set[str]:
    """Accept equivalent renderings (1.2 vs 1.20 vs 1.200, 141 vs 141.0)."""
    out = {canonical_value}
    try:
        number = float(canonical_value)
    except ValueError:  # pragma: no cover
        return out
    if number == int(number):
        out.add(str(int(number)))
        out.add(f"{int(number)}.0")
    for digits in (0, 1, 2, 3, 4):
        out.add(f"{number:.{digits}f}")
    # percentage renderings of a fraction (0.908 -> 90.8)
    if 0.0 < abs(number) <= 1.0:
        out.add(f"{number * 100:.1f}")
        out.add(f"{number * 100:.2f}")
        out.add(str(round(number * 100)))
    return out


def strip_noise(text: str) -> str:
    """Remove constructs whose digits are not quantities."""
    for pattern in (_HASH, _DATE, _VERSION, _SECTION, _ID):
        text = pattern.sub(" ", text)
    # standards numbers and language version strings are identifiers, not quantities
    text = re.sub(r"\b(?:RFC|ISO|IEC|IEEE|U\+)\s*\d+\b", " ", text)
    text = re.sub(r"\bPython\s+\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?", "Python", text)
    return text


def numbers_in(text: str, *, skip_fences: bool = True) -> list[tuple[str, int]]:
    """Return (token, line number) pairs for the quantities in a document."""
    if skip_fences:
        # keep line numbering: replace fenced blocks with the same number of blank lines
        text = _FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    cleaned = strip_noise(text)
    found: list[tuple[str, int]] = []
    for index, line in enumerate(cleaned.splitlines(), start=1):
        for match in _NUMBER.finditer(line):
            found.append((match.group(0), index))
    return found


def source_pool(paths: Iterable[Path], *, skip_fences: bool = False) -> set[str]:
    """Every canonical number appearing in the source texts (plus their variants)."""
    pool: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            continue
        for token, _ in numbers_in(text, skip_fences=skip_fences):
            pool |= variants(canonical(token))
    return pool


ARTIFACT_GLOBS: tuple[str, ...] = (
    "reports/**/*.json",
    "reports/**/*.csv",
    "reports/**/*.md",
    "data/bench/**/*.json",
    "data/corpus/manifest.json",
    "evals/**/manifest.json",
    "configs/**/*.yaml",
)


def artifact_paths(root: Path) -> list[Path]:
    """Committed artifacts that may back a published number."""
    found: list[Path] = []
    for pattern in ARTIFACT_GLOBS:
        found.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(found))


def _exception_for(token: str, level: str | Sequence[str]) -> Exception_ | None:
    levels = {level} if isinstance(level, str) else set(level)
    for entry in EXCEPTIONS:
        if levels & set(entry.level) and re.fullmatch(entry.pattern, token):
            return entry
    return None


def check(root: Path) -> tuple[Report, dict[str, set[str]]]:
    report = Report()
    readme = root / README
    results = root / RESULTS
    external = root / EXTERNAL
    requirements = root / REQUIREMENTS
    for path in (readme, results):
        if not path.is_file():
            raise SystemExit(f"check-numbers: {path} not found (wrong --root?)")

    evidence_docs = source_pool([results, external])
    artifact_docs = artifact_paths(root)
    evidence_artifacts = source_pool(artifact_docs)
    evidence_requirements = source_pool([requirements])

    def verify(document: Path, level: str, sources: set[str]) -> None:
        for token, line in numbers_in(document.read_text(encoding="utf-8")):
            report.checked += 1
            if level == "A":
                report.level_a_checked += 1
            else:
                report.level_b_checked += 1
            exception = _exception_for(token, level)
            if exception is not None:
                report.exceptions_used[exception.pattern] = (
                    report.exceptions_used.get(exception.pattern, 0) + 1
                )
                continue
            if variants(canonical(token)) & sources:
                continue
            report.unverified.append(
                Claim(number=token, canonical=canonical(token), source=str(document), line=line)
            )

    # Level A may quote the generated docs OR a committed artifact (e.g. a dataset manifest)
    verify(readme, "A", evidence_docs | evidence_artifacts)
    verify(results, "B", evidence_artifacts | evidence_requirements)
    return report, {"level_a": evidence_docs, "level_b": evidence_artifacts}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (default: parent of scripts/)"
    )
    parser.add_argument("--verbose", action="store_true", help="print every checked number")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict line")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    report, _sources = check(root)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "checked": report.checked,
                    "level_a": report.level_a_checked,
                    "level_b": report.level_b_checked,
                    "exceptions_used": report.exceptions_used,
                    "unverified": [
                        {"number": c.number, "source": c.source, "line": c.line}
                        for c in report.unverified
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if report.ok else 1

    if not args.quiet:
        print(
            f"numbers: {report.checked} quantities "
            f"(README {report.level_a_checked} · results {report.level_b_checked}); "
            f"{len(report.exceptions_used)} documented exception pattern(s) used"
        )
    if report.exceptions_used and not args.quiet:
        by_pattern = {e.pattern: e for e in EXCEPTIONS}
        for pattern, count in sorted(report.exceptions_used.items()):
            print(f"  exception {pattern!r} ×{count}: {by_pattern[pattern].reason}")
    if args.verbose:
        for claim in report.unverified:
            print(f"  UNVERIFIED {claim.number} at {claim.source}:{claim.line}")
    if not report.ok:
        print(f"\nFAIL: {len(report.unverified)} quantity(ies) have no artifact backing:")
        for claim in report.unverified[:40]:
            print(f"  - {claim.number}  ({claim.source}:{claim.line})")
        if len(report.unverified) > 40:
            print(f"  … and {len(report.unverified) - 40} more")
        print(
            "\n  fix: generate the number from an artifact (scripts/build-results.py), or add a"
            " documented exception with the command that proves it."
        )
        return 1
    print("\nOK: every published quantity traces to the generated docs or the committed artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
