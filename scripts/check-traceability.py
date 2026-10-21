#!/usr/bin/env python
"""Mechanical enforcement of the requirements traceability matrix (NFR-REP-03, CMD-T).

The contract (``docs/requirements.md`` §16, §17.2):

* every ``AT-…`` id that appears in the §16 matrix must appear as a marker somewhere in the
  code, tests or scripts that implement it;
* every module path named in the §16 matrix must resolve to a real file or directory;
* the checker exits non-zero and lists every gap it found.

Two independent failure modes are covered: a requirement whose acceptance test silently
disappeared (marker missing), and a matrix that points at code that was moved or renamed
(stale path). Both were real in the release candidate — see ``docs/review-report.md`` R-01 and
the change-control notes in ``docs/requirements.md`` §18.5.

Usage
-----
    uv run python scripts/check-traceability.py            # human summary, exit 0/1
    uv run python scripts/check-traceability.py --quiet    # only gaps, if any
    uv run python scripts/check-traceability.py --json     # machine-readable summary
    uv run python scripts/check-traceability.py --root DIR # check another checkout (tests use this)

Exit codes: 0 = every AT id has a marker and every path exists; 1 = gaps found;
2 = the matrix or the tree could not be read (a broken checkout, not a content gap).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REQUIREMENTS = Path("docs") / "requirements.md"

#: Trees scanned for ``AT-…`` evidence markers, relative to the repository root.
MARKER_TREES: tuple[str, ...] = ("src", "tests", "scripts", "web")

#: File suffixes that can carry a marker.
MARKER_SUFFIXES: tuple[str, ...] = (".py", ".ts", ".tsx", ".js", ".mjs", ".ps1", ".md")

#: Paths in the matrix that are repository-relative, not ``src/answerdesk/``-relative.
REPO_PREFIXES: tuple[str, ...] = (
    "src/",
    "tests/",
    "scripts/",
    "docs/",
    "data/",
    "evals/",
    "configs/",
    "web/",
    ".github/",
)

#: Repository-level files that may appear bare in a module cell.
REPO_FILES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "README.ru.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        ".env.example",
        ".gitignore",
    }
)

_MATRIX_ROW = re.compile(r"^\|\s*((?:N?FR)-[A-Z0-9]+-\d+)\s*\|")
_BACKTICKED = re.compile(r"`([^`]+)`")
_AT_ID = re.compile(r"AT-[A-Z0-9]+-\d+")
_MARKER_IN_TEXT = re.compile(r"AT-[A-Z0-9]+-\d+")


@dataclass
class Matrix:
    """The parsed §16 matrix: one entry per requirement row."""

    rows: int = 0
    at_ids: list[str] = field(default_factory=list)
    paths: list[tuple[str, str]] = field(default_factory=list)  # (requirement, token)
    prose_only: list[str] = field(default_factory=list)  # rows whose module cell names no path

    @property
    def unique_at_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for at_id in self.at_ids:
            seen.setdefault(at_id, None)
        return list(seen)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_matrix(requirements: Path) -> Matrix:
    """Extract rows, AT ids and module paths from the §16 tables."""
    text = requirements.read_text(encoding="utf-8")
    try:
        start = text.index("## 16. Traceability matrix")
        end = text.index("## 17. Verification index")
    except ValueError as exc:  # pragma: no cover - only on a gutted document
        raise SystemExit(
            f"check-traceability: cannot locate §16/§17 in {requirements}: {exc}"
        ) from exc

    matrix = Matrix()
    for line in text[start:end].splitlines():
        match = _MATRIX_ROW.match(line)
        if not match:
            continue
        requirement = match.group(1)
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        matrix.rows += 1
        module_cell, at_cell = cells[1], cells[3]
        found = _AT_ID.findall(at_cell)
        if not found:
            raise SystemExit(
                f"check-traceability: row {requirement} has no AT id in its acceptance column"
            )
        matrix.at_ids.extend(found)
        tokens = [token.strip() for token in _BACKTICKED.findall(module_cell) if token.strip()]
        if tokens:
            matrix.paths.extend((requirement, token) for token in tokens)
        else:
            matrix.prose_only.append(requirement)
    if matrix.rows == 0:  # pragma: no cover - guards a formatting accident
        raise SystemExit("check-traceability: §16 contains no requirement rows")
    return matrix


def resolve_path(token: str, root: Path) -> Path:
    """Resolve a matrix path token to a filesystem path.

    Module cells name either a repository-relative path (``docs/results.md``, ``web/src/``)
    or a path inside the package (``ingest/chunker.py``). Anything without a known repository
    prefix is resolved under ``src/answerdesk/``, which is where module code lives.
    """
    cleaned = token.strip()
    if cleaned in REPO_FILES or cleaned.startswith(REPO_PREFIXES):
        return root / cleaned
    return root / "src" / "answerdesk" / cleaned


def collect_markers(root: Path) -> dict[str, list[str]]:
    """Map every AT id found in the marker trees to the files that mention it."""
    markers: dict[str, list[str]] = {}
    for tree in MARKER_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in MARKER_SUFFIXES:
                continue
            if "node_modules" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - unreadable file is not a content gap
                continue
            if "AT-" not in text:
                continue
            for at_id in set(_MARKER_IN_TEXT.findall(text)):
                markers.setdefault(at_id, []).append(str(path.relative_to(root)))
    return markers


def check(root: Path) -> dict[str, object]:
    """Run both checks; returns a JSON-serialisable report."""
    requirements = root / REQUIREMENTS
    if not requirements.is_file():
        raise SystemExit(f"check-traceability: {requirements} not found (wrong --root?)")

    matrix = parse_matrix(requirements)
    markers = collect_markers(root)

    missing_markers = [at_id for at_id in matrix.unique_at_ids if at_id not in markers]
    unknown_markers = sorted(set(markers) - set(matrix.unique_at_ids))

    missing_paths: list[dict[str, str]] = []
    checked_paths = 0
    seen: set[tuple[str, str]] = set()
    for requirement, token in matrix.paths:
        if (requirement, token) in seen:
            continue
        seen.add((requirement, token))
        checked_paths += 1
        if not resolve_path(token, root).exists():
            missing_paths.append(
                {
                    "requirement": requirement,
                    "token": token,
                    "resolved": str(resolve_path(token, root).relative_to(root)),
                }
            )

    return {
        "rows": matrix.rows,
        "at_ids": len(matrix.unique_at_ids),
        "marked_at_ids": len(matrix.unique_at_ids) - len(missing_markers),
        "markers_found": len(markers),
        "path_tokens": checked_paths,
        "missing_markers": missing_markers,
        "missing_paths": missing_paths,
        "unknown_markers": unknown_markers,
        "prose_only_rows": matrix.prose_only,
        "ok": not missing_markers and not missing_paths,
    }


def _print_human(report: dict[str, object], *, quiet: bool) -> None:
    missing_markers: Sequence[str] = report["missing_markers"]  # type: ignore[assignment]
    missing_paths: Sequence[dict[str, str]] = report["missing_paths"]  # type: ignore[assignment]
    unknown: Sequence[str] = report["unknown_markers"]  # type: ignore[assignment]
    ok = bool(report["ok"])

    if not quiet or not ok:
        print(
            f"traceability: {report['rows']} rows · "
            f"{report['marked_at_ids']}/{report['at_ids']} AT ids marked · "
            f"{report['path_tokens']} path references checked"
        )
    if missing_markers:
        print(
            f"\nFAIL: {len(missing_markers)} AT id(s) have no marker in {'|'.join(MARKER_TREES)}:"
        )
        for at_id in missing_markers:
            print(f"  - {at_id}")
        print("  fix: add `# <AT id>` to the test that proves the requirement, or correct §16.")
    if missing_paths:
        print(f"\nFAIL: {len(missing_paths)} matrix path(s) do not exist:")
        for item in missing_paths:
            print(f"  - {item['requirement']}: `{item['token']}` → {item['resolved']}")
        print("  fix: point the row at the real module, or record the correction in §18.5.")
    if unknown:
        print(
            f"\nnote: {len(unknown)} marker id(s) are not in §16 (stale marker or typo): "
            + ", ".join(unknown[:12])
            + (" …" if len(unknown) > 12 else "")
        )
    prose = report["prose_only_rows"]
    if prose and not quiet:
        print(
            f"\nnote: {len(prose)} row(s) name no module path in prose form "  # type: ignore[arg-type]
            f"(not a failure): {', '.join(prose)}"
        )
    if ok:
        print("\nOK: every AT id has a marker and every matrix path resolves.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (default: parent of scripts/)"
    )
    parser.add_argument("--quiet", action="store_true", help="print only gaps")
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    parser.add_argument(
        "--strict-markers",
        action="store_true",
        help="also fail on markers whose AT id is absent from the matrix",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.root or repo_root()).resolve()
    report = check(root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report, quiet=args.quiet)
    if not report["ok"]:
        return 1
    if args.strict_markers and report["unknown_markers"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
