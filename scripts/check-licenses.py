#!/usr/bin/env python
"""Offline licence audit of the installed dependency tree (NFR-REP-04, AT-REP-04).

The repository ships MIT code and must not pull a copyleft dependency into the runtime tree.
CI has no network and ``pip-audit`` cannot run here, so this checker audits what the offline
evidence can actually prove:

1. **Metadata first** — every installed distribution's ``License-Expression`` / classifiers.
2. **Bundled licence text second** — for distributions whose metadata says ``UNKNOWN``, read the
   ``LICENSE``/``COPYING`` file shipped inside the distribution and classify its text.
3. **Allow/deny policy** — MIT, BSD-2/3/4, Apache-2.0, ISC, PSF, MPL-2.0, Zlib, 0BSD, CC0,
   Unlicense and MIT-CMU are allowed; GPL-family, AGPL, SSPL, BUSL, Commons Clause and
   non-commercial terms are rejected regardless of how they are spelled.
4. **Repository posture** — ``LICENSE`` is the MIT text, the allowlist is documented in
   ``docs/requirements.md`` §13.3, and no code from the known licence traps (Dify, FastGPT,
   Verba, R2R) is imported.

What it does **not** do: verify transitive licences of packages that are not installed, or the
upstream text of a classifier-only claim. Those are ``pip-audit``/review obligations and are
labelled as such in ``docs/reproducibility.md``.

Usage
-----
    uv run python scripts/check-licenses.py            # table + verdict, exit 0/1
    uv run python scripts/check-licenses.py --json
    uv run python scripts/check-licenses.py --quiet

Exit codes: 0 = every installed distribution is allowed or explicitly overridden; 1 = a
forbidden or unidentifiable licence; 2 = the repository posture checks failed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ALLOWED_PREFIXES: tuple[str, ...] = (
    "mit",
    "bsd",
    "apache",
    "isc",
    "psf",
    "python software foundation",
    "mpl",
    "mozilla public license",
    "zlib",
    "0bsd",
    "cc0",
    "unlicense",
    "mit-cmu",
    "historical permission",
)

FORBIDDEN = re.compile(
    r"(^|\W)("
    r"agpl|affero|"
    r"gpl(?!\s*-?\s*classpath)|general public license|"
    r"sspl|server side public license|"
    r"busl|business source license|"
    r"commons clause|non[- ]?commercial|proprietary|"
    r"cc[- ]by[- ]nc|"
    r"european union public licence"
    r")(\W|$)",
    re.IGNORECASE,
)

LICENCE_FILE = re.compile(
    r"(?:^|/)(licen[cs]e|copying|notice)(?:\.(?:txt|md|rst))?$", re.IGNORECASE
)

#: Distributions whose metadata *and* bundled text are silent. Each entry states the evidence.
OVERRIDES: dict[str, tuple[str, str]] = {
    "coverage": (
        "Apache-2.0",
        "classifier metadata absent; upstream is Apache-2.0 (LICENSE.rst not shipped in the wheel)",
    ),
    "python-dotenv": ("BSD-3-Clause", "classifier metadata absent; upstream is BSD-3-Clause"),
}

TRAPS = ("dify", "fastgpt", "verba", "r2r")


@dataclass
class Entry:
    name: str
    version: str
    licence: str
    source: str  # "metadata" | "licence-file" | "override" | "unknown"
    allowed: bool
    note: str = ""


def _classify_text(text: str) -> str:
    head = text[:4000].lower()
    if "gnu affero general public license" in head:
        return "AGPL"
    if "gnu general public license" in head or "general public license" in head:
        return "GPL"
    if "mozilla public license" in head:
        return "MPL-2.0"
    if "apache license" in head:
        return "Apache-2.0"
    if "redistribution and use in source and binary forms" in head:
        if "3." in head and "endorse" in head:
            return "BSD-3-Clause"
        return "BSD"
    if "permission is hereby granted, free of charge" in head:
        return "MIT"
    if "isc license" in head or "permission to use, copy, modify, and/or distribute" in head:
        return "ISC"
    if "python software foundation" in head or "psf" in head[:200]:
        return "PSF-2.0"
    if "zlib" in head[:400] and "altered source versions" in head:
        return "Zlib"
    if (
        "unlicense" in head
        or "this is free and unencumbered software released into the public domain" in head
    ):
        return "Unlicense"
    return ""


def _from_metadata(dist: metadata.Distribution) -> str:
    expression = (dist.metadata.get("License-Expression") or "").strip()
    if expression and expression.upper() != "UNKNOWN":
        return expression
    classifiers = dist.metadata.get_all("Classifier") or []
    parts = [c.split("::")[-1].strip() for c in classifiers if c.startswith("License ::")]
    if parts:
        return "; ".join(parts)
    plain = (dist.metadata.get("License") or "").strip()
    if plain and plain.upper() != "UNKNOWN" and len(plain) < 200:
        return plain
    return ""


def _from_licence_file(dist: metadata.Distribution) -> tuple[str, str]:
    for file in dist.files or []:
        if LICENCE_FILE.search(str(file).replace("\\", "/")):
            try:
                text = Path(dist.locate_file(file)).read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - unreadable file inside a wheel
                continue
            classified = _classify_text(text)
            if classified:
                return classified, str(file)
    return "", ""


def _allowed(licence: str) -> bool:
    low = licence.strip().lower()
    if not low:
        return False
    if FORBIDDEN.search(low):
        return False
    return any(low.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def audit() -> tuple[list[Entry], list[str]]:
    entries: list[Entry] = []
    problems: list[str] = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = (dist.metadata["Name"] or "?").strip()
        version = (dist.version or "?").strip()
        licence = _from_metadata(dist)
        source = "metadata"
        note = ""
        if not licence or licence.upper() == "UNKNOWN":
            licence, file_note = _from_licence_file(dist)
            source = "licence-file" if licence else "unknown"
            note = f"read from {file_note}" if file_note else ""
        if (not licence or licence.upper() == "UNKNOWN") and name.lower() in OVERRIDES:
            licence, note = OVERRIDES[name.lower()]
            source = "override"
        allowed = _allowed(licence)
        entries.append(Entry(name, version, licence or "UNKNOWN", source, allowed, note))
        if allowed or not licence:
            continue
        problems.append(f"{name} {version}: forbidden licence {licence!r}")
    unknown = [e for e in entries if e.source == "unknown"]
    return entries, problems + [
        f"{e.name} {e.version}: licence could not be identified" for e in unknown
    ]


def repo_posture(root: Path) -> list[str]:
    problems: list[str] = []
    license_path = root / "LICENSE"
    if not license_path.is_file():
        problems.append("LICENSE is missing")
    else:
        text = license_path.read_text(encoding="utf-8", errors="ignore")[:400]
        if not text.lstrip().lower().startswith("mit license"):
            problems.append("LICENSE does not start with the MIT text")
    requirements = root / "docs" / "requirements.md"
    if requirements.is_file():
        body = requirements.read_text(encoding="utf-8", errors="ignore")
        if "§13.3" not in body and "13.3" not in body:
            problems.append(
                "docs/requirements.md no longer documents the licence allowlist (§13.3)"
            )
    src = root / "src"
    if src.is_dir():
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for trap in TRAPS:
                if re.search(rf"^\s*(?:from|import)\s+{trap}\b", text, re.MULTILINE):
                    problems.append(f"{path.relative_to(root)} imports vendored code from {trap}")
    return problems


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=None, help="repository root (default: parent of scripts/)"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    entries, licence_problems = audit()
    posture_problems = repo_posture(root)

    if args.json:
        print(
            json.dumps(
                {
                    "checked": len(entries),
                    "problems": licence_problems + posture_problems,
                    "entries": [e.__dict__ for e in entries],
                },
                indent=2,
            )
        )
    else:
        if not args.quiet:
            print(f"licences: {len(entries)} installed distribution(s) audited")
            for entry in entries:
                flag = "ok " if entry.allowed else "FAIL"
                note = f"  ({entry.note})" if entry.note else ""
                print(
                    f"  {flag} {entry.name} {entry.version}: {entry.licence} [{entry.source}]{note}"
                )
        for problem in licence_problems + posture_problems:
            print(f"FAIL: {problem}")
        if not licence_problems and not posture_problems:
            print(
                "\nOK: every installed distribution is permissive; "
                "repository licence posture intact."
            )
    if posture_problems:
        return 2
    return 0 if not licence_problems else 1


if __name__ == "__main__":
    sys.exit(main())
