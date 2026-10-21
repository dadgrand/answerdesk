"""Canonical, line-ending-independent provenance hashing.

A digest published in ``docs/results.md`` must mean the same thing on every machine. Raw-byte
hashing does not: a Windows working tree checks out CRLF while a fresh Linux/macOS clone (and
``git archive``) yields LF, so the same committed file produced two different digests and the
reproducibility claim was false on a clean checkout.

Every provenance input is therefore hashed over **canonical LF bytes**: ``\\r\\n`` and lone
``\\r`` are normalised to ``\\n`` before hashing. The normalisation is intentionally limited to
line endings — content bytes, encodings and trailing newlines are untouched, so the digest still
changes when the file changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["canonical_bytes", "canonical_sha256", "canonical_sha256_bytes", "raw_sha256"]


def canonical_bytes(data: bytes) -> bytes:
    """Return ``data`` with every line ending normalised to a single ``\\n``."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def canonical_sha256(path: Path) -> str:
    """Digest of one file's canonical LF bytes; ``"missing"`` when the file is absent."""
    if not path.is_file():
        return "missing"
    return canonical_sha256_bytes(path.read_bytes())


def raw_sha256(path: Path) -> str:
    """Digest of the file exactly as stored — for tests that pin the difference."""
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()
