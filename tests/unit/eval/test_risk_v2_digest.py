"""R-34/R-03: the risk-v2 provenance digests are line-ending independent.

The defect this file pins: ``feature_digest`` hashed raw bytes, so a committed
``reports/risk-v2/features.csv`` produced ``93cf38…`` on a CRLF Windows checkout and
``62bc8d…`` on an LF ``git archive``; ``scripts/risk-v2.py --check`` therefore failed on the
clean checkout the release is validated on while passing on the developer tree.

The guarantee now: the digest is computed over canonical LF bytes, so the same *content*
hashes the same way whichever line endings the checkout happened to use.

    # AT-REP-02  # AT-DET-02
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from answerdesk.eval.risk.features import canonical_bytes, feature_digest
from answerdesk.eval.risk.runner import check

REPO_ROOT = Path(__file__).resolve().parents[3]
RISK_DIR = REPO_ROOT / "reports" / "risk-v2"


def test_crlf_lf_and_mixed_twins_share_one_digest(tmp_path: Path) -> None:
    lf = tmp_path / "lf.csv"
    lf.write_bytes(b"fact_id,score\r\n1,0.5\n2,0.7\n")
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(b"fact_id,score\r\n1,0.5\r\n2,0.7\r\n")
    mixed = tmp_path / "mixed.csv"
    mixed.write_bytes(b"fact_id,score\r\n1,0.5\n2,0.7\r\n")

    digests = {feature_digest(path) for path in (lf, crlf, mixed)}
    assert len(digests) == 1, digests
    # and the digest is over the LF projection, not over the raw bytes
    assert feature_digest(crlf) == hashlib.sha256(b"fact_id,score\n1,0.5\n2,0.7\n").hexdigest()
    assert canonical_bytes(b"a\r\nb\rc\n") == b"a\nb\nc\n"


@pytest.mark.skipif(not (RISK_DIR / "features.csv").is_file(), reason="no committed risk-v2 run")
def test_check_passes_on_an_lf_normalised_copy(tmp_path: Path) -> None:
    """Extract the committed run, rewrite every text artefact to LF, and re-run --check.

    This is the clean-`git archive` condition: the working tree here is CRLF, the archive is
    LF, and the gate must pass on both.
    """
    copy = tmp_path / "risk-v2"
    shutil.copytree(RISK_DIR, copy)
    for name in ("features.csv", "bench-features.csv", "split.json", "summary.json", "certificate.json"):
        path = copy / name
        if path.is_file():
            path.write_bytes(canonical_bytes(path.read_bytes()))
    for name in ("frontier.csv", "reliability.csv"):
        path = copy / name
        if path.is_file():
            path.write_bytes(canonical_bytes(path.read_bytes()))

    assert check(copy) == 0


@pytest.mark.skipif(not (RISK_DIR / "summary.json").is_file(), reason="no committed risk-v2 run")
def test_committed_summary_records_the_canonical_digest() -> None:
    summary = json.loads((RISK_DIR / "summary.json").read_text(encoding="utf-8"))
    stored = summary["populations"]["consistency"]["source_digest"]
    assert stored == feature_digest(RISK_DIR / "features.csv")
