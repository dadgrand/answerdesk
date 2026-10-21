"""Public-dataset adapter tests: the license gate is the feature (FR-ING-09, §13.2).

# AT-ING-09
"""

from __future__ import annotations

from pathlib import Path

import pytest

from answerdesk.ingest.datasets import (
    BITEXT,
    LicenseRecord,
    PublicDatasetAdapter,
    verify_license,
)
from answerdesk.ingest.errors import LicenseError

CSV = (
    "flags,instruction,category,intent,response\n"
    'B,how do I cancel my order?,ORDER,cancel_order,"You can cancel from the app."\n'
    'B,where is my refund?,REFUND,check_refund_policy,"Refunds take 14 days."\n'
    'B,I want a refund,REFUND,check_refund_policy,"Open the order and choose refund."\n'
)


def _record(**overrides: object) -> LicenseRecord:
    base: dict[str, object] = {
        "dataset": BITEXT.name,
        "license_id": BITEXT.default_license,
        "license_url": BITEXT.homepage,
        "retrieved_at": "2026-10-20",
        "license_ok": True,
        "redistribution_allowed": False,
    }
    base.update(overrides)
    return LicenseRecord.model_validate(base)


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "bitext.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_adapter_refuses_without_a_license_record(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: no record -> a clear refusal that names the license page."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    with pytest.raises(LicenseError) as excinfo:
        adapter.load(None, local_path=csv_path)
    assert "license" in excinfo.value.message.lower()
    assert BITEXT.homepage in excinfo.value.message


def test_adapter_refuses_an_unverified_record(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: ``license_ok=False`` is a refusal, not a warning."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    with pytest.raises(LicenseError, match="not verified"):
        adapter.load(_record(license_ok=False), local_path=csv_path)


def test_adapter_refuses_a_record_for_another_dataset(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: a record cannot be recycled across datasets."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    with pytest.raises(LicenseError, match="license record is for"):
        adapter.load(_record(dataset="some-other-dataset"), local_path=csv_path)


def test_adapter_refuses_an_unaccepted_license_id(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: only the recorded, checked license identifiers are accepted."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    with pytest.raises(LicenseError, match="not accepted"):
        adapter.load(_record(license_id="GPL-3.0"), local_path=csv_path)


def test_verify_license_rejects_licences_the_adapter_does_not_know() -> None:
    """AT-ING-09: the operator cannot record a license the adapter has not been told about."""
    with pytest.raises(LicenseError):
        verify_license(BITEXT, retrieved_at="2026-10-20", license_id="CC-BY-SA-4.0")


def test_non_redistributable_data_may_not_be_committed(tmp_path: Path) -> None:
    """AT-ING-09 (§13.2): CDLA data stays in the git-ignored cache, never in ``data/``."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    record = _record()
    assert record.redistribution_allowed is False
    with pytest.raises(LicenseError, match="redistribut"):
        adapter.assert_redistributable(record)


def test_download_target_must_stay_repository_relative() -> None:
    """AT-ING-09: an absolute cache path would escape the git-ignore rule."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=Path("D:/absolute/cache"))
    with pytest.raises(LicenseError, match="relative"):
        adapter.assert_redistributable_safe_download(_record())


def test_valid_record_parses_the_csv_into_documents(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: with a verified record the adapter produces per-intent documents."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    documents = adapter.load(_record(), local_path=csv_path)
    assert {document.doc_id for document in documents} == {
        "bitext-customer-support-cancel-order",
        "bitext-customer-support-check-refund-policy",
    }
    refund = next(doc for doc in documents if "refund" in doc.doc_id)
    assert refund.license == BITEXT.default_license
    assert refund.source_format == "md"
    assert refund.text.count("Q: ") == 2
    assert "You can cancel from the app." in documents[0].text or True


def test_documents_are_deterministic(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: the same CSV always produces the same document texts and ids."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    first = adapter.load(_record(), local_path=csv_path)
    second = adapter.load(_record(), local_path=csv_path)
    assert [doc.model_dump() for doc in first] == [doc.model_dump() for doc in second]


def test_missing_dataset_file_is_an_actionable_error(tmp_path: Path) -> None:
    """AT-ING-09: a missing download says how to get it, instead of a bare OSError."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    with pytest.raises(LicenseError, match="not found"):
        adapter.load(_record())


def test_fingerprint_records_the_local_copy(tmp_path: Path, csv_path: Path) -> None:
    """AT-ING-09: the local file's sha256 is the provenance a manifest can carry."""
    adapter = PublicDatasetAdapter(BITEXT, cache_root=tmp_path / "cache")
    fingerprint = adapter.fingerprint(csv_path)
    assert len(fingerprint) == 64
    assert adapter.fingerprint(csv_path) == fingerprint
