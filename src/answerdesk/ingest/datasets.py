"""License-gated adapter for public customer-support datasets (FR-ING-09, §13.2).

The rule this module exists to enforce: **no license record, no ingest.** A public
dataset may only be used when its license was checked and recorded (identifier, URL,
retrieval date, and whether redistribution is permitted). Data whose license does not
allow redistribution is downloaded at runtime into a git-ignored directory and is never
copied into the repository's ``data/corpus``.

The reference adapter targets the Bitext customer-support family
(``bitext/Bitext-customer-support-llm-chatbot-training-dataset`` on Hugging Face, CSV
columns ``flags, instruction, category, intent, response``). Its published license is
**CDLA-Sharing-1.0**, which permits use with conditions but is *not* a permissive
redistribution license for this repository, so the adapter's default record for it has
``redistribution_allowed=False`` — the CSV stays in ``data/external/bitext/`` (ignored by
git) and only derived, in-memory documents are produced.

Because a checked-in license record could silently go stale, the record is *supplied by
the operator* (:func:`verify_license`) rather than hard-coded: the adapter knows which
license identifiers it accepts, and refuses everything else.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from answerdesk.contracts import Document
from answerdesk.ingest.errors import LicenseError
from answerdesk.ingest.loaders import load_markdown_text
from answerdesk.ingest.structure import IngestModel


class LicenseRecord(IngestModel):
    """The proof that a dataset may be used, recorded in the corpus manifest."""

    dataset: str
    license_id: str
    license_url: str
    retrieved_at: str
    license_ok: bool = False
    redistribution_allowed: bool = False
    attribution: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the adapter needs to know about one public dataset."""

    name: str
    homepage: str
    download_url: str
    filename: str
    accepted_licenses: tuple[str, ...]
    default_license: str
    language: str
    license_note: str

    @property
    def cache_dir(self) -> str:
        return f"data/external/{self.name}"


#: The Bitext customer-support dataset (Hugging Face). See ``docs/ingestion.md`` §7.
BITEXT: DatasetSpec = DatasetSpec(
    name="bitext-customer-support",
    homepage=(
        "https://huggingface.co/datasets/bitext/"
        "Bitext-customer-support-llm-chatbot-training-dataset"
    ),
    download_url=(
        "https://huggingface.co/datasets/bitext/"
        "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    ),
    filename="Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    accepted_licenses=("CDLA-Sharing-1.0",),
    default_license="CDLA-Sharing-1.0",
    language="en",
    license_note=(
        "CDLA-Sharing-1.0: use is permitted with conditions; redistribution inside this "
        "repository is not assumed. The CSV is therefore downloaded at runtime into "
        "data/external/bitext/ (git-ignored) and is never committed."
    ),
)


def verify_license(
    spec: DatasetSpec,
    *,
    retrieved_at: str,
    redistribution_allowed: bool = False,
    license_id: str | None = None,
    attribution: str | None = None,
    notes: str | None = None,
) -> LicenseRecord:
    """Record an operator-verified license. Call this *after* reading the license page."""
    resolved = license_id or spec.default_license
    if resolved not in spec.accepted_licenses:
        known = ", ".join(spec.accepted_licenses)
        raise LicenseError(
            f"license {resolved!r} is not accepted for {spec.name}; accepted: {known}"
        )
    return LicenseRecord(
        dataset=spec.name,
        license_id=resolved,
        license_url=spec.homepage,
        retrieved_at=retrieved_at,
        license_ok=True,
        redistribution_allowed=redistribution_allowed and resolved == spec.default_license,
        attribution=attribution,
        notes=notes or spec.license_note,
    )


class PublicDatasetAdapter:
    """Turns a license-recorded public CSV into documents, without leaking the raw data."""

    def __init__(
        self,
        spec: DatasetSpec,
        *,
        cache_root: str | Path = "data/external",
    ) -> None:
        self.spec = spec
        self.cache_root = Path(cache_root)

    # -- license gate ---------------------------------------------------------------
    def require_license(self, record: LicenseRecord | None) -> LicenseRecord:
        """Refuse to run without a matching, verified, explicitly recorded license."""
        if record is None:
            raise LicenseError(
                f"no license record for {self.spec.name}: refusing to ingest. Verify the "
                f"license at {self.spec.homepage} and pass LicenseRecord(license_ok=True, "
                "license_id=..., license_url=..., retrieved_at=...)"
            )
        if not record.license_ok:
            raise LicenseError(
                f"license record for {self.spec.name} is not verified (license_ok=False); "
                "usage is refused until a human confirms the license"
            )
        if record.dataset != self.spec.name:
            raise LicenseError(f"license record is for {record.dataset!r}, not {self.spec.name!r}")
        if record.license_id not in self.spec.accepted_licenses:
            known = ", ".join(self.spec.accepted_licenses)
            raise LicenseError(
                f"license {record.license_id!r} is not accepted for {self.spec.name}; "
                f"accepted: {known}"
            )
        if not record.license_url.startswith("http"):
            raise LicenseError("a license record must carry the URL it was verified at")
        return record

    def assert_redistributable(self, record: LicenseRecord) -> None:
        """Raise unless the license allows committing the data to this repository."""
        if not record.redistribution_allowed:
            raise LicenseError(
                f"{self.spec.name} ({record.license_id}) may not be redistributed inside "
                "this repository: keep the download in the git-ignored cache and commit "
                "only derived statistics, never the raw rows",
                detail=self.spec.license_note,
            )

    # -- data access ----------------------------------------------------------------
    def cache_path(self) -> Path:
        return self.cache_root / self.spec.name / self.spec.filename

    def download(self, record: LicenseRecord | None) -> Path:
        """Download the CSV into the git-ignored cache (network; never in tests)."""
        checked = self.require_license(record)
        self.assert_redistributable_safe_download(checked)
        target = self.cache_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.spec.download_url, timeout=60) as response:
            payload = response.read()
        target.write_bytes(payload)
        return target

    def assert_redistributable_safe_download(self, record: LicenseRecord) -> None:
        """A download is always allowed *into the cache*; only committing is restricted."""
        if self.cache_root.is_absolute():
            raise LicenseError(
                "the dataset cache must be a repository-relative path so it stays "
                "git-ignored (got an absolute path)"
            )
        return None

    def load(
        self,
        record: LicenseRecord | None,
        *,
        local_path: str | Path | None = None,
        max_documents: int | None = None,
    ) -> list[Document]:
        """Parse the CSV into per-intent documents (in memory, deterministic order)."""
        checked = self.require_license(record)
        path = Path(local_path) if local_path is not None else self.cache_path()
        if not path.is_file():
            raise LicenseError(
                f"dataset file not found at {path}; run `answerdesk ingest --bitext` once "
                "with network access, or pass --path to a local copy"
            )
        return self._documents_from_csv(path, checked, max_documents=max_documents)

    # -- conversion -----------------------------------------------------------------
    def _documents_from_csv(
        self, path: Path, record: LicenseRecord, *, max_documents: int | None = None
    ) -> list[Document]:
        grouped: dict[str, list[tuple[str, str]]] = {}
        text = path.read_bytes().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            intent = (row.get("intent") or row.get("category") or "general").strip() or "general"
            instruction = (row.get("instruction") or "").strip()
            response = (row.get("response") or "").strip()
            if not instruction or not response:
                continue
            grouped.setdefault(intent, []).append((instruction, response))
        documents: list[Document] = []
        for intent in sorted(grouped):
            pairs = grouped[intent]
            doc_id = f"{self.spec.name}-{_slug(intent)}"
            body = [f"# {intent.replace('_', ' ').title()}", ""]
            for instruction, response in pairs:
                body.extend([f"Q: {instruction}", f"A: {response}", ""])
            markdown = self._front_matter(doc_id, intent, record) + "\n".join(body)
            documents.append(
                load_markdown_text(
                    markdown,
                    source_path=f"{self.spec.cache_dir}/{doc_id}.md",
                    doc_id=doc_id,
                    license=record.license_id,
                    language=self.spec.language,
                    title=f"{self.spec.name}: {intent}",
                )
            )
            if max_documents is not None and len(documents) >= max_documents:
                break
        return documents

    def _front_matter(self, doc_id: str, intent: str, record: LicenseRecord) -> str:
        return (
            "---\n"
            f'doc_id: "{doc_id}"\n'
            f'title: "{self.spec.name}: {intent}"\n'
            f'language: "{self.spec.language}"\n'
            f'license: "{record.license_id}"\n'
            f'license_url: "{record.license_url}"\n'
            f'retrieved_at: "{record.retrieved_at}"\n'
            'generator: "bitext-adapter-v1"\n'
            "---\n\n"
        )

    def fingerprint(self, path: str | Path) -> str:
        """sha256 of a local dataset file, recorded in the manifest as provenance."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", value.lower()).strip("-") or "general"


__all__ = [
    "BITEXT",
    "DatasetSpec",
    "LicenseRecord",
    "PublicDatasetAdapter",
    "verify_license",
]
