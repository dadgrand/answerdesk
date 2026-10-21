"""Typed ingestion errors (FR-ING-01/09, NFR-OPS-01 failure matrix).

Every failure mode of the ingestion layer is a subclass of
:class:`answerdesk.errors.AnswerdeskError`, so the service layer can map it onto an
HTTP envelope without inspecting message strings. The module never raises a bare
``ValueError`` for an operational failure.
"""

from __future__ import annotations

from answerdesk.errors import AnswerdeskError


class IngestError(AnswerdeskError):
    """Base class for a document that cannot be turned into a corpus entry."""

    code = "ingest_error"
    http_status = 422


class UnsupportedFormatError(IngestError):
    """The file suffix has no loader (FR-ING-01 lists the five supported formats)."""

    code = "unsupported_format"
    http_status = 415


class EmptyDocumentError(IngestError):
    """The source produced no text — an empty file, a scanned PDF, a blank DOCX.

    Raised instead of returning an empty :class:`~answerdesk.contracts.Document`,
    because an empty document silently becomes zero chunks and disappears from the
    corpus with no trace.
    """

    code = "empty_document"
    http_status = 422


class SourceNotFoundError(IngestError):
    """The configured source path does not exist or is not a file/directory."""

    code = "source_not_found"
    http_status = 404


class LicenseError(IngestError):
    """A dataset adapter was asked to run without a verified, recorded license.

    This is the enforcement point of FR-ING-09 / §13.2: no license record, no
    ingest — never a warning that gets ignored.
    """

    code = "license_not_verified"
    http_status = 403


class CorpusBuildError(IngestError):
    """The corpus could not be built (empty input, unwritable target, bad scale)."""

    code = "corpus_build_error"
    http_status = 500


__all__ = [
    "CorpusBuildError",
    "EmptyDocumentError",
    "IngestError",
    "LicenseError",
    "SourceNotFoundError",
    "UnsupportedFormatError",
]
