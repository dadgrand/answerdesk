"""Fixtures for the ingestion unit tests (fixtures are generated, never committed).

Every fixture here is produced in-process: a hand-written PDF (no reportlab), a DOCX via
``python-docx``, an HTML string and a Markdown string. That keeps the ingestion suite
offline and dependency-light while still exercising five real parsers.
"""

from __future__ import annotations

import io
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest


def build_pdf(lines: list[tuple[float, str]]) -> bytes:
    """Minimal valid single-page PDF with one text line per ``(font_size, text)``."""
    content_parts = ["BT"]
    y = 720.0
    for size, text in lines:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_parts.append(f"/F1 {size} Tf 72 {y:.1f} Td ({escaped}) Tj")
        y -= size + 8
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1")
    compressed = zlib.compress(stream)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(compressed)
        + compressed
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)


def build_docx() -> bytes:
    """A DOCX with two heading levels, body text, a numbered list and a table."""
    from docx import Document as DocxDocument

    document = DocxDocument()
    document.add_heading("Tariffs and fees", level=1)
    document.add_paragraph("The basic plan costs 50 RUB per month.")
    document.add_heading("Cash withdrawal", level=2)
    document.add_paragraph("The daily limit is 25 000 RUB.")
    document.add_paragraph("Open the app", style="List Number")
    document.add_paragraph("Choose a plan", style="List Number")
    table = document.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Indicator"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Monthly fee"
    table.cell(1, 1).text = "50 RUB"
    table.cell(2, 0).text = "Limit"
    table.cell(2, 1).text = "25 000 RUB"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


HTML_DOCUMENT = """<!doctype html>
<html lang="ru">
  <head><title>Возвраты и комиссии</title></head>
  <body>
    <h1>Возвраты и комиссии</h1>
    <p>Возврат обрабатывается за 14 дней.</p>
    <h2>Комиссии</h2>
    <p>Комиссия за перевод — 2,5 EUR.</p>
    <table>
      <tr><th>Показатель</th><th>Значение</th></tr>
      <tr><td>Комиссия</td><td>2,5 EUR</td></tr>
      <tr><td>Срок</td><td>14 дней</td></tr>
    </table>
    <ul><li>Проверьте реквизиты</li><li>Подтвердите операцию</li></ul>
    <dl>
      <dt>Сколько идёт возврат?</dt>
      <dd>Возврат идёт до 14 дней.</dd>
    </dl>
    <blockquote>Не сообщайте коды из SMS.</blockquote>
  </body>
</html>
"""

MARKDOWN_DOCUMENT = """---
doc_id: "md-fixture"
title: "Refund policy: cards and transfers"
language: "ru"
license: "CC0-1.0"
---

# Политика возвратов

Возврат обрабатывается за 14 дней с момента одобрения.

## Комиссии

Комиссия за перевод — 2,5 EUR. Лимит — 25 000 RUB.

| Показатель | Значение | Срок |
| --- | --- | --- |
| Комиссия | 2,5 EUR | 14 дней |
| Лимит | 25 000 RUB | сутки |

## Порядок действий

1. Проверьте реквизиты.
2. Подтвердите операцию.

## Вопросы

Q: Сколько идёт возврат?
A: Возврат идёт до 14 дней.

Q: Какая комиссия за перевод?
A: Комиссия за перевод — 2,5 EUR.
"""


@pytest.fixture()
def markdown_text() -> str:
    return MARKDOWN_DOCUMENT


@pytest.fixture()
def html_text() -> str:
    return HTML_DOCUMENT


@pytest.fixture()
def docx_bytes() -> bytes:
    return build_docx()


@pytest.fixture()
def make_pdf() -> Callable[[list[tuple[float, str]]], bytes]:
    return build_pdf


@pytest.fixture()
def write_bytes(tmp_path: Path) -> Callable[[str, bytes], Path]:
    def _write(name: str, data: bytes) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    return _write
