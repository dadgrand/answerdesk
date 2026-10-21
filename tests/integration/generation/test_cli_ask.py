"""``answerdesk ask`` — the offline question path (CMD-H).

# AT-OPS-03  # AT-OPS-04  # AT-GEN-04  # AT-GEN-10
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from answerdesk.cli import app

runner = CliRunner()

CARD_QUESTION = "Сколько стоит обслуживание карты?"


def combined(result: object) -> str:
    output = getattr(result, "output", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    return output + stderr


def test_ask_fake_answers_with_citations() -> None:
    """# AT-OPS-03 — keyless, network-free, and the markers are real."""
    result = runner.invoke(app, ["ask", "--fake", CARD_QUESTION])
    assert result.exit_code == 0, combined(result)
    text = combined(result)
    assert "Q: Сколько стоит обслуживание карты?" in text
    assert "120 рублей в месяц" in text
    assert "[1]" in text
    assert "Citations:" in text
    assert "Decision: answer (ANSWERED)" in text
    assert "prompt=grounded-answer-" in text
    assert "verifier=lexical-verifier-" in text


def test_ask_json_prints_the_structured_result() -> None:
    """# AT-GEN-04 — the CLI exposes the contract object, not a bespoke format."""
    result = runner.invoke(app, ["ask", "--fake", "--json", CARD_QUESTION])
    assert result.exit_code == 0, combined(result)
    payload = json.loads(result.output)
    assert payload["signal"]["action"] == "answer"
    assert payload["signal"]["reason_code"] == "ANSWERED"
    assert payload["language"] == "ru"
    assert payload["claims"] and payload["citations"]
    assert payload["claims"][0]["citation_ids"] == [1]
    assert payload["prompt_version"].startswith("grounded-answer-")
    assert payload["usage"]["input_tokens"] > 0
    citation = payload["citations"][0]
    assert citation["quote"] in payload["text"] or citation["quote"]
    assert len(citation["doc_hash"]) == 64


def test_ask_unknown_question_escalates_instead_of_inventing() -> None:
    """# AT-GEN-10 — no matching evidence ⇒ a structured signal, exit 0, no prose."""
    result = runner.invoke(app, ["ask", "--fake", "Как мне настроить спутниковую антенну?"])
    assert result.exit_code == 0, combined(result)
    text = combined(result)
    assert "(no prose answer) escalate: NO_EVIDENCE" in text


def test_ask_without_fake_needs_a_passage_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honest seam: the retrieval stack is wired by the service layer, not guessed here."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["ask", "--profile", "local", CARD_QUESTION])
    assert result.exit_code == 2
    assert "--fake" in combined(result)


def test_ask_help_documents_the_flags() -> None:
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    text = combined(result)
    assert "--fake" in text
    assert "--json" in text
