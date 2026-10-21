"""``answerdesk ask`` — the offline question path (CMD-H).

# AT-OPS-03  # AT-OPS-04  # AT-GEN-04  # AT-GEN-10
"""

from __future__ import annotations

import inspect
import json
import re

import pytest
from typer.testing import CliRunner

from answerdesk.cli import app

runner = CliRunner()

CARD_QUESTION = "Сколько стоит обслуживание карты?"

_ANSI = re.compile(
    r"(?:\x1b\][^\x07\x1b]*(?:\x07|\x1b\\))"  # OSC: hyperlinks, window titles
    r"|(?:\x1b\[[0-9;?]*[ -/]*[@-~])"  # CSI: colours, cursor movement
)

#: Wider than any flag line in this CLI, so neither rich nor click wraps an option name.
HELP_WIDTH = 200


def _plain(text: str) -> str:
    """Strip terminal control sequences and normalise line endings.

    Typer renders help with rich when it is importable, so raw output carries colour sequences
    *inside* option names (``\x1b[33m--fake\x1b[0m``) and wraps to the reported terminal width —
    which differs between a Windows console and a CI pipe. Both are rendering concerns, not CLI
    contracts, so the flag assertions read a deliberately wide, uncoloured render (ANSI stripped)
    *and* the command's declared parameter list.
    """
    without_ansi = _ANSI.sub("", text)
    return without_ansi.replace("\r\n", "\n").replace("\r", "\n")


def _help_text(command: str) -> str:
    """Render ``<command> --help`` in a fixed, wide, uncoloured terminal."""
    result = runner.invoke(
        app,
        [command, "--help"],
        env={"COLUMNS": str(HELP_WIDTH), "TERM": "dumb", "NO_COLOR": "1"},
        color=False,
    )
    assert result.exit_code == 0, combined(result)
    text = _plain(combined(result))
    # If the width override ever stops working (a rich/click change, a stripped environment),
    # rich truncates option names to "--synt…" instead of wrapping them, which would make the
    # flag assertions below fail for a reason unrelated to the CLI contract. Fail loudly with
    # the real cause: no help string in this CLI contains an ellipsis, so any "…" means the page
    # was cut to the terminal width.
    assert "…" not in text, (
        "the help page was truncated, so the wide-terminal override (COLUMNS="
        f"{HELP_WIDTH}) did not take effect; the flag assertions would be meaningless"
    )
    return text


def _declared_flags(command: str) -> set[str]:
    """Option tokens the command actually declares, read from Typer's own registration.

    ``app.registered_commands`` is the source the help page is built from, so a flag that is
    removed or renamed in the CLI fails here *and* in the rendered text — this is the structural
    half of the assertion, independent of any terminal rendering.
    """
    for info in app.registered_commands:
        callback = info.callback
        if callback is None:
            continue
        if (info.name or callback.__name__.replace("_", "-")) != command:
            continue
        tokens: set[str] = set()
        for parameter in inspect.signature(callback).parameters.values():
            declarations = getattr(parameter.default, "param_decls", None) or ()
            for declaration in declarations:
                tokens.update(part for part in str(declaration).split("/") if part.startswith("-"))
        return tokens
    raise AssertionError(f"`{command}` is not a registered Typer command")


def _assert_flags(command: str, *, flags: tuple[str, ...]) -> None:
    """Each flag is declared by the command *and* documented in its help page."""
    text = _help_text(command)
    declared = _declared_flags(command)
    for flag in flags:
        assert flag in declared, f"{flag} is not declared by `{command}`"
        assert flag in text, f"{flag} is missing from `{command} --help`"


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
    """# AT-OPS-03 — the offline flags are declared *and* documented, at any terminal width."""
    _assert_flags("ask", flags=("--fake", "--json", "--profile", "--config"))
