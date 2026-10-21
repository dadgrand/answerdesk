"""The structured-output schema for a grounded answer draft.

The schema is a checked-in JSON Schema file, not a Python dict, because it is the
contract with the *model*, not with the code: it is sent verbatim in
``response_format.json_schema`` (providers.py) and it must be valid for strict mode.

Strict mode has two hard requirements, both verified here rather than trusted
(``assert_strict_schema``): every object sets ``additionalProperties: false`` and lists
every property in ``required``. A schema that fails this is rejected at load time, so a
provider cannot silently fall back to loose JSON.

    # AT-GEN-01  # AT-GEN-04
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from answerdesk.generate.errors import GenerationError

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Checked-in schema path (a code constant: no new gated config key).
ANSWER_SCHEMA_RELATIVE_PATH = Path("configs") / "prompts" / "schemas" / "answer_draft.schema.json"

#: Stable name handed to the provider's json_schema block.
ANSWER_SCHEMA_NAME = "answerdesk_answer_draft"


def default_answer_schema_path() -> Path:
    return _REPO_ROOT / ANSWER_SCHEMA_RELATIVE_PATH


def load_answer_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Load the answer-draft JSON Schema and reject anything that is not strict."""
    file_path = Path(path) if path is not None else default_answer_schema_path()
    try:
        schema = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GenerationError(f"cannot read answer schema {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GenerationError(f"answer schema {file_path} is not valid JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise GenerationError(f"answer schema {file_path} must be a JSON object")
    assert_strict_schema(schema)
    return schema


def schema_digest(schema: dict[str, Any]) -> str:
    """Content hash of the schema, so the trace can say which contract the model saw."""
    payload = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schema_name(schema: dict[str, Any]) -> str:
    """A stable, machine-safe name for the json_schema block."""
    raw = str(schema.get("title", ANSWER_SCHEMA_NAME))
    cleaned = "".join(char if char.isalnum() else "_" for char in raw).strip("_")
    return cleaned or ANSWER_SCHEMA_NAME


def strict_violations(schema: dict[str, Any]) -> list[str]:
    """Every place where ``schema`` would fail OpenAI-style strict json_schema mode."""
    problems: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append(f"{path or '$'}: additionalProperties must be false")
            properties = node.get("properties")
            if isinstance(properties, dict):
                required = node.get("required")
                if not isinstance(required, list):
                    problems.append(f"{path or '$'}: every property must be listed in required")
                else:
                    missing = sorted(set(properties) - set(required))
                    if missing:
                        problems.append(f"{path or '$'}: properties not required: {missing}")
        for key, value in node.items():
            if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
                for name, child in value.items():
                    walk(child, f"{path}.{key}.{name}")
            elif key in {
                "items",
                "additionalProperties",
                "anyOf",
                "oneOf",
                "allOf",
                "not",
            } or isinstance(value, dict):
                walk(value, f"{path}.{key}")

    walk(schema, "")
    return problems


def assert_strict_schema(schema: dict[str, Any]) -> None:
    """Raise ``GenerationError`` when the schema is not strict-mode compatible."""
    problems = strict_violations(schema)
    if problems:
        raise GenerationError(
            "answer schema is not strict json_schema compatible: " + "; ".join(problems)
        )


__all__ = [
    "ANSWER_SCHEMA_NAME",
    "ANSWER_SCHEMA_RELATIVE_PATH",
    "assert_strict_schema",
    "default_answer_schema_path",
    "load_answer_schema",
    "schema_digest",
    "schema_name",
    "strict_violations",
]
