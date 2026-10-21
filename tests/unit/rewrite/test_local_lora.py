"""The LoRA backend without the stack: availability, typing, prompt and graceful failure.

These tests must pass on a base install (no torch), which is the point of the seam.

    # AT-GEN-15
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from answerdesk.rewrite.base import QueryRewriter
from answerdesk.rewrite.errors import RewriterUnavailable
from answerdesk.rewrite.local_lora import (
    LOCALE_NAMES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TRAINING_PACKAGES,
    LocalLoRARewriter,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_the_package_imports_without_torch() -> None:
    """NFR-PORT-03: importing answerdesk.rewrite must not pull in the training stack."""
    code = (
        "import sys; import answerdesk.rewrite; "
        "print('torch' in sys.modules, 'transformers' in sys.modules, 'peft' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False False", out.stdout


def test_availability_reflects_the_environment(tmp_path: Path) -> None:
    rewriter = LocalLoRARewriter(adapter_path=tmp_path / "adapter", base_model="Qwen/Qwen2.5-1.5B")
    assert isinstance(rewriter, QueryRewriter)
    missing = rewriter.missing
    assert any(entry.startswith("adapter:") for entry in missing)
    if not HAS_TORCH:
        assert {"torch", "transformers", "peft"} <= set(missing)
        assert rewriter.available is False


def test_load_raises_a_typed_error_when_the_adapter_is_absent(tmp_path: Path) -> None:
    rewriter = LocalLoRARewriter(adapter_path=tmp_path / "absent", base_model="Qwen/Qwen2.5-1.5B")
    with pytest.raises(RewriterUnavailable) as excinfo:
        rewriter.load()
    assert "adapter" in excinfo.value.message.lower()
    assert excinfo.value.missing


def test_the_error_names_the_install_command(tmp_path: Path) -> None:
    rewriter = LocalLoRARewriter(adapter_path=tmp_path / "absent", base_model="Qwen/Qwen2.5-1.5B")
    with pytest.raises(RewriterUnavailable, match="train"):
        rewriter.load()


def test_prompt_contract_is_versioned_and_localised() -> None:
    assert PROMPT_VERSION.startswith("rewrite-instruction-")
    assert "rewrite" in SYSTEM_PROMPT.lower()
    assert "Output only" in SYSTEM_PROMPT
    assert set(LOCALE_NAMES) == {"ru", "uz", "tg", "ky", "zh", "tr"}
    rewriter = LocalLoRARewriter(adapter_path=Path("x"), base_model="m")
    messages = rewriter.build_messages("zaym kak oformit", "ru")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Russian" in messages[1]["content"]
    assert "zaym kak oformit" in messages[1]["content"]
    unknown = rewriter.build_messages("hola", None)
    assert "customer's language" in unknown[1]["content"]


def test_defaults_are_conservative() -> None:
    rewriter = LocalLoRARewriter(adapter_path=Path("x"), base_model="m")
    assert rewriter.max_new_tokens == 64
    assert rewriter.timeout_ms == 2000.0
    assert rewriter.device == "auto"
    assert TRAINING_PACKAGES[:3] == ("torch", "transformers", "peft")
