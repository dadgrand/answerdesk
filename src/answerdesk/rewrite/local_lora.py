"""The optional local LoRA rewriter: weights are optional, the seam is not.

Everything heavy is imported *inside* the methods, so importing this module (and therefore
`answerdesk.rewrite`) costs nothing on a base install without torch — which is the whole point
of NFR-PORT-03 and of "off by default" in the requirements. When the stack or the adapter is
missing, :attr:`LocalLoRARewriter.available` is ``False`` and the backend degrades to identity
with a recorded reason instead of raising into the request path.

The prompt is a fixed, versioned instruction (``PROMPT_VERSION``) so a change of wording is
visible in the adapter card and in the training script.

    # AT-GEN-15
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from answerdesk.rewrite.base import RewriteResult
from answerdesk.rewrite.errors import RewriterUnavailable

#: Prompt contract for the adapter (bump when the wording changes; the card records it).
PROMPT_VERSION = "rewrite-instruction-1.0.0"

SYSTEM_PROMPT = (
    "You are a query normaliser for a multilingual customer-support search engine. "
    "Rewrite the user's message into one short, clean search query in the SAME language. "
    "Keep product names, amounts, dates and identifiers exactly as written. "
    "Remove greetings, filler words and politeness. Do not answer the question. "
    "Output only the rewritten query, with no quotes, labels or explanation."
)

#: Locale names used in the user turn (keeps the model from translating).
LOCALE_NAMES: dict[str, str] = {
    "ru": "Russian",
    "uz": "Uzbek",
    "tg": "Tajik",
    "ky": "Kyrgyz",
    "zh": "Chinese",
    "tr": "Turkish",
}

#: Packages the backend needs. Listed so the error message can name them.
TRAINING_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "safetensors",
)


def _missing_packages() -> list[str]:
    import importlib.util

    return [name for name in TRAINING_PACKAGES if importlib.util.find_spec(name) is None]


class LocalLoRARewriter:
    """Greedy decoder over a PEFT adapter on a small instruction-tuned base model."""

    name = "local_lora"

    def __init__(
        self,
        *,
        adapter_path: str | Path,
        base_model: str,
        device: str = "auto",
        max_new_tokens: int = 64,
        timeout_ms: float | None = 2000.0,
        batch_size: int = 8,
        max_prompt_tokens: int = 512,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.timeout_ms = timeout_ms
        self.batch_size = max(1, batch_size)
        self.max_prompt_tokens = max_prompt_tokens
        self.prompt_version = prompt_version
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_error: str | None = None
        self._loaded = False

    # -- availability -------------------------------------------------------------
    @property
    def missing(self) -> list[str]:
        """Packages and artefacts that are absent (an empty directory is not an adapter)."""
        missing = _missing_packages()
        if not (self.adapter_path / "adapter_config.json").is_file():
            missing.append(f"adapter:{self.adapter_path}")
        return missing

    @property
    def available(self) -> bool:
        if self._loaded:
            return self._load_error is None
        return not self.missing

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # -- loading ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        import torch

        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Import the stack, load the base model and attach the adapter (idempotent)."""
        if self._loaded:
            return
        missing = _missing_packages()
        if missing:
            raise RewriterUnavailable(
                "the local rewriter needs the `train` extra (torch/transformers/peft) and an "
                "adapter directory; install with `uv sync --extra train`",
                missing=missing,
            )
        if not (self.adapter_path / "adapter_config.json").is_file():
            raise RewriterUnavailable(
                f"adapter not found (no adapter_config.json): {self.adapter_path} — train it with "
                "`python scripts/train_query_rewriter.py --train --eval`",
                missing=[str(self.adapter_path)],
            )

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, str(self.adapter_path))
            model.eval()
        except Exception as exc:  # a broken adapter must not break the request path
            self._loaded = True
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise RewriterUnavailable(
                f"cannot load the rewriter adapter at {self.adapter_path}: {exc}",
                missing=[str(self.adapter_path)],
            ) from exc
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True
        self._load_error = None

    # -- inference ----------------------------------------------------------------
    def build_messages(self, query: str, locale: str | None) -> list[dict[str, str]]:
        language = LOCALE_NAMES.get(locale or "", "the customer's language")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Language: {language}\nMessage: {query.strip()}\nRewritten query:",
            },
        ]

    def rewrite(self, query: str, *, locale: str | None = None) -> RewriteResult:
        started = time.perf_counter()
        if not self._loaded:
            try:
                self.load()
            except RewriterUnavailable as exc:
                raise RewriterUnavailable(exc.message, missing=exc.missing) from exc

        import torch

        assert self._model is not None and self._tokenizer is not None
        messages = self.build_messages(query, locale)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip().strip('"').strip("«»").strip()
        if not text:
            text = query
        elapsed = (time.perf_counter() - started) * 1000.0
        return RewriteResult(
            original=query,
            query=text,
            backend=self.name,
            available=True,
            latency_ms=round(elapsed, 3),
            notes=f"adapter={self.adapter_path.name}; prompt={self.prompt_version}",
        )

    # -- batched inference (evaluation / offline batch jobs) -----------------------
    def rewrite_many(
        self, queries: list[str], *, locales: list[str | None] | None = None
    ) -> list[RewriteResult]:
        """Rewrite a batch of queries with one model call per ``batch_size`` chunk.

        Serving uses :meth:`rewrite` (one request, one rewrite); this exists so the evaluation
        harness can measure quality without paying single-query latency for every case.
        """
        if not queries:
            return []
        if not self._loaded:
            self.load()
        import torch

        assert self._model is not None and self._tokenizer is not None
        tokenizer = self._tokenizer
        model = self._model
        locales = locales or [None] * len(queries)
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        results: list[RewriteResult] = []
        try:
            for start in range(0, len(queries), self.batch_size):
                chunk = queries[start : start + self.batch_size]
                chunk_locales = locales[start : start + self.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        self.build_messages(query, locale),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for query, locale in zip(chunk, chunk_locales, strict=True)
                ]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_tokens,
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                completions = generated[:, encoded["input_ids"].shape[1] :]
                for query, text in zip(
                    chunk,
                    tokenizer.batch_decode(completions, skip_special_tokens=True),
                    strict=True,
                ):
                    cleaned = text.strip().split("\n")[0].strip().strip('"').strip("«»").strip()
                    results.append(
                        RewriteResult(
                            original=query,
                            query=cleaned or query,
                            backend=self.name,
                            available=True,
                            notes=f"batched; adapter={self.adapter_path.name}",
                        )
                    )
        finally:
            tokenizer.padding_side = previous_side
        return results


__all__ = [
    "LOCALE_NAMES",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "TRAINING_PACKAGES",
    "LocalLoRARewriter",
]
