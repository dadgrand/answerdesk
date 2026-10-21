#!/usr/bin/env python
"""Train, validate and evaluate the local query rewriter (t26).

Four modes, all reproducible from a clean checkout:

``--build-data``
    Rebuild ``data/training/{pairs.jsonl,splits.json,stats.json}`` from the frozen corpus
    (corpus v2.1 ``queries.jsonl``), then validate schema, dedup, splits and PII.

``--dry-run``
    Validate the data and the configuration and print the exact training plan (model, dtype,
    quantisation, steps, VRAM estimate) **without importing torch**. This is the command the
    repository's verification runs, so it must work on a base install.

``--check``
    Probe the training environment: torch/CUDA/GPU, transformers/peft/bitsandbytes versions,
    adapter presence, and whether 4-bit QLoRA is usable here. Exits 0 with a verdict line.

``--train`` (default)
    QLoRA 4-bit if bitsandbytes works, otherwise fp16 LoRA. Records seeds, package versions,
    wall-clock and peak VRAM, writes the adapter plus ``training-run.json``. ``--eval`` then
    measures retrieval recall@10/20 against the identity baseline on the noisy slice, a
    held-out perturbation slice and a clean slice, with a paired bootstrap CI and the clean
    non-regression gate.

Nothing here is imported by the package: it is a script, and every heavy import is inside the
function that needs it.

    # AT-GEN-18  # AT-GEN-19
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "training"
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index" / "corpus-77d7d081813738a6"
FROZEN_CORPUS_VERSION = "corpus-77d7d081813738a6"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_RECORD = "training-run.json"

#: LoRA target modules for the Qwen2 architecture (attention + MLP projections).
QWEN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_fingerprint(directory: Path) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    return {
        "files": [str(path.relative_to(directory)) for path in files],
        "bytes": sum(path.stat().st_size for path in files),
        "sha256": sha256_file(directory / "adapter_model.safetensors")
        if (directory / "adapter_model.safetensors").is_file()
        else "",
    }


def environment_versions() -> dict[str, str]:
    """Best-effort versions of everything that affects a training run."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = "missing"
    try:
        import torch

        versions["cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["gpu_total_mib"] = str(
                round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
            )
    except Exception:
        versions.setdefault("cuda", "unknown")
    driver = os.environ.get("ANSWERDESK_NVIDIA_DRIVER", "")
    if not driver:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            driver = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        except Exception:
            driver = ""
    versions["nvidia_driver"] = driver or "unknown"
    return versions


# --------------------------------------------------------------------------------------
# Data validation (shared by --dry-run and --build-data)
# --------------------------------------------------------------------------------------


def load_dataset(data_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    from answerdesk.rewrite.dataset import (
        assert_no_leakage,
        assert_no_pii,
        load_pairs,
    )

    pairs_path = data_dir / "pairs.jsonl"
    splits_path = data_dir / "splits.json"
    if not pairs_path.is_file() or not splits_path.is_file():
        raise SystemExit(
            f"training data missing in {data_dir}: run "
            "`python scripts/train_query_rewriter.py --build-data` first"
        )
    pairs = load_pairs(pairs_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assert_no_leakage(pairs)
    assert_no_pii(pairs)
    return pairs, splits


def plan_payload(
    args: argparse.Namespace, pairs: Sequence[Any], splits: dict[str, Any]
) -> dict[str, Any]:
    by_split = Counter(record.split for record in pairs)
    train_pairs = [record for record in pairs if record.split == "train"]
    steps_per_epoch = max(1, len(train_pairs) // max(1, args.batch_size * args.grad_accum))
    steps = min(args.max_steps, steps_per_epoch * args.epochs)
    return {
        "mode": "dry-run",
        "generated_at": utc_now(),
        "data": {
            "dir": str(args.data_dir),
            "pairs": len(pairs),
            "splits": dict(sorted(by_split.items())),
            "by_source": dict(sorted(Counter(r.source for r in pairs).items())),
            "by_locale": dict(sorted(Counter(r.locale for r in pairs).items())),
            "by_style": dict(sorted(Counter(r.style or "clean" for r in pairs).items())),
            "hash": splits.get("hash", ""),
        },
        "model": {
            "base_model": args.base_model,
            "quantisation": "4-bit nf4 (bitsandbytes)" if args.qlora else "none (fp16/bf16 LoRA)",
            "quantisation_note": (
                "run --check to confirm; on Windows/bitsandbytes 0.50 the 4-bit path is "
                "unavailable and the documented fallback is --no-qlora"
            ),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(args.target_modules),
        },
        "schedule": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "effective_steps": steps,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
        "outputs": {"adapter_dir": str(args.out), "run_record": str(args.out / RUN_RECORD)},
        "estimate": {
            "peak_vram_gib": round(
                (2.2 if args.qlora else 4.6) + args.batch_size * args.max_seq_len * 2e-6, 2
            ),
            "note": "rough; --check reports the real device and --train records the real peak",
        },
        "environment": environment_versions(),
    }


def run_dry(args: argparse.Namespace) -> int:
    pairs, splits = load_dataset(args.data_dir)
    payload = plan_payload(args, pairs, splits)
    if not args.quiet:
        print("dry-run: data and configuration are valid")
        data = payload["data"]
        print(
            f"  data      {data['pairs']} pairs {data['splits']} from {data['dir']}\n"
            f"  locales   {data['by_locale']}\n"
            f"  styles    {data['by_style']}"
        )
        schedule = payload["schedule"]
        print(
            f"  model     {payload['model']['base_model']} "
            f"[{payload['model']['quantisation']}], LoRA r={payload['model']['lora_r']}\n"
            f"  steps     {schedule['effective_steps']} "
            f"(batch {schedule['batch_size']}x{schedule['grad_accum']}, "
            f"lr {schedule['lr']}, seq {schedule['max_seq_len']}, seed {schedule['seed']})\n"
            f"  output    {payload['outputs']['adapter_dir']}"
        )
        env = payload["environment"]
        trainable = env.get("torch") != "missing" and env.get("cuda_available") == "True"
        print(
            f"  env       torch {env['torch']}, cuda {env.get('cuda_available')}, "
            f"gpu {env.get('gpu', 'n/a')} → "
            f"{'trainable here' if trainable else 'not trainable here (dry-run only)'}"
        )
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------------------
# Environment check
# --------------------------------------------------------------------------------------


def run_check(args: argparse.Namespace) -> int:
    env = environment_versions()
    report: dict[str, Any] = {"mode": "check", "generated_at": utc_now(), "environment": env}

    torch_ok = env.get("torch") != "missing"
    cuda_ok = env.get("cuda_available") == "True"
    four_bit: str = "not attempted"
    #: Always defined: the report reads it on every path (CUDA absent included), and a
    #: missing initialisation turned a clean "not attempted" verdict into an UnboundLocalError
    #: on CPU-only machines.
    four_bit_stderr: str = ""
    if torch_ok and cuda_ok:
        stderr_capture = io.StringIO()
        try:
            import torch
            from transformers import BitsAndBytesConfig

            _ = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # The config object alone does not prove the kernels load; a one-layer probe does.
            # bitsandbytes writes its diagnostic to stderr before raising, so capture it: the
            # recorded artefact must carry the real Windows error, not just "AssertionError".
            import bitsandbytes as bnb

            with contextlib.redirect_stderr(stderr_capture):
                linear = bnb.nn.Linear4bit(
                    16, 16, compute_dtype=torch.bfloat16, quant_type="nf4", device="cuda"
                )
                out = linear(torch.randn(2, 16, device="cuda", dtype=torch.bfloat16))
                torch.cuda.synchronize()
            del out, linear
            torch.cuda.empty_cache()
            four_bit = "ok"
        except Exception as exc:
            four_bit = f"failed: {type(exc).__name__}: {exc}"
        four_bit_stderr = stderr_capture.getvalue().strip()
    report["four_bit"] = four_bit
    report["four_bit_stderr"] = four_bit_stderr
    report["four_bit_note"] = (
        "a non-empty stderr here is bitsandbytes' own diagnostic; on Windows 0.50.2 it is "
        '"FP4 quantization state not initialized" immediately before the AssertionError'
    )
    report["adapter_present"] = (args.out / "adapter_model.safetensors").is_file()
    report["data_present"] = (args.data_dir / "pairs.jsonl").is_file()
    verdict = (
        "qlora" if four_bit == "ok" else ("fp16-lora" if torch_ok and cuda_ok else "unavailable")
    )
    report["verdict"] = verdict
    if not args.quiet:
        print(
            f"check: torch {env['torch']}, cuda {env.get('cuda_available')}, "
            f"gpu {env.get('gpu', 'n/a')}, driver {env.get('nvidia_driver')}"
        )
        print(
            f"  transformers {env['transformers']}, peft {env['peft']}, "
            f"bitsandbytes {env['bitsandbytes']}, accelerate {env['accelerate']}"
        )
        print(f"  4-bit probe: {four_bit}")
        print(
            f"  adapter present: {report['adapter_present']} "
            f"· data present: {report['data_present']}"
        )
        print(f"  verdict: {verdict}")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "environment-check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


# --------------------------------------------------------------------------------------
# Dataset rebuild
# --------------------------------------------------------------------------------------


def run_build_data(args: argparse.Namespace) -> int:
    from answerdesk.rewrite.dataset import build_dataset

    stats = build_dataset(
        corpus_dir=args.corpus_dir, out_dir=args.data_dir, seed=args.seed, quiet=args.quiet
    )
    if not args.quiet:
        print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _prompt_and_answer(tokenizer: Any, record: Any) -> tuple[str, str]:
    from answerdesk.rewrite.local_lora import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Language: {record.locale}\nMessage: {record.noisy}\nRewritten query:",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, record.clean


def _encode(tokenizer: Any, record: Any, max_len: int) -> dict[str, list[int]]:
    prompt, answer = _prompt_and_answer(tokenizer, record)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    attention = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def _collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append(item["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def _write_run_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_train(args: argparse.Namespace) -> int:
    started_wall = time.time()
    record: dict[str, Any] = {
        "mode": "train",
        "started_at": utc_now(),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "func"
        },
        "environment": environment_versions(),
        "status": "failed",
    }
    record_path = args.out / RUN_RECORD
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from answerdesk.rewrite.dataset import load_pairs

        pairs = load_pairs(args.data_dir / "pairs.jsonl")
        train_pairs = [record_ for record_ in pairs if record_.split == "train"]
        val_pairs = [record_ for record_ in pairs if record_.split == "val"]
        if args.limit:
            train_pairs = train_pairs[: args.limit]
            val_pairs = val_pairs[: max(16, args.limit // 8)]
        if not train_pairs:
            raise SystemExit("no training pairs after filtering")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if args.qlora:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_config,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        record["parameters"] = {
            "trainable": trainable,
            "total": total,
            "trainable_pct": round(100.0 * trainable / max(1, total), 4),
        }

        encoded = [_encode(tokenizer, record_, args.max_seq_len) for record_ in train_pairs]
        val_encoded = [_encode(tokenizer, record_, args.max_seq_len) for record_ in val_pairs]

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.train()
        pad_id = tokenizer.pad_token_id or 0
        steps_per_epoch = max(1, len(encoded) // max(1, args.batch_size * args.grad_accum))
        total_steps = min(args.max_steps, steps_per_epoch * args.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.03
        )
        rng = random.Random(args.seed)
        losses: list[float] = []
        step = 0
        tokens_seen = 0
        train_started = time.time()
        while step < total_steps:
            order = list(range(len(encoded)))
            rng.shuffle(order)
            for batch_start in range(0, len(order), args.batch_size * args.grad_accum):
                if step >= total_steps:
                    break
                chunk = order[batch_start : batch_start + args.batch_size * args.grad_accum]
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0.0
                micro_batches = [
                    chunk[index : index + args.batch_size]
                    for index in range(0, len(chunk), args.batch_size)
                ]
                for micro in micro_batches:
                    if not micro:
                        continue
                    batch = _collate([encoded[index] for index in micro], pad_id)
                    tokens_seen += int(batch["attention_mask"].sum())
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast(
                        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
                    ):
                        outputs = model(**batch)
                        loss = outputs.loss / max(1, len(micro_batches))
                    loss.backward()
                    accumulated += float(loss.detach())
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                if step < total_steps - 1:
                    scheduler.step()
                step += 1
                losses.append(accumulated)
                if not args.quiet and (
                    step == 1 or step % args.log_every == 0 or step == total_steps
                ):
                    elapsed = time.time() - train_started
                    peak = (
                        torch.cuda.max_memory_allocated() / (1024**3)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    print(
                        f"  step {step}/{total_steps} loss {accumulated:.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e} "
                        f"{elapsed:.0f}s vram {peak:.2f} GiB"
                    )
                if step >= total_steps:
                    break

        # Validation loss on the held-out split (teacher-forced, no generation).
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for val_batch_start in range(0, len(val_encoded), args.batch_size * 4):
                micro = val_encoded[val_batch_start : val_batch_start + args.batch_size * 4]
                if not micro:
                    continue
                batch = _collate(micro, pad_id)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    outputs = model(**batch)
                val_losses.append(float(outputs.loss.detach()))
        train_wall = time.time() - train_started
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )

        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.out))
        tokenizer.save_pretrained(str(args.out))
        fingerprint = directory_fingerprint(args.out)
        record.update(
            {
                "status": "trained",
                "finished_at": utc_now(),
                "wall_clock_s": round(train_wall, 1),
                "total_wall_clock_s": round(time.time() - started_wall, 1),
                "steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "tokens_seen": tokens_seen,
                "loss_first": round(losses[0], 4) if losses else None,
                "loss_last": round(losses[-1], 4) if losses else None,
                "loss_mean_last_10": round(sum(losses[-10:]) / max(1, len(losses[-10:])), 4)
                if losses
                else None,
                "val_loss": round(sum(val_losses) / len(val_losses), 4) if val_losses else None,
                "val_pairs": len(val_encoded),
                "peak_vram_gib": round(peak_vram, 3),
                "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "quantisation": "4bit-nf4" if args.qlora else "none-bf16",
                "adapter": fingerprint,
                "adapter_size_mib": round(fingerprint["bytes"] / (1024**2), 3),
                "reproduce": (
                    "python scripts/train_query_rewriter.py --train --eval "
                    f"--base-model {args.base_model} --seed {args.seed} "
                    f"--max-steps {args.max_steps} --lr {args.lr} "
                    f"--batch-size {args.batch_size} --grad-accum {args.grad_accum} "
                    f"--lora-r {args.lora_r} --max-seq-len {args.max_seq_len} "
                    f"--{'qlora' if args.qlora else 'no-qlora'}"
                ),
            }
        )
        _write_run_record(record_path, record)
        if not args.quiet:
            print(
                f"trained {total_steps} steps in {train_wall:.0f}s; "
                f"peak VRAM {peak_vram:.2f} GiB; val loss {record['val_loss']}; "
                f"adapter {record['adapter_size_mib']} MiB at {args.out}"
            )
    except SystemExit:
        raise
    except Exception as exc:
        record.update(
            {
                "status": "infeasible",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "total_wall_clock_s": round(time.time() - started_wall, 1),
            }
        )
        _write_run_record(record_path, record)
        print(
            f"training failed: {type(exc).__name__}: {exc}\n"
            f"  wrote {record_path} (status=infeasible) — this is a documented negative result "
            "path, not a silent failure",
            file=sys.stderr,
        )
        return 2
    return 0


# --------------------------------------------------------------------------------------
# Evaluation against the retrieval pipeline
# --------------------------------------------------------------------------------------


class _PipelineSearcher:
    """Adapter from the retrieval pipeline to the harness's ``Searcher`` protocol."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def search_ids(self, query: str, k: int) -> list[str]:
        return [passage.chunk.chunk_id for passage in self.pipeline.search(query, k)]


def _build_searcher(index_dir: Path, corpus_version: str) -> tuple[Any, str]:
    from answerdesk.config import load_settings
    from answerdesk.index.store import IndexBundle
    from answerdesk.retrieval.pipeline import build_pipeline
    from answerdesk.testing.doubles import HashingEmbedder

    settings = load_settings(profile="offline")
    embedder = HashingEmbedder(settings.embedding.dimension)
    index = IndexBundle.load(index_dir, embedder)
    pipeline = build_pipeline(settings, index)
    return _PipelineSearcher(pipeline), index.manifest.index_version


def _build_eval_slices(
    corpus_dir: Path, *, seed: int, data_dir: Path
) -> tuple[dict[str, list[Any]], Any]:
    from answerdesk.index.store import ChunkStore
    from answerdesk.retrieval.datasets import build_cases, load_query_rows
    from answerdesk.rewrite.dataset import MANGLERS, perturb
    from answerdesk.rewrite.evaluate import EvalCase

    store = ChunkStore.from_corpus(corpus_dir)
    rows = load_query_rows(corpus_dir)
    cases = build_cases(list(store.chunks), rows)

    noisy = [
        EvalCase(
            case_id=case.case_id,
            locale=case.locale,
            query=case.query,
            gold_chunk_ids=list(case.gold_chunk_ids),
            style=case.style or "",
            category=case.category,
            split=case.split,
        )
        for case in cases
        if case.category == "noisy_freeform"
    ]
    clean = [
        EvalCase(
            case_id=case.case_id,
            locale=case.locale,
            query=case.query,
            gold_chunk_ids=list(case.gold_chunk_ids),
            category=case.category,
            split=case.split,
        )
        for case in cases
        if case.category in {"direct", "multi_hop", "numeric_date", "identifier"}
    ]
    # Held-out perturbation set: clean questions whose text never appears in the training
    # pairs, mangled with a seed family (90000+) the dataset builder never uses, and filtered
    # once more so no evaluation string can be a training string.
    from answerdesk.rewrite.dataset import load_pairs, normalise

    pairs_path = data_dir / "pairs.jsonl"
    seen: set[str] = set()
    if pairs_path.is_file():
        for record in load_pairs(pairs_path):
            seen.add(normalise(record.clean))
            seen.add(normalise(record.noisy))
    unseen = [case for case in clean if normalise(case.query) not in seen]
    held_out: list[EvalCase] = []
    styles = list(MANGLERS)
    for index, case in enumerate(unseen[:200]):
        style = styles[index % len(styles)]
        mangled = perturb(case.query, style, seed=seed + 90000 + index)
        if normalise(mangled) in seen:
            continue
        held_out.append(
            EvalCase(
                case_id=f"ho-{case.case_id}",
                locale=case.locale,
                query=mangled,
                gold_chunk_ids=list(case.gold_chunk_ids),
                style=style,
                category="held_out_perturbation",
                split="test",
            )
        )
    # Keep the clean slice comparable in size to the noisy one to bound runtime.
    clean_sample = clean[: max(240, len(noisy) // 2)]
    return {"noisy_freeform": noisy, "held_out": held_out, "clean": clean_sample}, cases


def run_eval(args: argparse.Namespace, *, rewriter: Any = None, searcher: Any = None) -> int:
    from answerdesk.rewrite.config import load_rewrite_config
    from answerdesk.rewrite.evaluate import evaluate_rewriter, write_report

    if rewriter is None and args.rewrite_backend == "deterministic":
        from answerdesk.rewrite.deterministic import DeterministicRewriter

        rewriter = DeterministicRewriter()
    elif rewriter is None:
        config = load_rewrite_config(args.rewrite_config)
        if not (args.out / "adapter_model.safetensors").is_file():
            print(
                f"no adapter at {args.out}: evaluate the deterministic baseline with "
                "`--eval-only --rewrite-config configs/rewrite/deterministic.yaml`",
                file=sys.stderr,
            )
            return 2
        from answerdesk.rewrite.local_lora import LocalLoRARewriter

        rewriter = LocalLoRARewriter(
            adapter_path=args.out,
            base_model=args.base_model,
            max_new_tokens=config.local_lora.max_new_tokens,
            timeout_ms=config.local_lora.timeout_ms,
            batch_size=args.eval_batch_size,
        )
        rewriter.load()
    if searcher is None:
        searcher, index_version = _build_searcher(args.index_dir, FROZEN_CORPUS_VERSION)
    else:
        index_version = "fake"
    slices, _cases = _build_eval_slices(args.corpus_dir, seed=args.seed, data_dir=args.data_dir)
    report, traces = evaluate_rewriter(
        rewriter=rewriter,
        slices=slices,
        searcher=searcher,
        corpus_version=FROZEN_CORPUS_VERSION,
        index_version=index_version,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    out_dir = args.eval_out or (args.data_dir / "eval")
    written = write_report(report, out_dir, traces=traces if args.save_traces else None)
    if not args.quiet:
        print(
            f"evaluated {report.n_cases} cases with backend `{report.backend}` "
            f"(corpus {report.corpus_version})"
        )
        for entry in report.slices:
            print(
                f"  {entry['name']:<16} n={entry['n']:<4} "
                f"recall@20 {entry['recall@20']:.4f} → {entry['recall@20_rewritten']:.4f} "
                f"Δ {entry['delta@20']:+.4f} "
                f"[{entry['ci_low@20']:+.4f}, {entry['ci_high@20']:+.4f}]"
            )
        print(
            f"  clean gate: {'passed' if report.clean_gate_passed else 'FAILED'} "
            f"(Δ {report.clean_gate_delta:+.4f}); latency p50 "
            f"{report.latency_ms.p50} ms; wrote {', '.join(p.name for p in written)}"
        )
    return 0 if report.clean_gate_passed else 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_query_rewriter",
        description="Train and evaluate the local QLoRA query rewriter (t26).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate data/config, no training")
    mode.add_argument("--check", action="store_true", help="probe the training environment")
    mode.add_argument("--build-data", action="store_true", help="rebuild data/training from corpus")
    mode.add_argument("--eval-only", action="store_true", help="evaluate an existing adapter")
    parser.add_argument("--train", action="store_true", help="run training (default action)")
    parser.add_argument("--eval", action="store_true", help="evaluate after training")
    parser.add_argument("--save-traces", action="store_true", help="include per-case traces")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "adapter")
    parser.add_argument("--eval-out", type=Path, default=None)
    parser.add_argument(
        "--rewrite-config", type=Path, default=REPO_ROOT / "configs" / "rewrite" / "default.yaml"
    )
    parser.add_argument(
        "--rewrite-backend",
        choices=["local_lora", "deterministic"],
        default="local_lora",
        help="which rewriter the evaluation measures",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=320)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(QWEN_TARGETS))
    parser.add_argument("--qlora", dest="qlora", action="store_true", default=True)
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    parser.add_argument("--limit", type=int, default=0, help="cap training pairs (smoke runs)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.build_data:
        return run_build_data(args)
    if args.check:
        return run_check(args)
    if args.dry_run:
        return run_dry(args)
    if args.eval_only:
        return run_eval(args)
    code = run_train(args)
    if code != 0:
        return code
    if args.eval:
        return run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
