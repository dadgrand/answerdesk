# Adapter card — `answerdesk-query-rewriter` v1

**What it is:** a LoRA adapter on a small instruction-tuned model that rewrites a noisy
customer-support message (telegraphic, run-on, typo-ridden, Latin-transliterated, code-switched)
into one clean retrieval query in the same language.

**Status:** **trained locally on the owner's GPU** — this is not a placeholder. QLoRA 4-bit was
the intended primary path and **failed on Windows** (see "Hardware" below); the adapter was
produced with the documented fp16/bf16 LoRA fallback.

| field | value |
|---|---|
| Adapter | `data/training/adapter/` |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Qwen licence: Apache-2.0) |
| Adapter licence | MIT (same as this repository) |
| Method | LoRA, r=16, α=32, dropout 0.05, all attention + MLP projections |
| Precision | bf16 weights (no quantisation — bitsandbytes 4-bit unavailable here) |
| Trainable parameters | 18,464,768 of 1,562,179,072 (**1.18%**) |
| Adapter size | **85.6 MiB** directory / 70.5 MiB `adapter_model.safetensors` (73,911,112 bytes) |
| Adapter sha256 | `acf0cae3520bec399e6de55faee054ed77c4afab3ee2c8f7f86e84ebf50cf5a8` (`adapter_model.safetensors`) |
| Training data | `data/training/pairs.jsonl` (1430 pairs; 1153 train / 139 val / 138 test) |
| Dataset card | `data/training/dataset-card.md` |
| Training run record | `data/training/adapter/training-run.json` |
| Prompt | `rewrite-instruction-1.0.0` (`answerdesk.rewrite.local_lora.SYSTEM_PROMPT`) |

## Hardware and wall-clock

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB |
| Driver / CUDA | 610.88 / CUDA 12.6 wheels (`torch 2.14.1+cu126`) |
| Peak VRAM | **7.58 GiB** (of 16 GiB) |
| Training time | **84 s** for 144 steps (2 epochs, effective batch 16, seq len 320) |
| Validation loss | 0.2645 (train loss 0.7614 → 0.0213) |
| Seeds | 1337 (data, torch, shuffling) |
| Stack | torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.2, accelerate 1.15.0, datasets 3.6.0, bitsandbytes 0.50.2 (installed, unusable for 4-bit) |

### Why not QLoRA

`bitsandbytes 0.50.2` on Windows cannot initialise its 4-bit CUDA path here:

```
FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.
4-bit probe: failed: AssertionError:
```

`BitsAndBytesConfig` constructs fine; the `bnb.nn.Linear4bit` forward pass asserts. That is a
known Windows limitation of the wheel, not a configuration error. The training script keeps
QLoRA as the **default** code path (`--qlora`), so on Linux or a working bnb build the same
command produces a 4-bit adapter; this card describes the adapter that exists.

## Reproduce

```bash
# 1) data (frozen corpus v2.1 must be present at data/corpus)
python scripts/train_query_rewriter.py --build-data

# 2) environment check (reports qlora | fp16-lora | unavailable)
python scripts/train_query_rewriter.py --check

# 3) training + evaluation (this is the command that produced this card)
python scripts/train_query_rewriter.py --train --eval --no-qlora \
    --base-model Qwen/Qwen2.5-1.5B-Instruct --seed 1337 \
    --max-steps 300 --lr 2e-4 --batch-size 4 --grad-accum 4 \
    --lora-r 16 --max-seq-len 320

# on a machine where bitsandbytes 4-bit works, drop --no-qlora
```

Weights download needs network on first use (`HF_HOME` must point at a writable cache).

## How it is meant to be used

Off by default. Enable per deployment in `configs/rewrite/default.yaml`:

```yaml
rewrite:
  enabled: true
  backend: local_lora
  local_lora:
    base_model: Qwen/Qwen2.5-1.5B-Instruct
    adapter_path: data/training/adapter
```

`answerdesk rewrite status` (or `rewrite_status()` in the service) reports
`active_backend`/`available`/`fallback` without loading weights, and the backend degrades to
identity — logging the reason — when torch/peft or the adapter directory is missing. A rewrite
that raises or exceeds `timeout_ms` returns the original query with `fallback=True`.

## Measured behaviour

See `docs/query-rewriter.md` for the full table (retrieval recall@10/20 on the noisy slice, the
held-out perturbation slice and the clean slice, paired-bootstrap CI, latency, and the clean
non-regression gate). Summary of the honest position:

* it **does** normalise the styles it was trained on, and the retrieval measurement quantifies
  what that buys;
* it was trained on **synthetic** pairs written by a generator, not by customers;
* no native speaker reviewed its outputs in ru/uz/tg/ky/zh/tr;
* the validation loss (0.26) is far above the training loss (0.02), i.e. the adapter has partly
  memorised 1,153 templated pairs — expect weaker generalisation to unseen phrasings than the
  retrieval numbers suggest;
* 1.5B parameters was chosen for a 16 GiB card and a short wall-clock; a 3B QLoRA run is the
  same script with `--base-model Qwen/Qwen2.5-3B-Instruct`.

## Open hypothesis (t11, not tested here)

All measurements used the hashing-embedder index. A **dense BGE-M3 index** may tolerate
paraphrase-induced lexical change far better, which is the one scenario where this adapter could
still pay off. That experiment belongs to t11 and is specified in
[`query-rewriter.md`](query-rewriter.md) §4.1; until it runs, this adapter is **not recommended
for enablement** (`configs/rewrite/default.yaml` keeps `enabled: false`).

The machine-checked environment verdict lives in `data/training/environment-check.json`
(`verdict: fp16-lora`, `four_bit_stderr: "FP4 quantization state not initialized …"`).

## Limitations (explicit)

1. **Synthetic training data** — no human-written noisy queries, no HITL correction loop.
2. **No native-speaker validation** of outputs in any of the six locales.
3. **Low-resource locales** (tg, ky, uz) have the fewest training pairs and the weakest
   expected quality.
4. **Single-turn, query-only** — it does not see conversation history or retrieved passages, and
   it must not answer the question (the prompt forbids it).
5. **Identifier preservation is not guaranteed by training alone**; the backend's prompt asks for
   it and the pipeline's identifier probe is the safety net.
6. **Windows/bitsandbytes** means this artefact is full-precision LoRA, ~86 MiB, not a 4-bit
   adapter.
7. **No safety fine-tune**: PII/injection handling belongs to the policy layer, which runs
   before and after the rewriter, not inside it.
