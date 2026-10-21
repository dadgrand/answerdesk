# Query rewriter: local QLoRA training, measured uplift, and the honest verdict

Owner: **t26** (`answerdesk.rewrite`, `scripts/train_query_rewriter.py`, `data/training/`).
Inputs: frozen corpus `corpus-77d7d081813738a6` (t22), retrieval pipeline (t5/t23),
policy PII audit (t8). Adapter card: [`adapter-card.md`](adapter-card.md).

## 1. What was asked, what was built, what it measured

The task: train a small instruction model on the owner's GPU to normalise noisy customer
messages (telegraphic, run-on, typos, Latin transliteration, code-switching) into a clean
retrieval query for ru/uz/tg/ky/zh/tr, then **measure** whether that improves retrieval.

Built and delivered:

| piece | path |
|---|---|
| training data + card | `data/training/{pairs.jsonl,splits.json,stats.json,dataset-card.md}` |
| training script (QLoRA primary, fp16/bf16 LoRA fallback) | `scripts/train_query_rewriter.py` |
| run record (seeds, VRAM, wall-clock, versions) | `data/training/adapter/training-run.json` |
| environment probe | `data/training/environment-check.json` |
| adapter (1.5B, LoRA r=16, 85.6 MiB) | `data/training/adapter/` |
| optional backend + config | `src/answerdesk/rewrite/`, `configs/rewrite/default.yaml` |
| measurement harness | `src/answerdesk/rewrite/evaluate.py` |
| measurement artefacts | `data/training/eval*/rewriter-eval.{json,md}` |

**Headline result: the rewriter does not improve retrieval on this corpus, and it degrades clean
queries. The feature ships off by default and is not recommended for enablement.** The rest of
this document is the evidence.

## 2. Hardware and the QLoRA question

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB, driver 610.88 |
| CUDA stack | `torch 2.14.1+cu126`, `transformers 4.57.6`, `peft 0.21.2`, `accelerate 1.15.0`, `bitsandbytes 0.50.2`, `datasets 3.6.0` |
| Peak VRAM (bf16 LoRA, 1.5B) | **7.58 GiB / 16 GiB** at seq 320, batch 4 × accum 4 |
| Training wall-clock | 144 steps = **84 s**; 432 steps = **228 s** (both measured) |
| Adapter | 85.6 MiB directory, `adapter_model.safetensors` 73,911,112 bytes, sha256 `acf0cae3…f5a8` |

**QLoRA 4-bit is unavailable on this Windows host.** `BitsAndBytesConfig` constructs, but the
first `bnb.nn.Linear4bit` forward pass asserts:

```
FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.
4-bit probe: failed: AssertionError:
```

`scripts/train_query_rewriter.py --check` reports the verdict (`qlora` | `fp16-lora` |
`unavailable`); here it is `fp16-lora`, and the adapter was trained with bf16 LoRA. The QLoRA
code path is the script's **default** (`--qlora`), so on Linux or a working bitsandbytes build
the same command produces the 4-bit adapter. Nothing in the base install depends on any of it.

## 3. Training data

1,430 pairs from the frozen corpus (1,153 train / 139 val / 138 test), six locales, five noise
styles; details and construction rules in
[`data/training/dataset-card.md`](../data/training/dataset-card.md). Two properties matter here:

* splits are **text- and document-disjoint** (no clean or noisy string crosses a boundary, no
  document straddles one), so the test slice is genuinely unseen;
* the PII audit excluded 86 rows (identifier false positives in the policy detector at the time;
  fixed upstream by the captain afterwards).

## 4. The measurement

Protocol (`src/answerdesk/rewrite/evaluate.py`):

* **slices** — `noisy_freeform` (474 corpus rows with a noisy style), `held_out` (156 questions
  whose text never appears in the training pairs, mangled with a seed family the builder never
  uses), `clean` (240 clean questions, the non-regression arm);
* **arms** — identity (the pipeline as shipped), deterministic normaliser, LoRA adapter;
* **metric** — `recall_any@10/@20` against the gold chunk ids, computed by running the real
  retrieval pipeline twice per case (same case, same gold, only the query differs);
* **uncertainty** — paired bootstrap over cases, 1,000 resamples, seed 20261021 (the statistic
  t23 uses for its ablation);
* **latency** — wall-clock of the rewrite call; the LoRA arm is measured batched (8 per forward
  pass) and reported as amortised, i.e. a lower bound on single-request latency.

### Results (corpus `corpus-77d7d081813738a6`, index `index-dc9909c648367626`)

| arm | slice | n | recall@20 base → rewritten | Δ@20 | 95% CI | recall@10 base → rewritten | Δ@10 | 95% CI |
|---|---|---|---|---|---|---|---|---|
| deterministic | noisy_freeform | 474 | 0.8228 → 0.8249 | **+0.0021** | [−0.0063, +0.0105] | 0.5485 → 0.5464 | −0.0021 | [−0.0105, +0.0042] |
| deterministic | held_out | 156 | 0.9679 → 0.9679 | 0.0000 | [−0.0192, +0.0192] | 0.8654 → 0.8910 | **+0.0256** | [+0.0064, +0.0513] |
| deterministic | clean | 240 | 0.9208 → 0.8917 | **−0.0292** | [−0.0542, −0.0083] | 0.7542 → 0.7375 | −0.0167 | [−0.0333, −0.0042] |
| LoRA 144 steps | noisy_freeform | 474 | 0.8228 → 0.8186 | −0.0042 | [−0.0316, +0.0232] | 0.5485 → 0.6055 | **+0.0570** | [+0.0169, +0.0992] |
| LoRA 144 steps | held_out | 156 | 0.9679 → 0.7885 | **−0.1795** | [−0.2500, −0.1154] | 0.8654 → 0.5769 | **−0.2885** | [−0.3718, −0.2051] |
| LoRA 144 steps | clean | 240 | 0.9208 → 0.8000 | **−0.1208** | [−0.1667, −0.0792] | 0.7542 → 0.6042 | **−0.1500** | [−0.2042, −0.1000] |
| LoRA 432 steps | noisy_freeform | 474 | 0.8228 → 0.8312 | +0.0084 | [−0.0211, +0.0359] | 0.5485 → 0.6118 | **+0.0633** | [+0.0253, +0.1076] |
| LoRA 432 steps | held_out | 156 | 0.9679 → 0.7949 | **−0.1731** | [−0.2436, −0.1090] | 0.8654 → 0.5769 | **−0.2885** | [−0.3718, −0.2115] |
| LoRA 432 steps | clean | 240 | 0.9208 → 0.7958 | **−0.1250** | [−0.1708, −0.0833] | 0.7542 → 0.6125 | **−0.1417** | [−0.1917, −0.0958] |

Latency: deterministic **0.025 ms** p50 per query; LoRA **110.6 ms** p50 / 149.8 ms p95 when
batched 8-at-a-time (amortised, a lower bound) and **419 ms p50 / 540 ms max for a true
single-request call** (18 calls, warm model) — i.e. ~4 orders of magnitude more than the
deterministic normaliser and far above the retrieval budget itself.

**Clean-query non-regression gate (Δ@20 ≥ −0.01): FAILED for every learned and rule-based arm**
(−0.0292 deterministic, −0.1208 LoRA-144, −0.1250 LoRA-432).

### Reading the numbers honestly

1. **Both adapters move top-10 on noisy input, and nothing else.** Δrecall@10 is +5.7pp
   [+1.7, +9.9] (144 steps) and +6.3pp [+2.5, +10.8] (432 steps), while Δrecall@20 is flat
   (−0.4pp and +0.8pp, both intervals straddling zero). Reordering the top of the ranking is not
   the same as finding more of the right documents: at the cut-off the product actually uses
   (20), there is no gain.
2. **The gain does not transfer to unseen noise.** On the held-out perturbation slice both
   adapters collapse: −18.0pp @20 / −28.9pp @10 (144 steps) and −17.3pp @20 / −28.9pp @10
   (432 steps). The 144-step adapter was trained on 178 synthetic perturbations and learned
   those strings, not the transformation.
3. **Every arm damages clean queries**, badly for the adapter (−12pp @20, −15pp @10). Rewriting a
   clean question into a paraphrase removes the exact lexical forms BM25 was tuned on. The gate
   threshold (−1pp) is missed by an order of magnitude.
4. **Longer training made generalisation worse** (validation loss 0.2897 vs 0.2645 at 144 steps,
   training loss 0.0004): the adapter memorised 1,153 templated pairs.
5. **Qualitative failure modes** in the 144-step adapter's own outputs, verbatim from the probe:
   `karta xizmati qancha turadi` (Uzbek) → `Карта обслуживание сколько стоит?` (Russian — a
   language violation the prompt forbids); `zaym kak oformit` → `Как оформить займ или кредит?`
   (an invented product expansion); `dengi kak poluchit` and `perevod tadzhikistan dengi ne
   doshli` both collapse to the same string `Деньги не пришли: как проверить перевод?` (distinct
   questions become one query, which is exactly how a cache/rewriter pair produces a wrong hit).
6. **The deterministic arm is the only positive signal, and it is tiny**: +2.6pp recall@10 on the
   held-out perturbation slice [+0.6, +5.1], zero at @20, and a −2.9pp clean regression. It costs
   0.025 ms, but it still fails the non-regression gate, so it too stays off by default.
7. **The pipeline already does this work.** t23 added noise probes, transliteration and
   typo-tolerant matching *inside* retrieval, with its own measured ablation. A rewriter in front
   of that duplicates the mechanism, and a redundant step that changes the query can only lose
   information.

That is the finding: on this corpus, with this retrieval stack, **query rewriting is not the
lever** — the ceiling was already reached inside the pipeline. This is exactly the outcome the
task asked to be reported instead of a fabricated win.

## 5. How to reproduce (or skip) the training

```bash
# 0) base install, no torch: validate data + config only (what CI runs)
uv run --extra train python scripts/train_query_rewriter.py --dry-run

# 1) environment verdict: qlora | fp16-lora | unavailable
uv run --extra train python scripts/train_query_rewriter.py --check

# 2) dataset from the frozen corpus (deterministic, seed 1337)
python scripts/train_query_rewriter.py --build-data

# 3) training (this reproduces the shipped 144-step adapter)
python scripts/train_query_rewriter.py --train --eval --no-qlora \
    --base-model Qwen/Qwen2.5-1.5B-Instruct --seed 1337 --max-steps 300 \
    --lr 2e-4 --batch-size 4 --grad-accum 4 --lora-r 16 --max-seq-len 320
#    drop --no-qlora where bitsandbytes 4-bit works; use --base-model Qwen/Qwen2.5-3B-Instruct
#    for the 3B variant (fits 16 GiB in 4-bit, ~2× the wall-clock)

# 4) measurement only, against the frozen index
python scripts/train_query_rewriter.py --eval-only --out data/training/adapter
python scripts/train_query_rewriter.py --eval-only --rewrite-backend deterministic
```

Skip training entirely if you only need the numbers above: they are committed under
`data/training/eval*/`. The dataset can be rebuilt offline; the weights need network on first
use (`HF_HOME` at a writable path — `HF_HUB_CACHE` may point somewhere unwritable on this host).

## 6. The optional backend

Off by default; `configs/rewrite/default.yaml` is the only switch.

```yaml
rewrite:
  enabled: false          # keep false: see §4
  backend: identity       # identity | deterministic | local_lora
```

* `answerdesk.rewrite.build_rewriter()` returns the identity rewriter unless enabled, imports no
  ML library, and degrades to identity with a logged reason when torch/peft or
  `adapter_config.json` is missing (`rewrite_status()` reports `available`/`fallback`/`reason`).
* `TimedRewriter` catches every exception and enforces `timeout_ms`, returning the original query
  with `fallback=True`; the request path never fails because of a rewriter.
* Base install stays torch-free (`tests/unit/rewrite/test_local_lora.py` asserts that importing
  `answerdesk.rewrite` does not load `torch`/`transformers`/`peft`).

## 7. Limitations

1. **Synthetic data, no native speakers.** Training pairs are generator-written; no native
   speaker reviewed the rewrites in any of the six locales.
2. **One corpus, one retrieval stack.** The measurement is on `corpus-77d7d081813738a6` with the
   hashing-embedder index; a dense BGE-M3 index (t11) could change the picture, because a
   semantic retriever is less sensitive to paraphrase-induced lexical loss. That experiment was
   **not** run here and the conclusion should be re-tested before the feature is discarded
   permanently.
3. **Small model, short schedule.** 1.5B and ≤432 steps is a proof, not a tuning study; no
   hyper-parameter search, no data augmentation beyond the five manglers.
4. **Latency measured both ways.** 110.6 ms p50 batched (amortised, lower bound) and 419 ms p50
   for a true single-request call — either way it dwarfs the 0.025 ms deterministic normaliser
   and is too slow for the serving path on this card.
5. **No HITL loop.** Nothing feeds real operator corrections back into the pairs.
6. **Low-resource locales** (tg/ky/uz) have the fewest pairs and the weakest expected quality.
7. **QLoRA untested in practice here** — the code path exists and is the default, but on this
   Windows host it has never produced an adapter.

## 8. Requirement map

| requirement | where |
|---|---|
| training data from corpus v2.1 + perturbations, dedup, frozen splits, card | `src/answerdesk/rewrite/dataset.py`, `data/training/`, `tests/unit/rewrite/test_dataset.py` |
| QLoRA primary + fp16 fallback, seeds/VRAM/wall-clock/versions, `--dry-run` | `scripts/train_query_rewriter.py`, `data/training/adapter/training-run.json` |
| measured uplift with paired bootstrap, latency, clean gate | `src/answerdesk/rewrite/evaluate.py`, §4 above |
| adapter delivered with checksum/size/reproduction, adapter card | `data/training/adapter/`, [`adapter-card.md`](adapter-card.md) |
| optional backend, off by default, graceful absence, offline tests | `src/answerdesk/rewrite/{backend,local_lora}.py`, `configs/rewrite/default.yaml`, `tests/unit/rewrite/` |
| honest limitations and reproduction/skip path | §4, §5, §7 |
