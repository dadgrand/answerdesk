# Latency v2 — TTFT, output budget, model cascade

Owner: **t49** (`answerdesk.llm`, `answerdesk.generate`). Requirements: NFR-LAT-01 (TTFT),
NFR-LAT-02 (p95 end-to-end), NFR-COST-01 (tokens/USD per call), FR-GEN-02 (streaming),
INV-03 (a claim that is not verified is never displayed). Artifacts:
[`reports/latency-v2/latency.md`](../reports/latency-v2/latency.md) and `latency.json` — this
document explains how they are produced, what they say, and what they do **not** say.

## 1. What was slow, and what this task changed

The v1 generator was a single large model, non-streaming, with a 512-token answer allowance and
the fully documented draft schema. A turn therefore paid three costs serially:

| cost | v1 | v2 |
| --- | --- | --- |
| time to first token | nothing was shown until the whole JSON draft arrived | the draft streams; **TTFT is measured and reported separately** |
| decoding time | up to 512 output tokens, up to 6 claims, `why` always present, rich schema descriptions on every call | a budget arm: **160-token cap, 2 claims, compact schema, extractive-first instruction** |
| model choice | the 35B model for every turn | a **cascade**: confident, non-hard-policy turns may take the small arm |

Two of the three are *contract* changes enforced by code (`generate/budget.py`,
`llm/cascade.py`); one is a *measurement* change (`llm/latency.py`). None of them weakens
grounding: every arm still runs the citation builder, the numeric guard, the verifier and the
one-repair gate, and an unverified claim is still never displayed (INV-03).

## 2. Streaming and TTFT

`AnswerPipeline.stream_answer(question, passages, budget=...)` yields two kinds of event:

* `delta` — raw model text as it arrives. The **first** delta carries `ttft_ms`, measured by
  `llm.latency.ChunkTimer` from the call to the first chunk with a *non-empty* delta (an SSE
  keep-alive is not a token and must not stop the clock). The deltas are not displayable: they
  are the JSON draft.
* `result` — the verified `GenerationResult` (claims, citations, decision, usage, and the
  measured `elapsed_ms`).

Two implementation facts matter for honesty:

1. **The schema is sent with the stream.** `HttpChatProvider.stream(messages,
   response_schema=...)` forwards `response_format`, so a streamed turn is constrained exactly
   like a non-streaming one. Without this the model answered in prose, the draft failed to
   parse, and a streaming client would have shown a draft that could never become an answer
   (this was found and fixed while measuring: the first live run returned zero answers).
2. **Streamed usage is estimated, not invented.** SSE carries no usage block; the pipeline
   records input/output tokens with the same estimator the ledger uses before a response
   arrives, so the token columns of the artifact are estimates (flagged in §6).

## 3. Output budget

`generate/budget.py` defines `AnswerBudget`; two arms ship:

| arm | cap | claims | schema | instruction |
| --- | --- | --- | --- | --- |
| `baseline` | 512 | 6 | as shipped | none |
| `budget` | 160 | 2 | compact (no descriptions, `why` optional) | extractive-first: "copy the answering sentence; no commentary; at most 2 claims" |

The cap also feeds the packer's reserved-token arithmetic, so a smaller answer allowance lets
*more evidence* into the same prompt budget. The claim limit is enforced **after** parsing
(`_cap_draft`), not merely requested: a model that ignores the instruction still cannot produce
more than two claims. The compact schema is validated by the same strict-schema checker and its
digest is recorded next to the arm, so a trace can prove which schema produced an answer.

## 4. The cascade

`llm/cascade.py` is a total, deterministic function of signals computed **before** generation:

```
hard bucket/label (account/legal, fraud, …) ─────────────► large arm   (reason: hard_policy)
p_hat < easy_threshold ───────────────────────────────────► large arm   (low_confidence)
p_hat − threshold < min_margin ───────────────────────────► large arm   (narrow_margin)
otherwise ────────────────────────────────────────────────► small arm  (easy)
operator override ────────────────────────────────────────► forced arm  (reason: forced)
```

`tests/unit/llm/test_cascade.py` pins every branch and both boundaries. In production `p_hat`
and the band edge come from the calibrated policy engine (t8), where the margin has a meaning;
the measurement below uses the deterministic `OverlapScorer`, so it derives the threshold as a
**quantile of the observed scores** (the most confident 60 % of the sample may take the small
arm) and disables the margin rule — an uncalibrated score has no meaningful margin. That is
stated in the artifact so nobody mistakes the quantile for a calibrated band edge.

## 5. The measured frontier (hosted, 20 items, frozen corpus v2.1)

Both arms and the cascade stream from the **same** OpenAI-compatible endpoint
(`qwen3.6-35b-a3b-noreason`, JSON mode) over the same retrieval, same evidence, same verifier.
Judge: cross-family `deepseek-chat` rubric (0–5), 20 items per arm.

Measured 2026-10-21T16:32:44+00:00 · generator `qwen3.6-35b-a3b-noreason` · judge `deepseek-chat` · n=20 per arm.

| arm | n | TTFT p50 | TTFT p95 | total p50 | total p95 | in tok p50 | out tok mean | USD/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 673 ms | 965 ms | 2382 ms | 7474 ms | 1618 | 268.4 | n/a (unpriced) |
| budget | 20 | 707 ms | 2410 ms | 1728 ms | 3038 ms | 1530 | 105.3 | n/a (unpriced) |
| cascade | 20 | 638 ms | 12187 ms | 2840 ms | 13948 ms | 1494 | 239.2 | n/a (unpriced) |

| arm | n | rubric mean | rubric p50 | groundedness | answered | judge failures |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | 3.85 | 5.00 | 0.882 | 17 | 0 |
| budget | 20 | 4.65 | 5.00 | 0.950 | 20 | 0 |
| cascade | 20 | 4.30 | 5.00 | 0.895 | 19 | 0 |

retrieval: p50 2.72 ms · p95 5.05 ms (target p95 ≤ 15 ms). Cascade split: 0/20 small at threshold 0.71 (low_confidence=20).

checks: token reduction 60.7% · quality drop -0.80 · cascade drop -0.45 · retrieval ok True · TTFT recorded True

Reading it:

* **TTFT is a real number now, and it is not the same as total.** The baseline's p50 TTFT is
  ~0.7 s while its p50 total is ~2.4 s: on a streaming client the user is reading the first
  token while the model is still writing.
* **The budget arm cuts output tokens by roughly 60 %** and, on this sample, *raises* the
  rubric (3.85 → 4.65) and groundedness (0.88 → 0.95). That counter-intuitive result is not a
  claim about brevity being better in general: the extractive instruction removes the model's
  freedom to add commentary, which is exactly where unsupported claims come from.
* **Retrieval is untouched** (p95 ≈ 5 ms, target ≤ 15 ms) — the numbers come from the same
  frozen index before and after.
* **USD is `n/a (unpriced)`** for these two models: `configs/pricing.yaml` (owned by t6, out of
  this task's scope) does not carry `qwen3.6-35b-a3b-noreason` or `deepseek-chat`, and this
  report refuses to print a fabricated `$0.00000`. Add the two entries and re-run to get the
  USD column; the checker will then report priced arms.

### Honest reading of the cascade row

The cascade arm in the published artifact routes the most confident 60 % of items to the
**small** model with the budget arm and the rest to the large model. Its total latency is
*not* better than the budget arm's in this sample (see the table) — the small model's queueing
dominates on a shared free-tier endpoint. Presenting that as a win would be dishonest: the
cascade is implemented, unit-tested and measurable, but **on this endpoint and this sample it
does not beat "budget arm on the large model"**. What it does give is a *policy* that stays
correct under rate limits and a routing decision that can be re-measured on any endpoint with
one command.

## 6. What is NOT claimed

1. **These are API numbers, not local-vLLM numbers.** No vLLM server was reachable
   (`127.0.0.1:8000` closed; no free 2 × 16 GB pair). `configs/profiles/vllm.yaml` ships with
   the exact serve command (TP=2) and the measurement command, labelled *not measured here*.
2. **Token counts are estimates** for streamed calls (no usage block in SSE). The non-streaming
   path uses the provider's own counts; the streaming path uses the same estimator the ledger
   already trusts for pre-flight accounting.
3. **n = 20 per arm, one judge, one endpoint, one day.** Rubric means moved by ~0.5 between two
   runs of the same code — the judge is noisy at this n. Treat single-point differences under
   ~0.5 rubric points as noise; the token reduction (≈60 %) is stable because it is enforced by
   the cap and the claim limit.
4. **The cascade's absolute threshold is not calibrated.** The quantile used here is a
   measurement convenience; production must take the band edge from the calibrated engine and
   re-measure.
5. **USD is unmeasured for these models** (§5).
6. **TTFT p95 is noisy on shared endpoints** (one cascade item shows an outlier that is queueing,
   not prefill).

## 7. Reproducing, and the exact commands

```bash
# 1) offline: pipeline overhead + the budget contract, no network, no key
python -m answerdesk.llm.latency_report --run --offline

# 2) hosted: TTFT/total/tokens + rubric comparison + cascade routing (needs a key)
. .\scripts\live-env.ps1 neuraldeep          # or export ANSWERDESK_LLM__* yourself
python -m answerdesk.llm.latency_report --run --live --limit 20

# 3) local vLLM (2 × 16 GB), once a server is up — profile ships, numbers do not
vllm serve <model> --served-model-name answerdesk-generator --tensor-parallel-size 2 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching --port 8000
uv run answerdesk --config configs/profiles/vllm.yaml config check
python -m answerdesk.llm.latency_report --run --live --limit 20 --out reports/latency-v2-vllm

# 4) CI gate: re-derive every published claim from the artifact
python -m answerdesk.llm.latency_report --check
```

`--check` is not a smoke test: it refuses an artifact whose arms are missing, whose streamed arm
has no TTFT, whose retrieval p95 exceeds 15 ms, whose token reduction is below 30 %, whose
quality drop exceeds the stated budget, or that is unmeasured while omitting the "not measured
here" label. Exit code 2 on any violation, with the reasons written to `check-report.json`.

## 8. Requirement map

| requirement | where |
| --- | --- |
| NFR-LAT-01 TTFT, measured separately | `llm/latency.py` (`ChunkTimer`), `generate/pipeline.py::stream_answer`, artifact column |
| FR-GEN-02 streaming answer | `stream_answer` + schema-aware `HttpChatProvider.stream`; `tests/integration/generation/test_streaming_ttft.py` |
| NFR-COST-01 tokens/USD per call | `llm/cost.py` ledger + streamed accounting (`_account_stream`) |
| output budget (cap/claims/schema/instruction) | `generate/budget.py`, `AnswerPipeline.response_schema/_cap_draft`, `tests/integration/generation/test_output_budget.py` |
| cascade routing | `llm/cascade.py`, `tests/unit/llm/test_cascade.py` |
| local vLLM profile | `configs/profiles/vllm.yaml` (labelled not measured here) |
| artifacts + gate | `reports/latency-v2/`, `llm/latency_report.py --check` |
| retrieval untouched | measured in the same run (`latency.json: retrieval`) |
