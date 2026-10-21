# Grounded generation: providers, prompts, citations

Owner: **t6** (`answerdesk.llm`, `answerdesk.generate`). Normative sources: `docs/requirements.md`
§7 (citation contract), §5 (NFR-COST, NFR-PORT), §3 (invariants), and §16.3/§16.11 for the
traceability rows. This document is the operator-facing companion: it says how the layer is
built, how to point it at a model, and how to prove the behaviour.

```text
                     ┌──────────────────────────── answerdesk.llm ─────────────────────────────┐
  Settings ─────────►│ presets.py ──► factory.build_provider ──► HttpChatProvider (+Chain)    │
  (config.py)        │                                   │            │                        │
                     │                                   │            ├─► circuit breaker       │
                     │                                   │            ├─► retry policy (jitter) │
                     │                                   │            └─► CostLedger (USD)      │
                     │ recorded.py (replay)  stream.py (lazy chunks)                              │
                     └───────────────────────────────────┬────────────────────────────────────┘
                                                         │ Completion / CompletionChunk
                     ┌───────────────────────────────────▼──────────── answerdesk.generate ───┐
  passages ─────────►│ ContextPacker (budget, dedupe, lost-in-the-middle, labels p1..pN)      │
                     │        └─► PromptRegistry (versioned, hashed) ──► provider.complete      │
                     │                └─► parse_answer_draft ──► CitationBuilder (CR-2…CR-6)   │
                     │                        └─► LexicalVerifier ──► one repair pass ──► drop │
                     │                                └─► render_answer_text (markers)        │
                     └────────────────────────────────────────────────────────────────────────┘
```

---

## 1. The provider layer: one client, six presets

NFR-PORT-01 and ADR-003: **exactly one** model HTTP client. Presets are *data*
(`answerdesk.llm.presets.PRESETS`, mirrored from `answerdesk.config.PROVIDER_PRESETS`), and no
module compares against a preset name. `tests/unit/llm/test_provider_http.py` proves this with an
AST check: `providers.py` may not contain a comparison or `match` on a preset-name literal
(auditable, not aspirational — CL-PORT-01).

| Preset | Default `base_url` | Default model | Key | Kind |
|---|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` | remote |
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-chat` | `DEEPSEEK_API_KEY` | remote |
| `openrouter` | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | remote |
| `openai_compatible` | `http://127.0.0.1:8000/v1` | `default` | none | local |
| `vllm` | `http://127.0.0.1:8000/v1` | `Qwen/Qwen3-8B-Instruct` | none | local |
| `llama_cpp` | `http://127.0.0.1:8080/v1` | `local-model` | none | local |
| `ollama` | `http://127.0.0.1:11434/v1` | `qwen3:8b` | none | local |
| `fake` | `memory://fake` | `fake-model` | none | CI double only |

*Input aliases* (`llama.cpp`, `llama-cpp`, `openai-compatible`) are normalised to the canonical
keys before lookup, because the requirements spell the preset `llama.cpp` while Python identifiers
cannot.

### 1.1 Pointing it at a local server

Nothing to install in Python and nothing to change in code:

```bash
# Ollama
ollama serve & ollama pull qwen3:8b
ANSWERDESK_LLM__PRESET=ollama uv run answerdesk config check

# vLLM (OpenAI-compatible server)
vllm serve Qwen/Qwen3-8B-Instruct --port 8000
ANSWERDESK_LLM__PRESET=vllm uv run answerdesk config check

# llama.cpp server (OpenAI API mode)
llama-server -m model.gguf --port 8080 --chat-template <template>
ANSWERDESK_LLM__PRESET=llama_cpp ANSWERDESK_LLM__MODEL=local-model \
  uv run answerdesk config check

# Or simply use the local profile (defaults to ollama, enables the local embedder/reranker)
uv run answerdesk --profile local config show
```

`answerdesk config check` prints the resolved `ProviderConfig` with the key redacted. A local
preset never reads a key: `resolve_api_key` returns `None` without consulting the environment, and
the request carries no `Authorization` header (asserted in
`test_local_server_needs_no_key_and_sends_no_authorization_header` and, over a real socket, in
`tests/integration/generation/test_local_http_server.py`).

### 1.2 Transport features and where they are tested

| Feature | Implementation | Test |
|---|---|---|
| Endpoint + auth | `HttpChatProvider._endpoint`, `_headers` | `test_request_uses_the_chat_completions_endpoint_and_bearer_key` |
| Structured output | `response_format.json_schema` (`strict: true`) when a schema is passed, `{"type":"json_object"}` for `llm.json_mode`, nothing otherwise | `test_structured_output_sends_a_strict_json_schema`, `test_json_mode_falls_back_to_json_object` |
| Streaming (SSE) | `stream()` → `ChunkStream`; `data:` lines, `[DONE]`, tolerant of comments/keep-alives | `test_streaming_yields_deltas_then_done` |
| Timeout | `httpx.Timeout(config.timeout_s)`; timeouts raise `ProviderTimeoutError` | `test_timeout_is_retried_then_raised_with_the_configured_budget` |
| Retries | `RetryPolicy`: `delay = min(max_delay, base·2^attempt) · (1 − jitter + 2·jitter·rng())` | `test_nominal_backoff_doubles_and_caps`, `test_jitter_is_symmetric_and_seeded` |
| Retry-After | honoured (delta-seconds only), capped at 30 s, wins over the backoff guess | `test_retry_after_header_is_respected` |
| Non-retryable statuses | 400/401/403/404/413/422 fail immediately | `test_retryable_status_is_retried_and_fatal_status_is_not` |
| Circuit breaker | closed → open after `failure_threshold`, half-open probe after `recovery_timeout_s` | `test_circuit_opens_after_repeated_failures_and_fails_fast`, `test_breaker_opens_then_recovers` |
| Fallback chain | `ProviderChain` implements `ChatProvider`; primary → fallbacks; streaming falls back only before the first delta | `test_fallback_chain_answers_from_the_next_provider`, `test_chain_streaming_falls_back_before_the_first_delta` |
| Cost accounting | every call (success or failure) appends a `CostRecord` | `test_every_call_records_tokens_usd_latency_and_model` |
| Tool definitions | `complete(..., tools=[...], tool_choice=...)` forwards the OpenAI `tools` subset verbatim; the answer path never sends tools (autonomous action is a v1.0 non-goal) | `test_tool_definitions_are_forwarded_verbatim` |

Fallbacks are configuration: `llm.fallback_presets: [deepseek, vllm]`. A fallback whose key is not
set is skipped with a warning (it is "not available", not an error); a fallback equal to the primary
is ignored.

### 1.3 Failure modes (NFR-OPS-01 rows owned here)

| Failure | Behaviour | Reason code / error |
|---|---|---|
| Provider timeout | retried with backoff, then `ProviderTimeoutError`; chain moves on | `PROVIDER_ERROR` if no provider answers |
| Provider 5xx / 429 | retried, then `ProviderHTTPError`; chain moves on | `PROVIDER_ERROR` |
| Provider 401/403 | not retried (a key will not fix itself) | `PROVIDER_ERROR` |
| Unreachable local server | `ProviderConnectionError` after retries | `PROVIDER_ERROR` |
| Malformed model JSON | one strict re-ask, then escalation; never a crash | `PROVIDER_ERROR` |
| Empty evidence | no model call at all | `NO_EVIDENCE` |
| Model declares insufficiency | structured refusal, no prose | `NO_EVIDENCE` |
| Unsupported claim | one repair pass, then drop; if nothing survives | `VERIFIER_REJECT` / `CITATION_FAILED` |
| Injection-shaped evidence | no model call at all | `INJECTION_SUSPECTED` |
| Broken ambient proxy (e.g. Windows `socks4` registry entry) | client falls back to `trust_env=False` with a warning instead of failing at startup | — |

---

## 2. Cost and latency ledger (NFR-COST-01, NFR-COST-02)

`answerdesk.llm.cost` is the only place a dollar amount is computed.

* `CostRecord` — one row per model call: `request_id`, `session_id`, `stage`, `provider`, `preset`,
  `model`, `input_tokens`, `output_tokens`, `estimated_input_tokens`, `usage_source`
  (`provider` | `estimated` | `none`), `usd`, `priced`, `pricing_source`, `latency_ms`, `attempts`,
  `prompt_version`, `ok`, `error_code`.
* `CostLedger.summary()` — per-session aggregate: calls, ok/failed, tokens, `usd_total`,
  `usd_mean_per_call`, `usd_p95`, latency p50/p95, per-model breakdown, `unpriced_models`,
  `usage_sources`, and `estimate_mismatch`.
* `CostReport.from_summaries()` — report-level total across sessions (t7/t9 consume it).
* Prices live in `configs/pricing.yaml`, each row carrying `retrieved_at` and `source`
  (`PricingEntry`). A price change is a YAML edit — `test_a_price_change_is_a_config_edit` proves
  the code path takes the number from the file.

Token counts: provider usage is authoritative whenever the provider reports it. The local estimate
(`estimate_tokens`) exists so the packer can enforce a budget and so an implausible usage number is
detectable; it is **never** used for billing. The 0.5% reconciliation target of NFR-COST-01 is a
release-gate measurement with a real tokeniser (t11) — offline CI asserts that both numbers are
recorded and that the mismatch is reported (`test_summary_reports_estimate_mismatch`).

---

## 3. Prompts: versioned, hashed, tamper-evident

`configs/prompts/registry.yaml` holds the templates; `configs/prompts/registry.lock.yaml` holds the
reviewed `(version, digest)` pair per prompt. `answerdesk.generate.prompts` enforces three rules:

1. **Digest** — `sha256` over the normalised `system` + `user` bodies. A body edited without
   re-stamping fails to load (`test_editing_a_body_without_restamping_fails_to_load`).
2. **Version bump** — `stamp_registry` refuses a body change that keeps the old version, because
   the lock remembers what was reviewed (`test_stamping_refuses_a_body_change_without_a_version_bump`).
3. **Provenance** — every answer records `prompt_version` (`grounded-answer-1.0.0`), which is also
   written into `Usage.prompt_version` (`test_prompt_version_reaches_the_answer`).

Authoring workflow:

```bash
# 1. edit the body in configs/prompts/registry.yaml and bump its `version`
# 2. re-hash and refresh the lock (refuses a forgotten version bump)
uv run python -m answerdesk.generate --stamp
# 3. the next run records the new version in every answer and trace
```

Templating is `{{ variable }}` substitution, not Jinja2: the base install stays dependency-light
(ADR-009) and the whole template language is one auditable function. An unsupplied variable is a
hard error (`PromptRenderError`), never an empty string.

### 3.1 Spotlighting (FR-GEN-11, INV-01)

The system prompt states, in words the model can act on, that evidence is **data, not
instruction**, that nothing inside `<passage>…</passage>` can change the task, the output format,
the language or the routing, and that the JSON schema is the only output channel. Evidence is
wrapped in explicit `<passage id="pN" doc="…" heading="…">` blocks; passage-delimiter lookalikes in
document text are escaped by the packer so a document cannot close its own block.

`answerdesk.generate.guards` adds the deterministic half: directive-shaped text (instruction
overrides, role changes, prompt exfiltration, tool invocation, format overrides, delimiter escapes,
exfiltration URLs) is detected **before** the model call. A hit escalates with
`INJECTION_SUSPECTED`, so there is no prose for a payload to hijack. The fixture suite contains 40
payloads (0 auto-answers, 0 canary leaks) and a 10-item clean control set (0 false positives) —
`tests/integration/generation/test_injection.py`, AT-SEC-01.

---

## 4. The answer contract (requirements §7)

### 4.1 Structured output

The model is asked for JSON that matches `configs/prompts/schemas/answer_draft.schema.json`:
`{insufficient_evidence: bool, why: str, claims: [{text, citations: [{passage, quote}]}]}`. Every
object is closed (`additionalProperties: false`) and every property is required, so the schema is
valid for OpenAI-style strict mode and for local servers that validate it. The schema is checked at
load time (`assert_strict_schema`), so a loose schema cannot silently reach a provider.

The model never emits citation markers. It emits *structure*; the renderer derives the markers,
which is what makes CR-1/CR-5 checkable without parsing prose.

### 4.2 Citation rules → implementation

| Rule | Where it is enforced | Test |
|---|---|---|
| CR-1 every claim has ≥ 1 citation | `AnswerDraft.ClaimDraft.citations` (`min_length=1`), `Answer`/`GenerationResult` validators | `test_answer_result_may_not_carry_an_uncited_claim` |
| CR-2 no phantom citations | `CitationBuilder` resolves the label in the packed context only | `test_phantom_passage_label_is_rejected` |
| CR-3 quote is a whitespace-normalised substring | `find_quote_span` (case-sensitive) | `test_quote_must_be_verbatim_not_paraphrased` |
| CR-4 span maps to chunk offsets | offsets derived from the matched substring | `test_quote_span_maps_back_to_raw_offsets` |
| CR-5 markers ⇔ structure, numbering by first appearance | `CitationBuilder.build`, `render_answer_text`, `validate_render` | `test_numbering_is_by_first_appearance`, `test_rendered_markers_match_claim_structure` |
| CR-6 `doc_hash` recorded | `Citation.doc_hash = chunk.doc_hash` | `test_citation_resolves_to_the_packed_chunk` |
| CR-7 unsupported ⇒ repair once, then remove/escalate | pipeline state machine | `tests/integration/generation/test_repair.py` |
| CR-8 a refusal carries no prose | `GenerationResult._answer_shape` | `test_escalation_result_may_not_carry_prose` |
| CR-9 numeric/date guard | `numeric.canonical_numeric_tokens` + `numeric_violations` | `tests/integration/generation/test_numeric_guard.py` |
| CR-10 answer language = query language, else a notice | `language.detect_language`, `language_notice` | `test_wrong_language_answer_gets_a_visible_notice` |
| CR-11 document text is data | prompt spotlighting + delimiter neutralisation | `test_delimiter_escape_is_neutralised_before_the_prompt` |
| CR-12 streaming never reveals a retracted citation | citations are built only after verification; the stream carries no markers | `test_streamed_and_final_citations_are_identical` |

The numeric guard is deliberately strict and model-free: a number or date in a displayed claim must
appear, after canonicalisation, in a chunk that claim cites. The canonicalisation table (thousand
separators, decimal comma, leading zeros, ISO/dotted/month-name dates, times) is documented in
`answerdesk/generate/numeric.py` and pinned by parametrised tests.

### 4.3 Verification, repair and refusal

```text
draft ──► CitationBuilder ──► LexicalVerifier (quote → numeric → coverage)
                                   ├─ supported   → keep
                                   ├─ partial     → keep, support=partial
                                   └─ unsupported → ONE repair pass (repair-answer prompt)
                                                        ├─ supported now → keep
                                                        └─ still bad    → drop the claim
                                                                         └─ nothing left → escalate
```

* The repair budget is **one** pass (`generate.repair_passes`, config) — unbounded repair loops are
  a latency and cost hazard.
* After the pass, surviving claims are pruned to the citations they actually use, and renumbered by
  first appearance, so the displayed numbering never depends on a dropped claim.
* If nothing survives, the reason code distinguishes a citation failure (quote or numeric problem)
  from a verifier rejection (low lexical support).
* The default verifier (`lexical`) is deterministic so offline CI can prove the *state machine*.
  F-1/F-2/F-3/F-9 are judge-mediated release numbers measured by t11 on the frozen split; the marker
  `# AT-GEN-08` in `pipeline.py` records the hook, not a claim that CI proves quality.

### 4.4 Context packing (FR-GEN-05)

* **Hard budget**: `generate.token_budget` is a request ceiling, not an evidence ceiling. The
  pipeline reserves tokens for the system prompt, the question, the answer
  (`llm.max_tokens`) and a small margin, then the packer spends the rest on evidence. Passages that
  do not fit are dropped **whole** — a truncated passage would break CR-3, so recall loss is
  measured instead of truncation being hidden.
* **Dedupe**: token-count cosine ≥ `retrieval.dedupe_cosine` (default 0.95) collapses near
  duplicates; exact duplicates are dropped too.
* **Window**: `generate.context_k_min..context_k_max` (default 8–12).
* **Lost-in-the-middle order**: strongest first, second-strongest last, remainder descending in
  between: `[s1, s3, s4, …, sn, s2]`.
* **Labels**: `p1 … pN` assigned in *prompt* order, so the model reads the strongest evidence first.

---

## 5. Language policy (FR-GEN-06)

`detect_language` is a deterministic script + cue-word detector for ru, tg, uz, ky, tr, zh and en;
it answers `None` when the text carries no signal (one word, digits only) and the pipeline falls
back to `en`. The prompt receives both the code and the English name ("Answer in Russian"). If the
model answers in another language, the answer gets a localised, visible notice
(`language_notice`) and `GenerationResult.language_notice = True` — the pipeline never silently
relabels the answer.

---

## 6. Determinism and test doubles (FR-GEN-09)

| Double | What it is for |
|---|---|
| `FakeProvider` (`answerdesk.testing`) | plumbing: reply derived only from the last user message |
| `RecordedProvider` (`answerdesk.llm.recorded`) | replay a captured transcript by request digest; a stale fixture raises `RecordedFixtureMiss` instead of replaying a neighbour |
| `RecordingProvider` | authoring: wraps a live provider and writes the fixture |
| `ScriptedProvider` (test-local) | queue of draft strings, one per completion — the only way to script "bad answer, then repaired answer" |
| `MockOpenAI` (test-local) | an HTTP handler driving `httpx.MockTransport`: real request objects, real SSE bytes, scripted failures, no socket |

`tests/integration/generation/test_determinism.py` records a run, replays it twice and compares the
canonical JSON of the whole `GenerationResult` (claims, citations, signal, usage, provenance). The
committed fixture contains digests and a content-free preview (`messages=2 roles=system,user
chars=1234`) — never prompt text, so a fixture cannot leak a customer question (INV-05).

The suite is network-free by construction: `tests/conftest.py` patches `socket.connect`, so any
attempt to open a socket fails with a message naming NFR-DET-01. The one genuinely socket-based
test (`test_local_http_server.py`) binds `127.0.0.1` and runs only with
`ANSWERDESK_ALLOW_NETWORK=1`.

---

## 7. Command reference

```bash
uv run answerdesk ask --fake "Сколько стоит обслуживание карты?"   # CMD-H: offline demo path
uv run answerdesk ask --fake --json "Сколько стоит обслуживание карты?"  # the contract object
uv run python -c "from answerdesk.llm.presets import PRESETS; print(sorted(PRESETS))"  # CMD-I
uv run pytest -q tests/unit/llm tests/integration/generation       # CMD-G
uv run python -m answerdesk.generate --stamp                       # re-hash prompts
```

`ask --fake` answers from the checked-in demo pack (`answerdesk/generate/demo.py`: eight synthetic
support chunks in the six locales). A question that matches no demo case yields an empty evidence
pack and therefore an escalation with `NO_EVIDENCE` — the demo path never invents an answer.

**Known seam.** Without `--fake`, `ask` needs a passage source; the retrieval stack is owned by t5
and wired into the CLI/service by t9/t14. Until then `run_ask` raises a `ConfigError` that says so
rather than answering from the demo corpus in production mode. `AnswerPipeline.answer(question,
passages)` is the integration point: it takes any `Sequence[RetrievedPassage]`.

---

## 8. Requirement → code → test map

| Requirement | Code | Test |
|---|---|---|
| FR-GEN-01 | `pipeline.py`, `citations.render_answer_text` | `test_answer_is_a_structured_cited_object`, `test_completion_schema_is_sent_to_the_provider` |
| FR-GEN-02 | `citations.py`, `numeric.py` | `tests/integration/generation/test_citations.py` |
| FR-GEN-03 | `citations.py` (`CitationBuilder`, `render_answer_text`) | `tests/integration/generation/test_streaming.py` |
| FR-GEN-04 | `prompts.py`, `schema.py`, `configs/prompts/` | `tests/integration/generation/test_prompts.py` |
| FR-GEN-05 | `packer.py` | `tests/integration/generation/test_packer.py` |
| FR-GEN-06 | `language.py`, prompt contract | `tests/integration/generation/test_language.py` |
| FR-GEN-07 | `verifier.py`, `pipeline.py` | `tests/integration/generation/test_repair.py` |
| FR-GEN-08 | pipeline + t7 metrics | release gate (t11) |
| FR-GEN-09 | `llm/recorded.py`, `testing/doubles.py` | `tests/integration/generation/test_determinism.py` |
| FR-GEN-10 | `pipeline.py` refusal channel | `test_empty_evidence_escalates_without_calling_the_model`, `test_twenty_no_evidence_queries_produce_twenty_structured_signals` |
| FR-GEN-11 | `configs/prompts/registry.yaml`, `guards.py` | `tests/integration/generation/test_injection.py` |
| FR-GEN-12 | `numeric.py`, `pipeline.py` | `tests/integration/generation/test_numeric_guard.py`, `test_repair_that_does_not_help_drops_the_claim` |
| FR-OPS-03 | `presets.py`, `factory.py` | `tests/unit/llm/test_presets.py`, `test_factory.py` |
| NFR-COST-01/02 | `cost.py`, `configs/pricing.yaml` | `tests/unit/llm/test_cost_ledger.py` |
| NFR-PORT-01 | `providers.py`, `stream.py`, `factory.py` | `tests/unit/llm/test_provider_http.py` |
| NFR-SEC-01 | `guards.py`, prompt spotlighting | `tests/integration/generation/test_injection.py` |
| NFR-ML-02 | no translation endpoint is ever called | `test_only_chat_completions_is_ever_called` |

---

## 9. Limitations (stated, not hidden)

1. **Quality numbers are not asserted here.** F-1…F-10 are judge-mediated release measurements
   (t7/t11). Offline CI proves the *structure*: no uncited claim, no phantom citation, no displayed
   number that is absent from its evidence.
2. **The lexical verifier is a baseline.** It trades recall for determinism. A judge- or NLI-based
   verifier is a drop-in behind `ClaimVerifier`; it inherits the judge-bias caveats of §10.5.
3. **`estimate_tokens` is a heuristic** (≈4 characters per token, 1 per CJK ideograph). It bounds
   the packer and flags implausible usage; it is not a billing tokeniser.
4. **The demo pack is not retrieval.** `ask --fake` answers from eight curated synthetic chunks; it
   proves the pipeline, not retrieval quality (that is CMD-D/CMD-F, t5).
5. **Streaming is chunk-level, not token-level, in the doubles.** The HTTP provider forwards real
   SSE deltas; `FakeProvider`/`RecordedProvider` chunk by words so that assertions stay stable.
6. **Provider-specific extensions are out of scope.** Anything outside the chat-completions subset
   (vendor-only tool formats, native vision payloads) is not implemented, by design (ADR-003).
