# `domain/` — support-domain slice from the synthetic corpus (CC0-1.0)

| Path | What |
|---|---|
| `<lang>.jsonl` | 40 items per locale for `ru`, `uz`, `tg`, `ky`, `zh`, `tr` |
| `exclusions.json` | the t11 documents and questions excluded from the slice, with the hashes of the files that produced them |
| `LICENSE-CC0-1.0.txt` | the public-domain dedication |

The data is derived from the repository's synthetic customer-support corpus
(`data/corpus/`, generator `answerdesk-synthetic-v2.0.0`, seed `20261020`), which is
**CC0-1.0**. There is no third-party text in this directory. The repository's MIT licence
applies to code only; CC0-1.0 applies to these files.

## Categories

| `category` | n | `expected_behavior` | Evidence |
|---|---|---|---|
| `direct` | 48 | `answer_from_evidence` | one verbatim FAQ answer of the article |
| `numeric_date` | 48 | `answer_from_evidence` | one verbatim FAQ answer containing the amount/date |
| `multi_hop` | 48 | `answer_from_evidence` | two verbatim FAQ answers of the same article |
| `pii` | 36 | `answer_without_echo` | one FAQ answer; `pii_tokens`/`must_not_echo` carry the reserved identifier |
| `injection` | 36 | `answer_from_evidence` | one FAQ answer; `attack.instruction` is the injected text |
| `unanswerable` | 24 | `escalate` | none (by design) |

`reference_answer` is always a verbatim slice of `data/corpus/docs/<lang>/<doc_id>.md` — for
`multi_hop` it is the two quotes joined by a space. `--verify` re-checks every quote against
the corpus and recomputes the category invariants.

## What the categories do and do not measure

* `pii` measures **non-echoing** of a reserved identifier and grounding in the article — not
  whether the assistant handled a real privacy incident.
* `injection` measures whether the assistant ignores an in-band instruction and answers from
  evidence. The reference answer is the article's first FAQ answer, mirroring the corpus
  generator's own construction; it is a *safe-reply* reference, not a semantic answer to the
  injected request.
* `multi_hop` concatenates two FAQ questions of one article and expects the concatenation of
  their answers. It is a two-evidence test, not open-domain reasoning.
* `unanswerable` uses the four distinct out-of-domain questions per locale that the corpus
  defines; duplicating them to raise n would inflate coverage without adding information.

## Safety

Only reserved identifiers appear: RFC 5737-style phone ranges (`+7 000 000-00-00`,
`+998 00 000-00-00`, …), the `example.com` mailbox, documented never-issued test PANs
(`4000 0000 0000 0002`), reserved order ids (`ORD-2026-…`) and `example.*` domains. No real
person, bank or brand is named. The corpus builder refuses to emit a corpus that violates
that policy (`docs/requirements.md` §13.4), and the slice inherits the guarantee.
