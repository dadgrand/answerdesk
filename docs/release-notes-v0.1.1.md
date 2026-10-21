# answerdesk v0.1.1 — security patch

**Prompt injection is now blocked in two layers, and the second layer fails closed.**

- Deterministic multilingual rules catch the obvious phrasings before any provider call.
- A strict-JSON classifier runs before generation; any `true` type blocks with
  `INJECTION_SUSPECTED`. Malformed JSON, schema violations, timeouts and provider errors block
  by default (`security.fail_closed=true`; operators can opt out explicitly).
- Measured live: the Russian probe phrasing blocks in 0.04 s with zero provider calls; an
  English obfuscated phrasing the rules miss is blocked by the classifier in 1.09 s; a benign
  pricing question pays 1.73 s of security latency and is answered with five citations.
- Trace gains a `security` block; 30 new tests; `docs/security-agent.md` documents the design,
  the cost and the limits.

This is a patch on top of v0.1.0. The honest limitations of the project are unchanged: risk
control at 5% is still absent, NFR-LAT-01 is still a MISS, judged metrics are still diagnostic,
and cross-language consistency is still low. Those are recorded in `CHANGELOG.md` and
`docs/results.md`.
