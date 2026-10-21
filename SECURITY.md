# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately to the maintainers
listed in [`CITATION.cff`](CITATION.cff), or through the repository's private security-advisory
channel (GitHub → *Security* → *Report a vulnerability*).

Include: affected version/commit, the command or request that triggers it, what an attacker gains,
and — if you have one — a minimal reproduction against the offline profile.

We aim to acknowledge within 3 working days and to ship a fix or a documented mitigation within 30
days for high-severity issues. You will be credited unless you ask otherwise.

## What is in scope

| Area | Examples |
|---|---|
| Grounding invariants | a displayed claim without a resolving citation; a citation whose quote is not in the cited chunk; a numeric claim absent from its evidence (`INV-02`, `INV-03`, `INV-04`) |
| Prompt injection | document or retrieved text that changes routing, tool use, output schema or guard decisions (`INV-01`) |
| PII / secrets | PII reaching the index, a log, a trace file or an outbound provider request body (`INV-05`); secrets in logs or error envelopes |
| Service surface | auth bypass on `/admin/*`, idempotency-key abuse, rate-limit bypass, unbounded body/header handling, error envelopes leaking internals |
| Console | XSS via model or document text, unsafe link handling, clickjacking |
| Cache | a stale or wrong hit that survives the document/version guards (`FR-CAC-02…05`) |
| Supply chain | a dependency with a known high/critical advisory at release time; a dependency whose licence is incompatible with MIT |

## What is out of scope

- Findings that require a real production deployment — this repository ships no deployment.
- Self-XSS requiring the operator to paste a payload into devtools.
- Denial of service by an operator with the admin token (that is a trusted role by design).
- Model-provider vulnerabilities (report them upstream); we can only bound what our layer sends.
- Missing headers on a service that is documented as loopback-only (`127.0.0.1`).

## How this project defends itself (read before reporting)

- **Retrieved text is data, not control flow.** No tool call, routing decision or policy bypass can
  originate from document content; the decision path consumes only guards, scores and configuration.
- **Three redaction boundaries:** before indexing, before logging/tracing, before any outbound
  provider request. The service tests assert absence of fixture PII in trace payloads and in captured
  provider bodies.
- **Citations are validated deterministically** (span resolution, verbatim quote, numeric guard), and
  a failing validation turns the turn into an escalation instead of an answer.
- **The offline profile never opens a socket**; tests fail on any outbound connection attempt.
- Fuzzing and hostile-input findings are catalogued in [`docs/robustness.md`](docs/robustness.md);
  the guard-boundary audit is in [`docs/audit-t24.md`](docs/audit-t24.md).

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes (current) |
| < 0.1 | no |
