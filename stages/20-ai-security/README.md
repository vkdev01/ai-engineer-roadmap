# Stage 20 — AI Security

An AI application is a program whose input is text written by an untrusted stranger, whose output is executed by your tools, and whose context is filled with documents you did not write. This stage teaches you to design that system so that a hostile input is a handled case, not an incident. It sits after you can already build and deploy an LLM app (stages 07–19) and before you design one at scale (stage 21).

**Time:** 12–18 hours · **You need first:** Stage 08 (Prompt and Context) and Stage 17 (AI Backend) · **You leave with:** a CI-gated injection test suite and a threat model for your own app

## What you will be able to do

- Explain the difference between direct and indirect prompt injection, and show why "ignore previous instructions" is the weakest possible attack case.
- Separate data from instructions using structural means — instruction hierarchy, capability narrowing, output-side validation — rather than wording tricks.
- Red-team your own application with a repeatable corpus, and define a refusal boundary and acceptable-use policy you can defend.
- Detect and redact PII with regex plus NER, and choose the right point in the pipeline to redact.
- Sandbox tool execution so that a compromised model cannot touch the shell, the network or the filesystem beyond what a task needs.
- Reason about model and package provenance, licence checks and untrusted model files.
- Measure disparity across groups in data, prompts, retrieval and evaluation, and write the governance artefacts (DPIA-style note, audit trail, model card) that a reviewer will ask for.
- Ship an injection test suite that fails CI when a defence regresses.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Prompt Injection](01-prompt-injection.md) | Direct vs indirect attacks, and structural defences that hold |
| 2 | [Jailbreaks](02-jailbreaks.md) | What refusal training misses, and how to red-team your own app |
| 3 | [Data Leakage](03-data-leakage.md) | System prompt extraction, tenant bleed, over-broad context, PII in logs |
| 4 | [PII Redaction and Detection](04-pii-redaction.md) | Regex plus NER, pseudonymisation, and where to redact |
| 5 | [Sandboxing Tool Execution](05-sandboxing.md) | Allowlists, filesystem and network limits, timeouts, resource caps |
| 6 | [Supply Chain Security](06-supply-chain.md) | Model provenance, pinned versions, licences, untrusted weights |
| 7 | [Bias and Fairness](07-bias-fairness.md) | Where bias enters an LLM app, and how to measure disparity |
| 8 | [Governance](08-governance.md) | Data residency, retention, audit trails, model cards, DPDP-style duties |
| 9 | [Guardrails](09-guardrails.md) | Layered filters, schema constraints, approval gates, and their limits |
| 10 | [Ship: Injection Test Suite](10-ship-injection-tests.md) | A corpus-driven suite that fails CI on a defence regression |

## The shipping project

[10-ship-injection-tests.md](10-ship-injection-tests.md) — you build a test harness that replays a corpus of labelled attack cases against your own app's public entry point, asserts on behaviour (not on exact strings), and exits non-zero in CI. It proves the defences from topics 1, 3, 5 and 9 still hold after every prompt or model change.

## How to know you are done

- You can state, for your own app, which inputs are trusted, which are untrusted, and which tools each untrusted path can reach.
- Your app has at least three independent defences against injection, and you can name the failure mode each one covers.
- Your test suite runs the attack corpus and fails on a deliberately weakened defence (you have tried breaking it).
- You can answer "what happens to this data at rest, for how long, and who can read it" without guessing.
- You have written a one-page governance note for a real feature, and you have labelled it as engineering input, not legal advice.

---
[← Back to roadmap](../../README.md) · [Next stage: System Design](../21-system-design/README.md)
