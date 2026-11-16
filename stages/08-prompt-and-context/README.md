# Stage 08 — Prompt and Context

The model does not see your intent. It sees the tokens you put in the window, in the order you put them. This stage is about taking that sentence seriously: writing instructions that actually constrain behaviour, assembling context deliberately, versioning prompts like code, and proving with tests that a prompt still does what it did yesterday.

It sits right after the APIs because that is the moment you stop being impressed that a model answers at all, and start being annoyed that it answers differently on Tuesday.

**Time:** 2-3 weeks · **You need first:** [Stage 07 — LLM APIs](../07-llm-apis/README.md) · **You leave with:** a versioned prompt library with a test suite that runs in CI

## What you will be able to do

- Write a system instruction with explicit scope, refusal behaviour and output contract — and test it against adversarial input
- Choose between zero-shot, few-shot and chain-of-thought, and know when reasoning-out-loud makes an answer worse
- Build prompt templates with typed variables that fail loudly instead of silently rendering `{undefined}`
- Compress a long document into a token budget without losing the facts the question needs
- Version prompts as artefacts with ids, changelogs and rollback
- Defend an application against prompt injection from untrusted content
- Measure a prompt with assertions and rubrics instead of vibes
- Engineer a context window: what to include, what to drop, what order, and why position matters

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [System Instructions That Actually Constrain](01-system-instructions.md) | Scope, refusals and output contracts that hold under pressure |
| 2 | [Few-Shot Prompting and Chain of Thought](02-few-shot-cot.md) | Examples and reasoning, and when each one backfires |
| 3 | [Prompt Templates with Variables and Validation](03-templates.md) | Typed templates that fail loudly, not silently |
| 4 | [Context Compression](04-context-compression.md) | Fitting the right facts into a fixed token budget |
| 5 | [Prompt Versioning as Code](05-prompt-versioning.md) | Prompts as diffable, rollback-able artefacts |
| 6 | [Prompt-Injection Defence](06-injection-defense.md) | Separating data from instructions, and checking the output |
| 7 | [Prompt Evals](07-prompt-evals.md) | Assertions, rubrics and a pass rate you can gate on |
| 8 | [Context Engineering](08-context-engineering.md) | Budgeting, ordering and recency in a finite window |
| 9 | [Ship: A Prompt Test Suite](09-ship-prompt-test-suite.md) | A CI gate that fails when a prompt regresses |

## The shipping project

[Ship: A Prompt Test Suite](09-ship-prompt-test-suite.md) — you take the prompts from this stage, freeze them into fixtures with expected properties, add deterministic assertions and a rubric-based grader, and wire the whole thing into a single command that prints a scorecard and exits non-zero when quality drops. This is the artefact that lets you change a prompt on a Friday without fear.

## How to know you are done

- Your system instruction survives ten adversarial inputs you wrote yourself, including one injection attempt
- Every prompt in your app loads from a versioned file, not a string literal
- You can state your prompt's pass rate on a fixed fixture set, and you have watched it drop when you broke the prompt on purpose
- You can explain, with an example from your own testing, why the middle of a long context is a bad place for the instruction that matters most
- Your test suite runs in one command and fails the build on regression

---
[← Back to roadmap](../../README.md) · [Next stage: Vector Databases](../09-vector-databases/README.md)
