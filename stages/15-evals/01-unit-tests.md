# 1. Unit Tests for LLM Code

> **Stage 15 · Evals and Testing** — ordinary software tests, applied to a system that is not deterministic.
>
> **You'll be able to:** separate the deterministic parts of an LLM pipeline from the model call, fake the model in tests, and use golden files to catch silent prompt regressions.
>
> **Prerequisites:** [5. Git and GitHub](../01-programming-basics/05-git-github.md) · [3. Templates](../08-prompt-and-context/03-templates.md)

## Why this exists

The reason people say "you cannot unit test LLM code" is that they are testing the wrong layer. The model's output is not deterministic, so asserting on it is a coin flip. But everything around the model is ordinary code: prompt construction, context assembly, chunk selection, JSON parsing, retry logic, token budgeting, error mapping. Those are deterministic, they are where most bugs live, and they are fully testable.

The trick is to make the boundary explicit. A function that builds a prompt should not also call the API. A function that parses a response should take a string, not a client. Once the model call sits behind a narrow interface, you can fake it — return a canned string, an error, a malformed JSON body — and test all the branches in milliseconds, with no network and no cost.

The mental model: **test the seams, fake the model. Behaviour of the model is measured by evals, not by asserts.**

## The mental model

```text
┌──────────── deterministic, unit-tested ─────────────┐   ┌─ evals ─┐
prompt build ──► context assembly ──► [ model call ] ──► parse ──► validate ──► post-process
                                          ▲
                                    faked in tests
```

| Layer | Test style | Speed |
|---|---|---|
| Prompt building, templating | exact string asserts, golden files | ms |
| Parsing, validation, retries | fakes returning canned/edge responses | ms |
| Retrieval assembly, budgets | fixtures, property checks | ms |
| Answer quality | evals with datasets and graders | seconds-minutes |

## Core concepts

### Put the model behind a protocol, then fake it

Define a tiny interface — one method, `complete(prompt) -> str` or a structured call — and have your business logic depend on that, not on a vendor SDK. In tests, pass a fake that returns a queue of prepared responses. This lets you test the interesting branches: the model returns prose instead of JSON, returns valid JSON with a missing field, times out, or returns an empty string. Every one of those is a real production incident you can now reproduce in a test.

### Golden files catch prompt drift

A golden file is the exact prompt your code produces for a fixed input, committed to the repo. When someone edits the template, the test fails and shows a diff. This is the single highest-value test in a prompt-heavy codebase, because prompt changes are the most common accidental regression: a reordered section, a dropped instruction, a variable that silently became `None`. Regenerate goldens deliberately — an update flag plus a reviewed diff — never automatically in CI.

### Assert on invariants, not on wording

For anything that touches generated text, assert properties rather than exact strings: the output parses as JSON; it contains exactly the required keys; every cited id exists in the retrieved set; the answer is under the token budget; the refusal message appears when context is empty. Invariants survive model upgrades and prompt rewording, and they catch the failures that matter. Exact-string assertions on model output are flaky tests waiting to be deleted.

### No network in unit tests, ever

A unit test that calls a real API is slow, costs money, and fails when the provider has a bad minute — which trains your team to ignore red builds. Enforce it: block sockets in the test session (`pytest-socket` or a fixture that patches the HTTP layer), and keep anything that needs a model in a separate, explicitly-marked eval suite. The rule "unit tests run offline in under a minute" is worth defending.

## Code

A pipeline split at the seam, with the model call behind a protocol. Note how little the business logic knows about the vendor.

```python
from __future__ import annotations

import json
from typing import Protocol

class Model(Protocol):
    async def complete(self, prompt: str, *, max_tokens: int = 512) -> str: ...

class SupportPipeline:
    def __init__(self, model: Model) -> None:
        self.model = model

    def build_prompt(self, question: str, context: list[str]) -> str:
        blocks = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context))
        return (
            "Answer using only the context. Cite block ids like [0].\n"
            f"Context:\n{blocks}\n\nQuestion: {question}"
        )

    async def run(self, question: str, context: list[str]) -> dict:
        raw = await self.model.complete(self.build_prompt(question, context))
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            return {"answer": raw.strip(), "citations": [], "degraded": True}
        return {"answer": rec.get("answer", ""),
                "citations": [c for c in rec.get("citations", []) if 0 <= c < len(context)],
                "degraded": False}
```

The tests: a fake model with a queue of responses, a golden-file check for the prompt, and invariant assertions for the output. Every branch here maps to a production failure.

```python
from pathlib import Path

import pytest

GOLDEN = Path("tests/golden/support_prompt.txt")

class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        return self.responses.pop(0)

@pytest.mark.asyncio
async def test_prompt_matches_golden():
    pipe = SupportPipeline(FakeModel([]))
    prompt = pipe.build_prompt("Where is order 4411?", ["Shipped yesterday.", "Due Thursday."])
    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(prompt)
    assert prompt == GOLDEN.read_text(), "prompt template changed; review the diff"

@pytest.mark.asyncio
async def test_non_json_response_degrades_without_raising():
    pipe = SupportPipeline(FakeModel(["It left the warehouse yesterday."]))
    out = await pipe.run("Where is 4411?", ["Shipped yesterday."])
    assert out["degraded"] is True and out["answer"]

@pytest.mark.asyncio
async def test_out_of_range_citations_are_dropped():
    pipe = SupportPipeline(FakeModel(['{"answer": "Thursday", "citations": [0, 7]}']))
    out = await pipe.run("When?", ["Due Thursday."])
    assert out["citations"] == [0]        # invariant: only real blocks survive
```

A property-style check that survives prompt rewording: whatever the model says, the pipeline never emits a citation the model could not have seen, and never emits an empty answer.

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    '{"answer": "ok", "citations": [0]}',
    '{"answer": "", "citations": [0]}',
    '{"citations": [1]}',
    'not json at all',
])
async def test_invariants_hold_for_any_model_output(raw: str):
    out = await SupportPipeline(FakeModel([raw])).run("q", ["ctx one", "ctx two"])
    assert isinstance(out["answer"], str)
    assert all(0 <= c < 2 for c in out["citations"])
```

## Real use case

A document-QA team shipped a prompt refactor — they reordered the instructions so the citation rule came last — and answers stopped including citations. Nobody noticed for three days, because answers were still correct and fluent; only the citation links were gone, so reviewers had nothing to click.

They added two tests in an afternoon. A golden file pins the assembled prompt, so any template edit shows up as a reviewed diff. And a fake model with four canned responses covers the parse branches: valid JSON, JSON without citations, out-of-range citations, and plain prose. The refactor that caused the incident now fails in CI in under a second with a diff pointing at the moved line. The team's rule became: the prompt is code, so the prompt has a test.

## Common mistakes

- **Mistake:** asserting on a real model's exact output — **Instead:** assert invariants, and measure quality in the eval suite.
- **Mistake:** letting unit tests hit the network — **Instead:** block sockets in the session and keep model calls behind a fake.
- **Mistake:** no golden file for the assembled prompt — **Instead:** commit the exact prompt for a fixed input and review diffs on change.
- **Mistake:** testing only the happy path — **Instead:** fake malformed JSON, missing keys, empty strings and timeouts; those are your incidents.
- **Mistake:** regenerating goldens automatically in CI — **Instead:** an explicit update flag plus a human-reviewed diff.

## Checkpoint

1. Refactor one function so the model call sits behind a protocol, then write a fake and test three branches.
2. Add a golden-file test for your prompt assembly and confirm it fails when you reorder one instruction.
3. Add socket blocking to your test session and count how many tests break — those were the slow, flaky ones.

**Self-check:** the model call is the only non-deterministic part, so putting it behind a narrow interface lets you fake every branch of the surrounding code and keep unit tests offline and fast.

## Go deeper

- Next: [2. LLM-as-Judge, Done Properly](02-llm-as-judge.md) — measuring what unit tests cannot.
- Prompt hygiene: [5. Prompt Versioning](../08-prompt-and-context/05-prompt-versioning.md).
- Wiring this into CI: [8. Regression Tests and CI Gates](08-regression-tests.md).

---
[← Previous: Stage 14 — Multimodal](../14-multimodal/README.md) · [Stage index](README.md) · [Next: 2. LLM-as-Judge, Done Properly](02-llm-as-judge.md)
