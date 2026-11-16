# 9. Ship: A Prompt Test Suite

> **Stage 08 · Prompt and Context** — fixtures, assertions and a CI gate that blocks prompt regressions.
>
> **You'll be able to:** ship a prompt test suite with recorded fixtures, layered assertions and a pass-rate gate wired into CI.
>
> **Prerequisites:** [Prompt Evals](07-prompt-evals.md) · [Prompt Versioning as Code](05-prompt-versioning.md)

## Why this exists

The previous files gave you the pieces: a template with typed variables, a versioned prompt file, a fixture set with assertions, and a gate. Shipped separately, they rot — the fixtures stop matching the template, the gate is commented out because it flaked once, and the recorded outputs go stale. The project is what turns them into one artefact a team actually uses.

The design constraint: **the suite must run in CI in under two minutes, offline by default, and fail loudly on a real regression.** Everything else — cost, speed, flake — follows from that.

The mental model: **fixtures on disk, a runner that either replays recorded outputs or calls the model, layered assertions, and a threshold that blocks merge.**

## The mental model

```text
tests/prompts/
  fixtures/*.json        input + expected properties + recorded output
  test_ticket.py         assertions, no network by default
  baseline.json          pass rate per prompt version
  conftest.py            record / replay switch

CI: pytest tests/prompts --replay
      |
      +-- pass rate >= floor
      +-- pass rate >= baseline - tolerance
      +-- every fixture's recorded output still matches the template hash
```

| Mode | Network | Speed | Used for |
|---|---|---|---|
| replay | no | seconds | every CI run and every local edit |
| record | yes | minutes | after a deliberate prompt change |
| live | yes | minutes | nightly drift check |

## Core concepts

### Record once, replay forever

Store each fixture's model output in the fixture file, keyed by prompt version and content hash. CI replays those outputs against your assertions: free, fast and deterministic. Re-record only when the prompt changes on purpose, and review the diff of recorded outputs in the pull request — that diff is the most informative review artefact you will get.

### Assertions run in layers

Layer 1 is structural and free: the output parses, the sentinel appears, required facts are present. Layer 2 is behavioural: length, forbidden phrases, no invented identifiers. Layer 3 is graded: a rubric judge, run only in the nightly live job because it costs money and is less stable. A CI gate should depend on layers 1 and 2 only.

### The gate needs two numbers

An absolute floor catches a badly broken prompt. A no-regression comparison against the baseline catches a small, quiet drop. Require both, and make the baseline update an explicit, reviewed commit rather than an automatic write — otherwise the gate drifts downward one convenient exception at a time.

### Stale fixtures are worse than no fixtures

If a template's content hash changes and the recorded outputs are not re-recorded, the suite is testing yesterday's prompt and passing. Assert the hash in the test: a mismatch is a failure telling you to re-record deliberately.

### Make the failure message actionable

When a fixture fails, print the fixture id, the assertion that failed, the expected property and the first 120 characters of the actual output. A CI log that says "assertion failed" wastes the ten minutes the suite was supposed to save.

## Code

A pytest suite with record/replay, layered assertions and the two-number gate.

```python
# tests/prompts/conftest.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def replay_enabled() -> bool:
    # default to replay: CI must never need an API key to run assertions
    return os.environ.get("PROMPT_TESTS_MODE", "replay") == "replay"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURES


@pytest.fixture
def mode() -> str:
    return "replay" if replay_enabled() else "live"
```

The test itself is deliberately boring — the assertions are the product.

```python
# tests/prompts/test_ticket.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.prompts import load_active          # your versioned prompt loader
from app.prompt_runner import run_prompt     # system+user -> text

FIXTURES = Path(__file__).parent / "fixtures"
FLOOR = 0.80
TOLERANCE = 0.05


def load_cases() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURES.glob("*.json"))]


def check(case: dict, answer: str) -> list[str]:
    """Layers 1 and 2 only: free, deterministic, CI-safe."""
    fails: list[str] = []
    exp = case["expect"]

    if "exact_prefix" in exp and not answer.strip().startswith(exp["exact_prefix"]):
        fails.append(f"expected prefix {exp['exact_prefix']!r}")
    for needle in exp.get("must_contain", []):
        if needle.lower() not in answer.lower():
            fails.append(f"missing {needle!r}")
    for needle in exp.get("must_not_contain", []):
        if needle.lower() in answer.lower():
            fails.append(f"forbidden {needle!r}")
    if "max_chars" in exp and len(answer) > exp["max_chars"]:
        fails.append(f"too long: {len(answer)} chars")
    return fails


@pytest.mark.asyncio
async def test_prompt_suite() -> None:
    prompt = load_active("ticket-answer")
    cases = load_cases()
    assert cases, "no fixtures found"

    failures: dict[str, list[str]] = {}
    for case in cases:
        # stale-fixture guard: a changed body must force a deliberate re-record
        if case.get("prompt_hash") not in (None, prompt.content_hash):
            failures[case["id"]] = [
                f"recorded against {case['prompt_hash']}, template is "
                f"{prompt.content_hash} - re-record deliberately"
            ]
            continue
        answer = await run_prompt(prompt, case["input"], mode=case.get("mode", "replay"))
        fails = check(case, answer)
        if fails:
            failures[case["id"]] = fails

    pass_rate = (len(cases) - len(failures)) / len(cases)

    # actionable failure output: id, what broke, what came back
    if failures:
        report = "\n".join(f"{k}: {'; '.join(v)}" for k, v in failures.items())
        print(f"pass rate {pass_rate:.2f}\nfailures:\n{report}")

    baseline = json.loads((Path(__file__).parent / "baseline.json").read_text())
    prev = baseline.get(prompt.version, FLOOR)

    assert pass_rate >= FLOOR, f"below floor: {pass_rate:.2f} < {FLOOR}"
    assert pass_rate >= prev - TOLERANCE, (
        f"regression vs {prompt.version} baseline {prev:.2f}: {pass_rate:.2f}"
    )
```

The recording script is separate and run by a human after a deliberate prompt change. It writes the fixture files, and their diff goes into the pull request.

```python
# scripts/record_prompts.py
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.prompts import load_active
from app.prompt_runner import run_prompt

FIXTURES = Path("tests/prompts/fixtures")


async def main() -> None:
    prompt = load_active("ticket-answer")
    for path in sorted(FIXTURES.glob("*.json")):
        case = json.loads(path.read_text())
        answer = await run_prompt(prompt, case["input"], mode="live")
        case["recorded"] = answer
        case["prompt_hash"] = prompt.content_hash
        case["prompt_version"] = prompt.version
        path.write_text(json.dumps(case, indent=2) + "\n")
        print(f"recorded {path.name}")


if __name__ == "__main__":
    asyncio.run(main())
```

The CI job is four lines of YAML and is the whole point of the exercise.

```yaml
name: prompt-tests
on: [pull_request]
jobs:
  prompts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: pytest tests/prompts -q          # replay mode, no API key needed
```

## Real use case

A team of four maintained twelve prompts for a document-intelligence product. Prompt edits were reviewed by reading the diff, and twice in one quarter a "harmless wording change" removed a constraint that a customer noticed before the team did.

They built this suite: 34 fixtures across the twelve prompts, recorded outputs committed alongside the prompt files, layers 1 and 2 in CI, and a nightly live job that added the rubric judge. Recording a deliberate change now produces a fixture diff showing exactly which answers changed, which turned prompt review from "read the words" into "read the behaviour change". The gate caught a regression in the first week: a shortened system prompt pushed one fixture below the floor. The failure mode before was review by intuition on a system whose behaviour was not observable until a customer found it.

## Common mistakes

- **Mistake:** requiring an API key in CI. — **Instead:** default to replay of recorded outputs; keep live runs for a nightly job.
- **Mistake:** letting recorded outputs go stale after a prompt edit. — **Instead:** assert the prompt content hash in the fixture and fail on mismatch.
- **Mistake:** updating the baseline automatically when the gate fails. — **Instead:** make baseline changes a reviewed commit, or the gate drifts down.
- **Mistake:** putting the LLM judge in the blocking gate. — **Instead:** gate on deterministic layers; run grading nightly and review trends.
- **Mistake:** failing with a bare assertion message. — **Instead:** print fixture id, failed property and the first part of the actual answer.

## Checkpoint

1. Add five fixtures to an existing prompt, record them, and confirm a replay run passes with no network access.
2. Break the prompt deliberately (remove the sentinel rule) and confirm the gate fails with the offending fixture ids printed.
3. Add a nightly live job that runs the same suite plus a rubric judge and writes a JSON report you can diff week to week.

**Self-check:** you replay recorded outputs keyed by prompt content hash, assert on deterministic properties, and require both an absolute floor and a no-regression comparison against the stored baseline.

## Go deeper

- [Prompt Evals](07-prompt-evals.md) — the assertion and grading techniques this project assembles.
- [Prompt Versioning as Code](05-prompt-versioning.md) — the version and hash the fixtures are keyed to.
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — the operational pipeline around this gate.
- [Regression Tests](../15-evals/08-regression-tests.md) — broader regression strategy beyond prompts.

---
[← Previous: Context Engineering](08-context-engineering.md) · [Stage index](README.md) · [Next: Stage 09 — Vector Databases](../09-vector-databases/README.md)
