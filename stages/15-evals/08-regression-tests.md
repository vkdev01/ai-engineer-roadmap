# 8. Regression Tests

> **Stage 15 · Evals and Testing** — putting a score behind a gate so a prompt change on Friday cannot silently make Monday worse.
>
> **You will be able to:** build a baseline, compare runs against it, handle flaky cases, and design a CI gate that catches regressions without blocking every pull request.
>
> **Prerequisites:** [Golden Datasets](03-golden-datasets.md) · [Agent Evals](06-agent-evals.md) · [Unit Tests for LLM Code](01-unit-tests.md)

## Why this exists

An LLM system has no compiler. Nothing stops you merging a prompt edit that drops faithfulness by six points, and nobody notices until a customer does. Regression testing is how you buy back the safety net that typed languages give you for free.

The hard part is not running the tests. It is choosing a threshold that catches real regressions without crying wolf every time a provider updates a model, because a gate that fails randomly gets disabled, and a disabled gate is worse than none.

The mental model: **a baseline plus a tolerance, enforced on every change, with an escape hatch that requires a written reason.**

## The mental model

```text
   change ──► run suite (300 stratified cases)
                 │
                 ├─ deterministic tests fail? ──► block (no debate)
                 ├─ score < baseline - tolerance? ──► block (investigate)
                 ├─ cost or steps up > X%? ──► block (performance regression)
                 └─ otherwise ──► merge, update baseline on main
```

Three gates, three different meanings. Deterministic tests are binary; scores have tolerance; cost has a budget. Mixing them into one number is how gates become unmaintainable.

## Core concepts

### Baselines are stored artefacts, not memories

Store the last accepted report as JSON in the repo, keyed by prompt version and model id. The comparison is then a diff between two files, which is reviewable in a pull request.

```json
{
  "prompt_version": "support-v7",
  "model": "gpt-4o-mini",
  "dataset": "dataset.v3",
  "run_at": "2026-11-02",
  "metrics": {
    "pass_rate": 0.847,
    "faithfulness": 0.912,
    "citations_valid": 1.0,
    "median_steps": 3,
    "cost_per_case": 0.0031
  }
}
```

### Tolerance, chosen from observed variance

Run the same suite twice on an unchanged system and you will get different numbers. Measure that noise first: run three times, take the spread, and set tolerance a little wider. If your pass rate wobbles by ±2 points run to run, a 1-point gate will fail constantly.

```python
def compare(current: dict, baseline: dict, tolerance: dict[str, float]) -> list[str]:
    """Return a list of regressions. Empty list means safe to merge."""
    problems: list[str] = []
    for key, tol in tolerance.items():
        cur = current["metrics"].get(key)
        base = baseline["metrics"].get(key)
        if cur is None or base is None:
            continue
        if key in {"median_steps", "cost_per_case"}:
            if cur > base * (1 + tol):
                problems.append(f"{key} regressed: {cur} > {base} (+{tol:.0%} allowed)")
        elif cur < base - tol:
            problems.append(f"{key} regressed: {cur} < {base} (-{tol} allowed)")
    return problems
```

### Flaky cases are findings, not noise

A case that passes two runs in three is telling you the system is non-deterministic in a way users will experience. Do not delete it. Either pin it (temperature 0, fixed seed where the provider supports it) or record it as a known-flaky case with an expected pass rate and exclude it from the gate while it is tracked.

```python
def stable_cases(history: list[dict[str, bool]], min_runs: int = 3, min_pass: float = 0.99) -> set[str]:
    """Cases that passed every recorded run — the ones safe to gate on."""
    counts: dict[str, list[bool]] = {}
    for run in history:
        for cid, ok in run.items():
            counts.setdefault(cid, []).append(ok)
    return {
        cid for cid, results in counts.items()
        if len(results) >= min_runs and sum(results) / len(results) >= min_pass
    }
```

### The escape hatch that keeps the gate alive

Sometimes a change intentionally trades a metric: a stricter refusal rule lowers pass rate but prevents a compliance risk. Allow an override in the PR description — `EVAL-OVERRIDE: faithfulness -0.03, reason: ...` — and make CI print the override in the summary. The gate stays trusted because the exceptions are visible and argued.

## Code

A gate script that CI can call, plus the workflow that runs it.

```python
# evals/gate.py
from __future__ import annotations

import json
import os
import re
import sys

TOLERANCE = {"pass_rate": 0.03, "faithfulness": 0.02, "citations_valid": 0.0,
             "median_steps": 0.25, "cost_per_case": 0.30}


def main() -> int:
    current = json.load(open("evals/report.json"))
    baseline = json.load(open("evals/baselines/latest.json"))

    if current.get("dataset") != baseline.get("dataset"):
        print(f"dataset changed ({baseline.get('dataset')} -> {current.get('dataset')}): "
              "re-baseline deliberately, do not compare")
        return 2

    problems = compare(current, baseline, TOLERANCE)

    override = os.environ.get("EVAL_OVERRIDE", "")
    if override:
        print(f"OVERRIDE in effect: {override}")
        waived = re.findall(r"([a-z_]+)\s*-?[\d.]+", override)
        problems = [p for p in problems if not any(p.startswith(w) for w in waived)]

    if problems:
        print("REGRESSIONS:")
        for p in problems:
            print("  -", p)
        print("\nFix, or add EVAL-OVERRIDE with a reason in the PR description.")
        return 1

    print("no regressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```yaml
# .github/workflows/evals.yml
name: evals
on:
  pull_request:
    paths: ["prompts/**", "src/**", "evals/**"]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e . && pip install pytest
      - name: deterministic tests
        run: pytest -q tests/            # these must never be flaky
      - name: eval suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          EVAL_OVERRIDE: ${{ github.event.pull_request.body }}
        run: python evals/run.py --out evals/report.json
      - name: gate
        run: python evals/gate.py
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-report
          path: evals/report.json
```

Note the two-stage split: `pytest` for deterministic logic (parsing, chunking, routing, schema), the eval suite for model behaviour. The first must be fast and never flaky; the second is allowed to be slow and is where tolerance applies.

### Cost of the gate

A 300-case suite with a small judge is cheap per run but not free, and it runs on every pull request. Two practical economies: gate on 100 cases for pull requests and run the full 300 nightly; and cache results keyed by prompt version plus dataset hash so re-runs of an unchanged prompt cost nothing.

## Real use case

A team of six engineers ships prompt changes weekly. Their first gate is a hard threshold on pass rate and it fails on roughly half of all pull requests, mostly from run-to-run noise. Within a month, people start adding `--no-verify`-style overrides by default and the gate is decorative.

They fix it in three moves. They measure variance (three identical runs) and set tolerance at twice the observed spread. They split the gate into deterministic tests (blocking, no tolerance) and scored evals (tolerance, override required). And they reduce the pull-request suite to 100 stratified cases, keeping the full set nightly.

Result: the gate fails a few times a month, and every failure is a real regression. That ratio — signal, not noise — is the only thing that keeps a gate respected.

## Common mistakes

- **Mistake:** a hard threshold with no tolerance — **Instead:** measure run-to-run variance and set tolerance at roughly twice it.
- **Mistake:** comparing runs against different datasets — **Instead:** store the dataset hash in the report and refuse to compare across versions.
- **Mistake:** mixing deterministic tests into the scored suite — **Instead:** keep binary tests binary; they block without debate.
- **Mistake:** deleting flaky cases — **Instead:** track them; flakiness is a user-visible property of your system.
- **Mistake:** no override path — **Instead:** allow a written override in the PR and print it in the summary.
- **Mistake:** running the full suite on every push — **Instead:** 100 on pull requests, 300 nightly, cached by prompt version.

## Checkpoint

1. Run your suite three times unchanged and record the spread of each metric. Set tolerance from that spread and write the numbers down.
2. Build `gate.py` and confirm it fails when you deliberately weaken a prompt, and passes when you revert.
3. Add the GitHub Actions workflow and open a pull request that trips the gate. Then add an override and confirm it is printed and honoured.

**Self-check:** if your gate has never failed on a real regression, it is probably too loose — break a prompt on purpose and check it notices.

## Go deeper

- [Online and Offline Evals](09-online-offline-evals.md) — what production traffic tells you that the gate cannot
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — the MLOps view of the same pipeline
- [Ship: An Eval Suite](10-ship-eval-suite.md) — the whole suite, packaged

---
[← Previous: Human Evals](07-human-evals.md) · [Stage index](README.md) · [Next: Online and Offline Evals](09-online-offline-evals.md)
