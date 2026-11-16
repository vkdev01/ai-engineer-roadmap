# 10. Ship: An Eval Suite

> **Stage 15 · Evals and Testing** — one command that runs deterministic tests, judge-graded cases and adversarial inputs, prints a scorecard, and exits non-zero on regression.
>
> **You'll be able to:** package everything in this stage into a suite you can hand to a colleague, and defend its numbers in a review.
>
> **Prerequisites:** [Regression Tests](08-regression-tests.md) · [Agent Evals](06-agent-evals.md) · [LLM as Judge](02-llm-as-judge.md)

## Why this exists

The pieces in this stage only pay off when they are one command. A suite spread across four scripts and a spreadsheet is a suite nobody runs on a Friday afternoon.

This project is the packaging discipline: a runner, a dataset, graders, thresholds, a baseline and a report. When you are done, "how do you know it works" has a one-line answer and a number attached.

The mental model: **one command, one scorecard, one exit code.**

## The mental model

```text
   python -m evals run
        │
        ├── tests/            pytest, deterministic, no model calls  ──┐
        ├── cases: golden     assertions + judge rubric              ──┤
        ├── cases: adversarial injection, leakage, off-scope         ──┤
        └── cases: agent      outcome + trajectory assertions        ──┘
                                    │
                              scorecard + baseline diff
                                    │
                             exit 0 (pass) / 1 (regression)
```

## Core concepts

### Three layers, in this order

1. **Deterministic** — parsing, chunking, routing, schema validation, citation-id checks. Fast, no model calls, never flaky. If these fail, stop; the rest is meaningless.
2. **Behavioural** — assertions plus judge grading on the golden dataset. Tolerant thresholds.
3. **Adversarial** — injection attempts, prompt-extraction attempts, off-scope requests, and refusal cases. These are pass/fail with no tolerance.

Running them in this order means the cheap failures surface first and you do not pay for a judge run on a system whose parser is broken.

### The scorecard is the deliverable

Print a fixed-format table so that reports from different weeks can be diffed. One row per metric, with the value, the threshold and the delta from baseline.

```text
$ python -m evals run --dataset evals/dataset.v3.jsonl

deterministic            42/42 passed          0.6s
behavioural
  pass_rate              0.847  (>= 0.82)   +0.012
  faithfulness           0.912  (>= 0.90)   +0.004
  citations_valid        1.000  (== 1.00)    0.000
  answer_relevance       0.881  (>= 0.85)   -0.003
adversarial
  injection_blocked      8/8    (== 8/8)     0.000
  refusal_correct        5/5    (== 5/5)     0.000
cost
  median_steps           3      (<= 4)       0
  cost_per_case          $0.0031 (<= $0.004) -$0.0002
RESULT: PASS
```

### Thresholds live in one file

Not in code, not in CI YAML. One `thresholds.toml`, reviewed in pull requests, so that loosening a threshold is a visible commit with a reason.

```toml
[behavioural]
pass_rate = 0.82
faithfulness = 0.90
citations_valid = 1.0
answer_relevance = 0.85

[adversarial]
injection_blocked = 1.0
refusal_correct = 1.0

[cost]
median_steps = 4
cost_per_case = 0.004
```

### What to do when a threshold is wrong

Sometimes a threshold is genuinely too strict — a new refusal rule legitimately lowers pass rate while improving safety. The process is: change the threshold in its own commit, with a sentence explaining what you traded. Never change a threshold inside a feature pull request; that is how gates rot.

## Code

The runner, as a small package.

```text
evals/
├── __init__.py
├── __main__.py          CLI: run / baseline / report
├── runner.py            orchestration, ordering, exit code
├── graders.py           deterministic assertions + judge calls
├── dataset.v3.jsonl     300 cases, versioned
├── thresholds.toml
├── baselines/latest.json
└── report.json          written by the last run
```

```python
# evals/runner.py
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from .graders import grade_case, load_cases, stable_subset


def run(dataset: str, limit: int | None = None) -> dict:
    thresholds = tomllib.loads(Path("evals/thresholds.toml").read_text())

    # layer 1: deterministic — bail out immediately if broken
    det = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], capture_output=True, text=True)
    if det.returncode != 0:
        print("DETERMINISTIC TESTS FAILED\n" + det.stdout[-2000:])
        raise SystemExit(1)

    cases = load_cases(dataset)
    if limit:
        cases = stable_subset(cases, limit)

    results = [grade_case(c) for c in cases]

    behavioural = [r for r in results if r["layer"] == "behavioural"]
    adversarial = [r for r in results if r["layer"] == "adversarial"]

    def mean(key: str, rows: list[dict]) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    metrics = {
        "pass_rate": round(sum(r["passed"] for r in behavioural) / max(len(behavioural), 1), 3),
        "faithfulness": mean("faithfulness", behavioural),
        "citations_valid": mean("citations_valid", behavioural),
        "answer_relevance": mean("relevance", behavioural),
        "injection_blocked": round(sum(r["passed"] for r in adversarial) / max(len(adversarial), 1), 3),
        "median_steps": sorted(r["steps"] for r in results)[len(results) // 2],
        "cost_per_case": round(sum(r["cost"] for r in results) / max(len(results), 1), 5),
    }

    failures = check(metrics, thresholds)
    report = {"dataset": Path(dataset).stem, "metrics": metrics, "failures": failures}
    Path("evals/report.json").write_text(json.dumps(report, indent=2))
    print_scorecard(metrics, thresholds, failures)
    return report


def check(metrics: dict, thresholds: dict) -> list[str]:
    flat = {**thresholds.get("behavioural", {}), **thresholds.get("adversarial", {}), **thresholds.get("cost", {})}
    failures = []
    for key, limit in flat.items():
        value = metrics.get(key)
        if value is None:
            continue
        if key in {"median_steps", "cost_per_case"}:
            if value > limit:
                failures.append(f"{key} {value} > {limit}")
        elif value < limit:
            failures.append(f"{key} {value} < {limit}")
    return failures
```

```python
# evals/__main__.py
import argparse
import sys

from .runner import run


def main() -> int:
    p = argparse.ArgumentParser(prog="evals")
    p.add_argument("command", choices=["run", "baseline"])
    p.add_argument("--dataset", default="evals/dataset.v3.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    report = run(args.dataset, args.limit)
    if args.command == "baseline":
        open("evals/baselines/latest.json", "w").write(__import__("json").dumps(report, indent=2))
        print("baseline updated")
        return 0
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

### The README section that sells it

> **How we know it works.** `python -m evals run` runs 42 deterministic tests, 300 behavioural cases and 13 adversarial cases, prints a scorecard and exits non-zero on regression. Current: pass rate 0.847, faithfulness 0.912, citations valid 1.000, injection blocked 8/8. The judge is calibrated at kappa 0.64 against 60 human-labelled cases. Cost per case $0.0031; the suite costs roughly $1 per full run and runs nightly, with a 100-case subset on pull requests.

That paragraph is worth more in a portfolio than any architecture diagram.

## Real use case

A team of four engineers ships an internal document assistant. Before the suite, "is it working" was answered by whoever had used it that morning; after, it is a table. The practical difference shows up in their third month, when a provider silently updates a model and faithfulness drops from 0.91 to 0.86.

Nobody notices from usage — the answers still look fine. The nightly run catches it, the gate blocks the next release, and the fix (pinning the model version, then re-evaluating the new one deliberately) takes an hour instead of a week of confused debugging.

That is the return on this project: **you stop being surprised.**

## Common mistakes

- **Mistake:** one script that runs everything with no ordering — **Instead:** deterministic first, and stop on failure.
- **Mistake:** thresholds scattered across code and CI — **Instead:** one file, reviewed, with changes in their own commit.
- **Mistake:** a report that is a wall of JSON — **Instead:** a fixed-format scorecard with deltas; JSON goes to the artefact store.
- **Mistake:** no adversarial cases in the suite — **Instead:** injection, extraction and refusal cases are pass/fail with no tolerance.
- **Mistake:** running the full suite on every push — **Instead:** subset on pull requests, full nightly, cached by prompt version.
- **Mistake:** a README that describes the architecture but not the score — **Instead:** lead with the numbers and the command that produces them.

## Checkpoint

1. Assemble the suite with all three layers and run it end to end. Record the scorecard and the total cost of one run.
2. Break a prompt on purpose and confirm the suite exits non-zero with a specific failing metric.
3. Write the "How we know it works" paragraph for your README, including the judge's calibration number and the cost per run.

**Self-check:** if you cannot state your suite's cost and runtime, you will not run it often enough — measure both and put them in the README next to the scores.

## Go deeper

- [Regression Tests](08-regression-tests.md) — the gate mechanics in detail
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — running this in a pipeline
- [Eval Platform](../22-real-projects/12-eval-platform.md) — the multi-project version

---
[← Previous: Online and Offline Evals](09-online-offline-evals.md) · [Stage index](README.md) · [Next stage: Inference](../16-inference/README.md)
