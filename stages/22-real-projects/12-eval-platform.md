# 12. Eval Platform

> **Stage 22 · Real Projects** — datasets, graders, runs and a dashboard, so any prompt or model change gets a verdict instead of an opinion.
>
> **You'll be able to:** build the internal tool that makes quality measurable: versioned datasets, composable graders, judge calibration, run comparison with an honest confidence interval, and a regression gate.
>
> **Prerequisites:** [LLM as Judge](../15-evals/02-llm-as-judge.md) · [Golden Datasets](../15-evals/03-golden-datasets.md)

## Why this exists

Every team starts with a notebook that calls the model on ten examples and prints results. It works until the second person, the second prompt, or the second week — then nobody remembers which examples were used, the judge prompt drifted, and "it feels better" becomes the release criterion.

An eval platform is the least glamorous project in this stage and the one that most distinguishes a senior engineer. It is also the project most likely to be adopted at a company where you intern, because the gap it fills is universal.

Buyers are the team itself, then the org. The value is not dashboards; it is that a decision becomes a number with a range, and a regression gets caught before a customer sees it.

Mental model: an eval run is a *repeatable experiment* — fixed dataset version, fixed prompt version, fixed graders, recorded model ids — and the output is a comparison with uncertainty, not a single score.

## The mental model

```text
dataset (versioned, frozen)
      |
      +---- case: input, expected (or rubric), tags, slice
      |
[runner] --prompt version + model id --> outputs (cached by hash)
      |
[graders] --deterministic--> exact/JSON/schema/regex
      |----judge----------> rubric -> structured verdict + reason
      |----human----------> spot-check labels
      v
[run: per-case scores + metadata + cost]
      |
[compare: vs baseline, bootstrap CI, per-slice deltas]
      |
[gate: pass/fail in CI]  -->  [dashboard: trend by version]
```

## Core concepts

### Scope and non-goals

In scope: dataset CRUD with immutable versions, a runner with caching, deterministic and judge graders, run comparison with confidence intervals, per-slice reporting, a dashboard, and a CI gate.

Non-goals: annotation tooling for large teams, model training integration, and A/B testing in production (that is [Online vs Offline](../15-evals/09-online-offline-evals.md)). Also non-goal: a universal quality score — a single number invites gaming and hides slices.

### Datasets are immutable versions, not files

A dataset has a name, a version, and a content hash. Editing cases creates a new version; runs reference the version they used, forever. Each case carries `input`, optional `expected`, an optional `rubric`, `tags` for slicing, and a `source` so you know if it came from production or was written by hand. Freeze a held-out slice that you look at rarely — if you tune against it, it stops measuring anything.

### Graders are small, composable, and testable

Write graders as functions with a uniform signature returning `{score, passed, detail}`. Deterministic graders first: exact match, JSON schema validity, required-field presence, regex, numeric tolerance, latency budget. They are free, fast and never drift. Use a judge only where correctness is semantic, and always pair a judge with a deterministic pre-check — if the output is not valid JSON, do not ask a judge whether it is good JSON.

Judges need calibration, which is the step almost everyone skips: take a sample of cases, label them yourself, and measure agreement between your labels and the judge's verdicts. If agreement is poor, the rubric is ambiguous — fix the rubric, not the sample. Record the judge model id per run, because a judge upgrade silently changes historical comparability.

### Comparison needs uncertainty

A run on 40 cases cannot distinguish a two-point difference from noise. Compute a bootstrap confidence interval over per-case scores, and report the delta with its interval. Two rules that follow: never declare a winner when the interval spans zero, and prefer paired comparison (same cases, both versions) over comparing to a stored baseline number from another day.

### Milestones, data model, budget

Four weeks: week 1 dataset model and runner with caching; week 2 deterministic graders and the run store; week 3 judge plus calibration; week 4 comparison, CI gate, dashboard.

| table | key fields |
|-------|-----------|
| `dataset` | id, name, version, content_hash, created_at |
| `case` | id, dataset_id, input, expected, rubric, tags[], source |
| `run` | id, dataset_id, prompt_version, model_id, judge_model, git_sha, cost_usd |
| `result` | id, run_id, case_id, output, scores_json, passed |
| `baseline` | id, dataset_id, run_id, promoted_at, promoted_by |

Cost, illustrative: eval cost is (cases × graders) model calls. A judge call per case at 1k input and 200 output tokens is the dominant term; cache outputs by (input, prompt version, model id) so re-scoring with a new grader does not re-generate. Run the full suite nightly and a smoke subset (10-20 cases) on every pull request, or nobody will wait for CI.

## Code

The grader protocol plus a judge that must return a structured verdict keeps scores comparable:

```python
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class Grade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: str = ""


class Grader(Protocol):
    name: str
    def __call__(self, case: dict, output: str) -> Grade: ...


def json_validity(case: dict, output: str) -> Grade:
    import json
    try:
        json.loads(output)
        return Grade(score=1.0, passed=True, detail="parsed")
    except json.JSONDecodeError as exc:
        return Grade(score=0.0, passed=False, detail=f"invalid json: {exc.msg}")


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "partial", "fail"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


async def judge(case: dict, output: str, client, judge_model: str) -> Grade:
    """Deterministic pre-check first: never ask a judge about unparseable output."""
    if case.get("requires_json") and not json_validity(case, output).passed:
        return Grade(score=0.0, passed=False, detail="pre-check failed: invalid json")

    resp = await client.messages.create(
        model=judge_model,          # record this id on the run; upgrades break comparability
        max_tokens=300,
        messages=[{"role": "user", "content": (
            f"RUBRIC:\n{case['rubric']}\n\nINPUT:\n{case['input']}\n\n"
            f"ANSWER:\n{output}\n\nReturn JSON with verdict and reason."
        )}],
    )
    data = parse_json(resp.text, VERDICT_SCHEMA)
    score = {"pass": 1.0, "partial": 0.5, "fail": 0.0}[data["verdict"]]
    return Grade(score=score, passed=data["verdict"] == "pass", detail=data["reason"])
```

And the comparison, which is the difference between an eval tool and a score printer:

```python
import random

def bootstrap_ci(deltas: list[float], n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Paired per-case deltas -> interval for the mean difference."""
    means = []
    for _ in range(n):
        sample = [random.choice(deltas) for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return lo, hi


def verdict(base: list[float], new: list[float]) -> str:
    assert len(base) == len(new), "paired comparison needs the same cases"
    deltas = [n - b for b, n in zip(base, new)]
    lo, hi = bootstrap_ci(deltas)
    if lo > 0:
        return f"improved (+{sum(deltas)/len(deltas):.3f}, CI {lo:.3f}..{hi:.3f})"
    if hi < 0:
        return f"regressed ({sum(deltas)/len(deltas):.3f}, CI {lo:.3f}..{hi:.3f})"
    return f"no detectable difference (CI {lo:.3f}..{hi:.3f})"
```

## Real use case

A product team ships prompt changes weekly across four features. Before: each change was reviewed by reading a handful of outputs, and two regressions reached customers — one made refusals too eager, another broke JSON output on a rare label.

After: every feature has a dataset with a frozen slice, a smoke subset runs on each pull request, and the dashboard shows per-slice trends. The gate is a paired comparison with an interval, so a "no detectable difference" result is a legitimate outcome that unblocks a refactor.

The design choice that made it work is pairing and intervals: it removed the argument about whether a small drop was real, which was where review time went.

Demo script, two minutes: show a dataset version and a run, run a smoke subset against a deliberately worse prompt, show the interval verdict, then the per-slice breakdown that localises the regression. Talking points: why datasets are immutable, how you calibrated the judge, what a confidence interval changes about decisions, and the cost of the nightly suite.

## Common mistakes

**Mistake:** editing a dataset in place — **Instead:** new version, runs pinned to versions.
**Mistake:** one composite score — **Instead:** per-slice scores plus a small set of named metrics.
**Mistake:** trusting a judge without calibration — **Instead:** label a sample, measure agreement, fix the rubric.
**Mistake:** declaring winners on tiny differences — **Instead:** bootstrap intervals and paired comparisons.
**Mistake:** re-generating outputs to try a new grader — **Instead:** cache outputs by input and model, re-score only.
**Mistake:** running the full suite on every pull request — **Instead:** a smoke subset in CI, full suite nightly.

## Checkpoint

1. Build a grader that checks a numeric field within a tolerance, and a case where it should fail. Confirm the score is 0.
2. Calibrate your judge on 20 hand-labelled cases and compute agreement. Rewrite one rubric clause and re-measure.
3. Add per-slice reporting to the comparison and find a slice where the "improved" version is worse. What do you ship?

**Self-check:** Exercise 1 fails as expected because tolerance graders must compare parsed numbers, not strings — string comparison makes `1.0` and `1.00` different answers.

## Go deeper

- [Faithfulness](../15-evals/04-faithfulness.md) and [Hallucinations](../15-evals/05-hallucinations.md) — the graders this platform is usually built for
- [Regression Tests](../15-evals/08-regression-tests.md) — turning the gate into a habit
- [Fine-Tuning Pipeline](11-fine-tuning-pipeline.md) — a consumer of this platform
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — wiring the gate into GitHub Actions

---

[← Previous: Fine-Tuning Pipeline](11-fine-tuning-pipeline.md) · [Stage index](README.md) · [Next: AI SaaS](13-ai-saas.md)
