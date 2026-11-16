# 7. CI/CD for Evals

> **Stage 18 · MLOps and LLMOps** — making the eval suite a gate that blocks bad prompt changes, instead of a script someone remembers to run.
>
> **You'll be able to:** put an eval job in a pull request, set regression thresholds that catch real drops without crying wolf, and handle flaky LLM-judged tests honestly.
>
> **Prerequisites:** [Regression Tests](../15-evals/08-regression-tests.md) · [DVC for Data and Models](04-dvc.md)

## Why this exists

You have a golden set and a judge. Someone edits a system prompt to fix one complaint, runs the eval locally, sees 0.79 versus 0.78, and merges. Two weeks later a customer reports that the assistant stopped citing sources, and you discover the new prompt quietly dropped citation instructions — a regression that a full eval run would have caught, on a metric nobody looked at because the mean score went up.

The mean is the problem. A single averaged score hides which examples broke. CI for evals exists to run the full suite on every change, compare per-metric against a baseline, and refuse the merge when something that used to pass now fails.

The mental model: **the eval suite is your test suite, and a PR that lowers quality is a failing test, not a judgement call.**

## The mental model

```text
PR opened  ->  ci/evals job
                 |-- dvc pull golden set (hash-pinned)
                 |-- run evals on candidate prompt  -> report.json
                 |-- fetch baseline report for main  -> baseline.json
                 |-- diff per metric + per example
                 `-- gate: pass / warn / fail
                        fail -> block merge, comment with failing examples
                        warn -> allow merge, note the flaky metric
```

```text
metric            baseline   candidate   delta    gate
faithfulness        0.81       0.72      -0.09    FAIL  (> 0.03 drop)
citation_present    0.98       0.41      -0.57    FAIL  (> 0.05 drop)
mean_latency_ms     1180       1090       -90     ok
judge_agreement     0.86       0.85      -0.01    WARN  (flaky, n=240)
```

## Core concepts

### Compare against a baseline, not against zero

An absolute threshold ("fail if faithfulness < 0.75") sounds reasonable and is wrong: your quality drifts upward over months, and a fixed floor stops protecting you. Compare **candidate against the current main baseline** for the same dataset hash. If the dataset changed, the comparison is invalid — fail the job and tell the author to rebase or to bump the baseline deliberately.

Store baselines as committed artefacts (`evals/baseline.json`) updated by a workflow on merge to main. That keeps the baseline reviewable in git and immune to "which run was main?" ambiguity.

### Per-metric thresholds, and per-example diffs

Aggregate thresholds are necessary but blunt. Set a tolerance per metric based on how noisy it is: a deterministic metric like `citation_present` deserves a tight tolerance (0.05); a judge-scored metric with 240 examples needs a looser one because sampling noise alone moves it.

Then show the diffs. A failing gate that says "faithfulness -0.09" makes the author hunt. A gate that comments the six examples which flipped from pass to fail, with the old and new answer, makes the fix obvious in one read.

### Flaky tests: detect, quarantine, do not ignore

LLM evals are flaky by nature: temperature above zero, judge model variance, retrieval ordering ties. Three responses, in order of preference:

- **Reduce the flakiness.** Set temperature to 0 for eval runs, pin the judge model and judge prompt version, fix the random seed where the SDK allows it, and use a fixed retrieval index snapshot.
- **Quarantine the metric.** If a metric's variance across repeated identical runs is larger than its threshold, it is not a gate, it is noise. Run it three times, take the median, and widen the tolerance. Record the measured variance in the workflow so the number is justified.
- **Never silently retry until green.** A job that retries a failing eval until it passes trains everyone to ignore red builds.

A useful diagnostic: run the same prompt against the same dataset twice in CI and diff the two reports. Whatever differs is your noise floor. Set thresholds above it, not below.

### Cost and time of the gate

An eval job that costs a lot per PR and takes 20 minutes will be disabled within a month. Two-tier it: a **smoke tier** (30 examples, deterministic metrics, under two minutes) on every push, and a **full tier** (240+ examples, judge-based) on PRs to main and nightly. The smoke tier catches most breakage; the full tier catches drift.

Also: cache the judge calls. If a prompt change does not alter the rendered prompt for an example, its judged score cannot change — key the cache on the rendered prompt hash plus judge version and skip the call. On a 240-example set where 200 prompts are unchanged, that is most of the bill gone.

## Code

The gate script. It reads two reports, applies per-metric tolerances, collects the flipped examples, and exits non-zero on failure so the CI job blocks. The tolerance table is data, so widening a flaky threshold is a reviewable diff.

```python
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricRule:
    tolerance: float   # max allowed drop vs baseline
    flaky: bool = False


RULES: dict[str, MetricRule] = {
    "faithfulness": MetricRule(tolerance=0.03),
    "citation_present": MetricRule(tolerance=0.05),
    "answer_correct": MetricRule(tolerance=0.04),
    "judge_agreement": MetricRule(tolerance=0.06, flaky=True),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def gate(baseline: dict, candidate: dict) -> tuple[bool, list[str], list[dict]]:
    if baseline["dataset_hash"] != candidate["dataset_hash"]:
        return False, ["dataset hash differs: rebase or bump the baseline deliberately"], []

    lines: list[str] = []
    failed = False
    for metric, rule in RULES.items():
        base = baseline["metrics"].get(metric)
        cand = candidate["metrics"].get(metric)
        if base is None or cand is None:
            continue
        delta = cand - base
        status = "ok"
        if delta < -rule.tolerance:
            status = "WARN" if rule.flaky else "FAIL"
            failed = failed or status == "FAIL"
        lines.append(f"{metric}: {base:.3f} -> {cand:.3f} ({delta:+.3f}) {status}")

    # Per-example flips are what make a failure actionable.
    base_pass = {r["id"] for r in baseline["rows"] if r["passed"]}
    flipped = [
        {"id": r["id"], "question": r["question"], "old": r.get("old_answer"), "new": r["answer"]}
        for r in candidate["rows"]
        if r["id"] in base_pass and not r["passed"]
    ]
    return not failed, lines, flipped


if __name__ == "__main__":
    ok, lines, flipped = gate(load(Path(sys.argv[1])), load(Path(sys.argv[2])))
    print("\n".join(lines))
    if flipped:
        print(f"\n{len(flipped)} examples regressed:")
        for f in flipped[:6]:
            print(f"  - [{f['id']}] {f['question']}")
    sys.exit(0 if ok else 1)
```

The workflow that runs it. Note the two tiers, the DVC pull with a hash-pinned dataset, and that the job writes a comment-friendly artefact rather than relying on logs.

```yaml
# .github/workflows/evals.yml
name: evals
on:
  pull_request:
    paths: ["prompts/**", "src/**", "dvc.lock"]
  schedule:
    - cron: "0 2 * * *"   # nightly full tier

jobs:
  smoke:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: dvc pull --no-run-cache
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.DVC_AWS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.DVC_AWS_SECRET }}
      - name: Run smoke evals
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m src.evals --tier smoke --out reports/candidate.json
      - name: Gate
        run: python -m src.gate evals/baseline.json reports/candidate.json
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-report
          path: reports/candidate.json
```

## Real use case

A documentation assistant has 240 golden questions and a faithfulness judge. The team's first attempt at gating used one mean score with a 0.02 tolerance. It failed on almost every PR because the judge itself varied by about 0.03 between identical runs, so the team set the gate to warn-only and then ignored it.

The second attempt did three things. It pinned temperature to 0 and the judge prompt to a version, which cut variance to about 0.01. It split metrics, so `citation_present` — a deterministic string check — became a hard gate with a tight tolerance. And it printed the flipped examples in the PR comment. The first week the tight citation gate fired on a prompt refactor that had removed the "cite the section id" instruction; the author saw eleven flipped examples in the comment and fixed it in ten minutes. Before the gate existed, that same class of change had shipped twice and produced two customer escalations.

## Common mistakes

**Mistake:** a single mean score as the gate — **Instead:** per-metric tolerances, with deterministic metrics gated tightly and judge metrics loosely.

**Mistake:** comparing against a hardcoded floor — **Instead:** compare against the committed baseline for the same dataset hash.

**Mistake:** running the full judge suite on every push — **Instead:** a fast smoke tier on push, the full tier on PRs to main and nightly.

**Mistake:** auto-retrying a red eval job until green — **Instead:** measure the noise floor by running the same prompt twice, then set thresholds above it and quarantine genuinely noisy metrics.

**Mistake:** no caching of judge calls — **Instead:** key the judge cache on rendered prompt hash plus judge version, so unchanged examples cost nothing.

**Mistake:** a gate nobody can read — **Instead:** post the metric table and the first six flipped examples as a PR comment.

## Checkpoint

1. Write the gate script against two hand-edited report JSONs — one that should pass, one that should fail — and prove both exit codes.
2. Wire the smoke tier into your repo's PR workflow with DVC pull, and open a PR that deliberately removes a citation instruction. Confirm the gate blocks it and the flipped examples appear.
3. Measure your noise floor: run the same prompt twice through the judge, diff the reports, and set each tolerance above the observed difference. Document the numbers in the workflow file.

**Self-check:** exercise 1 is done when the failing report exits with code 1 and prints a metric line showing a negative delta below the configured tolerance.

## Go deeper

- [Regression Tests](../15-evals/08-regression-tests.md) — designing the suite the gate runs.
- [Golden Datasets](../15-evals/03-golden-datasets.md) — the hash-pinned dataset behind every comparison.
- [DVC for Data and Models](04-dvc.md) — how the eval job gets the exact dataset.
- [Ship: A Trace Dashboard](08-ship-trace-dashboard.md) — the online counterpart to this offline gate.

---

[← Previous: Cost Monitoring and Budgets](06-cost-monitoring.md) · [Stage index](README.md) · [Next: Ship: A Trace Dashboard](08-ship-trace-dashboard.md)
