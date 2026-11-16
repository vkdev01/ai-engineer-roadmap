# 6. Agent Evals

> **Stage 15 · Evals and Testing** — grading the trajectory as well as the outcome, because an agent that got lucky is not a working agent.
>
> **You'll be able to:** evaluate an agent on outcome, on trajectory and on cost, write tool-call assertions, and detect the failure modes that outcome-only evals hide.
>
> **Prerequisites:** [Loop Engineering](../11-agents/10-loop-engineering.md) · [Agent Architecture](../11-agents/02-agent-architecture.md) · [Golden Datasets](03-golden-datasets.md)

## Why this exists

A single model call is easy to grade: compare the output. An agent is a program that chose its own steps, so the same correct answer can arrive after two tool calls or after nine, one of which wrote to a database it should not have touched.

Outcome-only evals reward luck. Trajectory-only evals punish valid creativity. Real agent evals grade three things: did it achieve the goal, did it take a defensible path, and what did it cost.

The mental model: **grade the answer, the route, and the bill.**

## The mental model

```text
                    ┌─────────────────────────────┐
   task ──► agent ──┤  outcome: is the goal met?  │──► pass/fail
                    ├─────────────────────────────┤
                    │  trajectory: right tools,   │──► assertions
                    │  no forbidden calls, order  │
                    ├─────────────────────────────┤
                    │  cost: steps, tokens, time  │──► budgets
                    └─────────────────────────────┘

   an agent that passes outcome but fails trajectory is a liability
   an agent that fails outcome but has a clean trajectory is a prompt problem
```

## Core concepts

### Outcome checks: assertions, not vibes

Most agent tasks have a checkable end state: the file exists and parses, the record is in the database with the right fields, the answer contains the required value. Write that check as code. Where the outcome is prose, use a judge with a rubric.

### Trajectory assertions: the four that catch most problems

1. **Required tools called** — the task cannot be done without `search`; assert it was called at least once.
2. **Forbidden tools never called** — assert no write tool ran without approval.
3. **Ordering** — assert `retrieve` happened before `answer`, when the design requires it.
4. **Step ceiling** — assert the run finished within the budget; a pass at step 11 of a 6-step budget is a fail.

```python
@dataclass
class Trajectory:
    tool_calls: list[str]          # names in order
    steps: int
    tokens: int
    cost_usd: float
    approved_writes: set[str]

def assert_trajectory(t: Trajectory, required=(), forbidden=(), max_steps=6, max_cost=0.05) -> list[str]:
    problems: list[str] = []
    for name in required:
        if name not in t.tool_calls:
            problems.append(f"required tool not called: {name}")
    for name in forbidden:
        if name in t.tool_calls and name not in t.approved_writes:
            problems.append(f"forbidden tool called without approval: {name}")
    if t.steps > max_steps:
        problems.append(f"step budget exceeded: {t.steps} > {max_steps}")
    if t.cost_usd > max_cost:
        problems.append(f"cost budget exceeded: ${t.cost_usd:.3f} > ${max_cost:.3f}")
    return problems
```

### Cost is part of the score

Report cost and steps next to pass rate, always. An agent whose pass rate rose from 0.78 to 0.82 while its median step count went from 3 to 7 has regressed in practice, and a pass-rate-only scorecard will call that an improvement.

### The three failure shapes you are looking for

| Shape | Signature | Meaning |
|---|---|---|
| Lucky pass | outcome ok, trajectory violates a rule | hidden liability; will fail on a different day |
| Honest fail | outcome wrong, trajectory clean | prompt or tool description problem |
| Loop | steps at ceiling, repeated tool calls | no-progress detection missing |

Counting these three is more informative than a single pass rate, and it is what you put in the report.

## Code

An agent eval runner over a small task set.

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field

@dataclass
class Task:
    tid: str
    instruction: str
    check: str                  # name of an outcome checker
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_steps: int = 6


def run_suite(agent, tasks: list[Task], checkers: dict) -> dict:
    rows = []
    for task in tasks:
        run = agent.run(task.instruction, max_steps=task.max_steps)
        outcome_ok = checkers[task.check](run)                 # bool
        traj_problems = assert_trajectory(
            Trajectory(run.tool_calls, run.steps, run.tokens, run.cost_usd, run.approved_writes),
            required=task.required_tools, forbidden=task.forbidden_tools, max_steps=task.max_steps,
        )
        shape = (
            "lucky_pass" if outcome_ok and traj_problems
            else "honest_fail" if not outcome_ok and not traj_problems
            else "loop" if run.steps >= task.max_steps
            else "clean_pass" if outcome_ok
            else "broken"
        )
        rows.append({"tid": task.tid, "outcome": outcome_ok, "problems": traj_problems,
                     "steps": run.steps, "cost": run.cost_usd, "shape": shape})

    n = len(rows) or 1
    return {
        "n": len(rows),
        "pass_rate": round(sum(r["outcome"] for r in rows) / n, 3),
        "clean_pass_rate": round(sum(1 for r in rows if r["shape"] == "clean_pass") / n, 3),
        "lucky_passes": [r["tid"] for r in rows if r["shape"] == "lucky_pass"],
        "honest_fails": [r["tid"] for r in rows if r["shape"] == "honest_fail"],
        "loops": [r["tid"] for r in rows if r["shape"] == "loop"],
        "median_steps": sorted(r["steps"] for r in rows)[len(rows) // 2],
        "total_cost": round(sum(r["cost"] for r in rows), 4),
    }


if __name__ == "__main__":
    report = run_suite(agent, load_tasks("evals/agent_tasks.jsonl"), CHECKERS)
    print(json.dumps(report, indent=2))
```

A task file where every case is checkable:

```json
{"tid":"a1","instruction":"Find the refund window for damaged goods and state it.","check":"answer_contains_30_days","required_tools":["search"],"max_steps":4}
{"tid":"a2","instruction":"Issue a replacement for order 88213.","check":"order_updated","required_tools":["search","update_order"],"forbidden_tools":[],"max_steps":6}
{"tid":"a3","instruction":"Issue a replacement for order 90001 without approval.","check":"order_unchanged","forbidden_tools":["update_order"],"max_steps":6}
```

TypeScript, for CI:

```ts
type SuiteReport = { pass_rate: number; clean_pass_rate: number; loops: string[] };

export function gate(r: SuiteReport, minClean = 0.75): { pass: boolean; why: string } {
  if (r.loops.length) return { pass: false, why: `loops: ${r.loops.join(", ")}` };
  if (r.clean_pass_rate < minClean) {
    return { pass: false, why: `clean pass ${r.clean_pass_rate} < ${minClean} (pass rate ${r.pass_rate})` };
  }
  return { pass: true, why: "ok" };
}
```

## Real use case

A team ships an agent that handles customer refund requests. Its pass rate is 0.81 and everyone is pleased. Then an incident: the agent issued a refund twice for one request because a retry re-ran the write tool.

The trajectory eval would have caught it before launch. Their first assertion set is three lines — `update_order` may not run without an approval token, no tool may run twice with the same arguments, and step count must stay within budget — and the duplicated refund fails immediately. Retrospectively they find four other "lucky passes" in the suite: correct outcomes reached through paths that violated a rule, each one a latent incident.

Their scorecard now leads with `clean_pass_rate` rather than `pass_rate`, because the difference between them is exactly the amount of luck they were relying on.

## Common mistakes

- **Mistake:** grading only the final answer — **Instead:** assert on required tools, forbidden tools, ordering and step ceiling.
- **Mistake:** no case where a write must *not* happen — **Instead:** include negative tasks; the safest agent is one tested on restraint.
- **Mistake:** reporting pass rate without cost or steps — **Instead:** always show median steps and total cost beside it.
- **Mistake:** a suite of five tasks — **Instead:** twenty is a floor for agents, because variance is high.
- **Mistake:** re-running a flaky agent task until it passes — **Instead:** run each task three times and report the pass rate per task; flakiness is a finding.
- **Mistake:** fixing lucky passes by loosening the assertion — **Instead:** tighten the tool or the prompt; the assertion was right.

## Checkpoint

1. Write five agent tasks with outcome checkers, including one where the correct behaviour is to do nothing.
2. Add trajectory assertions for a forbidden tool and a step ceiling, and confirm a deliberately broken run fails both.
3. Run the suite three times and record pass rate, clean pass rate, median steps and total cost. Compare the two rates.

**Self-check:** if `clean_pass_rate` is much lower than `pass_rate`, you are relying on luck — fix the trajectory rules before adding features.

## Go deeper

- [Loop Engineering](../11-agents/10-loop-engineering.md) — the budgets these assertions enforce
- [Human Evals](07-human-evals.md) — when the outcome check needs a person
- [Regression Tests](08-regression-tests.md) — gating CI on clean pass rate

---
[← Previous: Hallucinations](05-hallucinations.md) · [Stage index](README.md) · [Next: Human Evals](07-human-evals.md)
