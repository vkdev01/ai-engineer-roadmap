# 3. Reasoning and Planning

> **Stage 11 · AI Agents** — how the model decides what to do next: interleaved reasoning (ReAct), plan-then-execute, task decomposition, and the cases where planning makes things worse.
>
> **You'll be able to:** choose between reactive and planned agents for a task, and implement both on the same tool registry.
>
> **Prerequisites:** [Agent Architecture](02-agent-architecture.md) · [Few-Shot and Chain of Thought](../08-prompt-and-context/02-few-shot-cot.md)

## Why this exists

A loop with tools is not yet an agent that can handle a job like "compare our three pricing pages and tell me which is cheapest per seat at 40 users". That job needs several lookups in a sensible order. Without any reasoning step, the model often starts fetching before it knows what it needs, or fetches one page and answers from memory.

Two ideas fix this, and they pull in opposite directions. **ReAct** interleaves a short thought with every action, so the model re-plans after each observation. **Plan-then-execute** writes the whole plan first, then runs it. Reactive agents adapt; planned agents are cheap and inspectable. Most production agents are reactive with a light plan at the top, and knowing why is the point of this file.

The mental model: **planning is spending tokens now to avoid spending steps later.** It is only worth it when steps are expensive or irreversible.

## The mental model

```text
ReAct (interleaved)                    Plan-then-execute (upfront)
──────────────────                     ──────────────────────────
think → act → observe                  plan: [s1, s2, s3, s4]
  ↑        │                           for s in plan: act(s)
  └────────┘  re-plan every step       re-plan only on failure
adapts to surprises                     cheap, auditable, brittle
more tokens, more steps                 fewer tokens, fixed order
```

## Core concepts

### ReAct: thought, action, observation, repeat

ReAct is not a special architecture — it is a prompting convention inside the loop you already built. Each turn the model writes a one-or-two-sentence thought ("I need the seat count before pricing, so fetch the pricing page first"), then emits a tool call. The observation comes back and the next thought can contradict the last one, which is the whole value: the plan is rebuilt from fresh evidence every step.

Keep thoughts short and require them to end in a decision. Long free-form reasoning burns context and encourages the model to argue with itself instead of acting.

### Plan-then-execute: one plan, then a runner

Here the model produces a structured plan — a list of steps, each naming a tool and its arguments — and your code executes it. The plan is data, so you can print it, get it approved, cache it, or run steps in parallel. It is also where the failure mode lives: if step two reveals that the customer has no subscription, a fixed plan keeps going.

Use it when the steps are known and cheap (a report template, a nightly digest) or when a human must approve the approach before any action happens.

### Decomposition: split by data, not by verbs

Bad decomposition splits a task into vague verbs: "research, analyse, write". Good decomposition splits by data: "for each of these three URLs, extract the seat price; then compare the three extracted records". Data-shaped subtasks can be run in parallel, retried individually, and verified one at a time. Verb-shaped subtasks cannot be checked at all.

### When planning hurts

Planning hurts in four situations: the task is a single lookup; the environment changes faster than the plan can be written; the plan is longer than the budget for executing it; and the model is bad at the domain, so the plan is confidently wrong. In each case the plan adds tokens and anchors the agent on a wrong path. A useful heuristic: if a competent human would not write the steps down before starting, do not make the model do it.

### Verify the plan, not just the answer

A cheap check that pays for itself: after planning, ask the model (or a validator) whether each step's output is a valid input to the next step. Most bad plans fail this test immediately — step three needs an `order_id` that step one never produces.

## Code

The same registry, two drivers. First, a ReAct turn: the thought is part of the assistant message, so the model sees its own reasoning next turn.

```python
from __future__ import annotations

import json
from typing import Any, Protocol

class Model(Protocol):
    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> dict[str, Any]: ...

REACT_SYSTEM = """You are a research agent.
Each turn: write one short Thought (max 2 sentences) ending in a decision,
then either call exactly one tool or give the final answer.
Do not repeat a tool call you already made with the same arguments."""

async def react_step(model: Model, task: str, history: list[dict[str, Any]],
                     tools: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [{"role": "system", "content": REACT_SYSTEM},
                {"role": "user", "content": task}, *history]
    reply = await model.complete(messages, tools)
    # Persist the thought with the action so the next turn sees its own plan.
    history.append({"role": "assistant",
                    "content": f"Thought: {reply.get('thought', '')}\n"
                               f"Action: {reply.get('tool') or 'final'}"})
    return reply
```

Now plan-then-execute. The plan is validated as data before a single tool runs, which is what makes it safe to show a human.

```python
from pydantic import BaseModel, Field, ValidationError

class PlanStep(BaseModel):
    tool: str
    args: dict[str, Any]
    produces: str = Field(description="name of the value this step yields")

class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]

async def plan_then_execute(model: Model, task: str, registry: dict[str, Any],
                            tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = await model.complete(
        [{"role": "user", "content": f"Write a JSON plan for: {task}"}], tools)
    try:
        plan = Plan.model_validate_json(raw["text"])
    except ValidationError as exc:
        raise ValueError(f"Unusable plan: {exc.error_count()} schema errors") from exc

    available = set(registry)
    produced: set[str] = set()
    for step in plan.steps:                       # static check before any side effect
        if step.tool not in available:
            raise ValueError(f"Plan uses unknown tool {step.tool}")
        produced.add(step.produces)

    results: list[dict[str, Any]] = []
    for step in plan.steps:
        # Inject earlier outputs by name; the plan never hardcodes a value it cannot know.
        results.append({"tool": step.tool, "output": await registry[step.tool](step.args)})
    return results
```

The TypeScript variant matters when the plan is shown to a user before it runs — render it as a checklist and let them deselect steps.

```ts
export type PlanStep = { tool: string; args: Record<string, unknown>; produces: string };

export function validatePlan(steps: PlanStep[], tools: readonly string[]): string[] {
  const known = new Set(tools);
  const problems: string[] = [];
  steps.forEach((s, i) => {
    if (!known.has(s.tool)) problems.push(`step ${i + 1}: unknown tool ${s.tool}`);
    if (!s.produces) problems.push(`step ${i + 1}: declares no output name`);
  });
  return problems;
}
```

## Real use case

A procurement team's agent answers "which supplier is cheaper for 500 units, including shipping?". Reactive-only, it fetched the first supplier's page and answered with unit price, ignoring shipping. Plan-first, it produced a fixed four-step plan that was wrong whenever a supplier's page had no shipping table — it kept going and compared a real number against a missing one.

The shipped version is hybrid: a short plan is generated up front and shown in the UI, but each step's observation can trigger a re-plan. When the shipping table is missing, the agent inserts a "fetch shipping policy page" step instead of continuing with a hole. Accuracy on this class of question improved while the number of steps stayed flat, because the plan stopped it from wandering.

## Common mistakes

- **Mistake:** asking for a long reasoning trace on every step — **Instead:** one or two sentences that end in a decision.
- **Mistake:** hardcoding a fixed plan for a changing environment — **Instead:** allow re-planning on any failed or surprising observation.
- **Mistake:** decomposing by verbs ("research, analyse") — **Instead:** decompose by data ("for each of these three URLs…").
- **Mistake:** executing a plan without checking tool names and produced values — **Instead:** validate the plan as data before the first side effect.
- **Mistake:** planning a single lookup — **Instead:** go reactive; a plan there is pure overhead.

## Checkpoint

1. Add a `produces`/`consumes` field pair to `PlanStep` and reject plans where a step consumes a name no earlier step produced.
2. Turn the ReAct loop into a hybrid: plan once, execute steps, and re-plan only when a tool returns `no results`.
3. Measure it — run ten tasks with both drivers, record steps and tool calls per task, and write down which class of task each driver wins.

**Self-check:** Exercise 1 — keep a running set of `produces` values while walking the plan, and raise if `step.consumes` is not in that set.

## Go deeper

- [Tool Calling](04-tool-calling.md) — the schemas your plans are built from.
- [Reflection and Self-Critique](09-reflection.md) — checking a plan or answer against evidence.
- [Multi-Agent Patterns](07-multi-agent.md) — decomposition across agents rather than steps.
- [Graph Engineering](../12-orchestration/03-graph-engineering.md) — plans expressed as routable graphs.

---

[← Previous: Agent Architecture](02-agent-architecture.md) · [Stage index](README.md) · [Next: Tool Calling](04-tool-calling.md)
