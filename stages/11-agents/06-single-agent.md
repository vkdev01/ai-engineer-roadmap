# 6. Single-Agent Patterns

> **Stage 11 · AI Agents** — the patterns that make one agent enough: a tight tool set, a router in front, a workflow behind, and a clear boundary between what the model decides and what your code decides.
>
> **You'll be able to:** pick a single-agent pattern for a task, and recognise the moment when adding a second agent would make things worse.
>
> **Prerequisites:** [Memory and State](05-memory-and-state.md) · [Reasoning and Planning](03-reasoning-planning.md)

## Why this exists

Multi-agent systems get the attention; single agents do the work. Almost every task people reach for a "team of agents" to solve is really one agent with a better tool set, or a deterministic workflow with one model call inside it. The single-agent patterns below are the ones that hold up in production, and they are the baseline you must beat before adding complexity.

The reason is arithmetic. Two agents mean two contexts, two prompts to maintain, and a handoff that loses information. One agent with eight well-designed tools has one trace, one budget and one place to debug. Start there; move on only when a concrete limitation forces you.

The mental model: **let the model decide what it cannot know, let code decide everything else.**

## The mental model

```text
                  ┌──────────────────────────────────────┐
 task ──► [router]│  one-shot        loop-with-tools     │
           (code  │  (single call)   (ReAct, budgets)    │
           or LLM)│  workflow        agent-as-tool       │
                  │  (fixed steps)   (nested call)       │
                  └──────────────────────────────────────┘
   rule of thumb: fewest model calls that reliably finish the job
```

| Pattern | Model calls | Use when |
|---|---|---|
| One-shot | 1 | Input is complete; no lookup needed |
| Workflow with one model step | 1-2 | Order of steps is known |
| Loop with tools | 2-10 | Steps depend on what you find |
| Agent-as-tool | nested | A subtask needs its own budget and tools |

## Core concepts

### One-shot with a structured answer

If the task needs no external facts, do not build an agent. One call, a strict output schema, validation, retry once. This handles classification, extraction, rewriting and most "summarise this" work — and it is cheap, fast and testable. Reach for the loop only when the answer depends on something the model cannot see.

### Workflow with a model step inside

Most "agentic" business processes are workflows: fetch, transform, call the model once, write. Expressing them as a workflow gives you retries per step, a clear failure point, and a log a non-engineer can follow. Use the model where judgement is needed and code everywhere else. A workflow that branches on a model decision is still a workflow.

### Router in front, specialists behind

When you have three or four genuinely different task types, put a cheap classifier in front and give each branch its own small prompt and tool set. This beats one giant prompt that must handle everything: each branch's context is smaller, its instructions are sharper, and you can evaluate them separately. The router can be a keyword table, an embedding similarity check, or a small model call — try the cheapest first.

### Agent-as-tool: nesting with a budget

Sometimes one subtask deserves its own loop: "find and verify the delivery date" may need four tool calls. Wrap it as a tool whose implementation runs a nested agent with its own small budget, and return only its final answer to the outer loop. The outer agent sees a tool that takes a question and returns a fact — its context stays clean, and the nested agent cannot spend the outer budget.

### Write the escalation rule down

For each pattern, decide in advance what happens when it fails: fall back to a human, fall back to a cheaper answer, or retry once. Without this rule, every failure becomes an exception, and exceptions in agents are usually silent wrong answers.

## Code

A router plus a specialist, and an agent-as-tool wrapper. Note that the router returns a name, not a free-text decision — the branch is code, so it cannot be misheard.

```python
from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

Route = Literal["billing", "technical", "other"]

async def route(question: str, model: Any) -> Route:
    # Cheap classifier first: keywords cover most traffic before any model call.
    lowered = question.lower()
    if any(w in lowered for w in ("invoice", "refund", "charge", "gst")):
        return "billing"
    if any(w in lowered for w in ("error", "crash", "login", "api")):
        return "technical"
    out = await model.complete([{"role": "user",
        "content": f"Classify as billing, technical or other. Answer with one word.\n{question}"}], [])
    word = out["text"].strip().lower()
    return word if word in ("billing", "technical") else "other"

SPECIALISTS: dict[Route, str] = {
    "billing": "You are a billing specialist. Use only billing tools. Quote amounts in rupees.",
    "technical": "You are a support engineer. Ask for the error text before guessing.",
    "other": "Answer briefly; if unsure, say you will route to a human.",
}

async def handle(question: str, model: Any, run_loop: Callable[..., Awaitable[str]]) -> str:
    branch = await route(question, model)
    # Each branch gets its own prompt and its own tool subset.
    return await run_loop(system=SPECIALISTS[branch], task=question, branch=branch)
```

The agent-as-tool wrapper keeps the nested run's mess out of the parent's context — only the final answer crosses the boundary.

```python
from dataclasses import dataclass

@dataclass
class SubAgent:
    name: str
    description: str
    system: str
    tools: dict[str, Any]
    max_steps: int = 6

def as_tool(sub: SubAgent, run_loop: Callable[..., Awaitable[str]]) -> dict[str, Any]:
    async def run(args: dict[str, Any]) -> str:
        question = args["question"]
        try:
            answer = await run_loop(system=sub.system, task=question,
                                    tools=sub.tools, max_steps=sub.max_steps)
        except Exception as exc:                       # never leak a nested traceback
            return f"{sub.name} failed: {type(exc).__name__}. Answer without it."
        return f"{sub.name} says: {answer}"            # one line crosses the boundary

    return {
        "name": sub.name,
        "description": sub.description,
        "schema": {"type": "object", "required": ["question"],
                   "properties": {"question": {"type": "string"}}},
        "run": run,
        "side_effect": False,
    }
```

In TypeScript, the router is often the whole product surface: a single endpoint that returns which specialist handled the message, so the UI can show the right panel.

```ts
export type Branch = "billing" | "technical" | "other";

export function routeByKeyword(question: string): Branch {
  const q = question.toLowerCase();
  if (/\b(invoice|refund|charge|gst)\b/.test(q)) return "billing";
  if (/\b(error|crash|login|api)\b/.test(q)) return "technical";
  return "other";
}

export async function handle(
  question: string,
  callBranch: (branch: Branch, question: string) => Promise<string>,
): Promise<{ branch: Branch; answer: string }> {
  const branch = routeByKeyword(question);
  return { branch, answer: await callBranch(branch, question) };
}
```

## Real use case

A SaaS support team ran a single prompt with fourteen tools. It was mediocre at everything: it offered refund tools on login questions and asked for error text on billing questions. Adding a second "supervisor agent" made it slower and no more accurate, because the supervisor had the same fuzzy boundary.

The version that shipped dropped the supervisor and added a router. Billing, technical and account questions went to three small agents, each with three to five tools and a one-paragraph prompt. Misrouted tool calls fell sharply because each branch simply could not see the wrong tools, and each branch could be evaluated and tuned on its own.

## Common mistakes

- **Mistake:** building a loop for a task with no external lookups — **Instead:** one call with a strict schema.
- **Mistake:** one prompt with every tool for every user — **Instead:** route first, then expose only the branch's tools.
- **Mistake:** letting a nested agent's transcript into the parent context — **Instead:** return one summary line across the boundary.
- **Mistake:** using a model call to route when a keyword table would do — **Instead:** cheapest router that works; upgrade only on measured errors.
- **Mistake:** no defined fallback when a branch fails — **Instead:** decide the escalation rule per branch and write it in the code.

## Checkpoint

1. Add a `confidence` return to `route` and send low-confidence questions to `other`, then measure how often that happens on a sample of twenty questions.
2. Turn the billing branch into an `as_tool` sub-agent and check the parent's context length before and after.
3. Add an eval that runs the same twenty questions through the router and reports branch accuracy.

**Self-check:** Exercise 1 — have the model answer with `branch confidence`, parse both, and treat anything below your threshold as `other`; log the count so you can tune the threshold.

## Go deeper

- [Multi-Agent Patterns](07-multi-agent.md) — when one agent genuinely is not enough.
- [Agent Evals](../15-evals/06-agent-evals.md) — measuring per-branch quality.
- [Agent Design](../21-system-design/03-agent-design.md) — designing the boundary at scale.
- [Trade-offs](../12-orchestration/10-tradeoffs.md) — the cost of each topology.

---

[← Previous: Memory and State](05-memory-and-state.md) · [Stage index](README.md) · [Next: Multi-Agent Patterns](07-multi-agent.md)
