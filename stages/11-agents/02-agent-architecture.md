# 2. Agent Architecture

> **Stage 11 · AI Agents** — the minimal agent: prompt, tool registry, execute, observe, repeat, with explicit stop conditions.
>
> **You'll be able to:** write a complete agent loop in about forty lines and name the four ways it can end.
>
> **Prerequisites:** [Harness Engineering](01-harness-engineering.md) · [Tool Calling](../07-llm-apis/05-tool-calling.md)

## Why this exists

People talk about agents as if they were a new kind of model. They are not. An agent is a `while` loop around the same completion call you already know, plus a rule for turning model output into an action and action results back into input. Once you have written that loop yourself, every framework in [Stage 12](../12-orchestration/README.md) becomes a convenience instead of a mystery.

What breaks without an explicit loop is subtle. If you call the model once and hope, you get one-shot answers that abandon the task halfway. If you loop without stop conditions, you get an expensive machine that keeps calling the same search tool because nothing tells it that it is finished.

The mental model: **agent = model + tools + loop + stop rule.** Take away any of the four and you have a chatbot.

## The mental model

```text
  task ──► [assemble prompt]
                │
                ▼
          ┌──► model call ──► final text? ──yes──► return answer  (stop 1: answered)
          │        │
          │        no, tool call
          │        ▼
          │   validate args ──bad──► error message back to model (stop 0: none, retry)
          │        │ good
          │        ▼
          │   run tool (sandboxed) ──► observation
          │        │
          └────────┘
               also stop on: step budget (2), repeated no-progress (3), fatal error (4)
```

Four exits: **answered**, **budget exhausted**, **no progress**, **error**. A loop with fewer than four is a loop you cannot operate.

## Core concepts

### One model call per turn, one action per turn

Let the model emit either a final answer or exactly one tool call. One action per turn keeps the trace readable, makes approval gates trivial (you pause before the single action), and prevents the "batch of five tool calls where the third was garbage" failure. Providers do support parallel tool calls; add that later, deliberately, once the single-action loop is trustworthy.

### The registry is a dictionary with rules

A tool registry maps a name to three things: a JSON schema for the arguments, a Python function, and a policy (read-only or side-effecting). The schema is what the model sees; the function is what runs; the policy is what your loop checks before running. Keeping policy separate from the function means the same tool can be allowed in one agent and gated in another.

### Observations must be short and self-describing

The tool result goes back into the context, so its length is a cost and its shape is a comprehension problem for the model. Return a compact, labelled string: what you looked for, how many results, and the useful fields. Never return a raw API blob. If a result is empty, say `no results for query="x"` — a bare empty string reads to the model like a broken tool and invites a retry.

### Stop conditions are data, not vibes

The model does not know your budget. Your loop decides. Track steps, tool calls, and a progress signal (new information this turn versus the same tool with the same arguments). When the progress signal repeats, stop and return what you have with an honest partial answer. This is the difference between a demo and something you can run unattended.

### Determinism belongs at the edges

Keep temperature low for tool-argument generation, validate everything before executing, and make the loop's own decisions deterministic. You cannot make the model reproducible; you can make everything around it reproducible, which is what makes debugging possible.

## Code

A complete loop with a registry, validation, and all four exits. The `Model` protocol is the only thing you swap for a real SDK — for OpenAI or Anthropic, translate the tool schema to their function/tool format and read the tool call from the response; check the provider's current docs for the exact field names.

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol

@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]          # JSON Schema for the arguments
    run: Callable[[dict[str, Any]], Awaitable[str]]
    side_effect: bool = False       # True means: needs an approval gate

class Model(Protocol):
    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> dict[str, Any]: ...

@dataclass
class Outcome:
    status: Literal["answered", "budget", "no_progress", "error"]
    text: str
    steps: int
    seen: list[str] = field(default_factory=list)

async def run_agent(task: str, model: Model, tools: dict[str, ToolSpec],
                    max_steps: int = 10) -> Outcome:
    messages = [{"role": "user", "content": task}]
    schemas = [{"name": t.name, "description": t.description, "schema": t.schema}
               for t in tools.values()]
    seen: list[str] = []                       # progress ledger: "name|args-json"

    for step in range(1, max_steps + 1):
        reply = await model.complete(messages, schemas)

        if "tool" not in reply:                # exit 1: the model answered
            return Outcome("answered", reply["text"], step, seen)

        name, args = reply["tool"], reply.get("args", {})
        spec = tools.get(name)
        if spec is None:                       # unknown tool: teach, do not crash
            messages.append({"role": "tool",
                             "content": f"Unknown tool {name}. Available: {sorted(tools)}"})
            continue

        key = f"{name}|{json.dumps(args, sort_keys=True)}"
        if key in seen:                        # exit 3: same call twice = no progress
            return Outcome("no_progress", "Stopped: repeated identical tool call.", step, seen)
        seen.append(key)

        try:
            observation = await spec.run(args)
        except Exception as exc:               # exit 4 handled as a teachable message
            observation = f"Tool {name} failed: {exc}. Adjust the arguments and retry."
        messages.append({"role": "tool", "content": observation})

    return Outcome("budget", "Stopped: step budget exhausted.", max_steps, seen)
```

On the TypeScript side, the same loop runs in a Node service and pushes events to a browser. The client renders the trace as it happens, which is what makes an agent feel trustworthy rather than magic.

```ts
type Step = { kind: "tool"; name: string; args: unknown } | { kind: "final"; text: string };

export async function runAgent(
  task: string,
  callModel: (messages: unknown[], tools: unknown[]) => Promise<Step>,
  registry: Record<string, (args: never) => Promise<string>>,
  maxSteps = 10,
): Promise<{ status: "answered" | "budget"; text: string }> {
  const messages: unknown[] = [{ role: "user", content: task }];
  for (let step = 0; step < maxSteps; step++) {
    const out = await callModel(messages, Object.keys(registry));
    if (out.kind === "final") return { status: "answered", text: out.text };
    const run = registry[out.name];
    const result = run ? await run(out.args as never) : `Unknown tool ${out.name}`;
    messages.push({ role: "tool", content: result });
  }
  return { status: "budget", text: "Step budget exhausted." };
}
```

## Real use case

A fintech built an internal "invoice chaser". The agent reads a customer's overdue invoices, drafts an email, and files it as a draft in the billing inbox. The first version had one stop condition: the model said it was done. It regularly filed the same draft twice because it never saw confirmation of the previous one, and occasionally gave up silently after a tool error.

The rebuild kept the same model and tools and changed the loop. Each tool returned a labelled confirmation with an id, the ledger blocked identical repeat calls, and the side-effecting `create_draft` tool required approval. Duplicate drafts stopped because the observation carried the draft id the model needed to notice it had already acted.

## Common mistakes

- **Mistake:** looping until the model says "done" in prose — **Instead:** require a structured final answer, and add budgets anyway.
- **Mistake:** raising on a tool exception and killing the run — **Instead:** return the error as an observation the model can act on, then let the budget stop real thrash.
- **Mistake:** returning raw JSON blobs as observations — **Instead:** return a short labelled summary with the ids the model needs.
- **Mistake:** letting the model call five tools per turn before you trust one — **Instead:** one action per turn until the loop is boring.
- **Mistake:** putting approval logic inside the tool function — **Instead:** keep policy in the loop so the same tool can be gated or free per agent.

## Checkpoint

1. Add a `max_seconds` check to `run_agent` and return a `budget` outcome that reports how far it got.
2. Add a `search` tool and an `Outcome.seen` report so you can see the exact call sequence for a failed run.
3. Make the no-progress rule fuzzy: treat two calls to the same tool with near-identical arguments (same query, different whitespace) as a repeat.

**Self-check:** Exercise 1 — record `start = time.monotonic()` before the loop and check `time.monotonic() - start > max_seconds` at the top of each step, returning `Outcome("budget", ...)`.

## Go deeper

- [Reasoning and Planning](03-reasoning-planning.md) — what the model should think about before it calls a tool.
- [Loop Engineering](10-loop-engineering.md) — termination rules that hold up under load.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — making the final answer machine-readable.
- [LangGraph](../12-orchestration/02-langgraph.md) — the same loop, expressed as a state graph.

---

[← Previous: Harness Engineering](01-harness-engineering.md) · [Stage index](README.md) · [Next: Reasoning and Planning](03-reasoning-planning.md)
