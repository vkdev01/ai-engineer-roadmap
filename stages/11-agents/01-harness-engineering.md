# 1. Harness Engineering

> **Stage 11 · AI Agents** — the scaffold around the model — tools, context assembly, loop control, budgets, observability — that turns a text generator into something that finishes work.
>
> **You'll be able to:** describe an agent as a harness of five replaceable parts and write a working skeleton of each in plain Python.
>
> **Prerequisites:** [Context Engineering](../08-prompt-and-context/08-context-engineering.md) · [Tool Calling](../07-llm-apis/05-tool-calling.md)

## Why this exists

Give a model a tool and it will sometimes use it well. Ask it to do a five-step job and it will drift: forget step three, call the same tool twice, answer before checking, or invent a file that does not exist. The instinct is to blame the model. Usually the model is fine and the harness is thin — nobody gave it a budget, nobody showed it what happened last step, nobody stopped it when it stalled.

The mental model is one sentence: **the model proposes, the harness decides.** The model emits tokens; your code decides whether those tokens become an action, whether the action is allowed, what the model sees next, and when the whole thing ends. Every reliability property you want lives in your code, not in the weights.

This is also the good news for your career. Model ids change quarterly; the harness is yours. Teams that win with agents are rarely using a cleverer model — they have a tighter loop, better error text, and a trace they can read.

## The mental model

```text
                 ┌─────────────── harness (your code) ───────────────┐
 user task  ───► │ 1 context assembly   (system, history, tool output)│
                 │ 2 model call         (the only non-determinism)   │
                 │ 3 parse              (text? tool call? final?)     │
                 │ 4 policy             (allowed? needs approval?)   │
                 │ 5 execute            (sandboxed tool run)          │
                 │ 6 record             (trace: step, cost, result)   │
                 │ 7 decide             (continue? stop? escalate?)   │
                 └───────────────────────────────────────────────────┘
                          ▲                                    │
                          └──── feedback: observation ─────────┘
```

Five parts you own: **tools** (what can be done), **context assembly** (what the model sees), **loop control** (when to go again), **budgets** (how much it may spend), **observability** (what you can prove afterwards).

## Core concepts

### The harness is the product surface

Two teams ship the same model. One agent works; one flails. The difference is almost always in the four seams: the tool descriptions, the error strings the model reads, the truncation policy on old turns, and the stop conditions. Treat all four as product decisions with owners, not plumbing.

### Context assembly is a budget, not a log

You cannot append every tool result forever. Assemble each turn deliberately: a stable system prompt, a short task statement, the last few turns verbatim, a compact summary of older turns, and the newest observation in full. Everything else is dropped on purpose. If you have never decided what to drop, you have decided to drop whatever happens to fall off the window.

### Loop control is a state machine, not a `while True`

The loop has states: `planning`, `awaiting_tool`, `awaiting_approval`, `done`, `failed`, `budget_exhausted`. Writing them down as an enum makes interrupts, retries and resumption possible later — this is exactly what the frameworks in [Stage 12](../12-orchestration/README.md) formalise as a graph.

### Budgets are three numbers, not one

Cap **steps** (model calls), **tool calls**, and **wall-clock seconds** separately. A step budget catches thrash; a tool budget catches a tool that returns nothing useful so the model retries; a time budget catches a slow dependency. Record all three in the trace so you can tune them from data.

### Observability means the run is explainable from a file

One JSONL line per step: step index, model, token counts, latency, the tool name, the validated arguments, the result summary, and the reason the step happened. When a user says "it went weird at 3pm", you read the trace instead of re-running and hoping.

## Code

A harness skeleton with the five parts named. It runs against a fake model so you can see the machinery without an API key; swap `FakeModel.complete` for a real SDK call and nothing else changes. The model id must be checked against the provider's current docs.

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

State = Literal["planning", "awaiting_tool", "done", "failed", "budget_exhausted"]

@dataclass
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 12
    max_seconds: float = 60.0
    steps: int = 0
    tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)

    def spend_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps or time.monotonic() - self.started > self.max_seconds:
            raise RuntimeError("budget_exhausted")

@dataclass
class Trace:
    path: str
    def record(self, **event: Any) -> None:
        # One line per step: append-only, greppable, replayable.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), **event}) + "\n")

Tool = Callable[[dict[str, Any]], Awaitable[str]]

class Harness:
    def __init__(self, model: Any, tools: dict[str, Tool], budget: Budget, trace: Trace) -> None:
        self.model, self.tools, self.budget, self.trace = model, tools, budget, trace

    def assemble_context(self, task: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Deliberate truncation: task + last 6 turns, newest observation kept whole.
        return [{"role": "system", "content": "You are a careful agent. Use one tool at a time."},
                {"role": "user", "content": task}, *history[-6:]]

    async def run(self, task: str) -> str:
        history: list[dict[str, Any]] = []
        state: State = "planning"
        while state not in ("done", "failed", "budget_exhausted"):
            self.budget.spend_step()
            msg = await self.model.complete(self.assemble_context(task, history))
            self.trace.record(step=self.budget.steps, kind="model", state=state)

            if msg.get("tool"):
                name, args = msg["tool"], msg.get("args", {})
                if name not in self.tools:
                    history.append({"role": "tool", "content": f"No tool '{name}'. Available: {list(self.tools)}"})
                    continue
                self.budget.tool_calls += 1
                result = await self.tools[name](args)
                self.trace.record(step=self.budget.steps, kind="tool", name=name, args=args)
                history.append({"role": "tool", "content": result})
                state = "planning"
            else:
                history.append({"role": "assistant", "content": msg["text"]})
                state = "done"
                return msg["text"]
        return ""
```

The TypeScript side of the same idea is what a UI consumes — a typed event stream, so the browser can show "calling search…" instead of a spinner. Stream these over SSE from your agent server (see [Streaming APIs](../17-ai-backend/09-ship-streaming-api.md)).

```ts
export type HarnessEvent =
  | { type: "step"; index: number; state: string }
  | { type: "tool_start"; name: string; args: Record<string, unknown> }
  | { type: "tool_end"; name: string; ok: boolean; ms: number }
  | { type: "final"; text: string };

export function toSse(event: HarnessEvent): string {
  // Named events let the client switch on type without parsing the payload twice.
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}
```

## Real use case

A support team built an agent that looks up a customer's plan, then refunds a duplicate charge. Version one appended every tool result to the conversation and had no step cap. On a ticket with five past orders, the context grew until the model started re-querying orders it had already seen, and one run issued the same refund twice.

The fix was pure harness work. The context assembler kept only the current ticket plus a summarised list of already-checked order ids, the tool schema marked `refund_charge` as requiring an approval gate, and the refund tool became idempotent on `charge_id`. Step count per ticket dropped and duplicate refunds stopped — with the same model.

## Common mistakes

- **Mistake:** trusting the model to remember what it already did — **Instead:** keep an explicit ledger in the harness (checked ids, written files) and put it in the context.
- **Mistake:** one global timeout — **Instead:** separate step, tool-call and wall-clock budgets, and record all three.
- **Mistake:** logging only the final answer — **Instead:** log every step with arguments and a result summary; the trace is your only debugging tool.
- **Mistake:** growing the system prompt every time a bug appears — **Instead:** fix the seam that broke (tool description, error text, truncation) and add an eval case.
- **Mistake:** hiding the harness behind a framework before you understand it — **Instead:** write the loop yourself once, then adopt [LangGraph](../12-orchestration/02-langgraph.md) for the parts you now understand.

## Checkpoint

1. Add a `max_tool_calls` check to the skeleton above so a tool-heavy task stops cleanly instead of raising.
2. Change `assemble_context` to keep the newest tool result in full but truncate older ones to 200 characters, and note what breaks.
3. Add a `state="awaiting_approval"` branch that pauses before a write tool and resumes when a flag is set to true.

**Self-check:** Exercise 1 — compare `budget.tool_calls` against `max_tool_calls` before executing, set `state = "budget_exhausted"`, and return a partial answer instead of raising.

## Go deeper

- [Agent Architecture](02-agent-architecture.md) — the loop in full, with every stop condition.
- [Loop Engineering](10-loop-engineering.md) — budgets and termination as a discipline.
- [Tracing](../18-mlops-llmops/05-tracing.md) — turning your JSONL trace into something a team can read.
- [Context Engineering](../08-prompt-and-context/08-context-engineering.md) — the assembly rules in depth.

---

[← Previous: Stage 10 — RAG](../10-rag/README.md) · [Stage index](README.md) · [Next: Agent Architecture](02-agent-architecture.md)
