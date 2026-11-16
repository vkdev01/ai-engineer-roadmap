# 10. Trade-offs

> **Stage 12 · Orchestration** — an honest comparison of the frameworks in this stage, and the case for writing your own loop.
>
> **You'll be able to:** choose an orchestration approach for a given job, defend the choice in a review, and name the specific cost you accepted.
>
> **Prerequisites:** [LangGraph](02-langgraph.md) · [Graph Engineering](03-graph-engineering.md) · [Agent Architecture](../11-agents/02-agent-architecture.md)

## Why this exists

Every framework in this stage claims to make agents easy. They do — for the first version. The cost shows up in the second version, when you need to inspect why a run took 14 model calls, or cap spend per tenant, or resume after a deploy, and the abstraction you chose is now between you and the answer.

This file exists so that the choice is deliberate. There is no best framework; there is a best fit for a job, a team and a timeline, and being able to say *why* is the actual skill.

The mental model: **a framework is a loan against future flexibility, spent on present speed.**

## The mental model

```text
        more abstraction  ──────────────────────────────►

  plain loop      typed graph      framework graph    role-based crew

  you write       you write        you configure      you describe
  everything      control flow     both, in YAML      agents and hope

  ◄────────────────────────────────────────────────────
        more control, more code to write
```

Ask two questions before choosing. **How many branches does this workflow really have?** and **who debugs it at 2am?** If the answer to the second is "whoever is on call, using the traces", then observability outranks elegance.

## Core concepts

### The comparison that matters

| | Plain loop | LangGraph | LlamaIndex | Haystack | Pydantic AI | DSPy | CrewAI / AutoGen |
|---|---|---|---|---|---|---|---|
| Primary abstraction | your code | state graph | indexes + query engines | pipeline of components | typed agent | signature + module | roles / conversation |
| Control flow | explicit | explicit edges | mostly retrieval-driven | explicit wiring | explicit | compiled | model-decided |
| Learning curve | none | medium | medium | low-medium | low | high | low |
| Debuggability | best | good | medium | good | good | poor (compiled prompts) | poor |
| Persistence / resume | you build it | built in (checkpoints) | partial | partial | you build it | n/a | limited |
| Typed outputs | you build it | you build it | partial | partial | first-class | first-class | weak |
| Best fit | ≤3 steps, full control | branching + approval | document Q&A | search pipelines | typed extraction agents | optimisation with a metric | prototyping a decomposition |
| Worst fit | multi-role orchestration | tiny scripts | general agents | agent loops | free-form chat | single prompts, no data | anything with hard budgets |

### The three costs nobody puts in the README

**Dependency surface.** A framework pulls in a dependency tree, and that tree is a security and upgrade surface. Stage 20 and Stage 19 both assume you know what is in your image. Every framework you add is another thing to patch, another thing that can pin a transitive dependency you need.

**Abstraction leakage.** Frameworks leak at the worst moments: a retry you did not ask for, a message format you cannot control, a token count you cannot see. Each leak costs an afternoon. The question is not "does it leak" but "does it leak in the place where my requirements are strictest".

**Bus factor of the abstraction.** If the framework's maintainers change direction, your options are to migrate or to freeze the version. Both are fine if you have tests (Stage 15) and painful if you do not. This is an argument for keeping your *business logic* outside the framework's types wherever the framework allows it.

### A decision procedure that fits on a card

1. **Write the workflow as a plain loop first.** If it fits in under ~150 lines and has at most two branches, stop. You are done, and you will debug it faster than anyone with a framework.
2. **Add a graph when you need** a conditional edge, a checkpoint, or a human interrupt. Those three needs are exactly what a state graph buys, and they appear in that order in real systems.
3. **Add a data framework** (LlamaIndex, Haystack) when your problem is mostly retrieval and ingestion, not control flow.
4. **Add a typed agent** (Pydantic AI) when your pain is malformed output rather than orchestration.
5. **Add an optimiser** (DSPy) only after you have a metric and ≥20 labelled examples — not before.
6. **Add a role-based framework** for a spike, then pin the graph for production.

### What "write your own" actually costs

People overestimate this. A minimal harness — tool registry, loop with a step budget, trace log, typed state, retry — is roughly 200 lines of Python, and you own every line of it. What you do *not* get for free: persistence, streaming, interrupt/resume, and the accumulated edge-case handling. LangGraph's checkpointing in particular is annoying to rebuild well.

So the honest calculus: **write your own when control flow is simple and your requirements are unusual; adopt a graph framework when you need resumability and interrupts; adopt a data or optimisation framework when the pain is in retrieval or in tuning.**

## Code

The same three-step workflow in two shapes, so you can feel the difference.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------- shape 1: plain loop, ~40 lines, fully inspectable ----------

@dataclass
class Run:
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def loop(question: str, tools: dict[str, Callable[..., str]], llm, max_steps: int = 6) -> Run:
    run = Run(messages=[{"role": "user", "content": question}])
    while run.steps < max_steps:
        run.steps += 1
        reply = llm(run.messages, tools=list(tools))          # returns tool call or answer
        run.trace.append({"step": run.steps, "kind": reply["kind"]})
        if reply["kind"] == "answer":
            return run
        result = tools[reply["name"]](**reply["args"])        # side effect happens here
        run.messages.append({"role": "tool", "content": result, "name": reply["name"]})
    run.trace.append({"step": run.steps, "kind": "budget_exhausted"})
    return run
```

```python
# ---------- shape 2: the same workflow as a graph ----------
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


class S(TypedDict, total=False):
    question: str
    route: Literal["cheap", "retrieve"]
    context: str
    answer: str


def router(s: S) -> S:
    s["route"] = "retrieve" if len(s["question"]) > 40 else "cheap"
    return s


def retrieve(s: S) -> S:
    s["context"] = search(s["question"])       # your retriever
    return s


def answer(s: S) -> S:
    s["answer"] = llm_answer(s["question"], s.get("context", ""))
    return s


g = StateGraph(S)
g.add_node("router", router)
g.add_node("retrieve", retrieve)
g.add_node("answer", answer)
g.set_entry_point("router")
g.add_conditional_edges("router", lambda s: s["route"], {"cheap": "answer", "retrieve": "retrieve"})
g.add_edge("retrieve", "answer")
g.add_edge("answer", END)
app = g.compile(checkpointer=MemorySaver())

out = app.invoke({"question": "..."}, config={"configurable": {"thread_id": "run-1"}})
```

Both are readable. The graph wins on three things the loop cannot do without extra work: `thread_id` gives you resumption, the conditional edge is declarative (so it can be drawn and tested), and the checkpointer is a solved problem rather than your weekend. The loop wins on everything else: fewer imports, no state schema to maintain, and a trace you wrote yourself.

## Real use case

A payments company builds a support assistant. The first version is a plain loop with three tools and a 6-step budget, shipped in a week. It handles the majority of traffic, and on-call engineers debug it by reading a JSONL trace they wrote.

Six months later two requirements arrive that the loop cannot meet cleanly: refunds above a threshold need a human approval that may come hours later, and runs must survive deployments. Approval-plus-resume is exactly what a checkpointer and an interrupt give you, so the workflow moves to LangGraph — but only the refund branch. The rest of the loop stays a loop, called as a single node inside the graph.

That hybrid is the most common production shape in practice, and it is the answer to "framework or not": **framework where you need its features, plain code everywhere else.** The team's rule, written in the design doc, is that a node may contain arbitrary code but the graph may not contain business rules that are not also in the test suite.

## Common mistakes

- **Mistake:** choosing a framework before writing the plain loop — **Instead:** write the loop first; you will discover which feature you actually need, and it is usually not the one the README advertised.
- **Mistake:** evaluating frameworks on a demo, not on your hardest case — **Instead:** port your nastiest real workflow (the one with the approval and the retry) and see which framework makes it ugly.
- **Mistake:** letting framework objects into your domain types — **Instead:** convert at the boundary; your business logic should not import the framework.
- **Mistake:** assuming a framework gives you observability — **Instead:** check that you can see every model call, token count, latency and tool argument; if not, add tracing yourself (Stage 18).
- **Mistake:** picking a framework for its ecosystem of integrations — **Instead:** count how many you will actually use; three integrations are cheaper to write than a dependency you must maintain.
- **Mistake:** mixing two orchestration frameworks in one service — **Instead:** one orchestrator, many plain functions; two frameworks means two tracing models and two upgrade paths.

## Checkpoint

1. Take the plain loop above and add (a) a step budget, (b) a trace log, (c) a hard timeout. Note the line count. That number is your baseline for "framework or not".
2. Port the same workflow to LangGraph and measure: lines of code, time to first successful run, and what you get for free.
3. Write a one-page decision record for a workflow of your choice: requirements, options considered, choice, and the specific cost you accepted. Keep it — this is exactly what Stage 21 asks you to produce.

**Self-check:** if your decision record does not name a cost you accepted, you have written marketing, not a decision.

## Go deeper

- [Ship: A LangGraph Flow](11-ship-langgraph-flow.md) — the hybrid shape, built
- [Agent Design](../21-system-design/03-agent-design.md) — the system-design view of the same choice
- [Tracing](../18-mlops-llmops/05-tracing.md) — what to insist on from any orchestrator

---
[← Previous: CrewAI and AutoGen](09-crewai-autogen.md) · [Stage index](README.md) · [Next: Ship: A LangGraph Flow](11-ship-langgraph-flow.md)
