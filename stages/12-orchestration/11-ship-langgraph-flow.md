# 11. Ship: A LangGraph Flow

> **Stage 12 · Orchestration** — rebuild the harness agent as a state graph with routing, persistence, an approval interrupt and a trace you can read.
>
> **You'll be able to:** ship a graph-based agent with typed state, conditional routing, checkpoint resume and a human approval node, and prove it behaves.
>
> **Prerequisites:** [LangGraph](02-langgraph.md) · [Human in the Loop](../11-agents/08-human-in-the-loop.md) · [Loop Engineering](../11-agents/10-loop-engineering.md)

## Why this exists

You already built an agent loop in Stage 11. It worked, and you could read all of it. What it could not do: survive a deploy mid-run, pause for an approval that arrives hours later, or show a reviewer a diagram of its control flow.

This project is the upgrade, and the discipline is deliberate — you are not replacing the loop with a framework, you are *wrapping* the loop in a graph where the graph owns the parts that need to survive a restart, and the loop keeps doing what it is good at.

The mental model: **the graph owns state and control; the nodes own work.**

## The mental model

```text
                       ┌─────────────┐
        question ─────►│   router    │  (cheap model, ~30 tokens out)
                       └──────┬──────┘
                 simple ┌─────┴─────┐ needs_evidence
                        ▼           ▼
                   ┌────────┐   ┌──────────┐
                   │ answer │   │ retrieve │
                   └────┬───┘   └────┬─────┘
                        │            ▼
                        │      ┌──────────┐
                        │      │  reason  │◄──┐  (agent loop lives here,
                        │      └────┬─────┘   │   bounded by step budget)
                        │           │ tool?   │
                        │           ├─────────┘
                        │           ▼
                        │      ┌──────────┐   side_effect
                        │      │ approve  │◄── interrupt() → human
                        │      └────┬─────┘
                        ▼           ▼
                     ┌─────────────────┐
                     │      END        │   state persisted, trace emitted
                     └─────────────────┘
```

Four nodes, one conditional edge, one interrupt, one checkpointer. If you can draw it, you can debug it.

## Core concepts

### Typed state is the contract

State is a `TypedDict`. Everything a node needs must be in it; nothing a node invents may live outside it. This single rule is what makes resumption possible — a checkpoint is just the state, so anything not in the state is lost on restart.

```python
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["cheap", "full"]
    context: list[str]
    messages: Annotated[list, add_messages]   # reducer: appends instead of replacing
    pending_tool: dict | None
    approved: bool | None
    answer: str
    steps: int
    cost_usd: float
    trace_id: str
```

`add_messages` is a reducer: without it, each node would overwrite the message list. Getting reducers wrong is the most common LangGraph bug, and it presents as "my agent forgot the tool result".

### Routing is an edge, not an if-statement inside a node

The router's job is to decide *which path*, not to do the work. Keeping it tiny (one cheap model call, or even a heuristic) is what makes the cheap path actually cheap.

```python
def route(state: AgentState) -> str:
    # Heuristics first: a classifier call is not free and is not always needed.
    q = state["question"].lower()
    if any(k in q for k in ("policy", "clause", "document", "cite")):
        return "retrieve"
    if len(q) < 40:
        return "cheap"
    return "retrieve"
```

Then wire it declaratively so the diagram and the code agree:

```python
graph.add_conditional_edges("router", route, {"cheap": "answer", "retrieve": "retrieve"})
```

### Interrupts are how approval survives a restart

`interrupt()` suspends the graph and returns control to your application, with state safely in the checkpointer. Resuming with the same `thread_id` continues from exactly that point — even in a different process, hours later.

```python
from langgraph.types import interrupt

def approve(state: AgentState) -> AgentState:
    tool = state.get("pending_tool") or {}
    if tool.get("side_effect") != "write":
        state["approved"] = True
        return state
    decision = interrupt({"tool": tool.get("name"), "args": tool.get("args")})
    state["approved"] = bool(decision.get("approved"))
    return state
```

The application side resumes with `Command(resume={"approved": True})`. That is the whole human-in-the-loop pattern: no polling, no bespoke job table.

### Budgets belong in state

Step and cost counters live in state, so they are checkpointed and inspected like anything else. A graph that cannot say how much it has spent is a graph you cannot run unattended.

## Code

The project, as a package you can actually run.

```text
langgraph-agent/
├── pyproject.toml
├── README.md
├── src/agent/
│   ├── state.py        AgentState + reducers
│   ├── nodes.py        router, retrieve, reason, answer, approve
│   ├── graph.py        wiring + checkpointer
│   ├── tools.py        typed tool registry (from stage 11)
│   └── cli.py          run / resume / inspect
└── tests/
    └── test_graph.py   routing, budget, approval
```

```python
# src/agent/graph.py
from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .nodes import answer, approve, reason, retrieve, route, router
from .state import AgentState

CHECKPOINTS = "runs/checkpoints.sqlite"


def build(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("retrieve", retrieve)
    g.add_node("reason", reason)
    g.add_node("approve", approve)
    g.add_node("answer", answer)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route, {"cheap": "answer", "retrieve": "retrieve"})
    g.add_edge("retrieve", "reason")

    # the agent loop: reason either calls a tool (via approve) or finishes
    g.add_conditional_edges(
        "reason",
        lambda s: "tool" if s.get("pending_tool") else "done",
        {"tool": "approve", "done": "answer"},
    )
    g.add_conditional_edges("approve", lambda s: "again" if s.get("approved") else "answer",
                            {"again": "reason", "answer": "answer"})
    g.add_edge("answer", END)

    return g.compile(checkpointer=checkpointer)


def main() -> None:
    with SqliteSaver.from_conn_string(CHECKPOINTS) as saver:
        app = build(saver)
        cfg = {"configurable": {"thread_id": "demo-1"}}
        out = app.invoke({"question": "What is the refund policy for damaged goods?",
                          "steps": 0, "cost_usd": 0.0}, config=cfg)
        print(out["answer"])
```

```python
# src/agent/cli.py — run, resume after approval, inspect a run
from __future__ import annotations

import json
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .graph import CHECKPOINTS, build


def resume(thread_id: str, approved: bool) -> None:
    with SqliteSaver.from_conn_string(CHECKPOINTS) as saver:
        app = build(saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(cfg)
        if not state.next:
            print(json.dumps({"status": "already_finished", "answer": state.values.get("answer")}))
            return
        out = app.invoke(Command(resume={"approved": approved}), config=cfg)
        print(json.dumps({"status": "resumed", "answer": out.get("answer")}))


if __name__ == "__main__":
    resume(sys.argv[1], sys.argv[2] == "yes")
```

### How to prove it works

| Claim | How you check it | Pass condition |
|---|---|---|
| Routing sends simple questions down the cheap path | Run 20 labelled questions, log `route` per run | ≥90% agreement with your labels |
| A crash does not lose work | Kill the process during `reason`, restart, resume the thread | Run continues from the same step, no duplicate tool call |
| Approval gates write tools only | Run a write-tool request without resuming | Zero side effects recorded in the tool log |
| Budget stops runaway loops | Point a tool at a broken endpoint so `reason` never finishes | Run ends at `max_steps` with `budget_exhausted` in the trace |
| Cheap path is cheaper | Sum `cost_usd` for cheap vs full runs on the same 20 questions | Cheap path at least 3× lower per run |

The duplicate-tool-call row is the one people skip, and it is the one that costs money in production. Resumption must be idempotent: give each tool call a key derived from `thread_id` + step, and make the tool refuse a repeat.

```python
# tests/test_graph.py
def test_cheap_path_skips_retrieval(fake_llm, fake_retriever):
    app = build()
    out = app.invoke({"question": "hi there", "steps": 0, "cost_usd": 0.0})
    assert out["route"] == "cheap"
    assert fake_retriever.calls == 0


def test_budget_exhausted(fake_llm_that_always_calls_a_tool):
    app = build()
    out = app.invoke({"question": "loop forever", "steps": 0, "cost_usd": 0.0})
    assert out["steps"] <= MAX_STEPS
    assert any(e["kind"] == "budget_exhausted" for e in out.get("trace", []))
```

## Real use case

A logistics operator ships this graph as an internal assistant for operations staff. The cheap path handles the majority of questions ("where is my SLA defined", "what does code 47 mean") with a single model call and no retrieval. The full path handles the questions that need policy documents, and the write path — issuing a re-delivery — requires an approval from a shift supervisor who may not look at the queue for twenty minutes.

The design choice that made it work is the interrupt plus SQLite checkpointer: the request sits in state, the supervisor's phone shows a pending approval, and the resume happens in a different worker process. Before the graph, the team had a job table, a polling worker and a bespoke "paused" status that drifted out of sync with the agent's actual position. After, the paused state *is* the checkpoint, and there is no second source of truth.

Their measured result after a month: approval latency unchanged (it was always human-bound), but incidents caused by duplicate re-delivery requests dropped to zero once each tool call was keyed and made idempotent on resume.

## Common mistakes

- **Mistake:** putting business state outside `AgentState` (module globals, closures) — **Instead:** anything needed after a restart belongs in state; if it does not fit, your state schema is wrong.
- **Mistake:** missing reducers on list fields, so each node overwrites messages — **Instead:** use `add_messages` or an explicit `operator.add` reducer and test that a tool result survives one hop.
- **Mistake:** a router that does real work (retrieval, long generation) — **Instead:** keep routing to classification; the cheap path's value is that it is cheap.
- **Mistake:** resuming after a crash and repeating side effects — **Instead:** derive an idempotency key per tool call and make the tool reject duplicates.
- **Mistake:** using an in-memory checkpointer in production — **Instead:** SQLite or Postgres; in-memory dies with the process and your approvals vanish.
- **Mistake:** no test for the approval path — **Instead:** assert zero side effects when a write tool is requested and never approved; that is the test that protects you in a review.

## Checkpoint

1. Build the graph above with the SQLite checkpointer and run a question that needs retrieval. Inspect `app.get_state(cfg)` mid-run and confirm `context` and `steps` are there.
2. Kill the process during `reason`, then resume with the same `thread_id`. Confirm it continues rather than restarting, and that no tool ran twice.
3. Add a cost ceiling to state: when `cost_usd` exceeds a threshold, route straight to `answer` with a "budget exceeded, partial answer" message. Write the test that proves the ceiling holds.

**Self-check:** if your resume test cannot distinguish "continued" from "restarted", your trace is not recording step numbers — fix the trace before anything else.

## Go deeper

- [Trade-offs](10-tradeoffs.md) — why only the approval branch moved to a graph
- [Tracing](../18-mlops-llmops/05-tracing.md) — turning the trace log into something you can query
- [Ship: A Streaming API](../17-ai-backend/09-ship-streaming-api.md) — serving this graph to a browser

---
[← Previous: Trade-offs](10-tradeoffs.md) · [Stage index](README.md) · [Next stage: Fine-Tuning](../13-fine-tuning/README.md)
