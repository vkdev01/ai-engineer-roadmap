# 2. LangGraph

> **Stage 12 · Orchestration** — the agent loop as a state machine: typed state, nodes, edges, conditional edges, checkpointing and interrupts.
>
> **You'll be able to:** express an agent as a graph, route on state, persist runs, and pause for human approval without losing progress.
>
> **Prerequisites:** [LangChain](01-langchain.md) · [Memory and State](../11-agents/05-memory-and-state.md)

## Why this exists

Your Stage 11 loop is a graph already — you just wrote it as `while` and `if`. The moment you need a retry path, a human gate, a parallel branch or a resume after a crash, the implicit graph becomes hard to see and harder to test. LangGraph makes the graph explicit: state is a typed dictionary, nodes are functions that transform it, edges say where to go next.

The value is not that it is a different algorithm. It is that the same agent becomes **inspectable** (draw it), **testable** (call one node with a state), **resumable** (checkpointer) and **interruptible** (a first-class pause). Those four properties are what turn a demo into a service.

The mental model: **nodes transform state, edges decide the next node, the checkpointer remembers every step.**

## The mental model

```text
        ┌─────────┐   route_after_plan    ┌──────────┐
 start─►│  plan   │─────"tools"──────────►│  tools   │──┐
        └─────────┘                       └──────────┘  │
             │ "final"                        ▲         │
             ▼                                └─"more"──┘
        ┌─────────┐      needs_approval   ┌──────────┐
        │ respond │◄───"gated"────────────│ approval │
        └─────────┘                       └──────────┘
             │                                 (interrupt)
             ▼
            END          state: {messages, steps, plan, decision}
                         every node sees the whole state, returns a partial update
```

## Core concepts

### State is a typed dict with reducers

Define state once with a `TypedDict`, or a Pydantic model if you want validation. Fields that accumulate — messages, tool results — need a reducer, because the default behaviour replaces a key rather than appending to it. The reducer is the subtle part: `Annotated[list, add_messages]` appends instead of overwriting, and getting this wrong produces the classic "my history keeps disappearing" bug.

### Nodes are functions from state to partial state

A node receives the full state and returns only the keys it changes. That signature is what makes nodes testable: call `plan({"messages": [...], "steps": 0})` in a unit test, no graph required. Keep nodes small — one decision or one side effect each — and put the policy checks in a node of their own so they appear in the graph.

### Conditional edges are your `if`

`add_conditional_edges(node, router, mapping)` where the router is a pure function of state returning a label, and the mapping turns labels into destination nodes. Every label the router can return must exist in the mapping, or the run fails at runtime — so enumerate the labels as a literal type and let the type checker help.

### Checkpointing gives you resume, replay and time travel

Compile the graph with a checkpointer (in-memory for tests, SQLite or Postgres for real) and a `thread_id` per conversation. Every superstep is saved. You can then resume after a crash, replay a thread to a prior state, and inspect exactly what the state was before a bad decision. This is the feature most teams keep a framework for.

### Interrupts are pauses, not sleeps

`interrupt(payload)` suspends the graph and surfaces the payload to the caller; resuming with `Command(resume=value)` continues from the same node with that value available. That is an approval gate as a primitive: no blocking thread, no lost state. Wire it to the gate policy from [Human in the Loop](../11-agents/08-human-in-the-loop.md).

## Code

A three-node graph with a tool node, a conditional route, a checkpointer and an interrupt for the gated action. Package paths change between versions — check the installed version's docs if an import fails.

```python
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list[dict[str, Any]], add_messages]   # reducer: appends
    steps: int
    pending_tool: dict[str, Any] | None
    approved: bool

def plan(state: State) -> dict[str, Any]:
    """Decide the next action. Replace with a real model call."""
    if state["steps"] >= 6:
        return {"pending_tool": None}
    return {"steps": state["steps"] + 1, "pending_tool": {"name": "search", "args": {"query": "refund policy"}}}

def route_after_plan(state: State) -> Literal["tools", "gate", "respond"]:
    pending = state["pending_tool"]
    if pending is None:
        return "respond"
    return "gate" if pending["name"] == "save_report" else "tools"

def run_tool(state: State) -> dict[str, Any]:
    pending = state["pending_tool"]
    result = f"ran {pending['name']} with {pending['args']}"      # your registry call
    return {"messages": [{"role": "tool", "content": result}], "pending_tool": None}

def gate(state: State) -> dict[str, Any]:
    pending = state["pending_tool"]
    # Suspends the graph; the payload is what your UI renders on the approval card.
    decision = interrupt({"tool": pending["name"], "args": pending["args"],
                          "reason": "side-effecting action"})
    return {"approved": bool(decision.get("approved")), "pending_tool": None}

def respond(state: State) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": "Done. Report saved."}]}

builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("tools", run_tool)
builder.add_node("gate", gate)
builder.add_node("respond", respond)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", route_after_plan,
                             {"tools": "tools", "gate": "gate", "respond": "respond"})
builder.add_edge("tools", "plan")          # the loop edge, explicit
builder.add_edge("gate", "tools")
builder.add_edge("respond", END)

graph = builder.compile(checkpointer=SqliteSaver.from_conn_string("runs.sqlite"))
```

Driving it: one thread id per conversation, and a resume call for the interrupt.

```python
import asyncio

async def main() -> None:
    config = {"configurable": {"thread_id": "ticket-42"}}
    result = await graph.ainvoke({"messages": [], "steps": 0,
                                 "pending_tool": None, "approved": False}, config)

    snapshot = await graph.aget_state(config)
    if snapshot.next:                                  # a node is waiting: it interrupted
        pending = snapshot.tasks[0].interrupts[0].value
        print("needs approval:", pending["tool"], pending["args"])
        result = await graph.ainvoke(Command(resume={"approved": True}), config)

    print(result["messages"][-1]["content"])

asyncio.run(main())
```

The TypeScript package mirrors the Python API closely, which matters when the approval UI and the graph live in the same Node service.

```ts
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";

const State = Annotation.Root({
  steps: Annotation<number>({ reducer: (_prev, next) => next, default: () => 0 }),
  pendingTool: Annotation<string | null>({ reducer: (_p, n) => n, default: () => null }),
});

const graph = new StateGraph(State)
  .addNode("plan", (s) => ({ steps: s.steps + 1, pendingTool: "search" }))
  .addNode("tools", () => ({ pendingTool: null }))
  .addEdge(START, "plan")
  .addConditionalEdges("plan", (s) => (s.pendingTool ? "tools" : "done"), {
    tools: "tools",
    done: END,
  })
  .addEdge("tools", "plan")
  .compile();

export async function runOnce(threadId: string): Promise<number> {
  const out = await graph.invoke({ steps: 0 }, { configurable: { thread_id: threadId } });
  return out.steps;
}
```

## Real use case

An insurance team's claims assistant had to check policy coverage, gather documents, and file a decision — the filing being irreversible. Their Stage 11 loop worked until a claim needed a supervisor's sign-off; the run then died and the agent re-did the document gathering on the retry, sometimes with different documents.

Rebuilt as a LangGraph flow, the checkpointer saved every step and the filing node used `interrupt`. A supervisor's approval resumed the exact thread, so the documents gathered before the pause were the ones filed. The team also gained replay: when a claim was misfiled, they could load the thread and see the state at each superstep instead of reconstructing it from logs.

## Common mistakes

- **Mistake:** forgetting the reducer on a list field — **Instead:** annotate accumulating fields with a reducer and test that two nodes append.
- **Mistake:** one giant node that does everything — **Instead:** one decision or side effect per node; test each in isolation.
- **Mistake:** a conditional route returning a label missing from the mapping — **Instead:** type the labels and cover every branch.
- **Mistake:** no checkpointer in production — **Instead:** SQLite or Postgres with a `thread_id` per conversation.
- **Mistake:** blocking a thread while waiting for a human — **Instead:** `interrupt` and resume with `Command`.
- **Mistake:** no step cap because "the graph terminates" — **Instead:** keep a counter in state and route to `END` when it trips.

## Checkpoint

1. Add a `route_after_tools` that sends the run back to `plan` at most three times, then to `respond`.
2. Swap the SQLite checkpointer for an in-memory one in tests and assert that state survives a simulated restart with SQLite.
3. Add a node that validates the report body before the gate, and route invalid bodies back to `plan`.

**Self-check:** Exercise 1 — keep `steps` in state, and have the router return `"respond"` once `state["steps"] >= 3`, otherwise `"plan"`.

## Go deeper

- [Graph Engineering](03-graph-engineering.md) — designing the graph before you code it.
- [Ship: A LangGraph Flow](11-ship-langgraph-flow.md) — this pattern as a full project.
- [Human in the Loop](../11-agents/08-human-in-the-loop.md) — the policy behind the gate.
- [Tracing](../18-mlops-llmops/05-tracing.md) — reading a graph run in a trace viewer.

---

[← Previous: LangChain](01-langchain.md) · [Stage index](README.md) · [Next: Graph Engineering](03-graph-engineering.md)
