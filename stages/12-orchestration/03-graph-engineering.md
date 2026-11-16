# 3. Graph Engineering

> **Stage 12 · Orchestration** — designing an agent as a graph before you write it: naming the states, choosing the transitions, planning the failure edges, and knowing when a graph is the wrong shape.
>
> **You'll be able to:** draw an agent as a state machine on paper, defend every edge, and catch design errors before they become bugs.
>
> **Prerequisites:** [LangGraph](02-langgraph.md) · [Loop Engineering](../11-agents/10-loop-engineering.md)

## Why this exists

Most agent bugs are graph bugs wearing a prompt costume. The agent re-plans forever because there is no edge to `done` from the failure state. Two nodes both write the report because the graph has no single owner for that output. A run hangs because a node waits on something nobody routes to.

Graph engineering is the discipline of drawing the thing first: a small set of named states, a transition for every event, and an explicit failure edge out of every state. It takes twenty minutes with a pen and saves days of debugging, because the questions you cannot answer on paper are exactly the ones that will bite at runtime.

The mental model: **if you cannot draw it, you cannot operate it.**

## The mental model

```text
 every state needs three exits:

        ┌──────────────┐
        │    STATE     │
        └──┬────┬────┬─┘
   success │    │fail│   unknown / needs human
           ▼    ▼    ▼
       next   error  escalate
        │      │       │
        └──────┴───────┴──► all three lead somewhere. no dangling state.
```

| Question | Bad answer | Good answer |
|---|---|---|
| What are the states? | "the agent" | `intake`, `gather`, `check`, `act`, `report` |
| Who owns each output? | "whoever writes it" | exactly one node per artefact |
| What ends the run? | "the model decides" | a counter or an explicit terminal state |
| What if a node fails? | unhandled | an error edge to a named state |

## Core concepts

### Name states as nouns, transitions as events

A state is a situation the run can be in: `awaiting_documents`, `policy_checked`, `awaiting_approval`. An event is what moves it: `documents_found`, `policy_failed`, `approved`. Naming them separately prevents the classic confusion where a node called `check_policy` is really a state plus a transition plus a side effect. Once the names are clean, the graph almost draws itself.

### One owner per artefact

Every output the run produces — the report, the filed decision, the email — should be written by exactly one node. When two nodes can write the same artefact you get duplicates and races, and no amount of prompt discipline fixes it. This is the graph version of a single-writer rule.

### Failure edges are first-class

For each state, ask what happens on: a tool error, a timeout, an empty result, a validation failure, and an unparseable model reply. Each answer is an edge. Teams that skip this end up with an agent that works on the demo data and stalls on the first malformed PDF.

### Fan-out only where the branches are independent

Parallel edges are worth it when sub-results do not affect each other — five document summaries, four supplier lookups. They are a bug when one branch's result should change another's inputs, because the merge will silently pick a winner. If branches must talk, they are sequential states, not parallel ones.

### When a graph is the wrong shape

Three cases. **Truly linear work** — one prompt, one parse — needs no graph; a function is clearer. **A single free-form conversation** with no fixed stages is often better as a loop with tools than as five rigid states. **Highly dynamic planning**, where the model invents the steps, fits a loop better; a graph that must be edited for every new task shape is a maintenance tax.

## Code

The design artefact first: a tiny graph spec you can review in a pull request before any implementation. This is deliberately plain data — it is a design document that a script can validate.

```python
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class State:
    name: str
    owner_of: str | None = None                     # artefact this state writes
    on_success: str | None = None
    on_error: str = "escalate"
    terminal: bool = False

@dataclass
class GraphSpec:
    states: dict[str, State] = field(default_factory=dict)
    start: str = ""

    def add(self, state: State) -> "GraphSpec":
        self.states[state.name] = state
        return self

    def validate(self) -> list[str]:
        problems: list[str] = []
        for s in self.states.values():
            if not s.terminal and s.on_success is None:
                problems.append(f"{s.name}: no success edge")
            for target in filter(None, (s.on_success, s.on_error)):
                if target not in self.states:
                    problems.append(f"{s.name}: edge to unknown state {target}")
        if self.start not in self.states:
            problems.append(f"start state {self.start!r} not defined")

        owners = [s.owner_of for s in self.states.values() if s.owner_of]
        for artefact in set(owners):
            if owners.count(artefact) > 1:          # single-writer rule
                problems.append(f"artefact {artefact!r} has more than one owner")
        return problems

spec = (GraphSpec(start="intake")
        .add(State("intake", on_success="gather"))
        .add(State("gather", on_success="check"))
        .add(State("check", on_success="act", on_error="report"))
        .add(State("act", owner_of="decision", on_success="report", on_error="escalate"))
        .add(State("report", owner_of="report", on_success="done"))
        .add(State("escalate", on_success="report"))
        .add(State("done", terminal=True)))

assert spec.validate() == [], spec.validate()
```

Second, the same spec rendered as a Mermaid diagram for the pull request. Reviewers catch more in a picture than in a diff, and the diagram cannot drift because it is generated.

```python
def to_mermaid(spec: GraphSpec) -> str:
    lines = ["stateDiagram-v2"]
    for s in spec.states.values():
        if s.terminal:
            continue
        if s.on_success:
            lines.append(f"    {s.name} --> {s.on_success}: ok")
        lines.append(f"    {s.name} --> {s.on_error}: error")
    lines.append(f"    [*] --> {spec.start}")
    return "\n".join(lines)

print(to_mermaid(spec))
```

In TypeScript, the same spec drives a runtime router, so the design and the code cannot diverge — the spec is the single source of truth for legal transitions.

```ts
export type Transition = { from: string; event: "ok" | "error"; to: string };

export function makeRouter(spec: { states: string[]; transitions: Transition[] }) {
  const table = new Map<string, string>();
  for (const t of spec.transitions) table.set(`${t.from}:${t.event}`, t.to);

  return function next(state: string, event: "ok" | "error"): string {
    const target = table.get(`${state}:${event}`);
    if (!target) throw new Error(`no transition from ${state} on ${event}`);
    if (!spec.states.includes(target)) throw new Error(`unknown state ${target}`);
    return target;
  };
}
```

## Real use case

A logistics team built a delivery-exception agent: read the exception, check the customer's SLA, choose a remedy, notify, and log. Their first implementation was a single ReAct loop. It worked until a remedy involved a partial refund — then the agent sometimes both refunded and issued a voucher, because nothing in the loop owned the "remedy" decision.

Redrawn as a graph, `choose_remedy` became the single owner of the remedy artefact and every path had to pass through it. Two states were added that nobody had thought about: `sla_unknown` (escalate, do not guess) and `notify_failed` (retry once, then log and stop). Both had occurred in production but had been invisible, showing up as vague "agent gave up" reports.

## Common mistakes

- **Mistake:** drawing only the happy path — **Instead:** an error edge out of every state, with a named destination.
- **Mistake:** two nodes writing the same artefact — **Instead:** enforce a single-writer rule in a validator.
- **Mistake:** mixing state names with action names — **Instead:** states are nouns, transitions are events.
- **Mistake:** parallel branches that need each other's results — **Instead:** make them sequential states.
- **Mistake:** a graph per new task shape — **Instead:** if the model invents the steps, use a loop, not a graph.
- **Mistake:** a diagram in the wiki that nobody updates — **Instead:** generate the diagram from the spec.

## Checkpoint

1. Extend `validate` to detect unreachable states (no inbound edge except the start) and cycles with no exit.
2. Generate a Mermaid diagram for the Stage 11 harness agent and compare it with the code you wrote.
3. Add a `needs_human` state to the spec and route every `on_error` through it, then note how many states now need a manual decision.

**Self-check:** Exercise 1 — compute the set of states reachable from `start` by walking edges, then report any state not in that set as unreachable.

## Go deeper

- [LangGraph](02-langgraph.md) — turning the spec into a runnable graph.
- [Ship: A LangGraph Flow](11-ship-langgraph-flow.md) — a graph built spec-first.
- [Agent Design](../21-system-design/03-agent-design.md) — graph thinking at system scale.
- [Agent Architecture](../11-agents/02-agent-architecture.md) — the loop this discipline formalises.

---

[← Previous: LangGraph](02-langgraph.md) · [Stage index](README.md) · [Next: LlamaIndex](04-llamaindex.md)
