# 3. Agent Design

> **Stage 21 · System Design** — why a production agent is a state machine with a budget, not a `while True` loop around a model, and how to make its steps idempotent and resumable.
>
> **You'll be able to:** model an agent as explicit states and transitions, enforce step and cost budgets, make tools safe to retry, and resume a run after a crash.
>
> **Prerequisites:** [Agent Architecture](../11-agents/02-agent-architecture.md) · [Memory and State](../11-agents/05-memory-and-state.md)

## Why this exists

The demo agent is a loop: call the model, if it asks for a tool run the tool, repeat until it stops asking. It is twenty lines and it works. Then it runs in production and you discover the loop has no idea what it is doing — it cannot tell "I have gathered enough" from "I am stuck", it re-runs a payment twice after a timeout, it dies mid-run when the pod restarts, and nobody can answer "why did it take 34 steps".

Every one of those is a state problem, and state problems have a known answer: make the states explicit. An agent that is a state machine can be budgeted, inspected, resumed and tested. An agent that is a loop can only be watched.

The mental model: **the model proposes a next action; your code decides whether that action is legal, cheap enough, and already done.**

## The mental model

```text
            ┌──────────┐
   start ──►│ PLANNING │◄──────────────┐
            └────┬─────┘               │
                 │ plan ready          │ tool result
                 ▼                     │
            ┌──────────┐        ┌──────┴──────┐
            │ ACTING   │───────►│ OBSERVING   │
            └────┬─────┘  call  └──────┬──────┘
                 │                    │
        budget/│  no legal action     │ enough?
        step cap▼                     ▼
            ┌──────────┐        ┌───────────┐
            │  FAILED  │        │ COMPOSING │──► DONE
            └──────────┘        └───────────┘
                 ▲
                 └── needs approval ──► WAITING_FOR_HUMAN
```

| Design choice | Free loop | State machine |
|---|---|---|
| Termination | the model stops asking | explicit transitions and a step cap |
| Cost control | post-hoc | budget checked before each transition |
| Resumability | lost on crash | checkpoint the state after each step |
| Testability | run it and hope | assert transition legality directly |
| Human approval | awkward | a state that waits |

## Core concepts

### Explicit states beat emergent behaviour

Define the states you actually have — planning, acting, observing, waiting for approval, composing, done, failed — and the legal transitions between them. Then the model's job shrinks: at each step it chooses *an action from an allowed set*, not the shape of the whole run. That is easier to evaluate, easier to constrain (see [Guardrails](../20-ai-security/09-guardrails.md)) and much easier to debug.

The practical rule: **the model proposes, the machine disposes.** If the model returns an action that is not legal in the current state, that is a handled case with a log line, not an exception.

### Budgets on every dimension

A step cap alone is not enough. Enforce all four:

- **Steps** — max transitions per run (the cheapest guard against loops).
- **Wall clock** — max seconds, checked before each step, with the whole run cancellable.
- **Tokens** — cumulative input plus output across all model calls; this is your cost proxy.
- **Tool spend** — if a tool costs money (a paid API, a search quota), it has its own cap.

Budgets are checked *before* a transition, so the run stops cleanly with a partial result and a reason, rather than being killed mid-call. "Return the best answer so far, marked incomplete" is a much better product behaviour than a 500.

### Idempotent tools, because retries happen

Any step in a distributed system may run twice: a timeout where the work succeeded, a retry, a resumed run replaying from a checkpoint. So every tool must be safe to call twice with the same arguments:

- **Reads** are naturally idempotent. Cache them by argument hash to also save money.
- **Writes** need an idempotency key derived from `(run_id, step_index, tool, args_hash)` and honoured by the downstream system, or an upsert rather than an insert.
- **Side effects with no natural key** (sending an email, charging a card) get a dedupe table on your side, plus an approval gate.

This is the single most common production bug in agent systems: an agent that re-charges a customer because a tool call timed out after the charge succeeded.

### Resumability: checkpoint the state, not the transcript

After each transition, persist the state: current state, plan, completed steps with their results, budget consumed, and a version of the prompt and tool schema. Then a crash is recoverable — a worker picks the run up, validates that the tool schema version still matches, and continues. If it does not match, fail the run rather than continuing with mismatched tools.

Keep checkpoints small. Store tool *results* by reference (an object-store key) rather than inline when they are large, and redact before persisting (see [PII Redaction](../20-ai-security/04-pii-redaction.md)).

## Code

A state machine with legal transitions, budget enforcement before each step, and an idempotency key handed to the tool. The model's choice is validated against the allowed set for the current state.

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict


class State(str, Enum):
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPOSING = "composing"
    DONE = "done"
    FAILED = "failed"


LEGAL: dict[State, set[State]] = {
    State.PLANNING: {State.ACTING, State.COMPOSING, State.FAILED},
    State.ACTING: {State.OBSERVING, State.WAITING_FOR_HUMAN, State.FAILED},
    State.OBSERVING: {State.PLANNING, State.COMPOSING, State.FAILED},
    State.WAITING_FOR_HUMAN: {State.ACTING, State.FAILED},
    State.COMPOSING: {State.DONE, State.FAILED},
    State.DONE: set(),
    State.FAILED: set(),
}


@dataclass
class Budget:
    max_steps: int = 12
    max_tokens: int = 60_000
    used_steps: int = 0
    used_tokens: int = 0

    def exhausted(self) -> str | None:
        if self.used_steps >= self.max_steps:
            return "step budget"
        if self.used_tokens >= self.max_tokens:
            return "token budget"
        return None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    args: dict[str, str] = field(default_factory=dict)


ALLOWED_IN_STATE: dict[State, frozenset[str]] = {
    State.PLANNING: frozenset({"search_docs"}),
    State.ACTING: frozenset({"search_docs", "get_order_status"}),
    State.OBSERVING: frozenset(),
    State.WAITING_FOR_HUMAN: frozenset(),
    State.COMPOSING: frozenset(),
    State.DONE: frozenset(),
    State.FAILED: frozenset(),
}


def transition(state: State, target: State) -> State:
    if target not in LEGAL[state]:
        raise ValueError(f"illegal transition {state} -> {target}")
    return target


def idempotency_key(run_id: str, step: int, action: Action) -> str:
    blob = json.dumps(action.model_dump(), sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return f"{run_id}:{step}:{digest}"
```

The run loop below enforces the budget before each step, refuses illegal actions, and checkpoints after every transition so a crash is recoverable.

```python
from __future__ import annotations


async def run_agent(run_id: str, task: str, deps, budget: Budget | None = None) -> dict:
    budget = budget or Budget()
    state = State.PLANNING
    await deps.checkpoints.save(run_id, {"state": state, "steps": []})

    while state not in (State.DONE, State.FAILED):
        reason = budget.exhausted()
        if reason:
            # Stop cleanly with a partial result rather than being killed.
            return {"status": "incomplete", "reason": reason, "state": state}

        proposal = await deps.models.propose(state, task, deps.tool_schemas)
        budget.used_tokens += proposal.usage_tokens

        if proposal.action is None:
            state = transition(state, State.COMPOSING)
            continue

        action = proposal.action
        if action.tool not in ALLOWED_IN_STATE[state]:
            deps.audit.deny(run_id, state, action.tool)  # handled, not raised
            state = transition(state, State.COMPOSING)
            continue

        key = idempotency_key(run_id, budget.used_steps, action)
        state = transition(state, State.ACTING)
        result = await deps.tools.call(action.tool, action.args, idempotency_key=key)

        budget.used_steps += 1
        state = transition(state, State.OBSERVING)
        await deps.checkpoints.save(
            run_id,
            {"state": state, "budget": budget.__dict__, "last_result": result.ref},
        )
        state = transition(state, State.PLANNING)

    return {"status": "ok" if state is State.DONE else "failed", "state": state}
```

## Real use case

**A procurement research agent that compared supplier quotes.** It searched a supplier catalogue, fetched quote documents and produced a comparison table. In the first version — a free loop with no cap — a supplier's catalogue page that returned an empty result set for a valid query caused the agent to retry the same search with slightly different phrasing indefinitely. Two runs consumed a large share of the monthly model budget in one afternoon.

The rebuild made the loop a machine. States became explicit; the search tool could only be called from `PLANNING` and `ACTING`; a step cap of 12 and a token budget of 60,000 stopped the run cleanly with an "incomplete: step budget" result the UI could explain. Tool calls took an idempotency key so a retried quote fetch did not duplicate the row. Checkpoints after each step meant a worker restart resumed the run instead of restarting it — which mattered because the average run took several minutes. The bug that started it all became a test: an empty result set must terminate in `COMPOSING`, not in another search.

## Common mistakes

- **Mistake:** a `while True` loop with no step cap — **Instead:** explicit states, a step cap, and a token budget checked before each transition.
- **Mistake:** killing a run mid-call when the budget expires — **Instead:** check before the step and return a partial result with a reason.
- **Mistake:** tools that insert rather than upsert — **Instead:** idempotency keys from `(run_id, step, args_hash)` plus dedupe on your side.
- **Mistake:** storing the full transcript as the checkpoint — **Instead:** store state, budget, and references to results; redact what you persist.
- **Mistake:** letting the model choose any tool from any state — **Instead:** an allowlist per state, with a deny path that logs and terminates.

## Checkpoint

1. Write the state set and legal transition table for an agent you have built. Add one assertion per illegal transition.
2. Add a token budget and a step cap, then craft an input that hits each. Confirm the run returns a partial result rather than erroring.
3. Make one write tool idempotent with a key, then call it twice with the same arguments and confirm only one effect occurred.

**Self-check:** The transition table should be a dict of allowed targets per state, and each illegal pair should raise or log a deny — the point is that an action which is wrong for the current state is a handled case, not an exception.

## Go deeper

- [Model Routing](04-model-routing.md) — choosing a model per step instead of per app.
- [Fault Tolerance](07-fault-tolerance.md) — retries, breakers and degraded modes for each step.
- [Loop Engineering](../11-agents/10-loop-engineering.md) — the loop mechanics in more depth.
- [Human in the Loop](../11-agents/08-human-in-the-loop.md) — designing the approval state.

---
[← Previous: RAG Design at Scale](02-rag-design.md) · [Stage index](README.md) · [Next: Model Routing](04-model-routing.md)
