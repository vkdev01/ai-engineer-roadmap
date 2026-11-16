# 7. Multi-Agent Patterns

> **Stage 11 · AI Agents** — supervisor, handoff, debate and fan-out/fan-in: the four topologies worth knowing, and what each one costs you in tokens, latency and debuggability.
>
> **You'll be able to:** implement a supervisor and a fan-out/fan-in graph in plain Python, and justify the topology you chose with numbers rather than vibes.
>
> **Prerequisites:** [Single-Agent Patterns](06-single-agent.md) · [Reasoning and Planning](03-reasoning-planning.md)

## Why this exists

Multi-agent systems are sold as "a team of AI employees". In practice they are message passing between several prompts, and every hop loses fidelity: the specialist does not see what the supervisor saw, the summary of the summary drifts, and a failure at hop three is invisible from hop one. They are worth it in a narrow set of cases, and those cases are worth naming precisely.

The honest framing: **multi-agent is a context-management technique, not an intelligence technique.** You split because one context cannot hold everything or because subtasks can genuinely run in parallel — not because more agents sound smarter. The model that runs each sub-agent is usually the same model you could have called once.

## The mental model

```text
supervisor           handoff              debate            fan-out / fan-in
─────────            ───────              ──────            ─────────────────
   [sup] ──► A        [triage] ─► [spec]   A ◄─critique─► B   task ─┬─► A ─┐
     ▲  └──► B            (control        └──judge picks──┘        ├─► B ─┼─► merge
     └── results           moves)                                └─► C ─┘
 1 context per       1 shared thread     2+ full transcripts    N contexts,
 worker, sup holds   ownership passes    per round              results merged
 the plan            to one agent
```

## Core concepts

### Supervisor: one planner, many workers

A supervisor holds the plan and the user's goal. It delegates a narrow question to a worker, receives a short result, and decides the next delegation. The key discipline: **workers never talk to each other and never see the user's whole history.** They get a task statement and their own tools. This keeps each worker's context small and makes the supervisor's trace the single source of truth for what happened.

### Handoff: transfer ownership, do not copy

In a handoff, one agent decides it is no longer the right owner and passes the conversation, with state, to another. Unlike delegation, the receiving agent now owns the goal. This is right for triage ("this is a legal question, not billing") and wrong for research subtasks, where ownership should stay with the planner. The failure mode is ping-pong: two agents handing the same conversation back and forth, which you stop with a hop counter.

### Debate: two answers, one judge

Two agents answer independently, then critique each other, and a judge picks or synthesises. It genuinely helps on tasks with checkable reasoning — arithmetic, consistency checks, policy interpretation. It does not help on open-ended writing, where the judge simply picks the more confident style. Debate costs two to four times the tokens of a single pass, so reserve it for high-value, verifiable decisions.

### Fan-out/fan-in: parallelism with a merge contract

When subtasks are independent — summarise twelve documents, check five suppliers — run them concurrently and merge the results. The engineering is all in the merge: define the output shape each worker must return, cap concurrency, and decide what happens when one worker fails. Partial results with a stated gap beat a failed batch.

### Count the cost before you commit

Multi-agent multiplies tokens (each agent has its own system prompt and context), adds latency (sequential hops), and divides observability (N traces, one user). Write those three numbers down for the topology you plan to ship. If a single agent with better tools is within a small margin on quality, ship the single agent.

## Code

A supervisor that delegates with narrow task statements, and a fan-out/fan-in runner with a merge contract. Both reuse the loop you already wrote; nothing new is needed except the plumbing between agents.

```python
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

Worker = Callable[[str], Awaitable[str]]   # task statement in, short result out

async def supervisor(goal: str, workers: dict[str, Worker], model: Any,
                     max_delegations: int = 6) -> str:
    notes: list[str] = []
    for _ in range(max_delegations):
        prompt = (f"Goal: {goal}\nNotes so far:\n" + ("\n".join(notes) or "(none)") +
                  f"\nWorkers: {sorted(workers)}\n"
                  "Reply JSON: {\"worker\": name, \"task\": str} or {\"final\": str}.")
        reply = await model.complete([{"role": "user", "content": prompt}], [])
        decision = reply["decision"]

        if "final" in decision:                       # supervisor owns the answer
            return decision["final"]
        name, task = decision["worker"], decision["task"]
        if name not in workers:                       # a bad pick must not crash the run
            notes.append(f"delegation to unknown worker {name} rejected")
            continue

        # The worker sees only this task statement: no goal, no user history.
        result = await workers[name](task)
        notes.append(f"[{name}] {task} -> {result}")

    return "Stopped: delegation budget exhausted. Notes:\n" + "\n".join(notes)

async def fan_out(tasks: list[str], worker: Worker, merge: Callable[[list[Any]], str],
                  concurrency: int = 4) -> str:
    gate = asyncio.Semaphore(concurrency)             # protect rate limits
    async def one(task: str) -> dict[str, Any]:
        async with gate:
            try:
                return {"task": task, "ok": True, "result": await worker(task)}
            except Exception as exc:                  # partial results are the contract
                return {"task": task, "ok": False, "result": f"{type(exc).__name__}: {exc}"}
    results = await asyncio.gather(*(one(t) for t in tasks))
    return merge(list(results))

def merge_with_gaps(results: list[dict[str, Any]]) -> str:
    done = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    body = "\n".join(f"- {r['task']}: {r['result']}" for r in done)
    gap = f"\nGaps ({len(failed)}): " + "; ".join(r["task"] for r in failed) if failed else ""
    return f"Completed {len(done)}/{len(results)}\n{body}{gap}"
```

The TypeScript shape matters when the supervisor's delegations are shown live — a UI that renders the fan-out as a set of cards, each resolving independently.

```ts
export type Delegation = { worker: string; task: string; status: "running" | "done" | "failed" };

export async function fanOut(
  tasks: string[],
  worker: (task: string) => Promise<string>,
  onUpdate: (d: Delegation) => void,
): Promise<Delegation[]> {
  const cards: Delegation[] = tasks.map((task) => ({ worker: "doc", task, status: "running" }));
  cards.forEach(onUpdate);
  await Promise.all(
    cards.map(async (card) => {
      try {
        await worker(card.task);
        card.status = "done";
      } catch {
        card.status = "failed";
      }
      onUpdate(card);
    }),
  );
  return cards;
}
```

## Real use case

A research team built a "multi-agent analyst" with a planner, a searcher, a summariser and a critic. It produced plausible reports, took about four times the tokens of a single agent, and — the real problem — nobody could explain why a report was wrong, because the evidence was in the searcher's context and the claim was in the summariser's.

They kept two of the four. The planner stayed, because the task genuinely decomposed into independent questions. The searcher stayed as a fan-out worker. The summariser and critic were removed: the planner now receives worker results with source ids and writes the report itself, so every claim and its evidence are in one context. Reports got easier to audit, and cost fell to roughly half.

## Common mistakes

- **Mistake:** giving workers the full user history — **Instead:** pass a narrow task statement and return one line.
- **Mistake:** letting workers call each other — **Instead:** route every cross-worker need through the supervisor.
- **Mistake:** no hop or delegation cap — **Instead:** cap delegations and handoffs, and return the notes on exhaustion.
- **Mistake:** debate for open-ended writing — **Instead:** reserve debate for verifiable decisions.
- **Mistake:** a fan-out with no merge contract — **Instead:** fixed output shape per worker, plus an explicit gaps section.
- **Mistake:** adopting multi-agent before measuring the single agent — **Instead:** keep the single-agent baseline in your eval set.

## Checkpoint

1. Add a hop counter to a two-agent handoff loop and stop with an explicit message when it exceeds three.
2. Change `merge_with_gaps` to include source ids, and check the final answer cites them.
3. Run the same ten tasks with a single agent and with your supervisor, and record tokens, latency and steps for both in a table.

**Self-check:** Exercise 1 — keep a `hops` integer in the shared state, increment on each handoff, and return a `handoff_limit` result once it passes the cap.

## Go deeper

- [Human in the Loop](08-human-in-the-loop.md) — putting a person between agents.
- [CrewAI and AutoGen](../12-orchestration/09-crewai-autogen.md) — frameworks for role-based and conversational multi-agent.
- [Multi-Agent Research](../22-real-projects/04-multi-agent-research.md) — the same patterns in a real project.
- [Agent Evals](../15-evals/06-agent-evals.md) — measuring quality per topology.

---

[← Previous: Single-Agent Patterns](06-single-agent.md) · [Stage index](README.md) · [Next: Human in the Loop](08-human-in-the-loop.md)
