# 5. Memory and State

> **Stage 11 · AI Agents** — conversation state versus long-term memory: checkpoints you can resume, summaries that keep the thread, and retrieval memory that outlives the session.
>
> **You'll be able to:** separate the state of a run from the memory of a user, and persist both so an agent can pause, resume and remember.
>
> **Prerequisites:** [Tool Calling](04-tool-calling.md) · [Context Compression](../08-prompt-and-context/04-context-compression.md)

## Why this exists

An agent that forgets is not an assistant, it is a form. Two different things get called "memory" and mixing them up causes most of the bugs. **State** is what a single run knows: the task, the steps taken, the pending tool call. **Memory** is what survives across runs: the user's preferences, facts you learned last week, documents they uploaded.

State must be exact and resumable — a checkpoint you can reload after a crash or an approval pause. Memory must be selective and retrievable — you cannot put a year of history in the context, so you store it and fetch the relevant slice. An agent needs both, and neither is the chat transcript.

The mental model: **state is the run's stack; memory is the user's library.**

## The mental model

```text
 run state (short-lived, exact)        memory (long-lived, selective)
 ─────────────────────────────        ──────────────────────────────
 task, step index, ledger             profile facts (key → value)
 pending tool call / approval         episode summaries
 scratchpad of observations           retrieved documents (vectors)
        │                                     │
        └──── checkpoint after each step ─────┘
                     │                            │
                     ▼                            ▼
        resume the same run             assemble into context on demand
```

| Question | State | Memory |
|---|---|---|
| Lifetime | one run | many runs |
| Storage | JSON checkpoint, keyed by run id | DB table plus a vector index |
| Failure if lost | run restarts | assistant seems forgetful |
| Failure if too big | slow, expensive steps | irrelevant context, worse answers |

## Core concepts

### Checkpoint after every step, not at the end

Write the full loop state — messages, step index, budget spent, ledger of tool calls, pending approval — to durable storage after each step. This single habit gives you crash recovery, resumable approval gates, and time-travel debugging, because a checkpoint is a run you can replay. Keep the checkpoint small and versioned: a `schema_version` field costs nothing and saves a migration.

### Summarise old turns, keep recent ones verbatim

A rolling summary is the cheapest form of memory compression. Keep the last few turns exactly as they were and replace everything older with a short paragraph written by the model: what the user wants, what has been established, what is still open. Regenerate it every N turns, never mid-step, and store both the summary and the turn index it covers so you can rebuild.

### Retrieval memory: store facts, fetch by relevance

Long-term memory is a small RAG system over the user's own history. Write facts as short, self-contained statements with a timestamp and a source ("prefers invoices as PDF, told on 12 Nov"), embed them, and retrieve the top few at the start of each run. Self-contained matters: "he said it was fine" is useless a month later.

### Write memory deliberately, never automatically

The most common failure is dumping the whole transcript into a vector store. It grows without bound, retrieves near-duplicates, and drags stale contradictions into the context. Instead, define a small set of write triggers — a user states a preference, a task completes, an explicit "remember this" — and write only then. Reading is automatic; writing is a decision.

### The ledger is the part models forget

Models lose track of what they already did, especially after summarisation. Keep a structured ledger outside the conversation: ids checked, files written, tools called with which arguments. Put the ledger in the context, not the transcript. This is why the no-progress rule in [Loop Engineering](10-loop-engineering.md) works.

## Code

A checkpointed run state and a two-tier context assembler. Storage is a JSON file here; in production it is a row keyed by `run_id` in Postgres or Redis — see [Redis Caching](../17-ai-backend/04-redis-caching.md).

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class RunState:
    schema_version: int
    run_id: str
    task: str
    step: int = 0
    turns: list[dict[str, Any]] = field(default_factory=list)   # recent, verbatim
    summary: str = ""                                           # older, compressed
    summary_covers_upto: int = 0                                # turn index covered
    ledger: dict[str, list[str]] = field(default_factory=dict)  # tool -> keys seen
    pending_approval: dict[str, Any] | None = None

def save(state: RunState, root: str = "runs") -> None:
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f"{state.run_id}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(asdict(state), fh)
    os.replace(tmp, os.path.join(root, f"{state.run_id}.json"))  # atomic: no half writes

def load(run_id: str, root: str = "runs") -> RunState:
    with open(os.path.join(root, f"{run_id}.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != 1:
        raise ValueError("checkpoint from an older schema; migrate before resuming")
    return RunState(**data)

def assemble_context(state: RunState, keep_recent: int = 6) -> list[dict[str, Any]]:
    ledger_text = "; ".join(f"{k}: {len(v)} seen" for k, v in state.ledger.items())
    return [
        {"role": "system", "content": f"Task summary: {state.summary}\nProgress: {ledger_text}"},
        {"role": "user", "content": state.task},
        *state.turns[-keep_recent:],
    ]
```

Now the memory side: write facts deliberately, read them by relevance at the start of a run.

```python
from datetime import datetime, timezone
from pydantic import BaseModel

class Fact(BaseModel):
    user_id: str
    text: str                      # must be self-contained
    source: str                    # "chat" | "upload" | "tool"
    created_at: str

def remember(user_id: str, text: str, source: str, store: Any) -> None:
    if len(text) < 15 or text.lower().startswith("ok"):   # cheap noise filter
        return
    fact = Fact(user_id=user_id, text=text, source=source,
                created_at=datetime.now(timezone.utc).isoformat())
    store.upsert(fact)             # embeds and indexes the text

async def recall(user_id: str, query: str, store: Any, k: int = 5) -> str:
    hits = store.search(user_id=user_id, query=query, k=k)
    if not hits:
        return "No stored memory for this user yet."
    return "Known facts:\n" + "\n".join(f"- ({h.source}) {h.text}" for h in hits)
```

The TypeScript side is the shape a client needs: a run id it can reconnect to and resume after an approval pause.

```ts
export type Checkpoint = {
  runId: string;
  step: number;
  summary: string;
  pendingApproval: { tool: string; args: Record<string, unknown> } | null;
};

export async function resume(
  checkpoint: Checkpoint,
  approve: (tool: string, args: Record<string, unknown>) => Promise<boolean>,
): Promise<Checkpoint | null> {
  if (!checkpoint.pendingApproval) return null;          // nothing to resume
  const ok = await approve(checkpoint.pendingApproval.tool, checkpoint.pendingApproval.args);
  return { ...checkpoint, pendingApproval: null, summary: `${checkpoint.summary}\napproval:${ok}` };
}
```

## Real use case

A clinic's front-desk agent books appointments. It handled a booking over six turns and then lost the thread when the patient asked "and can you send it to my wife's number too?" — the phone number from turn two had fallen out of the window. Separately, when the appointment system needed a staff approval, the run died and the patient had to start over.

The fix was structural. Recent turns stayed verbatim, older turns became a summary that named the patient, the doctor and the slot, and the facts (phone numbers, preferred language) moved into retrieval memory so they could be recalled by relevance rather than by position in a window. The pending approval became a checkpoint field, so the run resumes the moment staff click approve — no restart, no repeated questions.

## Common mistakes

- **Mistake:** treating the chat transcript as memory — **Instead:** store facts and summaries, and fetch what is relevant.
- **Mistake:** checkpointing only at the end of a run — **Instead:** checkpoint after each step, atomically.
- **Mistake:** auto-writing every turn into the vector store — **Instead:** define write triggers and filter noise before embedding.
- **Mistake:** storing pronouns and fragments — **Instead:** write self-contained statements that make sense weeks later.
- **Mistake:** keeping the ledger inside the transcript where summarisation eats it — **Instead:** keep it as structured state and inject it each turn.

## Checkpoint

1. Add a `turn_count` field and regenerate `summary` every six turns, keeping the last six turns verbatim.
2. Make `recall` return nothing when the best hit is below a similarity threshold, and check that the agent does not hallucinate a preference.
3. Add a `pending_approval` branch to a run, save the checkpoint, and resume it in a second process using only the run id.

**Self-check:** Exercise 1 — increment `turn_count` per appended turn, and when `turn_count % 6 == 0`, summarise `turns[:-6]`, extend `summary`, and update `summary_covers_upto`.

## Go deeper

- [Human in the Loop](08-human-in-the-loop.md) — interrupts that depend on these checkpoints.
- [Loop Engineering](10-loop-engineering.md) — using the ledger to detect no progress.
- [Parent-Child Retrieval](../10-rag/08-parent-child.md) — a useful shape for memory retrieval too.
- [LangGraph](../12-orchestration/02-langgraph.md) — checkpointing as a first-class feature.

---

[← Previous: Tool Calling](04-tool-calling.md) · [Stage index](README.md) · [Next: Single-Agent Patterns](06-single-agent.md)
