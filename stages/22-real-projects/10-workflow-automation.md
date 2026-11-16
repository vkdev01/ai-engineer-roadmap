# 10. Workflow Automation

> **Stage 22 · Real Projects** — an event-driven system where a trigger starts an agent run, the run takes real actions, and a human approves anything that cannot be undone.
>
> **You'll be able to:** build durable, replayable workflows that survive crashes mid-step, never fire an action twice, and park themselves waiting for approval without holding a process open.
>
> **Prerequisites:** [Job Queues](../17-ai-backend/05-job-queues.md) · [Human in the Loop](../11-agents/08-human-in-the-loop.md)

## Why this exists

Chat is pull: the user asks, the system answers. Most business value is push: a new support ticket, a signed contract, a form submission, a scheduled check that a supplier's terms changed. Those events should start work, and the work should end in an action — a draft, a ticket update, a Slack message, a row in the CRM.

The difference from a chat agent is not intelligence, it is *durability*. A workflow may wait three days for an approval, may crash after the third of five steps, and may receive the same event twice because the source retried. An agent loop built on a `while` statement in a request handler cannot do any of that.

Buyers are operations teams paying people to copy things between systems. They measure hours saved and errors avoided, and they will not tolerate duplicate actions.

Mental model: a workflow is a *persisted state machine*. Each step reads state, does one thing with an idempotency key, and writes state; the queue decides when the next step runs.

## The mental model

```text
event source (webhook, poll, cron)
        |
        v
[ingest: verify signature, dedupe by event id] --> [queue]
        |
        v
[step runner] -- load run state --> [step 1: enrich] --> [step 2: agent decides]
                                          |
                                   [step 3: action]
                                          |
                        risk high? --yes--> [state=awaiting_approval]
                                          |          (process exits; nothing held open)
                                          |                   |
                                          |        human approves -> new event
                                          |                   v
                                          no ----------> [step 4: notify] --> done
                                          |
                            failure --> [retry with backoff] --> [dead letter]
```

Every transition is a row. Nothing important lives only in memory.

## Core concepts

### Scope and non-goals

In scope: webhook and cron triggers, signature verification, event deduplication, a persisted step machine, retries with backoff, an approval gate, a dead-letter queue with replay, and per-run traces.

Non-goals: a general visual builder, arbitrary user-authored code, sub-second latency, and fan-out to thousands of targets. Also non-goal: exactly-once delivery — you get at-least-once delivery plus idempotent steps, which is the only honest version.

### Durability: persist the cursor, not the loop

Store a run row with `current_step`, `state`, `attempt`, and a JSON payload. The runner picks up runs whose `next_run_at` is due, executes exactly one step, and writes the new state in the same transaction as the side-effect record. If the process dies after the side effect but before the state write, the retry re-executes the step — and the idempotency key makes the second execution a no-op. That is why the intent row from [Support Agent](05-support-agent.md) belongs here too.

### Waiting is a state, not a blocked thread

When a run needs approval, set `state = awaiting_approval` and return. The approval is a new event that re-enqueues the run. Never hold a worker, a connection or a `sleep`. This single rule is what makes three-day approvals free and what lets you deploy while a run is waiting.

### Retries, poison events, dead letters

Classify errors. Transient (timeout, rate limit) → retry with exponential backoff and jitter, capped attempts. Permanent (validation, 404, policy refusal) → fail the run and record why. Unknown → retry a couple of times, then dead-letter. A dead-letter queue with the full payload and error, plus a one-click replay, is the operational feature people thank you for. Without it, a schema change silently drops a day of events.

### Data model, stack, milestones

| table | key fields |
|-------|-----------|
| `event` | id, source, external_id, payload, received_at, processed_at |
| `run` | id, workflow, event_id, state, current_step, attempt, next_run_at |
| `step_log` | id, run_id, step, started_at, ended_at, status, error, cost_usd |
| `approval` | id, run_id, requested_at, decided_at, decided_by, decision, note |

Stack, loose: FastAPI for webhooks, Redis or Postgres as the queue (a `SELECT ... FOR UPDATE SKIP LOCKED` loop is enough), Postgres for state, your LLM SDK for decision steps, and the target systems' APIs for actions. Do not add a workflow engine until the hand-rolled one hurts; when it does, the state model transfers directly.

Four weeks: week 1 ingest, dedupe, queue; week 2 steps and idempotent actions; week 3 approvals and resume; week 4 dead letters, replay, traces, evals.

### Eval plan and budget

- **Exactly-once effect rate** — under a fault-injection test that kills the worker between side effect and state write, the count of duplicate actions must be zero.
- **Run success rate** and **mean steps to completion** per workflow.
- **Approval latency** — time in `awaiting_approval`; a workflow that always waits is not automating anything.
- **Dead-letter rate** with a reason breakdown.
- **Cost per run**, split by LLM and external API calls.

Budget, illustrative: a triage workflow might make one model call at 3k input and 400 output tokens per run, plus a few API calls. At volume, the fixed cost is your queue and database, not tokens — so measure cost per 1,000 runs and check the LLM share before optimising prompts. If the LLM share is small, spend your effort on reliability instead.

## Code

The runner is the whole idea: claim a due run, execute one step, commit state and side-effect record together.

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Run:
    id: str
    workflow: str
    state: str                 # running | awaiting_approval | done | failed
    current_step: int
    attempt: int
    payload: dict


async def claim_due(db, limit: int = 20) -> list[Run]:
    """SKIP LOCKED keeps many workers from grabbing the same run."""
    rows = await db.fetch(
        """
        UPDATE run SET state = 'running', locked_at = now()
        WHERE id IN (
            SELECT id FROM run
            WHERE state = 'running' AND next_run_at <= now()
            ORDER BY next_run_at
            FOR UPDATE SKIP LOCKED
            LIMIT $1
        )
        RETURNING id, workflow, state, current_step, attempt, payload
        """,
        limit,
    )
    return [Run(**r) for r in rows]


async def run_one(run: Run, steps: dict[str, list], db) -> None:
    step = steps[run.workflow][run.current_step]
    try:
        result = await step(run.payload)          # side effect inside, idempotent by run.id
    except TransientError as exc:
        delay = min(60 * 2 ** run.attempt, 3600)
        await db.execute(
            "UPDATE run SET attempt = attempt + 1, next_run_at = $2, state = 'running' WHERE id = $1",
            run.id, datetime.utcnow() + timedelta(seconds=delay),
        )
        raise
    except Exception as exc:
        await db.execute("UPDATE run SET state = 'failed', error = $2 WHERE id = $1", run.id, str(exc))
        return

    if result.get("needs_approval"):
        await db.execute("UPDATE run SET state = 'awaiting_approval' WHERE id = $1", run.id)
        return                                   # worker is released; nothing held open

    await db.execute(
        """UPDATE run SET current_step = current_step + 1, attempt = 0,
                          state = $2, next_run_at = now() WHERE id = $1""",
        run.id, "done" if run.current_step + 1 >= len(steps[run.workflow]) else "running",
    )
```

And the ingest side, where duplicate events are stopped cheaply:

```python
async def ingest(source: str, external_id: str, payload: dict, db) -> bool:
    inserted = await db.execute(
        """INSERT INTO event (source, external_id, payload)
           VALUES ($1, $2, $3) ON CONFLICT (source, external_id) DO NOTHING
           RETURNING id""",
        source, external_id, payload,
    )
    return bool(inserted)      # False means the source retried; drop it silently
```

## Real use case

A sales-operations team wants every inbound contract PDF to become a CRM record, a checklist in the deal channel, and a legal-review request when the terms deviate from the standard template. Before: an operations person copied fields by hand, and legal review requests arrived without context.

After: a webhook starts a run; extraction fills the record; the agent compares terms and either completes the checklist or parks the run in `awaiting_approval` for legal with a diff of the deviations. Duplicate webhooks are dropped at ingest, and a schema change that broke a step showed up in the dead-letter queue within minutes rather than at month end.

The design choice that made it work is persisted steps plus approval as a state: three-day legal waits cost nothing, and the team could deploy during them.

Demo script, two minutes: fire a webhook twice and show the second being ignored; watch a run complete; fire a deviating contract and show it parking for approval; approve and show resumption; show a dead letter and replay it. Talking points: at-least-once plus idempotency, why approval is a state, retry classification, and the dead-letter queue.

## Common mistakes

**Mistake:** running the whole workflow in the webhook handler — **Instead:** persist, enqueue, return 200 fast; handlers must not do work.
**Mistake:** sleeping until an approval arrives — **Instead:** park the run as a state and resume on an event.
**Mistake:** claiming exactly-once delivery — **Instead:** at-least-once plus idempotent steps, and test it with fault injection.
**Mistake:** retrying permanent errors — **Instead:** classify; validation failures should fail fast and loudly.
**Mistake:** no dead-letter queue — **Instead:** keep the payload and error, and offer replay.
**Mistake:** trusting webhook payloads — **Instead:** verify the signature and re-fetch the object when the payload drives an irreversible action.

## Checkpoint

1. Add fault injection that crashes the worker after the action succeeds. Prove the retry does not repeat the action.
2. Add a dead-letter table with a replay command, and replay one failed run end to end.
3. Add per-run traces (steps, timings, cost) and compute mean steps to completion for two workflows. Which step is the bottleneck?

**Self-check:** Exercise 1 passes because the action's idempotency key derives from `run.id` and the step index, so the second attempt reads the stored result.

## Go deeper

- [Support Agent](05-support-agent.md) — the same gate applied to a conversation
- [Job Queues](../17-ai-backend/05-job-queues.md) — queue mechanics without a framework
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — retries, timeouts and backpressure at system level
- [Tracing](../18-mlops-llmops/05-tracing.md) — making a run debuggable after the fact

---

[← Previous: Knowledge Assistant](09-knowledge-assistant.md) · [Stage index](README.md) · [Next: Fine-Tuning Pipeline](11-fine-tuning-pipeline.md)
