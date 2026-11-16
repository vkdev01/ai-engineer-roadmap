# 5. Support Agent

> **Stage 22 · Real Projects** — an agent that answers, acts, and hands off — with idempotent tool calls, confirmation before anything irreversible, and a human who receives the whole context.
>
> **You'll be able to:** build a support agent that can take real actions (refunds, address changes, ticket updates) without duplicating them, and escalate to a human without making the customer repeat themselves.
>
> **Prerequisites:** [Human in the Loop](../11-agents/08-human-in-the-loop.md) · [Tool Calling](../07-llm-apis/05-tool-calling.md)

## Why this exists

Support is where agents meet money. A read-only assistant that suggests an answer is a toy; the value is in doing things — issuing the refund, rescheduling the delivery, closing the ticket. The moment an agent can write, three new problems appear that no prompt solves.

It retries. The tool call times out after the refund succeeded, the loop retries, the customer is refunded twice. It is talked into things. A user says "my manager approved it", and the agent skips the policy check. And it gives up badly: it escalates by saying "let me connect you to a human", the customer lands in a queue with no history, and repeats everything.

Buyers are support and CX teams measured on resolution time and cost per ticket. They will not deploy anything that can double-pay a refund.

Mental model: the agent *proposes*, the runtime *executes*. Every write goes through a gate that knows the risk level, records intent before acting, and can be replayed safely.

## The mental model

```text
customer message
      |
      v
[classify: question | action | complaint | unknown]
      |
      v
[agent loop] --proposes tool call--> [policy gate]
      |                                   |
      |                    risk=read  -----+--> execute
      |                    risk=write -----+--> confirm with user, then execute
      |                    risk=money -----+--> confirm + idempotency key + audit
      v
[resolution] or [handoff packet: transcript, entities, attempted actions, why]
```

The gate is code, not prompt text. The agent cannot bypass it because it does not hold the credentials.

## Core concepts

### Scope and non-goals

In scope: FAQ answers from a knowledge base, order and subscription lookups, refunds within a policy limit, address and plan changes, ticket notes, and handoff with a context packet.

Non-goals: outbound calls, legal or medical advice, negotiating outside policy, and any action above a hard spend ceiling — those escalate. Say the ceiling in the README and enforce it in the gate, not in the prompt.

### Risk tiers, and who confirms

Classify every tool: `read` (lookup, no side effect), `write` (reversible state change), `money` (payment, refund, credit), `destructive` (delete, cancel). Reads run freely. Writes require an explicit user confirmation in the conversation. Money requires confirmation *plus* an idempotency key derived from the conversation and intent, plus an audit row. Destructive actions require a human. This is a table in your codebase, and it is the first thing an interviewer will ask about.

### Idempotency is the whole ballgame

An agent loop retries. The provider times out, the model re-plans, and the same refund is attempted again. Solve it where the effect happens: before calling the tool, write an intent row with a key computed from stable inputs (`conversation_id`, `tool`, normalised arguments). The tool wrapper checks that key against completed executions. If a completed execution exists, return its stored result instead of calling the API. If an intent exists without a result, the outcome is unknown — do not retry blindly; query the downstream system for the current state and reconcile. This is the difference between a demo and something a finance team will allow.

### Handoff is a data structure

Escalation should not be a message. Build a packet: customer identity, verified entities (order ids), the transcript summary, what the agent tried and the result of each attempt, the policy rule that blocked it, and the suggested next step. Write it into the ticket before routing. A human who opens a ticket with the packet resolves faster and trusts the agent more — which is what gets the project renewed.

### Data model, stack, milestones

| table | key fields |
|-------|-----------|
| `conversation` | id, customer_id, channel, status, assigned_to |
| `message` | id, conversation_id, role, content, created_at |
| `intent` | id, conversation_id, tool, args_hash, idem_key, state, result_json |
| `handoff` | id, conversation_id, reason, packet_json, queue, created_at |

Stack, loose: FastAPI, Postgres, Redis for conversation locks, your helpdesk API as the system of record, and one LLM SDK. Keep the tool registry declarative so adding a tool is a data change plus a function.

Four weeks: week 1 read-only answers with retrieval; week 2 the gate and confirmation flow; week 3 idempotency and audit; week 4 handoff packet, evals, demo.

### Eval plan and budget

- **Task success rate** on a scripted set of conversations with known outcomes (did the refund happen exactly once?).
- **Duplicate side effects** — must be zero across a fault-injection test where you kill the tool call mid-flight.
- **Policy compliance** — share of scripted adversarial conversations where the agent refused correctly.
- **Escalation precision** — of escalations, how many needed a human; of non-escalations, how many should have escalated.
- **Deflection rate** and **cost per resolved ticket**, both from traces, compared to your pre-agent baseline.

Budget, illustrative: assume 6 model turns per conversation at about 4k input tokens (growing transcript) and 300 output tokens, so roughly 25k tokens per conversation. Add retrieval and tool latency to your p95. The lever that matters is turn count — every unnecessary clarification costs a full turn; put the customer's order lookup in the system context up front and turns drop.

## Code

The gate plus idempotent execution is the hard part. This is the mechanism an interviewer wants to see.

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Awaitable, Callable


class Risk(IntEnum):
    READ = 0
    WRITE = 1
    MONEY = 2
    DESTRUCTIVE = 3


@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk: Risk
    fn: Callable[..., Awaitable[dict]]
    needs_confirmation: bool


def idem_key(conversation_id: str, tool: str, args: dict) -> str:
    """Stable across retries: same conversation + tool + normalised args."""
    payload = json.dumps({"c": conversation_id, "t": tool, "a": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


async def execute(spec: ToolSpec, conversation_id: str, args: dict, db) -> dict:
    key = idem_key(conversation_id, spec.name, args)
    prior = await db.get_intent(key)

    if prior and prior.state == "done":
        return prior.result_json                     # replay: no second refund
    if prior and prior.state == "pending":
        # outcome unknown: reconcile against the system of record, never blind-retry
        return await reconcile(spec, args, db)

    await db.insert_intent(key, spec.name, args, state="pending")   # write-ahead
    try:
        result = await spec.fn(**args)
    except Exception as exc:
        await db.set_intent(key, state="failed", error=str(exc))
        raise
    await db.set_intent(key, state="done", result_json=result)
    return result


def needs_human(spec: ToolSpec, amount: float, ceiling: float) -> bool:
    return spec.risk is Risk.DESTRUCTIVE or (spec.risk is Risk.MONEY and amount > ceiling)
```

And the handoff packet, which is mostly discipline about fields:

```python
from pydantic import BaseModel

class HandoffPacket(BaseModel):
    customer_id: str
    conversation_id: str
    entities: dict[str, str]          # order_id, subscription_id, ...
    summary: str
    attempts: list[dict]              # tool, args_hash, state, result
    blocked_by: str                   # the policy rule, quoted
    suggested_next_step: str

async def escalate(conv, reason: str, packet: HandoffPacket, helpdesk) -> None:
    await helpdesk.add_note(conv.ticket_id, packet.model_dump_json(indent=2))
    await helpdesk.assign(conv.ticket_id, queue=queue_for(reason))
    await conv.close(status="handed_off")
```

## Real use case

A subscription business routes billing questions to the agent. Before: refunds were handled by tier-one agents with a manual checklist, and a retry bug in an earlier automation double-refunded a small number of customers, which is why the automation was pulled.

After: the agent answers policy questions from the help centre, issues refunds under a ceiling after explicit confirmation, and escalates anything above it with a packet naming the blocked rule. The finance team's condition was idempotency plus an audit trail, and both are visible in the README.

Demo script, two minutes: ask a policy question, show the citation; request a refund and show the confirmation turn; kill the tool call in the demo to show the retry returns the stored result instead of paying twice; request an above-ceiling refund and show the handoff packet. Talking points: the risk table, the write-ahead intent, why confirmation is in the runtime, and how you measure escalation precision.

## Common mistakes

**Mistake:** enforcing policy in the system prompt — **Instead:** enforce it in the gate; prompts are advisory, code is not.
**Mistake:** letting the model construct idempotency keys — **Instead:** derive them from conversation and normalised args; the model will vary them.
**Mistake:** retrying on timeout — **Instead:** reconcile against the downstream system when the outcome is unknown.
**Mistake:** escalating with "connecting you to a human" — **Instead:** write the packet, then route; the customer should never repeat themselves.
**Mistake:** confirming every read — **Instead:** confirm only writes and money; over-confirmation trains users to click yes.
**Mistake:** evaluating only happy paths — **Instead:** script adversarial conversations, including one that claims approval you never granted.

## Checkpoint

1. Add a fault-injection test that raises after the tool call succeeds but before the intent is marked done. Prove no duplicate effect.
2. Add a hard spend ceiling per conversation and a test that the gate blocks above it regardless of what the model proposes.
3. Measure escalation precision on twenty scripted conversations and find the one rule that would remove the most unnecessary escalations.

**Self-check:** Exercise 1 passes because the intent row exists before the call, so a crash leaves `pending` and the retry reconciles rather than re-executes.

## Go deeper

- [Agent Security](../11-agents/11-agent-security.md) — prompt injection through customer messages
- [Guardrails](../20-ai-security/09-guardrails.md) — output filters and PII handling in transcripts
- [Workflow Automation](10-workflow-automation.md) — the same gate applied to background events
- [Agent Evals](../15-evals/06-agent-evals.md) — scoring multi-turn outcomes, not single replies

---

[← Previous: Multi-Agent Research](04-multi-agent-research.md) · [Stage index](README.md) · [Next: Coding Assistant](06-coding-assistant.md)
