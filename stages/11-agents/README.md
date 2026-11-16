# Stage 11 — AI Agents

This stage is the centre of gravity of the roadmap. Everything before it gave the model knowledge and a voice; here it gets hands. An agent is a model in a loop with tools, a budget and a way to stop — and the phrase to carry through the next twelve files is **build the harness, close the loop**. The harness is the code you write around the model; the loop is the thing that makes one model call into work.

**Time:** roughly 3-4 weeks of evenings · **You need first:** [Stage 8 — Prompt and Context](../08-prompt-and-context/README.md) and [Stage 10 — RAG](../10-rag/README.md) · **You leave with:** a harness agent with a typed tool registry, a step budget, a trace log, an approval gate and a small eval set.

## What you will be able to do

- Write the agent loop yourself in plain Python, with no framework, and explain every line of it.
- Design tool schemas the model can actually use, and validate their arguments before anything runs.
- Choose between single-agent, supervisor, handoff and fan-out topologies using cost and debuggability, not fashion.
- Stop runaway loops with step budgets, no-progress detection and retry semantics that terminate.
- Defend an agent against prompt injection arriving through tool output.
- Put a human approval gate in front of side-effecting actions without losing resumable state.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Harness Engineering](01-harness-engineering.md) | The scaffold around the model that makes it reliable |
| 2 | [Agent Architecture](02-agent-architecture.md) | The minimal prompt-tool-observe loop and its stop conditions |
| 3 | [Reasoning and Planning](03-reasoning-planning.md) | ReAct, plan-then-execute, decomposition, and when planning hurts |
| 4 | [Tool Calling](04-tool-calling.md) | Schemas, validation, idempotency, actionable errors, sandboxing |
| 5 | [Memory and State](05-memory-and-state.md) | Conversation state, checkpoints, summarisation, retrieval memory |
| 6 | [Single-Agent Patterns](06-single-agent.md) | One agent with good tools beats a crowd of vague ones |
| 7 | [Multi-Agent Patterns](07-multi-agent.md) | Supervisor, handoff, debate, fan-out/fan-in, and honest costs |
| 8 | [Human in the Loop](08-human-in-the-loop.md) | Approval gates, interrupts and resumable state |
| 9 | [Reflection and Self-Critique](09-reflection.md) | What self-review fixes and what it only pretends to fix |
| 10 | [Loop Engineering](10-loop-engineering.md) | Step budgets, no-progress detection, termination proofs |
| 11 | [Agent Security](11-agent-security.md) | Injection via tool output, least privilege, allowlists |
| 12 | [Ship: A Harness Agent](12-ship-harness-agent.md) | The full agent: registry, budget, trace, approval, evals |

## The shipping project

[12-ship-harness-agent.md](12-ship-harness-agent.md) — you build a research-and-report agent from scratch: a typed tool registry, a step budget with no-progress detection, a JSONL trace log, an approval gate for writes, and a small eval set that measures whether a change to the harness helped or hurt.

## How to know you are done

- You can write the loop from memory in under thirty lines and explain each stop condition.
- Every tool you ship has a schema, a validator and an error message that names the fix.
- Your agent has a hard step budget and a trace you can read to explain any single run.
- You have a side-effecting tool behind an approval gate, and you have resumed a run after the interrupt.
- Your eval set has at least ten cases, including two adversarial ones from tool output.

---

[← Back to roadmap](../../README.md) · [Next stage: Orchestration](../12-orchestration/README.md)
