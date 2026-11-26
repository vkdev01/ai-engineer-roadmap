# Stage 21 — System Design

Every stage before this one taught a component. This stage teaches the thing you are actually paid for: deciding what to build, what to leave out, and how to defend the choice when someone asks why the system costs what it costs.

The output of this stage is a document, not a repository. That is deliberate — the ability to write a design that survives review is the difference between an engineer who builds features and one who is trusted with a system.

**Time:** 2 weeks · **You need first:** [Stage 16 — Inference](../16-inference/README.md) and [Stage 20 — AI Security](../20-ai-security/README.md) · **You leave with:** a full design document with capacity arithmetic, failure modes and a cost model

## What you will be able to do

- Draw the reference architecture of a scalable AI app and place any component on it
- Design RAG at scale: freshness, sharding, re-embedding cost
- Choose between a state machine and a free agent loop, and bound either
- Route requests across model tiers and measure the routes
- Design multi-tenant isolation, quotas and cost attribution
- Layer caching (exact, semantic, prefix, retrieval) and invalidate it honestly
- Make a system survive provider failure with timeouts, breakers and degraded modes
- Estimate capacity and cost with arithmetic a reviewer can check
- Write a design document that names its own failure modes
- Walk the nine production layers of an agentic app and say what happens when each one is down

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Scalable AI Apps](01-scalable-ai-apps.md) | The reference architecture and its seams |
| 2 | [RAG Design](02-rag-design.md) | Indexing at scale, freshness, re-embedding |
| 3 | [Agent Design](03-agent-design.md) | State machines, budgets, idempotency |
| 4 | [Model Routing](04-model-routing.md) | Cost tiers, cascades, measured routes |
| 5 | [Multi-Tenant](05-multi-tenant.md) | Isolation levels, quotas, attribution |
| 6 | [Caching](06-caching.md) | Four cache layers and their invalidation |
| 7 | [Fault Tolerance](07-fault-tolerance.md) | Timeouts, breakers, degraded modes |
| 8 | [Cost and Latency](08-cost-latency.md) | Token budgets and unit economics |
| 9 | [Scaling Trade-offs](09-scaling-tradeoffs.md) | What you buy and what you pay |
| 10 | [Ship: A Design Document](10-ship-design-doc.md) | The document, with numbers |
| 11 | [The Nine-Layer Agentic Stack](11-nine-layer-agent-stack.md) | The production layers, what each prevents, and three project briefs |

## The shipping project

[Ship: A Design Document](10-ship-design-doc.md) — pick a real product (one of the Stage 22 briefs works well), write the requirements, do the capacity arithmetic with labelled assumptions, draw the components, enumerate the failure modes, and price it. Ten pages, no code required, every number checkable.

## How to know you are done

- Every number in your document can be traced to an assumption you stated
- You have named at least five failure modes and what the system does in each
- You can say what you deliberately did not build, and why
- Your cost model shows cost per request and per thousand requests at an assumed volume
- A reviewer could implement your design without asking you what you meant

---
[← Back to roadmap](../../README.md) · [Next stage: Real Projects](../22-real-projects/README.md)
