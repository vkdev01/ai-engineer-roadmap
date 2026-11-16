# 1. Scalable AI Apps

> **Stage 21 · System Design** — the reference architecture for an AI application at scale, drawn as a pipeline with named seams where you add async work and caching.
>
> **You'll be able to:** draw an end-to-end AI architecture, place the async and caching seams correctly, and estimate what each hop costs in latency and money.
>
> **Prerequisites:** [AI App Architecture](../17-ai-backend/08-ai-app-architecture.md) · [Async Processing](../17-ai-backend/03-async-processing.md)

## Why this exists

A prototype is one function: build a prompt, call a model, print the answer. It works until three things happen at once — traffic arrives, answers must be grounded in documents, and someone asks what it costs. At that point the single function becomes a system with six hops, and the design question changes from "does it answer" to "where does the time go and where does the money go".

The reason to learn a reference architecture is that these systems fail in the same places. Retrieval is slow and needs a cache. Model calls are the expensive, rate-limited, flaky hop. Long tasks must not hold an HTTP connection. Tenant identity must be attached before retrieval, not after. If you know the standard shape, you can design a new feature by filling in the boxes instead of starting from a blank page.

The mental model: **edge → gateway → orchestrator → retrieval → model tier → stores, with an async seam wherever a step is slow and a cache seam wherever a step is repeated.**

## The mental model

```text
 client
   │  (TLS, static assets, SSE)
   ▼
 [ edge / CDN ] ── static, cached GETs
   │
   ▼
 [ gateway ] auth · tenant id · rate limit · quotas · request id
   │
   ├──────────────────────────────► [ queue ]  ── long jobs (ingest, batch, agents)
   ▼                                              │
 [ orchestrator ] prompt build · routing · budgets│
   │            │                                 │
   ▼            ▼                                 ▼
 [ retrieval ] [ cache ]                    [ workers ]
  vector+BM25   exact/semantic/prefix         same orchestrator
   │            │
   ▼            ▼
 [ stores ]  Postgres · vector DB · object store · Redis · audit
   │
   ▼
 [ model tier ] small / mid / frontier, via one client with retries + budgets
```

Two seams, and they matter more than the boxes:

- **Async seam.** Anything over a couple of seconds — ingestion, agent loops, batch enrichment, report generation — becomes a job with an id. The client polls or subscribes. HTTP requests stay short so your connection pool and load balancer are not held hostage.
- **Cache seam.** Anything repeated — system prefix, retrieved chunks, whole answers, embeddings — gets a cache with a policy-versioned key. Caches are where most of your cost and latency wins live (see [Caching Layers](06-caching.md)).

## Core concepts

### The gateway owns identity, limits and money

Every request gets, at the edge of your control: an authenticated identity, a **tenant id derived from the token** (never from the body), a request id that follows the request through every hop, a rate limit, and a quota check. Doing this in one place means retrieval cannot accidentally run without a tenant filter, and cost attribution has a tenant to bill.

Quotas are the difference between a bad day and an outage: per-tenant token budgets, concurrency caps, and a hard ceiling that returns a clear error rather than degrading everyone.

### The orchestrator is the only place that knows the pipeline

Keep prompt construction, routing decisions, retrieval calls, tool calls and budgets in one component, so the sequence is testable and versioned. When it lives in a request handler, every new entry point (API, webhook, batch job, cron) re-implements it slightly differently — and one of those copies forgets a guardrail. The orchestrator should be a pure-ish function of (input, session, config) returning (answer, trace), with side effects behind interfaces.

### Retrieval is a service, not a library call

At scale, retrieval deserves its own latency budget, its own cache, and its own health check. It is the hop most likely to be slow (a cold vector index, an expensive filter) and the one whose failure you can most gracefully degrade from — answer without documents, with a visible caveat. Give it a timeout that is a fraction of the total budget.

### One model client, many models

All model calls go through a single client that owns retries, timeouts, rate-limit handling, token counting, cost accounting and provider fallback. This is what makes [Model Routing](04-model-routing.md) and [Fault Tolerance](07-fault-tolerance.md) possible: the routing decision is a parameter, not a different code path.

### Latency and cost budget: do the arithmetic

Every hop gets a latency budget that sums to your target, and a cost line that sums to your unit economics. Labelled example, with illustrative assumptions you must replace:

| Assumption | Value (illustrative) |
|---|---|
| Target p95 end-to-end | 3.0 s |
| Input tokens per request | 2,500 |
| Output tokens per request | 350 |
| Model price (placeholder) | $P_in / M input tokens, $P_out / M output tokens |
| Requests per day at launch | 20,000 |

Latency budget: gateway 40 ms + retrieval 300 ms + prompt build 20 ms + model time-to-first-token 700 ms + streaming 1,800 ms ≈ 2.86 s. If the model hop exceeds 2.4 s p95, you are over target and must either shrink the prompt or route to a faster tier.

Cost: `(2500 × P_in + 350 × P_out) / 1,000,000` per request. At 20,000 requests/day that is the daily spend; multiply by 30 for the month. Substitute the provider's current published price — do not trust a number from a blog post, including this one.

## Code

The orchestrator skeleton: a typed request, a latency budget that is actually enforced, and a trace object that records where the time went. Note that the retrieval timeout is derived from the remaining budget rather than fixed.

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str
    user_id: str
    request_id: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    degraded: bool = False
    trace: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Budget:
    total_s: float = 3.0
    retrieval_s: float = 0.4
    model_s: float = 2.4


async def orchestrate(
    session: Session, question: str, deps, budget: Budget = Budget()
) -> Answer:
    started = time.perf_counter()
    trace: dict[str, float] = {}
    degraded = False

    # 1. Retrieval with its own slice of the budget. Failure is survivable.
    t0 = time.perf_counter()
    try:
        chunks = await asyncio.wait_for(
            deps.retrieval.search(session, question), timeout=budget.retrieval_s
        )
    except (asyncio.TimeoutError, Exception):
        chunks = []
        degraded = True  # answer without grounding, and say so in the trace
    trace["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # 2. Prompt build is pure; the model call gets whatever budget is left.
    prompt = deps.prompts.build(question, chunks)
    remaining = budget.total_s - (time.perf_counter() - started)
    model_budget = min(budget.model_s, max(remaining, 0.5))

    t1 = time.perf_counter()
    text = await asyncio.wait_for(
        deps.models.complete(prompt, tier="mid", max_tokens=500), timeout=model_budget
    )
    trace["model_ms"] = (time.perf_counter() - t1) * 1000
    trace["total_ms"] = (time.perf_counter() - started) * 1000

    return Answer(text=text, degraded=degraded, trace=trace)
```

The gateway check below is the other half: identity and quota resolved once, before anything expensive. Tenant id comes from the verified token, and the quota is a token budget rather than a request count, because a request can cost a hundred times another.

```ts
import type { Request, Response, NextFunction } from "express";

export interface Caller {
  tenantId: string;
  userId: string;
  requestId: string;
}

const tokenQuota: Record<string, number> = {}; // replace with Redis INCRBY

export function gateway(verify: (h?: string) => Caller | null) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const caller = verify(req.header("authorization"));
    if (!caller) {
      res.status(401).json({ error: "unauthenticated" });
      return;
    }
    // Tenant id is derived from the token, never read from the body.
    res.locals.caller = caller;
    const used = tokenQuota[caller.tenantId] ?? 0;
    if (used > 5_000_000) {
      res.status(429).json({ error: "tenant token quota exhausted" });
      return;
    }
    res.setHeader("x-request-id", caller.requestId);
    next();
  };
}
```

## Real use case

**A customer-support assistant for a D2C retailer with a peak-season spike.** The first version answered in one handler: fetch orders, call a frontier model, return. At roughly a few hundred concurrent users it hit two walls — the connection pool was saturated because every request waited on the model, and the monthly bill was dominated by a handful of tenants re-sending a 6,000-token system prompt.

The redesign kept the product identical and changed the shape. A gateway resolved tenant and quota once. Retrieval moved behind its own timeout and cache, with a degraded "answer without documents" path. A prompt cache stored the static system prefix so repeat turns re-used it (see [Caching Layers](06-caching.md)). Long report jobs moved to a queue. The same orchestrator served the API, the queue worker and a nightly batch job, so a guardrail added once applied everywhere. Peak-season traffic rose without a latency incident, and the cost per request fell because the expensive part — the prefix — stopped being recomputed.

## Common mistakes

- **Mistake:** holding an HTTP connection open for a 40-second agent run — **Instead:** make it a job with an id, and poll or stream progress.
- **Mistake:** reading `tenant_id` from the request body — **Instead:** derive it from the verified token in the gateway, and pass it down as a typed session.
- **Mistake:** fixed timeouts on every hop that sum to more than your target — **Instead:** derive downstream timeouts from the remaining budget.
- **Mistake:** re-implementing the pipeline in each entry point — **Instead:** one orchestrator, many callers.
- **Mistake:** quoting a model price as a design fact — **Instead:** use a placeholder constant with a comment to substitute the current published price.

## Checkpoint

1. Draw the six-box architecture for an app you have built and label each hop with a measured or estimated latency. Find the hop that dominates.
2. Compute cost per request for your app using the formula above, then compute monthly spend at 10,000 requests/day. State every assumption.
3. Move one synchronous step to the async seam: add a job id, a queue, and a status endpoint. Measure the p95 of the synchronous path before and after.

**Self-check:** The dominating hop is almost always the model call, followed by retrieval on a cold index — which is why the model tier gets the largest budget slice and retrieval gets a cache and a fallback.

## Go deeper

- [RAG Design at Scale](02-rag-design.md) — the retrieval box in detail.
- [Caching Layers](06-caching.md) — the seam that pays for itself first.
- [Cost and Latency Engineering](08-cost-latency.md) — turning the arithmetic into a plan.
- [Ship: Design Document](10-ship-design-doc.md) — writing this architecture up for review.

---
[← Previous: Stage 20 — AI Security](../20-ai-security/README.md) · [Stage index](README.md) · [Next: RAG Design at Scale](02-rag-design.md)
