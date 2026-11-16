# 5. Multi-Tenant Design

> **Stage 21 · System Design** — serving many customers from one system without leaking between them, letting one noisy customer starve the rest, or losing track of who owes what.
>
> **You'll be able to:** choose an isolation level with reasons, enforce per-tenant quotas and indexes, draw the data boundaries, and attribute cost per tenant.
>
> **Prerequisites:** [Auth and Rate Limits](../17-ai-backend/06-auth-rate-limits.md) · [Data Leakage](../20-ai-security/03-data-leakage.md)

## Why this exists

Multi-tenancy is the economics of a SaaS product: one deployment, many customers. It is also where the worst failures live, because a mistake is not a slow page — it is one customer reading another customer's documents. And AI systems add two twists the classic playbook does not cover: the expensive resource is *tokens*, not CPU, and the data boundary has to hold inside a vector index and inside a prompt, not just in a database row.

The mental model: **tenant identity is an invariant that every layer receives and enforces, and every expensive resource is metered per tenant.**

## The mental model

```text
 token ──► gateway: verify → tenant_id (from token, never the body)
              │
              ├─ quota: tokens/min · concurrency · spend/month
              ├─ routing: plan tier (free / pro / enterprise)
              ▼
        orchestrator ──► retrieval: filter tenant_id INSIDE the query
              │            index: shared | per-tenant collection
              ├─ cache: key namespaced by tenant (+ policy version)
              ├─ stores: row-level tenant_id on every table
              └─ metering: tokens, tool calls, storage → per tenant
```

| Isolation level | Mechanism | Cost | Use when |
|---|---|---|---|
| Shared everything | `tenant_id` column + query filter | lowest | many small tenants, strong code review |
| Shared DB, per-tenant schema | schema or collection per tenant | medium | medium tenants, some contractual isolation |
| Database per tenant | separate DB instance | high | regulated customers, data-residency needs |
| Deployment per tenant | separate stack | highest | enterprise contracts, single-tenant promises |

## Core concepts

### Isolation is a ladder; pick a rung and say why

Start shared with a mandatory filter, and move up only for a stated reason: a contract, a regulator, a volume that justifies it, or a residency requirement. Moving up costs money and operational effort — every migration, backup and monitoring path multiplies. Moving down later is a migration nobody enjoys.

Two hard rules that apply at every rung. First, the tenant id is derived from the verified token and passed as a typed session object; it is never read from a request body, a query string or a model output. Second, the filter lives *inside* the query — in SQL, in the vector search filter, in the cache key — not as a post-filter in application code. See [Data Leakage](../20-ai-security/03-data-leakage.md) for why the post-filter version fails.

### Per-tenant quotas: tokens, concurrency, spend

Three different limits, for three different failure modes:

- **Rate limit** (requests or tokens per minute) — stops a runaway client.
- **Concurrency cap** — stops one tenant occupying all your model connections.
- **Spend ceiling** (tokens or currency per month) — stops a bug from becoming an invoice.

Enforce them where the cost is incurred: the concurrency cap in the gateway or orchestrator before the model call, and the spend ceiling as a metered counter that fails closed with a clear error. A quota that returns "429: monthly token quota reached, resets on the 1st" is a product feature; a 500 is not.

### Per-tenant indexes and the cost of separation

A shared vector index with a filter is the cheapest option and usually correct. Per-tenant collections are worth it when a tenant's corpus is large enough that filtering hurts recall or latency, when a contract requires physical separation, or when a tenant needs a different embedding model. Remember what each index costs: its own ingestion job, its own re-embedding run (see [RAG Design at Scale](02-rag-design.md)), its own delete verification.

### Data boundaries beyond the row

List every place tenant data lands, and put the boundary on each: the primary database, the vector index, the object store (key prefix per tenant), the cache (namespaced keys), the prompt logs and traces (tenant id as a field, content redacted), backups, and evaluation datasets (de-identify, and never mix tenants into a shared golden set without consent). The one people forget is the cache: a semantic cache keyed only on the query text will happily serve tenant A's answer to tenant B.

### Cost attribution, with the arithmetic shown

You cannot price a product you cannot measure. Meter the things that cost money, per tenant, per request: input tokens, output tokens, embedding tokens, tool calls, storage, and retrieval queries.

Labelled example — **prices are placeholders; substitute current published values**:

| Assumption | Value |
|---|---|
| Tenant monthly requests | 200,000 |
| Input tokens per request | 2,500 |
| Output tokens per request | 350 |
| Input price / output price (placeholder) | `$P_in`, `$P_out` per 1,000,000 tokens |
| Embedding tokens per request (queries only) | 40 |

Monthly tokens: `200,000 × 2,500 = 500,000,000` input and `200,000 × 350 = 70,000,000` output. Cost: `(500 × $P_in) + (70 × $P_out)`, plus embedding `(200,000 × 40) / 1,000,000 = 8` million tokens × the embedding price. Add fixed costs — vector storage, database, a share of idle capacity — and divide by the plan price to get a gross margin per tenant.

The design consequence: a tenant whose usage is ten times the median but pays the same plan price is a loss you can only see if you meter. Attribution turns that into a pricing conversation instead of a surprise.

## Code

A tenant-scoped session, a quota check, and a namespaced cache key. The important detail is that `Session` is constructed only from a verified token, so no downstream function can receive a tenant id that came from the client.

```python
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class QuotaExceeded(Exception):
    pass


@dataclass(frozen=True)
class Session:
    tenant_id: str          # from the verified token only
    user_id: str
    plan: str               # "free" | "pro" | "enterprise"
    request_id: str


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens_per_min: int
    concurrency: int
    tokens_per_month: int


PLAN_LIMITS: dict[str, Limits] = {
    "free": Limits(tokens_per_min=20_000, concurrency=2, tokens_per_month=1_000_000),
    "pro": Limits(tokens_per_min=200_000, concurrency=16, tokens_per_month=40_000_000),
    "enterprise": Limits(tokens_per_min=2_000_000, concurrency=64, tokens_per_month=400_000_000),
}


async def admit(session: Session, meters, estimated_tokens: int) -> None:
    limits = PLAN_LIMITS[session.plan]
    # All three checks are on the tenant, not the user: one team, one budget.
    minute_used = await meters.minute_tokens(session.tenant_id)
    if minute_used + estimated_tokens > limits.tokens_per_min:
        raise QuotaExceeded("tokens per minute exceeded; retry shortly")
    month_used = await meters.month_tokens(session.tenant_id)
    if month_used + estimated_tokens > limits.tokens_per_month:
        raise QuotaExceeded("monthly token quota reached; resets at period end")
    if await meters.in_flight(session.tenant_id) >= limits.concurrency:
        raise QuotaExceeded("too many concurrent requests for this tenant")


def cache_key(session: Session, kind: str, payload: str, policy_version: str) -> str:
    # Tenant first, so a key collision across tenants is impossible by construction.
    return f"t:{session.tenant_id}:{kind}:{policy_version}:{hash(payload)}"


async def meter(session: Session, meters, usage) -> None:
    await meters.add(
        tenant_id=session.tenant_id,
        request_id=session.request_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        tool_calls=usage.tool_calls,
    )
```

The retrieval filter, repeated here because it is the boundary that matters most. The assertion after the search is deliberate defence in depth: it turns a store misconfiguration into a loud error instead of a silent leak.

```python
from __future__ import annotations


async def tenant_search(store, session: Session, vector: list[float], k: int = 8):
    rows = await store.search(
        vector=vector,
        k=k,
        filter={"tenant_id": {"$eq": session.tenant_id}},  # inside the query
    )
    leaked = [r["doc_id"] for r in rows if r["tenant_id"] != session.tenant_id]
    if leaked:
        raise RuntimeError(f"isolation violation for {session.tenant_id}: {leaked[:3]}")
    return rows
```

## Real use case

**A document-intelligence platform used by accounting firms.** Each firm had its own clients, so a single shared index served roughly 300 tenants. Two problems appeared in the first quarter. A batch import from one firm — several hundred thousand pages — saturated the model connection pool and made every other firm slow for an afternoon. And the monthly model bill was concentrated: three firms accounted for a large majority of usage while paying the same plan price as the rest.

The fixes were the two halves of this file. Concurrency caps per tenant, sized by plan, went into the gateway, so a bulk import was throttled to its own share and the noisy-neighbour afternoon could not recur. Metering was added at the orchestrator: tokens in, tokens out, embedding tokens, tool calls, all keyed by tenant and request id. Within a month the team could show each firm's cost, which turned into a usage-based tier for the three heavy users and a fair-use clause in the terms. The shared index stayed, with the filter inside the query and a fail-loud assertion, because none of the firms' contracts required physical separation.

## Common mistakes

- **Mistake:** reading `tenant_id` from the request body or a header the client controls — **Instead:** derive it from the verified token and pass a typed `Session` down.
- **Mistake:** filtering retrieved documents in application code — **Instead:** filter inside the store query and assert on the result.
- **Mistake:** a semantic cache keyed only on the query text — **Instead:** namespace every key by tenant and policy version.
- **Mistake:** rate limiting per user only — **Instead:** limit per tenant too, plus a concurrency cap and a spend ceiling.
- **Mistake:** discovering cost concentration after the invoice — **Instead:** meter tokens and tool calls per tenant per request from day one.

## Checkpoint

1. Add a `Session` type to an app you have built and make it impossible to construct one without a verified token. Count how many functions change.
2. Implement the three quota checks against an in-memory meter and write a test for each: minute limit, monthly limit, concurrency limit.
3. Pick an isolation level for a hypothetical customer with a residency requirement and write the three sentences justifying it, including what it costs you operationally.

**Self-check:** The `Session` is constructed only inside the auth layer from the token claims, so every downstream function receives a tenant id that a client cannot influence — that is what makes the invariant enforceable rather than a convention.

## Go deeper

- [Caching Layers](06-caching.md) — namespacing, and why invalidation is per tenant.
- [Data Leakage](../20-ai-security/03-data-leakage.md) — the four doors this design closes.
- [Cost and Latency Engineering](08-cost-latency.md) — turning metering into unit economics.
- [Ship: Design Document](10-ship-design-doc.md) — writing isolation and quotas into a review.

---
[← Previous: Model Routing](04-model-routing.md) · [Stage index](README.md) · [Next: Caching Layers](06-caching.md)
