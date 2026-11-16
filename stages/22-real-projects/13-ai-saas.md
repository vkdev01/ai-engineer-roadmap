# 13. AI SaaS

> **Stage 22 · Real Projects** — the business layer: accounts, plans, quotas that are actually enforced, metered usage, and a cost model where you know your margin per customer.
>
> **You'll be able to:** put authentication, tenancy, quota reservation and usage billing around an AI feature, and compute cost and margin per tenant from your own logs.
>
> **Prerequisites:** [Auth and Rate Limits](../17-ai-backend/06-auth-rate-limits.md) · [Multi-Tenant Design](../21-system-design/05-multi-tenant.md)

## Why this exists

The difference between a project and a product is that a product can tell you who used what, stop them when they exceed their plan, and still make money. Every AI app has the same three failure modes at this layer, and all three are billing incidents rather than quality incidents.

The first is unbounded spend: one customer runs a long agent loop and your monthly inference bill triples. The second is a quota check that is a `SELECT` followed by an `INSERT` — under concurrent requests, ten parallel calls all see quota remaining and all proceed. The third is invisible margin: you charge a flat subscription and never compute per-tenant cost, so your heaviest customer is quietly unprofitable and you find out at the end of the quarter.

Buyers are your own finance and growth teams. What they need is metering that survives retries and quotas that are enforced atomically.

Mental model: every request *reserves* budget before it runs and *settles* actual usage after; cost is a first-class column, not a monthly surprise.

## The mental model

```text
request --> [auth: JWT -> tenant_id] --> [plan lookup: limits]
                                                |
                                    [reserve quota atomically]
                                     (UPDATE ... WHERE used + n <= cap)
                                                |
                                        ok? --no--> 429 with reset time
                                         |
                                        yes
                                         v
                                  [run the feature]
                                         |
                                    [settle: actual tokens, cost_usd]
                                         |
                        [usage ledger] --> [invoice lines] --> [margin per tenant]
```

Two writes per request: a reservation and a settlement. Everything else is reporting.

## Core concepts

### Scope and non-goals

In scope: sign-up and sessions, tenants and memberships, plans with limits, atomic quota reservation, a usage ledger with cost, a billing export, and a per-tenant cost dashboard.

Non-goals: a full billing system (use a payment provider), SSO, complex pricing experiments, and reselling third-party models. Also non-goal: hard real-time quota accuracy — approximate within a small tolerance is fine as long as over-spend is bounded.

### Tenant from the token, never from the body

The tenant id comes from the verified session, and every row carries it. Middleware sets a request-scoped context so handlers cannot accidentally query across tenants; the repository layer asserts the tenant is present. Add one test per endpoint that a token for tenant A cannot read tenant B's data. This is the same discipline as [Production RAG](03-production-rag.md), applied at the HTTP boundary.

### Quotas: reserve, then settle

Reservation is the only mechanism that holds under concurrency. Do it in one statement so the database's row lock does the arbitration:

```sql
UPDATE quota SET used = used + $2
WHERE tenant_id = $1 AND period = $3 AND used + $2 <= cap
RETURNING used, cap;
```

Zero rows updated means over quota — return 429 with the reset time, and do not call the model. After the request, settle with actuals: if actual is less than the reservation, credit the difference; if more (a long generation), record the overage and let the next request be rejected. Estimate generously — worst-case max tokens — so the reservation is an upper bound.

### Cost is computed, then stored

Store `cost_usd` per request from token counts and the rate table version you used, not recomputed at invoice time. Rate tables change; if you recompute history with new rates, your invoices become fiction. Keep a `rate_version` column and a table of rates with effective dates.

Then compute the only number that matters for the business: **margin per tenant** = revenue attributed to the tenant minus (inference + retrieval + storage + queue) cost, over a period. Show it per tenant and per feature. Most teams discover one feature drives most of the cost and one tenant drives most of the feature.

### Plan limits that match how people actually use it

Limit the thing that costs you money: tokens per period, agent steps per request, documents ingested per month, concurrent requests. A limit on "API calls" is meaningless when one call can be a 40-step agent run. Give each plan a monthly token allowance, a per-request step cap, and a concurrency cap, and put the concurrency cap behind a token-bucket in Redis so a single tenant cannot starve the pool.

### Milestones, data model, budget

Four weeks: week 1 auth, tenants, memberships; week 2 quota reservation and 429 path; week 3 usage ledger with cost and rate versions; week 4 billing export, margin dashboard, load test.

| table | key fields |
|-------|-----------|
| `tenant` | id, name, plan, created_at, status |
| `membership` | tenant_id, user_id, role |
| `quota` | tenant_id, period, used, cap |
| `usage` | id, tenant_id, user_id, feature, tokens_in, tokens_out, cost_usd, rate_version, request_id |
| `rate` | version, model, usd_per_mtok_in, usd_per_mtok_out, effective_from |

Budget arithmetic, illustrative: pick a plan price, then work backwards. If a subscription is priced at some amount per month and your blended cost per request is C, then the allowance you can afford is (price × target gross margin) / C requests. Compute C from your own ledger — tokens per request × current rates — rather than from a vendor's example. Set the plan allowance below the affordable figure so a heavy user still leaves margin, and use the concurrency cap to bound worst-case exposure from a single abusive tenant.

## Code

Reservation and settlement, with the atomicity that makes it correct:

```python
from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class Reservation:
    tenant_id: str
    period: str
    tokens: int


async def reserve(db, tenant_id: str, period: str, tokens: int) -> Reservation:
    """One statement: the row lock arbitrates concurrent requests."""
    row = await db.fetchrow(
        """
        UPDATE quota SET used = used + $2
        WHERE tenant_id = $1 AND period = $3 AND used + $2 <= cap
        RETURNING used, cap
        """,
        tenant_id, tokens, period,
    )
    if row is None:
        raise HTTPException(status_code=429, detail={"error": "quota_exceeded", "period": period})
    return Reservation(tenant_id=tenant_id, period=period, tokens=tokens)


async def settle(db, res: Reservation, actual_tokens: int, cost_usd: float,
                 rate_version: str, request_id: str, feature: str, user_id: str) -> None:
    """Return the unused part of the reservation and record real usage."""
    delta = res.tokens - actual_tokens
    async with db.transaction():
        if delta:
            await db.execute(
                "UPDATE quota SET used = GREATEST(used - $2, 0) WHERE tenant_id = $1 AND period = $3",
                res.tenant_id, delta, res.period,
            )
        await db.execute(
            """INSERT INTO usage (tenant_id, user_id, feature, tokens_in, tokens_out,
                                  cost_usd, rate_version, request_id)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
            res.tenant_id, user_id, feature, actual_tokens, 0, cost_usd, rate_version, request_id,
        )
```

Cost from a versioned rate table, so history stays true:

```python
def cost_usd(tokens_in: int, tokens_out: int, rate: dict) -> float:
    """rate comes from the row whose effective_from is newest but <= now."""
    return (tokens_in / 1_000_000) * rate["usd_per_mtok_in"] + \
           (tokens_out / 1_000_000) * rate["usd_per_mtok_out"]
```

The margin query is one `GROUP BY`, and it is the report you should be able to run at any time:

```sql
SELECT t.id,
       t.plan,
       SUM(u.cost_usd)                                   AS cost,
       SUM(u.tokens_in + u.tokens_out)                    AS tokens,
       COUNT(DISTINCT u.request_id)                       AS requests
FROM tenant t
JOIN usage u ON u.tenant_id = t.id
WHERE u.created_at >= date_trunc('month', now())
GROUP BY t.id, t.plan
ORDER BY cost DESC;
```

## Real use case

A small team ships a document-question-answering product with a flat monthly plan. Before: quota was checked with a read then a write, so a burst from one customer's batch script exceeded the allowance, and the team learned about it from the provider invoice. Cost per customer was unknown, so the plan price was a guess.

After: reservation in one statement bounds over-spend, a concurrency cap in Redis stops one tenant starving others, and the margin query showed that a single feature — re-answering with the largest context — was most of the cost. They capped steps per request and made the big-context mode a paid add-on.

The design choice that made it work is treating quota as a reservation: the concurrency bug disappeared because the database, not the application, decided who was over the line.

Demo script, two minutes: show a request settling into the ledger; fire parallel requests at the cap and show the 429 with reset time; show the rate table with versions; open the margin table and explain the feature that dominates cost. Talking points: atomic reservation, why cost is stored not recomputed, how you set the allowance, and what happens on a provider price change.

## Common mistakes

**Mistake:** read-then-write quota checks — **Instead:** a single conditional `UPDATE`; the row lock is the arbiter.
**Mistake:** taking the tenant from the request body — **Instead:** derive it from the verified session and assert it in the repository.
**Mistake:** recomputing historical cost with current prices — **Instead:** store cost and `rate_version` at request time.
**Mistake:** limiting API calls instead of tokens and steps — **Instead:** limit what costs money: tokens, steps, documents, concurrency.
**Mistake:** no concurrency cap — **Instead:** a per-tenant token bucket; one tenant must not degrade everyone.
**Mistake:** no per-tenant margin view — **Instead:** a monthly `GROUP BY` you actually look at, per tenant and per feature.

## Checkpoint

1. Load-test the quota path with parallel requests at exactly the cap and confirm the total used never exceeds it.
2. Add a per-tenant concurrency cap in Redis and show a burst from one tenant being throttled while another is unaffected.
3. Add a second rate version with a new effective date and verify that old usage rows keep their original cost while new requests use the new rate.

**Self-check:** Exercise 1 holds because the conditional `UPDATE` either increments within the cap or matches no row, so concurrent requests serialise on the row.

## Go deeper

- [Production RAG](03-production-rag.md) — the service this layer wraps
- [Cost Monitoring](../18-mlops-llmops/06-cost-monitoring.md) — dashboards and alerting on spend
- [Model Routing](../21-system-design/04-model-routing.md) — cheaper models as a margin lever
- [Scaling Tradeoffs](../21-system-design/09-scaling-tradeoffs.md) — what changes at ten times the tenants

---

[← Previous: Eval Platform](12-eval-platform.md) · [Stage index](README.md) · [Next: Stage 23 — Career](../23-career/README.md)
