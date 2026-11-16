# 8. Token Accounting and Cost Tracking

> **Stage 07 · LLM APIs** — knowing what each request cost and who to bill for it.
>
> **You'll be able to:** count tokens before and after a call, attribute cost to a request and a tenant, and detect when a feature's spend drifts.
>
> **Prerequisites:** [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [Tokenization](../06-llm-fundamentals/02-tokenization.md)

## Why this exists

LLM spend is unusual: it is metered per token, it scales with user behaviour you do not control, and it is invisible in normal infrastructure monitoring. There is no CPU graph that tells you a prompt got 40% longer after a refactor. Teams discover this when a bill arrives, which is the worst possible time.

You need two numbers per call: what you expected (a pre-call estimate, for admission control) and what actually happened (the provider's usage, for the ledger). Estimating alone drifts; actuals alone arrive too late to stop a runaway loop.

The mental model: **cost is a fact you record per call, keyed by tenant and feature, and prices are configuration that changes under you.**

## The mental model

```text
request --> estimate tokens --> budget check --> call --> usage numbers
                                  |                          |
                              reject if over             ledger row
                                                             |
                                              tenant x feature x day rollup
```

| Quantity | Where it comes from | Use for |
|---|---|---|
| Input tokens | provider `usage` | billing, ledger |
| Output tokens | provider `usage` | billing, ledger |
| Pre-call estimate | local tokeniser or heuristic | budget checks, prompt design |
| Price per token | your config, from provider pricing page | converting to money |
| Retry count | your wrapper | explaining spend spikes |

## Core concepts

### Prices belong in configuration, not code

Per-token prices differ by model, by input versus output, and they change. Keep a small table in config with an effective date, and record which version priced each row. If a price is unknown, store the tokens and leave cost null — a null you can backfill beats a wrong number you bill someone for.

### The provider's usage is the source of truth

Your tokeniser will disagree slightly with the provider's count, because the provider counts the chat template, the role markers and any tool schemas. Estimate with your own tokeniser for budgeting; record the provider's numbers for accounting. Never present an estimate as a bill.

### Cache hits and reasoning tokens change the arithmetic

Some providers charge less for cached input prefixes, and some report reasoning or thinking tokens separately from visible output. A ledger with one flat input price will overstate cached calls and understate reasoning ones. Model the fields you actually receive, and keep an `extra` JSON column for the rest.

### Attribution is the point, not the total

A total spend number cannot tell you which feature to fix. Key every row by `tenant_id`, `feature`, `model`, `prompt_version` and `request_id`. Then "why did spend double" becomes a group-by instead of an investigation.

### Streaming and retries need explicit handling

Streams may report usage only in a final chunk. Retries create multiple provider calls for one user action — record each attempt, and mark which one produced the answer. Otherwise your ledger undercounts exactly when the system is unhealthy.

## Code

An estimator, a price table, and a ledger row written from the provider's usage.

```python
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import tiktoken  # local tokeniser; pick the encoding for your model family

# Prices are configuration. Amounts here are ILLUSTRATIVE placeholders only:
# check the provider's pricing page and store the effective date.
PRICES: dict[str, dict[str, Decimal]] = {
    "model-a": {"in": Decimal("0.0000005"), "out": Decimal("0.0000015")},
    "model-b": {"in": Decimal("0.0000002"), "out": Decimal("0.0000008")},
}
PRICE_VERSION = "2026-11-01"


@dataclass
class LedgerRow:
    request_id: str
    tenant_id: str
    feature: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    price_version: str
    attempt: int = 1
    latency_ms: int = 0
    extra: dict[str, object] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def estimate_tokens(text: str, model: str) -> int:
    """Pre-call estimate. Good enough for budget checks, not for billing."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback encoding
    return len(enc.encode(text))


def price(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    table = PRICES.get(model)
    if table is None:
        return Decimal("0")     # unknown model: record tokens, flag cost later
    return table["in"] * input_tokens + table["out"] * output_tokens


def make_row(
    *, tenant_id: str, feature: str, model: str, prompt_version: str,
    usage_in: int, usage_out: int, attempt: int, latency_ms: int,
    extra: dict[str, object] | None = None,
) -> LedgerRow:
    return LedgerRow(
        request_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        feature=feature,
        model=model,
        prompt_version=prompt_version,
        input_tokens=usage_in,
        output_tokens=usage_out,
        cost_usd=price(model, usage_in, usage_out),
        price_version=PRICE_VERSION,
        attempt=attempt,
        latency_ms=latency_ms,
        extra=extra or {},
    )


def to_jsonl(row: LedgerRow) -> str:
    # Decimal is not JSON-serialisable; store as a string to keep precision
    data = asdict(row)
    data["cost_usd"] = str(row.cost_usd)
    return json.dumps(data)
```

Wiring it into a call means the ledger is a side effect you cannot forget:

```python
async def tracked_call(tenant: str, feature: str, prompt: str, model: str) -> str:
    budget = 8_000
    if estimate_tokens(prompt, model) > budget:
        raise ValueError("prompt exceeds per-request token budget")
    t0 = time.monotonic()
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    row = make_row(
        tenant_id=tenant, feature=feature, model=model, prompt_version="v3",
        usage_in=resp.usage.prompt_tokens, usage_out=resp.usage.completion_tokens,
        attempt=1, latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await ledger.append(to_jsonl(row))     # your sink: file, Postgres, queue
    return resp.choices[0].message.content or ""
```

In a Node service, the same row is a typed object; keep `costUsd` as a number of micro-dollars to avoid float drift in aggregation.

```ts
export type LedgerRow = {
  requestId: string;
  tenantId: string;
  feature: string;
  model: string;
  promptVersion: string;
  inputTokens: number;
  outputTokens: number;
  costMicroUsd: number; // integers: safe to sum
  priceVersion: string;
  attempt: number;
  latencyMs: number;
};

// ILLUSTRATIVE placeholder prices per token; read the provider pricing page.
const PRICES: Record<string, { in: number; out: number }> = {
  "model-a": { in: 0.5, out: 1.5 }, // micro-USD per token
};

export function buildRow(args: {
  tenantId: string;
  feature: string;
  model: string;
  promptVersion: string;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
}): LedgerRow {
  const p = PRICES[args.model] ?? { in: 0, out: 0 };
  return {
    requestId: crypto.randomUUID(),
    ...args,
    costMicroUsd: Math.round(p.in * args.inputTokens + p.out * args.outputTokens),
    priceVersion: "2026-11-01",
    attempt: 1,
  };
}

export async function record(row: LedgerRow): Promise<void> {
  await fetch(process.env.LEDGER_URL!, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(row),
  });
}
```

## Real use case

A B2B SaaS company offered an AI "summarise this thread" button on a free plan. Spend grew faster than usage, and the dashboard could not explain why. Adding `tenant_id` and `feature` to every ledger row showed that one free tenant was generating very long threads, and the summariser was sending the whole thread every time it was clicked.

Two changes followed: a per-tenant daily token cap enforced from the ledger, and a pre-call estimate that truncated or summarised long threads before sending. Spend came back in line without changing the model or the prompt. The failure mode before was a single undifferentiated spend total — it told them money was leaving but not which button, which customer, or which input shape was responsible.

## Common mistakes

- **Mistake:** hardcoding prices in business logic. — **Instead:** keep a versioned price table in config and stamp each row with its version.
- **Mistake:** billing from your own token estimate. — **Instead:** record the provider's usage numbers; estimate only for pre-call budgeting.
- **Mistake:** logging only the total per day. — **Instead:** key rows by tenant, feature, model and prompt version so you can group.
- **Mistake:** forgetting that retries are separate provider calls. — **Instead:** log every attempt, with an `attempt` field.
- **Mistake:** assuming a stream reports usage. — **Instead:** request the final usage chunk where supported, and mark rows without usage as estimated.

## Checkpoint

1. Add a `per_tenant_daily_cap` check that reads the last 24 hours of rows and rejects calls over the cap.
2. Group a day of ledger rows by `feature` and by `prompt_version`; identify the most expensive one.
3. Simulate 200 calls with a 10% retry rate and confirm your total cost equals the sum of per-attempt costs, not per-user-action costs.

**Self-check:** you record `resp.usage.prompt_tokens` and `resp.usage.completion_tokens` (or the provider's equivalents) multiplied by a versioned price table, and write one row per attempt keyed by tenant and feature.

## Go deeper

- [Routing and Fallbacks](09-routing-fallbacks.md) — cost is one of the inputs to a routing decision.
- [Cost Monitoring](../18-mlops-llmops/06-cost-monitoring.md) — turning a ledger into dashboards and alerts.
- [Cost and Latency](../21-system-design/08-cost-latency.md) — the system-design view of the same trade-off.
- [Tokenization](../06-llm-fundamentals/02-tokenization.md) — why token counts are not word counts.

---
[← Previous: Retries and Rate Limits](07-retries-rate-limits.md) · [Stage index](README.md) · [Next: Routing and Fallbacks](09-routing-fallbacks.md)
