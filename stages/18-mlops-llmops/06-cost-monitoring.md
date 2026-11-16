# 6. Cost Monitoring and Budgets

> **Stage 18 · MLOps and LLMOps** — knowing what each feature, tenant and successful answer costs, before the invoice tells you.
>
> **You'll be able to:** attribute model spend to features and tenants, set budgets with alerts that fire early, and compute a unit-economics number you can defend.
>
> **Prerequisites:** [Cost Tracking](../07-llm-apis/08-cost-tracking.md) · [Tracing Design](05-tracing.md)

## Why this exists

An LLM bill is one number at the end of the month. Your product is forty features and three hundred customers. The bill tells you nothing you can act on: you cannot tell whether the spike is the new summariser, one enterprise tenant hammering a batch endpoint, or an accidental retry loop introduced in Tuesday's deploy.

Worse, LLM cost is variable in a way server cost is not. A CPU-bound service costs roughly the same per request forever. An LLM service costs per token, and tokens depend on user input, retrieved context size, retry count, and whether the model decided to think for 900 tokens before answering. A feature that looks cheap per call can be your largest line item because it is called constantly.

The mental model: **cost is a per-request fact you must record at the moment of the call, then aggregate. It is not something you reconstruct from a bill.**

## The mental model

```text
per call:  tokens_in, tokens_out, model_id  ->  price table  ->  cost_usd
                                                                   |
per request: sum of generation costs  = request cost              |
per trace:   cost_usd on the root  ------------------------------ |
                                                                   v
aggregations:  by feature | by tenant | by prompt_version | by day
alerts:        daily budget, per-tenant monthly cap, cost per success
```

## Core concepts

### Price is data, not code

Never hardcode `0.15 / 1M` in a function. Put prices in a YAML table with an `effective_from` date, because providers change prices and you will want old traces to keep their old cost rather than silently being repriced.

```yaml
# config/prices.yaml — verify every number against the provider's pricing page.
models:
  gpt-4.1-mini:
    effective_from: 2026-01-01
    input_per_1m: 0.00
    output_per_1m: 0.00
    note: fill in from the provider pricing page; do not guess
  claude-sonnet-class:
    effective_from: 2026-01-01
    input_per_1m: 0.00
    output_per_1m: 0.00
    note: same
```

Two things to internalise: the numbers above are deliberately blank because any number written here would be stale or invented, and **cached input tokens and batch discounts are priced differently** from ordinary input. If you use prompt caching, your price table needs a third rate per model, or your cost estimate will be wrong in the direction that matters.

### Attribute at the call site, aggregate later

The only reliable place to know what a call cost is where you make it: the response carries token usage, and you know the model. Attach `cost_usd` to the generation span and to the request record. Everything downstream — dashboards, alerts, tenant invoices — is a group-by on that field.

The common failure is computing cost in a nightly job from application logs that recorded the answer text but not the token counts. You cannot recover token counts reliably after the fact without re-tokenising, and re-tokenising with the wrong tokenizer gives wrong answers.

### Four numbers worth watching

- **Cost per request** by feature. Catches a prompt change that doubled context size.
- **Cost per successful request.** Divides by success rate; a feature with 60% failure is twice as expensive per good outcome as it looks.
- **Cost per tenant per day** with a monthly projection. Catches the one customer whose integration retries every failed call.
- **Share of spend by model.** If a cheap model is 5% of calls but 40% of spend, something is routing wrong; see [Model Routing](../21-system-design/04-model-routing.md).

### Budgets and alerts that fire before the invoice

A monthly budget alert at 100% is a post-mortem, not a control. Layer three:

1. **Soft alert at 50% of the monthly forecast**, to whoever owns the feature.
2. **Hard alert at 90%**, to the on-call channel, with the top three tenants by spend in the message.
3. **Circuit breaker** for per-tenant daily caps: reject with a clear error, or degrade to a cheaper model, before the tenant burns a shared budget.

The circuit breaker is the part people skip and later regret. It must be a deliberate product decision — what does the user see when their cap is hit? — so write it down with the product owner.

### Unit economics is the number you actually defend

Cost per request is an engineering metric. Unit economics is a business one: cost per resolved ticket, per document processed, per active user per month. Compute it as total feature cost divided by the business outcome count, and plot it against price. A feature whose cost per resolved ticket is rising while ticket volume is flat has a product problem, not a billing problem.

## Code

A cost recorder that reads the price table once, prices a call from real token usage, and writes both the per-call cost and a running per-tenant daily total to Redis. The Redis part is what makes the circuit breaker possible in a multi-instance deploy.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import redis
import yaml


@dataclass(frozen=True)
class Price:
    input_per_1m: Decimal
    output_per_1m: Decimal
    cached_input_per_1m: Decimal | None


def load_prices(path: str = "config/prices.yaml") -> dict[str, Price]:
    raw = yaml.safe_load(open(path))["models"]
    return {
        model: Price(
            input_per_1m=Decimal(str(cfg["input_per_1m"])),
            output_per_1m=Decimal(str(cfg["output_per_1m"])),
            cached_input_per_1m=(
                Decimal(str(cfg["cached_input_per_1m"])) if "cached_input_per_1m" in cfg else None
            ),
        )
        for model, cfg in raw.items()
    }


PRICES = load_prices()
r = redis.Redis(decode_responses=True)


def cost_usd(
    model: str, tokens_in: int, tokens_out: int, cached_in: int = 0
) -> Decimal:
    price = PRICES[model]
    # Cached input is billed at a different rate; the rest at the normal input rate.
    cached_rate = price.cached_input_per_1m or price.input_per_1m
    fresh_in = max(tokens_in - cached_in, 0)
    return (
        Decimal(fresh_in) / Decimal(1_000_000) * price.input_per_1m
        + Decimal(cached_in) / Decimal(1_000_000) * cached_rate
        + Decimal(tokens_out) / Decimal(1_000_000) * price.output_per_1m
    )


def record_spend(tenant_id: str, feature: str, amount: Decimal) -> Decimal:
    """Adds to today's per-tenant and per-feature counters; returns tenant total today."""
    day = date.today().isoformat()
    pipe = r.pipeline()
    tenant_key = f"spend:{day}:tenant:{tenant_id}"
    feature_key = f"spend:{day}:feature:{feature}"
    pipe.incrbyfloat(tenant_key, float(amount))
    pipe.expire(tenant_key, 60 * 60 * 48)
    pipe.incrbyfloat(feature_key, float(amount))
    pipe.expire(feature_key, 60 * 60 * 48)
    pipe.execute()
    return Decimal(str(r.get(tenant_key) or 0))
```

Second snippet: the check that runs before a call, not after. It degrades rather than failing hard, because a hard failure at the cap is a bad user experience for everyone on that tenant, including the ones behaving well.

```python
from __future__ import annotations

from decimal import Decimal

from app.cost import record_spend, r
from app.config import settings


class BudgetExceeded(Exception):
    pass


def guard(tenant_id: str, feature: str) -> None:
    day = __import__("datetime").date.today().isoformat()
    spent = Decimal(str(r.get(f"spend:{day}:tenant:{tenant_id}") or 0))
    if spent >= settings.tenant_daily_cap_usd:
        # Product decision: surface a clear message, not a generic 500.
        raise BudgetExceeded(
            f"daily model budget reached for this workspace ({spent:.2f} USD); "
            "resets at midnight UTC"
        )
```

## Real use case

A recruitment platform runs three LLM features: CV parsing, JD writing and interview-question generation. The monthly bill triples in one week. Application logs show no traffic increase, and nobody can say which feature is responsible because all three write to the same log stream with no cost field.

After instrumenting per-call cost and grouping by feature and tenant, the culprit is immediate: JD writing is 12% of requests and 61% of spend, because a recent prompt change inlined the entire company style guide — roughly 9,000 tokens of context on every call — to improve tone. Cost per JD went from a few cents to a multiple of that, and nobody noticed because per-call cost was never displayed anywhere.

The fix was to compress the style guide into a 400-token summary plus a retrieval step for the relevant sections. Cost per JD fell sharply and the tone scores held. The design choice that made it diagnosable was recording `cost_usd` on the generation span and `feature` on the trace root, so a single group-by answered a question that had previously needed a week of guessing.

## Common mistakes

**Mistake:** hardcoded prices in the code — **Instead:** a dated price table in config, so historical traces keep their historical cost.

**Mistake:** ignoring cached-input and batch pricing — **Instead:** a third rate per model; otherwise your estimate is wrong exactly where you optimised.

**Mistake:** only a monthly budget alert — **Instead:** soft alert, hard alert with top tenants, and a per-tenant circuit breaker.

**Mistake:** tracking cost per request but not per success — **Instead:** divide by success rate; a failing feature looks cheap per call and expensive per outcome.

**Mistake:** no feature or tenant tag on the trace — **Instead:** tag at the root before the first model call, or the spend is unattributable.

## Checkpoint

1. Add a `cost_usd` field to every generation span in your service, computed from a YAML price table you filled in from the provider's current pricing page.
2. Write a script that reads one day of traces and prints spend by feature and by tenant, plus cost per successful request for each. Identify the top three.
3. Implement the per-tenant daily cap with a graceful degradation path and a test that proves a capped tenant gets the friendly error while other tenants keep working.

**Self-check:** exercise 1 is done when a single request's trace shows a `cost_usd` that equals the sum of its generation costs, and that number changes when you switch to a more expensive model in config.

## Go deeper

- [Cost Tracking](../07-llm-apis/08-cost-tracking.md) — the per-call mechanics this file aggregates.
- [Cost and Latency](../21-system-design/08-cost-latency.md) — design-level trade-offs between the two.
- [Tracing Design](05-tracing.md) — why the attributes must be bounded and consistent.
- [Model Routing](../21-system-design/04-model-routing.md) — the lever that usually cuts spend the most.

---

[<- Previous: Tracing Design](05-tracing.md) · [Stage index](README.md) · [Next: CI/CD for Evals](07-ci-cd-evals.md)
