# 4. Model Routing

> **Stage 21 · System Design** — most requests do not need your most expensive model; this file is how you decide which one serves each request, and prove the decision is good.
>
> **You'll be able to:** design a tiered routing policy, implement classifier-based routing and a cascade with escalation, and evaluate whether a route is actually better than the alternative.
>
> **Prerequisites:** [Routing and Fallbacks](../07-llm-apis/09-routing-fallbacks.md) · [Model Routing Costs](../07-llm-apis/08-cost-tracking.md)

## Why this exists

A single-model app has one dial: which model. Set it to the frontier model and simple questions cost as much as hard ones; set it to the small model and the hard questions get worse answers, silently. The distribution of real requests is lopsided — in most products the majority of requests are simple, a minority are hard, and a small tail is genuinely difficult.

Routing is the discipline of matching the model to the request. Done well it cuts cost substantially with no measurable quality loss, because the requests that got cheaper were ones the small model already handled perfectly. Done badly it silently degrades quality for exactly the users who care most, because nobody measured quality per route.

The mental model: **routing is a classifier plus a policy, and the classifier's error in both directions has a cost you must measure.**

## The mental model

```text
 request ──► features ──► router ──► tier ──► answer
              │            │
              │            ├─ rules: length, task type, tenant tier, language
              │            ├─ classifier: predicted difficulty (cheap model)
              │            └─ cascade: try small, escalate on low confidence
              │
              └─ feedback: did the answer pass checks? (eval, thumbs, validator)

 tiers:  small (fast, cheap) → mid → frontier (slow, expensive)
```

| Strategy | How it decides | When to use | Risk |
|---|---|---|---|
| Static per feature | config | stable, well-understood tasks | leaves savings on the table |
| Rule-based | length, intent, tenant plan | predictable, auditable | brittle at the edges |
| Classifier | small model predicts difficulty | high volume, varied requests | classifier errors both ways |
| Cascade | run small, escalate if unsure | quality-critical, cost-sensitive | double cost on escalated requests |
| Tenant policy | plan or contract | enterprise SLAs | complexity in pricing |

## Core concepts

### Tiers, not models

Name tiers by role — `small`, `mid`, `frontier` — and map each to a concrete model id in config. Model ids change, prices change and providers deprecate; the routing policy should not need a code change when they do. Keep the mapping in one versioned file, and record the tier (and the resolved model id) in every trace so you can attribute quality and cost later.

### Features you can compute for free

Before you pay for a classifier, use what you already have: input length, presence of code, number of retrieved chunks and their scores, conversation depth, requested output format, tenant plan, language, and the task's own label. A large share of routing decisions are decidable by rules, and rules are auditable in a way a classifier is not.

A useful default policy: short input plus high retrieval confidence plus a structured-output task routes `small`; long input, low retrieval confidence, or a multi-step task routes `mid`; anything flagged as high-stakes (a tool call with a side effect, an explicit "be thorough") routes `frontier`.

### Cascades: pay twice only when it matters

A cascade runs the cheap model first, applies a **cheap confidence check**, and escalates only when the check fails. The check is the design: a schema validator, a groundedness check against retrieved documents, a small judge, or a self-reported uncertainty signal.

The arithmetic is worth doing before you build one. Labelled example, **prices are placeholders — substitute current published values**:

| Assumption | Value |
|---|---|
| Requests per month | 1,000,000 |
| Escalation rate (measured on a sample) | 15% |
| Cost per request, small tier | `$S` |
| Cost per request, frontier tier | `$F`, with `$F ≈ 10 × $S` for illustration |
| Cost of the confidence check | `$C`, roughly `0.1 × $S` |

Cascade cost per request: `$S + $C + 0.15 × $F`. With `$F = 10$S` and `$C = 0.1$S`: `$S + 0.1$S + 1.5$S = 2.6$S`. Compare with `$F = 10$S` for always-frontier: the cascade is about a quarter of the cost, provided quality on the escalated subset matches. If the escalation rate were 60%, cascade cost would be `0.1$S + 0.6 × 10$S ≈ 6.1$S` — still cheaper, but the latency tail gets worse because those requests now make two model calls.

Two consequences you must state: **p95 latency rises** for escalated requests (two calls instead of one), and **your cost is now sensitive to a number you must monitor** — the escalation rate. A prompt change that makes the small model look less confident can double your bill with no quality change.

### Evaluate routes, not just the app

The mistake is measuring quality on the whole app and calling the router good. Measure **per route, per slice**: the pass rate of `small`-routed answers versus what `frontier` would have produced on the same inputs. That requires a golden set with the route recorded and, ideally, a paired comparison — run a sample of `small`-routed requests through `frontier` offline and compare.

Track four numbers: cost per request by tier, quality by route, escalation rate, and the **misroute rate in both directions** — hard requests sent to `small` (quality risk) and easy requests sent to `frontier` (money wasted). The first is the one that hurts; set an alert on it.

## Code

A router with rules first, a classifier second, and a policy that returns a tier plus the reason. The reason is not decoration — it is what makes routing debuggable when quality drops.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

Tier = Literal["small", "mid", "frontier"]

# One place to change model ids. Ids and prices change: check provider docs.
TIER_MODELS: dict[str, str] = {
    "small": "<small-model-id>",
    "mid": "<mid-model-id>",
    "frontier": "<frontier-model-id>",
}


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    retrieved_chunks: int = 0
    top_score: float = 0.0
    wants_structured: bool = False
    high_stakes: bool = False
    tenant_plan: Literal["free", "pro", "enterprise"] = "free"


@dataclass(frozen=True)
class Route:
    tier: Tier
    reason: str


def route(req: Request) -> Route:
    # 1. Rules you can audit, in priority order.
    if req.high_stakes:
        return Route("frontier", "rule:high_stakes")
    if req.tenant_plan == "enterprise" and not req.wants_structured:
        return Route("mid", "rule:enterprise_floor")

    # 2. Free features.
    if req.wants_structured and len(req.text) < 600 and req.top_score >= 0.75:
        return Route("small", "rule:short_structured_grounded")
    if req.retrieved_chunks == 0 and len(req.text) < 300:
        return Route("small", "rule:short_no_context")
    if req.top_score < 0.5 or len(req.text) > 4_000:
        return Route("frontier", "rule:weak_retrieval_or_long")

    return Route("mid", "default:mid")


def resolve(route_: Route) -> str:
    return TIER_MODELS[route_.tier]  # never hardcode a model id at the call site
```

The cascade, with the confidence check that decides escalation. Note that the escalated call reuses the same prompt — you are not asking the small model to hand over its reasoning, which would cost tokens and leak style.

```python
from __future__ import annotations


async def complete_cascaded(client, prompt: str, validator) -> tuple[str, str]:
    """Try small, escalate once if the validator is not satisfied."""
    cheap = await client.complete(resolve(Route("small", "cascade")), prompt)
    ok, _ = validator(cheap.text)  # schema / groundedness / judge
    if ok:
        return cheap.text, "small"

    # Escalation: one attempt only. Never cascade more than two deep.
    strong = await client.complete(resolve(Route("frontier", "cascade")), prompt)
    return strong.text, "frontier:escalated"


def groundedness_validator(retrieved: list[str]):
    def check(answer: str) -> tuple[bool, str]:
        # Cheap proxy: does the answer share enough rare terms with the context?
        words = {w.lower() for w in answer.split() if len(w) > 5}
        context = {w.lower() for w in " ".join(retrieved).split() if len(w) > 5}
        if not words:
            return False, "empty answer"
        overlap = len(words & context) / len(words)
        return overlap >= 0.35, f"overlap={overlap:.2f}"  # tune on your data

    return check
```

## Real use case

**A coding assistant inside a CI product, used by developers for commit-message suggestions and for explaining test failures.** Both used the frontier model, at roughly a million requests a month. Commit messages are short, structured and easy; test-failure explanations are long, context-heavy and hard. The uniform model meant paying frontier prices for one-line summaries.

The team introduced two tiers and a rule set: commit messages route `small` (short, structured, no retrieval); failure explanations route `mid` by default, and `frontier` when the log exceeds 4,000 characters or the retrieval confidence is low. Quality was measured per route against a golden set, with a paired offline comparison of `small` versus `frontier` on 500 commit-message cases — the small model matched on the same share of cases the team required before switching. Cost per request for the commit-message path fell sharply, and the frontier path was unchanged. Six weeks later a prompt change pushed the weak-retrieval rule's hit rate up; the misroute alert on "hard requests sent to small" caught it before the quality numbers in the weekly report did.

## Common mistakes

- **Mistake:** hardcoding model ids at call sites — **Instead:** a tier map in one versioned config file, with the tier recorded in the trace.
- **Mistake:** a cascade with no escalation cap — **Instead:** escalate at most once, and monitor the escalation rate as a first-class metric.
- **Mistake:** measuring app-wide quality and calling the router validated — **Instead:** measure per route, with a paired comparison against the alternative tier.
- **Mistake:** ignoring the latency tail a cascade creates — **Instead:** report p95 by route, and let high-stakes or latency-sensitive requests skip the cascade.
- **Mistake:** quoting a price ratio between tiers as fact — **Instead:** use placeholders, and recompute the cascade arithmetic when prices change.

## Checkpoint

1. Write a rule-based router for an app you have built, returning a tier and a reason string. Log both and run it over a week of sampled traffic.
2. Compute the cascade cost for your traffic using the formula above, with your own escalation rate. Compare it to always-frontier.
3. Build a paired comparison on 200 golden cases: run each through both tiers, and report the share where the cheaper tier passed. Decide whether the switch is safe, and write the sentence you would say in a review.

**Self-check:** The route function returns a `Route` with tier and reason, resolved to a model id only through the config map — that separation is what lets you swap models and audit decisions without touching routing logic.

## Go deeper

- [Cost and Latency Engineering](08-cost-latency.md) — turning route data into unit economics.
- [Fault Tolerance](07-fault-tolerance.md) — routing as a fallback mechanism when a tier is down.
- [Online and Offline Evals](../15-evals/09-online-offline-evals.md) — measuring quality per route in production.
- [Ship: Design Document](10-ship-design-doc.md) — defending the routing policy in a review.

---
[← Previous: Agent Design](03-agent-design.md) · [Stage index](README.md) · [Next: Multi-Tenant Design](05-multi-tenant.md)
