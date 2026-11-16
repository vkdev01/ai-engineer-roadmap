# 9. Routing and Fallbacks

> **Stage 07 · LLM APIs** — choosing a model at runtime, and degrading instead of failing.
>
> **You'll be able to:** route requests by task and cost, fall back across providers on error or bad output, and avoid retry storms.
>
> **Prerequisites:** [Retries and Rate Limits](07-retries-rate-limits.md) · [Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md)

## Why this exists

One model for every request is the simplest design and usually the wrong one. A classification call does not need your most expensive model; a legal-document summary should not run on your cheapest. And when a provider has a bad ten minutes, a single-provider system has no answer except an error page.

Routing and fallback are two halves of one idea: **a request carries requirements, and a policy picks a model that meets them, with alternatives behind it.** Routing is the happy-path decision; fallback is what happens when the choice fails.

The mental model: **a list of candidates in preference order, plus rules for advancing down the list.**

## The mental model

```text
request(requirements)
    |
    v
[ router ] --> candidate list: [A, B, C]
    |
    v
try A --ok--> answer
    |
   fail (error | timeout | invalid output | too slow)
    |
try B --ok--> answer  (+ mark A degraded)
    |
   fail
    |
try C --> answer, or typed AllProvidersFailed
```

| Trigger to advance | Detects | Cost of getting it wrong |
|---|---|---|
| HTTP error / timeout | infrastructure | low — clear signal |
| Invalid output | schema or quality | medium — needs validation |
| Latency over budget | slow tail | medium — needs a clock |
| Cost over budget | spend | low — known before the call |
| Confidence below floor | quality | high — needs an estimator |

## Core concepts

### Route on task, not on vibes

Make the routing key explicit: `task="classify"` or `task="long_summary"`. Each task maps to a policy with a preferred model, a max token budget and a fallback list. That table is reviewable, testable and easy to change when prices move.

### Fallback needs a failure predicate

"Fall back on error" is easy. The valuable cases are quality failures: the model returned prose instead of JSON, or an answer shorter than the question implies. Define a validator per task — a schema check, a length check, a refusal check — and advance the candidate list when it fails.

### Fallback across providers beats fallback within one

Two models from the same provider share a status page. If the provider is down or you are rate-limited on the account, a second model from the same vendor does not help. Keep at least one candidate on a different provider, and verify that your prompt works there — templates, system-prompt handling and tool schemas differ.

### Degrade visibly, recover quietly

When you serve from a fallback, record it. Users do not need to know, but your metrics do: fallback rate is the earliest signal that a provider is degrading. Hide it in the UI; never hide it in the logs.

### The router itself must not become a bottleneck

A router that calls another LLM to decide which LLM to call adds latency and a new failure point. Start with a static table keyed by task. Add learned routing only when you have data showing the table is wrong.

## Code

A policy table, a candidate walker with a failure predicate, and a per-provider adapter registry.

```python
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

Task = Literal["classify", "summarise", "reason"]


@dataclass(frozen=True)
class Candidate:
    provider: str          # "openai" | "anthropic" | "hosted"
    model: str             # from config; check docs for current ids
    timeout_s: float = 20.0


@dataclass(frozen=True)
class Policy:
    candidates: tuple[Candidate, ...]   # preference order
    max_output_tokens: int = 800
    min_chars: int = 1                  # crude quality floor


POLICIES: dict[Task, Policy] = {
    "classify": Policy(
        candidates=(
            Candidate("hosted", "small-open-model", 10.0),   # cheap first
            Candidate("openai", "small-model", 15.0),
        ),
        max_output_tokens=64,
    ),
    "summarise": Policy(
        candidates=(
            Candidate("openai", "mid-model", 30.0),
            Candidate("anthropic", "mid-model", 30.0),       # other provider
        ),
    ),
    "reason": Policy(
        candidates=(
            Candidate("anthropic", "strong-model", 60.0),
            Candidate("openai", "strong-model", 60.0),
        ),
        max_output_tokens=2_000,
        min_chars=40,
    ),
}

# registry: provider name -> callable(model, prompt) -> text
Adapters = dict[str, Callable[[str, str], Awaitable[str]]]


class AllProvidersFailed(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures))
        self.failures = failures


@dataclass
class RouteResult:
    text: str
    model: str
    provider: str
    attempts: list[str] = field(default_factory=list)


async def route(
    task: Task, prompt: str, adapters: Adapters
) -> RouteResult:
    policy = POLICIES[task]
    failures: list[str] = []

    for cand in policy.candidates:
        call = adapters.get(cand.provider)
        if call is None:
            failures.append(f"{cand.provider}: no adapter")
            continue
        try:
            text = await asyncio.wait_for(
                call(cand.model, prompt), timeout=cand.timeout_s
            )
        except asyncio.TimeoutError:
            failures.append(f"{cand.provider}/{cand.model}: timeout")
            continue
        except Exception as exc:                      # provider error
            failures.append(f"{cand.provider}/{cand.model}: {exc}")
            continue

        # quality gate: advance the list on a bad answer, not just an error
        if len(text.strip()) < policy.min_chars:
            failures.append(f"{cand.provider}/{cand.model}: too short")
            continue
        return RouteResult(text=text, model=cand.model,
                           provider=cand.provider, attempts=failures)

    raise AllProvidersFailed(failures)
```

The caller gets one shape and a record of what was tried. Note that `attempts` doubles as your fallback metric.

In TypeScript, express the same thing with a small helper that races a timeout, which `fetch` does not give you directly.

```ts
export type Candidate = { provider: string; model: string; timeoutMs: number };
export type Adapter = (model: string, prompt: string) => Promise<string>;

export class AllProvidersFailed extends Error {
  constructor(readonly failures: string[]) {
    super(failures.join("; "));
  }
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    p,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms),
    ),
  ]);
}

export async function route(
  candidates: Candidate[],
  prompt: string,
  adapters: Record<string, Adapter>,
  minChars = 1,
): Promise<{ text: string; provider: string; model: string; failures: string[] }> {
  const failures: string[] = [];
  for (const c of candidates) {
    const adapter = adapters[c.provider];
    if (!adapter) {
      failures.push(`${c.provider}: no adapter`);
      continue;
    }
    try {
      const text = await withTimeout(adapter(c.model, prompt), c.timeoutMs);
      if (text.trim().length < minChars) {
        failures.push(`${c.provider}/${c.model}: too short`);
        continue;
      }
      return { text, provider: c.provider, model: c.model, failures };
    } catch (err) {
      failures.push(`${c.provider}/${c.model}: ${String(err)}`);
    }
  }
  throw new AllProvidersFailed(failures);
}
```

## Real use case

A document-review product ran every request on one large model. Two problems appeared: the classification step ("is this clause risky?") was the most frequent call and the most expensive per unit of value, and one provider's incident took the whole product down for half an hour.

The team introduced three tasks with their own policies. Classification went to a hosted open-weight model first with a small closed model as backup; summarisation used one provider with the other behind it; reasoning kept the strong model but added a second provider candidate. The incident class disappeared: a provider outage produced a degraded answer instead of an error, and the fallback-rate metric flagged it within minutes. The failure mode before was architectural — a single model id in a single code path, with no notion of an alternative.

## Common mistakes

- **Mistake:** putting all fallback candidates on one provider. — **Instead:** keep at least one candidate on a different provider and test your prompt against it.
- **Mistake:** falling back only on HTTP errors. — **Instead:** add a quality predicate per task — schema, length, refusal detection.
- **Mistake:** falling back on a 400 caused by your own bad request. — **Instead:** let client errors fail fast; every candidate will reject them identically.
- **Mistake:** silently serving degraded answers with no metric. — **Instead:** record which candidate answered, and alert on fallback rate.
- **Mistake:** using an LLM to choose the model on every request. — **Instead:** start with a static task-to-policy table you can review.

## Checkpoint

1. Add a `fast` task whose policy prefers the cheapest candidate and falls back on any answer longer than the `max_output_tokens` cap implies truncation.
2. Force the first candidate to time out and verify the second answers within the caller's deadline.
3. Emit a counter per `(task, provider, outcome)` and compute the fallback rate for a synthetic day of traffic.

**Self-check:** you iterate the policy's candidate list in order, catch timeout and error per candidate, apply a quality predicate before accepting an answer, and raise a typed error listing every failure if all candidates fail.

## Go deeper

- [Retries and Rate Limits](07-retries-rate-limits.md) — retry before you fall back, not instead of it.
- [Cost Tracking](08-cost-tracking.md) — per-model cost is what makes routing decisions measurable.
- [Model Routing](../21-system-design/04-model-routing.md) — the system-design version, with learned routers.
- [Ship: A Multi-Model Client](10-ship-multi-model-client.md) — this policy inside a real library.

---
[← Previous: Token Accounting and Cost Tracking](08-cost-tracking.md) · [Stage index](README.md) · [Next: Ship: A Multi-Model Client](10-ship-multi-model-client.md)
