# 7. Fault Tolerance

> **Stage 21 · System Design** — models rate-limit, providers go down, retrieval goes slow; this file is how the system degrades predictably instead of failing all at once.
>
> **You'll be able to:** set timeouts from a budget, retry with a total retry budget rather than per-call optimism, add circuit breakers, and define a degraded mode for each dependency.
>
> **Prerequisites:** [Retries and Rate Limits](../07-llm-apis/07-retries-rate-limits.md) · [Scalable AI Apps](01-scalable-ai-apps.md)

## Why this exists

An AI app depends on things that fail in ways ordinary web services mostly do not. Model providers rate-limit by token and by request, and the limits change. A model can be up but slow, returning tokens at a crawl. Retrieval degrades gradually as an index grows. A provider can be healthy for small requests and refusing large ones.

Naive handling makes it worse. Retry every failure three times and a rate-limit event becomes a retry storm that keeps you limited longer. Set a 60-second timeout on a 3-second product and every incident becomes a queue of stuck requests. Have one dependency, one path, and an outage is total.

The mental model: **every dependency gets a timeout, a bounded retry, a breaker, and a defined behaviour when it is unavailable — decided before the incident, not during it.**

## The mental model

```text
 request ─► [budget] ─► call
                  │      │
                  │      ├─ timeout (from remaining budget, not a constant)
                  │      ├─ retry: bounded, jittered, only idempotent + retryable
                  │      └─ breaker: open after N failures → fail fast
                  │
                  └─ degraded modes, in order of preference:
                     1. fallback provider (different model, same tier)
                     2. smaller tier (quality down, latency up acceptable)
                     3. cached / stale answer with a caveat
                     4. answer without retrieval, flagged
                     5. honest error with a retry-after
```

| Failure | Wrong response | Right response |
|---|---|---|
| 429 rate limit | retry immediately ×3 | honour `Retry-After`, back off with jitter, cap total attempts |
| Provider outage | keep trying until timeout | breaker opens, fall back to another provider or tier |
| Slow model (tokens crawl) | wait indefinitely | streaming idle timeout; abort and serve a partial or a fallback |
| Retrieval down | fail the request | answer without documents, visibly caveated |
| Tool timeout | retry a write | only retry if the tool is idempotent; otherwise check state first |

## Core concepts

### Timeouts derived from a budget

A timeout is not a preference; it is a share of your latency target. If p95 must be 3 seconds and the model hop gets 2.4 seconds, that is the timeout — and it shrinks as the request consumes budget, so a slow retrieval does not leave the model 2.4 seconds plus whatever retrieval took.

Apply timeouts at every hop: retrieval, model, each tool, and the whole request. For streaming responses, add an **idle** timeout (no token for N seconds) because a stream that never finishes looks like a hang, not an error.

### Retries with a total budget, not per-call optimism

Two rules make retries safe. **Only retry what is safe to retry**: rate limits, connection errors, 5xx, and timeouts on idempotent calls — never a 400, never a write without an idempotency key. And **budget them across the request**: at most 3 attempts total across all hops, with exponential backoff and full jitter, so a fleet of clients does not synchronise into a thundering herd.

Honour the provider's `Retry-After` when it sends one. If a rate limit says wait 20 seconds and your budget is 3, do not wait — fall back to another provider or return a clear error.

### Circuit breakers, and what they actually buy you

A breaker wraps a dependency and tracks failures. Closed: calls pass. Open: calls fail immediately for a cooldown window. Half-open: one probe decides whether to close.

The value is not preventing the failing call — it is preventing *your* resources from being consumed by calls that will fail. With a breaker open, requests fail in milliseconds instead of occupying workers for a full timeout, so the rest of the system stays responsive and your recovery is faster. Track breaker state as a metric and alert on transitions; an open breaker is an incident signal.

### Degraded modes, ranked and written down

For each dependency, decide what the product does when it is unavailable, and rank the options. A good default ladder: fallback provider at the same tier, then a smaller tier, then a stale cached answer with a visible caveat, then a partial answer (no retrieval), then an honest error with `Retry-After`.

Two requirements. **The degradation must be visible** — a `degraded: true` flag in the response and a field in the trace, so you can measure how often you are serving the fallback rather than guessing. And **degradation must not weaken security**: never skip a guardrail, an output check or a tenant filter because you are in a degraded path. That is how incidents become breaches (see [Guardrails](../20-ai-security/09-guardrails.md)).

### Load shedding beats collapse

Under sustained overload, the honest move is to serve fewer requests well rather than all requests badly. Shed by priority: drop batch and background traffic first, then free-tier requests, then everything beyond a concurrency cap, returning 429 with `Retry-After`. A system that sheds load stays up; a system that accepts everything falls over and takes minutes to recover.

## Code

A retry wrapper with a total attempt budget, jitter, and a rule about what is retryable — plus a breaker whose state you can observe. The important line is that a 400 is never retried.

```python
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field


class RetryableError(Exception):
    """Raised by the client for 429 / 5xx / connection errors only."""


class PermanentError(Exception):
    """Raised for 4xx (except 429): retrying cannot help."""


@dataclass
class Breaker:
    failure_threshold: int = 5
    cooldown_s: float = 30.0
    _failures: int = 0
    _opened_at: float = 0.0
    state: str = "closed"

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and time.monotonic() - self._opened_at > self.cooldown_s:
            self.state = "half_open"  # one probe gets through
            return True
        return False

    def record(self, ok: bool) -> None:
        if ok:
            self._failures, self.state = 0, "closed"
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self.state, self._opened_at = "open", time.monotonic()


@dataclass
class RetryBudget:
    max_attempts: int = 3
    base_delay_s: float = 0.4
    deadline_s: float = 2.4
    attempts: int = field(default=0, init=False)

    def can_retry(self, elapsed: float) -> bool:
        return self.attempts < self.max_attempts and elapsed < self.deadline_s


async def call_with_retry(fn, breaker: Breaker, budget: RetryBudget | None = None):
    budget = budget or RetryBudget()
    started = time.monotonic()

    while True:
        if not breaker.allow():
            raise RetryableError("circuit open")  # fail fast; caller degrades
        budget.attempts += 1
        try:
            result = await fn()
            breaker.record(True)
            return result
        except PermanentError:
            breaker.record(True)  # a 4xx is not the dependency's fault
            raise
        except RetryableError:
            breaker.record(False)
            elapsed = time.monotonic() - started
            if not budget.can_retry(elapsed):
                raise
            # Full jitter: spreads a fleet of clients instead of synchronising them.
            delay = random.uniform(0, budget.base_delay_s * (2 ** (budget.attempts - 1)))
            await asyncio.sleep(min(delay, max(budget.deadline_s - elapsed, 0)))
```

The degradation ladder as a single function, so every caller degrades the same way and the trace records which rung was used.

```python
from __future__ import annotations


async def answer_with_fallback(deps, session, question: str) -> dict:
    trace: dict[str, str] = {}
    try:
        text = await deps.models.complete(tier="mid", question=question)
        return {"text": text, "degraded": False, "trace": trace}
    except Exception:
        trace["primary"] = "failed"

    try:
        text = await deps.models.complete(tier="small", question=question)
        return {"text": text, "degraded": True, "trace": {**trace, "rung": "small_tier"}}
    except Exception:
        trace["small_tier"] = "failed"

    cached = await deps.cache.get_answer(session.tenant_id, question)
    if cached:
        # Stale but useful: label it, and never skip output validation here.
        return {
            "text": cached + "\n\n(Note: served from cache; may be out of date.)",
            "degraded": True,
            "trace": {**trace, "rung": "stale_cache"},
        }

    return {
        "text": "The assistant is temporarily unavailable. Please retry in a minute.",
        "degraded": True,
        "trace": {**trace, "rung": "error"},
    }
```

## Real use case

**A customer-facing product-recommendation assistant on a provider that rate-limited by tokens.** During a marketing campaign, traffic tripled. The app retried every 429 three times with no jitter and no total budget, so each rate-limited request became three more, the provider's limiter kept everyone in the penalty box for over an hour, and the connection pool filled with waiting requests.

Three changes fixed it, and the order matters. Timeouts were derived from a 3-second product budget, so no request could occupy a worker for a minute. Retries became a single budget of three attempts across the whole request, with full jitter and `Retry-After` honoured — which turned a retry storm into a queue that drained. A breaker on the primary provider opened after five consecutive failures and routed to a second provider at the same tier, and when that was also unavailable, the ladder fell to a stale cached answer with a visible caveat. Requests that could not be served got a 429 with `Retry-After` instead of a timeout, and background batch traffic was shed first. The campaign's peak was higher than the one that had caused the outage, and it passed without a visible incident.

## Common mistakes

- **Mistake:** retrying every error three times — **Instead:** retry only rate limits, connection errors, 5xx and idempotent timeouts, with jitter and a total attempt budget.
- **Mistake:** fixed per-hop timeouts that sum to more than the product target — **Instead:** derive each hop's timeout from the remaining budget.
- **Mistake:** skipping guardrails or tenant filters in the degraded path — **Instead:** degrade quality and freshness, never security.
- **Mistake:** silent fallbacks nobody measures — **Instead:** set `degraded: true`, record the rung in the trace, and alert on the rate.
- **Mistake:** accepting all traffic until the system collapses — **Instead:** shed by priority and return 429 with `Retry-After`.

## Checkpoint

1. Wrap one dependency call in a timeout derived from a total budget and verify with a deliberately slow stub that the request still returns within the target.
2. Implement the breaker and write three tests: it opens after N failures, it fails fast while open, and it closes after a successful probe.
3. Write the degradation ladder for an app you have built — four rungs, in order — and make one of them visible in the response.

**Self-check:** The breaker fails fast so that requests cost milliseconds instead of a full timeout, which keeps workers free and the rest of the system responsive — the state transition is also a metric you should alert on.

## Go deeper

- [Cost and Latency Engineering](08-cost-latency.md) — the budget these timeouts come from.
- [Scaling Trade-offs](09-scaling-tradeoffs.md) — when redundancy is worth its complexity.
- [Model Routing](04-model-routing.md) — fallback as a routing decision.
- [Ship: Design Document](10-ship-design-doc.md) — the failure-mode table a reviewer expects.

---
[← Previous: Caching Layers](06-caching.md) · [Stage index](README.md) · [Next: Cost and Latency Engineering](08-cost-latency.md)
