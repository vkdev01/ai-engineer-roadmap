# 7. Retries and Rate Limits

> **Stage 07 · LLM APIs** — surviving 429s and 5xx without making the outage worse.
>
> **You'll be able to:** implement exponential backoff with jitter, distinguish retryable from terminal failures, and respect provider rate limits.
>
> **Prerequisites:** [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [HTTP and REST APIs](../01-programming-basics/07-http-rest-apis.md)

## Why this exists

LLM endpoints fail more often than a database does. They rate-limit aggressively, they have bad minutes, and they occasionally return a 200 with a truncated or empty body. A single un-retried call turns a transient blip into a user-visible error; a naive retry loop turns a blip into a self-inflicted outage, because every client retries at the same instant and multiplies load exactly when the provider is struggling.

The mental model: **retries are a queueing problem, not a loop.** You are trying to spread your demand over time so the system recovers, and jitter is what stops all your clients from retrying in lockstep.

## The mental model

```text
attempt 1  ---> 429 (Retry-After: 2s)
   sleep 2s + jitter
attempt 2  ---> 503
   sleep 4s + jitter
attempt 3  ---> 200  -> done

give up after N attempts OR total deadline exceeded
```

| Status | Meaning | Retry? |
|---|---|---|
| 429 | rate limited | yes, honour `Retry-After` |
| 500 / 502 / 503 / 504 | provider-side failure | yes, with backoff |
| 408, connection reset | timeout | yes, with backoff |
| 400 | bad request / bad schema | no — your bug |
| 401 / 403 | auth or permission | no |
| 404 | wrong model id or path | no |
| 422 | validation rejected | no |

## Core concepts

### Retry only what is retryable

The most damaging bug is retrying a 400. It burns your quota, delays the real error, and hides a broken prompt behind three slow attempts. Classify the status first, then decide. Anything caused by your request being wrong must fail fast.

### Exponential backoff with full jitter

Wait `base * 2**attempt`, then multiply by a random factor. Full jitter means sleeping a uniform random duration in `[0, computed_delay]`; equal jitter means half the delay plus random half. Either spreads the herd. Without jitter, every instance that failed at second zero retries at second two together.

### Honour Retry-After, and know your limits

Providers send a `Retry-After` header on 429s and expose rate-limit headers on successful calls too. Reading them lets you slow down *before* you get blocked. Limits are usually counted in both requests per minute and tokens per minute, and a long prompt can trip the token limit while your request rate looks fine.

### Bound by attempts and by deadline

Two independent caps. Attempt count stops a hot loop; a wall-clock deadline stops a slow death where each attempt times out at 60 seconds and the user waits four minutes. Set the deadline from the caller's patience, not from the retry policy's convenience.

### Idempotency matters when the call has effects

A retried chat completion is harmless. A retried tool call that charged a card is not. If a retry can duplicate a side effect, pass an idempotency key or make the operation safe to repeat.

## Code

A retry helper with classification, jitter, `Retry-After` support and a deadline.

```python
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class RetriesExhausted(RuntimeError):
    pass


def _delay(attempt: int, base: float, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after            # the provider told us exactly how long
    # exponential with full jitter: uniform in [0, base * 2**attempt]
    return random.uniform(0, base * (2**attempt))


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base: float = 0.5,
    deadline_s: float = 30.0,
) -> T:
    started = time.monotonic()
    last: Exception | None = None

    for attempt in range(attempts):
        try:
            return await fn()
        except RateLimitError as exc:
            last = exc
            retry_after = _header_float(exc, "retry-after")
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS:
                raise                 # 400/401/404: our bug, fail fast
            last = exc
            retry_after = _header_float(exc, "retry-after")
        except (APITimeoutError, APIConnectionError) as exc:
            last = exc
            retry_after = None

        remaining = deadline_s - (time.monotonic() - started)
        sleep_for = _delay(attempt, base, retry_after)
        if sleep_for >= remaining:
            break                     # no point sleeping past the deadline
        await asyncio.sleep(sleep_for)

    raise RetriesExhausted(f"gave up after {attempts} attempts") from last


def _header_float(exc: APIStatusError, name: str) -> float | None:
    raw = exc.response.headers.get(name) if exc.response is not None else None
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None
```

Usage is then a one-line wrapper around any call:

```python
async def ask(prompt: str) -> str:
    return await with_retries(lambda: call_model(prompt), attempts=3, deadline_s=20)
```

The TypeScript version is the same policy, written by hand because `fetch` gives you the status code directly.

```ts
export class RetriesExhausted extends Error {}

const RETRYABLE = new Set([408, 409, 429, 500, 502, 503, 504]);

export async function withRetries<T>(
  fn: () => Promise<Response>,
  { attempts = 4, base = 500, deadlineMs = 30_000 } = {},
): Promise<T> {
  const started = Date.now();
  let lastError: unknown;

  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const res = await fn();
      if (res.ok) return (await res.json()) as T;

      if (!RETRYABLE.has(res.status)) {
        // 400/401/404: retrying cannot help, surface it now
        throw new Error(`terminal ${res.status}: ${await res.text()}`);
      }
      lastError = new Error(`retryable ${res.status}`);
      const retryAfter = Number(res.headers.get("retry-after"));
      const waitMs = Number.isFinite(retryAfter)
        ? retryAfter * 1000
        : Math.random() * base * 2 ** attempt; // full jitter

      if (Date.now() - started + waitMs > deadlineMs) break;
      await new Promise((r) => setTimeout(r, waitMs));
    } catch (err) {
      lastError = err;
      if (String(err).includes("terminal")) throw err;
      const waitMs = Math.random() * base * 2 ** attempt;
      if (Date.now() - started + waitMs > deadlineMs) break;
      await new Promise((r) => setTimeout(r, waitMs));
    }
  }
  throw new RetriesExhausted(`gave up after ${attempts} attempts: ${lastError}`);
}
```

## Real use case

A batch job summarising support tickets ran overnight with 20 worker coroutines, each making one call per ticket. During peak load the provider returned 429s, the workers retried immediately, and the job made the rate limiting worse — it finished late and left a backlog for the morning team.

The fix had three parts: read `Retry-After` and honour it, add full jitter so the twenty workers stopped marching in step, and add a global concurrency limit so the job stayed under the token-per-minute ceiling. The job then finished within its window without a single unhandled 429. The failure mode before was not too little retrying — it was retrying in a way that amplified the load that caused the failure.

## Common mistakes

- **Mistake:** retrying every exception. — **Instead:** retry 429, 5xx and timeouts only; let 4xx fail fast.
- **Mistake:** fixed delays or no jitter. — **Instead:** exponential backoff with full jitter, or honour `Retry-After`.
- **Mistake:** retrying forever. — **Instead:** cap both attempts and total wall-clock time.
- **Mistake:** ignoring token-per-minute limits because request counts look fine. — **Instead:** track input tokens per window; a big prompt counts against you.
- **Mistake:** retrying a call with side effects. — **Instead:** make it idempotent or use an idempotency key before you retry.

## Checkpoint

1. Simulate a 400 and confirm your helper raises immediately without sleeping.
2. Log every retry with attempt number, delay and status, then plot the delays for one bad request to see the jitter spread.
3. Add a semaphore limiting concurrent calls and verify the 429 rate drops under a synthetic burst.

**Self-check:** your helper inspects the status code first, re-raises immediately for 4xx client errors, and only sleeps with backoff for 429, 5xx and network timeouts.

## Go deeper

- [Cost Tracking](08-cost-tracking.md) — retries cost money; count them.
- [Routing and Fallbacks](09-routing-fallbacks.md) — the next step after retries: try another model.
- [Rate Limits and Auth](../17-ai-backend/06-auth-rate-limits.md) — the same discipline on your own API surface.
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — retries as one layer of a wider resilience design.

---
[← Previous: Multimodal APIs](06-multimodal-apis.md) · [Stage index](README.md) · [Next: Token Accounting and Cost Tracking](08-cost-tracking.md)
