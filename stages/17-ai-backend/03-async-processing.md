# 3. Async Processing

> **Stage 17 · AI Backend** — keeping the event loop free so concurrency actually happens.
>
> **You'll be able to:** find blocking calls in an async service and move them off the loop deliberately.
>
> **Prerequisites:** [Async Python](../01-programming-basics/01-python-async.md) · [FastAPI and Flask](01-fastapi-flask.md)

## Why this exists

An async server runs all your request handlers on one thread, one event loop. That is how it
holds hundreds of in-flight requests cheaply: while one handler awaits a model response, the
loop runs another handler. There is no second thread to rescue you.

So if any handler does something that *blocks* — a synchronous HTTP call, a big pandas
operation, a CPU-heavy tokenizer pass, a file read on a slow disk — the loop stops. Every other
request stops with it. Your service degrades to one-at-a-time throughput and looks mysteriously
slow, with low CPU usage that points nowhere.

The mental model: **the event loop is a single lane. Anything that does not `await` blocks the
lane, no matter how harmless it looks.**

## The mental model

```text
GOOD: awaiting I/O                       BAD: blocking the loop
loop: [req A await ...][req B ...]       loop: [req A -- blocking 2s --][req B ...]
      [req C await ...][req D ...]             B waited 2s, C waited 2s, D waited 2s
      10 requests in ~2s                      10 requests in ~20s
```

| Work type | Correct tool |
|---|---|
| HTTP to a provider, async SDK | `await` |
| HTTP to a provider, sync SDK | `run_in_threadpool` |
| Database with async driver | `await` |
| Database with sync driver | `run_in_threadpool` |
| pandas / numpy over big frames | worker process or job queue |
| Tokenizing large batches, PDF parsing | worker process or job queue |
| `time.sleep`, `requests.get` | never in a handler |

## Core concepts

### The arithmetic that makes it obvious

Ten requests each needing a 2-second wait:

- Awaiting properly: all ten overlap, wall clock is about 2 seconds.
- Blocking: each takes 2 seconds in the lane, wall clock is about 20 seconds.

The ratio is the concurrency you threw away. Under load this compounds: at 100 requests the
difference is 200 seconds versus 2, and the service effectively falls over.

### `run_in_threadpool` is the escape hatch, not the plan

FastAPI gives you `run_in_threadpool` (and Starlette's threadpool) to run a synchronous function
without blocking the loop. It works because the GIL is released during I/O, so a parked thread
does not stop the loop from running other handlers.

It is not a plan because each concurrent call consumes a thread from a bounded pool. Once the
pool is saturated, the loop is free but your sync work queues. Use it for a legacy SDK you cannot
replace; do not use it to run a 3-second CPU-bound job, because CPU-bound Python holds the GIL
and you have re-blocked the loop with extra steps.

### CPU-bound work belongs in a process

Parsing a large PDF, embedding 5,000 chunks, running a local tokenizer over a long document:
these hold the GIL. The right home is a separate process — a worker via a job queue
([Job Queues](05-job-queues.md)), or `run_in_executor` with a `ProcessPoolExecutor` for short
bursts. The rule of thumb: if it is CPU-bound and takes more than a few tens of milliseconds,
it does not belong in a request handler.

### Detect it instead of guessing

You cannot reliably eyeball this in code review. Two practical instruments: enable asyncio debug
mode to get "executing <Task> took X seconds" warnings, and measure event-loop lag in production
by scheduling a task that sleeps 50 ms and recording how late it wakes. If lag spikes when a
particular endpoint is called, you have found the blocker.

## Code

First, the difference made measurable. Both functions do the same "slow work" ten times; only
one of them keeps the loop responsive.

```python
from __future__ import annotations

import asyncio
import time

from fastapi.concurrency import run_in_threadpool


def blocking_io(i: int) -> str:
    """Stands in for a synchronous SDK or a legacy client. Blocks its thread."""
    time.sleep(0.2)
    return f"result-{i}"


async def bad_gather() -> float:
    """Sequential because each call blocks the only thread there is."""
    start = time.perf_counter()
    for i in range(10):
        blocking_io(i)                 # loop is frozen here
    return time.perf_counter() - start


async def good_gather() -> float:
    """Concurrent: the threadpool runs the blocking calls while the loop stays free."""
    start = time.perf_counter()
    await asyncio.gather(*[run_in_threadpool(blocking_io, i) for i in range(10)])
    return time.perf_counter() - start


async def loop_lag_probe(stop: asyncio.Event, lag: list[float]) -> None:
    """Samples how late a 50 ms sleep wakes up; high lag means a blocked loop."""
    while not stop.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(0.05)
        lag.append((time.perf_counter() - t0 - 0.05) * 1000.0)


async def main() -> None:
    print(f"blocking:  {await bad_gather():.2f}s")     # expect ~2.0s
    print(f"threadpool: {await good_gather():.2f}s")   # expect ~0.2s

    stop, lag = asyncio.Event(), []
    probe = asyncio.create_task(loop_lag_probe(stop, lag))
    await bad_gather()                                  # same blocking work, now measured
    stop.set()
    await probe
    print(f"peak loop lag during blocking work: {max(lag):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
```

You should see roughly a tenfold wall-clock difference and a loop-lag peak in the hundreds of
milliseconds — the number that tells you a production endpoint would stall every other request.

Second, the pattern that keeps a real service honest: an async provider client with retries and
a hard deadline, plus a CPU-bound path pushed to a worker. This is the shape you will reuse in
the shipping project.

```python
from __future__ import annotations

import asyncio
import random

import httpx


async def call_provider(
    client: httpx.AsyncClient,
    payload: dict,
    attempts: int = 3,
    base_delay: float = 0.4,
    deadline: float = 20.0,
) -> dict:
    """Retry with exponential backoff and jitter, bounded by an overall deadline."""
    async with asyncio.timeout(deadline):
        for attempt in range(1, attempts + 1):
            try:
                resp = await client.post("/v1/chat/completions", json=payload)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()
                # 5xx: retryable. 4xx would already have raised above.
                raise httpx.HTTPStatusError(
                    "server error", request=resp.request, response=resp)
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                if attempt == attempts:
                    raise
                # Jitter prevents every client retrying in lockstep.
                await asyncio.sleep(base_delay * 2 ** (attempt - 1) * (0.5 + random.random()))
    raise RuntimeError("unreachable")
```

## Real use case

An AI customer-support backend handled 40 requests per second at peak with healthy CPU
utilisation, yet p95 latency was eight seconds and support engineers could not find the slow
code. Profiling individual requests showed nothing: every handler finished in under 3 seconds in
isolation.

Event-loop lag measurement found it in a day. One endpoint generated a weekly PDF summary by
calling a synchronous PDF library directly in the handler. That library took 1.5 to 2 seconds of
pure CPU. It ran on maybe two requests a minute, so in isolation it looked harmless — but each
run froze the loop for two seconds, during which the other ~80 in-flight requests all stalled.
Averaged over the day it was invisible; during the Monday report window it was the whole p95.

They moved the PDF work to a worker process behind a job queue, returning a job id and a
polling endpoint. Loop lag dropped back to single-digit milliseconds and p95 fell to under two
seconds without touching the model path at all. The team added the lag metric to their dashboard
permanently, because the incident had taught them that a 40 req/s service can be brought down by
a job that runs twice a minute.

## Common mistakes

- **Mistake:** calling `requests.get` or `time.sleep` in an async handler — **Instead:** use `httpx.AsyncClient` and `asyncio.sleep`.
- **Mistake:** wrapping CPU-heavy work in `run_in_threadpool` — **Instead:** move it to a worker process; the GIL means threads do not help.
- **Mistake:** retrying without jitter — **Instead:** add randomised backoff so a provider blip does not become a synchronised retry storm.
- **Mistake:** retrying 4xx errors — **Instead:** retry only timeouts, connection errors and 5xx; a 400 will fail identically every time.
- **Mistake:** assuming async makes CPU work faster — **Instead:** async buys concurrency for waiting, nothing else.

## Checkpoint

1. Twenty requests each wait 1.5 seconds. What is the wall clock when awaited concurrently, and when serialised by a blocking call?
2. Add a loop-lag probe to the FastAPI app from [FastAPI and Flask](01-fastapi-flask.md) and log the maximum lag per minute. Deliberately introduce a `time.sleep(1)` in one endpoint and confirm the probe catches it.
3. Take a synchronous function that takes about 300 ms of CPU (for example, JSON-encoding a 50 MB structure) and measure it three ways: called directly, via `run_in_threadpool`, and via a `ProcessPoolExecutor`. Explain the numbers.

**Self-check:** 1.5 seconds concurrent, 30 seconds serialised — a 20x loss of concurrency from one blocking call.

## Go deeper

- [Job Queues](05-job-queues.md) — where CPU-bound and long-running work actually belongs.
- [Streaming with SSE and WebSockets](02-websockets.md) — blocking is especially visible mid-stream.
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — retries and deadlines as system properties.
- [Retries and Rate Limits](../07-llm-apis/07-retries-rate-limits.md) — provider-side retry behaviour you must respect.

---
[← Previous: Streaming with SSE and WebSockets](02-websockets.md) · [Stage index](README.md) · [Next: Redis Caching](04-redis-caching.md)
