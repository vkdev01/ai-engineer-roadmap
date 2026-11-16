# 1. Python Async

> **Stage 01 · Programming Basics** — lets your program wait on the network without wasting the CPU on the waiting.
>
> **You'll be able to:** take a script that calls twenty APIs one after another and make it finish in the time of the slowest call, with a timeout and an error path on every one.
>
> **Prerequisites:** [Stage 01 — Programming Basics](README.md) · [HTTP and REST APIs](07-http-rest-apis.md)

## Why this exists

Almost everything an AI engineer writes is waiting. Waiting for an embedding API, waiting for a model to stream a response, waiting for Postgres. In a normal Python script the call to `requests.post(...)` freezes the whole program until bytes come back. That is fine for one call and fatal for four hundred: your script spends its life idle, and a batch job that should take thirty seconds takes five minutes.

You could spawn threads, but threads bring locks, races, and a debugger that stops making sense. Async is the other answer: one thread, many paused tasks, and an explicit `await` marking every point where your code is willing to step aside.

The one-sentence mental model: **async is a scheduler for waiting, not a way to use more CPU.**

## The mental model

```text
event loop (one thread)
  |
  +-- task A: fetch(a)  --await--> [paused, waiting on socket]
  +-- task B: fetch(b)  --await--> [paused, waiting on socket]
  +-- task C: fetch(c)  --await--> [paused, waiting on socket]
  |
  (a's socket has bytes) -> resume A -> A parses, awaits next thing
```

While every task is paused on I/O, the loop has nothing to run. That is the point: you pay for latency, not for CPU.

| You have | You want | Use |
|---|---|---|
| Many network calls | Overlap the waiting | `asyncio` |
| Heavy CPU maths | Use all cores | `multiprocessing` |
| Blocking library you cannot change | Get it off the loop | `asyncio.to_thread` |

## Core concepts

### A coroutine is a function that can pause

`async def` does not run anything when called; it returns a coroutine object. The body only executes when something drives it — `await` from another coroutine, or `asyncio.run` at the top level.

```python
async def ping() -> str:
    return "pong"

coro = ping()        # nothing has run yet
print(coro)          # <coroutine object ping at 0x...>
```

### `await` is where you hand control back

`await something` means "start this, and while it is not ready, let the loop run other tasks." Only awaitable things can follow `await`. Awaiting a coroutine that finishes instantly gives you no concurrency at all — that is why `await` alone in a loop is still sequential.

### `gather` is what actually makes it concurrent

To run N coroutines at once, create them first, then hand the whole set to `asyncio.gather`. Each coroutine becomes a task on the loop, and the loop interleaves them.

```python
results = await asyncio.gather(*(fetch(u) for u in urls))
```

The subtle bug: `[await fetch(u) for u in urls]` runs them one at a time. `gather` runs them together.

### Bound your concurrency

Firing 400 requests at once gets you rate-limited or out of file descriptors. A `Semaphore` caps how many run simultaneously; the rest wait their turn. This is the single most useful pattern in production async code.

### Cancellation and timeouts are part of the API

`asyncio.timeout` (Python 3.11+) wraps a block and cancels whatever is inside when the clock runs out. Cancellation raises `CancelledError`, so cleanup code must not swallow it.

## Code

Sequential first, then the same work gathered. Run it and compare the printed elapsed time: the second version should be close to the slowest single call rather than the sum.

```python
from __future__ import annotations

import asyncio
import time

import httpx

URLS = [f"https://httpbin.org/delay/1?i={i}" for i in range(5)]


async def fetch(client: httpx.AsyncClient, url: str) -> int:
    resp = await client.get(url, timeout=10.0)
    resp.raise_for_status()
    return len(resp.content)


async def sequential() -> list[int]:
    async with httpx.AsyncClient() as client:
        # One await at a time: total time is the sum of all calls.
        return [await fetch(client, u) for u in URLS]


async def concurrent() -> list[int]:
    async with httpx.AsyncClient() as client:
        # Coroutines are created first, then all scheduled together.
        return await asyncio.gather(*(fetch(client, u) for u in URLS))


if __name__ == "__main__":
    for name, fn in (("sequential", sequential), ("concurrent", concurrent)):
        start = time.perf_counter()
        asyncio.run(fn())
        print(f"{name}: {time.perf_counter() - start:.2f}s")
```

Production version: a semaphore, a per-call timeout, and `return_exceptions=True` so one dead endpoint does not kill the batch.

```python
from __future__ import annotations

import asyncio

import httpx

LIMIT = asyncio.Semaphore(8)  # at most 8 requests in flight


async def fetch_one(client: httpx.AsyncClient, url: str) -> str | Exception:
    async with LIMIT:  # queue here rather than opening 400 sockets
        try:
            async with asyncio.timeout(15):
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
        except Exception as exc:  # noqa: BLE001 - we want the value, not a crash
            return exc


async def fetch_all(urls: list[str]) -> dict[str, str | Exception]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(fetch_one(client, u) for u in urls), return_exceptions=True
        )
    return dict(zip(urls, results))
```

A TypeScript caller of the same idea, since Node has the same shape (`Promise.all` instead of `gather`, `AbortSignal` instead of `asyncio.timeout`):

```ts
export async function fetchAll(urls: string[], limit = 8): Promise<Array<string | Error>> {
  const queue = [...urls];
  const worker = async (): Promise<Array<string | Error>> => {
    const out: Array<string | Error> = [];
    for (let url = queue.shift(); url !== undefined; url = queue.shift()) {
      try {
        const res = await fetch(url, { signal: AbortSignal.timeout(15_000) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        out.push(await res.text());
      } catch (err) {
        out.push(err instanceof Error ? err : new Error(String(err)));
      }
    }
    return out;
  };
  const chunks = await Promise.all(Array.from({ length: limit }, worker));
  return chunks.flat();
}
```

## Real use case

A support team wants every inbound ticket tagged with a product area before it reaches an agent. The tagging step calls a small LLM once per ticket. With a sequential script, a 400-ticket nightly backfill took long enough that the team ran it during lunch and hoped nobody wrote in; one slow provider response stalled the entire queue behind it.

The rewrite kept the same prompt but changed three things: tickets became tasks on one loop, a semaphore of 8 kept the provider inside its rate limit, and each task returned either a tag or an exception instead of raising. The nightly job now reports "392 tagged, 8 failed, 8 reasons listed" and the failures are retried by ticket id rather than by re-running everything.

The design choice that made it work is `return_exceptions=True` plus per-item results: partial success became a first-class outcome, which is what batch AI work actually is.

## Common mistakes

- **Mistake:** `await`ing inside a list comprehension and calling it concurrency. — **Instead:** build the coroutines first and pass them to `asyncio.gather`.
- **Mistake:** calling a blocking library (`requests`, `time.sleep`, heavy pandas) inside a coroutine. — **Instead:** it freezes the loop for every task; use `asyncio.to_thread` or an async client.
- **Mistake:** no cap on in-flight requests. — **Instead:** a `Semaphore` sized to the provider's rate limit.
- **Mistake:** catching `CancelledError` with a bare `except Exception` and continuing. — **Instead:** let cancellation propagate; clean up in `finally` and re-raise.
- **Mistake:** assuming async makes CPU work faster. — **Instead:** measure; if the profiler shows CPU, async will not help.

## Checkpoint

1. Wrap the sequential example in `asyncio.timeout(2)` and predict what happens. Then run it.
2. Change `concurrent()` to use a semaphore of 2 and explain the new elapsed time in one sentence.
3. Write `fetch_all` so it retries a failed URL twice with a short exponential backoff, and prove with a deliberately broken URL that only that item is retried.

**Self-check:** the timeout cancels the whole block, so instead of a slow success you get a `TimeoutError` raised from the `async with` — the sequential run never reaches the last URL.

## Go deeper

- [Generators](03-generators.md) — the other half of "process data without holding it all in memory".
- [HTTP and REST APIs](07-http-rest-apis.md) — status codes, headers and pagination, which your async client has to handle.
- [Async Processing](../17-ai-backend/03-async-processing.md) — the same ideas behind a web server, where the loop is shared by every user.

---
[← Previous: Stage 01 — Programming Basics](README.md) · [Stage index](README.md) · [Next: Type Hints and Pydantic v2](02-type-hints-pydantic.md)
