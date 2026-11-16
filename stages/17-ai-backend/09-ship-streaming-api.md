# 9. Ship: A Streaming Chat API

> **Stage 17 · AI Backend** — one service that streams tokens, authenticates, meters, caches, queues and degrades, with a load test that reports p95 latency and cost per request.
>
> **You'll be able to:** ship a production-shaped LLM API and prove its behaviour under load, failure and abuse.
>
> **Prerequisites:** [AI App Architecture](08-ai-app-architecture.md) · [Job Queues](05-job-queues.md) · [Auth and Rate Limits](06-auth-rate-limits.md)

## Why this exists

You have written the pieces in this stage separately. The skill that gets hired is putting them in the right order: where the quota check goes relative to the cache, what happens when the provider dies mid-stream, and which parts of the request must be persisted before the response starts.

This project is the integration test for everything above it. It is also the artefact you demo, because it behaves like a product rather than a notebook.

The mental model: **the request path is a chain of cheap gates, and the expensive call happens last, exactly once.**

## The mental model

```text
 client
   │  POST /chat  (Authorization: Bearer ...)
   ▼
 ┌──────────────┐  reject 401/429 early, before spending anything
 │ auth + quota │─────────────────────────────► error response
 └──────┬───────┘
        ▼
 ┌──────────────┐  hit → stream cached answer, cost 0
 │ cache lookup │─────────────────────────────► stream
 └──────┬───────┘
        ▼
 ┌──────────────┐  write audit row BEFORE the call (you must know what you spent)
 │ persist audit│
 └──────┬───────┘
        ▼
 ┌──────────────┐  provider timeout → degraded answer, not a hang
 │ model call   │──► SSE stream ──► client (disconnect cancels the call)
 └──────────────┘
        │ long work? → enqueue job, return job id
        ▼
    worker (retries, idempotency key)
```

## Core concepts

### Gates before spend

Order matters more than any single component. Auth and quota are cheap; the model call is expensive. Anything that can reject a request must run before anything that costs money. This sounds obvious and is violated constantly — usually because the quota check was added after the endpoint existed.

```python
async def guard(request: Request, tenant: Tenant, cost_estimate: int) -> None:
    if not tenant.is_active:
        raise HTTPException(403, "tenant inactive")
    if await usage_today(tenant.id) + cost_estimate > tenant.daily_tokens:
        raise HTTPException(429, "daily token quota exceeded")
    if not await limiter.allow(tenant.id):        # token bucket per tenant
        raise HTTPException(429, "rate limited", headers={"Retry-After": "1"})
```

### Streaming with cancellation

A stream is a generator that lives as long as the client's connection. If you do not propagate the disconnect, the model keeps generating and you keep paying for tokens nobody will read.

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/chat")
async def chat(request: Request, body: ChatIn, tenant: Tenant = Depends(auth)):
    await guard(request, tenant, cost_estimate=body.max_tokens)

    async def gen():
        try:
            async for delta in provider.stream(body.messages, max_tokens=body.max_tokens):
                if await request.is_disconnected():     # stop paying for nobody
                    break
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except ProviderTimeout:
            # headers are already sent: degrade in-band, never hang
            yield f"data: {json.dumps({'error': 'degraded', 'partial': True})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

The `X-Accel-Buffering: no` header is not decoration: behind nginx, without it, your stream arrives in one lump after completion and you have built a very expensive non-streaming API.

### Caching that is honest about what it caches

Exact-match caching is safe and boring. Semantic caching saves more and can be wrong — two questions can be similar and need different answers ("refund for order 123" versus "refund for order 456"). If you use it, store the threshold you chose, and exclude anything with an identifier or a date in it.

```python
async def cached_answer(question: str, tenant_id: str) -> str | None:
    key = f"q:{tenant_id}:{sha256(question.strip().lower().encode()).hexdigest()}"
    return await redis.get(key)


async def store_answer(question: str, tenant_id: str, answer: str, ttl: int = 3600) -> None:
    if re.search(r"\b\d{3,}\b", question):      # identifiers make answers user-specific
        return
    key = f"q:{tenant_id}:{sha256(question.strip().lower().encode()).hexdigest()}"
    await redis.set(key, answer, ex=ttl)
```

### Long work leaves the request path

Anything over a few seconds — batch document processing, an agent run with many steps, a fine-tuning job submission — belongs in a queue with an idempotency key, so a retry cannot double-charge or double-write.

```python
@app.post("/jobs", status_code=202)
async def submit(body: JobIn, tenant: Tenant = Depends(auth)) -> dict:
    job_id = await queue.enqueue(
        "run_agent",
        args={"question": body.question, "tenant_id": tenant.id},
        idempotency_key=body.idempotency_key or str(uuid4()),
    )
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}
```

## Code

The load test is part of the deliverable, because "it streams" is not a measurement.

```python
# loadtest.py — measures p95 latency, TTFT and cost per request
from __future__ import annotations

import asyncio
import statistics
import time

import httpx

URL = "http://localhost:8000/chat"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}      # from env, never inline in a repo
QUESTIONS = ["what is the refund window", "how do I reset my password",
             "where is my invoice", "can I change my plan mid-cycle"]


async def one(client: httpx.AsyncClient, q: str) -> dict:
    t0 = time.perf_counter()
    ttft = None
    async with client.stream("POST", URL, json={"messages": [{"role": "user", "content": q}]},
                             headers=HEADERS, timeout=30) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and ttft is None:
                ttft = time.perf_counter() - t0
    return {"ttft": ttft or 0.0, "total": time.perf_counter() - t0, "status": r.status_code}


async def main(n: int = 50, concurrency: int = 10) -> None:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded(q: str) -> dict:
            async with sem:
                return await one(client, q)

        results = await asyncio.gather(*[bounded(QUESTIONS[i % len(QUESTIONS)]) for i in range(n)])

    ok = [r for r in results if r["status"] == 200]
    print(f"n={n} ok={len(ok)} concurrency={concurrency}")
    print(f"ttft p50={statistics.median(r['ttft'] for r in ok):.2f}s "
          f"p95={sorted(r['ttft'] for r in ok)[int(len(ok) * 0.95) - 1]:.2f}s")
    print(f"total p95={sorted(r['total'] for r in ok)[int(len(ok) * 0.95) - 1]:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

TypeScript client that handles in-band errors and disconnects properly:

```ts
export async function* chatStream(messages: { role: string; content: string }[]) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify({ messages }),
    signal: AbortSignal.timeout(45_000),
  });
  if (!res.ok || !res.body) throw new Error(`chat failed: ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    const chunks = buf.split("\n\n");
    buf = chunks.pop() ?? "";
    for (const c of chunks) {
      if (!c.startsWith("data: ")) continue;
      const payload = c.slice(6);
      if (payload === "[DONE]") return;
      const evt = JSON.parse(payload) as { delta?: string; error?: string };
      if (evt.error) throw new Error(evt.error);       // surface degradation to the UI
      if (evt.delta) yield evt.delta;
    }
  }
}
```

### What to put in the README

| Metric | Value | How measured |
|---|---|---|
| TTFT p95 | measured | `loadtest.py`, 50 requests, concurrency 10 |
| Total p95 | measured | same |
| Cache hit rate | measured | Redis keyspace hits / requests, over an hour |
| Cost per 1k requests | computed | mean tokens × your provider's current price |
| Degraded-mode rate | measured | force provider timeout, count in-band errors |
| Quota enforcement | pass/fail | one tenant at limit cannot proceed; another is unaffected |

## Real use case

A B2B SaaS adds an AI assistant to its dashboard. The first version streams correctly and falls over in a week: one power user's scripted loop consumes the month's token budget, and a provider slowdown turns into hundreds of hung requests because nothing had a timeout.

The rewrite is this project. Quota checks move in front of the cache, a per-tenant token bucket caps burst rate, every provider call gets an explicit timeout with an in-band degraded message, and the long-running "analyse all my documents" feature moves to a queue with an idempotency key. Cost per request falls measurably — most of it from exact-match caching of the same twenty questions asked by every new user — and the incident class disappears.

## Common mistakes

- **Mistake:** checking quota after the model call — **Instead:** gate before spend; the cheapest request is the one you reject.
- **Mistake:** no `request.is_disconnected()` check in the stream — **Instead:** break out and let the provider call cancel; otherwise you pay for abandoned tokens.
- **Mistake:** buffering the stream behind a proxy — **Instead:** disable proxy buffering explicitly and test through the real deployment, not just locally.
- **Mistake:** semantic cache keyed without tenant isolation — **Instead:** include tenant in every cache key; cross-tenant cache hits are a data leak.
- **Mistake:** retrying a queue job with side effects and no idempotency key — **Instead:** key the job and make the handler check whether the effect already happened.
- **Mistake:** reporting average latency only — **Instead:** p95 or p99; users experience the tail, and the tail is where streaming saves you.

## Checkpoint

1. Stand up the service and run the load test at concurrency 10. Record TTFT p95 and total p95.
2. Add a deliberate 5-second provider delay and confirm clients receive the degraded event instead of hanging past your timeout.
3. Set a tenant's daily quota to a small number, exceed it, and confirm a second tenant is unaffected. Write both assertions into your test suite.

**Self-check:** if your cache hit rate is near zero on a real workload, check the key — trailing whitespace and case differences defeat exact-match caching, and hashing the raw question is usually the bug.

## Go deeper

- [AI App Architecture](08-ai-app-architecture.md) — the reference shape this implements
- [Cost and Latency](../21-system-design/08-cost-latency.md) — the system-design view of these budgets
- [Ship: A Docker Deploy](../19-cloud-devops/09-ship-docker-deploy.md) — getting this into production

---
[← Previous: AI App Architecture](08-ai-app-architecture.md) · [Stage index](README.md) · [Next stage: MLOps and LLMOps](../18-mlops-llmops/README.md)
