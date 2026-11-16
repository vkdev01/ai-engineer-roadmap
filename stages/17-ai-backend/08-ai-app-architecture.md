# 8. AI App Architecture

> **Stage 17 · AI Backend** — the five layers every production AI application has, and what each one owns.
>
> **You'll be able to:** draw and defend the reference architecture for an AI app, and place any feature in the right layer.
>
> **Prerequisites:** [API Security](07-api-security.md) · [RAG Architecture](../10-rag/01-rag-architecture.md)

## Why this exists

The first version of most AI apps is one file: receive request, call model, return text. It works
until you need a second client, a second model, a cache, a quota, a retry policy and an audit log.
Then every feature gets added in the same function, and changing the model means changing
everything.

The architecture that survives is boring and layered. Each layer has one job, talks only to its
neighbours, and can be replaced without touching the others. You will meet it in every serious AI
product, whatever the marketing calls it.

The mental model: **client → gateway → orchestrator → model and tools → store, with observability
cutting across all five.**

## The mental model

```text
+---------+   +-----------+   +--------------+   +-----------+   +--------+
| client  |-->| gateway   |-->| orchestrator |-->| models    |   | store  |
| web/app |   | auth      |   | prompt build |   | provider  |<->| cache  |
| mobile  |<--| quotas    |<--| tool loop    |<--| self-host |   | vector |
+---------+   | routing   |   | budget       |   | router    |   | db, S3 |
              | logging   |   +--------------+   +-----------+   +--------+
                    |                 |                 |             |
                    +-----------------+-----------------+-------------+
                              traces, metrics, cost, evals
```

| Layer | Owns | Must not own |
|---|---|---|
| Client | rendering, streaming UI, local state | provider keys, prompt logic |
| Gateway | identity, quotas, rate limits, routing, request logging | prompt construction, business rules |
| Orchestrator | prompt assembly, tool loop, retries, deadlines, fallbacks | auth, transport details |
| Model layer | one interface over many providers and engines | product logic |
| Store | cache, vectors, documents, job state | anything latency-critical that can be cached above |

## Core concepts

### The gateway is a policy layer, not a proxy

The gateway decides *whether* and *how much*, never *what*. It authenticates, checks quota,
applies rate limits, attaches a trace id, and picks a route. Because it is the only place
requests enter, one implementation covers every endpoint — which is the argument for putting
limits here rather than in handlers.

### The orchestrator is where the product lives

Everything specific to your product belongs here: which prompt version, which retrieved context,
which tools are available, what to do when the model asks for a tool, when to give up. Keep it
free of HTTP concerns so it can be unit tested with fake models and run in a worker as easily as in
a request.

### One model interface, many implementations

Wrap every provider and engine behind a small interface: `complete(messages, **params)` and
`stream(messages, **params)`. That is what lets you fail over from one provider to another, route
cheap requests to a small model, or swap a hosted model for a vLLM server you run yourself
([vLLM and SGLang](../16-inference/08-vllm-sglang.md)) without touching product code.

### A latency budget makes trade-offs concrete

Give every request a total budget and allocate it across layers. For a 3-second target:

```text
gateway (auth, quota, rate limit)        10 ms
cache lookup                              5 ms
retrieval + rerank                      250 ms
model prefill (first token)             600 ms
generation (300 tokens at 6 ms)        1800 ms
serialisation and network                50 ms
                                    -----------
total                                 2715 ms
```

The budget tells you what you can afford. If retrieval grows to 800 ms, you have a conversation
about token counts, not a vague feeling that things are slow. It also tells you when to stream:
2.7 seconds of total work with 600 ms to first token is perfectly acceptable if the user sees
progress at 600 ms.

### Every layer needs a degraded mode

What happens when the model provider is down, the vector store is slow, or the cache is
unreachable? Decide in advance:

| Failure | Degraded behaviour |
|---|---|
| Provider error or timeout | serve cache if available, else a clear apology with a retry hint |
| Provider slow but working | reduce `max_tokens`, switch to a faster model |
| Vector store down | answer without retrieval, and say so |
| Redis down | skip cache and rate limiting to a local fallback, never fail the request |
| Queue down | reject the long job with 503 and a retry-after, keep chat working |

A service that fails one feature is fine. A service that fails all of them because a cache is down
is a design error.

## Code

An orchestrator with a deadline, a fallback chain and an explicit degraded path. Note that it
knows nothing about HTTP — it takes a request object and returns a stream of events, so the same
code can be called from a route or a worker.

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum


class Route(StrEnum):
    FAST = "fast"
    STRONG = "strong"


@dataclass(frozen=True)
class ChatIn:
    tenant: str
    messages: list[dict]
    route: Route = Route.STRONG
    max_tokens: int = 800


@dataclass(frozen=True)
class ChatEvent:
    kind: str          # "token" | "degraded" | "done"
    text: str = ""


class Orchestrator:
    def __init__(self, models: dict[str, object], cache, retriever) -> None:
        self.models = models          # name -> client implementing complete/stream
        self.cache = cache
        self.retriever = retriever

    def _chain(self, route: Route) -> list[str]:
        # Order matters: cheap and available first for FAST, quality first for STRONG.
        return ["strong", "fast"] if route is Route.STRONG else ["fast", "strong"]

    async def run(self, req: ChatIn, budget_s: float = 25.0) -> AsyncIterator[ChatEvent]:
        try:
            async with asyncio.timeout(budget_s):
                if (hit := await self.cache.lookup(req)) is not None:
                    yield ChatEvent("token", hit.answer)
                    yield ChatEvent("done")
                    return

                context = await self.retriever.search(req.messages[-1]["content"], k=6)
                messages = self._build(req.messages, context)

                last_error: Exception | None = None
                for name in self._chain(req.route):
                    collected: list[str] = []
                    try:
                        async for delta in self.models[name].stream(
                            messages, max_tokens=req.max_tokens
                        ):
                            collected.append(delta)
                            yield ChatEvent("token", delta)
                        if collected:
                            await self.cache.store(req, "".join(collected))
                        yield ChatEvent("done")
                        return
                    except (TimeoutError, ConnectionError) as exc:
                        last_error = exc        # try the next model in the chain
                        continue

                # All models failed: degrade rather than 500.
                yield ChatEvent("degraded", "The model is unavailable right now.")
                yield ChatEvent("done")
        except TimeoutError:
            yield ChatEvent("degraded", "This answer took too long. Try a shorter question.")
            yield ChatEvent("done")

    def _build(self, messages: list[dict], context: list[str]) -> list[dict]:
        system = "Answer from the context. If it is missing, say so."
        if context:
            system += "\n\nContext:\n" + "\n\n".join(context)
        return [{"role": "system", "content": system}, *messages]
```

The gateway side, in TypeScript, since gateways are often a thin Node service in front of
Python workers. This middleware stamps a trace id, enforces a body-size cap and forwards the
principal downstream.

```ts
import { randomUUID } from "node:crypto";
import type { NextFunction, Request, Response } from "express";

const MAX_BODY_BYTES = 256 * 1024;

export function gateway(upstreamUrl: string) {
  return async function handler(req: Request, res: Response, next: NextFunction): Promise<void> {
    const traceId = req.header("x-trace-id") ?? randomUUID();
    const contentLength = Number(req.header("content-length") ?? "0");
    if (contentLength > MAX_BODY_BYTES) {
      res.status(413).json({ detail: "payload too large", trace_id: traceId });
      return;
    }
    const apiKey = req.header("x-api-key");
    if (apiKey === undefined) {
      res.status(401).json({ detail: "missing api key", trace_id: traceId });
      return;
    }

    try {
      const upstream = await fetch(`${upstreamUrl}${req.path}`, {
        method: req.method,
        headers: {
          "content-type": "application/json",
          "x-api-key": apiKey,
          "x-trace-id": traceId,
        },
        body: req.method === "GET" ? undefined : JSON.stringify(req.body),
        signal: AbortSignal.timeout(30_000),      // outer deadline, longer than inner ones
      });
      res.status(upstream.status);
      res.setHeader("x-trace-id", traceId);
      if (upstream.body === null) { res.end(); return; }
      // Pipe the stream straight through: no buffering, or SSE breaks.
      for await (const chunk of upstream.body as unknown as AsyncIterable<Uint8Array>) {
        res.write(chunk);
      }
      res.end();
    } catch (err) {
      next(err);
    }
  };
}
```

## Real use case

A 30-person company built an internal "ask anything" assistant over its Notion, Jira and Drive
content. The first version was a single FastAPI file with the provider call inline, and it grew for
six months: a Slack integration, a web app, a nightly digest job, a per-team budget, a cache. By the
end, changing the model required edits in eleven places, and nobody could say what a request
actually cost.

They refactored to the five layers, not by rewriting but by extraction. The provider calls moved
behind one interface, which immediately allowed a cheap model for the Slack quick-answer path and
the strong model for the web app. Prompt assembly and retrieval moved into an orchestrator, which
made it unit-testable with a fake model — their eval suite from
[RAG Evals](../10-rag/12-rag-evals.md) started running in seconds instead of minutes. Quotas and
rate limits moved into the gateway, which is where they finally applied to every entry point at
once rather than to whichever handlers someone had remembered.

The measurable outcome they cared about was incident behaviour. Previously, a provider outage
produced 500s across all surfaces and a support queue. After the refactor, the orchestrator's
fallback chain and degraded mode meant Slack got a short cached answer, the web app showed a clear
message with a retry link, and the nightly digest queued and ran later. Same outage, no incident.

## Common mistakes

- **Mistake:** putting prompt logic in the HTTP handler — **Instead:** move it to an orchestrator you can test without a server.
- **Mistake:** calling providers directly from several places — **Instead:** one model interface, one place to change.
- **Mistake:** implementing quotas per endpoint — **Instead:** one gateway policy layer for all traffic.
- **Mistake:** designing only the happy path — **Instead:** write the degraded behaviour for each dependency before launch.
- **Mistake:** budgeting latency by feel — **Instead:** allocate milliseconds per layer and check the sum against your target.

## Checkpoint

1. Draw the five layers for a feature you have built, and mark which layer each existing function belongs in. Note anything that appears in two layers.
2. Implement the orchestrator's fallback chain with two fake model clients, one that always times out. Verify the stream still completes with a degraded event.
3. Add a per-layer timing histogram to the orchestrator and the gateway, run a load test, and compare the measured p95 breakdown against the latency budget you wrote down first.

**Self-check:** with two models in the chain and a 25-second budget, the first attempt may consume the whole budget — so give each attempt its own shorter timeout, for example 15 seconds, or the fallback never gets a chance to run.

## Go deeper

- [Ship: A Streaming Chat API](09-ship-streaming-api.md) — this architecture as a running service.
- [Scalable AI Apps](../21-system-design/01-scalable-ai-apps.md) — the same layers under real load.
- [Multi-Tenant Design](../21-system-design/05-multi-tenant.md) — tenancy through all five layers.
- [Tracing](../18-mlops-llmops/05-tracing.md) — the cross-cutting observability this architecture depends on.

---
[← Previous: API Security](07-api-security.md) · [Stage index](README.md) · [Next: Ship: A Streaming Chat API](09-ship-streaming-api.md)
