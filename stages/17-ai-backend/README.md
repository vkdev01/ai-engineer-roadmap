# Stage 17 — AI Backend

An LLM call is one HTTP request. A product is a backend: streaming responses that survive a client disconnect, caches that cut cost, queues that absorb long jobs, auth and quotas that stop one user spending your budget, and timeouts everywhere so a slow provider does not become a slow product.

This stage is the one most AI tutorials skip, and it is the one that decides whether anything you built earlier survives contact with users.

**Time:** 2-3 weeks · **You need first:** [Stage 07 — LLM APIs](../07-llm-apis/README.md) · **You leave with:** a streaming chat API with auth, quotas, caching, background jobs and graceful degradation

## What you will be able to do

- Serve streaming responses over SSE and WebSockets, and handle disconnects correctly
- Keep the event loop unblocked: async I/O, thread pools for CPU work, no sync SDK calls in async handlers
- Add exact and semantic caching with honest invalidation
- Move long work into a job queue with retries, idempotency and dead-letter handling
- Enforce auth, per-tenant quotas and rate limits at the edge
- Harden the API: input validation, secrets, timeouts, egress control
- Draw the reference architecture of an AI app and place every component on it
- Degrade gracefully when the model provider is slow or down

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [FastAPI and Flask](01-fastapi-flask.md) | The request lifecycle, and where Flask still fits |
| 2 | [WebSockets](02-websockets.md) | Bidirectional streaming and connection state |
| 3 | [Async Processing](03-async-processing.md) | Not blocking the loop, and where CPU work goes |
| 4 | [Redis Caching](04-redis-caching.md) | Exact and semantic caches, and invalidation |
| 5 | [Job Queues](05-job-queues.md) | Retries, idempotency, dead letters |
| 6 | [Auth and Rate Limits](06-auth-rate-limits.md) | Keys, JWTs, per-tenant quotas |
| 7 | [API Security](07-api-security.md) | Validation, secrets, timeouts, egress |
| 8 | [AI App Architecture](08-ai-app-architecture.md) | The reference shape of the whole system |
| 9 | [Ship: A Streaming Chat API](09-ship-streaming-api.md) | All of it, running, with degradation |

## The shipping project

[Ship: A Streaming Chat API](09-ship-streaming-api.md) — a FastAPI service that streams tokens, authenticates requests, enforces per-tenant quotas, caches repeated questions, pushes long jobs to a worker, and returns a useful degraded response when the provider times out. You load-test it and report p95 latency and cost per request.

## How to know you are done

- A client that disconnects mid-stream does not leave a model call running
- Your handler contains no blocking call on the event loop, and you can prove it
- Repeated identical questions hit the cache, and you can state your hit rate
- A long job survives a worker restart without duplicating its side effect
- One tenant cannot exhaust another tenant's quota
- You can describe your degraded mode and you have triggered it deliberately

---
[← Back to roadmap](../../README.md) · [Next stage: MLOps and LLMOps](../18-mlops-llmops/README.md)
