# 1. FastAPI and Flask

> **Stage 17 · AI Backend** — the HTTP layer your model sits behind, and the one decision that matters about it.
>
> **You'll be able to:** build a typed async endpoint around a model call and justify your framework choice.
>
> **Prerequisites:** [HTTP and REST APIs](../01-programming-basics/07-http-rest-apis.md) · [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md)

## Why this exists

An AI backend is mostly waiting. A request arrives, you spend 50 milliseconds on your own
logic, then 2 to 20 seconds waiting for a model provider. If your server handles one request
per worker while waiting, you pay for a worker per concurrent user, and most of that money is
spent idle.

FastAPI is built for that shape: it is async-native, it validates requests with the same
Pydantic models you already use, and it generates an OpenAPI schema so clients are not guessing.
Flask is the older, simpler framework: synchronous by default, with an enormous ecosystem and
almost no learning curve.

The mental model: **FastAPI is a concurrency story, Flask is a simplicity story.** Neither is
wrong; the difference is how many users you can hold while waiting on someone else's GPU.

## The mental model

```text
              FastAPI (ASGI)                    Flask (WSGI)
        +---------------------+            +---------------------+
        | one process, one    |            | one thread per      |
        | event loop, many    |            | request, blocks on   |
        | in-flight awaits    |            | every wait          |
        +---------------------+            +---------------------+
                  |                                  |
        await provider(...)                provider(...)   <-- thread parked
        await redis.get(...)               redis.get(...)  <-- thread parked
        await db.fetch(...)                db.fetch(...)   <-- thread parked
```

| | FastAPI | Flask |
|---|---|---|
| Concurrency model | async/await on ASGI | sync threads on WSGI |
| Validation | Pydantic, built in | manual, or an extension |
| Streaming | `StreamingResponse`, native | `Response(generator)`, works |
| WebSockets | built in | needs an extension |
| Docs | OpenAPI generated | manual |
| Best fit | I/O-heavy AI services | small tools, existing sync codebases |

## Core concepts

### Waiting is the thing you are optimising

If a provider call takes 2 seconds and you have 4 worker processes, your ceiling is 2 requests
per second, no matter how fast your own code is. That is `4 / 2 s = 2 req/s`. On one async
process with no CPU work in the handler, the same 2-second wait overlaps across hundreds of
requests, and your ceiling becomes whatever the provider will give you.

The catch is that async only helps if *everything* on the path is async. One blocking call in a
handler stops the whole loop, and you are back to one request at a time — worse than Flask,
because at least Flask would have had other threads. See
[Async Processing](03-async-processing.md).

### Validate at the edge with Pydantic

Every field a client sends is an attack surface and a source of 500s. A Pydantic model makes
the contract explicit: types are checked before your code runs, unknown fields can be rejected,
and constraints like maximum message length are declared once. It also documents the API for
free.

### Configure once at startup, not per request

Provider clients, connection pools and Redis handles should be created in a lifespan handler
and shared. Creating an HTTP client per request means a new TLS handshake per request, which on
a 20-second model call is a rounding error but on a cached 5-millisecond response is most of
your latency.

### Where Flask still wins

Flask is the right choice when the service is small and synchronous anyway: an internal tool
that runs a nightly batch, a webhook receiver that writes to a database, a small admin page, or
an existing codebase where the model call is one function among hundreds. If your work is
mostly CPU-bound Python and your concurrency needs are low, async buys you nothing and costs you
a debugging model.

## Code

A minimal FastAPI service with a shared client, a typed request model, and an async provider
call with a timeout. The model id and the SDK call shape must be checked against your provider's
current docs.

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    model_config = {"extra": "forbid"}          # reject unknown fields loudly


class ChatResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One client for the whole process: reuses connections and the TLS session.
    app.state.llm = AsyncOpenAI(timeout=20.0, max_retries=2)
    yield
    await app.state.llm.close()


app = FastAPI(lifespan=lifespan, title="chat-api")


def get_llm() -> AsyncOpenAI:
    return app.state.llm


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    llm: Annotated[AsyncOpenAI, Depends(get_llm)],
) -> ChatResponse:
    try:
        # asyncio.timeout bounds the whole call, including retries inside the SDK.
        async with asyncio.timeout(25.0):
            result = await llm.chat.completions.create(
                model="<current-model-id>",
                messages=[{"role": "user", "content": body.message}],
                max_tokens=body.max_tokens,
                temperature=0.2,
            )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="model timed out") from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="model provider error") from exc

    usage = result.usage
    return ChatResponse(
        text=result.choices[0].message.content or "",
        model=result.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
```

Run it with an ASGI server such as `uvicorn app:app --workers 2` and post a request with `curl`.
You should get a typed JSON response, a 422 for a message over 8,000 characters, and a 504
instead of a hung connection when the provider is slow.

The equivalent Flask handler is shorter but blocks its thread for the whole model call. Note the
explicit timeout: without one, a stuck provider parks that thread indefinitely.

```python
from __future__ import annotations

import httpx
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.post("/v1/chat")
def chat():
    body = request.get_json(force=True)
    message = (body or {}).get("message", "")
    if not isinstance(message, str) or not 1 <= len(message) <= 8000:
        return jsonify({"detail": "message must be 1-8000 chars"}), 422
    try:
        # Synchronous client: this thread is parked for the whole call.
        resp = httpx.post(
            "https://api.example-provider.com/v1/chat/completions",
            json={"model": "<current-model-id>",
                  "messages": [{"role": "user", "content": message}]},
            timeout=25.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        return jsonify({"detail": "model provider error"}), 502
    return jsonify({"text": resp.json()["choices"][0]["message"]["content"]})
```

## Real use case

A team built an internal "ask the handbook" tool on Flask because it was the framework they
knew. It worked for a year with a few dozen users. Then the company rolled it out to 900
employees, and latency collapsed in a way that looked like a model problem but was not: the
provider's own status page was green, and single requests from a laptop were fast.

The arithmetic explained it. Their deployment was four sync workers behind gunicorn. Each
request held a worker for the full retrieval plus generation cycle, about 4 seconds at their
median. That is `4 / 4 s = 1 request per second` of capacity. At 900 employees, a Monday
morning burst of 60 simultaneous questions meant the last person waited a minute before their
request even started.

They moved the chat path to FastAPI, kept the client shared, and made retrieval and the model
call properly async. The same four-core box then held the burst comfortably, because waiting
overlapped instead of occupying a worker. They deliberately left the admin pages on Flask: they
are low traffic, synchronous, and rewriting them would have been motion without benefit.

## Common mistakes

- **Mistake:** choosing async and then calling a synchronous SDK inside the handler — **Instead:** use the async client, or `run_in_threadpool`, as [Async Processing](03-async-processing.md) shows.
- **Mistake:** creating a new HTTP client or Redis connection per request — **Instead:** build them in a lifespan handler and inject them.
- **Mistake:** letting Pydantic models accept any extra field — **Instead:** set `extra="forbid"` so client mistakes surface as 422 rather than silent behaviour.
- **Mistake:** returning raw provider exceptions to clients — **Instead:** map them to 502 or 504 and log the detail server-side only.
- **Mistake:** rewriting a working internal Flask tool to "modernise" it — **Instead:** migrate the hot path that is actually blocked; leave the rest.

## Checkpoint

1. Your provider call takes 3 seconds and you run 8 sync workers. What is your maximum request rate, and how many workers would you need for 20 requests per second?
2. Add a `/v1/chat/batch` endpoint that accepts up to 10 messages and runs the provider calls with `asyncio.gather`. Confirm the wall-clock time for 10 calls is close to one call, not ten.
3. Add Pydantic validation that rejects any message containing a `{{` template marker, and write a test that proves a bad payload returns 422 without calling the provider.

**Self-check:** `8 / 3 s ≈ 2.7` requests per second; for 20 per second you would need 60 workers, which is the arithmetic argument for async.

## Go deeper

- [Streaming with SSE and WebSockets](02-websockets.md) — the same endpoint, but sending tokens as they arrive.
- [Async Processing](03-async-processing.md) — the blocking trap in detail.
- [API Security](07-api-security.md) — validation is the first layer, not the only one.
- [FastAPI and Pydantic habits](../01-programming-basics/02-type-hints-pydantic.md) — the models this file relies on.

---
[← Previous: Stage 16 — Inference](../16-inference/README.md) · [Stage index](README.md) · [Next: Streaming with SSE and WebSockets](02-websockets.md)
