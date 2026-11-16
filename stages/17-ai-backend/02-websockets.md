# 2. Streaming with SSE and WebSockets

> **Stage 17 · AI Backend** — getting tokens to the browser as they are generated instead of after.
>
> **You'll be able to:** stream a model response over SSE and over WebSockets, and choose correctly between them.
>
> **Prerequisites:** [FastAPI and Flask](01-fastapi-flask.md) · [Streaming](../07-llm-apis/03-streaming.md)

## Why this exists

A 400-token answer takes several seconds to generate. Delivered as one JSON body, the user sees
nothing for those seconds and then everything at once. Delivered as a stream, they see the first
words in a few hundred milliseconds and read while it generates. The total time is the same; the
perceived speed is completely different.

There are two mainstream ways to push data from server to browser. **Server-Sent Events (SSE)**
is a one-way text stream over plain HTTP. **WebSockets** is a full-duplex connection that stays
open both ways. Both work for chat; they differ in what you can send back up.

The mental model: **SSE is a long HTTP response that never quite finishes; a WebSocket is a
socket you both talk on.**

## The mental model

```text
SSE (one way)                            WebSocket (two way)
client ---- POST /chat --------->        client <--- upgrade ---> server
client <=== data: tok ======                            <===> tokens both ways
client <=== data: tok ======                            <===> user interrupts
client <=== data: [DONE] ====                            <===> client sends more context
```

| | SSE | WebSocket |
|---|---|---|
| Direction | server to client | both |
| Transport | HTTP/1.1 or HTTP/2 response | upgraded TCP connection |
| Client API | `EventSource` (or `fetch` reader) | `WebSocket` |
| Auto-reconnect | built into `EventSource` | you implement it |
| Proxies and CDNs | works like any HTTP response | needs upgrade support end to end |
| Auth | normal headers, cookies | headers on the handshake only |
| Best for | chat completions, progress, notifications | voice, collaborative editing, interruptible agents |

## Core concepts

### The SSE wire format is trivial, which is the point

An SSE response is a `text/event-stream` body of records separated by a blank line. Each record
is one or more `field: value` lines. For token streaming you need only `data:`:

```text
data: {"delta": "The"}

data: {"delta": " contract"}

data: [DONE]

```

Two details cause most bugs. First, the blank line between records is mandatory — without it the
browser buffers forever. Second, any proxy that buffers responses will hold your tokens until
the end; you must send a buffering-control header and, ideally, a comment line (`: ping`) early
to force the headers out.

### Framing overhead is small but real

Each token costs you the framing bytes: roughly 20 to 30 bytes for `data: {"delta":"..."}` plus
the blank line. If your average token is 4 characters, the payload is about 5 bytes and the
overhead dominates the wire size — but at 50 tokens per second that is still around 1.5 KB/s,
which is nothing. Do not batch tokens to "save bandwidth"; batch only if your client renders
jankily at very high rates.

### Cancellation is the feature people forget

When a user closes the tab or presses stop, you want to stop paying the provider immediately.
With SSE, the server sees the request disconnect and the generator is cancelled. Catch that
cancellation and close the upstream stream, or you keep generating tokens nobody will read.

### WebSockets earn their complexity when you need to talk back

A voice assistant needs to send audio up and tokens down at the same time. An agent UI needs to
let the user interrupt mid-generation or answer a clarification question. Those are
bidirectional by nature, and SSE would force a second HTTP request per message. If your
interaction is one question, one answer, SSE is simpler and cheaper to operate.

### Auth differs in a way that surprises people

`EventSource` cannot set custom headers, so an SSE endpoint usually authenticates by cookie or a
short-lived query token. A WebSocket authenticates on the handshake, and the connection then
stays authorised until it closes — so a revoked key does not take effect until you disconnect
the socket. Plan for that with short-lived tokens or periodic re-checks.

## Code

A FastAPI SSE endpoint. It sets the headers that defeat proxy buffering, streams deltas, emits a
final `[DONE]`, and closes the upstream stream if the client goes away.

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

app = FastAPI()
llm = AsyncOpenAI(timeout=60.0)


def sse(payload: dict | str) -> str:
    """One SSE record. The trailing blank line is what flushes it to the client."""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


async def token_stream(prompt: str) -> AsyncIterator[str]:
    yield ": connected\n\n"          # comment line: forces headers out through proxies
    stream = await llm.chat.completions.create(
        model="<current-model-id>",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    try:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield sse({"delta": delta})
    finally:
        # Runs on client disconnect too: stop paying for tokens nobody reads.
        await stream.close()
    yield sse("[DONE]")


@app.get("/v1/chat/stream")
async def chat_stream(q: str) -> StreamingResponse:
    return StreamingResponse(
        token_stream(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # nginx: do not buffer this response
            "Connection": "keep-alive",
        },
    )
```

Test it with `curl -N 'http://localhost:8000/v1/chat/stream?q=hello'`. You should see records
arrive one at a time rather than in one burst at the end — that is the whole check.

The browser side. This client renders deltas as they arrive and handles the done marker, using
`fetch` with a stream reader so it can send an auth header (`EventSource` cannot).

```ts
export async function streamChat(
  prompt: string,
  onDelta: (text: string) => void,
  apiKey: string,
): Promise<void> {
  const res = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ message: prompt }),
  });
  if (!res.ok || res.body === null) throw new Error(`stream failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Records are separated by a blank line; a partial record stays in the buffer.
    const records = buffer.split("\n\n");
    buffer = records.pop() ?? "";

    for (const record of records) {
      const line = record.split("\n").find((l) => l.startsWith("data:"));
      if (line === undefined) continue;              // comment or keep-alive
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      const parsed = JSON.parse(payload) as { delta: string };
      onDelta(parsed.delta);
    }
  }
}
```

## Real use case

A recruiting platform added an AI feature that wrote a first draft of a job description. It
worked, but candidates in usability testing kept clicking the button a second time because
"nothing happened". The endpoint returned a single JSON body after the full generation, and at
600 tokens that was several seconds of nothing.

They switched to SSE, and the same latency became acceptable without any model change. Two
details made the rollout work. Their CDN buffered responses by default, which produced the exact
same "nothing happens" bug in staging — solved with `X-Accel-Buffering: no` and an immediate
comment line so the headers flushed. And when they added a stop button, they discovered the
server kept generating after the tab closed; wrapping the upstream stream in `try/finally` and
closing it cut their provider spend by a noticeable margin, because abandoned drafts were
common.

They kept SSE rather than moving to WebSockets, because the interaction is one prompt, one
draft. When they later added "revise this paragraph" inside the draft, they still used a fresh
SSE request rather than a persistent socket — simpler to authorise and to debug.

## Common mistakes

- **Mistake:** forgetting the blank line after each `data:` record — **Instead:** always end with `\n\n`; without it the client buffers forever.
- **Mistake:** letting a proxy buffer the stream — **Instead:** set `X-Accel-Buffering: no` and send an early comment line.
- **Mistake:** leaving the upstream provider stream open on disconnect — **Instead:** close it in a `finally` block so cancelled requests stop costing money.
- **Mistake:** using `EventSource` when you need to send an auth header — **Instead:** use `fetch` with a body reader, as above.
- **Mistake:** choosing WebSockets for a request-response chat — **Instead:** use SSE; you will spend a week on reconnect and scaling logic you did not need.

## Checkpoint

1. Add a `Content-Type: application/json` POST version of the streaming endpoint that reads `message` from the body, and confirm `curl -N` still streams incrementally.
2. Measure the wire bytes of a 200-token stream and divide by the token count. How much of each record is framing versus content?
3. Add an `AbortController` to the TypeScript client wired to a stop button, and verify server-side that the upstream provider call is cancelled when you press it.

**Self-check:** a record like `data: {"delta":" word"}\n\n` is roughly 24 bytes for about 5 bytes of content, so framing dominates the wire — but at tens of tokens per second that is still well under a kilobyte per second.

## Go deeper

- [Async Processing](03-async-processing.md) — what else can block a stream mid-flight.
- [Streaming](../07-llm-apis/03-streaming.md) — the provider side you are proxying.
- [Ship: A Streaming Chat API](09-ship-streaming-api.md) — this endpoint with auth, quotas and caching around it.
- [Scaling Trade-offs](../21-system-design/09-scaling-tradeoffs.md) — why long-lived connections change your capacity model.

---
[← Previous: FastAPI and Flask](01-fastapi-flask.md) · [Stage index](README.md) · [Next: Async Processing](03-async-processing.md)
