# 3. Streaming Responses

> **Stage 07 · LLM APIs** — delivering tokens as they are generated, from server to browser.
>
> **You'll be able to:** consume a streamed response with a Python async generator and relay it to a browser over Server-Sent Events.
>
> **Prerequisites:** [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [Generators](../01-programming-basics/03-generators.md)

## Why this exists

A 900-token answer takes tens of seconds to generate. Without streaming, the user stares at a spinner for all of it and then reads the whole answer at once. With streaming, the first words appear in about a second and the experience feels like a person typing. Nothing about the total time changed — only when the value arrives.

Streaming is not just a UI nicety. It changes your error model: failures can happen *after* you have already sent bytes to the client, so you can no longer return a clean 500. It also changes your cost accounting: usage numbers may only arrive in the final chunk.

The mental model: **the response is a sequence of small deltas, and your job is to concatenate them while staying honest about partial failure.**

## The mental model

```text
server                                    client
------                                    ------
POST /chat (stream: true)
  |  data: {"delta": "Hel"}          -->   append "Hel"
  |  data: {"delta": "lo"}           -->   append "lo"
  |  data: {"delta": " world"}       -->   append " world"
  |  data: {"usage": {...}}          -->   record cost
  |  data: [DONE]                    -->   close

rule: the first token is a UI win,
      the last token is a correctness win
```

| Layer | Mechanism | Failure you must handle |
|---|---|---|
| Provider to server | SSE frames over HTTP | connection drop mid-stream |
| Server to client | SSE frames over HTTP | client disconnect |
| Server internals | async generator | exception after first yield |

## Core concepts

### Server-Sent Events are just text lines

SSE is a tiny protocol: `data: <payload>\n\n` per event, with optional `event:` and `id:` lines. Providers use it for token deltas. You do not need a library to parse it, and on the server side you write it with a `text/event-stream` content type and a flush after each frame.

### Providers wrap deltas differently

OpenAI streams `choices[0].delta.content`. Anthropic streams typed events, where text arrives as `content_block_delta` with a `text_delta` payload, plus lifecycle events for message start and stop. Both end with something you must detect rather than assume — OpenAI sends a `[DONE]` sentinel, Anthropic sends a `message_stop` event.

### The async generator is the right server abstraction

One `async def` that yields strings can be consumed by a terminal loop, a FastAPI `StreamingResponse`, or a test. It keeps the provider-specific parsing in one place and gives the caller a clean `async for chunk in stream(...)`.

### Partial failure needs a policy

Decide before you ship: if the stream dies at token 300, do you (a) close the connection and let the client retry the whole request, (b) send an explicit error event and keep the partial text, or (c) transparently restart. Transparent restart duplicates text already shown. Most products choose (b) — send an error event, keep what arrived, and mark the message as incomplete.

### Usage arrives late or not at all

Some providers send usage in a final chunk, some only when you pass an option, some not on streams at all. If cost matters, request the final usage chunk explicitly and check the docs. Never estimate a streamed cost from the number of deltas you happened to receive.

## Code

An async generator that normalises both providers into plain text deltas, and returns usage at the end.

```python
from __future__ import annotations

import os
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
anthropic_client = AsyncAnthropic()

# Placeholders; verify current model ids in the provider docs.
OPENAI_MODEL = os.environ["OPENAI_MODEL"]
ANTHROPIC_MODEL = os.environ["ANTHROPIC_MODEL"]


async def stream_openai(system: str, user: str) -> AsyncIterator[str]:
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        # ask for usage on the final chunk; option name varies, check docs
        stream_options={"include_usage": True},
    )
    async for event in stream:
        if not event.choices:
            continue                      # usage-only chunk has no choices
        delta = event.choices[0].delta.content
        if delta:
            yield delta


async def stream_anthropic(system: str, user: str) -> AsyncIterator[str]:
    async with anthropic_client.messages.stream(
        model=ANTHROPIC_MODEL,
        system=system,
        max_tokens=800,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        # the SDK gives typed events; we only care about text deltas
        async for text in stream.text_stream:
            yield text
```

Consuming it is three lines, which is the payoff:

```python
import asyncio


async def main() -> None:
    async for piece in stream_anthropic("You are terse.", "Explain SSE in one line."):
        print(piece, end="", flush=True)
    print()


asyncio.run(main())
```

On the Node side, relay to a browser. This is the pattern that makes a chat UI feel alive, and it needs no framework.

```ts
// Express-style handler; `req`/`res` are the usual Node request/response.
import type { Request, Response } from "express";
import OpenAI from "openai";

const openai = new OpenAI();

export async function chatStream(req: Request, res: Response): Promise<void> {
  res.setHeader("content-type", "text/event-stream");
  res.setHeader("cache-control", "no-cache");
  res.setHeader("connection", "keep-alive");
  res.flushHeaders(); // send headers now so the browser starts listening

  const abort = new AbortController();
  req.on("close", () => abort.abort()); // client went away: stop spending

  try {
    const stream = await openai.chat.completions.create(
      {
        model: process.env.OPENAI_MODEL!, // model id from config
        messages: [{ role: "user", content: String(req.body?.prompt ?? "") }],
        stream: true,
      },
      { signal: abort.signal },
    );

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        // SSE frame: one data line, blank line to terminate
        res.write(`data: ${JSON.stringify({ delta })}\n\n`);
      }
    }
    res.write("data: [DONE]\n\n");
  } catch (err) {
    // after headers are sent, this is the only way to report failure
    res.write(`event: error\ndata: ${JSON.stringify({ message: String(err) })}\n\n`);
  } finally {
    res.end();
  }
}
```

The browser side is a loop over `EventSource` or a `fetch` reader; either way you append `delta` to a DOM node and stop on `[DONE]`.

## Real use case

A coding-assistant web app answers questions about a customer's repository. Answers often run long, with code blocks. Before streaming, the median perceived wait was long enough that users clicked away and re-submitted, doubling spend on abandoned requests.

The team added an SSE endpoint and rendered deltas into a `<pre>`, with an explicit `event: error` frame so a mid-stream failure showed "response incomplete, retry" instead of a silently truncated answer. They also wired `req.on("close")` to an `AbortController`, which cut spend on abandoned requests because generation stopped when the tab closed. The failure mode before was not slowness — it was that a truncated answer looked complete, so users trusted a half-sentence as the whole explanation.

## Common mistakes

- **Mistake:** buffering the whole stream server-side and sending it at the end. — **Instead:** flush each frame; buffering throws away the entire benefit.
- **Mistake:** trying to send a 500 after the first byte. — **Instead:** use an explicit error event frame and mark the message incomplete.
- **Mistake:** ignoring client disconnect. — **Instead:** abort the provider request on `close`, or you pay for tokens nobody reads.
- **Mistake:** assuming usage numbers arrive on streams. — **Instead:** request the usage chunk if the provider supports it, and check docs for the option name.
- **Mistake:** parsing SSE by splitting on `\n` and assuming one JSON object per line. — **Instead:** split on the blank-line frame boundary, because data can span lines.

## Checkpoint

1. Write an async generator that yields `(delta, is_last)` tuples so the caller knows when the stream ends.
2. Add a token counter to the consumer and abort after 200 deltas, confirming the provider request is cancelled.
3. Serve a stream from FastAPI's `StreamingResponse` and open it in a browser tab with `EventSource`; then kill the server mid-stream and record what the client sees.

**Self-check:** you yield each delta as it arrives and flush it, so the caller sees text before generation finishes; the `is_last` flag (or a sentinel) tells the caller the stream ended normally rather than by failure.

## Go deeper

- [Generators](../01-programming-basics/03-generators.md) — the Python mechanics behind `yield` and `async for`.
- [Structured Outputs](04-structured-outputs.md) — the awkward middle ground: streaming a JSON object.
- [WebSockets](../17-ai-backend/02-websockets.md) — when SSE is not enough because you need bidirectional traffic.
- [Ship: A Streaming API](../17-ai-backend/09-ship-streaming-api.md) — production hardening of this pattern.

---
[← Previous: Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md) · [Stage index](README.md) · [Next: Structured Outputs](04-structured-outputs.md)
