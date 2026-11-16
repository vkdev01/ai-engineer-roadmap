# 1. The OpenAI and Anthropic Chat APIs

> **Stage 07 · LLM APIs** — the two chat APIs most production LLM code is written against.
>
> **You'll be able to:** send a chat request, read the response object, and know which parts of it your code should depend on.
>
> **Prerequisites:** [Tokenization](../06-llm-fundamentals/02-tokenization.md) · [Context Windows](../06-llm-fundamentals/03-context-windows.md)

## Why this exists

A model is not a function you import. It is a service behind HTTPS: you post a list of messages, it posts back a list of content blocks and some numbers. Before chat APIs existed, you had to host the model and write the sampler yourself. Now the hard part is not the model, it is the shape of the conversation you send and the parts of the answer you trust.

The reason to learn the raw shape — not just the SDK — is that SDKs change, and every provider borrows from the same two shapes. OpenAI uses `messages` with a `role` and a `content` string. Anthropic uses `messages` plus a separate `system` parameter, and `content` is a list of blocks. Learn both once and every other provider becomes a mapping exercise.

The mental model: **a chat call is stateless.** The model remembers nothing between calls. The conversation is an array you rebuild and resend each time, which is why long chats cost more and why context-window limits are your problem, not the provider's.

## The mental model

```text
your code                     provider
--------                     --------
messages: [                  request
  {role: system, ...}   -->  validate + count tokens
  {role: user,   ...}        run the model
  {role: assistant,..}       sample tokens
  {role: user,   ...}   <--  response + usage numbers
]
                             nothing is stored server-side
```

| Concept | OpenAI shape | Anthropic shape |
|---|---|---|
| System prompt | a `system` role message | top-level `system` string |
| Message content | string, or list of parts | always a list of blocks |
| Reply location | `choices[0].message.content` | `content[0].text` |
| Stop reason | `finish_reason` | `stop_reason` |
| Usage | `usage.prompt_tokens` / `completion_tokens` | `usage.input_tokens` / `output_tokens` |
| Model id | `model="..."` | `model="..."` |

## Core concepts

### The three roles are the whole protocol

A message is `role` plus `content`. `system` sets standing behaviour, `user` is what the human said, `assistant` is what the model said (including its own earlier turns). You replay the whole array every call, so trimming old turns is a decision you make, not the provider's.

### Content is either a string or a list of parts

OpenAI accepts a plain string for simple text. Anthropic always wants a list. The moment you add an image, a document or a tool result, both go to the list form — so write your code in the list form from day one and you will never rewrite it.

### Sampling parameters are your output shape controls

`temperature` flattens or sharpens the distribution, `max_tokens` (or `max_output_tokens`) caps length and therefore cost, `stop` sequences end generation early. For extraction and classification tasks, low temperature is the default you want; for brainstorming, higher. Never assume defaults match across providers — pass them explicitly.

### Usage numbers are the only ground truth for cost

Both providers return token counts on every response. These are authoritative in a way that your own tokeniser estimate is not, because they include the chat template overhead. Store them. If a provider omits usage on a streamed call, that is a reason to request the final usage chunk, not to guess.

### Model ids expire

Model names are versioned strings and older ones get retired on a schedule. Read the model id from configuration, never from a hardcoded literal buried in a function, and check the provider's current docs before you ship. The code below uses a placeholder for exactly this reason.

## Code

A minimal typed wrapper around both providers. Note that the return type is ours, not the SDK's — that is the habit that makes provider swapping possible later.

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Sequence

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# Check each provider's docs for current model ids; these are placeholders.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "set-me-in-config")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "set-me-in-config")


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


openai_client = AsyncOpenAI()          # reads OPENAI_API_KEY from env
anthropic_client = AsyncAnthropic()    # reads ANTHROPIC_API_KEY from env


async def openai_chat(system: str, turns: Sequence[Turn]) -> Reply:
    resp = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=800,
        messages=[{"role": "system", "content": system}]
        + [{"role": t.role, "content": t.text} for t in turns],
    )
    choice = resp.choices[0]
    return Reply(
        text=choice.message.content or "",
        model=resp.model,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        stop_reason=choice.finish_reason,
    )


async def anthropic_chat(system: str, turns: Sequence[Turn]) -> Reply:
    resp = await anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        system=system,                       # system is not a message here
        temperature=0.2,
        max_tokens=800,
        messages=[
            {"role": t.role, "content": [{"type": "text", "text": t.text}]}
            for t in turns
        ],
    )
    # content is a list of blocks; text blocks carry the prose
    text = "".join(b.text for b in resp.content if b.type == "text")
    return Reply(
        text=text,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )
```

You should see the same `Reply` object from either function, with real token counts. That symmetry is the point.

The TypeScript version is shorter because both SDKs return plain JSON. Keep the same discipline: a local type at the edge, mapped once.

```ts
// npm i openai @anthropic-ai/sdk
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

export type Turn = { role: "user" | "assistant"; text: string };
export type Reply = {
  text: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  stopReason: string | null;
};

const openai = new OpenAI();                       // OPENAI_API_KEY
const anthropic = new Anthropic();                 // ANTHROPIC_API_KEY

export async function openaiChat(
  system: string,
  turns: Turn[],
  model: string, // pass the model id in; never hardcode it in a helper
): Promise<Reply> {
  const res = await openai.chat.completions.create({
    model,
    temperature: 0.2,
    max_tokens: 800,
    messages: [
      { role: "system", content: system },
      ...turns.map((t) => ({ role: t.role, content: t.text })),
    ],
  });
  const choice = res.choices[0];
  return {
    text: choice.message.content ?? "",
    model: res.model,
    inputTokens: res.usage?.prompt_tokens ?? 0,
    outputTokens: res.usage?.completion_tokens ?? 0,
    stopReason: choice.finish_reason,
  };
}

export async function anthropicChat(
  system: string,
  turns: Turn[],
  model: string,
): Promise<Reply> {
  const res = await anthropic.messages.create({
    model,
    system,
    temperature: 0.2,
    max_tokens: 800,
    messages: turns.map((t) => ({
      role: t.role,
      content: [{ type: "text" as const, text: t.text }],
    })),
  });
  const text = res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  return {
    text,
    model: res.model,
    inputTokens: res.usage.input_tokens,
    outputTokens: res.usage.output_tokens,
    stopReason: res.stop_reason,
  };
}
```

## Real use case

A support-ticket triage service at a mid-sized SaaS company. Each incoming ticket is sent with a system prompt listing the product's categories and a user message containing the ticket text. The service needs a category and a one-line reason, so it uses low temperature, a small `max_tokens`, and stores `usage` on every call.

The first version hardcoded one model id inside the handler. When that model was retired, the deploy failed at 2am and nobody knew which files to touch. The second version moved the model id into config and wrapped both providers behind a `Reply` dataclass. When the team later added Anthropic for a second opinion on low-confidence tickets, the handler did not change: it received two `Reply` objects and compared them. The failure mode before was provider coupling — the code knew the SDK's shape, so it could not be pointed at a different one.

## Common mistakes

- **Mistake:** expecting the model to remember the previous request. — **Instead:** resend the whole message array each time and treat history as your own data structure.
- **Mistake:** hardcoding a model id in a function body. — **Instead:** read it from config or environment, because model ids are retired on a schedule.
- **Mistake:** assuming `content` is a string on every provider. — **Instead:** use the list-of-parts form, which is the only form that survives images and tool results.
- **Mistake:** ignoring the `usage` field and estimating cost from character counts. — **Instead:** store the provider's numbers; they include template overhead your estimate misses.
- **Mistake:** letting `max_tokens` default silently. — **Instead:** set it explicitly, since it is both a latency and a spend ceiling.

## Checkpoint

1. Write a function that takes a list of `Turn` objects and returns the total input tokens of the last call, for both providers.
2. Add a `stop` sequence so generation halts at the first newline, and verify the reply is a single line.
3. Build a tiny CLI that keeps history in a list, appends each user message and each assistant reply, and prints the running token total — then explain at which turn you would need to start trimming.

**Self-check:** you read `usage.prompt_tokens` for OpenAI and `usage.input_tokens` for Anthropic, from the response object, not from a local estimate.

## Go deeper

- [Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md) — the third provider shape, and how open models fit in.
- [Streaming Responses](03-streaming.md) — the same request, delivered incrementally.
- [Tokenization](../06-llm-fundamentals/02-tokenization.md) — why those usage numbers are not word counts.
- [Instruction Tuning](../06-llm-fundamentals/06-instruction-tuning.md) — why a chat-shaped model follows `system` at all.

---
[← Previous: Stage 06 — LLM Fundamentals](../06-llm-fundamentals/README.md) · [Stage index](README.md) · [Next: Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md)
