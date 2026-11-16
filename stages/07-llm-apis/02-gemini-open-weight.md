# 2. Gemini and Open-Weight Models via Hosted Endpoints

> **Stage 07 · LLM APIs** — the third provider shape, and how to call open models without owning a GPU.
>
> **You'll be able to:** call Gemini-style and OpenAI-compatible endpoints, and decide when an open-weight model is the right choice.
>
> **Prerequisites:** [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [Open vs Closed Models](../06-llm-fundamentals/08-open-vs-closed.md)

## Why this exists

Two providers are not a market. Gemini-style APIs differ in one important way — the roles are `user` and `model`, and the content is a list of parts keyed by type. Open-weight models differ in a more important way: the *weights* are downloadable, but you almost never run them on your laptop for production. You call them through a hosted endpoint that speaks either the provider's own protocol or the OpenAI-compatible one.

That last point is the practical key. A large share of inference hosts expose an OpenAI-compatible `/v1/chat/completions`. If your client is written against that shape, swapping in a hosted open model is a base-URL and model-id change. The mental model: **an "open-weight model" is a file; the endpoint you call is just a server that loaded the file.**

## The mental model

```text
             +---------------------------+
             |  your client (one shape)  |
             +------------+--------------+
                          |
      +-------------------+-------------------+
      |                   |                   |
 Gemini-style      OpenAI-compatible     Provider-native
 (contents/parts)  (/v1/chat/...)        (bespoke SDK)
      |                   |                   |
   Gemini API      hosted open-weight      Anthropic
                   (vLLM, TGI, hosts)      messages API
```

| Question | Gemini-style | OpenAI-compatible host |
|---|---|---|
| Role names | `user`, `model` | `user`, `assistant` |
| Content | list of parts | string or list of parts |
| System prompt | system instruction field | `system` role message |
| Auth header | API key in header or query | `Authorization: Bearer ...` |
| Swap cost | new adapter | base URL + model id |

## Core concepts

### Gemini's shape: contents and parts

Gemini models requests as `contents`, a list of turns, each with a `role` and `parts`. A part is typed — `text`, `inline_data` for bytes, `file_data` for an uploaded reference. Roles are `user` and `model`, not `assistant`. Generation settings such as temperature and max output tokens live in a separate config object rather than being top-level arguments.

This is not a cosmetic difference. Any abstraction that assumes `messages[].content` as a string will need a real adapter for Gemini, not a rename.

### The OpenAI-compatible surface is the de facto standard

Most hosted inference providers implement the same endpoint path, the same JSON body and the same response shape as OpenAI. That is why "open-weight model support" in practice means "point the OpenAI SDK at a different `base_url`". You get streaming, tool calling and JSON mode only to the extent the host implements them, so verify per host rather than assuming.

### Hosted versus self-hosted is a cost and control trade

Hosted endpoints give you a URL and a bill. Self-hosting gives you a GPU bill and full control over quantisation, context length and data residency. For a learner, hosted is the right default; self-hosting is a later optimisation and belongs to the inference stage. The API code you write should not care which one is behind the URL.

### Open weights change the compliance conversation

With a closed API, your data leaves your network. With a hosted open-weight model, it also leaves — unless you self-host. The licence of the weights matters too: some permit commercial use, some add restrictions. Read the licence before you build a product on it; this is a legal question, not a technical one.

### Model ids and capabilities vary per host

The same weights served by two hosts may support different context lengths, different quantisations and different feature flags. Treat the model id and base URL as a pair, sourced from config, and check the host's docs for what is actually enabled.

## Code

A Gemini-style call, then the same logical call against an OpenAI-compatible host. Note how the adapter converts our `Turn` list into `contents`.

```python
from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.genai import types
from openai import AsyncOpenAI

# Placeholders: check each provider's docs for current model ids and hosts.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "set-me-in-config")
OPEN_HOST_URL = os.environ.get("OPEN_HOST_URL", "https://your-host.example/v1")
OPEN_HOST_MODEL = os.environ.get("OPEN_HOST_MODEL", "set-me-in-config")


@dataclass(frozen=True)
class Turn:
    role: str      # "user" | "assistant"
    text: str


gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# OpenAI-compatible hosts reuse the OpenAI SDK, only the base_url changes.
hosted = AsyncOpenAI(
    base_url=OPEN_HOST_URL,
    api_key=os.environ.get("OPEN_HOST_KEY", "not-needed"),
)


async def gemini_chat(system: str, turns: list[Turn]) -> str:
    contents = [
        types.Content(
            # Gemini calls the assistant role "model"
            role="model" if t.role == "assistant" else "user",
            parts=[types.Part(text=t.text)],
        )
        for t in turns
    ]
    resp = await gemini.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=800,
        ),
    )
    return resp.text or ""


async def hosted_chat(system: str, turns: list[Turn]) -> str:
    resp = await hosted.chat.completions.create(
        model=OPEN_HOST_MODEL,
        temperature=0.2,
        max_tokens=800,
        messages=[{"role": "system", "content": system}]
        + [{"role": t.role, "content": t.text} for t in turns],
    )
    return resp.choices[0].message.content or ""
```

Both functions return a string, so the caller is untouched. That is the adapter pattern in its smallest form.

From Node, the OpenAI-compatible host needs no special client at all — a plain `fetch` against the same path is enough, which keeps your dependency count low.

```ts
export async function hostedChat(
  system: string,
  turns: { role: "user" | "assistant"; text: string }[],
): Promise<string> {
  const baseUrl = process.env.OPEN_HOST_URL; // e.g. https://host.example/v1
  const model = process.env.OPEN_HOST_MODEL;
  if (!baseUrl || !model) throw new Error("host config missing");

  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.OPEN_HOST_KEY ?? ""}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 800,
      messages: [
        { role: "system", content: system },
        ...turns.map((t) => ({ role: t.role, content: t.text })),
      ],
    }),
  });

  if (!res.ok) {
    // keep the body: hosts put the real reason there
    throw new Error(`host error ${res.status}: ${await res.text()}`);
  }
  const json = (await res.json()) as {
    choices: { message: { content: string | null } }[];
  };
  return json.choices[0]?.message.content ?? "";
}
```

## Real use case

An internal document-summarisation tool at a logistics company handles both customer contracts and internal memos. Contracts must not leave the company network, so they go to a self-hosted open-weight model behind an OpenAI-compatible URL. Memos are lower sensitivity and go to a closed API for better quality.

The team first wrote two separate code paths with duplicated prompt handling, and the summariser outputs drifted until nobody could tell which pipeline produced which summary. The fix was one `summarise(text)` function that took a `route` argument and dispatched to either adapter — same prompt, same output handling, different base URL. The failure mode before was duplicated logic across providers, which made every prompt change a two-place edit and eventually an inconsistent product.

## Common mistakes

- **Mistake:** assuming Gemini accepts `assistant` as a role name. — **Instead:** map to `model` inside the adapter and keep provider names out of your core code.
- **Mistake:** pointing the OpenAI SDK at a host and assuming tool calling and JSON mode work. — **Instead:** test each feature against the specific host; support varies by server and version.
- **Mistake:** treating "open weights" as "self-hosted". — **Instead:** check whether the endpoint is hosted by a third party, because that decides your data-residency answer.
- **Mistake:** ignoring the model licence. — **Instead:** read it before commercial use; some weights carry use restrictions.
- **Mistake:** hardcoding a base URL in the client constructor. — **Instead:** read it from config so a host migration is a deploy change, not a code change.

## Checkpoint

1. Add a `gemini_chat` branch to a dispatcher that takes `provider: Literal["gemini", "hosted"]` and returns the same `Reply`-like object for both.
2. Send one image to Gemini using an inline bytes part and describe what changes in the `contents` structure.
3. Point your client at a different OpenAI-compatible base URL and run the same prompt; write down which features broke and why.

**Self-check:** you convert `assistant` to `model` and wrap text in a typed part, because Gemini models turns as `contents` of `parts` with those role names.

## Go deeper

- [Open vs Closed Models](../06-llm-fundamentals/08-open-vs-closed.md) — the trade-offs behind the routing decision.
- [Multimodal APIs](06-multimodal-apis.md) — sending images and files as typed parts.
- [Routing and Fallbacks](09-routing-fallbacks.md) — putting two providers behind one call path on purpose.
- [vLLM and SGLang](../16-inference/08-vllm-sglang.md) — what the server on the other end is actually doing.

---
[← Previous: The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [Stage index](README.md) · [Next: Streaming Responses](03-streaming.md)
