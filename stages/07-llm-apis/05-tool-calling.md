# 5. Tool Calling

> **Stage 07 · LLM APIs** — letting the model ask your program to run something, then telling it what happened.
>
> **You'll be able to:** run a complete tool-calling loop, including tool results, and stop it safely.
>
> **Prerequisites:** [Structured Outputs](04-structured-outputs.md) · [HTTP and REST APIs](../01-programming-basics/07-http-rest-apis.md)

## Why this exists

A model can write "the weather in Bengaluru is 27°C" from training data, but it cannot know today's temperature. Tool calling closes that gap without fine-tuning: you describe functions in a schema, the model replies with a structured request to call one, your code executes it, and you feed the result back so the model can answer.

The crucial thing beginners miss: **the model never runs anything.** It emits a JSON object naming a function and arguments. Your process decides whether to run it. That separation is where all the safety lives, and it is why "the model deleted my table" is always a bug in your executor, not in the model.

The mental model: **a tool call is a structured output whose schema is a function signature, plus a protocol for sending the return value back.**

## The mental model

```text
user question
    |
    v
[ model ] --tool_call: get_weather{city:"Pune"}--> [ your executor ]
    ^                                                    |
    |                                         result: "31C, humid"
    |                                                    |
    +<---- messages += tool_result ----------------------+
    |
    v
final text answer  (or another tool_call, loop again)
```

| Step | Who acts | What is added to `messages` |
|---|---|---|
| 1 | model | an assistant message containing tool calls |
| 2 | your code | nothing — you execute |
| 3 | your code | a tool result message per call id |
| 4 | model | the final assistant text |

## Core concepts

### The schema is a JSON Schema, described in prose

Each tool has a name, a description and a parameter schema. The description is a prompt: it is how the model decides *when* to call. Vague descriptions produce wrong calls far more often than weak models do. Write them as you would to a new colleague.

### Tool calls come back with ids and must be answered

OpenAI returns `tool_calls` on the assistant message, each with an `id`. Anthropic returns `tool_use` blocks with an `id`. You must send back a result message that references that id — `role: "tool"` with `tool_call_id` on OpenAI, a `tool_result` block on Anthropic. Miss the id and the model loses the thread of which call it is answering.

### The loop needs a hard cap

The model can call tools repeatedly. A search tool that returns nothing useful can cause it to search again, forever. Cap iterations (three to five is typical), and make the cap produce a graceful "I could not find this" rather than an exception.

### Arguments are untrusted input

The model produces the arguments, so validate them with the same pydantic/zod schema before execution. A `city` field can contain anything, including an attempt to reach a different code path. Validate, then execute, then serialise the result — never `eval` and never string-format into SQL.

### Errors are results, not exceptions

If a tool fails, return the failure as the tool result. The model can then explain it, retry with different arguments, or ask the user. Swallowing errors and returning `""` teaches the model that the tool always succeeds.

## Code

A full loop with two tools, argument validation, an iteration cap and error-as-result handling.

```python
from __future__ import annotations

import json
import os
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

client = AsyncOpenAI()
MODEL = os.environ["OPENAI_MODEL"]  # check provider docs for current id


class WeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g. 'Pune'")
    unit: Literal["c", "f"] = "c"


class SearchArgs(BaseModel):
    query: str = Field(description="Short search query, under 10 words")


# The schema is a prompt: the description decides when the model calls it.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city. Use for any weather question.",
            "parameters": WeatherArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search internal product docs. Use for product questions.",
            "parameters": SearchArgs.model_json_schema(),
        },
    },
]

SCHEMAS = {"get_weather": WeatherArgs, "search_docs": SearchArgs}


async def execute(name: str, raw_args: str) -> str:
    """Validate arguments, then run. Always return a string, never raise."""
    schema = SCHEMAS.get(name)
    if schema is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        args = schema.model_validate_json(raw_args)
    except ValidationError as exc:
        return json.dumps({"error": "invalid arguments", "detail": str(exc)})

    if name == "get_weather":
        # a real implementation calls a weather API here
        return json.dumps({"city": args.city, "temp": 31, "unit": args.unit})
    return json.dumps({"hits": []})  # real implementation queries an index


async def answer(question: str, max_steps: int = 4) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Use tools when you need live data."},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        resp = await client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))  # keep tool_calls

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
            result = await execute(call.function.name, call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,       # must match the call
                    "content": result,
                }
            )
    return "I could not finish that with the tools available."
```

The Anthropic variant differs only in the result shape: you push a user message containing `tool_result` blocks.

```python
async def anthropic_tool_result(call_id: str, result: str) -> dict[str, Any]:
    # Anthropic: results go back as a user turn containing tool_result blocks
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": result}
        ],
    }
```

From a Node service, the loop is the same and `zod` plays the validator role. Note the `maxSteps` guard and the try/catch that becomes a tool result.

```ts
import { z } from "zod";

const WeatherArgs = z.object({ city: z.string(), unit: z.enum(["c", "f"]).default("c") });

export async function runTool(name: string, rawArgs: string): Promise<string> {
  if (name !== "get_weather") return JSON.stringify({ error: `unknown tool ${name}` });
  const parsed = WeatherArgs.safeParse(JSON.parse(rawArgs || "{}"));
  if (!parsed.success) {
    return JSON.stringify({ error: "invalid arguments", detail: parsed.error.message });
  }
  try {
    const weather = await fetchWeather(parsed.data.city); // your API call
    return JSON.stringify(weather);
  } catch (err) {
    return JSON.stringify({ error: String(err) }); // errors are results
  }
}

declare function fetchWeather(city: string): Promise<{ temp: number; unit: string }>;
```

## Real use case

A customer-support assistant for an e-commerce app. Tools: `order_status(order_id)`, `refund_eligibility(order_id)`, `search_policy(query)`. A user asks "where is my order and can I return it?" The model calls `order_status`, gets a delivered date, then calls `refund_eligibility`, gets `eligible: true, window_days: 7`, and answers with both facts and a link to the returns page.

The first version returned exceptions from the tool layer, so a bad order id crashed the handler and the user saw a generic error page. The second version returned errors as tool results, so the model said "I could not find that order id — could you check the number?" and asked for a correction. A third change capped the loop at four steps after a policy-search tool with zero hits caused repeated searching. The failure mode before was treating tool failures as server errors instead of conversational turns.

## Common mistakes

- **Mistake:** executing tool arguments without validating them. — **Instead:** validate against the declared schema first; the arguments are model output, not your data.
- **Mistake:** dropping the `tool_call_id` / `tool_use_id`. — **Instead:** echo it exactly, or the model cannot match result to request.
- **Mistake:** raising on tool failure. — **Instead:** return an error payload as the tool result so the model can recover or explain.
- **Mistake:** looping without a step cap. — **Instead:** cap at a small number and return a graceful fallback message.
- **Mistake:** writing tool descriptions like internal function docs. — **Instead:** write them as instructions for when to call and when not to.

## Checkpoint

1. Add a third tool that returns the current date, and confirm the model stops calling tools when the question needs no live data.
2. Force a tool to return `{"error": "timeout"}` and check that the final answer acknowledges the failure instead of inventing a value.
3. Log every tool call with its arguments and latency, then compute how many steps a typical question needs.

**Self-check:** you append an assistant message containing the tool call, then a `tool` message with `tool_call_id` set to the call's id and the result as `content`, then call the model again.

## Go deeper

- [Structured Outputs](04-structured-outputs.md) — the schema mechanics underneath tool calls.
- [Tool Calling in agents](../11-agents/04-tool-calling.md) — the same loop with planning and memory around it.
- [Agent Security](../11-agents/11-agent-security.md) — what changes when tools can write, not just read.
- [Multimodal APIs](06-multimodal-apis.md) — tools that return images.

---
[← Previous: Structured Outputs](04-structured-outputs.md) · [Stage index](README.md) · [Next: Multimodal APIs](06-multimodal-apis.md)
