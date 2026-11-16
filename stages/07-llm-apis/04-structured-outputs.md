# 4. Structured Outputs

> **Stage 07 · LLM APIs** — getting JSON your code can trust, not JSON that merely looks like JSON.
>
> **You'll be able to:** constrain a model to a schema, validate the result with pydantic or zod, and repair failures without hiding them.
>
> **Prerequisites:** [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) · [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md)

## Why this exists

The moment an LLM answer feeds a database, a branch in your code or a downstream API, free text stops being acceptable. You need fields with names and types. The naive approach — "please reply in JSON" in the prompt — works most of the time and fails in ways that are hard to debug: a trailing comma, a missing key, a string where you asked for a number, a wrapper object you did not expect.

Structured outputs solve this at three layers, and knowing which layer you are on matters. Prompting is a request. Schema-constrained decoding is an enforcement. Validation is your own guarantee. You want all three, in that order of trust.

The mental model: **the schema is a contract; the model is a supplier who usually complies, and your validator is the receiving inspector.**

## The mental model

```text
  schema (pydantic / zod)
        |
        v
  request:  "respond with this JSON schema"  or  json_schema mode
        |
        v
  model output  ---> parse ---> validate ---> use
                        |          |
                        |          +-- fails? repair loop (1 retry)
                        +-- fails? same loop, with the error text
```

| Approach | Enforcement | Portability | Use when |
|---|---|---|---|
| "Reply in JSON" in the prompt | none | any model | quick scripts, never production |
| JSON mode (valid JSON, no schema) | syntax only | most providers | you parse it yourself anyway |
| Schema-constrained output | shape enforced | provider-specific | production extraction |
| Tool/function schema for shape | shape enforced | widest support | the provider lacks a native schema mode |

## Core concepts

### Three levels of "structured", and they are not equal

JSON mode guarantees syntactically valid JSON but says nothing about keys. Schema mode guarantees the keys and types, usually by constraining which tokens can be emitted. Prompt-only structure guarantees nothing. Mixing these up is the most common source of "it worked in testing".

### Schema design decides success rate

Small, flat schemas with `enum` fields and explicit descriptions validate far more reliably than deeply nested ones with optional branches. If a field is genuinely optional, model it as such rather than making it nullable and then guessing. Keep field names self-describing — `severity` beats `s`.

### Validation is not the same as parsing

`json.loads` succeeding means the bytes were JSON. It does not mean `severity` is one of your allowed values. Always parse into a typed model — pydantic v2 in Python, zod in TypeScript — so the failure is a clear error at the boundary rather than a confusing crash three layers down.

### The repair loop is a bounded retry, not a loop

On validation failure, send the model its own output plus the validator's error message and ask for a corrected object. Cap it at one or two attempts, and count those attempts as cost. An unbounded repair loop is how a small bug becomes an outage.

### Streaming and structure fight each other

A streamed JSON object is unusable until it is complete, so partial parsing only helps for specific patterns (a top-level array of independent items). If you need both, stream progress to the UI and parse only at the end.

## Code

Extraction with a pydantic model as the contract, a schema-constrained call, and a bounded repair loop. The model id is a placeholder — check the docs for the current one and for the exact schema option name.

```python
from __future__ import annotations

import json
import os
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

client = AsyncOpenAI()
MODEL = os.environ["OPENAI_MODEL"]  # check docs: model ids rotate


class Ticket(BaseModel):
    """The contract. Every field is required and typed."""

    title: str = Field(description="Short summary, under 60 characters")
    severity: Literal["low", "medium", "high", "critical"]
    product_area: Literal["billing", "auth", "api", "ui", "other"]
    needs_human: bool = Field(description="True if a human must respond today")


SYSTEM = (
    "You extract structured ticket data from customer messages. "
    "Reply with JSON only. Never invent a product area; use 'other' if unsure."
)


async def extract(message: str, max_repairs: int = 1) -> Ticket:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": message},
    ]
    last_error = ""
    for attempt in range(max_repairs + 1):
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,  # extraction wants determinism, not creativity
            # schema-constrained output; option name/format varies by provider
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ticket",
                    "schema": Ticket.model_json_schema(),
                    "strict": True,
                },
            },
        )
        raw = resp.choices[0].message.content or ""
        try:
            return Ticket.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            # feed the failure back: the model is good at fixing named errors
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "That failed validation with this error:\n"
                        f"{last_error}\nReturn corrected JSON only."
                    ),
                },
            ]
    raise ValueError(f"could not produce a valid Ticket: {last_error}")
```

In TypeScript the same contract is a zod schema, and `safeParse` gives you the error text for the repair message.

```ts
import { z } from "zod";

export const TicketSchema = z.object({
  title: z.string().max(60),
  severity: z.enum(["low", "medium", "high", "critical"]),
  product_area: z.enum(["billing", "auth", "api", "ui", "other"]),
  needs_human: z.boolean(),
});
export type Ticket = z.infer<typeof TicketSchema>;

export async function extractTicket(
  message: string,
  callModel: (prompt: string) => Promise<string>,
): Promise<Ticket> {
  const first = await callModel(
    `Extract a ticket as JSON with keys title, severity, product_area, needs_human.\n\n${message}`,
  );

  const parsed = TicketSchema.safeParse(safeJson(first));
  if (parsed.success) return parsed.data;

  // one bounded repair attempt, carrying the validator's complaint
  const repaired = await callModel(
    `Your JSON failed validation: ${parsed.error.message}\n` +
      `Return corrected JSON only.\n\nPrevious output:\n${first}`,
  );
  return TicketSchema.parse(safeJson(repaired)); // fail loudly if still wrong
}

function safeJson(text: string): unknown {
  // models sometimes wrap JSON in prose or fences; strip the common cases
  const trimmed = text.trim().replace(/^```(?:json)?/, "").replace(/```$/, "");
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  return JSON.parse(start >= 0 ? trimmed.slice(start, end + 1) : trimmed);
}
```

## Real use case

An invoice-processing feature in an expense app reads a forwarded email and produces a `vendor`, `amount`, `currency` and `due_date`. The first version asked for JSON in the prompt and parsed with `json.loads`. It worked in demos and failed in production on roughly the cases that mattered: amounts as `"₹1,200"` strings, dates as "next Friday", and occasionally a chatty preamble before the object.

The rewrite defined a pydantic model with `amount: float`, `currency: Literal["INR","USD","EUR"]` and an ISO date string, used schema-constrained output, and added a single repair attempt that quoted the validator error. Amounts arriving as strings now failed validation and were repaired into numbers instead of silently reaching the ledger as text. The failure mode before was not bad model output — it was that the parser accepted output the domain could not.

## Common mistakes

- **Mistake:** treating JSON mode as schema enforcement. — **Instead:** add your own typed validation, because valid JSON can still have the wrong keys.
- **Mistake:** letting the repair loop run unbounded. — **Instead:** cap it at one or two attempts and raise a typed error after that.
- **Mistake:** parsing with `json.loads` and indexing keys directly. — **Instead:** validate into pydantic or zod so failures surface at the boundary.
- **Mistake:** designing a deeply nested schema with many optional branches. — **Instead:** flatten it, use enums, and split into two calls if it is genuinely two extractions.
- **Mistake:** forgetting that the repair call costs tokens. — **Instead:** log repair attempts as their own cost rows; a rising repair rate is a prompt-quality signal.

## Checkpoint

1. Add a `confidence: float` field with a validator that rejects values outside 0-1, and confirm a bad value triggers exactly one repair.
2. Convert the `Ticket` schema to zod and run the same five sample messages through both implementations, comparing where they disagree.
3. Write a test that feeds deliberately broken JSON (fenced, prose-wrapped, trailing comma) into your parser and asserts the repair path fires.

**Self-check:** you call `Ticket.model_validate_json` (or `safeParse`) and catch the validation error, then resend the model its output plus that error text once before failing.

## Go deeper

- [Tool Calling](05-tool-calling.md) — the same schema machinery used to make the model call functions.
- [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md) — measuring how often extraction actually validates.
- [Validation](../03-data-engineering/07-validation.md) — data-contract thinking applied to model output.
- [JSON and YAML](../01-programming-basics/08-json-yaml.md) — the parsing fundamentals this builds on.

---
[← Previous: Streaming Responses](03-streaming.md) · [Stage index](README.md) · [Next: Tool Calling](05-tool-calling.md)
