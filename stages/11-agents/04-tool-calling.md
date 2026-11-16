# 4. Tool Calling

> **Stage 11 · AI Agents** — designing tools the model can use correctly: schemas, validation, idempotency, error messages that name the fix, and sandboxing side effects.
>
> **You'll be able to:** turn any function into a tool the model calls reliably, and refuse the arguments that would hurt.
>
> **Prerequisites:** [Agent Architecture](02-agent-architecture.md) · [Tool Calling](../07-llm-apis/05-tool-calling.md)

## Why this exists

The model does not call your function. It emits JSON that your code may or may not turn into a function call. That gap is where agents fail: a required field left out, a date in the wrong format, a `limit` of 5000 that returns a 200 KB blob, a refund tool run twice because the model retried after a timeout.

Tool calling is therefore a **user-interface problem**, and the model is your user. A good tool has a short honest description, a schema that makes the wrong call hard to express, a validator that rejects nonsense with a message the model can act on, and an effect that is safe to repeat.

The mental model: **the schema is the contract, the validator is the enforcement, the error string is the documentation the model actually reads.**

## The mental model

```text
model intent ──► JSON args ──► [validate] ──bad──► "start_date must be YYYY-MM-DD,
                                    │               got '3 March'. Retry."
                                    good
                                    ▼
                            [policy: read-only? gated?] ──► [execute, idempotent]
                                    │
                                    ▼
                            compact labelled observation
```

## Core concepts

### Descriptions are prompts; write them like prompts

The description is the only documentation the model reliably sees. Say what the tool does, when to use it, when not to, and what it returns — in that order, in one or two sentences. "Searches the support KB. Use for product questions; do not use for billing, which has its own tool. Returns up to 5 passages with ids." A vague description produces vague calls, and no amount of schema strictness fixes it.

### Make wrong calls hard to express

Use enums instead of free strings, required fields instead of optional ones with defaults, and narrow types (`int` with `minimum: 1, maximum: 50`). If a parameter only accepts three values, an enum removes a whole failure class. If two parameters must travel together, model them as one object. Every constraint you put in the schema is a constraint you do not have to explain in prose.

### Validate before you execute

Schema validation by the provider is not enough — it is advisory, and models do drift. Validate again in your code with Pydantic (Python) or zod (TypeScript) and treat failure as a teaching moment, not an exception. The message you return becomes the next observation, so make it name the field, the value received, and the accepted format.

### Idempotency: make repeats harmless

Any tool with an external effect — send, create, charge, delete — should take an idempotency key. Derive it from the semantic content (customer id plus invoice id plus amount), not from a timestamp, and let the downstream system collapse duplicates. This is what saves you when a request times out after the server already committed.

### Sandbox the side effects

Split your tools into read-only and side-effecting, and treat them differently. Read-only tools run freely. Side-effecting tools get a dry-run mode, a policy check, and — in anything user-facing — an approval gate from [Human in the Loop](08-human-in-the-loop.md). For code execution or shell tools, run in a container with no network and a time limit; see [Sandboxing](../20-ai-security/05-sandboxing.md).

## Code

A typed registry where the schema, the validator and the function cannot drift apart, because the schema is generated from the Pydantic model. The error path returns a string for the model, never an exception.

```python
from __future__ import annotations

from datetime import date
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=200, description="Plain keywords, not a sentence.")
    limit: int = Field(default=5, ge=1, le=10, description="How many passages to return.")
    scope: Literal["docs", "billing", "all"] = Field(default="all")

class RefundArgs(BaseModel):
    charge_id: str = Field(pattern=r"^ch_[A-Za-z0-9]{6,}$")
    amount_paise: int = Field(gt=0, description="Amount in paise; must not exceed the charge.")
    reason: str = Field(min_length=10, max_length=200)
    idempotency_key: str = Field(description="Stable key: charge_id + amount_paise.")

ToolFn = Callable[[Any], Awaitable[str]]

def make_tool(name: str, description: str, args_model: type[BaseModel],
              fn: ToolFn, *, side_effect: bool) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "schema": args_model.model_json_schema(),   # one source of truth
        "validate": args_model.model_validate,
        "run": fn,
        "side_effect": side_effect,
    }

async def call_tool(tool: dict[str, Any], raw_args: dict[str, Any]) -> str:
    try:
        args = tool["validate"](raw_args)
    except ValidationError as exc:
        # The model reads this string next turn. Name the field and the fix.
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first["loc"])
        return (f"Invalid arguments for {tool['name']}: field '{field}' {first['msg']}. "
                f"Expected schema: {list(tool['schema'].get('properties', {}))}. Retry once.")
    return await tool["run"](args)

async def search(args: SearchArgs) -> str:
    rows = await _query_index(args.query, args.scope, args.limit)   # your retriever
    if not rows:
        return f'no results for query="{args.query}" scope={args.scope}. Try broader keywords.'
    lines = [f"{len(rows)} results for query=\"{args.query}\":"]
    lines += [f"- [{r.id}] {r.title}: {r.snippet[:160]}" for r in rows]
    return "\n".join(lines)                                          # compact, labelled
```

The same discipline in TypeScript, where zod gives you the JSON schema for free and the parse doubles as your runtime guard.

```ts
import { z } from "zod";

export const RefundArgs = z.object({
  chargeId: z.string().regex(/^ch_[A-Za-z0-9]{6,}$/),
  amountPaise: z.number().int().positive(),
  reason: z.string().min(10).max(200),
  idempotencyKey: z.string().min(8),
});

export type Tool<I> = {
  name: string;
  description: string;
  schema: z.ZodType<I>;
  sideEffect: boolean;
  run: (args: I) => Promise<string>;
};

export function makeTool<I>(t: Tool<I>): Tool<I> {
  return t;
}

export async function callTool<I>(
  tool: Tool<I>,
  raw: unknown,
  approve: (name: string, args: I) => Promise<boolean>,
): Promise<string> {
  const parsed = tool.schema.safeParse(raw);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    return `Invalid arguments for ${tool.name}: '${issue.path.join(".")}' ${issue.message}. Retry once.`;
  }
  if (tool.sideEffect && !(await approve(tool.name, parsed.data))) {
    return `Action ${tool.name} was not approved. Do not retry; report back instead.`;
  }
  return tool.run(parsed.data);
}
```

## Real use case

A logistics dashboard let an agent reschedule deliveries. The original tool took `new_date: string`, and the model sent `"tomorrow"`, `"2026-11-20"` and `"20/11/2026"` on different days; two of those crashed the handler and the run ended. It also retried after a gateway timeout, creating duplicate reschedules on the same order.

Two changes fixed it. `new_date` became a `date` type with a description naming the accepted format, and the tool took an `idempotency_key` of `order_id + new_date`, which the scheduling service collapsed. Invalid arguments now return a sentence the model reads and corrects, and the agent completes the task on the second attempt instead of dying.

## Common mistakes

- **Mistake:** one tool that takes an `action` string and dispatches internally — **Instead:** one tool per action, with a schema that only fits that action.
- **Mistake:** returning raw API JSON — **Instead:** return a short labelled summary with the ids the next step needs.
- **Mistake:** raising on validation failure — **Instead:** return a corrective message naming the field and format.
- **Mistake:** side-effecting tools without idempotency keys — **Instead:** derive a stable key from semantic content and let the downstream collapse repeats.
- **Mistake:** letting a shell or code tool run with the app's credentials and network — **Instead:** sandbox it with no network, a timeout and a read-only mount.
- **Mistake:** a 30-tool registry for a 3-tool job — **Instead:** expose fewer tools per agent; selection accuracy falls as the list grows.

## Checkpoint

1. Add a `dry_run: bool` field to `RefundArgs` that returns the plan without calling the payments API.
2. Write a tool description for `search` that states when *not* to use it, then check whether the model stops calling it for billing questions.
3. Build a validator that rejects a `refund` whose `amount_paise` exceeds the charge, using an idempotent lookup of the charge first.

**Self-check:** Exercise 1 — pass `dry_run` through to the handler and return `"would refund {amount} for {charge_id}"` before the API call.

## Go deeper

- [Reasoning and Planning](03-reasoning-planning.md) — how tools get sequenced.
- [Agent Security](11-agent-security.md) — treating tool output as untrusted input.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — the schema mechanics underneath.
- [Sandboxing](../20-ai-security/05-sandboxing.md) — running dangerous tools safely.

---

[← Previous: Reasoning and Planning](03-reasoning-planning.md) · [Stage index](README.md) · [Next: Memory and State](05-memory-and-state.md)
