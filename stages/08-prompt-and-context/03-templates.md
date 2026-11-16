# 3. Prompt Templates with Variables and Validation

> **Stage 08 · Prompt and Context** — prompts as typed artefacts, not f-strings scattered through handlers.
>
> **You'll be able to:** define a prompt template with typed variables, validate inputs before rendering, and preview the exact text that will be sent.
>
> **Prerequisites:** [System Instructions That Actually Constrain](01-system-instructions.md) · [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md)

## Why this exists

The first version of every prompt is an f-string inside a function. It works. Then someone adds a conditional section, someone else adds an optional example block, a third person interpolates a user field directly, and suddenly nobody knows what text the model actually received on the request that went wrong.

A template fixes three problems: **what variables exist** (a schema), **what happens when one is missing or wrong** (validation), and **what exact text was sent** (a rendered preview you can log and diff). This is the difference between a prompt you edit and a prompt you can debug.

The mental model: **a prompt template is a function with a typed signature, whose body happens to be prose.**

## The mental model

```text
Template(id, version, variables: schema, body)
      |
      v
  render(values) --> validate --> fill slots --> Rendered(text, template_id, hash)
      |
      +-- missing/invalid var --> ValidationError (before any API call)
      |
      +-- Rendered.text logged with template_id + hash
```

| Bad practice | Template approach |
|---|---|
| `f"...{user_text}..."` inline | declared `user_text` variable, validated length |
| optional section via `if` in the handler | explicit `include_examples: bool` variable |
| silent empty string when a var is missing | validation error, no call made |
| "which prompt produced this?" | `template_id` + `version` + content hash on every log line |

## Core concepts

### Variables are a schema, not a string substitution

Declare each variable with a type and constraints: max characters, allowed values, required or not. Validation catches the real bugs — an empty retrieval result, a 200 KB document pasted into a field meant for a sentence, a `None` where a string was expected.

### Delimiting is part of the template

Any variable carrying untrusted content should be rendered inside fixed delimiters written by the template, not by the caller. That way the delimiter is always present and always matches what the system instruction describes. See [Prompt-Injection Defence](06-injection-defense.md).

### The template owns the structure, the caller owns the values

Callers pass values only. If a caller wants a different section order, that is a new template version. This keeps structure decisions in one reviewable file rather than spread across call sites.

### Rendered output is an artefact you keep

Return a `Rendered` object with the final text, the template id, the version and a content hash. Log the hash with every request. When a user reports a bad answer, the hash tells you exactly which text was sent without storing full prompts for every call.

### Never build prompts by concatenating rendered templates

Concatenating two templates makes a third, unversioned prompt. If a prompt genuinely has two parts, express that as one template with two sections.

## Code

A minimal template engine on pydantic: declared variables, validation, fixed delimiters, and a rendered artefact.

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator


class TicketVars(BaseModel):
    """The template's signature: everything the caller must supply."""

    question: str = Field(min_length=3, max_length=2_000)
    context: str = Field(default="", max_length=20_000)
    product_area: str = Field(default="general")
    tone: str = Field(default="neutral")

    @field_validator("product_area")
    @classmethod
    def known_area(cls, v: str) -> str:
        allowed = {"general", "billing", "auth", "api"}
        if v not in allowed:
            raise ValueError(f"product_area must be one of {sorted(allowed)}")
        return v

    @field_validator("tone")
    @classmethod
    def known_tone(cls, v: str) -> str:
        allowed = {"neutral", "concise", "warm"}
        if v not in allowed:
            raise ValueError(f"tone must be one of {sorted(allowed)}")
        return v


BODY = """\
Answer the question for the {product_area} area in a {tone} tone.

<context>
{context}
</context>

<question>
{question}
</question>
"""


@dataclass(frozen=True)
class Rendered:
    text: str
    template_id: str
    version: str
    content_hash: str


@dataclass(frozen=True)
class Template:
    template_id: str
    version: str
    body: str

    def render(self, values: dict[str, object]) -> Rendered:
        # validate BEFORE rendering: no API call happens with bad input
        v = TicketVars.model_validate(values)
        text = self.body.format(
            product_area=v.product_area,
            tone=v.tone,
            context=v.context or "(no context provided)",
            question=v.question,
        )
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return Rendered(text, self.template_id, self.version, digest)


TICKET_TEMPLATE = Template(template_id="ticket-answer", version="v3", body=BODY)
```

Because the delimiters live in `BODY`, a caller cannot forget them, and a malicious `context` value cannot escape its tags without the model seeing the closing tag first.

The TypeScript equivalent uses zod, which gives the same guarantee at the Node boundary — important when the values come from an HTTP request body.

```ts
import { createHash } from "node:crypto";
import { z } from "zod";

export const TicketVars = z.object({
  question: z.string().min(3).max(2000),
  context: z.string().max(20_000).default(""),
  productArea: z.enum(["general", "billing", "auth", "api"]).default("general"),
  tone: z.enum(["neutral", "concise", "warm"]).default("neutral"),
});
export type TicketVars = z.infer<typeof TicketVars>;

const BODY = `Answer the question for the {{productArea}} area in a {{tone}} tone.

<context>
{{context}}
</context>

<question>
{{question}}
</question>`;

export type Rendered = {
  text: string;
  templateId: string;
  version: string;
  contentHash: string;
};

export function renderTicket(values: unknown): Rendered {
  const v = TicketVars.parse(values); // throws before any model call
  const text = BODY.replace(/\{\{(\w+)\}\}/g, (_m, key: string) => {
    if (key === "context") return v.context || "(no context provided)";
    return String(v[key as keyof TicketVars] ?? "");
  });
  return {
    text,
    templateId: "ticket-answer",
    version: "v3",
    contentHash: createHash("sha256").update(text).digest("hex").slice(0, 16),
  };
}
```

## Real use case

A product-docs assistant had its prompt inline in a FastAPI handler. One bug took two days to find: a caller passed `context=None` when retrieval returned nothing, the f-string rendered the literal text `None`, and the model confidently answered "the document says None" for a subset of questions.

Moving to a template with a pydantic schema fixed it structurally: `context` is a string with a default of `""`, and the template substitutes an explicit "(no context provided)" phrase that the system instruction tells the model to treat as a missing-context signal. The same refactor added a `content_hash` to every log line, which is what let them confirm afterwards that no other caller was sending malformed values. The failure mode before was unvalidated interpolation — the prompt had no idea what a valid value looked like.

## Common mistakes

- **Mistake:** building prompts with `f"..."` inside handlers. — **Instead:** one template artefact with declared variables and validation.
- **Mistake:** letting the caller add delimiters. — **Instead:** put delimiters in the template body so they are always present and consistent.
- **Mistake:** defaulting a missing variable to `""` silently. — **Instead:** either require it or substitute an explicit sentinel the model is told to recognise.
- **Mistake:** concatenating templates at call time. — **Instead:** make it one template with sections, so the version covers the whole prompt.
- **Mistake:** logging full prompts for every request. — **Instead:** log template id, version and content hash; store full text only on failure or sampling.

## Checkpoint

1. Add an optional `examples` variable that renders a section only when non-empty, and confirm the delimiter appears exactly once.
2. Write a test that passes `context=None` and asserts a validation error is raised before any API call.
3. Log the `content_hash` on 20 calls and find two requests that produced identical prompts — then explain whether that is expected.

**Self-check:** the schema declares `context` as a bounded string with a default, and the template substitutes an explicit "(no context provided)" marker rather than rendering `None` or an empty gap.

## Go deeper

- [Prompt Versioning as Code](05-prompt-versioning.md) — what to do when a template changes.
- [Prompt-Injection Defence](06-injection-defense.md) — why delimiters belong to the template.
- [Few-Shot Prompting and Chain of Thought](02-few-shot-cot.md) — example blocks as a template section.
- [Validation](../03-data-engineering/07-validation.md) — the same discipline applied to data pipelines.

---
[← Previous: Few-Shot Prompting and Chain of Thought](02-few-shot-cot.md) · [Stage index](README.md) · [Next: Context Compression](04-context-compression.md)
