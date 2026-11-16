# 2. Type Hints and Pydantic v2

> **Stage 01 · Programming Basics** — turns loose dictionaries into named, validated objects so bad data fails at the door.
>
> **You'll be able to:** define a pydantic v2 model for an API payload, parse untrusted JSON into it, and get a readable error naming the exact field that was wrong.
>
> **Prerequisites:** [Python Async](01-python-async.md) · [JSON and YAML](08-json-yaml.md)

## Why this exists

Most production bugs in AI applications are shape bugs, not logic bugs. A model returns `{"score": "0.87"}` where you expected a float, or the field is `null`, or the key is `confidence` today and `score` next week. Nothing crashes at the boundary; the bad value travels four functions deep and then breaks something that has nothing to do with the cause. Debugging it means printing dictionaries.

Type hints are for humans and tools: they let your editor and `mypy` see the shape of a value before you run anything. Pydantic goes further and enforces that shape at runtime, converting "0.87" to `0.87` where it safely can and refusing where it cannot.

The mental model: **a pydantic model is a boundary object — you hand it anything, and it either gives you a trusted object or an error that says which field lied.**

## The mental model

```text
untrusted input                boundary                  inside your program
{"score": "0.87",       ->  Model.model_validate(...)  ->  obj.score: float
 "label": "billing"}          |
                              +-- wrong type/range --> ValidationError
                                                          (field, message, input)
```

Two layers of typing, two jobs:

| Layer | Checked when | Catches |
|---|---|---|
| Type hints + `mypy` | before you run | wrong argument, typo'd attribute |
| Pydantic models | at runtime | wrong JSON, missing field, bad range |

## Core concepts

### Annotations are documentation the machine can read

`def embed(text: str, dim: int = 768) -> list[float]:` tells a reader the contract in one line. Python does not enforce it, but `mypy` does, and it will catch `embed(123)` before your test suite does. Add `from __future__ import annotations` so annotations are never evaluated at import time.

### `BaseModel` + field types = validation

A pydantic model declares fields with types. `Model.model_validate(raw)` parses a dict (or JSON string) and raises `ValidationError` if anything is off. In v2 the parsing core is Rust, so this is fast enough to run on every request.

### Field constraints beat hand-written checks

`Field(ge=0.0, le=1.0)`, `min_length`, `max_length`, `pattern` put the rule in the declaration. `Annotated[float, Field(ge=0, le=1)]` is the modern spelling and keeps the type readable.

### Validators handle what types cannot express

`@field_validator("label")` for one field, `@model_validator(mode="after")` for cross-field rules — "if `status` is `done`, `finished_at` must be set". Validators run after type coercion, so you are working with parsed values.

### `model_dump` / `model_dump_json` are the way out

Never hand-roll `json.dumps(obj.__dict__)`. `model_dump()` gives a plain dict, `model_dump_json()` gives a JSON string that respects your field aliases and serialisers — which is what you send back over HTTP.

### Strictness where it matters

`model_config = ConfigDict(extra="forbid", strict=True)` rejects unknown keys and refuses "5" → 5 coercion. Use it for config files and internal contracts; stay lenient for third-party payloads you do not control.

## Code

A model for the kind of structured output an LLM returns, with constraints and a cross-field rule.

```python
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TicketTag(BaseModel):
    # Unknown keys are an error: we want to notice when a provider adds fields.
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(pattern=r"^T-\d{4,}$")
    label: Literal["billing", "bug", "feature", "account"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def low_confidence_needs_rationale(self) -> TicketTag:
        # A weak tag is only acceptable if the model explained itself.
        if self.confidence < 0.5 and len(self.rationale) < 20:
            raise ValueError("confidence below 0.5 needs a rationale of 20+ chars")
        return self


raw = {"ticket_id": "T-1042", "label": "billing", "confidence": "0.87",
       "rationale": "Invoice ID mentioned twice in the body."}

tag = TicketTag.model_validate(raw)   # "0.87" is coerced to 0.87
print(tag.confidence, type(tag.confidence))   # 0.87 <class 'float'>
print(tag.model_dump_json())

try:
    TicketTag.model_validate({**raw, "confidence": 1.4})
except ValidationError as exc:
    print(exc.errors()[0]["loc"], exc.errors()[0]["msg"])  # ('confidence', ...)
```

Validating a whole batch, keeping the good rows and reporting the bad ones — the pattern you will use on every dataset and every model response.

```python
from __future__ import annotations

from pydantic import BaseModel, ValidationError


class Row(BaseModel):
    id: str
    text: str
    score: float


def parse_rows(raw_rows: list[dict[str, object]]) -> tuple[list[Row], list[dict[str, object]]]:
    good: list[Row] = []
    bad: list[dict[str, object]] = []
    for index, raw in enumerate(raw_rows):
        try:
            good.append(Row.model_validate(raw))
        except ValidationError as exc:
            # Keep the position so the caller can point at the source line.
            bad.append({"index": index, "errors": exc.errors(include_url=False)})
    return good, bad
```

The same contract on the frontend, so a TypeScript consumer and a Python producer cannot drift apart silently:

```ts
import { z } from "zod";

export const TicketTagSchema = z.object({
  ticket_id: z.string().regex(/^T-\d{4,}$/),
  label: z.enum(["billing", "bug", "feature", "account"]),
  confidence: z.number().min(0).max(1),
  rationale: z.string().min(1).max(400),
});

export type TicketTag = z.infer<typeof TicketTagSchema>;

export function parseTags(body: unknown): TicketTag[] {
  return z.array(TicketTagSchema).parse(body); // throws with a path to the bad field
}
```

## Real use case

A document-intelligence product extracts `{vendor, invoice_date, total, currency, line_items}` from scanned invoices using an LLM. Early on, the extraction prompt returned free-form JSON that was parsed with `json.loads` and read with `.get()`. Two failures reached customers: a total of `"1,250.00"` (a string with a thousands separator) silently became `0.0` when cast, and a missing `currency` became `None` and printed as `None INR` on the PDF.

The fix moved every read behind a `Invoice` model: `total: Decimal` with a validator that strips separators, `currency: Literal["INR","USD","EUR"]` with no default, and `line_items` summed and compared to `total` in a model validator, raising if they disagree by more than a rupee. Extraction failures now land in a rejected queue with the offending field named, the model gets re-asked with the error text, and nothing invalid reaches the renderer. The failure mode changed from "wrong invoice sent to a customer" to "this invoice needs a second look".

## Common mistakes

- **Mistake:** typing a parameter `dict` and reading keys inside. — **Instead:** define a model at the boundary and pass the model.
- **Mistake:** giving a required field a default of `None` to silence errors. — **Instead:** if the field is required, leave it required; let the error tell you the upstream is broken.
- **Mistake:** using `extra="ignore"` everywhere. — **Instead:** use `forbid` on contracts you own so provider changes surface early.
- **Mistake:** catching `ValidationError` and re-raising `ValueError("bad input")`. — **Instead:** log `exc.errors()` — the field path is the whole value of the error.
- **Mistake:** mutating a model after construction. — **Instead:** models are meant to be treated as immutable; build a new one with `model_copy(update={...})`.

## Checkpoint

1. Add a `tags: list[str]` field capped at five items and prove that a sixth item raises.
2. Write a `model_validator` that rejects any `total` that is negative or has more than two decimal places.
3. Take a real JSON response from any API you have access to and model it with `extra="forbid"`; every `ValidationError` you hit is a field you would have guessed wrong.

**Self-check:** `tags: list[str] = Field(default_factory=list, max_length=5)` — `max_length` on a list bounds the item count in pydantic v2.

## Go deeper

- [JSON and YAML](08-json-yaml.md) — the formats your models parse and serialise.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — how to make a model produce JSON that fits a schema like this one.
- [Data Validation](../03-data-engineering/07-validation.md) — the same discipline applied to whole datasets.

---
[← Previous: Python Async](01-python-async.md) · [Stage index](README.md) · [Next: Generators](03-generators.md)
