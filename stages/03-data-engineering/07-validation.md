# 7. Schema Validation with Pydantic

> **Stage 03 · Data Engineering** — the boundary where malformed records stop
>
> **You'll be able to:** define a pydantic v2 model for a dataset record, validate a whole file against it, and report every failure with its line number.
>
> **Prerequisites:** [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md) · [Cleaning Messy Data](02-data-cleaning.md)

## Why this exists

A `dict` is a promise nobody keeps. `record["messages"]` is fine until the day a
record has `message` instead of `messages`, and then your training job fails forty
minutes in, on record 31,204, with a `KeyError` that tells you nothing about which
file or which upstream change caused it.

Validation moves that failure to the boundary, before the data is stored, and turns
it into a message that names the field, the value and the reason. Pydantic does it by
declaring the shape of a valid record as a Python class: you write the types once, and
pydantic coerces, checks and reports.

Mental model: **a pydantic model is an executable schema — the docstring and the
check are the same object.**

## The mental model

```text
raw dict / JSON line
        |
        v
  +-------------------+        valid        +-----------------+
  |  Model.model_      |  ----------------> | typed instance  |  -> store / train
  |  validate_json()  |                    +-----------------+
  +---------+---------+
            | invalid
            v
  +--------------------------------------------+
  | ValidationError                            |
  |  loc: ("messages", 1, "role")              |
  |  msg: "Input should be 'user' or ..."      |
  |  input: "human"                            |
  +--------------------------------------------+
            |
            v
   count it, log the line number, decide:
   reject the file  |  quarantine the record
```

## Core concepts

### Declare the shape, let pydantic coerce

Field types are constraints. `str` rejects `None`; `int` accepts `"42"` and coerces it;
`Literal["user", "assistant"]` rejects `"human"`. `Field(min_length=1)` rejects the
empty string. You are writing the schema once, in a form that both type checkers and
the runtime understand.

### Validators encode the rules a type cannot express

`@field_validator` runs after the type check, for things like "content must not be
whitespace only". `@model_validator(mode="after")` sees the whole record, which is
where cross-field rules live: the last message must be from the assistant, the
`messages` list must alternate roles, `end_id` must be greater than `start_id`.

### Validate a stream line by line, and count

`model_validate_json(line)` parses and checks one line. Wrap it in a loop over the
file, catch `ValidationError`, and record the line number. A file where 3 of 10,000
lines fail is a data problem you can quarantine; a file where 3,000 fail is a broken
upstream and you should stop. Report both numbers — the counts are the signal.

### Strict where it matters, lax where it helps

Pydantic v2 is lax by default (`"42"` becomes `42`). Use `model_config =
ConfigDict(strict=True)` on records you generate yourself, where a wrong type means a
bug. Keep lax mode on records you ingest from other people's exports, where a number
arriving as a string is normal. Choosing per model is a real design decision, not a
formality.

### The model is documentation the next engineer can run

A well-written model with `Field(description=...)` is the data card for your dataset.
When the upstream team asks "what do you expect?", you send the class. That is worth
more than any README paragraph.

## Code

A model for a fine-tuning record with cross-field rules, then a validator that reads a
JSONL file and reports per-line failures. You should see one line per bad record with
its line number, then a summary count.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1, description="Non-empty message text")


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")   # unknown keys are a schema drift signal

    id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=2)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def last_is_assistant(self) -> "Example":
        if self.messages[-1].role != "assistant":
            raise ValueError("last message must be from the assistant")
        return self


def validate_jsonl(path: Path) -> tuple[list[Example], list[tuple[int, str]]]:
    good: list[Example] = []
    bad: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                good.append(Example.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                bad.append((lineno, str(exc).splitlines()[0]))
    return good, bad


if __name__ == "__main__":
    ok, failures = validate_jsonl(Path("train.jsonl"))
    print(f"valid: {len(ok)}  invalid: {len(failures)}")
    for lineno, msg in failures[:5]:
        print(f"  line {lineno}: {msg}")
    # A high failure rate means the upstream changed; do not train on a fraction.
    rate = len(failures) / max(1, len(ok) + len(failures))
    if rate > 0.01:
        raise SystemExit(f"failure rate {rate:.1%} is too high — upstream schema drift")
```

The same contract on the TypeScript side, for the small CLI that inspects a dataset
before upload. `zod` gives the identical shape: parse, catch, report the line.

```ts
import { readFile } from "node:fs/promises";
import { z } from "zod";

const Message = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().min(1),
});

const Example = z
  .object({ id: z.string().min(1), messages: z.array(Message).min(2) })
  .strict()
  .refine((e) => e.messages[e.messages.length - 1].role === "assistant", {
    message: "last message must be from the assistant",
  });

export async function check(path: string): Promise<{ ok: number; bad: string[] }> {
  const lines = (await readFile(path, "utf8")).split("\n").filter((l) => l.trim() !== "");
  const bad: string[] = [];
  let ok = 0;
  lines.forEach((line, i) => {
    const parsed = Example.safeParse(JSON.parse(line));
    if (parsed.success) ok += 1;
    else bad.push(`line ${i + 1}: ${parsed.error.issues[0].message}`);
  });
  return { ok, bad };
}
```

## Real use case

A team fine-tuned a support-reply model on an export from their ticketing system. The
first run produced a model that sometimes answered in the wrong language. The cause
was upstream: the export had begun including internal agent notes in the assistant
turn, and those notes were written in whatever language the agent preferred.

A pydantic model would have caught it in one line — the team added a
`Literal["customer_visible"]` field to the source record, plus a validator that
rejects any message whose `visibility` is not that value. Validation now runs as a
step in the pipeline, before the file is written, and prints a per-line failure list.
The failure count is also an alert: if it goes above a small threshold, the export
schema has drifted and the run stops instead of training on a mixed corpus.

## Common mistakes

- **Mistake:** validating after the write — **Instead:** validate before the record reaches storage; a bad row in Parquet is a bad row forever.
- **Mistake:** `except Exception: pass` around validation — **Instead:** count failures and log the line number; silence is the failure mode you are trying to remove.
- **Mistake:** one model for ingestion and generation — **Instead:** lax mode for other people's exports, strict mode for records you produce.
- **Mistake:** no `extra="forbid"` — **Instead:** forbid extra keys so upstream drift becomes visible instead of silently dropped.
- **Mistake:** failing on the first bad record — **Instead:** validate all, report the count, then decide whether to reject the file or quarantine the rows.

## Checkpoint

1. Write a model for a record with `id`, `text`, and `label` where `label` is one of three fixed values. Validate a list of ten dicts, three of them bad, and print every failure.
2. Add a cross-field rule to your model (for example, `text` must be longer than `label`), then confirm it fires on a record where both fields are individually valid.
3. Point your validator at a real JSONL file and report the valid/invalid split. Decide and justify a threshold at which the pipeline should stop instead of quarantine.

**Self-check:** the three bad dicts each produce a `ValidationError` whose `loc` names the failing field, and validation continues to the next record rather than stopping.

## Go deeper

- [Dataset Preparation](08-dataset-prep.md) — validation is the gate before dedup and splits.
- [Data Pipelines](06-data-pipelines.md) — where the validation step belongs in a run.
- [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md) — the language features underneath.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — the same models validating model output instead of file input.

---
[← Previous: Data Pipelines](06-data-pipelines.md) · [Stage index](README.md) · [Next: Dataset Preparation](08-dataset-prep.md)
