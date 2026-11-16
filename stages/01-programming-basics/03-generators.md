# 3. Generators

> **Stage 01 · Programming Basics** — lets you walk over a million records using the memory of one.
>
> **You'll be able to:** replace a function that builds a huge list with a generator that yields one item at a time, and chain stages into a pipeline that never holds the whole dataset.
>
> **Prerequisites:** [Type Hints and Pydantic v2](02-type-hints-pydantic.md) · [Python Async](01-python-async.md)

## Why this exists

The naive data script reads a file into a list, transforms the list into another list, filters it into a third, then writes it out. For a thousand rows nobody notices. For a two-gigabyte JSONL export of documents, `f.readlines()` alone will exhaust the memory of a small cloud instance, and every intermediate list doubles the damage.

A generator is the fix: a function that produces values one at a time, on demand, keeping only the state it needs. The consumer pulls; the producer computes only when pulled.

The mental model: **a list is the whole answer in a box; a generator is a machine you turn by hand, one item per turn.**

## The mental model

```text
for row in pipeline("docs.jsonl"):        # pull one value
        |
        +-- filter_empty(row)             # pull one from below, maybe drop it
                |
                +-- parse_json(line)      # pull one line
                        |
                        +-- read_lines()  # one line from disk at a time

memory held at any moment: current line + current row
```

| | `list` | generator |
|---|---|---|
| Memory | all items at once | one item |
| Reusable | yes | no, exhausted after one pass |
| Speed to first result | after everything is computed | immediately |
| Length | `len(x)` | count as you consume |

## Core concepts

### `yield` turns a function into a factory of values

When Python sees `yield` in a function body, calling the function returns a generator object instead of running the body. Each `next()` runs until the next `yield` and pauses there, remembering local variables and position.

```python
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1
```

Nothing prints until you iterate. `list(countdown(3))` gives `[3, 2, 1]`.

### Generators are lazy, so errors are lazy too

A generator does not validate anything until the bad row is reached. That is a feature — you can process 900,000 good rows before hitting a malformed one — and a trap, because a `try` around the call will not catch it. Wrap the consumption loop, not the creation.

### Generator expressions and `itertools` compose cheaply

`(parse(l) for l in lines)` is a generator expression: same laziness, one line. `itertools.islice`, `chain`, `groupby`, `batched` (3.12+) give you windowing and chunking without materialising lists. For batching before an API call, `itertools.batched` is the clean answer; on 3.11 write a small chunker.

### Pipelines stay readable when each stage is a small generator

The pattern that scales: one generator per concern — read, parse, filter, transform — glued in a single `for` loop. Each stage is testable alone with `list(stage(input))`.

### `yield from` delegates to a sub-generator

`yield from inner()` forwards every value from `inner`, which keeps wrappers short and preserves laziness through layers.

### Async generators for streaming responses

`async def` + `yield` gives an async generator, consumed with `async for`. This is exactly how you read a streaming LLM response token by token without buffering the whole answer.

## Code

A three-stage pipeline over a JSONL file. The file never gets loaded whole; at most one line is in memory.

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:          # iterating a file yields one line at a time
            line = line.strip()
            if line:
                yield line


def parse_json(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    for number, line in enumerate(lines, start=1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Skip bad rows but keep the pipeline alive; log the line number.
            print(f"skip line {number}: not JSON")
            continue


def only_long(rows: Iterator[dict[str, Any]], min_chars: int = 200) -> Iterator[dict[str, Any]]:
    for row in rows:
        text = row.get("text")
        if isinstance(text, str) and len(text) >= min_chars:
            yield row


def pipeline(path: Path) -> Iterator[dict[str, Any]]:
    return only_long(parse_json(read_lines(path)))


if __name__ == "__main__":
    total = sum(1 for _ in pipeline(Path("docs.jsonl")))
    print(total)   # counts without ever building a list
```

Chunking for a batched API call, plus an async generator that streams a model response — the two generator patterns you will use most in AI work.

```python
from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Iterator
from typing import TypeVar

T = TypeVar("T")


def chunks(items: Iterator[T], size: int) -> Iterator[list[T]]:
    while batch := list(itertools.islice(items, size)):
        yield batch            # last batch may be shorter; loop ends on empty


async def stream_text(client: object, prompt: str) -> AsyncIterator[str]:
    # Shape matches the anthropic/openai streaming SDKs; check current docs
    # for the exact event names of the model you use.
    stream = await client.messages.create(  # type: ignore[attr-defined]
        model="<check-provider-docs>", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}], stream=True,
    )
    async for event in stream:
        if getattr(event, "type", "") == "content_block_delta":
            yield event.delta.text
```

## Real use case

An internal "chat with our docs" prototype indexed a 1.4 GB JSONL dump of exported wiki pages. The first ingestion script did `rows = [json.loads(l) for l in open(path)]`, embedded them in batches of 32, and was killed by the container's memory limit partway through — repeatedly, always around the same point, which made it look like one corrupt document.

Rewriting ingestion as the pipeline above changed the failure from "the process dies" to "the process streams". Peak memory dropped to a few hundred megabytes because only one batch of 32 documents was ever resident, and the run became resumable: the writer tracked the last completed document id, so a restart skipped everything already embedded. Malformed lines were counted and reported instead of crashing the job.

The design choice that mattered was laziness at every stage: read → parse → filter → chunk → embed all stayed generators, so the only place with a real list was the single batch handed to the embedding API.

## Common mistakes

- **Mistake:** returning a generator from a function and iterating it twice, getting nothing the second time. — **Instead:** generators are single-use; call the factory again or wrap a list if you truly need two passes.
- **Mistake:** putting the `try/except` around `pipeline(path)` instead of around the `for` loop. — **Instead:** exceptions surface during consumption, not creation.
- **Mistake:** `len(gen)` or `gen[0]`. — **Instead:** count by consuming (`sum(1 for _ in gen)`) or use `itertools.islice(gen, 1)`.
- **Mistake:** sorting a generator with `sorted(gen)`. — **Instead:** that materialises everything; sort on disk or keep the data ordered upstream.
- **Mistake:** holding a file handle open across a long-running generator and never closing it. — **Instead:** open inside the generator with `with`, so exhaustion or garbage collection closes it.

## Checkpoint

1. Turn `only_long` into a generator expression and confirm the output is identical.
2. Add a `write_jsonl` stage that consumes the pipeline and appends to an output file, keeping memory flat.
3. Write a `batched_embed` that takes a generator of texts, chunks it into 32s, calls your embedding function per chunk, and yields one vector at a time — then show peak memory stays flat as the input grows.

**Self-check:** `rows = (r for r in parse_json(read_lines(path)) if len(r.get("text", "")) >= 200)` — identical results, same laziness, less code.

## Go deeper

- [Python Async](01-python-async.md) — async generators extend this idea to streaming network data.
- [NumPy and Pandas](../03-data-engineering/01-numpy-pandas.md) — where laziness stops and columnar memory begins.
- [Parquet and JSONL](../03-data-engineering/05-parquet-jsonl.md) — the file formats your pipelines will actually stream.

---
[← Previous: Type Hints and Pydantic v2](02-type-hints-pydantic.md) · [Stage index](README.md) · [Next: Core Algorithms](04-algorithms.md)
