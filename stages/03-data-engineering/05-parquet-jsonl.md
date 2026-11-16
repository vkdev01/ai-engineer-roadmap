# 5. Parquet and JSONL

> **Stage 03 · Data Engineering** — the two file formats an AI engineer lives in: one for analysis, one for models
>
> **You'll be able to:** choose between Parquet and JSONL for a given job, convert between them safely, and read either without loading it all into memory.
>
> **Prerequisites:** [NumPy and Pandas](01-numpy-pandas.md) · [JSON and YAML](../01-programming-basics/08-json-yaml.md)

## Why this exists

CSV is a text format designed for a spreadsheet, and it is a poor fit for everything
else. It has no types, so a reader has to guess whether `007` is a number. It stores
values row by row, so a query for one column still reads every byte of every column.
It has no compression story beyond gzip over the whole file.

Two formats replace it, for two different jobs. **Parquet** is columnar and typed: it
is what you use for analytics, for anything a query engine will scan, and for large
datasets you want to store compactly. **JSONL** (JSON Lines) is one JSON object per
line: it is what you use for LLM training and eval data, because each record is a
self-contained conversation, it streams one record at a time, and appending a record
never rewrites the file.

Mental model: **Parquet is a table on disk; JSONL is a list of records on disk.**

## The mental model

```text
Parquet (columnar)                    JSONL (line-delimited)
+--------+--------+--------+          {"id": 1, "text": "hi", "label": "ok"}
| id     | 1 2 3  |                {"id": 2, "text": "yo", "label": "bad"}
+--------+--------+--------+          {"id": 3, "text": "hm", "label": "ok"}
| text   | a b c  |
+--------+--------+--------+          read a line -> one record
| label  | ok ok  |                  append   -> one line
+--------+--------+--------+
                                      nested shapes are natural:
read "label" only                     {"messages": [{"role": "user", ...},
  -> touches only the label             {"role": "assistant", ...}]}
     column's bytes

row group 0 | row group 1 | ...       no schema, no types, no index
 each holds N rows of all columns     validate every record yourself
```

## Core concepts

### Columnar layout is why Parquet is fast

Parquet stores all values of one column together, in row groups. A query for
`label` reads only the `label` bytes. Values in a column share a type, so they
compress far better than interleaved rows. Each row group stores min/max statistics
per column, so a reader can skip a whole group without decompressing it.

### Types are declared, not guessed

A Parquet file carries a schema: `id` is `int64`, `created_at` is a timestamp,
`label` is a string. That schema is what makes round-tripping safe — no more
`007` becoming `7`. Pandas, Polars, DuckDB, Spark and Arrow all read the same file
and agree on the types.

### JSONL is the format of LLM data

Every fine-tuning and eval file you will meet is JSONL. The reason is structural:
a training example is a nested object (messages, tools, metadata), records are
processed one at a time, and a run that crashes halfway should not lose the records
already written. Line-delimited JSON gives all three for free. The cost is that a
single malformed line breaks a naive reader, so you validate every line.

### Read in chunks or by column, never all at once

`pd.read_parquet(path, columns=[...])` reads one column. `pd.read_json(path,
lines=True, chunksize=10_000)` streams JSONL in batches. Both let you process a file
larger than memory, which is the normal case once a dataset passes a few gigabytes.

### Convert deliberately and keep the record count

Parquet and JSONL are not interchangeable: Parquet holds a table, JSONL holds nested
records. Convert a table to JSONL by `to_dict(orient="records")` per chunk; convert
back by reading line by line. Always assert the record count on both sides — a
conversion that loses records silently is the classic bug.

## Code

Write a Parquet file with partitioning, then read back only one column. You should see
a directory of files and a Series whose dtype is the declared type, not a guess.

```python
from __future__ import annotations

import pandas as pd

df = pd.DataFrame({
    "doc_id": [f"d{i}" for i in range(10_000)],
    "source": ["web", "pdf"] * 5_000,
    "n_tokens": range(10_000),
})

# Partition by source: one directory per value, so a filtered read skips the rest.
df.to_parquet("lake/documents", engine="pyarrow", partition_cols=["source"],
              compression="zstd", index=False)

# Column pruning: only n_tokens bytes are read.
tokens = pd.read_parquet("lake/documents", columns=["n_tokens"])
print(len(tokens), tokens.dtypes.to_dict())   # 10000 {'n_tokens': dtype('int64')}

# Predicate pushdown: only rows with n_tokens > 9000 are materialised.
big = pd.read_parquet("lake/documents", filters=[("n_tokens", ">", 9_000)])
print(len(big))
```

JSONL for a fine-tuning file, streamed in chunks. Each line is one training example
with a `messages` list, the shape the chat fine-tuning APIs expect.

```python
from __future__ import annotations

import json
from pathlib import Path

rows = [
    {"id": "e1", "messages": [
        {"role": "user", "content": "What is an index?"},
        {"role": "assistant", "content": "A sorted side structure that speeds lookups."},
    ]},
    {"id": "e2", "messages": [
        {"role": "user", "content": "What is a CTE?"},
        {"role": "assistant", "content": "A named intermediate result inside a query."},
    ]},
]

out = Path("train.jsonl")
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")   # one record, one line

# Stream it back; never json.load() the whole file.
n = 0
with out.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)          # a bad line raises here, loudly
        assert rec["messages"][-1]["role"] == "assistant"
        n += 1
print("records:", n, "bytes:", out.stat().st_size)
```

TypeScript, because the JSONL dataset usually ends up in a Node tool or a small
validator in CI. `zod` checks the nested shape line by line.

```ts
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { z } from "zod";

const Message = z.object({ role: z.enum(["system", "user", "assistant"]), content: z.string().min(1) });
const Example = z.object({ id: z.string(), messages: z.array(Message).min(2) });

export async function validateJsonl(path: string): Promise<{ ok: number; bad: number }> {
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let ok = 0;
  let bad = 0;
  for await (const line of rl) {
    if (line.trim() === "") continue;
    const parsed = Example.safeParse(JSON.parse(line));
    if (parsed.success) ok += 1;
    else bad += 1; // report the line number in real code
  }
  return { ok, bad };
}
```

## Real use case

A document-intelligence team stored its extracted corpus as one CSV per month,
gzipped. Rebuilding a retrieval index meant reading every file and every column,
because a CSV read cannot skip a column — several hours per rebuild, and the pipeline
had to hold the whole corpus in memory.

Moving to Parquet partitioned by `source_month` changed the shape of the job: the
index rebuild now reads only `doc_id` and `text`, skips months that have not changed
by checking row-group statistics, and streams. The same team kept JSONL for the eval
set, because each eval record carries a nested `messages` list and a `metadata`
object, and those do not fit a flat table without awkward JSON-in-a-column.

| Need | Format | Why |
|---|---|---|
| Scan 40M rows for 2 columns | Parquet | Column pruning, compression |
| Monthly partition rebuild | Parquet | Skip unchanged partitions |
| 5,000 nested eval examples | JSONL | One self-contained record per line |
| Append during a running job | JSONL | Append never rewrites the file |

## Common mistakes

- **Mistake:** storing training data as CSV to "keep it readable" — **Instead:** use JSONL; it is equally readable, streams, and holds nested records.
- **Mistake:** `json.load(open("train.jsonl"))` — **Instead:** iterate line by line with `json.loads`; a whole-file load fails and exhausts memory on real datasets.
- **Mistake:** writing thousands of tiny Parquet files — **Instead:** target files in the hundreds-of-MB range; small files destroy scan performance.
- **Mistake:** converting formats without asserting the record count — **Instead:** count records before and after and fail on a mismatch.
- **Mistake:** one giant Parquet file with no partitioning — **Instead:** partition on the column you filter by (date, source) so reads can skip.
- **Mistake:** trusting every JSONL line — **Instead:** validate each line with pydantic or zod and report the line number of any failure.

## Checkpoint

1. Take any CSV, write it to Parquet, read back two columns, and compare dtypes to the original. Note where the CSV had lost type information.
2. Convert a Parquet table to JSONL in chunks of 1,000 records and assert the counts match. Then corrupt one line and confirm your reader reports it rather than crashing silently.
3. Partition a dataset by month, then time a read of a single month against a read of the unpartitioned file.

**Self-check:** a column-only Parquet read is faster because Parquet stores values column by column, so reading `n_tokens` never touches the `source` or `doc_id` bytes.

## Go deeper

- [Data Pipelines](06-data-pipelines.md) — writing these files as idempotent pipeline outputs.
- [Dataset Preparation](08-dataset-prep.md) — JSONL is the format your splits will be saved in.
- [Schema Validation with Pydantic](07-validation.md) — validating records before they are written.
- [SQL on PostgreSQL](04-sql-postgresql.md) — moving between a table and a file.

---
[← Previous: SQL on PostgreSQL](04-sql-postgresql.md) · [Stage index](README.md) · [Next: Data Pipelines](06-data-pipelines.md)
