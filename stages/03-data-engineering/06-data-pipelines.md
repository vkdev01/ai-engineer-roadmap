# 6. Data Pipelines

> **Stage 03 · Data Engineering** — making a data job you can run twice without fear
>
> **You'll be able to:** write an idempotent, incremental pipeline that survives a crash and re-run, and explain why each of those words matters.
>
> **Prerequisites:** [Parquet and JSONL](05-parquet-jsonl.md) · [Generators](../01-programming-basics/03-generators.md)

## Why this exists

A notebook that builds a dataset works exactly once. Run it again tomorrow and you
have appended a second copy of everything; run it after a crash and you cannot tell
which rows made it. When someone asks "is this file from before or after the fix?"
you have no answer, because the job has no memory and no contract.

A pipeline is the same transformation with three promises added: running it twice
gives the same result, running it after a failure resumes instead of restarting, and
each run records what it read and wrote. Those promises are what let a data job be
scheduled, retried and trusted.

Mental model: **a pipeline is a function `outputs = f(inputs, state)`, where the state
is what makes re-running safe.**

## The mental model

```text
  source            stage 1            stage 2           stage 3
  raw.csv  ---->  extract      ---->  transform  ---->  load
                  (read + hash)      (clean, cast)     (write partition)
                      |                  |                  |
                      v                  v                  v
                 +----------------------------------------------+
                 |  state / manifest:  run_id, input_hash,      |
                 |  rows_in, rows_out, watermark, status        |
                 +----------------------------------------------+
                                 |
        re-run with same input_hash -> stage skipped (idempotent)
        re-run after crash          -> resumes from last watermark
        new day's data              -> only new rows processed (incremental)
```

## Core concepts

### Idempotent: same input, same output, no duplicates

A step is idempotent if running it twice equals running it once. The reliable way to
get there is **write-by-replacement, not by appending**: compute the output partition
for a given date, write it to a temporary path, then atomically replace the target.
If the job dies mid-write, the old partition is still intact. Appending to a live file
is the opposite and is where double-counted rows come from.

### Incremental: process what is new, keyed by a watermark

Re-processing 40 million rows to add yesterday's 20,000 is waste. Keep a
**watermark** — the highest `created_at` or the highest id you have successfully
loaded — and read only rows beyond it. The watermark must be committed *after* the
write succeeds, in the same transaction or manifest update, or a crash between the two
loses data.

### Deterministic: sort and fix the clock

Two runs that produce different files are not idempotent even if the rows match. Sort
by a stable key before writing, avoid `now()` inside the transform, and pass the run
date as a parameter. Then two runs on the same input hash produce byte-identical
output, and you can diff them.

### The manifest is the pipeline's memory

A small JSON file per run — run id, input path and hash, row counts, output paths,
status, duration — costs almost nothing and answers every "what happened" question.
It is also what lets a scheduled job skip work: if `input_hash` matches the last
successful run, there is nothing to do.

### Fail loudly, resume cleanly

A pipeline should stop on a schema surprise rather than write a partially wrong
partition. Validate before write (see [Schema Validation with Pydantic](07-validation.md)),
raise, and leave the previous partition untouched. Then a retry is safe, because
nothing was half-written.

## Code

A minimal pipeline with a manifest, a watermark and an atomic write. Run it twice: the
second run should print `skipped` and change nothing.

```python
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("lake")
STATE = ROOT / "_state" / "manifest.json"


def input_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_state() -> dict[str, object]:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def run(run_date: date) -> None:
    src = Path("raw") / f"orders_{run_date.isoformat()}.csv"
    target = ROOT / "orders" / f"dt={run_date.isoformat()}"
    h = input_hash(src)
    state = load_state()

    # Idempotency gate: identical input already loaded -> do nothing.
    if state.get("input_hash") == h and state.get("status") == "ok":
        print("skipped: input unchanged")
        return

    df = pd.read_csv(src, dtype={"order_id": "string"})
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df = df.dropna(subset=["order_id"]).sort_values("order_id")  # deterministic order

    # Atomic replace: write to a temp dir, then swap. A crash leaves the old one.
    tmp = target.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    df.to_parquet(tmp, index=False)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "run_date": run_date.isoformat(),
        "input_hash": h,
        "rows_in": int(len(df)),
        "watermark": str(df["order_id"].max()),   # committed after the write
        "status": "ok",
    }, indent=2))
    print(f"wrote {len(df)} rows to {target}")


if __name__ == "__main__":
    run(date(2025, 3, 1))
```

The incremental half: read only rows past the watermark, so a daily run touches only
new data. This is the shape you want when the source is a database rather than a file.

```python
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg://app:secret@localhost:5432/app")


def fetch_since(watermark: datetime | None, limit: int = 50_000) -> pd.DataFrame:
    # Keyset pagination on (created_at, id): stable under concurrent inserts,
    # unlike OFFSET, which can skip or repeat rows when the table changes.
    query = text("""
        SELECT id, customer_id, amount_inr, created_at
        FROM orders
        WHERE (:wm IS NULL)
           OR (created_at, id) > (:wm, :wm_id)
        ORDER BY created_at, id
        LIMIT :lim
    """)
    with engine.connect() as conn:
        return pd.read_sql(
            query, conn,
            params={"wm": watermark, "wm_id": 0, "lim": limit},
        )


batch = fetch_since(None)
print(len(batch), batch["created_at"].max() if len(batch) else None)
```

## Real use case

A retail analytics team ran a nightly notebook that appended the day's orders to a
single `all_orders.parquet`. One night the job crashed halfway through the write, and
because Parquet writes are not append-safe at that size, the file was unreadable — and
because there was no backup of the previous version, a week of dashboards broke.

The rewrite is the code above: one partition per date, written to a temp directory and
swapped in atomically, with a manifest recording the input hash and row count. The next
crash cost one day's partition, which the retry rebuilt. The manifest also gave the team
something they had never had — the ability to answer "did last night's run actually
include the new customer table?" without re-running anything.

| Before | After |
|---|---|
| Append to one large file | One partition per date, atomic swap |
| Crash = corrupted file | Crash = yesterday's partition intact |
| Re-run duplicates rows | `input_hash` gate skips or replaces |
| No record of what ran | Manifest per run |

## Common mistakes

- **Mistake:** appending to the output on every run — **Instead:** write the partition to a temp path and `os.replace` it into place.
- **Mistake:** using `now()` inside the transform — **Instead:** pass the run date in; otherwise two runs can never be identical.
- **Mistake:** committing the watermark before the write — **Instead:** update the watermark only after the write succeeds, so a crash re-reads rather than loses rows.
- **Mistake:** `OFFSET`-based pagination over a changing table — **Instead:** use keyset pagination on `(created_at, id)`.
- **Mistake:** catching exceptions and writing partial output — **Instead:** validate, raise, and leave the previous partition untouched.
- **Mistake:** unsorted writes — **Instead:** sort by a stable key so runs are diffable.

## Checkpoint

1. Run the pipeline twice on the same input and confirm the second run prints `skipped` and the output file's modification time does not change.
2. Add an incremental mode: read only rows with `created_at` greater than the manifest's watermark, and prove with a row count that yesterday's rows were not re-read.
3. Break it on purpose: delete half of a partition mid-write (or raise just before the swap) and confirm the previous partition is still readable, then re-run and confirm recovery.

**Self-check:** the second run skips because the input hash matches the last successful run's hash and status is `ok` — nothing to recompute, nothing to rewrite.

## Go deeper

- [Schema Validation with Pydantic](07-validation.md) — the check that goes before the write.
- [Dataset Preparation](08-dataset-prep.md) — the transformation a pipeline usually wraps.
- [MLflow and Weights & Biases](../18-mlops-llmops/01-mlflow-wandb.md) — tracking runs and artifacts in a larger system.
- [DVC](../18-mlops-llmops/04-dvc.md) — versioning the data a pipeline produces.

---
[← Previous: Parquet and JSONL](05-parquet-jsonl.md) · [Stage index](README.md) · [Next: Schema Validation with Pydantic](07-validation.md)
