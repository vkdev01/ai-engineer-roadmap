# 2. Cleaning Messy Data

> **Stage 03 · Data Engineering** — turning a real-world export into something you can defend in a review
>
> **You'll be able to:** diagnose and fix duplicates, type drift, missing values and broken dates, and record every fix so the cleaning is reproducible.
>
> **Prerequisites:** [NumPy and Pandas](01-numpy-pandas.md) · [Probability and Bayes](../02-math-and-ml-basics/03-probability-bayes.md)

## Why this exists

Nobody hands you a clean CSV. You get a file where `phone` is sometimes a string and
sometimes a float, where one city is spelled four ways, where 12% of `price` is empty,
and where 300 rows are the same order entered twice. If you train or evaluate on that,
your model learns the mess.

The danger is not the mess itself — it is the *invisible* mess. Dropping rows with
missing values looks like a one-line fix and quietly deletes every customer from one
region, because that region's export never filled the field. Cleaning is a set of
decisions, and a decision you cannot see in the output is a decision nobody can audit.

Mental model: **cleaning is a pipeline of named, reversible steps, each one logging
how many rows it touched.**

## The mental model

```text
raw.csv
  |
  +-- 1. profile      count nulls, dtypes, duplicates   -> report, no changes
  +-- 2. normalise    strip, casefold, unify units      -> rows unchanged
  +-- 3. type-cast    ids to str, dates to datetime     -> bad rows flagged
  +-- 4. dedupe       exact then fuzzy key              -> rows removed (logged)
  +-- 5. missing      drop / fill / flag                -> decision recorded
  +-- 6. outliers     clamp or flag, never silently     -> values changed (logged)
  |
  v
clean.parquet + cleaning_log.json
```

## Core concepts

### Profile before you touch anything

A profile is a read-only summary: row count, dtype per column, null count, distinct
count, and a few quantiles. You run it first because every later decision depends on
it, and you keep it because it is the "before" in your before/after. `df.info()`,
`df.describe(include="all")` and `df.isna().sum()` cover most of it.

### Normalise text before comparing it

`"Bengaluru"`, `"bengaluru"`, `" Bengaluru "` and `"BANGALORE"` are one city and four
strings. Normalise with `.str.strip().str.lower()` and then map through an explicit
alias table. Do not let a fuzzy matcher decide your canonical spellings — write the
mapping down so the next person can argue with it.

### Cast types explicitly, and count what fails

`pd.to_numeric(s, errors="coerce")` turns junk into `NaN` instead of raising. That is
useful, but you must then count the `NaN`s it created: those are rows you could not
parse, and they need a decision, not silence. Same for `pd.to_datetime(..., errors="coerce")`.
Ids are always strings — an id with leading zeros is not a number.

### Dedupe on a key you chose, not on the whole row

Exact duplicates (`df.drop_duplicates()`) catch identical rows. The expensive ones are
near-duplicates: the same order submitted twice, 40 seconds apart, with a different
`updated_at`. Fix those by defining a natural key — `(customer_id, amount_inr, date)` —
sorting by `updated_at`, and keeping the last. Record how many rows that removed.

### Missing values are three different problems

Missing *at random* (a sensor glitch) can be dropped or imputed. Missing *by group*
(one region never fills `gstin`) must be kept and flagged, because dropping it deletes
a whole segment. Missing *by meaning* (`delivered_at` is null because the order is not
delivered) is information, not absence. Ask which one you have before choosing.

## Code

A cleaning function that returns the frame *and* a log of what it did. You should see a
dict with a count for each step, and a frame with no duplicate keys and no unparsed
dates.

```python
from __future__ import annotations

import pandas as pd

CITY_ALIASES = {"bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai"}


def clean_orders(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = raw.copy()
    log: dict[str, int] = {"rows_in": len(df)}

    # 1. Normalise text columns.
    df["city"] = (
        df["city"].astype("string").str.strip().str.lower()
        .replace(CITY_ALIASES)
    )

    # 2. Explicit casts; count what failed instead of raising.
    df["order_id"] = df["order_id"].astype("string").str.strip()
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    log["unparsed_amount"] = int(df["amount_inr"].isna().sum())
    log["unparsed_date"] = int(df["created_at"].isna().sum())

    # 3. Exact duplicates, then near-duplicates on a natural key.
    before = len(df)
    df = df.drop_duplicates()
    log["exact_dupes_removed"] = before - len(df)

    before = len(df)
    df = (
        df.sort_values("created_at")
          .drop_duplicates(subset=["customer_id", "amount_inr", "created_at"], keep="last")
    )
    log["near_dupes_removed"] = before - len(df)

    # 4. Keep rows with no amount but flag them; drop only unparseable dates.
    df["amount_missing"] = df["amount_inr"].isna()
    df = df.dropna(subset=["created_at", "order_id"])
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log
```

Same idea in TypeScript, for the case where the cleaning happens in a Node service
before records hit your API. This version rejects a batch instead of silently fixing
it, which is usually what you want at a service boundary.

```ts
import { z } from "zod";

const OrderSchema = z.object({
  orderId: z.string().min(1),
  city: z.string().transform((s) => s.trim().toLowerCase()),
  amountInr: z.coerce.number().nonnegative(),
  createdAt: z.coerce.date(),
});

export type Order = z.infer<typeof OrderSchema>;

export function cleanBatch(rows: unknown[]): { ok: Order[]; failed: number } {
  const ok: Order[] = [];
  let failed = 0;
  for (const row of rows) {
    const parsed = OrderSchema.safeParse(row);
    if (parsed.success) ok.push(parsed.data);
    else failed += 1; // count, never swallow
  }
  return { ok, failed };
}
```

## Real use case

A grocery delivery app imported its weekly order export into a demand-forecasting job.
The export contained a `city` column typed by three different internal tools, and the
forecast model treated `Bangalore`, `bangalore` and `BLR` as three separate markets —
so each appeared to have a third of its real demand and the model under-ordered in all
of them. Nothing errored; the numbers were just quietly low.

The fix was this file's shape: normalise and alias-map `city`, dedupe on
`(customer_id, amount_inr, created_at)` to remove the double-submitted orders that
inflated weekend demand, and write `cleaning_log.json` next to the output. The log is
what made the fix stick — the next time the import changed, `rows_out` dropped sharply
and the alert fired before the forecast ran.

## Common mistakes

- **Mistake:** `df.dropna()` as the first line — **Instead:** profile nulls by group first; drop only columns where missingness is genuinely random, and flag the rest.
- **Mistake:** fixing values by hand in the CSV — **Instead:** put the mapping in code (`CITY_ALIASES`) so the next run reproduces it.
- **Mistake:** `errors="coerce"` without counting — **Instead:** count the `NaN`s you created and log them; they are rows you failed to parse.
- **Mistake:** deduping on all columns — **Instead:** dedupe on a natural key plus a timestamp, keeping the last write.
- **Mistake:** cleaning in place with no record — **Instead:** return the frame and a log dict, and persist the log alongside the output.
- **Mistake:** treating ids as numbers — **Instead:** keep ids as strings; leading zeros and 19-digit ids both break under integer casting.

## Checkpoint

1. Write a `profile(df)` function that returns a DataFrame with one row per column: dtype, null count, distinct count, and the top value. Run it on any real CSV you can find.
2. Add near-duplicate detection to the code above using a one-day window instead of an exact timestamp match, and log how many rows it removes.
3. Take a column with 15% missing values, decide whether the missingness is random or by group using a groupby on another column, and write one paragraph defending your choice.

**Self-check:** a profile changes nothing and tells you dtype, null count, distinct count and the most common value per column — it is the "before" picture every later decision depends on.

## Go deeper

- [Exploratory Data Analysis](03-eda.md) — the charts that reveal group-level missingness.
- [Schema Validation with Pydantic](07-validation.md) — enforce these rules at a boundary instead of in a notebook.
- [Data Pipelines](06-data-pipelines.md) — run cleaning as a repeatable step, not a manual pass.
- [NumPy and Pandas](01-numpy-pandas.md) — the masks and casts this file builds on.

---
[← Previous: NumPy and Pandas](01-numpy-pandas.md) · [Stage index](README.md) · [Next: Exploratory Data Analysis](03-eda.md)
