# 1. NumPy and Pandas

> **Stage 03 · Data Engineering** — the two libraries you will use to touch every dataset in this roadmap
>
> **You'll be able to:** load tabular data, reshape arrays without guessing at shapes, and answer a question about a dataset in four lines of Pandas.
>
> **Prerequisites:** [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md) · [Generators](../01-programming-basics/03-generators.md)

## Why this exists

Plain Python is fine for a hundred rows and hopeless for a million. A list of lists
stores every number as a separate object with its own type tag, so a loop over
500,000 rows of two columns does 500,000 interpreter steps. NumPy stores the same
numbers as one contiguous block of raw bytes and pushes the loop into compiled C.
That single change is why data work happens in arrays and not in lists.

Pandas sits on top of NumPy and adds the thing NumPy lacks: **labels**. An array
knows position, not meaning. A DataFrame knows that column `price_inr` exists, that
row 41 is order `A-1183`, and how to join two tables on a shared key.

The mental model to carry: **NumPy is a typed block of numbers with a shape; a
DataFrame is a dict of NumPy arrays that share one row index.**

## The mental model

```text
numpy array (2-D)                pandas DataFrame
  shape (3, 4)                     index   a      b     c
  dtype float64                    ---------------------------
  +----+----+----+----+            r0   1.0   2.0   3.0
  |  1 |  2 |  3 |  4 |            r1   4.0   5.0   6.0
  +----+----+----+----+            r2   7.0   8.0   9.0
  |  4 |  5 |  6 |  7 |
  +----+----+----+----+          df["a"].to_numpy()
  |  7 |  8 |  9 | 10 |          -> array([1., 4., 7.])
  +----+----+----+----+

axis 0 = down the rows     axis 1 = across the columns
reduce over axis 0 -> one value per column
```

## Core concepts

### A shape is (rows, columns) — read it out loud

Every NumPy operation has a shape you can predict before you run it. `(1000, 3)`
means a thousand rows and three columns. Indexing with `arr[i]` drops axis 0, so
`arr[0].shape` is `(3,)`. Slicing with `arr[:, 1]` keeps it flat, `arr[:, 1:2]` keeps
it 2-D as `(1000, 1)`. Most shape bugs are someone assuming one and getting the other.

### Vectorise instead of looping

A vectorised expression describes the whole array at once. `prices * 1.18` adds tax
to every row with no loop, and it is both shorter and faster than the `for` version.
The rule: if you are writing a `for` loop over array elements, there is almost always
an array expression that replaces it.

### Boolean masks select rows

A comparison on a column returns an array of `True`/`False`, and indexing with it
keeps only the `True` rows. Combine masks with `&` and `|`, wrapped in parentheses —
`df[(df.a > 1) & (df.b < 9)]`. This is the single most used Pandas idiom.

### Groupby is split, apply, combine

`df.groupby("city")["amount"].sum()` splits rows by city, sums each group, and
stitches the results back into a Series indexed by city. Everything else in Pandas
analysis is a variation of this: `.agg()` for several statistics, `.transform()` when
you want the group result broadcast back to the original rows.

### Joins need matching keys and matching meaning

`pd.merge` mirrors SQL joins. Check three things before you merge: do the key columns
have the same dtype (an `int64` id will not match an `object` id), is the key unique
on the side you assume is unique, and how many rows do you expect back. A merge that
grows your row count silently is the most common data bug in production.

## Code

Load a CSV, inspect it, and answer a question with a mask, a groupby and a merge.
You should see one printed shape, then a small summary table with one row per city.

```python
from __future__ import annotations

import numpy as np
import pandas as pd

orders = pd.read_csv("orders.csv", parse_dates=["created_at"])
print(orders.shape)          # (5000, 5) — rows, columns

# Cost per row, vectorised: no loop over 5000 rows.
orders["amount_inr"] = orders["qty"] * orders["unit_price_inr"]

# Boolean mask: keep only paid orders created this year.
mask = (orders["status"] == "paid") & (orders["created_at"].dt.year == 2025)
paid = orders.loc[mask, ["order_id", "city", "amount_inr"]]

# Split-apply-combine. .agg gives one row per city.
by_city = paid.groupby("city", as_index=False).agg(
    orders=("order_id", "count"),
    revenue=("amount_inr", "sum"),
)

# Merge on a shared key. validate= raises if the key is not unique on the right.
cities = pd.read_csv("cities.csv")            # columns: city, region
joined = by_city.merge(cities, on="city", how="left", validate="many_to_one")

print(joined.sort_values("revenue", ascending=False).head())
```

The NumPy half of the same idea — reshaping and broadcasting. Note the annotated
shapes; predict each one before you run it.

```python
from __future__ import annotations

import numpy as np

x = np.arange(12, dtype=np.float64)   # shape (12,)
grid = x.reshape(3, 4)                # shape (3, 4)
col = grid[:, 1:2]                    # shape (3, 1) — kept 2-D on purpose
row_mean = grid.mean(axis=1)          # shape (3,)    — one value per row

# Broadcasting: (3, 1) - (3,)  ->  (3, 1) - (1, 3)  ->  (3, 3)
diff = col - row_mean
print(grid.shape, col.shape, row_mean.shape, diff.shape)
# (3, 4) (3, 1) (3,) (3, 3)

# Standardise columns: mean 0, unit variance, per column.
z = (grid - grid.mean(axis=0)) / grid.std(axis=0)   # shapes (3,4) vs (4,) -> (3,4)
print(np.round(z.mean(axis=0), 6))                  # ~[0. 0. 0. 0.]
```

## Real use case

A support-analytics product shows an operations team how many tickets each agent
closes per week. The first version loaded `tickets.csv` and looped over rows to count
per agent, then looped again to attach each agent's team from `agents.csv`. It took
about a minute on a 200,000-row export, and the join was a nested loop that silently
dropped agents whose id column had been read as a string in one file and an integer
in the other — 4 agents vanished from the dashboard with no error.

The rewrite is the code above: `read_csv` with explicit dtypes, one `groupby().agg()`,
one `merge(validate="many_to_one")`. Runtime dropped to well under a second for the
same file. More importantly, the `validate=` argument turned a silent data loss into
an exception on the first run, which is how the dtype mismatch was found.

| Before | After |
|---|---|
| Nested loops over 200k rows | One groupby + one merge |
| Missing agents, no error | `validate=` raises on a bad key |
| Shapes guessed, shapes wrong | Shapes asserted in a test |

## Common mistakes

- **Mistake:** `df[df.a > 1 and df.b < 2]` — **Instead:** use `&` with parentheses, `df[(df.a > 1) & (df.b < 2)]`; `and` is Python's scalar operator and raises on arrays.
- **Mistake:** chained assignment like `df[df.a > 1]["b"] = 0` — **Instead:** use `.loc`, `df.loc[df.a > 1, "b"] = 0`, which writes to the frame rather than a copy.
- **Mistake:** merging without checking row counts — **Instead:** assert `len(merged) == len(expected)` and pass `validate=` so a bad key fails loudly.
- **Mistake:** letting Pandas infer dtypes on ids and dates — **Instead:** pass `dtype=` and `parse_dates=` explicitly; ids read as strings cause silent join misses.
- **Mistake:** `for i in range(len(arr))` to fill an array — **Instead:** allocate with `np.zeros(shape)` and fill with a vectorised expression or slicing.

## Checkpoint

1. Build a `(6, 3)` array of random integers, then produce a `(6,)` array holding the mean of each row, and a `(3,)` array holding the max of each column. Write down each shape before running it.
2. From a DataFrame of orders with `city` and `amount_inr`, produce revenue per city and the share of total revenue each city contributes, in one chain of Pandas calls.
3. Load two CSVs that share a key, prove the key is unique on one side, then merge and assert the row count is what you predicted. Break it on purpose by changing one file's dtype and confirm `validate=` catches it.

**Self-check:** the row-mean array is `(6,)` because `mean(axis=1)` collapses the column axis, leaving one value per row; the column max is `(3,)` because `max(axis=0)` collapses the row axis.

## Go deeper

- [Cleaning Messy Data](02-data-cleaning.md) — what to do when the CSV you just loaded is not trustworthy.
- [Exploratory Data Analysis](03-eda.md) — turning these primitives into answers.
- [SQL on PostgreSQL](04-sql-postgresql.md) — the same groupby and join, expressed where the data already lives.
- [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md) — why shapes and broadcasting are the algebra you already met.

---
[← Previous: Stage 02 — Math and ML Basics](../02-math-and-ml-basics/README.md) · [Stage index](README.md) · [Next: Cleaning Messy Data](02-data-cleaning.md)
