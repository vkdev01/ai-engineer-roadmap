# 4. SQL on PostgreSQL

> **Stage 03 · Data Engineering** — querying where the data lives, with the three SQL features that separate beginners from useful engineers
>
> **You'll be able to:** write joins, CTEs, window functions and JSON queries against PostgreSQL, and decide when SQL beats Pandas.
>
> **Prerequisites:** [NumPy and Pandas](01-numpy-pandas.md) · [HTTP and REST APIs](../01-programming-basics/07-http-rest-apis.md)

## Why this exists

Pandas requires the data to fit in your process's memory. Production tables do not
fit, and the copy you make is stale the moment you make it. SQL runs where the data
is, returns only the rows you asked for, and is the one language every data team
shares — the query you write is the query an analyst can read.

Three features carry most of the weight. **Joins** combine tables. **CTEs** let you
name an intermediate result so a query reads top to bottom instead of inside out.
**Window functions** compute a value per row that depends on other rows *without
collapsing them*, which is the thing `GROUP BY` cannot do and the reason most
"give me the running total" questions are painful without them.

Mental model: **`GROUP BY` gives you one row per group; a window function gives you
every row, each carrying a value computed over its group.**

## The mental model

```text
orders                        customers
------                        ---------
id  customer_id  amount_inr   id  name    city
1   7            499          7   Asha    Pune
2   7            1200         8   Vikram  Kochi
3   8            300

FROM orders o JOIN customers c ON o.customer_id = c.id

step 1  join      -> one wide row per order (3 rows)
step 2  WHERE     -> filter rows
step 3  GROUP BY  -> collapse to one row per group   (aggregate)
   or   OVER ()   -> keep all rows, add a computed column (window)
step 4  HAVING    -> filter groups
step 5  ORDER BY  -> sort, last
```

## Core concepts

### A CTE names a step so the query reads forward

`WITH` defines a named result you can use below. The point is legibility: instead of
a subquery inside a subquery, you get a list of steps. A CTE is not automatically
materialised — PostgreSQL may inline it — so treat it as a readability tool, not a
performance one.

### Window functions keep every row

`SUM(amount_inr) OVER (PARTITION BY customer_id)` adds a column holding each
customer's total, repeated on each of their rows. `ROW_NUMBER() OVER (PARTITION BY
customer_id ORDER BY created_at DESC)` numbers each customer's orders newest first,
which gives you the standard "latest row per group" query: number them, then keep
where the number is 1. `LAG()` reads the previous row's value, which is how you
compute a gap between events without a self-join.

### JSONB is a real column type

Postgres can store a JSON document in a `jsonb` column and index into it. Use it for
payloads whose shape genuinely varies — raw model responses, tool-call arguments,
provider metadata. Query with `->` (returns JSON) and `->>` (returns text);
`jsonb_path_ops` GIN indexes make containment queries (`@>`) fast. Do not use `jsonb`
for fields you filter on constantly; those deserve real columns.

### Indexes trade write speed for read speed

An index is a sorted side structure. It makes `WHERE customer_id = 7` fast and makes
every `INSERT` slightly slower. Index the columns you filter and join on, not every
column. `EXPLAIN ANALYZE` tells you whether the planner used it — read that before
guessing why a query is slow.

### SQL or Pandas? Ask where the data is

If the data already lives in Postgres and you need an aggregate, do it in SQL and
return a small frame. If you need to iterate over rows in Python, reshape, or fit a
model, pull a narrow result set into Pandas. Pulling a whole table to aggregate it in
Pandas is the anti-pattern.

## Code

A single query that answers a real question: each customer's latest order, their
lifetime spend, and the days since that order. Note that the window functions keep
one row per order, and the final filter collapses to one row per customer.

```sql
-- One row per customer: latest order, lifetime spend, recency.
WITH ranked AS (
    SELECT
        o.id            AS order_id,
        o.customer_id,
        o.amount_inr,
        o.created_at,
        SUM(o.amount_inr) OVER (PARTITION BY o.customer_id) AS lifetime_inr,
        ROW_NUMBER()     OVER (PARTITION BY o.customer_id
                               ORDER BY o.created_at DESC)  AS rn
    FROM orders o
    WHERE o.status = 'paid'
),
latest AS (
    SELECT * FROM ranked WHERE rn = 1
)
SELECT
    c.name,
    c.city,
    l.order_id          AS last_order,
    l.amount_inr        AS last_amount_inr,
    l.lifetime_inr,
    CURRENT_DATE - l.created_at::date AS days_since
FROM latest l
JOIN customers c ON c.id = l.customer_id
ORDER BY l.lifetime_inr DESC
LIMIT 50;
```

A gap between events with `LAG`, plus a JSONB column for a model call log. You should
see one row per call with the latency delta from that session's previous call.

```sql
-- Latency between consecutive calls in the same session, from a jsonb payload.
SELECT
    id,
    session_id,
    payload ->> 'model'                        AS model,
    (payload -> 'usage' ->> 'completion_tokens')::int AS completion_tokens,
    created_at - LAG(created_at) OVER (
        PARTITION BY session_id ORDER BY created_at
    ) AS since_previous
FROM llm_calls
WHERE payload @> '{"status": "ok"}'      -- jsonb containment, GIN-indexable
ORDER BY session_id, created_at;
```

```sql
CREATE INDEX llm_calls_payload_gin ON llm_calls USING GIN (payload jsonb_path_ops);
```

And the same aggregate from Python, so you can see where the boundary sits — the
query does the work, Python only receives the result.

```python
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://app:secret@localhost:5432/app")

QUERY = """
SELECT c.city, COUNT(*) AS paid_orders, SUM(o.amount_inr) AS revenue
FROM orders o
JOIN customers c ON c.id = o.customer_id
WHERE o.status = 'paid' AND o.created_at >= now() - interval '90 days'
GROUP BY c.city
HAVING COUNT(*) >= 5
ORDER BY revenue DESC
"""

with engine.connect() as conn:
    by_city = pd.read_sql(QUERY, conn)   # small frame, aggregation done in SQL
print(by_city.head())
```

## Real use case

A fintech's credit team needed "the customer's most recent KYC document and whether
it is expiring in the next 30 days", across 4 million rows. The first version pulled
every document row into a Python service and used Pandas `groupby().tail(1)`, which
took minutes, allocated several gigabytes, and returned stale data if the service
cached it.

The rewrite is the `ranked`/`latest` CTE above, with `ROW_NUMBER()` doing the
"latest per customer" work inside Postgres and a partial index on `status = 'paid'`.
The API now returns a 200-row page in milliseconds and holds no data in memory.
The `LAG()` query became the basis of a separate latency dashboard, built from the
`jsonb` call log the team already had, so no new table was needed.

## Common mistakes

- **Mistake:** `SELECT *` in production queries — **Instead:** name the columns; it prevents breakage when a column is added and lets the planner use an index-only scan.
- **Mistake:** filtering a window result in `WHERE` — **Instead:** window functions are evaluated after `WHERE`; wrap the query in a CTE and filter on the alias.
- **Mistake:** using `jsonb` for a field you filter on every day — **Instead:** promote hot fields to real typed columns; keep `jsonb` for genuinely variable payloads.
- **Mistake:** indexing every column — **Instead:** index what you filter and join on, then confirm with `EXPLAIN ANALYZE`.
- **Mistake:** pulling a full table into Pandas to aggregate it — **Instead:** aggregate in SQL and return the small result.
- **Mistake:** forgetting that `COUNT(*)` counts rows while `COUNT(col)` skips nulls — **Instead:** choose deliberately and comment which you meant.

## Checkpoint

1. Write a query that returns each customer's first and last order date, using `MIN`/`MAX` with `GROUP BY`.
2. Rewrite it to also include each customer's most recent order id, using `ROW_NUMBER()` in a CTE. Confirm both versions return the same number of rows.
3. Using `LAG()`, find sessions where the gap between two consecutive calls exceeds one minute, and join that back to a `customers` table to see who is affected.

**Self-check:** `MIN(created_at)` and `MAX(created_at)` per `customer_id` give first and last order dates; the window version keeps the extra columns because it never collapses the rows.

## Go deeper

- [Parquet and JSONL](05-parquet-jsonl.md) — the file formats that sit on either side of a database export.
- [Data Pipelines](06-data-pipelines.md) — running these queries as scheduled, idempotent steps.
- [Python and SQL Interviews](../23-career/03-python-sql-interviews.md) — these three features are what interviewers probe.
- [NumPy and Pandas](01-numpy-pandas.md) — the same joins and groupbys on the Python side.

---
[← Previous: Exploratory Data Analysis](03-eda.md) · [Stage index](README.md) · [Next: Parquet and JSONL](05-parquet-jsonl.md)
