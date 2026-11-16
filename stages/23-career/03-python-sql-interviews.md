# 3. Python and SQL Interviews

> **Stage 23 · Career** — the coding and query rounds that come before anyone asks you about models, answered with the structure interviewers are listening for.
>
> **You'll be able to:** answer async, generator and complexity questions with working code, and write window-function and indexing answers that show you have used SQL on real data.
>
> **Prerequisites:** [Python Async](../01-programming-basics/01-python-async.md) · [SQL and PostgreSQL](../03-data-engineering/04-sql-postgresql.md)

## Why this exists

For an AI-engineer role, the Python round is rarely LeetCode-hard. It is a filter for whether you can write code that does not deadlock, does not load a million rows into memory, and can explain why one approach is slower than another. The SQL round filters for the same thing: whether you can express a ranking or a funnel without three self-joins.

These rounds are winnable by preparation because the archetypes repeat. There are maybe eight Python patterns and six SQL patterns, and they cover most of what you will be asked.

Mental model: say the shape of the answer before writing it — approach, complexity, edge cases — then write code. Interviewers score the narration as much as the code.

## The mental model

| Round | Archetype | What they are really checking |
|---|---|---|
| Python | run N async calls with a concurrency cap | do you know `gather` vs `TaskGroup`, and why a semaphore |
| Python | generator that batches / streams | do you know memory does not scale with input |
| Python | dedupe or group preserving order | dict/set semantics and complexity |
| Python | complexity of a nested loop | can you reason, not memorise |
| SQL | top-N per group | window functions |
| SQL | find rows missing in another table | anti-join vs `NOT IN` with NULLs |
| SQL | deduplicate keeping latest | `DISTINCT ON` or row_number |
| SQL | why is this query slow | indexing, selectivity, `EXPLAIN` |

## Core concepts

### Async: the cap is the point

The question is usually "call this API for 500 ids". The naive answer awaits in a loop and takes forever; the naive fix `asyncio.gather` on 500 tasks hammers the provider and gets rate-limited. The answer they want: bound concurrency with a semaphore, handle per-task failure so one bad id does not kill the batch, and know that `gather` with `return_exceptions=True` returns results and exceptions in order. Mention that `asyncio.TaskGroup` cancels siblings on failure — that difference is the follow-up question.

### Generators: memory, not syntax

A generator question is a memory question in disguise. "Read a large file and yield batches" tests whether you know that `readlines()` materialises everything and that yielding batches decouples IO size from batch size. Say the memory profile out loud: O(batch), not O(file). Bonus point: generators are single-use, so if the caller needs two passes, they need a list or a re-open.

### Complexity: reason from the data structure

They are not asking you to memorise the table. They are asking whether you can spot the hidden cost: `x in list` inside a loop is O(n²); the same with a set is O(n). A dict lookup is average O(1) but hash collisions and bad `__hash__` matter. Sorting inside a loop is the other classic. If you can name the data structure and its cost, you have answered.

### SQL: window functions replace self-joins

"Latest row per user", "top 3 per category", "running total" — all window functions. The pattern: `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` inside a CTE, filter `= 1` outside. Filtering on the window result in the same `WHERE` fails because window functions are evaluated after `WHERE`; that is the trap and mentioning it earns credit.

### SQL: joins, NULLs and the anti-join trap

`NOT IN (SELECT id FROM b)` returns no rows if `b.id` contains a NULL, because `x <> NULL` is unknown. Use `NOT EXISTS` or a `LEFT JOIN ... WHERE b.id IS NULL`. Say this without being asked; it is the most common production SQL bug in analytics code. Also know that `LEFT JOIN` with a filter on the right table in `WHERE` silently becomes an inner join.

### Indexing: what an index does and does not fix

A B-tree index helps equality and range predicates on a leading column, and helps `ORDER BY` when the order matches. It does not help `LIKE '%term%'`, a function applied to the column (`WHERE lower(email) = ...` needs an expression index), or a low-selectivity predicate where a sequential scan is cheaper. Composite index order matters: `(tenant_id, created_at)` serves tenant-scoped recency queries; `(created_at, tenant_id)` does not. Say `EXPLAIN ANALYZE` and what you look for: row estimate versus actual, and whether it is a seq scan when you expected an index scan.

## Code

A worked answer to the async archetype, with the narration you would say out loud: bound concurrency, isolate failures, keep order.

```python
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def fetch_all(
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    concurrency: int = 8,
) -> list[R | Exception]:
    """Run fn over items with a concurrency cap; failures do not cancel the batch.

    Approach: semaphore-bounded tasks + gather(return_exceptions=True).
    Complexity: O(n) wall-clock chunks of `concurrency`; memory O(n) for results.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(item: T) -> R:
        async with sem:                       # at most `concurrency` in flight
            return await fn(item)

    return await asyncio.gather(*[one(i) for i in items], return_exceptions=True)
```

The generator archetype, with the memory claim stated:

```python
def batches(path: str, size: int = 500):
    """Yield lists of lines; memory is O(size), not O(file)."""
    buf: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:                       # iterating a file streams it
            buf.append(line.rstrip("\n"))
            if len(buf) >= size:
                yield buf
                buf = []                      # new list: the yielded one is the caller's
    if buf:
        yield buf
```

The two SQL archetypes, both of which come up in some form in most data rounds:

```sql
-- top 3 events per user by recency, and the trap: window fns run after WHERE
WITH ranked AS (
  SELECT user_id,
         event_type,
         created_at,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
  FROM events
)
SELECT user_id, event_type, created_at
FROM ranked
WHERE rn <= 3;

-- rows in `documents` with no chunks: anti-join that survives NULLs
SELECT d.id
FROM documents d
WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id);

-- the index that makes a tenant-scoped recency query cheap
CREATE INDEX CONCURRENTLY idx_events_tenant_created
  ON events (tenant_id, created_at DESC);
```

## Real use case

In a real interview loop, a candidate was asked to "write code that embeds 10,000 chunks with an API". They wrote the loop version, then corrected themselves to `gather` with a semaphore, then added `return_exceptions=True` and a retry on rate-limit errors — and said the trade-off out loud: higher concurrency finishes sooner and triggers more 429s, so the cap is a policy decision, not a constant.

The design choice that made the answer strong is bounding concurrency while isolating failures: it is exactly the shape of every real ingestion script, and it shows the interviewer you have run one.

## Common mistakes

**Mistake:** `gather` on thousands of tasks with no cap — **Instead:** a semaphore, and say why the cap exists.
**Mistake:** letting one exception cancel a batch silently — **Instead:** `return_exceptions=True` or `TaskGroup` with explicit handling, and state the difference.
**Mistake:** reading a whole file to process it in batches — **Instead:** iterate the file handle and yield batches.
**Mistake:** `NOT IN` against a nullable column — **Instead:** `NOT EXISTS` or a `LEFT JOIN ... IS NULL`.
**Mistake:** filtering on a window function in `WHERE` — **Instead:** compute it in a CTE, filter outside.
**Mistake:** claiming an index helps any query — **Instead:** name the predicate shape, the leading column, and when a seq scan wins.

## Checkpoint

1. Rewrite `fetch_all` using `asyncio.TaskGroup` and state what changes when one task raises.
2. Write a query returning the latest chunk per document, then one returning documents with zero chunks. Check both against a table where `chunk.document_id` has a NULL.
3. Take the tenant-scoped query above, run `EXPLAIN ANALYZE` with and without the index, and explain the plan change in two sentences.

**Self-check:** Exercise 1 changes failure semantics — `TaskGroup` cancels the remaining tasks when one raises, so it suits all-or-nothing work, not best-effort ingestion.

## Go deeper

- [Generators](../01-programming-basics/03-generators.md) — the pattern behind every streaming pipeline
- [Data Pipelines](../03-data-engineering/06-data-pipelines.md) — async ingestion in production form
- [ML and LLM Basics](04-ml-llm-basics.md) — the next round in most loops
- [System Design Practice](05-system-design-practice.md) — where these primitives become architecture

---

[← Previous: Resume](02-resume.md) · [Stage index](README.md) · [Next: ML and LLM Basics](04-ml-llm-basics.md)
