# 8. Ship: A Trace Dashboard

> **Stage 18 · MLOps and LLMOps** — the project that proves you can turn raw traces into latency, cost, error and eval numbers per prompt version.
>
> **You'll be able to:** build an ingest endpoint plus a dashboard that answers "which prompt version is slow, expensive or wrong" from your own data.
>
> **Prerequisites:** [Tracing Design](05-tracing.md) · [Cost Monitoring and Budgets](06-cost-monitoring.md)

## Why this exists

Every topic in this stage produces data. This project makes that data answer questions. You are building the small version of what an observability vendor sells: an endpoint that accepts trace records, a store, and four aggregations that a human can read in ten seconds.

Doing it yourself is the fastest way to learn what the vendors are actually computing, and it gives you a fallback when a vendor's pricing or retention policy does not fit. It is also a strong portfolio piece, because it shows you understand the join keys — prompt version, feature, tenant — that make LLM systems debuggable.

The mental model: **ingest writes one flat row per request; the dashboard is four `GROUP BY` queries over that row.**

## The mental model

```text
AI app --POST /ingest--> trace row
                          (trace_id, ts, feature, tenant, prompt_version,
                           model, latency_ms, tokens_in, tokens_out,
                           cost_usd, error, eval_score)
                                    |
              +---------------------+---------------------+
              v                     v                     v
      /metrics/latency      /metrics/cost        /metrics/errors
      p50/p95 by version    spend + per-success  rate by version
              |
              v
      /metrics/eval  mean score by version (from CI + online judges)
              |
              v
      one HTML page: four tables, version as the join key
```

## Core concepts

### What you build

A single Python service with four parts. Keep it one repo and one deployable; splitting ingest from query is a later optimisation.

1. **`POST /ingest`** — accepts a batch of trace records, validates with pydantic, writes to a table. Must be fast and must never reject the caller's whole request because of one malformed row: validate per row, count rejects, return `202`.
2. **A store** — SQLite is enough for this project and keeps the deploy trivial; use Postgres if you already have one. The only schema decision that matters is that every row carries `prompt_version`, `feature` and `tenant_id`.
3. **Four aggregation endpoints** — latency percentiles, cost, error rate, eval score, all grouped by prompt version, with a `since` filter.
4. **`GET /`** — an HTML page that fetches the four endpoints and renders tables, server-rendered with a tiny template. No frontend build step; you are demonstrating data plumbing, not React.

## Code

The ingest side: the record schema and the writer. Note that `cost_usd` is sent by the app that made the call, not recomputed here, because only the caller knows the token counts and the price that applied at the time.

```python
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()
DB = "traces.db"


class TraceRecord(BaseModel):
    trace_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature: str
    tenant_id: str
    prompt_version: str
    model: str
    latency_ms: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    error: bool = False
    error_type: str | None = None
    eval_score: float | None = Field(default=None, ge=0, le=1)


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  feature TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_usd REAL NOT NULL,
  error INTEGER NOT NULL,
  error_type TEXT,
  eval_score REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_version ON traces(prompt_version, ts);
"""


@app.on_event("startup")
def init_db() -> None:
    with sqlite3.connect(DB) as conn:
        conn.executescript(SCHEMA)


@app.post("/ingest", status_code=202)
def ingest(batch: list[dict]) -> dict[str, int]:
    ok = 0
    rejected = 0
    rows: list[tuple] = []
    for item in batch:
        try:
            r = TraceRecord.model_validate(item)
        except ValueError:
            rejected += 1          # one bad row must not sink the batch
            continue
        rows.append((r.trace_id, r.ts.isoformat(), r.feature, r.tenant_id,
                     r.prompt_version, r.model, r.latency_ms, r.tokens_in,
                     r.tokens_out, r.cost_usd, int(r.error), r.error_type, r.eval_score))
        ok += 1
    with sqlite3.connect(DB) as conn:
        conn.executemany("INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return {"accepted": ok, "rejected": rejected}
```

The query side: percentiles are computed in SQL rather than in Python, and cost per success divides by successes so a broken version cannot look cheap. `PERCENTILE_CONT` is not in SQLite, so use a window-function approximation, which is exact enough at these volumes.

```python
from __future__ import annotations

import sqlite3

from fastapi import FastAPI, Query

app = FastAPI()
DB = "traces.db"


def q(sql: str, params: tuple) -> list[dict]:
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


@app.get("/metrics/latency")
def latency(since: str = Query("1970-01-01")) -> list[dict]:
    # Rank within version, then pick the row at the 50th and 95th percentile rank.
    return q(
        """
        WITH ranked AS (
          SELECT prompt_version, latency_ms,
                 ROW_NUMBER() OVER (PARTITION BY prompt_version ORDER BY latency_ms) AS rn,
                 COUNT(*)    OVER (PARTITION BY prompt_version) AS n
          FROM traces WHERE ts >= ? AND error = 0
        )
        SELECT prompt_version,
               MAX(CASE WHEN rn = CAST(n * 0.50 AS INTEGER) + 1 THEN latency_ms END) AS p50_ms,
               MAX(CASE WHEN rn = CAST(n * 0.95 AS INTEGER) + 1 THEN latency_ms END) AS p95_ms,
               MAX(n) AS requests
        FROM ranked GROUP BY prompt_version ORDER BY p95_ms DESC
        """,
        (since,),
    )


@app.get("/metrics/cost")
def cost(since: str = Query("1970-01-01")) -> list[dict]:
    return q(
        """
        SELECT prompt_version,
               ROUND(SUM(cost_usd), 4) AS spend_usd,
               COUNT(*) AS requests,
               SUM(CASE WHEN error = 0 THEN 1 ELSE 0 END) AS successes,
               ROUND(SUM(cost_usd) / NULLIF(SUM(CASE WHEN error = 0 THEN 1 ELSE 0 END), 0), 6)
                 AS cost_per_success_usd
        FROM traces WHERE ts >= ?
        GROUP BY prompt_version ORDER BY spend_usd DESC
        """,
        (since,),
    )


@app.get("/metrics/errors")
def errors(since: str = Query("1970-01-01")) -> list[dict]:
    return q(
        """
        SELECT prompt_version,
               ROUND(AVG(error) * 100, 2) AS error_pct,
               SUM(error) AS failures
        FROM traces WHERE ts >= ? GROUP BY prompt_version ORDER BY error_pct DESC
        """,
        (since,),
    )


@app.get("/metrics/eval")
def eval_score(since: str = Query("1970-01-01")) -> list[dict]:
    return q(
        """
        SELECT prompt_version, ROUND(AVG(eval_score), 3) AS mean_score, COUNT(eval_score) AS scored
        FROM traces WHERE ts >= ? AND eval_score IS NOT NULL
        GROUP BY prompt_version ORDER BY mean_score DESC
        """,
        (since,),
    )
```

### How to prove it works

Three steps, in order. Do not skip the third — it is the one that makes the project worth showing.

1. **Seed with recorded data.** Write a script that replays a week of synthetic-but-realistic traces through `/ingest`: two prompt versions, one feature, two tenants, a small error rate. Check that each endpoint returns a table you would believe.
2. **Prove the aggregations are right.** Hand-compute p50 for one version from the raw rows in a notebook and compare to the endpoint. Percentile code is where these dashboards usually lie quietly.
3. **Ship a regression and catch it.** Send a batch for a new `prompt_version` with roughly double the tokens and a higher error rate. Confirm the dashboard shows the new version with worse cost per success and a higher error percentage, in the same table as the healthy version. That side-by-side is your evidence.

A short before/after table makes the result legible:

| prompt_version | p95 ms | spend USD | cost per success | error % | mean score |
|---|---|---|---|---|---|
| answer.v3 | 1180 | 4.10 | 0.0031 | 1.2 | 0.78 |
| answer.v4 | 1540 | 9.60 | 0.0074 | 6.4 | 0.74 |

Fill this with your own replayed numbers. The shape is the point: v4 costs more, fails more and scores worse, and one glance is enough.

## Real use case

A two-person team ships a support assistant and uses this dashboard as their release check. Before each prompt release they replay one hour of production traffic through the candidate version in a staging feature key, then compare the two rows. On the third release, the candidate showed a better mean eval score but a p95 latency almost double and a cost per success nearly triple, because the new prompt asked the model to reason before answering and the reasoning tokens were billed.

Without the dashboard they would have shipped it — the quality metric was the only one they had been looking at, and it was genuinely better. The dashboard turned a judgement call into a trade-off they could make deliberately: they kept the reasoning prompt for the paid tier and kept the fast prompt for the free tier, splitting by tenant plan. That split later became the basis of their pricing page.

## Common mistakes

**Mistake:** recomputing cost in the dashboard from tokens with today's prices — **Instead:** accept `cost_usd` from the caller, so historical rows keep historical prices.

**Mistake:** averaging latency instead of using percentiles — **Instead:** p50 and p95; a mean hides the tail that users complain about.

**Mistake:** grouping by `model` only — **Instead:** group by `prompt_version`, which is what actually changed, and keep model as a second dimension.

**Mistake:** an ingest endpoint that returns 400 for the whole batch on one bad row — **Instead:** validate per row, count rejects, return 202.

**Mistake:** no `since` filter, so the dashboard scans everything — **Instead:** default to the last 24 hours and index `(prompt_version, ts)`.

## Checkpoint

1. Stand up ingest plus the four endpoints with seeded data and confirm each returns rows grouped by prompt version.
2. Verify one percentile by hand against the raw rows, and add a test that fails if the endpoint and the hand computation disagree.
3. Replay a deliberately worse version and produce the before/after table above from your own numbers.

**Self-check:** exercise 1 is done when `GET /metrics/cost` returns one row per `prompt_version` with a `cost_per_success_usd` that differs from `spend_usd / requests` whenever the version has failures.

## Go deeper

- [Stage 19 — Cloud and DevOps](../19-cloud-devops/README.md) — containerise and deploy this service.
- [CI/CD for Evals](07-ci-cd-evals.md) — feed offline eval scores into the same table.
- [Cost Monitoring and Budgets](06-cost-monitoring.md) — the alerting layer that sits on top of these numbers.
- [Online and Offline Evals](../15-evals/09-online-offline-evals.md) — deciding which scores belong in `eval_score`.

---

[← Previous: CI/CD for Evals](07-ci-cd-evals.md) · [Stage index](README.md) · [Next: Stage 19 — Cloud and DevOps](../19-cloud-devops/README.md)
