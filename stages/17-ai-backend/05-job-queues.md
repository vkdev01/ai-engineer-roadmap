# 5. Job Queues

> **Stage 17 · AI Backend** — moving work that takes minutes out of the request path.
>
> **You'll be able to:** enqueue a long AI task with retries and idempotency, and give clients a way to follow it.
>
> **Prerequisites:** [Async Processing](03-async-processing.md) · [Redis Caching](04-redis-caching.md)

## Why this exists

Some work does not fit in an HTTP request. Ingesting a 300-page PDF, generating embeddings for
10,000 chunks, running an eval suite, transcribing an hour of audio: these take minutes, may need
retries, and will exceed any sane proxy timeout. Holding the connection open is not a design, it
is a countdown to a 504.

A job queue separates *accepting* work from *doing* work. The API writes a job record, returns
immediately with an id, and a separate worker process picks the job up, retries it on failure, and
writes the result where the client can fetch it.

The mental model: **the request path promises, the worker delivers.**

## The mental model

```text
client --POST /jobs--> API --enqueue--> queue (Redis) --> worker --retry x3--> provider
                        |                                   |
                        +--> 202 {"job_id": "..."}           +--> result store
client --GET /jobs/{id}--> API --read--> {status, progress, result_url}
```

| | Celery | RQ | arq |
|---|---|---|---|
| Model | mature, feature-rich | simple, Redis only | asyncio-native, Redis only |
| Async support | partial, historically sync-first | sync workers | async first |
| Scheduling, chains | yes, rich primitives | basic | basic |
| Learning cost | high | low | low |
| Good fit | large existing Celery estates | small sync jobs | FastAPI async services |

## Core concepts

### Retries need to be safe, so tasks must be idempotent

A queue retries by re-running. If the first attempt actually succeeded but the acknowledgement was
lost, you will run it twice — paying the provider twice and possibly writing duplicate rows.
Make the task idempotent: derive a deterministic key from the job's inputs, and check it before
doing the expensive work.

```text
idempotency_key = sha256(tenant_id + document_id + task_name + params_version)
```

Store that key with a status when you start, and short-circuit if it is already `done`. This is
the same pattern as an HTTP `Idempotency-Key` header, applied one layer down.

### Retry policy is a cost decision

Exponential backoff with jitter, capped attempts, and a dead-letter destination. Concretely: 3
attempts at 2, 4 and 8 seconds, plus jitter. Retrying a provider rate-limit error needs backoff;
retrying a 400 does not, because it will fail identically forever. Classify errors before you
retry — [Retries and Rate Limits](../07-llm-apis/07-retries-rate-limits.md) covers the provider
side of that.

### Progress must be observable

A job that takes six minutes needs a status a client can poll or subscribe to. Write progress
into the same store as the result: `queued`, `running`, `progress: 0.42`, `done`,
`failed: <reason>`. Without this, users assume the system is broken and re-submit, which doubles
your load and your bill.

### One queue per cost profile, not per feature

Separate queues by resource shape, not by product area: a CPU-bound parsing queue, an
embedding queue, a provider-call queue. Otherwise a batch of 500 PDFs starves the one urgent job
that a user is watching. Workers then scale per queue, so you can add embedding capacity without
adding PDF-parsing capacity.

### The result store is not the queue

Keep job state and results in something durable and queryable — Postgres, S3, or Redis with
persistence enabled — rather than relying on the queue's internals. Queues are for delivery; they
are not a database.

## Code

An `arq` worker with an idempotent task, retries and progress reporting. It uses Redis for both
the queue and the job status, which is enough for a small service.

```python
from __future__ import annotations

import hashlib
import json

from arq import Retry, create_pool
from arq.connections import RedisSettings

JOB_TTL = 24 * 3600


def idempotency_key(tenant: str, doc_id: str, task: str, version: str) -> str:
    raw = f"{tenant}:{doc_id}:{task}:{version}".encode()
    return hashlib.sha256(raw).hexdigest()


async def set_status(ctx, key: str, **fields) -> None:
    redis = ctx["redis"]
    current = json.loads(await redis.get(f"job:{key}") or "{}")
    current.update(fields)
    await redis.set(f"job:{key}", json.dumps(current), ex=JOB_TTL)


async def ingest_document(ctx, tenant: str, doc_id: str, pages: list[str]) -> dict:
    """Chunk, embed and store a document. Safe to retry: keyed on inputs."""
    key = idempotency_key(tenant, doc_id, "ingest", "v1")
    redis = ctx["redis"]

    existing = json.loads(await redis.get(f"job:{key}") or "{}")
    if existing.get("status") == "done":
        return {"skipped": True, "chunks": existing.get("chunks", 0)}

    await set_status(ctx, key, status="running", progress=0.0, attempts=ctx["job_try"])

    chunks: list[str] = []
    for i, page in enumerate(pages):
        chunks.extend(c for c in page.split("\n\n") if c.strip())
        await set_status(ctx, key, progress=round((i + 1) / max(len(pages), 1) * 0.5, 3))

    try:
        # Replace with your embedding call; batched to keep provider calls few.
        vectors = await ctx["embed_many"](chunks)
    except TimeoutError:
        # Retryable: arq re-runs the task, and the idempotency check above
        # prevents duplicate work once the first attempt has finished.
        if ctx["job_try"] < 3:
            raise Retry(defer=2 ** ctx["job_try"]) from None
        await set_status(ctx, key, status="failed", error="embedding timeout")
        raise

    await ctx["store"].upsert(tenant=tenant, doc_id=doc_id, chunks=chunks, vectors=vectors)
    await set_status(ctx, key, status="done", progress=1.0, chunks=len(chunks))
    return {"skipped": False, "chunks": len(chunks)}


async def startup(ctx) -> None:
    ctx["embed_many"] = build_embedder()          # your provider client
    ctx["store"] = build_vector_store()


class WorkerSettings:
    functions = [ingest_document]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn("redis://localhost:6379/0")
    max_tries = 3
    job_timeout = 900                              # 15 minutes
```

Enqueue from the API and return immediately. The client then polls the status key, so a 15-minute
job never holds an HTTP connection.

```python
from __future__ import annotations

from arq import create_pool
from arq.connections import RedisSettings


async def enqueue_ingest(tenant: str, doc_id: str, pages: list[str]) -> str:
    pool = await create_pool(RedisSettings.from_dsn("redis://localhost:6379/0"))
    job = await pool.enqueue_job("ingest_document", tenant, doc_id, pages)
    if job is None:                     # arq returns None if a duplicate is already queued
        raise RuntimeError("job already queued")
    return job.job_id
```

## Real use case

A legal-tech product let firms upload bundles of case documents and ask questions across them.
Ingestion — parsing, chunking and embedding a 400-page bundle — took several minutes. The first
implementation did it inside the upload request, and it failed in three distinct ways: reverse
proxies cut the connection at 60 seconds, users refreshed the page and uploaded again, and the
retry-on-timeout logic created duplicate embeddings that showed up as repeated citations in
answers.

They moved ingestion to a queue. The upload endpoint wrote the file to object storage and
enqueued a job keyed on `tenant:document:content_hash`, returning a job id in under a second. The
worker reported progress, so the UI showed a real bar instead of a spinner. Duplicate uploads
collapsed onto the same idempotency key, so refreshing the page cost nothing and created no
duplicate chunks.

They also split queues by resource shape. PDF parsing and OCR went to a CPU queue with many small
workers; embedding went to a queue with a small number of workers sized to the provider's rate
limit. Before the split, a firm uploading 200 documents at once would occupy every worker and the
single urgent re-index one user was waiting on would sit behind all of it. After the split, that
urgent job had its own lane.

## Common mistakes

- **Mistake:** doing long work inside the request handler and hoping the timeout is generous — **Instead:** enqueue and return 202 with a job id.
- **Mistake:** writing tasks that are not idempotent — **Instead:** key on inputs and short-circuit when the work is already done.
- **Mistake:** retrying non-retryable errors — **Instead:** classify; only timeouts, connection errors and 5xx deserve a retry.
- **Mistake:** one queue for everything — **Instead:** separate by resource profile so urgent small jobs are not starved.
- **Mistake:** storing job results only in the queue's internal structures — **Instead:** write status and results to a durable store you control.

## Checkpoint

1. A task fails 20 percent of the time with a retryable error. With 3 attempts, what fraction of jobs end in failure, assuming independent attempts?
2. Add a `GET /jobs/{id}` FastAPI endpoint that reads the status key and returns `{status, progress, error}`. Confirm it reports progress while a long job runs.
3. Make the worker handle a non-retryable error (a 400 from the provider) by marking the job `failed` immediately and not consuming the remaining attempts. Test it with a deliberately malformed input.

**Self-check:** `0.2^3 = 0.008`, so about 0.8 percent of jobs fail after three attempts — but only if the failures are genuinely independent, which is why backoff matters.

## Go deeper

- [Async Processing](03-async-processing.md) — why CPU-bound work must leave the loop in the first place.
- [Redis Caching](04-redis-caching.md) — the same Redis instance usually serves both jobs and cache.
- [Data Pipelines](../03-data-engineering/06-data-pipelines.md) — batch ingestion at a larger scale.
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — dead letters, backpressure and degraded modes.

---
[← Previous: Redis Caching](04-redis-caching.md) · [Stage index](README.md) · [Next: Auth and Rate Limits](06-auth-rate-limits.md)
