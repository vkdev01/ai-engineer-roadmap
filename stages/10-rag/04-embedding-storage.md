# 4. Embedding and Storage Design

> **Stage 10 · RAG** — the schema decisions around each vector: metadata, filters, tenancy, and the re-indexing path you will need sooner than you think.
>
> **You'll be able to:** design a chunk record, choose filter fields deliberately, and plan a re-embed without downtime.
>
> **Prerequisites:** [Chunking](03-chunking.md) · [Chroma and pgvector](../09-vector-databases/08-chroma.md)

## Why this exists

The vector is the easy part. What surrounds it — the metadata schema, the tenancy boundary, the model id, the ability to re-embed everything — decides whether the system survives its first year. Every one of these is cheap to design in week one and expensive to retrofit.

The specific trap: you will change embedding models. Better models arrive, a multilingual requirement appears, or you find the current one is weak on your domain. At that moment you have a collection whose vectors are meaningless under the new model, and no way to tell which rows are which unless you planned for it.

The mental model: **a chunk record is a row in a table whose model id is part of the schema, and whose tenancy is part of the key.**

## The mental model

```text
chunk record
─────────────────────────────────────────────────────────────
id            "sha256:doc7::sec3::c04"     stable, content-addressed
vector        float32[384]                 unit length, from model_id
model_id      "all-MiniLM-L6-v2"           part of the schema, not a comment
dim           384                           mismatch with the model is a bug
tenant        "acme"                       filter, indexed
doc_id        "doc7"                       joins back to the source
source        "handbook.pdf"               shown to the user
page          12                           citation target
heading_path  "Leave > Carryover"          shown to the model
lang          "en"                         filter
version       3                            re-index generation
content_hash  "sha256:9f2c..."             dedup + skip re-embedding
─────────────────────────────────────────────────────────────

re-index = new collection "chunks_v2"  →  backfill  →  verify  →  swap alias
           (old one stays until the new one passes evals)
```

## Core concepts

### Metadata has three jobs, and they need different fields

**Filtering** fields go into the query engine: `tenant`, `lang`, `doc_type`, `updated_at`. Keep them scalar and low-cardinality where possible, and index the ones you filter on constantly.

**Citation** fields go back to the user: `source`, `page`, `heading_path`, `doc_id`. Without these, an answer cannot be verified, and unverifiable answers get ignored in serious settings.

**Debugging** fields never reach a user but save you: `model_id`, `content_hash`, `version`, `ingested_at`. When something is wrong, these tell you which generation of the index produced it.

A common mistake is one `metadata` blob with everything in it. Split the concerns in your record type so it is obvious which fields a query may filter on.

### Multi-tenancy: hard partitions versus filters

Two valid designs. **Partitioned**: a collection or namespace per tenant. Isolation is structural, a filter bug cannot leak data, and small tenants cost a little overhead. **Filtered**: one collection, `tenant` as an indexed filter field, enforced in the engine. Cheaper at scale, and correct only if every single query path applies the filter.

Choose partitioned when the tenant count is small and data sensitivity is high; choose filtered when there are thousands of tenants and you can enforce the filter in one place. Either way, enforce it in the data-access layer, not at each call site — one `search(tenant=...)` function that always injects the filter, with no path around it.

### Re-indexing is a blue-green deployment

Never mutate vectors in place. Create `chunks_v2`, backfill with the new model, run your eval harness against both, then swap an alias or a config pointer. Keep `chunks_v1` for a defined window so rollback is a config change, not a re-embed.

Two details that make it painless. Store `model_id` and `version` on every record so a mixed collection is detectable. And serve reads through an alias so the swap is atomic from the application's point of view.

### Skip re-embedding what has not changed

Hash the chunk text and store `content_hash`. On re-ingest, chunks whose hash already exists with the same `model_id` do not need an embedding call. On a corpus that changes 5% per month, this turns a full re-embed into a small incremental one — and it is the single biggest cost saver in the whole pipeline.

The corollary: chunk ids must be **content-derived**, not positional. If ids shift when a document is edited, you lose the ability to match old chunks to new ones.

### Measure throughput before promising a date

You cannot estimate a re-index without a throughput number. Embed 1,000 chunks, time it, and extrapolate: at 1,000 chunks in `T` seconds, a 2-million-chunk corpus takes about `2000 × T`. Do this arithmetic in a script, not in your head, and include the store's upsert time — it is often comparable to the embedding time.

## Code

The record type and a re-ingest that skips unchanged chunks by content hash. This is the function that makes model migrations cheap.

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, Field

class ChunkRecord(BaseModel):
    id: str = Field(description="content-derived, stable across edits")
    text: str
    vector: list[float] | None = None
    model_id: str
    dim: int
    tenant: str
    doc_id: str
    source: str
    page: int
    heading_path: str = ""
    lang: str = "en"
    version: int = 1
    content_hash: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

def chunk_id(doc_id: str, heading_path: str, text: str) -> str:
    # hash of the text makes the id stable; the path keeps collisions meaningful
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}::{heading_path or '-'}::{h}"

def needs_embedding(rec: ChunkRecord, existing: dict[str, str]) -> bool:
    """existing maps chunk_id -> model_id currently stored."""
    return existing.get(rec.id) != rec.model_id      # same id, new model => re-embed
```

The blue-green migration, with the throughput arithmetic that lets you promise a window you can meet.

```python
from __future__ import annotations

import time

def measure_throughput(embed_batch, sample: list[str], batch_size: int = 64) -> float:
    """Return chunks per second, measured on a real sample."""
    t0 = time.perf_counter()
    for i in range(0, len(sample), batch_size):
        embed_batch(sample[i : i + batch_size])
    elapsed = time.perf_counter() - t0
    return len(sample) / elapsed

def plan_reindex(total_chunks: int, chunks_per_sec: float, workers: int = 1) -> dict[str, float]:
    seconds = total_chunks / (chunks_per_sec * workers)
    return {
        "total_chunks": total_chunks,
        "chunks_per_sec": round(chunks_per_sec, 1),
        "hours": round(seconds / 3600, 2),
        "days_at_8h": round(seconds / 3600 / 8, 2),
    }

# measure on 1,000 real chunks, then extrapolate before you commit to a date
rate = measure_throughput(embed_batch, sample_texts)   # e.g. 60 chunks/sec on a laptop
print(plan_reindex(2_000_000, rate, workers=4))
```

With a measured 60 chunks/second on one worker, 2 million chunks is about 33,333 seconds — roughly 9.3 hours single-threaded, or about 2.3 hours with four workers, before upsert time. Those are arithmetic, not benchmarks: run the measurement on your own hardware.

## Real use case

A support platform indexed 1.8 million chunks with a small English-only model. Eighteen months later they won customers in three more languages, and the model's cross-lingual behaviour was poor: a Hindi question rarely matched an English passage.

Their first instinct was to re-embed in place. They had no `model_id` column and no versioned collection, so "re-embed in place" meant a partially migrated collection where half the vectors came from each model — a state that produces plausible-looking garbage, since the two models' spaces are unrelated.

They rebuilt properly: a new `chunks_v2` collection with `model_id`, `version` and `content_hash` populated, a backfill that skipped unchanged chunks by hash (which cut the job by the share of chunks that had not changed), and an eval run comparing v1 and v2 on a 60-question set split by language before swapping the alias. Rollback stayed a one-line config change for two weeks.

The design choice that would have saved them the mess: `model_id` on every record from day one, and a versioned collection name behind an alias.

## Common mistakes

- **Mistake:** no `model_id` on stored vectors — **Instead:** make it a required field, and refuse to serve a query whose model differs from the manifest's.
- **Mistake:** positional chunk ids (`doc7_chunk_12`) — **Instead:** content-derived ids so edits do not scramble identity.
- **Mistake:** re-embedding unchanged content — **Instead:** compare `content_hash` and skip.
- **Mistake:** mutating vectors in place during a model change — **Instead:** new collection, backfill, eval, swap alias, keep the old one briefly.
- **Mistake:** enforcing tenancy at each call site — **Instead:** one data-access function that always injects the tenant filter.
- **Mistake:** promising a re-index window without a throughput measurement — **Instead:** time 1,000 chunks and extrapolate in code.

## Checkpoint

1. Define the `ChunkRecord` for your corpus, filling every field with real values from one document. Which fields are filters, which are citations, which are debug-only?
2. Ingest 200 chunks twice: once cold and once after editing ten of them. Confirm the second run embeds only the changed chunks by counting embedding calls.
3. Measure embedding throughput on 1,000 of your chunks and produce a re-index plan for a corpus ten times larger, with a worker count and an honest time window.

**Self-check:** filters are `tenant`, `lang`, `doc_type`; citations are `source`, `page`, `heading_path`; debug-only are `model_id`, `content_hash`, `version`, `ingested_at` — and the second ingest embeds exactly the ten edited chunks, because their hashes changed while the rest still match.

## Go deeper

- [Query Transformation](05-query-transform.md) — the query side of the same record design.
- [DVC](../18-mlops-llmops/04-dvc.md) — versioning the corpus that feeds the index.
- [Multi-Tenant](../21-system-design/05-multi-tenant.md) — the architecture view of the tenancy decision.

---
[← Previous: 3. Chunking](03-chunking.md) · [Stage index](README.md) · [Next: 5. Query Transformation](05-query-transform.md)
