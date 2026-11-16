# 2. RAG Design at Scale

> **Stage 21 · System Design** — retrieval-augmented generation as an engineering system: an indexing pipeline, a freshness contract, a sharding plan, and the arithmetic of re-embedding.
>
> **You'll be able to:** design a RAG system that stays correct as documents change, size its index and its re-embedding cost with labelled assumptions, and choose a sharding and multi-index strategy.
>
> **Prerequisites:** [RAG Architecture](../10-rag/01-rag-architecture.md) · [Vector Indexing](../09-vector-databases/04-vector-indexing.md)

## Why this exists

A demo RAG system embeds twenty PDFs once. A production one ingests documents continuously from four sources, deletes them when a customer deletes them, keeps an answer about a policy current to the day, serves several tenants without mixing their data, and survives an embedding model upgrade that invalidates every vector it holds.

Those are system design problems, not prompting problems. The retrieval quality you measure in an eval is downstream of the pipeline that built the index — chunking, freshness, deduplication, deletions — and the pipeline is where most production RAG failures actually live.

The mental model: **the index is a derived view of the truth, so it needs the same care as any materialised view: a build path, a freshness guarantee, a delete path, and a rebuild cost you have computed.**

## The mental model

```text
 sources ─► fetch ─► parse ─► clean ─► chunk ─► embed ─► upsert ─► index
   │         │        │        │        │        │         │        │
   │      changed?  text+meta  dedupe  by doc   batch     version  shard
   │         │                 (hash)  structure          tag      key
   └── deletes ─────────────────────────────────────────► delete path
                                                          (must exist)
 query ─► embed ─► filter(tenant, acl) ─► ANN + BM25 ─► rerank ─► top-k
```

| Concern | Decision to make | Default that works |
|---|---|---|
| Freshness | how stale may an answer be? | minutes for tickets, hours for docs, daily for archives |
| Delete | how does a delete propagate? | delete by `doc_id`, verified by a count check |
| Sharding | one index or many? | one collection per tenant above a size threshold |
| Versioning | how do you re-embed? | `embedding_model` tag on every row, dual-write then switch |
| Cost | what does a rebuild cost? | compute it before you need it (see arithmetic below) |

## Core concepts

### The indexing pipeline is where quality is won

Five stages, each with a failure that shows up as a bad answer:

1. **Fetch** — track a content hash per source document. Re-embedding an unchanged document is pure waste, and skipping a changed one is a stale answer.
2. **Parse** — PDFs, HTML and spreadsheets lose structure differently. Keep headings, tables and page numbers as metadata; a chunk without its heading is ambiguous.
3. **Chunk** — split on document structure (headings, sections), with overlap, and carry parent ids so you can expand to a bigger context at answer time (see [Parent-Child](../10-rag/08-parent-child.md)).
4. **Deduplicate** — near-identical copies of the same policy inflate the top-k and crowd out other documents. Hash normalised text and collapse.
5. **Embed in batches** — batch for throughput, record the model id and dimension on every row, and make the write idempotent on `(doc_id, chunk_index, embedding_model)` so a retry does not duplicate.

### Freshness is a contract, not a hope

Write it down: "an answer about a policy reflects changes within 15 minutes". Then build the mechanism that meets it: incremental sync on a schedule or a change feed, a priority queue for documents a user is actively asking about, and a `indexed_at` timestamp on every chunk so you can measure staleness. The most common production complaint about RAG is not relevance — it is "it gave me last quarter's number".

Also design the **delete** path at the same time as the write path. A document removed from the source but present in the index is a data-protection problem, not just a quality problem (see [Governance](../20-ai-security/08-governance.md)). Verify deletes with a count query, and test them.

### Sharding and multi-index

Three reasons to shard: **isolation** (a tenant's data must not be retrievable by another), **size** (an index larger than memory degrades), and **heterogeneity** (code chunks and prose chunks want different chunk sizes and rerankers).

The practical ladder: start with one collection and a mandatory tenant filter (see [Multi-Tenant Design](05-multi-tenant.md)); move to one collection per tenant when a tenant's volume justifies it or a contract requires physical separation; use separate indexes per *content type* when chunking strategy differs materially. Each index you add is another pipeline, another re-embedding job and another place for a delete to fail — add them for a reason you can state.

### Re-embedding cost, with the arithmetic shown

An embedding model upgrade means re-embedding everything. Estimate it before the vendor deprecates the model, not after.

Labelled example — **all figures illustrative, substitute your own**:

| Assumption | Value |
|---|---|
| Chunks in the index | 2,000,000 |
| Average chunk length | 400 tokens |
| Total tokens to embed | 2,000,000 × 400 = 800,000,000 |
| Embedding price (placeholder) | `$E` per 1,000,000 tokens — check current pricing |
| Embedding throughput (placeholder) | 500 chunks/s at batch size 128 on your workers |

Cost: `800 × $E`. If `$E` were 0.02, that is 800 × 0.02 = **$16** — cheap, which is why re-embedding is usually a *time and correctness* problem rather than a money problem. Time: `2,000,000 / 500 = 4,000 s ≈ 67 minutes` of pure embedding, plus fetch and parse, so plan a few hours of wall clock.

Storage: `2,000,000 chunks × 1,024 dimensions × 4 bytes = 8,192,000,000 bytes ≈ 8.2 GB` of raw float32 vectors, before index overhead and metadata — roughly double it for a graph or HNSW index. Query load: at 50 queries/s with 1,024-dimension vectors, you are scanning for neighbours 50 times a second, so index type and memory matter more than raw vector bytes.

The design consequence: use **dual-write** — write new vectors tagged with the new model id alongside the old, verify with an eval, then switch reads and delete the old. A big-bang re-embed means an outage window you cannot roll back.

## Code

An idempotent upsert with content hashing and model tagging — the two fields that make incremental sync and re-embedding safe. The hash skips unchanged documents; the model tag makes a partial migration legible.

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    heading: str | None


def content_hash(text: str) -> str:
    normalised = " ".join(text.lower().split())  # whitespace-insensitive
    return hashlib.sha256(normalised.encode()).hexdigest()


def chunk_key(c: Chunk, model_id: str) -> str:
    # Idempotency key: retries and re-runs cannot create duplicates.
    return f"{c.doc_id}:{c.chunk_index}:{model_id}"


async def sync_document(store, embedder, doc_id: str, text: str,
                        model_id: str) -> dict[str, int]:
    digest = content_hash(text)
    previous = await store.get_doc_hash(doc_id)
    if previous == digest:
        return {"embedded": 0, "skipped": 1}  # unchanged: no cost, no work

    chunks = split_on_headings(text)  # structure-aware; keep parent ids
    vectors = await embedder.embed([c.text for c in chunks], model=model_id)

    rows = [
        {
            "key": chunk_key(c, model_id),
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "embedding": vec,
            "embedding_model": model_id,  # makes partial migrations visible
            "content_hash": digest,
        }
        for c, vec in zip(chunks, vectors)
    ]
    await store.upsert_many(rows)          # idempotent on `key`
    await store.set_doc_hash(doc_id, digest)
    return {"embedded": len(rows), "skipped": 0}


async def delete_document(store, doc_id: str) -> int:
    removed = await store.delete_by_doc(doc_id)
    remaining = await store.count_by_doc(doc_id)
    if remaining:
        raise RuntimeError(f"delete incomplete for {doc_id}: {remaining} rows left")
    return removed
```

The freshness and rebuild arithmetic as a script, so the numbers in your design doc are reproducible rather than remembered.

```python
from __future__ import annotations

CHUNKS = 2_000_000
TOKENS_PER_CHUNK = 400
DIMS = 1024
BYTES_PER_FLOAT = 4
CHUNKS_PER_SEC = 500
EMBED_PRICE_PER_MTOK = 0.02  # PLACEHOLDER: substitute current published price

total_tokens = CHUNKS * TOKENS_PER_CHUNK
print(f"tokens to embed      : {total_tokens:,}")
print(f"embedding cost (USD) : {total_tokens / 1e6 * EMBED_PRICE_PER_MTOK:,.2f}")
print(f"embedding time (h)   : {CHUNKS / CHUNKS_PER_SEC / 3600:,.2f}")
print(f"vector bytes (GB)    : {CHUNKS * DIMS * BYTES_PER_FLOAT / 1e9:,.1f}")
print(f"with index overhead  : ~{CHUNKS * DIMS * BYTES_PER_FLOAT * 2 / 1e9:,.1f} GB")
```

## Real use case

**A compliance-policy assistant for a mid-sized bank.** Roughly 40,000 policy documents, revised on a rolling schedule, with a regulator-facing requirement that a deleted policy must not be retrievable. The first build re-indexed everything nightly: the job took most of the night, so answers were up to 24 hours stale, and the delete path did not exist at all — a withdrawn circular stayed answerable for a day after withdrawal.

The rebuild made three changes. Content hashing made the nightly job incremental, so it ran in minutes and could run hourly, which turned freshness into a stated contract. The delete path became a first-class operation with a verification count and a test. Every chunk got an `embedding_model` tag, so when the embedding model was upgraded the team dual-wrote new vectors, compared retrieval quality on a golden set, switched reads, then removed the old rows — no outage, no rollback risk. Index size was around two million chunks, which made the re-embedding arithmetic a few dollars and a couple of hours, not a project.

## Common mistakes

- **Mistake:** rebuilding the whole index nightly — **Instead:** hash content and sync incrementally, so the job is cheap enough to run often.
- **Mistake:** no delete path until someone asks — **Instead:** design delete alongside write, and verify it with a count check in tests.
- **Mistake:** storing vectors without the embedding model id and dimension — **Instead:** tag every row; it is what makes a migration possible and debuggable.
- **Mistake:** one giant shared collection with a post-filter — **Instead:** filter inside the query, and shard per tenant or per content type when you can state why.
- **Mistake:** quoting re-embedding cost from memory — **Instead:** recompute from chunk count, token count and the current published price; write the arithmetic in the doc.

## Checkpoint

1. Compute the re-embedding cost and wall-clock time for your own index using the script above. State each assumption and which one you are least sure of.
2. Add a `content_hash` check to a sync job and measure how much work the second run does when nothing changed.
3. Write the freshness contract for one RAG system in a single sentence, then list the mechanism that enforces it and how you would measure a violation.

**Self-check:** The second run should embed zero chunks and skip every document — if it re-embeds, the hash is computed over unstable text (whitespace, timestamps, or HTML that changes per fetch).

## Go deeper

- [Multi-Tenant Design](05-multi-tenant.md) — isolation levels for the index.
- [Caching Layers](06-caching.md) — caching retrieval results and their invalidation.
- [Embedding Storage](../10-rag/04-embedding-storage.md) — the mechanics behind the upsert above.
- [Ship: Design Document](10-ship-design-doc.md) — putting this arithmetic into a review.

---
[← Previous: Scalable AI Apps](01-scalable-ai-apps.md) · [Stage index](README.md) · [Next: Agent Design](03-agent-design.md)
