# 3. Production RAG

> **Stage 22 · Real Projects** — retrieval that serves many tenants from one index, stays fresh as documents change, and refuses to deploy when quality drops.
>
> **You'll be able to:** ship a RAG service with tenant isolation you can prove, incremental ingestion with deletion, and an eval gate that blocks a bad prompt change.
>
> **Prerequisites:** [RAG with Evals](../10-rag/13-ship-rag-with-evals.md) · [Multi-Tenant Design](../21-system-design/05-multi-tenant.md)

## Why this exists

A demo RAG works because the corpus is small, static and yours. A product breaks on three things the demo never had. Documents change and the index goes stale, so the assistant confidently quotes last quarter's policy. Documents get deleted, and the assistant surfaces a contract that no longer exists. Several customers share one index, and a retrieval bug leaks one customer's documents into another's answer — which is not a quality bug, it is an incident.

This is the project that makes you employable, because it is the one every company is actually running and most candidates cannot describe. The buyers are internal platform teams and B2B SaaS companies; the thing they pay for is not retrieval quality in the abstract, it is *trustworthy* retrieval — fresh, isolated, measured.

Mental model: an index is a *cache of the document store*, with the same correctness rules as any cache — it can be stale, it must be invalidated, and every read must be scoped to an owner.

## The mental model

```text
[source of truth: object store + row per doc version]
        |  webhook / poll / CDC
        v
[ingest worker] --hash compare--> skip | re-chunk | embed | upsert
        |                                         |
        | delete event                            v
        +------------------------------> [vector store: tenant_id, doc_id, version]
                                                  |
  query --> [auth: tenant from token] --> [retrieve: filter tenant_id AND version=live]
                                                  |
                                     [rerank] --> [answer with citations]
                                                  |
                                     [eval gate in CI: golden set + faithfulness]
```

Tenant comes from the token, never from the request body. Version comes from the document row, never from the chunk.

## Core concepts

### Scope and non-goals

In scope: many tenants, incremental ingestion, deletion, hybrid retrieval with reranking, citations, an offline eval gate, and per-tenant usage metrics.

Non-goals: fine-tuning, agentic multi-hop retrieval (that is [Agentic RAG](../10-rag/11-agentic-rag.md)), a document editor, and per-tenant physical indexes. One index with a mandatory filter is the right v1; shard later only when a tenant is large enough to justify it.

### Isolation is a filter you cannot forget

The dangerous version is `search(query, tenant_id=...)` where the caller passes the tenant. One forgotten argument and you have a leak. Make the retriever impossible to call without a tenant: construct it from the authenticated request context, so the tenant is a constructor argument and the query method has no tenant parameter at all. Then add a defensive assertion that every returned chunk's `tenant_id` matches, and a test that asserts cross-tenant retrieval returns nothing even when the query text is copied verbatim from another tenant's document.

### Freshness is versioning plus deletion, not reindexing

Give every document a monotonically increasing `version`. Ingest computes a content hash; if it matches the stored hash, skip — most poll cycles change nothing, and this is where your embedding cost goes to zero. If it differs, write new chunks with the new version and flip `live` on the document row in one transaction. Retrieval filters on the live version, so readers never see a half-updated document. Deletion writes a tombstone and hard-deletes chunks in a background sweep, because retrieval must stop immediately but storage can lag.

### Chunking is a product decision

Chunk size trades context against precision. Parent-child chunking — small chunks for matching, larger parents for the prompt — usually beats a single size, and it is cheap to implement: store `parent_id` on each chunk and fetch parents after retrieval. Always store the heading path (`Policy > Leave > Carry forward`) with the chunk; it improves both embedding and citation display. See [Parent-Child](../10-rag/08-parent-child.md).

### Data model and stack

| table | key fields |
|-------|-----------|
| `tenant` | id, name, plan, quota_tokens_month |
| `document` | id, tenant_id, source_uri, version, hash, live, deleted_at |
| `chunk` | id, tenant_id, document_id, version, parent_id, heading_path, text |
| `query_log` | id, tenant_id, query, retrieved_ids[], answer_id, latency_ms, cost_usd |
| `eval_run` | id, git_sha, prompt_version, metrics_json, passed |

Stack, loose: FastAPI, Postgres for metadata, Qdrant or pgvector for vectors, an embedding model behind your own thin wrapper so you can swap it, a cross-encoder or hosted reranker, and your eval suite from [RAG Evals](../10-rag/12-rag-evals.md). Pin nothing in the README; note that model ids and prices change and say where you check them.

### Milestones: five weeks

- Week 1 — single-tenant ingest with hashes and versions. Success: re-running ingest costs no embedding calls.
- Week 2 — retrieval with filters, hybrid search, reranking, citations. Success: golden-set recall beats your baseline.
- Week 3 — tenancy. Success: the cross-tenant test passes and the retriever cannot be built without a tenant.
- Week 4 — deletion and freshness. Success: a deleted document is unreachable within your stated window.
- Week 5 — eval gate and dashboards. Success: a deliberately worse prompt fails CI.

### Eval plan and budget arithmetic

Metrics on a frozen golden set per tenant shape (write it before tuning): **recall@k** of the document containing the answer, **answer faithfulness** by LLM judge on a sample you also read, **citation support** (does the cited chunk contain the sentence), **p95 latency**, **cost per 1,000 queries**, and **staleness** (time from source change to retrievable).

Budget, illustrative: a query embeds roughly 30 tokens, retrieves 40 chunks of 300 tokens to rerank, and sends 6 chunks of 300 tokens plus a 400-token answer budget to the model. Reranking 40 chunks costs 40 short forward passes — cheap on a local cross-encoder, priced per search on hosted ones. The dominant term is usually answer generation, then reranking, then embedding; measure all three separately or you will optimise the wrong one. For p95 under a few seconds, cap rerank candidates at 40 and stream the answer.

## Code

Isolation and freshness are the two mechanisms worth showing. First, a retriever whose tenant is fixed at construction:

```python
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


@dataclass(frozen=True)
class Chunk:
    id: str
    tenant_id: str
    document_id: str
    text: str
    heading_path: str


class Retriever:
    """Tenant is bound at construction: there is no way to pass it per call."""

    def __init__(self, client: AsyncQdrantClient, collection: str, tenant_id: str) -> None:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self._client = client
        self._collection = collection
        self._tenant = tenant_id

    async def search(self, vector: list[float], k: int = 40) -> list[Chunk]:
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
            query_filter=Filter(          # scope enforced server-side, not in Python
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=self._tenant))],
                must_not=[FieldCondition(key="deleted", match=MatchValue(value=True))],
            ),
        )
        chunks = [to_chunk(h) for h in hits]
        assert all(c.tenant_id == self._tenant for c in chunks)  # cheap tripwire
        return chunks
```

Second, ingestion that skips unchanged documents and versions changed ones atomically:

```python
import hashlib

async def ingest(document_id: str, raw: bytes, db, store) -> str:
    new_hash = hashlib.sha256(raw).hexdigest()
    row = await db.fetch_document(document_id)
    if row and row.hash == new_hash:
        return "skipped"                       # the common case, zero embed cost

    version = (row.version + 1) if row else 1
    parents = chunk_parents(extract_text(raw))
    chunks = [c for p in parents for c in chunk_children(p, size=300, overlap=40)]

    vectors = await embed_batch([c.text for c in chunks])
    await store.upsert(chunks, vectors, version=version, tenant_id=row.tenant_id)

    async with db.transaction():               # flip live only after chunks exist
        await db.write_document(document_id, version=version, hash=new_hash, live=True)
    await store.mark_superseded(document_id, keep_version=version)
    return f"indexed v{version} ({len(chunks)} chunks)"
```

Deletion is the same shape: tombstone the document row in the transaction, then let a sweep hard-delete chunks whose version is not live.

## Real use case

A B2B support product indexes each customer's help centre and internal runbooks. Before: one shared collection, tenant filtered in application code, and a prompt change shipped on Friday silently dropped recall because the chunker started splitting tables mid-row. After: the tenant is bound in the retriever, deletion is a tombstone, and every prompt or chunker change runs the golden set in CI with recall and faithfulness thresholds.

The design choice that made it work is the version-plus-tombstone model: freshness and deletion became ordinary database semantics instead of an index-rebuild script someone remembers to run.

Demo script, two minutes: ask a question and show the citation opening the exact chunk; change a source document and show the re-index and the new answer; copy tenant A's text into tenant B's query and show an empty result; show a CI run that failed on a worse prompt. Talking points: why the tenant is a constructor argument, what "live version" buys you, how you measure staleness, and what you would shard and when.

## Common mistakes

**Mistake:** passing `tenant_id` as a search parameter — **Instead:** bind it at construction and assert on results; a parameter is a thing someone forgets.
**Mistake:** re-embedding the whole corpus on every sync — **Instead:** hash, compare, skip; only changed documents cost money.
**Mistake:** deleting chunks before flipping the document version — **Instead:** write new, flip live, sweep old; readers never see a gap.
**Mistake:** tuning chunk size by feel — **Instead:** measure recall@k on the golden set for two or three configurations and keep the winner.
**Mistake:** one global eval average — **Instead:** report per tenant shape; a large tenant with easy documents hides the small one that fails.
**Mistake:** storing the prompt in the deployment image only — **Instead:** version prompts and log the version on every query, or you cannot attribute a regression.

## Checkpoint

1. Write the cross-tenant test: index two tenants, query tenant B with tenant A's exact text, assert zero hits. Run it in CI.
2. Add a staleness metric: time from a source change to a retrievable chunk. Plot it for a week of edits.
3. Add a rerank step and measure recall@5 before and after, plus p95 latency. Decide whether the quality gain pays for the latency.

**Self-check:** Exercise 1 passes only if the filter is applied server-side and the retriever cannot be constructed without a tenant — a Python-side filter would still pass this test but fails the audit.

## Go deeper

- [Multi-Tenant](../21-system-design/05-multi-tenant.md) — quotas, noisy neighbours and per-tenant limits
- [Hybrid Retrieval](../10-rag/07-hybrid-retrieval.md) and [Reranking](../10-rag/06-reranking.md) — the two quality levers with the best ratio
- [Regression Tests](../15-evals/08-regression-tests.md) — turning the eval gate into a habit
- [AI SaaS](13-ai-saas.md) — billing and quotas on top of this service

---

[← Previous: Document Intelligence](02-document-intelligence.md) · [Stage index](README.md) · [Next: Multi-Agent Research](04-multi-agent-research.md)
