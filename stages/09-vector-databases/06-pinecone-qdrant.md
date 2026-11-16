# 6. Pinecone and Qdrant

> **Stage 09 · Vector Databases** — two production-grade stores: one fully managed, one you can run yourself, both with the filtering and namespaces you need.
>
> **You'll be able to:** create a collection, upsert with payloads, and run filtered vector search in either store from Python.
>
> **Prerequisites:** [Vector Indexing Structures](04-vector-indexing.md) · [Approximate Nearest Neighbour Search](05-ann-search.md)

## Why this exists

Once your index does not fit in a process, you need a server: something that keeps vectors on disk or in memory, maintains the graph, handles concurrent queries, and survives a restart. You have two broad choices — pay someone to run it, or run it yourself.

Pinecone is the managed answer: an API, no servers, no tuning of the underlying graph beyond a few knobs. Qdrant is the self-hostable answer: an open-source engine in Rust, available as a Docker image, with payload filtering as a first-class feature.

The two are close enough in concepts that learning one teaches the other. What differs is operations: who pages at 3 a.m., and how your data leaves if you change your mind.

## The mental model

```text
                    Pinecone                          Qdrant
hosting       managed SaaS, single API          docker run / cloud / embedded local mode
unit of data  index + namespace                 collection + point (id, vector, payload)
filtering     metadata filter dict              payload filter objects, indexable fields
tuning        pod/serverless spec, top_k        HNSW params: m, ef_construct, hnsw_ef
auth          API key                           API key or none (local)
the bet       no ops, vendor dependency         ops, portability, full control
```

## Core concepts

### The data model is the same shape everywhere

A **point** (Qdrant) or **vector record** (Pinecone) is: an id, a float array, and a JSON-ish payload of metadata. Queries return ids, scores, and payloads. Every operation you need is one of four: create, upsert, query, delete.

Internalise this and you can move between engines in an afternoon. The differences live in the filter syntax and in what you are allowed to tune.

### Namespaces and payload filters solve different problems

A **namespace** (Pinecone) or a **separate collection** (Qdrant) is a hard partition: queries never cross it. Use it for tenancy or for an environment boundary — `prod` versus `staging` — where a bug must not leak data.

A **payload filter** is a soft condition inside one partition: `tenant = "acme"`, `lang = "en"`, `updated_at > "2026-01-01"`. Use it for anything users can toggle.

The rule that matters: filters must be pushed into the query, never applied after it. Both engines support that, and both let you get it wrong by fetching top-50 and filtering in your application — which returns fewer than 10 results when the filter is selective.

### Filterable fields must be declared (Qdrant) or are auto-indexed (Pinecone)

Qdrant requires you to create a payload index on a field before filtering on it is efficient; without one, filtering still works but scans. Pinecone indexes metadata automatically but has limits on filter complexity and on the size of the metadata per record. Check the current limits in each vendor's docs before designing a schema with 40 metadata fields.

### Serverless changes the cost model, not the API

Pinecone's serverless indexes bill on usage rather than on provisioned pods, and pricing and free-tier limits change — check the current docs rather than trusting a blog post. The API shape is the same, so your code does not care. What does change is latency behaviour: a serverless index may have higher tail latency than a warm pod-based one, so measure p95 under your own load.

### Qdrant's local mode is the underrated feature

`QdrantClient(path="./qdrant_data")` runs the same engine embedded in your process, on disk, with no server. It is a genuine way to develop and test: the same filter syntax and the same semantics as production, without a Docker dependency in CI.

## Code

Qdrant first: create a collection, index a filterable field, upsert with payloads, then query with a filter pushed into the engine.

```python
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

DIM = 384  # must match your embedding model

client = QdrantClient(path="./qdrant_data")   # embedded for dev; QdrantClient(url=..., api_key=...) for a server

client.recreate_collection(
    collection_name="handbook",
    vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
)
# without this, filtering on "tenant" works but scans every point
client.create_payload_index(collection_name="handbook", field_name="tenant", field_schema="keyword")

points = [
    PointStruct(id=1, vector=vec_a, payload={"tenant": "acme", "source": "leave.md", "page": 2}),
    PointStruct(id=2, vector=vec_b, payload={"tenant": "globex", "source": "leave.md", "page": 2}),
]
client.upsert(collection_name="handbook", points=points)

hits = client.query_points(                      # older clients: client.search(...)
    collection_name="handbook",
    query=query_vector,
    limit=5,
    query_filter=Filter(must=[FieldCondition(key="tenant", match=MatchValue(value="acme"))]),
    with_payload=True,
).points

for h in hits:
    print(f"{h.score:.4f}", h.payload["source"], h.payload["page"])
```

Pinecone next. Note the namespace for the hard partition and the metadata filter for the soft one — and that metadata rides along with each vector in the upsert tuple.

```python
from __future__ import annotations

import os

from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])   # never hardcode the key
index = pc.Index("handbook")                            # create it once, with a spec from current docs

index.upsert(
    vectors=[
        ("doc-1", vec_a, {"tenant": "acme", "source": "leave.md", "page": 2}),
        ("doc-2", vec_b, {"tenant": "acme", "source": "billing.md", "page": 1}),
    ],
    namespace="acme",        # hard partition: other namespaces are unreachable here
)

res = index.query(
    vector=query_vector,
    top_k=5,
    namespace="acme",
    include_metadata=True,
    filter={"source": {"$eq": "leave.md"}},   # soft filter, evaluated inside the engine
)

for m in res["matches"]:
    print(f"{m['score']:.4f}", m["metadata"]["source"], m["metadata"]["page"])
```

Both snippets assume `vec_a`, `vec_b` and `query_vector` come from your embedding model, normalised, at `DIM` dimensions. Client method names differ across SDK versions — `query_points` replaced `search` in newer `qdrant-client` releases — so check the installed version's docs when something raises `AttributeError`.

## Real use case

A healthcare-adjacent SaaS with two deployment modes — a managed cloud offering and an on-premise install for hospital customers — needed the same retrieval code in both. On-premise customers could not send documents to a third party, so the managed store was not an option for them.

They put both engines behind one internal interface: `upsert(points)`, `search(vector, k, filters)`, `delete(ids)`. Qdrant with an API key served the on-premise installs inside the customer's network; Pinecone served the cloud product with no operations burden. The interface was about 80 lines and each adapter was about 120.

Two details made it work. Both adapters mapped their native filter syntax onto one small filter model (equality and range only), which kept the surface honest instead of trying to support every predicate. And both stored `model_id` in the payload, so a re-embedding migration was the same script against either backend.

The failure mode before: the first prototype filtered by tenant in application code after fetching top-50, which silently returned 3 results for small tenants. Pushing the filter into the engine fixed it in one line per adapter.

## Common mistakes

- **Mistake:** hardcoding API keys in the script — **Instead:** read them from the environment and fail loudly when missing.
- **Mistake:** forgetting the payload index in Qdrant — **Instead:** create it at collection setup, in the same migration that creates the collection.
- **Mistake:** filtering after the query — **Instead:** pass the filter to the engine so ranking happens inside the allowed set.
- **Mistake:** mixing namespaces as if they were filters — **Instead:** namespaces for hard boundaries (tenant, environment), payload filters for toggles.
- **Mistake:** assuming self-hosted is free — **Instead:** cost the node, the disk, the backups, the upgrades, and the on-call time before comparing.
- **Mistake:** trusting a blog post for current limits or pricing — **Instead:** check the provider's docs; these change often.

## Checkpoint

1. Build the same 500-point collection in Qdrant local mode and in Pinecone. Confirm the top-5 ids and payloads match for three queries.
2. Add a `lang` payload field, index it in Qdrant, and query with a filter for `lang = "en"` plus `tenant = "acme"`. Verify that the engine returns 5 hits even when those conditions match only a small slice.
3. Write the two-adapter interface described above for `upsert` and `search` with equality filters, and run the same test suite against both backends.

**Self-check:** identical vectors with the same metric produce the same top-5 ids in both engines, up to tie-breaking at equal scores; if they differ beyond that, one side is using a different metric or a different normalisation.

## Go deeper

- [Weaviate and Milvus](07-weaviate-milvus.md) — when a schema-first or sharded engine is the better fit.
- [Chroma and pgvector](08-chroma.md) — the local and SQL-native alternatives.
- [Multi-Tenant](../21-system-design/05-multi-tenant.md) — designing the tenancy boundary properly.

---
[← Previous: 5. Approximate Nearest Neighbour Search](05-ann-search.md) · [Stage index](README.md) · [Next: 7. Weaviate and Milvus](07-weaviate-milvus.md)
