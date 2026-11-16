# 7. Weaviate and Milvus

> **Stage 09 · Vector Databases** — the heavier engines: Weaviate's schema-first GraphQL-flavoured model, and Milvus's distributed, index-choice-heavy design.
>
> **You'll be able to:** decide when either engine is worth its operational weight, and write a working create/insert/search flow in both.
>
> **Prerequisites:** [Pinecone and Qdrant](06-pinecone-qdrant.md) · [Vector Indexing Structures](04-vector-indexing.md)

## Why this exists

For most projects under a few million vectors, Qdrant, Chroma or pgvector is enough and you should stop reading there. Two situations push past that: you need **built-in vectorisation and hybrid modules** so the database calls the model for you (Weaviate's angle), or you need to **scale past a single node with billions of vectors and control over index type** (Milvus's angle).

Both are real capabilities and both cost real operational effort. This file is about knowing which one you are buying, and what you will be maintaining afterwards.

## The mental model

```text
                     Weaviate                              Milvus
shape          schema-first: classes, properties,      collection with typed schema,
               vectorizer modules built in             index chosen per field
query          GraphQL / client with near_vector,      client.search with metric and
               near_text, hybrid                      search_params per call
killer feature hybrid (BM25 + vector) in one query,   index zoo (HNSW, IVF, DiskANN,
               modules vectorize text for you         GPU indexes), sharding, partitions
runs as        single binary / docker, or cloud        standalone (embedded etcd+MinIO)
                                                       or full distributed cluster
cost of entry  low: one container                      high: object store, metadata store,
                                                       message queue for distributed mode
```

## Core concepts

### Weaviate: the schema is the API

Weaviate asks you to define a collection (historically a "class") with typed properties before inserting. That sounds like ceremony, and it is, but it buys two things: BM25 keyword search works out of the box because every text property is indexed for it, and filters are type-checked rather than stringly-typed.

The distinctive feature is **modules**. Configure a vectorizer module and Weaviate calls the embedding model itself: you insert `{"text": "..."}` and it stores the vector. That removes a whole service from your architecture — at the cost of coupling your data model to the database's supported model list, and paying for embedding inside the database's call path.

Its second distinctive feature is **native hybrid search**: one query that runs BM25 and vector search and fuses them with a relative-score weight, rather than you implementing fusion in application code. That is genuinely convenient; [Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) shows the DIY version and when it is better.

### Milvus: choose your index, choose your scale

Milvus separates the compute and storage concerns of a vector database and can run as a standalone process or as a distributed cluster with separate components for metadata (etcd), object storage (MinIO or S3), and a message log. Standalone is one container and fine for development; the distributed deployment is a real system to operate.

What it offers in return: a wide range of index types (HNSW, IVF variants, disk-based indexes for collections larger than memory, GPU indexes), partitioning by field value, and sharding across query nodes. If your problem is genuinely "a billion vectors, memory is the constraint", that disk-based index family is the reason people pick it.

Index parameters are per-field, set at index-creation time, and `search_params` are per-query — the same `ef`/`nprobe` story as everywhere else, just spelled in Milvus's own parameter names.

### The decision that actually matters

Ask three questions in order:

1. **Does the corpus fit on one node's RAM with room to spare?** If yes, a simpler store is almost always the right answer.
2. **Do you want the database calling your embedding model?** If yes, Weaviate's modules are a real saving; if no, they are a coupling you will fight when you change models.
3. **Do you need more vectors than one node can hold?** If yes, Milvus's disk-based indexes and sharding are the honest answer, and you should plan for the operational work.

If none of the three is a strong yes, the correct move is to stay on a lighter store.

## Code

Weaviate: define a collection with a filterable text property, insert with an explicit vector (vectorizer disabled so the model choice stays yours), then query.

```python
from __future__ import annotations

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

client = weaviate.connect_to_local()          # or connect_to_weaviate_cloud(...)

if not client.collections.exists("Chunk"):
    client.collections.create(
        name="Chunk",
        vectorizer_config=Configure.Vectorizer.none(),   # we supply vectors ourselves
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="tenant", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
        ],
    )

chunks = client.collections.get("Chunk")
chunks.data.insert(properties={"text": "Carry over up to 10 days of leave.",
                               "tenant": "acme", "source": "leave.md"},
                   vector=vec_a)               # normalised, 384 floats

res = chunks.query.near_vector(
    near_vector=query_vector,
    limit=5,
    return_properties=["text", "source"],
    filters=Filter.by_property("tenant").equal("acme"),   # server-side filter
)
for obj in res.objects:
    print(f"{obj.metadata.distance:.4f}", obj.properties["source"])
```

Milvus: schema first, index with explicit parameters, then a search that carries its own `ef`.

```python
from __future__ import annotations

from pymilvus import DataType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")   # standalone for dev

if not client.has_collection("chunks"):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=384)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("tenant", DataType.VARCHAR, max_length=64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},   # build-time cost
    )
    client.create_collection("chunks", schema=schema, index_params=index_params)

client.insert("chunks", data=[{"id": 1, "vector": vec_a,
                               "text": "Carry over up to 10 days of leave.",
                               "tenant": "acme"}])

hits = client.search(
    collection_name="chunks",
    data=[query_vector],
    limit=5,
    filter='tenant == "acme"',
    search_params={"params": {"ef": 64}},          # query-time recall/latency knob
    output_fields=["text", "tenant"],
)
for h in hits[0]:
    print(f"{h['distance']:.4f}", h["entity"]["text"][:50])
```

Both snippets assume `vec_a` and `query_vector` come from your own model at 384 dimensions. SDK call shapes differ between major versions — check the installed version's docs, and note that Milvus's field names for distance and entity depend on the version too.

## Real use case

A news archive with roughly 300 million article paragraphs needed search across 14 languages, with per-language and per-date filters. They had started on a single-node store, which stopped fitting in memory at about a fifth of the corpus, and the naive fix — bigger machine — was not available at their budget.

They moved to a Milvus distributed deployment with a disk-based index for the vector field and partitions by language, so a Hindi query never touched the English segments. Their filterable fields (language, publication, date) went into the schema as scalar fields. Recall was measured against a flat index over a 1-million-vector sample before cutover, and the harness from [Approximate Nearest Neighbour Search](05-ann-search.md) was reused unchanged.

Two things went wrong and are worth planning for: the message-log component needed its own disk sizing, and the initial index build took long enough that they had to schedule it as a batch job with progress reporting. Neither is exotic; both are the operational weight that this file is warning about.

They evaluated Weaviate too, and rejected it for this project only because they wanted to own the embedding model per language rather than take the database's supported list.

## Common mistakes

- **Mistake:** choosing a distributed engine for a 200k-vector corpus — **Instead:** start on a single node and migrate only when memory forces it.
- **Mistake:** enabling a built-in vectorizer and then changing embedding models — **Instead:** either accept the coupling deliberately, or disable the module and supply vectors yourself.
- **Mistake:** setting Milvus `search_params` once in code and never tuning — **Instead:** sweep `ef` and record recall against a flat reference, exactly as in the previous files.
- **Mistake:** treating a Weaviate schema change as free — **Instead:** plan property additions before ingest; altering types later is painful.
- **Mistake:** forgetting the scalar index or partition key for a field you filter on constantly — **Instead:** declare partitions and index the filter fields you know you will use.
- **Mistake:** benchmarking a distributed cluster with an empty collection — **Instead:** test at realistic size and segment count, since behaviour changes with both.

## Checkpoint

1. Run Weaviate in Docker, create the `Chunk` collection above, and insert 100 vectors. Query with and without the tenant filter and confirm the filter reduces the result set server-side.
2. Do the same in Milvus standalone. Compare the two queries' latency for the same 100 vectors and explain the difference in two sentences.
3. Write a one-page decision note for a corpus of 50 million vectors that must support per-language filtering, comparing Milvus distributed against a sharded single-node deployment.

**Self-check:** the filtered query returns up to 5 results from the matching tenant only, and the unfiltered one returns the global top 5 — the difference proves the filter is evaluated before the limit, not after.

## Go deeper

- [Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) — the fusion Weaviate does for you, built by hand.
- [Scaling Trade-offs](../21-system-design/09-scaling-tradeoffs.md) — sharding and partitioning decisions.
- [Ship: A Semantic Search Service](10-ship-semantic-search.md) — the project that ties this stage together.

---
[← Previous: 6. Pinecone and Qdrant](06-pinecone-qdrant.md) · [Stage index](README.md) · [Next: 8. Chroma and pgvector](08-chroma.md)
