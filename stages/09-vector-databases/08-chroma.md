# 8. Chroma and pgvector

> **Stage 09 · Vector Databases** — the two lowest-friction options: Chroma for a local prototype in five lines, pgvector for vectors living inside the database you already run.
>
> **You'll be able to:** prototype retrieval with Chroma and move the same collection into Postgres with an HNSW index and SQL filtering.
>
> **Prerequisites:** [Semantic Search End to End](03-semantic-search.md) · [SQL and PostgreSQL](../03-data-engineering/04-sql-postgresql.md)

## Why this exists

Most projects do not need a vector database. They need a table with a vector column, or a folder with a local index. Reaching for a managed service on day one adds a network hop, a bill, a credential, and a second source of truth — for a corpus that fits in RAM.

Chroma gives you the fastest possible start: persist to a directory, add documents, query. pgvector gives you the most boring possible production answer: vectors are rows, filters are `WHERE` clauses, backups are your existing backups, and your team already knows how to operate it.

The mental model: **both are "a table plus a nearest-neighbour operator", with very different operational stories.**

## The mental model

```text
Chroma (local-first)                    pgvector (SQL-native)
────────────────────                    ────────────────────
PersistentClient("./chroma")            CREATE EXTENSION vector
  collection                            table: id, tenant, content, embedding vector(384)
    id / document / metadata / vector    index: hnsw (embedding vector_cosine_ops)
  query(n_results, where=...)            SELECT ... ORDER BY embedding <=> $1 LIMIT 5
                                        filters are WHERE, joins are joins
embedded in your process;               separate server; one source of truth
great for prototypes and tests          great for products that already use Postgres
```

## Core concepts

### Chroma stores documents, not just vectors

Chroma's unit is a record with an `id`, an optional `document` string, an optional `metadata` dict, and an `embedding`. If you do not pass embeddings, Chroma computes them with its configured embedding function — convenient for a prototype, and a hidden dependency for production, since the model is then part of the database's configuration rather than your code.

Pass embeddings explicitly and the model choice stays in your application. That is the same trade you saw with Weaviate's modules.

### Chroma's distance is a collection setting

The metric is set when the collection is created, and it cannot be changed afterwards — you recreate and re-add. Set `{"hnsw:space": "cosine"}` explicitly rather than relying on a default, because the default has changed across versions and the resulting scores land on different scales.

`where` supports equality and `$in`, `$and`, `$or` on metadata. Metadata values must be scalars (string, int, float, bool) — nested dicts are not filterable, so flatten anything you plan to filter on.

### pgvector is ordinary SQL, with three operators

Create the extension, add a `vector(n)` column, and query with a distance operator:

| Operator | Distance | Use with |
|---|---|---|
| `<=>` | cosine distance = `1 − cosine similarity` | `vector_cosine_ops` |
| `<#>` | negative inner product | `vector_ip_ops` |
| `<->` | euclidean (L2) | `vector_l2_ops` |

Note that `<=>` returns **distance**, so `ORDER BY embedding <=> $1` sorts ascending and the best match is smallest. To get a similarity you can show a user, compute `1 - (embedding <=> $1)`. Getting this backwards is the single most common pgvector bug.

### Index type: HNSW by default, IVFFlat when building is too slow

pgvector supports HNSW and IVFFlat. HNSW gives better recall per query at the cost of a slower build and more memory; IVFFlat builds fast but needs a training pass and a `lists` parameter tuned to row count. For most corpora HNSW with default `m` and `ef_construction`, plus a tuned `hnsw.ef_search` per session, is the right starting point.

### Filtering plus ANN is the sharp edge in pgvector

A `WHERE tenant = 'acme'` combined with `ORDER BY embedding <=> ...` can behave badly: the planner may scan the HNSW index and discard non-matching rows, so a selective filter yields fewer rows than your `LIMIT`. Newer pgvector versions offer iterative index scans to mitigate this; on older versions, the practical fixes are to over-fetch and re-filter, to use partial indexes per tenant, or to accept a filtered exact scan when the filter is very selective.

This is the one place where a purpose-built vector store is meaningfully ahead: it plans filters and ANN together. Say so honestly in your design note.

## Code

Chroma, with embeddings supplied by you and an explicit metric, plus a metadata filter.

```python
from __future__ import annotations

import chromadb

client = chromadb.PersistentClient(path="./chroma_data")   # a directory, not a server

collection = client.get_or_create_collection(
    name="handbook",
    metadata={"hnsw:space": "cosine"},     # set at creation; cannot be changed later
)

collection.add(
    ids=["c1", "c2", "c3"],
    embeddings=[vec_a, vec_b, vec_c],      # normalised, from your own model
    documents=["Carry over up to 10 days of leave.",
               "Refunds within 30 days of purchase.",
               "Report lost devices within 24 hours."],
    metadatas=[{"tenant": "acme", "source": "leave.md"},
               {"tenant": "acme", "source": "billing.md"},
               {"tenant": "globex", "source": "security.md"}],
)

res = collection.query(
    query_embeddings=[query_vector],
    n_results=2,
    where={"tenant": "acme"},              # metadata filter, applied inside the query
    include=["documents", "metadatas", "distances"],
)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"{1 - dist:.4f}", meta["source"], doc[:40])   # convert distance to similarity
```

pgvector: schema, index, and a query that returns similarity while filtering on tenant.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id        bigserial PRIMARY KEY,
    tenant    text      NOT NULL,
    source    text      NOT NULL,
    content   text      NOT NULL,
    embedding vector(384) NOT NULL          -- dimension must match your model
);

-- HNSW with cosine distance; build after bulk load for speed
CREATE INDEX chunks_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX chunks_tenant_idx ON chunks (tenant);

SET hnsw.ef_search = 64;                    -- per session; raise for better recall

SELECT id, source, content,
       1 - (embedding <=> $1) AS similarity   -- <=> is distance, so subtract from 1
FROM chunks
WHERE tenant = $2
ORDER BY embedding <=> $1                     -- ascending distance = best first
LIMIT 5;
```

Insert with `psycopg` by passing the vector as a string like `'[0.021,-0.118,...]'`, or with the `pgvector` Python package's adapter, which handles the serialisation for you. Always parameterise `$1`; never build SQL by string formatting an embedding.

## Real use case

A 30-person company ran an internal knowledge assistant on Postgres already — it held users, documents metadata and permissions. Their first vector prototype used a separate hosted vector store, which meant two systems to keep in sync, and permissions had to be duplicated. A permission change in Postgres did not propagate, and for a week some staff could retrieve chunks from documents they had lost access to.

They moved the vectors into the same database: a `chunks` table with a `vector(384)` column, an HNSW index, and permissions enforced by joining to their existing access table in the same query. Deleting a document became a cascade, not a second API call.

The cost was honest: filtered ANN in pgvector needed care, so they used partial indexes for their two largest tenants and an over-fetch-then-refilter pattern elsewhere, and they documented the over-fetch factor with measured numbers. Their recall harness from [Approximate Nearest Neighbour Search](05-ann-search.md) ran against `1 - (embedding <=> ...)` scores with no changes.

Chroma was used for the whole development cycle — same chunking code, local directory, no server — which is what let a two-person team iterate before touching production infrastructure.

## Common mistakes

- **Mistake:** relying on Chroma's default embedding function in production — **Instead:** pass embeddings explicitly so the model lives in your code and your config.
- **Mistake:** assuming Chroma's default distance is cosine — **Instead:** set `{"hnsw:space": "cosine"}` at creation and store that choice in your docs.
- **Mistake:** sorting pgvector `<=>` results descending — **Instead:** `ORDER BY ... <=>` ascending; convert to similarity only for display.
- **Mistake:** creating the HNSW index before a bulk load — **Instead:** load first, then index, or expect a long insert path.
- **Mistake:** combining a very selective `WHERE` with HNSW and expecting `LIMIT` rows — **Instead:** over-fetch and refilter, use partial indexes, or enable iterative scans.
- **Mistake:** nesting metadata dicts in Chroma and expecting to filter on them — **Instead:** flatten to scalar fields you can actually query.

## Checkpoint

1. Build the Chroma collection above and query it with and without `where`. Confirm the filter changes the result set and that `1 - distance` behaves like a similarity.
2. Create the pgvector table, insert 1,000 rows, and run the query with `hnsw.ef_search` at 10 and at 200. Compare latency and whether the top-5 changes.
3. Take one tenant with a selective filter and compare three approaches: plain filtered HNSW, over-fetch 20× then refilter, and a partial index for that tenant. Report rows returned and p95 latency for each.

**Self-check:** the filtered query returns only rows for that tenant, and `1 - (embedding <=> $1)` produces a number that rises for better matches while the `ORDER BY` still uses the raw distance ascending.

## Go deeper

- [Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) — add Postgres full-text search to the same table.
- [SQL and PostgreSQL](../03-data-engineering/04-sql-postgresql.md) — the query planning basics behind the filtering caveat.
- [Ship: A Semantic Search Service](10-ship-semantic-search.md) — use either of these as your project backend.

---
[← Previous: 7. Weaviate and Milvus](07-weaviate-milvus.md) · [Stage index](README.md) · [Next: 9. Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md)
