# Stage 09 — Vector Databases

This stage turns text into numbers and then into a searchable index. It sits after prompts because retrieval is the first thing an LLM application needs that a prompt alone cannot give it: knowledge the model was never trained on, filtered by who is asking.

**Time:** 2–3 weeks · **You need first:** Stage 06 (LLM Fundamentals) and Stage 08 (Prompt and Context) · **You leave with:** a semantic search service with a measured recall number and a query API

## What you will be able to do

- Explain what an embedding is geometrically and why cosine, dot product and euclidean distance rank results identically once vectors are unit-length.
- Choose between flat, IVF and HNSW indexes using a recall/latency/memory trade-off you can defend with numbers.
- Measure recall@k honestly against brute force, instead of trusting a demo that "looks right".
- Stand up the same collection in Qdrant, Chroma and pgvector, and know when a managed service earns its cost.
- Fuse BM25 and vector results with reciprocal rank fusion and explain why raw score addition fails.
- Ship a search endpoint that returns ranked hits, latency, and a recall report.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Embeddings: Text as Points in Space](01-embeddings.md) | The geometry: direction is meaning, length is not |
| 2 | [Similarity Metrics](02-similarity-metrics.md) | Cosine, dot, L2 and when they agree |
| 3 | [Semantic Search End to End](03-semantic-search.md) | Embed, store, query, rank, return — the whole loop |
| 4 | [Vector Indexing Structures](04-vector-indexing.md) | Flat, IVF, HNSW and what each one costs |
| 5 | [Approximate Nearest Neighbour Search](05-ann-search.md) | Tuning nprobe and efSearch, measuring recall |
| 6 | [Pinecone and Qdrant](06-pinecone-qdrant.md) | Managed and self-hosted, with metadata filters |
| 7 | [Weaviate and Milvus](07-weaviate-milvus.md) | The heavier options: schemas, shards, scale |
| 8 | [Chroma and pgvector](08-chroma.md) | The local-first path and SQL-native vectors |
| 9 | [Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) | Keyword plus vector, fused by rank not score |
| 10 | [Ship: A Semantic Search Service](10-ship-semantic-search.md) | The project: recall report plus query API |

## The shipping project

[10-ship-semantic-search.md](10-ship-semantic-search.md) — you build a search service over a document corpus you choose, with a `/search` endpoint, metadata filtering, and a script that measures recall@10 against brute-force ground truth and prints a before/after table for two index settings.

## How to know you are done

- You can compute cosine similarity by hand for two small vectors and say what the number means.
- Your index has a recall@10 figure you measured yourself, with the query set and k stated.
- You can name the memory cost of your collection: vectors × dimensions × bytes, plus index overhead.
- Your search API returns results and per-stage latency, and rejects malformed filters.
- You can explain, in two sentences, why hybrid search beats pure vectors on part-number queries.

---
[← Back to roadmap](../../README.md) · [Next stage: RAG](../10-rag/README.md)
