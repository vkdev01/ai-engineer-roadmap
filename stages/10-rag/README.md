# Stage 10 — RAG

Retrieval-augmented generation is the workhorse of applied AI: fetch the right context, put it in the window, answer from it. It is also the stage where most systems quietly fail, because retrieval looks fine in a demo and collapses on the tenth real question.

This stage builds it properly — ingestion, chunking, storage, query transformation, reranking, hybrid retrieval, parent-child, graph and agentic variants — and measures every step, so you always know which part is losing your recall.

**Time:** 3-4 weeks · **You need first:** [Stage 09 — Vector Databases](../09-vector-databases/README.md) · **You leave with:** a RAG service with a measured baseline, citations and an eval harness

## What you will be able to do

- Draw a RAG architecture and say where each failure mode appears
- Parse real documents — PDFs, tables, scanned pages — into retrievable text
- Choose a chunking strategy from evidence, not from a blog post
- Rewrite, expand and decompose queries to fix retrieval misses
- Add a reranker and hybrid retrieval, and measure what each bought
- Serve small chunks to the index and large sections to the model
- Build graph and agentic retrieval when the question needs multi-hop reasoning
- Score context recall, precision, faithfulness and answer relevance

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [RAG Architecture](01-rag-architecture.md) | The pipeline, and where it breaks |
| 2 | [Ingestion and Parsing](02-ingestion-parsing.md) | PDFs, tables, layout and OCR fallbacks |
| 3 | [Chunking](03-chunking.md) | Size, overlap and structure, chosen by measurement |
| 4 | [Embedding and Storage Design](04-embedding-storage.md) | Metadata, filters, multi-tenancy, re-indexing |
| 5 | [Query Transformation](05-query-transform.md) | Rewriting, HyDE and decomposition |
| 6 | [Reranking](06-reranking.md) | Cross-encoders on a shortlist |
| 7 | [Hybrid Retrieval](07-hybrid-retrieval.md) | BM25 plus vectors, fused |
| 8 | [Parent-Child Retrieval](08-parent-child.md) | Index small, answer with the whole section |
| 9 | [Multi-Query Retrieval](09-multi-query.md) | Several phrasings, merged results |
| 10 | [Graph RAG](10-graph-rag.md) | Entities and relations for multi-hop questions |
| 11 | [Agentic RAG](11-agentic-rag.md) | Letting the model decide what to search |
| 12 | [RAG Evaluation](12-rag-evals.md) | Context recall, faithfulness, answer relevance |
| 13 | [Ship: A RAG Service with Evals](13-ship-rag-with-evals.md) | The whole thing, measured and served |

## The shipping project

[Ship: A RAG Service with Evals](13-ship-rag-with-evals.md) — you take a real corpus, build the naive pipeline, measure it, then add hybrid retrieval, a reranker and parent-child in three measured steps. You ship a service that answers with citations and a README that reports each step's score.

## How to know you are done

- You can state your retrieval score before and after every change you made
- You have a question in your eval set whose answer spans two chunks, and your pipeline now handles it
- Your answers carry citations you can click through to the source
- You know which of your failures are retrieval failures and which are generation failures, and you can show the split
- Your eval harness runs in one command against a frozen dataset

---
[← Back to roadmap](../../README.md) · [Next stage: AI Agents](../11-agents/README.md)
