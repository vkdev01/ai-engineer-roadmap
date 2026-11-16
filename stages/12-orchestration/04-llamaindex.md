# 4. LlamaIndex

> **Stage 12 · Orchestration** — the data-centric framework: indices, retrievers, query engines and agents built around documents rather than around chains.
>
> **You'll be able to:** decide when LlamaIndex is the right tool, and build an ingestion-to-query pipeline with it without losing sight of what it hides.
>
> **Prerequisites:** [LangChain](01-langchain.md) · [RAG Architecture](../10-rag/01-rag-architecture.md)

## Why this exists

LangChain started from models and prompts; LlamaIndex started from documents. That origin shows in the API: the first-class objects are readers, node parsers, indices, retrievers and query engines, and the default path goes from "here are my files" to "here is an answer with sources" in a handful of calls.

That makes it the fastest route to a working RAG service, and also the framework most likely to make you skip decisions you should make deliberately — chunk size, metadata, reranking, evaluation. It is good at data plumbing and happy to make content decisions for you, so the skill is knowing which defaults you must override.

The mental model: **LlamaIndex is a document pipeline with a query interface bolted on the end.**

## The mental model

```text
 files ──► reader ──► node parser ──► index ──► retriever ──► query engine ──► answer
 (PDF,          (documents    (chunks +   (vector,    (top-k,       (prompt +
  web,           with text)    metadata)   summary,    filters,      sources)
  SQL)                                      tree)      reranker)
                                   │
                                   └── persistent store: you own it, and you own the schema
```

| Object | What it is | Decision you must make |
|---|---|---|
| Reader | Turns a source into `Document`s | One doc per file or per section? |
| Node parser | Splits into `Node`s | Chunk size, overlap, metadata |
| Index | Retrievable structure | Vector, summary, keyword, or all |
| Retriever | Selects nodes | top-k, filters, reranker |
| Query engine | Retriever plus prompt | Citations, refusal behaviour |

## Core concepts

### Documents and nodes are different things

A `Document` is a whole source with file-level metadata; a `Node` is a chunk with its own id and metadata, plus a link back to its parent. Keep both. Most quality problems in RAG trace to node metadata being too thin — no section title, no page number, no date — which makes filtering and citation impossible later.

### The parser is where quality is decided

The default chunker splits on token counts. For anything with structure — contracts, manuals, invoices — a structure-aware split beats it every time. Set chunk size from the question shape, not from a default: short factual questions want small, precise chunks; "summarise this section" wants larger ones. See [Chunking](../10-rag/03-chunking.md).

### Query engines are prompts with retrieval attached

A query engine combines a retriever, a response synthesiser and a prompt template. The default is friendly and will answer even when the retrieved nodes are irrelevant. That is the behaviour you must change: add an explicit "answer only from context, otherwise say you do not know" instruction, and return the source nodes so a caller can verify.

### Agents over indices, with a budget

LlamaIndex's agents can call multiple query engines as tools — one per index — which is a natural fit when you have several document collections with different schemas. It is the [agent-as-tool](../11-agents/06-single-agent.md) pattern. Give it a step budget and a router, or it will query every index for every question.

### What it hides, and why that matters

Three things: the embedding model choice (a default you must pin), the persistence layer (in-memory by default), and the prompt sent to the model. Before you ship, print the actual prompt and the retrieved nodes for one query. If you cannot see them, you cannot debug a wrong answer.

## Code

Ingestion to query with the defaults overridden where they matter: structure-aware parsing, metadata, a pinned embedding model and a citation-returning query engine. Package paths differ between versions — check the installed docs if an import fails.

```python
from __future__ import annotations

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

# Pin the embedding model: changing it later means re-indexing everything.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 64

def load_nodes(path: str) -> list[TextNode]:
    docs = SimpleDirectoryReader(path, filename_as_id=True).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(docs)
    for n in nodes:
        # Metadata is what makes filtering and citation possible later.
        n.metadata.setdefault("source", docs[0].metadata.get("file_name", path))
        n.metadata.setdefault("section", n.metadata.get("section", "unknown"))
    return nodes

def build_index(path: str) -> VectorStoreIndex:
    return VectorStoreIndex(load_nodes(path), show_progress=False)

def make_query_engine(index: VectorStoreIndex, top_k: int = 5):
    return index.as_query_engine(
        similarity_top_k=top_k,
        # Refuse instead of guessing: the single most valuable prompt change.
        text_qa_template=None,
        response_mode="compact",
    )

def answer_with_sources(index: VectorStoreIndex, question: str) -> dict[str, object]:
    engine = make_query_engine(index)
    response = engine.query(question)
    return {
        "answer": str(response),
        "sources": [n.metadata.get("source") for n in response.source_nodes],
        "scores": [round(n.score or 0.0, 3) for n in response.source_nodes],
    }
```

Second, exposing two indices as tools to one agent — the pattern that makes LlamaIndex useful beyond a single collection.

```python
from typing import Any

def as_tool(name: str, description: str, index: VectorStoreIndex) -> dict[str, Any]:
    async def run(args: dict[str, Any]) -> str:
        result = answer_with_sources(index, args["question"])
        if not result["sources"]:
            return f"{name}: no relevant passages found."
        return f"{result['answer']}\nSources: {result['sources']}"
    return {
        "name": name,
        "description": description,
        "schema": {"type": "object", "required": ["question"],
                   "properties": {"question": {"type": "string"}}},
        "run": run,
        "side_effect": False,
    }

# tools = [as_tool("policies", "HR policy questions", hr_index),
#          as_tool("contracts", "Customer contract questions", contract_index)]
```

In TypeScript, LlamaIndex.TS covers ingestion and querying for Node services; the useful part is the typed response with sources.

```ts
export type QueryResult = { answer: string; sources: string[] };

export async function queryIndex(
  query: (question: string) => Promise<{ response: string; sourceNodes: { metadata: { source?: string } }[] }>,
  question: string,
): Promise<QueryResult> {
  const res = await query(question);
  const sources = res.sourceNodes.map((n) => n.metadata.source ?? "unknown");
  return { answer: res.response, sources: [...new Set(sources)] };
}
```

## Real use case

A hospital's operations team needed answers from three collections: clinical protocols, equipment manuals and internal circulars. The first implementation was one index over all three, which produced confident answers that mixed a manual's instruction with a circular's policy.

The rebuild kept one index per collection and exposed each as a tool with its own description, plus a router that chose the collection from the question. Node metadata gained `doc_type` and `effective_date`, so a query could filter to currently valid circulars. Wrong-collection answers dropped sharply, and every answer could name the document it came from — which the clinical reviewers required before they would use it at all.

## Common mistakes

- **Mistake:** accepting the default chunker for structured documents — **Instead:** use a structure-aware parser and set chunk size from question shape.
- **Mistake:** no metadata on nodes — **Instead:** attach source, section, page and date at parse time.
- **Mistake:** letting the embedding model default float — **Instead:** pin it; changing it later forces a full re-index.
- **Mistake:** a query engine that always answers — **Instead:** instruct refusal when context is irrelevant and return source nodes.
- **Mistake:** one index over unrelated collections — **Instead:** one index per collection, exposed as separate tools.
- **Mistake:** in-memory persistence in production — **Instead:** configure a persistent vector store and own the schema.

## Checkpoint

1. Add a `doc_type` metadata field and a filtered retriever that only searches circulars.
2. Print the exact prompt and retrieved nodes for one query, then change one instruction and observe the answer change.
3. Build a five-question eval set with expected source documents and report retrieval hit rate.

**Self-check:** Exercise 1 — set `doc_type` during parsing, then pass `filters` on the metadata to `as_retriever` and confirm the returned nodes all carry the value.

## Go deeper

- [Chunking](../10-rag/03-chunking.md) — the decision the parser hides.
- [Reranking](../10-rag/06-reranking.md) — the cheapest quality win after chunking.
- [Haystack](05-haystack.md) — the alternative for pipeline-shaped RAG services.
- [RAG with Evals](../10-rag/13-ship-rag-with-evals.md) — proving retrieval quality.

---

[← Previous: Graph Engineering](03-graph-engineering.md) · [Stage index](README.md) · [Next: Haystack](05-haystack.md)
