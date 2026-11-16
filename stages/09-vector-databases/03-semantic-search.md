# 3. Semantic Search End to End

> **Stage 09 · Vector Databases** — assembles embed, store, query, rank and return into one loop you can run before any database exists.
>
> **You'll be able to:** build a complete semantic search over a folder of documents in one script, and name the five stages where it can go wrong.
>
> **Prerequisites:** [Embeddings: Text as Points in Space](01-embeddings.md) · [Similarity Metrics](02-similarity-metrics.md)

## Why this exists

Every vector database in the rest of this stage is an optimisation of one loop. If you do not have the loop working in its dumbest possible form — a numpy array and a dot product — you cannot tell whether a bad result came from the index, the embeddings, or the chunking. People reach for Pinecone on day one and then spend a week blaming it for a problem that was 300-word chunks.

So this file builds the whole thing with no infrastructure. Load text, chunk it, embed it, keep a matrix in memory, score a query, return top-k with source attribution.

The mental model: **semantic search is `argmax` over a dot product, plus the bookkeeping that tells you which document the winning vector came from.**

## The mental model

```text
offline (once per corpus)                     online (per query, ~ms)
─────────────────────────                     ────────────────────────
files ─► parse ─► chunk ─► embed ─► matrix     query ─► embed ─► dot ─► top-k ─► hydrate ─► JSON
                                     + ids                        │
                                     + metadata                   └─ filter by tenant,
                                                                    language, date
```

Five failure points, in order of how often they bite: bad parsing (text lost), bad chunking (answer split in half), wrong embedding model or missing prefix, no metadata filtering (wrong tenant's document returned), and a metric mismatch.

## Core concepts

### Chunking is part of search, not an afterthought

An embedding compresses a passage into one vector. If a chunk contains three topics, the vector is an average of three topics and matches none of them well. If a chunk is one sentence, it loses the context that made the sentence meaningful. Start at roughly 200–500 tokens with a small overlap, and treat the number as a parameter to test — [Chunking](../10-rag/03-chunking.md) does this properly.

### The id-to-text mapping is the actual product

The vector index returns integers: row 4,217. Useless on its own. You need a parallel structure — a list, a table, a payload — that maps row 4,217 to `{"text": ..., "source": "handbook.pdf", "page": 12}`. Every vector store calls this differently (payload, metadata, stored fields), but it always exists. Without it you can rank but you cannot cite, and an uncited answer is not trustworthy in any product.

### Filters are applied before ranking, or the ranking lies

If a user can only see their own tenant's documents, you must exclude other tenants *before* you take the top 10, not after. Filtering after top-k gives you fewer than 10 results, or wrong ones. Every real store supports a metadata filter pushed into the search; in the numpy version you mask the score array. This distinction is the reason [Embedding and Storage Design](../10-rag/04-embedding-storage.md) insists on filterable metadata from the start.

### Normalise once, then dot product everywhere

Normalise at write time and the query path becomes a single matrix-vector multiply. For 50,000 chunks at 384 dimensions that is about 19 million multiply-adds — a few milliseconds in numpy, single-digit milliseconds in a compiled index. You do not need an ANN index until you are in the hundreds of thousands of vectors, or you need sub-10ms at high concurrency.

### Always return latency and scores

Two habits that pay for themselves: return the score with each hit (so you can debug relevance), and time each stage (embed, search, hydrate). When a user says "search is slow", the split tells you whether it is the model or the database within seconds.

## Code

The full loop in about 40 lines, using a local model and no database. `search()` returns text plus metadata, with a tenant filter applied as a mask.

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims; check the hub for current ids

@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int
    tenant: str

def chunk_text(text: str, source: str, tenant: str, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Character-based chunking, good enough to start. Tokens are better; see 10-rag/03-chunking."""
    step = size - overlap
    out: list[Chunk] = []
    for i, start in enumerate(range(0, max(len(text) - overlap, 1), step)):
        piece = text[start : start + size].strip()
        if piece:
            out.append(Chunk(id=i, text=piece, source=source, page=start // 1800 + 1, tenant=tenant))
    return out

class InMemoryIndex:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)
        vecs = self.model.encode([c.text for c in chunks], normalize_embeddings=True)
        self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])

    def search(self, query: str, k: int = 5, tenant: str | None = None) -> list[dict]:
        t0 = time.perf_counter()
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q                       # cosine, because rows are unit length
        if tenant is not None:                         # filter BEFORE top-k
            mask = np.array([c.tenant == tenant for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        hits = [
            {"score": round(float(scores[i]), 4), "text": self.chunks[i].text[:120],
             "source": self.chunks[i].source, "page": self.chunks[i].page}
            for i in top if np.isfinite(scores[i])
        ]
        return hits  # latency added by the caller; see the API snippet below
```

Feed it three documents and query it. You should get back the chunk that answers the question even when the wording differs, and a `score` around 0.5–0.8 for a good match with this model — exact values vary by model build.

```python
docs = {
    "handbook.md": "Employees may carry over up to 10 days of unused leave into the next year. "
                   "Leave requests go through the HR portal at least two weeks in advance.",
    "billing.md": "Annual plans are billed upfront. Refunds are available within 30 days of purchase.",
    "security.md": "All laptops must use full-disk encryption. Report lost devices to security within 24 hours.",
}

model = SentenceTransformer(MODEL_ID)
index = InMemoryIndex(model)
for name, body in docs.items():
    index.add(chunk_text(body, source=name, tenant="acme"))

for hit in index.search("how many vacation days can I keep for next year", k=3):
    print(hit["score"], hit["source"], hit["text"][:60])
```

## Real use case

A 6-person startup shipped an internal "ask the docs" search over 2,000 Confluence pages using exactly this in-memory loop, in a FastAPI process with the matrix loaded at boot. It answered in single-digit milliseconds per query, which was far inside their 200ms budget, and needed no infrastructure to operate.

The design choices that mattered: chunking at ~400 characters with 50 overlap so a paragraph and its heading stayed together; tenant filtering as a mask *before* the top-k slice, because the same instance served two customers; and returning `source` and `page` on every hit so support staff could jump straight to the page instead of re-searching.

The failure mode before: Confluence's own search matched titles and tags, so a page titled "Time off policy" never surfaced for "vacation carryover". Users had learned to search for the page name rather than their question — the workaround is always the tell that search is broken.

They outgrew the in-memory version at roughly 200k chunks across tenants, when a single numpy multiply plus result hydration crossed their latency budget under concurrent load. That is when an ANN index earns its place — see [Vector Indexing Structures](04-vector-indexing.md).

## Common mistakes

- **Mistake:** filtering by tenant after taking top-k — **Instead:** push the filter into the search so ranking happens inside the allowed set.
- **Mistake:** chunking on fixed character counts without checking the output — **Instead:** print 20 random chunks and read them; broken chunks are visible immediately.
- **Mistake:** storing only vectors, no source metadata — **Instead:** store `source`, `page`, and any filter fields next to every vector from the first commit.
- **Mistake:** returning a bare list of strings — **Instead:** return `{score, text, source, page, latency_ms}` so the UI can cite and you can debug.
- **Mistake:** rebuilding the whole index at every process start — **Instead:** persist the matrix and metadata (`.npy` plus JSONL) and load them at boot.
- **Mistake:** judging quality from three queries you wrote yourself — **Instead:** collect 50 real queries and measure, as [Approximate Nearest Neighbour Search](05-ann-search.md) shows.

## Checkpoint

1. Run the loop over five of your own documents and query it with a paraphrase of a sentence you know is in one of them. Record the score and rank of the correct chunk.
2. Add a `lang` field to `Chunk` and make `search()` filter on it. Confirm that filtering before top-k returns 5 hits while filtering after top-k returns fewer.
3. Persist the index to disk (matrix as `.npy`, chunks as JSONL), reload it in a fresh process, and confirm identical results for the same query. Time the cold load.

**Self-check:** the correct chunk should come back at rank 1 with a cosine score in the middle of the model's range for this corpus — a paraphrase normally scores lower than an exact quote, which is expected, not a bug.

## Go deeper

- [Vector Indexing Structures](04-vector-indexing.md) — what to replace the numpy multiply with, and when.
- [Chunking](../10-rag/03-chunking.md) — the parameter you will tune most.
- [Stage 17 — FastAPI](../17-ai-backend/01-fastapi-flask.md) — wrap this loop in an endpoint properly.

---
[← Previous: 2. Similarity Metrics](02-similarity-metrics.md) · [Stage index](README.md) · [Next: 4. Vector Indexing Structures](04-vector-indexing.md)
