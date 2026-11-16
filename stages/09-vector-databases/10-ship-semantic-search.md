# 10. Ship: A Semantic Search Service

> **Stage 09 · Vector Databases** — the project: a search service over your own corpus, with a query API, metadata filters, and a measured recall report.
>
> **You'll be able to:** ship a service whose relevance claim is backed by a number you produced, not a demo you curated.
>
> **Prerequisites:** [Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) · [Approximate Nearest Neighbour Search](05-ann-search.md)

## Why this exists

Everything in this stage is now a set of parts: an embedding model, a store, an index, a fusion step, a recall harness. A project that wires them together is what turns knowledge into a skill — and the recall report is what makes it different from every tutorial demo you have seen.

The deliverable is small on purpose. One corpus, one endpoint, one measurement script, one README table. If you can do this end to end and defend the numbers, you can do it at any scale.

## The mental model

```text
  build_index.py                     serve.py                      eval_recall.py
  ──────────────                     ────────                      ──────────────
  parse corpus                       FastAPI /search               load flat reference
  chunk (400 tok, 50 overlap)   ───► embed query                   load ANN collection
  embed (batched)                    filter (tenant, lang)    ───► 200 queries, k=10
  upsert + payload  ──────────────►  search top-k            ───► compare, per group
  write manifest.json                return hits + latency         print table
       │                                  │                            │
       └──────────── manifest: model_id, dim, metric, index params ────┘
```

## Core concepts

### Step 1 — Freeze the manifest before you index

Write a `manifest.json` at build time containing: `model_id`, `dim`, `metric`, `chunk_size`, `chunk_overlap`, `index_type`, index parameters, and the corpus fingerprint (file count and total characters). Every later mystery — "why did recall drop" — is answered by diffing the manifest. This one file is the difference between a debuggable system and a haunted one.

### Step 2 — Build the index and the flat reference together

Write your vectors to the ANN store *and* to a flat `.npy` matrix with a parallel JSONL of payloads. The flat copy is your ground truth for recall and your escape hatch if the store misbehaves. For a corpus up to a few hundred thousand chunks it costs little.

### Step 3 — Serve with filters pushed down

The endpoint takes `q`, `k`, and optional `tenant` / `lang`. It embeds the query once, runs one filtered search, and returns `{score, text, source, page}` plus a `latency_ms` breakdown. Validate the request; a `k` of 10,000 should be rejected, not honoured.

### Step 4 — Measure recall@10 and publish it

Take 200 real queries, compute exact top-10 from the flat matrix, compare with the store's top-10, and report recall overall **and per query group**. State the query count next to the number. If you tuned `ef`/`nprobe`, show the sweep as a table.

### Step 5 — Do not confuse recall with relevance

Recall@10 says the index found the same neighbours as brute force. It does not say those neighbours answer the question. Note that explicitly in your README and point at the evaluation work in [RAG Evaluation](../10-rag/12-rag-evals.md).

## Code

The measurement script. It reads the manifest, loads the flat reference, runs the query set against both, and prints a table you can paste into your README.

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

K = 10

def load_reference(dir_path: Path) -> tuple[np.ndarray, list[dict], dict]:
    matrix = np.load(dir_path / "vectors.npy")                     # float32, unit length
    payloads = [json.loads(l) for l in (dir_path / "payloads.jsonl").read_text().splitlines()]
    manifest = json.loads((dir_path / "manifest.json").read_text())
    return matrix, payloads, manifest

def exact_top_k(matrix: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    scores = matrix @ q
    return np.lexsort((np.arange(len(scores)), -scores))[:k].tolist()

def recall_report(store_search, queries: list[dict], matrix: np.ndarray, payloads: list[dict]) -> str:
    """store_search(query_vector, k, filters) -> list of row indices, in rank order."""
    groups: dict[str, list[float]] = {}
    for item in queries:
        qv = np.asarray(item["vector"], dtype=np.float32)
        truth = set(exact_top_k(matrix, qv, K))
        got = store_search(qv, K, item.get("filters", {}))
        groups.setdefault(item["group"], []).append(len(truth & set(got)) / K)

    lines = [f"recall@{K} over {len(queries)} queries"]
    for name, vals in sorted(groups.items()):
        lines.append(f"  {name:<14} n={len(vals):<4} {sum(vals) / len(vals):.3f}")
    allv = [v for vs in groups.values() for v in vs]
    lines.append(f"  {'ALL':<14} n={len(allv):<4} {sum(allv) / len(allv):.3f}")
    return "\n".join(lines)
```

The service. Two endpoints — `search` and `health` — with the filter and `k` validated, and latency broken down by stage so a slow query is diagnosable.

```python
from __future__ import annotations

import time
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="semantic-search")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # must match the index

class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=50)          # reject absurd k, do not honour it
    tenant: str | None = None
    mode: Literal["vector", "hybrid"] = "vector"

class Hit(BaseModel):
    score: float
    text: str
    source: str
    page: int

class SearchResponse(BaseModel):
    hits: list[Hit]
    latency_ms: dict[str, float]

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    qv = model.encode([req.q], normalize_embeddings=True)[0]
    t1 = time.perf_counter()
    # store.search pushes the tenant filter into the engine; see 06-pinecone-qdrant
    rows = store.search(qv, k=req.k, filters={"tenant": req.tenant} if req.tenant else {})
    t2 = time.perf_counter()
    hits = [Hit(score=r.score, text=r.text[:400], source=r.source, page=r.page) for r in rows]
    return SearchResponse(
        hits=hits,
        latency_ms={"embed": round((t1 - t0) * 1000, 2), "search": round((t2 - t1) * 1000, 2)},
    )
```

A Node consumer, because the search box usually lives in a browser app and this is where people get the request shape wrong. It validates the response with `zod` rather than trusting it.

```ts
import { z } from "zod";

const Hit = z.object({
  score: z.number(),
  text: z.string(),
  source: z.string(),
  page: z.number().int(),
});
const SearchResponse = z.object({
  hits: z.array(Hit),
  latency_ms: z.record(z.number()),
});

export async function search(
  q: string,
  opts: { k?: number; tenant?: string; signal?: AbortSignal } = {},
): Promise<z.infer<typeof SearchResponse>> {
  const res = await fetch("http://localhost:8000/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ q, k: opts.k ?? 5, tenant: opts.tenant }),
    signal: opts.signal,                     // lets the UI cancel stale keystrokes
  });
  if (!res.ok) throw new Error(`search failed: ${res.status}`);
  return SearchResponse.parse(await res.json());   // throws on shape drift, loudly
}
```

## Real use case

A documentation team shipped this project as the search behind a 3,000-page product manual. Their first release used a hosted store with default index settings and no measurement. Support reported that "search is worse than before" within a week, but nobody could say worse than what.

The rebuild added three things: the manifest, the flat reference, and a 200-query set drawn from actual site search logs, split into "short keyword" and "full question" groups. The report showed the index's recall@10 was fine for both groups — so the problem was not the index. It was chunking: their parser had dropped table content, so pricing-table answers were simply absent from the corpus.

That is the value of measuring index recall separately from relevance: it ruled out an entire layer in one afternoon, and pointed at ingestion — which is exactly where [Ingestion and Parsing](../10-rag/02-ingestion-parsing.md) starts.

## Common mistakes

- **Mistake:** shipping without a query set — **Instead:** collect 200 real queries before launch; without them you cannot tell a regression from a mood.
- **Mistake:** storing the model id only in code — **Instead:** write it into the manifest and refuse to serve if the manifest's model id differs from the loaded model's.
- **Mistake:** reporting a single blended recall number — **Instead:** report per group, with `n` for each.
- **Mistake:** letting the client send `k = 1000` — **Instead:** validate with a maximum and return 422 for anything larger.
- **Mistake:** claiming "high accuracy" from three hand-picked queries — **Instead:** state the metric, the query count, and what it does not cover.
- **Mistake:** rebuilding the flat reference only at launch — **Instead:** rebuild it in the same job as the index, so the two never drift.

## Checkpoint

1. Build the index, the flat reference and the manifest for a corpus of at least 500 chunks. Run the recall script and paste the table into your README with the query count.
2. Add a `mode: "hybrid"` branch that fuses BM25 with vectors using RRF, and report recall per mode. Does the ANN recall stay the same? Why?
3. Deploy the service, point the TypeScript client at it, and add a `p95` measurement over 100 sequential requests. Write two sentences on where the time goes.

**Self-check:** hybrid mode does not change ANN recall@10 against a vector-only ground truth, because recall here measures the index's ability to reproduce exact vector neighbours — the fusion happens after retrieval and is a relevance question, not an index question.

## Go deeper

- [Stage 10 — RAG](../10-rag/README.md) — next: turn retrieval into grounded answers.
- [RAG Evaluation](../10-rag/12-rag-evals.md) — measure relevance, not just recall.
- [Stage 17 — FastAPI](../17-ai-backend/01-fastapi-flask.md) — harden the endpoint: auth, rate limits, streaming.

---
[← Previous: 9. Hybrid Search and Reciprocal Rank Fusion](09-hybrid-search.md) · [Stage index](README.md) · [Next: Stage 10 — RAG](../10-rag/README.md)
