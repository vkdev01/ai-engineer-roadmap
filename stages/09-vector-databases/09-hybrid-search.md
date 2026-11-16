# 9. Hybrid Search and Reciprocal Rank Fusion

> **Stage 09 · Vector Databases** — combines keyword search and vector search, fused by rank instead of score, because the two score scales are not comparable.
>
> **You'll be able to:** run BM25 and vector search side by side, fuse with RRF, and explain why adding raw scores fails.
>
> **Prerequisites:** [Chroma and pgvector](08-chroma.md) · [Similarity Metrics](02-similarity-metrics.md)

## Why this exists

Vector search is bad at exact tokens. Ask for part number `AX-4471-B` and an embedding will happily return `AX-4471-C`, `AX-4417-B`, and a paragraph about "parts starting with AX" — all semantically close, all wrong. BM25 keyword search gets that query right instantly, because the token either matches or it does not.

Conversely, BM25 is bad at paraphrase. "How do I get my money back" does not share a token with "refund policy" and scores near zero.

Hybrid search runs both and merges. The interesting part is the merge: BM25 scores are unbounded sums of IDF-weighted term frequencies; cosine similarities live in `[-1, 1]`. You cannot add them.

## The mental model

```text
query ──┬─► BM25 (sparse, lexical)  ──► ranked list A: [d7, d2, d9, d1, ...]
        │                                    │
        └─► vectors (dense, semantic) ──► ranked list B: [d2, d5, d7, d4, ...]
                                             │
                              RRF: score(d) = Σ 1 / (k + rank_in_list)     k = 60
                                             │
                                     fused: [d2, d7, d5, d9, d1, ...]

  A document ranked well in BOTH lists beats one ranked first in only one.
```

## Core concepts

### BM25 in one paragraph

BM25 scores a document for a query by summing, over query terms, an IDF weight times a saturating term-frequency factor, normalised by document length:

```text
score = Σ_t  IDF(t) · ( tf · (k1 + 1) ) / ( tf + k1 · (1 − b + b · dl / avgdl) )
```

`k1` (commonly 1.2–2.0) controls term-frequency saturation — the tenth occurrence adds much less than the second. `b` (commonly 0.75) controls length normalisation. You rarely tune these; the important property is that exact rare tokens dominate, which is exactly the behaviour vectors lack.

### Why you cannot add the scores

Two problems. Scale: BM25 can return 14.2 for a great match while cosine returns 0.71 — the BM25 list wins every tie-break for no semantic reason. Distribution: a BM25 score of 8 means something different on a 200-word document than on a 4,000-word one, and cosine's range differs per model.

You can normalise — min-max each list to `[0, 1]` — but min-max depends on the top and bottom of each list, so a single outlier rescales everything, and the result is not stable across queries. Weighted score fusion works when you tune the weight per corpus, and it breaks when the corpus shifts.

### RRF: use the ranks, throw away the scores

Reciprocal rank fusion ignores scores entirely. For each document, sum `1 / (k + rank)` across the lists it appears in, where `rank` is 1-based and `k` is a smoothing constant, conventionally 60.

Worked example. Suppose document X is rank 1 in BM25 and rank 3 in vectors, and document Y is rank 2 in both:

```text
X: 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = 0.032266
Y: 1/(60+2) + 1/(60+2) = 0.016129 + 0.016129 = 0.032258
```

X wins by 0.000008 — effectively a tie, which is the point. Consistency across both retrievers is worth as much as being first in one. A document found by only one list at rank 1 scores `0.016393` and loses to any document in both lists near the top.

`k = 60` damps the difference between rank 1 and rank 2 (`1/61` vs `1/62`), so the fusion is robust to one retriever being confidently wrong. Smaller `k` sharpens the top; larger `k` flattens it. Sixty is a reasonable default and you should say in your docs that you did not tune it unless you did.

### Two implementation choices that matter

**Candidate depth.** Fuse over the top 50 from each list, not the top 5. A document at rank 30 in BM25 that is rank 2 in vectors deserves to surface; truncating at 5 throws away the benefit.

**Deduplication key.** Chunks from the two retrievers must share ids. If BM25 indexes paragraphs and vectors index 400-token chunks, the ids do not correspond and fusion is meaningless. Index the same chunk set into both.

## Code

A self-contained fusion with a tiny in-memory BM25, so you can see the mechanism without a search engine. It prints the two input lists and the fused list side by side.

```python
from __future__ import annotations

import math
import re
from collections import Counter

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())   # keeps AX-4471-B as one token

class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tokens = {i: tokenize(t) for i, t in docs.items()}
        self.avgdl = sum(len(v) for v in self.tokens.values()) / max(len(self.tokens), 1)
        df: Counter[str] = Counter()
        for toks in self.tokens.values():
            df.update(set(toks))
        n = len(self.tokens)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, limit: int = 50) -> list[str]:
        q = tokenize(query)
        scored: list[tuple[float, str]] = []
        for doc_id, toks in self.tokens.items():
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if s > 0:
                scored.append((s, doc_id))
        scored.sort(key=lambda p: (-p[0], p[1]))     # deterministic tie-break by id
        return [d for _, d in scored[:limit]]

def rrf(lists: list[list[str]], k: int = 60, limit: int = 10) -> list[tuple[str, float]]:
    """Fuse ranked id lists. k=60 damps rank differences."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:limit]
```

Run it on three documents with one exact-token query to see BM25 and vectors disagree, then fuse.

```python
docs = {
    "d1": "Replacement part AX-4471-B fits the model 300 pump.",
    "d2": "Spare components for industrial pumps and their ordering codes.",
    "d3": "How to return a faulty pump under warranty.",
}
bm25 = BM25(docs)
lexical = bm25.search("AX-4471-B")           # exact token match, BM25 wins big
semantic = ["d2", "d1", "d3"]                # stand-in for vector ranks

fused = rrf([lexical, semantic], k=60, limit=3)
for doc_id, score in fused:
    print(f"{score:.6f}", doc_id, docs[doc_id][:44])
```

`d1` should lead with roughly twice the fused score: it is rank 1 lexically and rank 2 semantically, so it collects two contributions. `d2` follows at a single contribution of `1/61 ≈ 0.0164`, and `d3` trails. That ordering — exact match first, semantically related second — is what a user expects from a search box.

## Real use case

An industrial-parts distributor's search box took model numbers, and their semantic search pilot quietly failed on exactly those queries: `AX-4471-B` returned `AX-4471-C` first, because the embeddings of the two strings are almost identical and the model has no notion of a checksum character mattering.

They added Postgres full-text search over the same chunk table and fused with RRF at `k = 60`, fusing the top 50 from each side. Chunk ids were shared because both indexes were built from the same chunking pass — the single decision that made fusion correct.

They then measured on 300 real search-log queries split into two groups: ones containing a part number and ones that were natural language. On part-number queries, hybrid put the exact part in the top 3 far more often than vectors alone; on natural-language queries, hybrid matched vectors closely and beat BM25 alone. Both numbers went into the release note with the query counts, because their earlier "it looks better" claim had not survived scrutiny.

The failure mode before: customers were pasting part numbers into a box that had been tuned for prose, and getting near-misses that looked authoritative.

## Common mistakes

- **Mistake:** adding normalised BM25 and cosine scores with fixed weights — **Instead:** use RRF, or tune weights per corpus and re-tune when the corpus changes.
- **Mistake:** fusing only the top 5 from each retriever — **Instead:** fuse the top 50 and cut to 10 after fusion.
- **Mistake:** letting the two retrievers index different chunk sets — **Instead:** build both from one chunking pass so ids match exactly.
- **Mistake:** tokenising part numbers into pieces — **Instead:** keep hyphens and digits in the token pattern so `AX-4471-B` stays one token.
- **Mistake:** assuming hybrid always wins — **Instead:** measure per query group; on some corpora it costs latency without improving relevance.
- **Mistake:** hardcoding `k = 60` without knowing why — **Instead:** keep the default, but write down that it was not tuned, so nobody assumes it was.

## Checkpoint

1. Build the BM25 above over ten documents and run a query that is a pure paraphrase. Confirm BM25 returns nothing or something poor, and explain why in terms of IDF.
2. Add a third ranked list (say, title matches) and fuse three lists. Confirm a document appearing at moderate rank in all three beats one at rank 1 in a single list.
3. Implement min-max score fusion alongside RRF and compare the two on 30 queries. Write three sentences on where they disagree and which you would ship.

**Self-check:** BM25 returns nothing for a paraphrase because none of the query's tokens appear in any document, so every term contributes zero regardless of IDF — there is no partial credit in lexical matching.

## Go deeper

- [Hybrid Retrieval](../10-rag/07-hybrid-retrieval.md) — the same fusion inside a RAG pipeline, with a reranker after it.
- [Reranking](../10-rag/06-reranking.md) — the step that usually beats tuning fusion weights.
- [Ship: A Semantic Search Service](10-ship-semantic-search.md) — add hybrid as a mode in your project API.

---
[← Previous: 8. Chroma and pgvector](08-chroma.md) · [Stage index](README.md) · [Next: 10. Ship: A Semantic Search Service](10-ship-semantic-search.md)
