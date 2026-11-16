# 7. Hybrid Retrieval

> **Stage 10 · RAG** — running lexical and vector retrieval together inside the pipeline, then reranking the fused list.
>
> **You'll be able to:** wire BM25 and vector search into one retrieval stage, tune the fusion, and place the reranker correctly after it.
>
> **Prerequisites:** [Reranking](06-reranking.md) · [Hybrid Search and Reciprocal Rank Fusion](../09-vector-databases/09-hybrid-search.md)

## Why this exists

RAG questions contain two kinds of signal, and each retriever is blind to one. "What changed in policy 4.2.1" needs the exact token. "What happens if I resign mid-year" needs meaning. A pipeline with only vectors fails the first; a pipeline with only BM25 fails the second.

Stage 09 showed the mechanics of reciprocal rank fusion. This file is about the pipeline consequences: where the sparse index lives, how the fusion interacts with the reranker, and how to tune the whole chain rather than one link.

The mental model: **hybrid retrieval raises the ceiling on recall; the reranker then picks from a better candidate set.**

## The mental model

```text
                    ┌───────────────┐
 question ─────────►│ sparse: BM25  │──► top 50 ─┐
                    └───────────────┘            │
                    ┌───────────────┐            ├──► RRF fuse ──► rerank ──► top 5 ──► prompt
 question ─────────►│ dense: vectors│──► top 50 ─┘   (~60-80      (cross-
                    └───────────────┘                   unique)      encoder)

 order matters:  fuse BEFORE rerank. Reranking each list separately then
                 merging scores throws away the rank information RRF uses.
```

## Core concepts

### The sparse index should live next to your data

You have three practical homes for BM25: the vector database's built-in sparse support, your existing Postgres full-text search (`tsvector`), or a dedicated engine. Adding a third system for keyword search is rarely justified — Postgres full-text over the same chunk table is boring and works, and it keeps ids aligned by construction.

Whichever you choose, the requirement is the same: **the same chunk ids in both indexes**. If sparse indexes paragraphs and dense indexes 400-token chunks, fusion is meaningless, because rank lists describe different objects.

### Fuse, then rerank — in that order

The sequence that works is: two lists of 50 → RRF → about 60–80 unique candidates → rerank → top 5. Reranking first and merging scores afterwards loses the cross-list agreement signal that makes RRF robust, and doubles your reranker cost for no benefit.

One subtlety: reranking a fused list of 80 costs more than reranking 50. If latency is tight, cut each retriever's list to 30 and rerank the fused ~50.

### Weighted fusion is sometimes right, and always fragile

RRF treats both retrievers as equally trustworthy. For some corpora that is wrong — a corpus of dense technical prose may deserve more weight on lexical matching. You can weight RRF by multiplying each list's contribution, but the weight is corpus-specific and drifts as the corpus changes.

The honest approach: try equal-weight RRF first, measure, and only add weights if the measured gain is large enough to justify the maintenance. Write down the weight and the date it was tuned.

### Query type is a routing signal, not a fusion parameter

A useful refinement: classify the query before retrieval. Queries containing identifiers, codes or quoted phrases route to lexical-heavy retrieval; natural-language questions route to vector-heavy. This is often worth more than weight tuning, because it adapts per query instead of averaging over all of them.

Keep the classifier trivial — a regex for digits, hyphens and quoted strings is enough for most corpora — and log how many queries each path handles.

### Measure per query group, always

Hybrid retrieval's benefit is concentrated in a subgroup: identifier queries. A blended average will show a modest gain and hide the fact that one group improved a lot and another slightly regressed. Split your eval set by query type and report both, with counts.

## Code

Fusion and reranking composed into one retrieval function, with candidate depth as an explicit parameter and the ordering enforced in code.

```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int

def rrf_fuse(ranked_lists: list[list[str]], k: int = 60, limit: int = 80) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:limit]

def hybrid_retrieve(
    question: str,
    lexical_search,          # (query, k) -> list[str] of ids
    dense_search,            # (question, k) -> list[str] of ids
    fetch,                   # (ids) -> dict[str, Hit]
    reranker,                # (question, list[Hit], keep) -> list[Hit]
    depth: int = 50,
    fused_limit: int = 80,
    keep: int = 5,
) -> dict:
    lex_ids = lexical_search(question, depth)
    dense_ids = dense_search(question, depth)

    fused = rrf_fuse([lex_ids, dense_ids], k=60, limit=fused_limit)
    by_id = fetch([i for i, _ in fused])
    candidates = [by_id[i] for i, _ in fused if i in by_id]   # preserve fused order

    ranked = reranker(question, candidates, keep)
    return {
        "hits": ranked,
        "n_lexical": len(lex_ids),
        "n_dense": len(dense_ids),
        "n_fused": len(candidates),
        "overlap": len(set(lex_ids) & set(dense_ids)),        # log this; it is diagnostic
    }
```

Postgres full-text over the same chunk table, so ids align with the vector index by construction. This is the least glamorous and most reliable sparse index for a team already on Postgres.

```sql
-- one column, generated and indexed once
ALTER TABLE chunks
  ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(heading_path, '') || ' ' || content)) STORED;

CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);

-- lexical candidates for a query, filtered by tenant, ranked by ts_rank_cd
SELECT id, ts_rank_cd(tsv, q) AS rank
FROM chunks, websearch_to_tsquery('english', $1) AS q
WHERE tsv @@ q
  AND tenant = $2
ORDER BY rank DESC
LIMIT 50;
```

`websearch_to_tsquery` accepts quoted phrases and `-exclusions` the way a user would type them, which is exactly the syntax that makes lexical retrieval worth having.

## Real use case

A hardware manufacturer's field-service assistant answered from 12,000 service bulletins. Engineers searched by part number and error code — `E-2214`, `AX-4471-B` — and the vector-only pipeline returned bulletins about similar-looking codes, sometimes for the wrong product line.

They added Postgres full-text over the same chunk table, fused with RRF at depth 50, and reranked the fused list to 5. The `overlap` field in the response turned out to be the most useful diagnostic they added: for natural-language queries, lexical and dense lists overlapped substantially; for error codes they barely overlapped at all, which was the signature of the exact-match problem.

Their eval set was split into "code queries" and "description queries" and reported separately. Code queries improved a great deal; description queries stayed flat, which was the expected result and made the change easy to defend.

The failure mode before: engineers had learned to type the product line along with the code, as a workaround for near-miss retrieval. When a workaround like that appears in your logs, you are looking at a retrieval defect.

## Common mistakes

- **Mistake:** running BM25 over different chunks than the vector index — **Instead:** build both from one chunking pass so ids match.
- **Mistake:** reranking each list separately, then merging — **Instead:** fuse first with RRF, rerank the fused list.
- **Mistake:** fusing the top 5 from each side — **Instead:** use depth 30–50 per side and cut after fusion.
- **Mistake:** tuning fusion weights on a blended average — **Instead:** split the eval set by query type and report each group with counts.
- **Mistake:** standing up a separate search cluster for keyword matching — **Instead:** start with Postgres full-text on the same table.
- **Mistake:** not logging list overlap — **Instead:** log `n_lexical`, `n_dense` and `overlap` per request; it identifies exact-match failure in one glance.

## Checkpoint

1. Add Postgres full-text search to your chunk table and confirm that a part-number query returns hits lexically while the vector path returns near-misses.
2. Implement `hybrid_retrieve` and compare top-5 relevance with vectors-only on 30 queries split into code and description groups.
3. Change the fusion depth from 20 to 50 to 100 and measure both latency and the change in top-5 relevance. Write two sentences on where the returns flatten.

**Self-check:** for a part-number query the lexical list contains the exact chunk near rank 1 while the dense list contains semantically similar chunks with different codes, so the fused list puts the exact match first — and `overlap` between the two lists is near zero.

## Go deeper

- [Hybrid Search and Reciprocal Rank Fusion](../09-vector-databases/09-hybrid-search.md) — the fusion mechanics in more depth.
- [Parent-Child Retrieval](08-parent-child.md) — the next structural improvement to retrieval.
- [RAG Evaluation](12-rag-evals.md) — context recall, the metric hybrid retrieval moves.

---
[← Previous: 6. Reranking](06-reranking.md) · [Stage index](README.md) · [Next: 8. Parent-Child Retrieval](08-parent-child.md)
