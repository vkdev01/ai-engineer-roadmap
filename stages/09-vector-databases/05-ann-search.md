# 5. Approximate Nearest Neighbour Search

> **Stage 09 · Vector Databases** — how to tune an ANN index and measure recall honestly, instead of trusting a demo that looks fine on five queries.
>
> **You'll be able to:** define recall@k precisely, build a ground-truth set by brute force, and pick index parameters from a measured latency/recall curve.
>
> **Prerequisites:** [Vector Indexing Structures](04-vector-indexing.md) · [Semantic Search End to End](03-semantic-search.md)

## Why this exists

"Approximate" means the index may not return the true nearest neighbours. That is the deal you accept for speed. What you must not accept is not knowing how often it happens.

The usual failure: someone raises `nprobe` until the first ten queries "look right", ships, and later finds that queries in a particular language return nothing relevant — because those vectors sit in cells the index rarely probes. Recall was never measured, so the regression was invisible.

Recall is a number you can compute in an afternoon. This file is about computing it properly.

## The mental model

```text
ground truth (exact)                ANN result                recall@10
────────────────────                ──────────                ─────────
flat scan, all n vectors     vs      index, some vectors       |truth ∩ ann| / k
returns 10 ids                      returns 10 ids            averaged over queries

  truth = {4, 9, 12, 31, 44, 57, 61, 70, 88, 93}
  ann   = {4, 9, 12, 31, 44, 57, 61, 88, 90, 91}
  overlap = 8            recall@10 = 8/10 = 0.80

  sweep efSearch 16 → 512, plot recall against p95 latency, pick a point
```

## Core concepts

### Recall@k is an overlap ratio, per query, averaged

Define it once and never drift:

```text
recall@k = |ANN_k(q) ∩ Exact_k(q)| / k
```

Averaged over a query set. Two honest caveats. First, if the corpus has fewer than `k` vectors, the denominator is the corpus size. Second, ties matter: if five vectors share the exact same score at the k boundary, exact answers can differ between runs; break ties deterministically by id, or accept a small noise floor.

A related metric, **precision of the returned set against a labelled answer set**, is different — that is relevance, not index quality. Keep them separate: recall@k here measures *the index*, not *your embeddings*.

### The query set decides whether the number means anything

Your recall number is only as good as the queries. Three rules:

- Use **real queries** from logs or support tickets, not sentences copied out of your documents. Copied sentences are easy mode and inflate recall.
- Cover the **sub-populations** you care about: each language, each tenant, short queries, long queries.
- Use **enough** of them. Thirty queries give a recall estimate with a wide error bar; a few hundred is comfortable. Report the query count with the number.

### Ground truth must come from the same vectors and metric

Build truth by brute force over the *same* matrix the index holds, with the *same* metric. If you compute truth with cosine and query the index with L2, you are measuring metric mismatch, not index quality.

### Tune the query-time knob, not the build-time one, for latency

`efSearch` (HNSW) and `nprobe` (IVF) trade latency for recall at query time — no rebuild needed. `M`, `efConstruction` and `nlist` are build-time and change memory or build duration. In production, sweep the query-time parameter, pick the value that meets your p95 latency target, and leave headroom: a node under load runs slower, and latency-driven systems often shrink `efSearch` adaptively.

### Recall is not the same as usefulness

An index with recall@10 = 0.95 can still produce a bad product if the embedding model is wrong for your domain, or if the answer is split across two chunks. Recall measures the index; usefulness needs an end-to-end eval — [RAG Evaluation](../10-rag/12-rag-evals.md) is where that lives.

## Code

A complete, honest measurement harness. It builds ground truth by brute force, compares against the ANN result at several settings, and reports recall per query group so a weak sub-population cannot hide in the average.

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

@dataclass
class RecallReport:
    k: int
    per_group: dict[str, list[float]] = field(default_factory=dict)

    def add(self, group: str, hit: int) -> None:
        self.per_group.setdefault(group, []).append(hit / self.k)

    def table(self) -> str:
        lines = [f"recall@{self.k}"]
        for group, vals in sorted(self.per_group.items()):
            mean = sum(vals) / len(vals)
            lines.append(f"  {group:<16} n={len(vals):<4} {mean:.3f}")
        allv = [v for vs in self.per_group.values() for v in vs]
        lines.append(f"  {'ALL':<16} n={len(allv):<4} {sum(allv) / len(allv):.3f}")
        return "\n".join(lines)

def exact_top_k(matrix: np.ndarray, q: np.ndarray, k: int) -> set[int]:
    """Brute-force truth. Tie-break by id so repeated runs agree."""
    scores = matrix @ q
    order = np.lexsort((np.arange(len(scores)), -scores))  # score desc, then id asc
    return set(order[:k].tolist())

def ann_top_k(matrix: np.ndarray, q: np.ndarray, k: int, ef: int) -> set[int]:
    """Stand-in for a real index: probes ef candidates then keeps k.

    Replace the body with client.search(..., params={"ef": ef}) for HNSW.
    Here we simulate the mechanism: a cheap prefilter of ef candidates by a
    coarse projection, then exact ranking inside them.
    """
    coarse = matrix[:, :16] @ q[:16]           # cheap approximation of the score
    cand = np.argsort(-coarse)[:ef]            # the index's candidate set
    scores = matrix[cand] @ q
    order = cand[np.lexsort((cand, -scores))]
    return set(order[:k].tolist())
```

Then the sweep. Notice that recall is reported per query group, and that latency is measured, not assumed.

```python
import time

rng = np.random.default_rng(3)
n, dim, k = 20_000, 128, 10
matrix = rng.normal(size=(n, dim))
matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

queries = {"short": [], "long": []}
for group, count in (("short", 60), ("long", 60)):
    for _ in range(count):
        q = rng.normal(size=dim); q /= np.linalg.norm(q)
        queries[group].append(q)

truth = {g: [exact_top_k(matrix, q, k) for q in qs] for g, qs in queries.items()}

for ef in (16, 64, 256, 1024):
    rep, t0 = RecallReport(k=k), time.perf_counter()
    for group, qs in queries.items():
        for q, t in zip(qs, truth[group]):
            got = ann_top_k(matrix, q, k, ef)
            rep.add(group, len(got & t))
    ms = (time.perf_counter() - t0) * 1000 / sum(len(v) for v in queries.values())
    print(f"ef={ef:<5} {ms:6.2f} ms/query")
    print(rep.table())
```

You should see recall climb as `ef` grows and latency climb with it — then pick the smallest `ef` that clears your recall target. Do not expect high absolute values from this snippet: on random uniform data a 16-dimension projection is a weak prefilter, so recall stays low even at `ef = 1024` (a twentieth of the corpus). Real HNSW reaches much higher recall at the same candidate count because the graph search is smarter than a projection. What transfers is the shape of the trade-off and the method of measuring it, not the absolute numbers — which is exactly why you run the sweep on your own data.

## Real use case

A legal-research product had a 900k-chunk collection in HNSW with the library's default `efSearch`. Relevance complaints were concentrated on Hindi queries, while English queries were fine. Nothing in the logs looked wrong: p95 latency was excellent and results were non-empty.

They added a harness of 240 queries split by language, built ground truth from a flat index over the same vectors, and measured recall@10. English sat high; Hindi was materially lower, because Hindi chunks were fewer and the graph had fewer neighbours to walk through in that region of the space. Raising `efSearch` for all traffic recovered most of the gap at roughly double the p95 latency, which their budget allowed. They then documented the recall number per language in the service's README, with the query count.

The design choice from this file: **recall reported per group, not just averaged**, which is what made the Hindi regression visible at all. A single blended number would have hidden it.

## Common mistakes

- **Mistake:** reporting recall from ten hand-written queries — **Instead:** use a few hundred real queries, report the count, and split by sub-population.
- **Mistake:** computing ground truth with a different metric than the index uses — **Instead:** derive truth from the same matrix and metric, ideally from a flat index in the same system.
- **Mistake:** benchmarking latency on an idle node — **Instead:** measure under representative concurrency and report p95, not the mean.
- **Mistake:** tuning `efSearch` once and never again — **Instead:** re-run the harness when the corpus grows, the model changes, or a new language is added.
- **Mistake:** assuming high recall means good answers — **Instead:** pair index recall with an end-to-end eval on labelled query/answer pairs.
- **Mistake:** ignoring tie-breaking — **Instead:** sort deterministically by score then id, so repeated runs match.

## Checkpoint

1. Take 200 vectors, `k = 5`, and compute recall@5 of a deliberately bad candidate set (say 10 candidates chosen at random). Confirm the expected value is around `10/200 = 0.05` and explain why it is not exactly that.
2. Run the sweep above on your own corpus and produce a table of `ef`, recall@10, and p95 latency. Choose a setting and write one sentence justifying it.
3. Add a third query group of one-word queries. Re-run, and if recall differs from the other groups, write down one hypothesis for why and one experiment to test it.

**Self-check:** with 10 random candidates out of 200, the chance that a given true neighbour is in the candidate set is 10/200 = 0.05, so the expected overlap is `5 × 0.05 = 0.25` out of 5, i.e. recall ≈ 0.05 — and the observed value scatters around that because candidates are drawn per query.

## Go deeper

- [Pinecone and Qdrant](06-pinecone-qdrant.md) — the same tuning knobs, in a real client.
- [Ship: A Semantic Search Service](10-ship-semantic-search.md) — where this harness becomes part of your project.
- [RAG Evaluation](../10-rag/12-rag-evals.md) — measuring answer quality, which recall does not cover.

---
[← Previous: 4. Vector Indexing Structures](04-vector-indexing.md) · [Stage index](README.md) · [Next: 6. Pinecone and Qdrant](06-pinecone-qdrant.md)
