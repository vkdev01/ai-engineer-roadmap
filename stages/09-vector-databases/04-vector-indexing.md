# 4. Vector Indexing Structures

> **Stage 09 · Vector Databases** — explains flat, IVF and HNSW as three different trades between recall, latency and memory, with the arithmetic shown.
>
> **You'll be able to:** size a collection, predict its memory, and choose an index family with a reason you can defend.
>
> **Prerequisites:** [Semantic Search End to End](03-semantic-search.md) · [Similarity Metrics](02-similarity-metrics.md)

## Why this exists

Brute force is exact and simple: multiply the query against every vector. It is also linear in corpus size, so at some point it stops fitting the latency budget. Every index is an attempt to look at fewer vectors — and every one of them gives up exactness to do it.

The trouble is that "use HNSW" is repeated as folklore. It has a memory cost, a build cost, and a parameter set that decides how often it misses the true nearest neighbour. You should be able to say *why* you picked it and *what you gave up*.

## The mental model

```text
                    how the search space is pruned         memory over raw vectors
flat      every vector compared, exactly            │  none (raw matrix only)
          ████████████████████████████████            │

IVF       cluster into nlist cells, probe nprobe    │  + centroids (tiny)
          ██ ██ ██ ·  ·  ·  ·  ·  ·  ·                │
          only probed cells are scanned             │

HNSW      layered graph, greedy descent then local  │  + ~M links per vector
          search on the bottom layer                │  (the expensive one)
          layer 2:   ●───●                            │
          layer 1:  ●─●─●─●─●                         │
          layer 0: ●●●●●●●●●●●●                       │
```

## Core concepts

### Memory comes first, because it decides what is even possible

The raw cost of a collection is `n × dim × bytes_per_float`. For 1,000,000 vectors at 1536 dimensions:

```text
float32: 1e6 × 1536 × 4 = 6.144 GB
float16: 1e6 × 1536 × 2 = 3.072 GB
int8   : 1e6 × 1536 × 1 = 1.536 GB
```

That is before any index. Do this sum before you choose a database, not after the bill arrives. Quantising to int8 cuts memory by 4× at a small recall cost — see [Quantization](../16-inference/06-quantization.md) for the general idea.

### Flat: exact, and correct to keep as your reference

A flat index is the numpy multiply from the last file, wrapped in an API. Its recall@k is 1.0 by definition. It is genuinely the right choice below a few hundred thousand vectors, and it is *always* the right choice as ground truth when you evaluate an approximate index — you cannot measure recall without an exact answer to compare against.

Its cost is O(n) per query. For 100,000 vectors at 384 dimensions a single query is a ~38M multiply-add scan, which in numpy is a few milliseconds; multiply that by 50 concurrent users and the budget is gone.

### IVF: cluster first, then scan a few cells

IVF (inverted file) runs k-means over the corpus to produce `nlist` centroids. Each vector is assigned to its nearest centroid, forming an inverted list per cell. At query time you find the nearest `nprobe` centroids and scan only their lists.

The arithmetic is the whole trade-off. With `nlist = 1000` and `nprobe = 10`, you scan about 1% of the corpus per query — a 100× reduction in distance computations. But if the true nearest neighbour sits in a cell whose centroid was the 11th closest, you miss it. Larger `nprobe` means better recall and more work; the relationship is roughly linear in scanned vectors.

IVF has two known weak spots worth stating plainly: you must **train** it on a representative sample before adding vectors, and its recall degrades when the query distribution shifts away from the training sample. Adding many vectors after training also skews cell sizes.

### HNSW: a navigable graph, and the current default

HNSW (hierarchical navigable small world) builds a multi-layer proximity graph. Search starts at a sparse top layer, greedily walks toward the query, then descends; the bottom layer holds every vector and gets a local best-first search. It gives high recall at low latency without training, and it handles incremental inserts.

The costs are real. Each vector stores up to `M` links on the bottom layer (and `M` links on upper layers, which is why implementations often budget roughly `2M` links), each link being an integer id — for `M = 16` that is tens of bytes per vector on top of the float payload, plus per-node overhead. Build time is superlinear-ish in practice because each insert does its own graph search. And deletion is awkward: most systems mark vectors as deleted and rebuild later.

Three parameters decide everything:

| Parameter | Controls | Raise it and you get |
|---|---|---|
| `M` | links per node | better recall, more memory, slower build |
| `efConstruction` | search width while building | better graph quality, slower build (no query cost) |
| `efSearch` | search width at query time | better recall, higher latency |

`efSearch` must be ≥ k. Setting `efSearch = k` is the classic under-tuned default and produces visibly poor recall.

### The comparison table you should be able to reproduce

| Index | Recall | Query latency | Memory overhead | Build | Handles inserts |
|---|---|---|---|---|---|
| Flat | exact (1.0) | O(n) | none | none | trivially |
| IVF | tunable, needs training | O(nprobe/nlist × n) | centroids only | k-means pass | yes, degrades |
| HNSW | high, tunable | ~O(log n) | ~M links per vector | slow, per-insert search | yes |

## Code

Two things worth writing yourself: the memory arithmetic as a function you actually call, and a naive IVF to see the recall loss with your own numbers.

```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Sizing:
    n: int
    dim: int
    dtype_bytes: int
    m_links: int = 0        # 0 for flat, e.g. 16 for HNSW

    @property
    def vector_gb(self) -> float:
        return self.n * self.dim * self.dtype_bytes / 1e9

    @property
    def graph_gb(self) -> float:
        # ~2M link slots (bottom + upper layers), 4 bytes per id
        return self.n * 2 * self.m_links * 4 / 1e9

    def report(self) -> str:
        total = self.vector_gb + self.graph_gb
        return f"vectors {self.vector_gb:.2f} GB + graph {self.graph_gb:.2f} GB = {total:.2f} GB"

print(Sizing(1_000_000, 1536, 4).report())          # 6.14 GB + 0.00 = 6.14 GB
print(Sizing(1_000_000, 1536, 1, m_links=16).report())  # 1.54 + 0.13 = 1.67 GB
```

A minimal IVF with k-means clustering, so the recall loss is something you measured rather than read. `nprobe=1` will miss neighbours that `nprobe=10` finds.

```python
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

def build_ivf(x: np.ndarray, nlist: int, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    km = KMeans(n_clusters=nlist, n_init=4, random_state=seed).fit(x)
    assign = km.labels_
    lists = [np.where(assign == c)[0] for c in range(nlist)]
    return km.cluster_centers_, lists

def ivf_search(q: np.ndarray, centers: np.ndarray, lists: list[np.ndarray],
               x: np.ndarray, k: int, nprobe: int) -> list[int]:
    probe = np.argsort(-(centers @ q))[:nprobe]           # nearest centroids first
    cand = np.concatenate([lists[c] for c in probe]) if probe.size else np.array([], dtype=int)
    if cand.size == 0:
        return []
    scores = x[cand] @ q
    return cand[np.argsort(-scores)[:k]].tolist()

def exact_search(q: np.ndarray, x: np.ndarray, k: int) -> list[int]:
    return np.argsort(-(x @ q))[:k].tolist()

rng = np.random.default_rng(0)
x = rng.normal(size=(5000, 64)); x /= np.linalg.norm(x, axis=1, keepdims=True)
q = rng.normal(size=64); q /= np.linalg.norm(q)

centers, lists = build_ivf(x, nlist=64)
truth = set(exact_search(q, x, k=10))
for nprobe in (1, 4, 16):
    got = set(ivf_search(q, centers, lists, x, k=10, nprobe=nprobe))
    print(nprobe, len(truth & got) / 10)   # recall rises with nprobe; 1.0 only when every cell is probed
```

Random uniform data is close to the worst case for IVF — real embeddings cluster, so recall is usually better. That is exactly why you measure on your own data.

## Real use case

A media company indexed 2.4 million article chunks at 768 dimensions. Raw float32 cost: `2.4e6 × 768 × 4 / 1e9 ≈ 7.4 GB`, which already exceeded their single-node budget once the database's own overhead was added, and left nothing for HNSW links.

They switched the stored vectors to int8 with a float32 copy kept only for reranking the top 100 candidates. Memory dropped to about 1.8 GB of vectors plus a small rerank buffer, which fit comfortably with an HNSW graph at `M = 16`. Measured recall@10 fell slightly versus their flat reference; the top-100 rescoring recovered most of it, and they published the before/after numbers in their own dashboard rather than quoting a vendor claim.

The failure mode before: they had provisioned by guessing, the node ran out of memory during a re-index, and the service was down for hours. The sizing function above would have predicted the problem in one line.

## Common mistakes

- **Mistake:** picking HNSW before doing the memory arithmetic — **Instead:** compute `n × dim × bytes`, add index overhead, then decide.
- **Mistake:** setting `efSearch` equal to `k` — **Instead:** set `efSearch` several times `k` and measure recall at your p95 latency.
- **Mistake:** training IVF on a sample that does not look like production data — **Instead:** sample randomly across tenants and languages, and retrain when the distribution shifts.
- **Mistake:** deleting the flat reference index once ANN works — **Instead:** keep a flat index over a sample as ground truth for your recall script.
- **Mistake:** assuming a rebuild is cheap — **Instead:** measure build time on 10% of the corpus and extrapolate before committing to a re-index window.

## Checkpoint

1. Compute the raw memory for 500,000 vectors at 768 dimensions in float32 and int8. Add HNSW overhead at `M = 16` and state the total for each.
2. Run the IVF snippet with `nlist` at 16, 64 and 256, and `nprobe` at 1, 4, 16. Tabulate recall@10 against the exact answer and find the smallest `nprobe` that reaches recall 0.95 at each `nlist`.
3. Swap the synthetic data for 5,000 real embeddings from your own corpus, redo exercise 2, and explain in three sentences why the numbers moved.

**Self-check:** 500,000 × 768 × 4 bytes is about 1.54 GB in float32 and about 0.38 GB in int8; HNSW at `M = 16` adds roughly `500000 × 2 × 16 × 4 ≈ 0.064 GB`, so about 1.60 GB and 0.45 GB respectively.

## Go deeper

- [Approximate Nearest Neighbour Search](05-ann-search.md) — measuring recall honestly and tuning against latency.
- [Quantization](../16-inference/06-quantization.md) — the general theory behind the int8 trick above.
- [Scaling Trade-offs](../21-system-design/09-scaling-tradeoffs.md) — when to shard a collection instead of growing one node.

---
[← Previous: 3. Semantic Search End to End](03-semantic-search.md) · [Stage index](README.md) · [Next: 5. Approximate Nearest Neighbour Search](05-ann-search.md)
