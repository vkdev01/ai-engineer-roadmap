# 2. Similarity Metrics

> **Stage 09 · Vector Databases** — decides what "close" means, and shows why the three common metrics collapse into one when you normalise.
>
> **You'll be able to:** pick a metric deliberately, prove the equivalence between cosine and dot product on unit vectors, and compute each one by hand.
>
> **Prerequisites:** [Embeddings: Text as Points in Space](01-embeddings.md) · [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md)

## Why this exists

You have vectors; now you need a number for "how alike". Three formulas get used in practice — cosine similarity, dot product, euclidean distance — and most tutorials present them as interchangeable. They are not. They disagree on *scale*, and if you do not know which one your index uses, a top-10 list can change silently when you swap a library.

The failure is quiet. Nothing errors. Your results just reorder, and a relevant document that was rank 3 becomes rank 14. Knowing the metric is how you avoid debugging that for a day.

## The mental model

```text
                 cosine                     dot                  euclidean (L2)
what it measures  angle only                angle × lengths      straight-line distance
range             [-1, 1]                   unbounded            [0, ∞)
higher is better? yes                       yes                  NO — lower is better
unit vectors?     unchanged                 == cosine            d² = 2 − 2·cos

  On unit vectors all three rank identically:
  higher cosine  ==  higher dot  ==  lower L2
```

## Core concepts

### Cosine similarity measures angle and ignores length

Cosine similarity is the dot product divided by both magnitudes, which is exactly the cosine of the angle between the vectors:

```text
cos(a, b) = (a · b) / (||a|| · ||b||)
```

Worked example: `a = [3, 4]`, `b = [4, 3]`. Dot = 12 + 12 = 24. Both norms are 5. So cos = 24/25 = **0.96**. Angle ≈ 16.3°. They are close but not identical.

Now `a = [1, 0]`, `b = [2, 0]`: dot = 2, norms 1 and 2, cos = **1.0**. Same direction, different length — cosine calls them identical.

### Dot product rewards magnitude, which is usually a bug

`a · b` is just the sum of elementwise products. It equals cosine × `||a||` × `||b||`. If your embedding model produces longer vectors for longer text, a plain dot product will systematically favour long documents — a length bias dressed up as relevance.

For text retrieval, that bias is almost never what you want, which is why dot product is only safe once you have normalised. Some models (and some model cards) are explicitly trained for dot-product retrieval with unnormalised vectors; follow the card.

### Euclidean distance is the same ordering, inverted

L2 distance is `sqrt(Σ(aᵢ − bᵢ)²)`. For unit vectors there is an exact identity:

```text
||a − b||² = ||a||² + ||b||² − 2(a · b) = 2 − 2·cos(a, b)
```

So if cos = 0.96, then `d² = 0.08` and `d = 0.2828`. If cos = 0.70, `d² = 0.60` and `d = 0.7746`. Distance falls monotonically as cosine rises, so **ranking is identical** — only the direction of the number flips. Many indexes return distance and expect you to sort ascending; getting that backwards gives you the ten *least* relevant documents, confidently.

### Squared L2 saves a square root

Inside a vector database, `||a − b||²` is enough for ranking because `sqrt` is monotonic. Most implementations skip the `sqrt` and store or return *squared* L2. If you compare a distance from one system to a threshold from another, check which one you have.

### Choosing in practice

| Situation | Use | Reason |
|---|---|---|
| Normalised text embeddings | cosine (or dot on unit vectors) | Length is noise; angle is meaning |
| Model card says "dot product" for retrieval | dot, unnormalised | The model was trained that way |
| Index only supports L2 | L2 on unit vectors | Same ranking as cosine |
| Mixed-length documents, no normalisation | cosine | Removes the length bias |

## Code

A single script that computes all three for the same pair and proves the ordering equivalence on random unit vectors. This is the check to run before you trust any new index.

```python
from __future__ import annotations

import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

a = np.array([3.0, 4.0])
b = np.array([4.0, 3.0])
print(f"cos={cosine(a, b):.4f}  dot={dot(a, b):.1f}  l2={l2(a, b):.4f}")
# cos=0.9600  dot=24.0  l2=1.4142

# identity check: l2^2 == 2 - 2*cos for unit vectors
an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
print(round(l2(an, bn) ** 2, 6), round(2 - 2 * cosine(an, bn), 6))  # both 0.08
```

Now the ordering test — the part people skip. Generate 1,000 random unit vectors, score them against one query with each metric, and confirm the top-10 sets are identical.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(7)
dim, n = 64, 1000
corpus = rng.normal(size=(n, dim))
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)   # unit length

q = rng.normal(size=dim)
q /= np.linalg.norm(q)

by_dot  = np.argsort(-(corpus @ q))[:10]                  # higher is better
by_cos  = np.argsort(-(corpus @ q))[:10]                  # identical on unit vectors
by_l2   = np.argsort(np.linalg.norm(corpus - q, axis=1))[:10]  # lower is better

print(set(by_dot) == set(by_cos) == set(by_l2))  # True
```

Note the two `argsort` directions: cosine and dot sort descending, L2 sorts ascending. If you get `False` here on unnormalised vectors, that is the length bias showing up, not a bug in numpy.

## Real use case

A document-intelligence team ran a Qdrant collection created with `Distance.COSINE` for the first year. When they migrated to a second index to cut cost, the new index defaulted to L2 and their evaluation script — which compared returned scores to a fixed `0.75` threshold — started returning zero results, because L2 distances for unrelated documents sit near `sqrt(2) ≈ 1.41` and for near-duplicates near `0`, on a completely different scale.

The documents were fine. The threshold was meaningless. Their fix was to make the metric explicit in configuration, store it in the index metadata, and convert every score at the boundary into cosine similarity before any threshold logic ran. After that, the same code path worked across both indexes.

The design choice from this file: **scores are converted to one canonical metric at the adapter layer**, so downstream filtering never depends on which engine answered.

## Common mistakes

- **Mistake:** sorting L2 distances descending — **Instead:** sort ascending, or convert to `1 - d/2` style similarity first.
- **Mistake:** using dot product on unnormalised vectors from a model trained for cosine — **Instead:** normalise at write time; then dot and cosine agree.
- **Mistake:** applying a cosine threshold to an L2 index's raw output — **Instead:** convert at the boundary, and keep the metric in config, not in code comments.
- **Mistake:** assuming a similarity of 0.8 means "80% relevant" — **Instead:** treat it as an ordering score; calibrate thresholds against labelled pairs for your corpus.
- **Mistake:** comparing squared L2 from one library to L2 from another — **Instead:** check whether the value is squared; the identity `d² = 2 − 2cos` lets you convert.

## Checkpoint

1. For `a = [1, 2, 0]` and `b = [2, 1, 0]`, compute cosine, dot, and L2. Then compute them again after normalising both. Which numbers changed?
2. Take any 500 vectors from a real embedding model, compute the top-10 for one query with dot and with L2 on unnormalised vectors, and count the overlap. Explain the difference in one sentence.
3. Build a `score_to_cosine(raw, metric)` helper that accepts a cosine score, a dot score on unit vectors, or an L2 distance and returns cosine similarity. Unit-test all three branches with the identity above.

**Self-check:** cosine is `4/5 = 0.8`, dot is `4`, and L2 is `sqrt(2) ≈ 1.414`. After normalising both vectors to unit length, cosine stays `0.8`, dot becomes `0.8`, and L2 becomes `sqrt(0.4) ≈ 0.632` — only the metrics that depended on length changed.

## Go deeper

- [Semantic Search End to End](03-semantic-search.md) — put the metric inside a working search loop.
- [Vector Indexing Structures](04-vector-indexing.md) — how the metric choice constrains which index you can use.
- [Probability and Bayes](../02-math-and-ml-basics/03-probability-bayes.md) — the vocabulary you need to calibrate a score threshold.

---
[← Previous: 1. Embeddings: Text as Points in Space](01-embeddings.md) · [Stage index](README.md) · [Next: 3. Semantic Search End to End](03-semantic-search.md)
