# 1. Linear Algebra

> **Stage 02 · Math and ML Basics** — the notation every model is written in: vectors, matrices, and the dot product that measures similarity.
>
> **You'll be able to:** read a shape like `[batch, seq, dim]`, multiply matrices by hand and in NumPy, and explain what a dot product and cosine similarity mean.
>
> **Prerequisites:** [Stage 02 — Math and ML Basics](README.md) · [Generators](../01-programming-basics/03-generators.md)

## Why this exists

An embedding is a vector. A prompt batch is a matrix. Every layer of a transformer is a matrix multiply plus a nonlinearity. When something breaks, the error is usually a shape mismatch, and the fix is almost always "you have `[batch, dim]` where you needed `[dim, batch]`".

You do not need to compute matrix inverses by hand. You need to hold three things clearly: what a vector is, what a matrix does to a vector, and why the dot product is the workhorse.

The mental model: **a matrix is a function that transforms vectors, and shapes are part of the meaning.**

## The mental model

```text
vector v = [0.2, -1.4, 0.9]        shape (3,)         a point / a direction
matrix W (2 rows, 3 cols)          shape (2, 3)       transforms 3-d into 2-d
                                     W @ v -> shape (2,)

batch of 4 texts, 3 dims           (4, 3)
layer weights (3 -> 5)             (3, 5)
(4, 3) @ (3, 5) -> (4, 5)          inner dims must match, outer dims survive
```

| Operation | Notation | Result shape | Meaning |
|---|---|---|---|
| Dot product | `a · b` | scalar | similarity / projection |
| Matrix-vector | `W @ v` | `(rows,)` | transform one vector |
| Matrix-matrix | `A @ B` | `(A.rows, B.cols)` | transform a batch |
| Norm | `‖v‖` | scalar | length |
| Transpose | `A.T` | swapped | same data, other orientation |

## Core concepts

### A vector is a list of numbers with a length and a direction

`[3, 4]` has length 5 by Pythagoras, and points up-and-right. Two vectors are similar when they point the same way, regardless of how long they are — which is exactly why cosine similarity divides by both lengths.

### The dot product measures agreement

`a · b = Σ aᵢbᵢ`. It is large and positive when the vectors point the same way, zero when perpendicular, negative when opposite. Attention scores, retrieval scores and neuron activations are all dot products. Multiply elementwise, then sum.

### A matrix is a stack of dot products

`(W @ v)ᵢ = W[i] · v` — row `i` of the matrix dotted with the vector. That single sentence explains every linear layer: each output number is one row's opinion about the input.

### Shapes must line up, and the rule is simple

For `A @ B`, `A`'s column count must equal `B`'s row count. The result takes `A`'s rows and `B`'s columns. In practice: `(batch, in) @ (in, out) -> (batch, out)`. When you get a shape error, print the shapes and read this rule.

### Cosine similarity is the dot product normalised

`cos(a,b) = (a·b) / (‖a‖‖b‖)`, giving a number between -1 and 1. Embedding search uses this because it ignores magnitude — a short document and a long one on the same topic still score high.

### Broadcasting is NumPy being helpful, and it hides bugs

`(4, 3) + (3,)` adds the row vector to every row. Convenient, and the reason a `(4, 1)` where you meant `(4,)` silently produces `(4, 4)`. Add `.shape` prints when in doubt.

## Code

Vectors, dot products and cosine similarity, written out so the formulas are not magic.

```python
from __future__ import annotations

import numpy as np


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))          # elementwise multiply, then sum


def norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * v)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = norm(a) * norm(b)
    return 0.0 if denom == 0 else dot(a, b) / denom


a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 4.0, 6.0])
print(dot(a, b), cosine(a, b))   # 28.0, 1.0 (same direction, different length)
print(cosine(a, np.array([-1.0, -2.0, -3.0])))   # -1.0 (opposite)
```

A linear layer done twice: once with explicit loops so you can see it, once as a matmul so you can see why shapes matter.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))      # batch of 4 vectors, 3 features each
W = rng.normal(size=(3, 5))      # layer: 3 in, 5 out
bias = rng.normal(size=(5,))

# Explicit: output[i, j] = row i of X dotted with column j of W, plus bias.
out_loop = np.zeros((4, 5))
for i in range(4):
    for j in range(5):
        out_loop[i, j] = float(np.dot(X[i], W[:, j]) + bias[j])

out_fast = X @ W + bias          # identical, and this is what real code uses
print(np.allclose(out_loop, out_fast), out_fast.shape)   # True (4, 5)

# Transposing to fix a mismatch: scores for "query i vs key j"
Q, K = rng.normal(size=(4, 8)), rng.normal(size=(6, 8))
scores = Q @ K.T                 # (4, 8) @ (8, 6) -> (4, 6)
print(scores.shape)
```

## Real use case

A retrieval prototype ranked results badly and nobody could explain why: the same query returned near-identical scores for clearly different documents. The code stored embeddings and scored with `np.linalg.norm(a - b)` — Euclidean distance on unnormalised vectors. Because one document set came from short titles and another from long pages, vector length dominated the distance, and long documents looked "far" from everything.

Switching the score to cosine similarity fixed the ranking without touching the embedding model. Then a second bug surfaced: a `(n,)` vector had been stored where `(n, 1)` was expected, and a broadcast turned an `(n,)` query into an `(n, n)` score matrix — which is why some queries appeared to match themselves. Printing `.shape` at the boundary and asserting the expected shape in a helper function caught it immediately.

The design choice that mattered was treating shapes as part of the contract: every scoring function asserts its input shapes before computing.

## Common mistakes

- **Mistake:** comparing embeddings with Euclidean distance without normalising. — **Instead:** normalise once at write time and use the dot product, or use cosine.
- **Mistake:** ignoring `@` versus `*`. — **Instead:** `*` is elementwise, `@` is matrix multiply; they are not interchangeable.
- **Mistake:** relying on broadcasting to fix a shape. — **Instead:** reshape deliberately with `.reshape`/`.T` and assert the result shape.
- **Mistake:** treating a batch dimension as a feature dimension. — **Instead:** write shapes in a comment above each line.
- **Mistake:** assuming `float32` and `float64` behave identically in comparisons. — **Instead:** use `np.allclose`, not `==`.

## Checkpoint

1. Implement `cosine` for a whole matrix of queries against a matrix of documents in one matmul, and verify it against a loop.
2. Take a `(2, 3)` matrix and a `(3, 4)` matrix; predict the output shape, then check.
3. Normalise 1,000 random vectors and confirm every pairwise cosine falls in `[-1, 1]`; then explain why the dot product alone is enough after normalisation.

**Self-check:** `Q @ D.T` gives `(n_queries, n_docs)`; after normalising both matrices row-wise, that matrix is the cosine similarity matrix directly.

## Go deeper

- [Calculus](02-calculus.md) — how these matrices get their numbers in the first place.
- [Similarity Metrics](../09-vector-databases/02-similarity-metrics.md) — cosine, dot, and when each one is wrong.
- [Embeddings](../09-vector-databases/01-embeddings.md) — vectors that come out of a model rather than a random generator.

---
[← Previous: Stage 02 — Math and ML Basics](README.md) · [Stage index](README.md) · [Next: Calculus](02-calculus.md)
