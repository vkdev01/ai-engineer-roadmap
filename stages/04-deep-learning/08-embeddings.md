# 8. Embeddings

> **Stage 04 · Deep Learning** — meaning as geometry, and the operation the whole LLM stack is built on
>
> **You'll be able to:** train a small embedding layer, measure similarity correctly, and explain what a vector space of meaning can and cannot represent.
>
> **Prerequisites:** [Transformers](07-transformers.md) · [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md)

## Why this exists

A model cannot multiply a word. So before anything else happens, each token id becomes
a vector — a row in a lookup table with one row per vocabulary entry. Training adjusts
those rows so that vectors of things used similarly end up close together. That is the
whole idea: **an embedding is a learned coordinate system where distance means
relatedness.**

This matters beyond the model's internals. The same property is what makes semantic
search work: embed the documents, embed the query, return the nearest vectors. Every
vector database, retrieval pipeline and RAG system in this roadmap is built on the fact
that a dot product between two of these vectors is a usable measure of relevance.

Mental model: **an embedding table is a dictionary from id to point; the model learns
where to put the points.**

## The mental model

```text
  token id  ->  embedding table E (vocab, d_model)  ->  vector (d_model,)

        E =
        +--------------------------------+
   "cat"|  0.21 -0.04  0.88  ...  0.11   |
   "dog"|  0.19 -0.02  0.84  ...  0.15   |   cat and dog: close
   "car"| -0.55  0.31  0.02  ... -0.40   |   car: far away
        +--------------------------------+
                    |
                    v
        geometry of the space (2-D sketch)

              cat .  . dog
                   .
             car
                          .  truck

  similarity = cosine(a, b) = (a·b) / (|a| |b|)   ->  -1 .. 1
```

## Core concepts

### The table is a parameter, trained like any other

`nn.Embedding(vocab, d_model)` is a matrix of shape `(vocab, d_model)` — no bias, no
nonlinearity. It is updated by gradient descent exactly like a weight matrix, and the
gradient for row `i` is non-zero only when token `i` appears in the batch. That
sparsity is why rare tokens learn slowly, and why subword tokenisation matters: it
shrinks the vocabulary so each row sees more updates.

### Cosine similarity is the standard measure, and normalisation matters

`cos(a, b) = (a·b) / (|a| |b|)`. It ignores magnitude and compares direction, which is
usually what you want: a long document and a short one about the same topic should score
high. If your vectors are already L2-normalised, cosine similarity equals the dot
product, which is why vector databases ask you to normalise on insert — it makes the
inner-product index equivalent to a cosine index.

### Distance and similarity are different, and mixing them is a bug

Euclidean distance grows with magnitude; cosine does not. Sorting by `-distance` is not
the same as sorting by `similarity`. Pick one metric, apply it consistently at index and
query time, and record which one your index was built with. A mismatch produces results
that are plausible and wrong — the hardest kind to notice.

### Contextual embeddings are not a lookup

In a transformer, the vector a token contributes to the next layer is not its table row
— it is the row plus positional information plus everything attention mixed in. So "bank"
in "river bank" and "bank account" start from the same row and end up different. Static
embedding models (one vector per word) and contextual models (one vector per occurrence)
answer different questions; the retrieval stack in this roadmap uses contextual ones.

### What the geometry cannot do

Embeddings encode co-occurrence, not truth. Similarity is not entailment: "not
approved" and "approved" are extremely close in vector space and opposite in meaning.
Nothing in the geometry knows about negation, ordering or scope. That is why retrieval
systems need reranking and why RAG systems need the model to read the passage rather
than trust the score.

## Code

Train an embedding table on a tiny corpus with a skip-gram-style objective, then look at
nearest neighbours. On a toy corpus the neighbours are only as good as the data — the
point is the mechanics, not the quality.

```python
from __future__ import annotations

import torch
from torch import nn

# Toy corpus: two clearly separate topics.
sentences = [
    "cat dog pet feed", "dog cat pet tail", "pet cat feed bowl",
    "server gpu cluster node", "gpu server node rack", "cluster node server gpu",
]
vocab = sorted({w for s in sentences for w in s.split()})
stoi = {w: i for i, w in enumerate(vocab)}
pairs = [(stoi[a], stoi[b]) for s in sentences for a, b in zip(s.split(), s.split()[1:])]
pairs += [(b, a) for a, b in pairs]                       # both directions

ids = torch.tensor(pairs)                                  # (n_pairs, 2)
model = nn.Embedding(len(vocab), 8)                        # (vocab, 8)
head = nn.Linear(8, len(vocab))                            # (8, vocab) — predict context
opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=0.05)

for step in range(400):
    opt.zero_grad()
    center, context = ids[:, 0], ids[:, 1]                 # both (n_pairs,)
    logits = head(model(center))                           # (n_pairs, 8) -> (n_pairs, vocab)
    loss = nn.functional.cross_entropy(logits, context)
    loss.backward()
    opt.step()
print("final loss:", round(loss.item(), 4))

E = model.weight.detach()                                  # (vocab, 8)
E = E / E.norm(dim=1, keepdim=True)                        # L2-normalise rows
sim = E @ E.T                                              # (vocab, vocab) cosine matrix
for word in ("cat", "gpu"):
    scores, idx = sim[stoi[word]].topk(3)
    print(word, "->", [vocab[i] for i in idx], [round(float(s), 2) for s in scores])
```

The retrieval half — embed, normalise, rank — with the metric discipline from above. You
should see the correct document first for each query.

```python
from __future__ import annotations

import numpy as np


def cosine_rank(query: np.ndarray, docs: np.ndarray) -> list[int]:
    """query (d,), docs (n, d) -> indices sorted by cosine similarity, best first."""
    q = query / (np.linalg.norm(query) + 1e-12)
    d = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    sims = d @ q                                  # (n, d) @ (d,) -> (n,)
    return np.argsort(-sims).tolist()             # descending similarity


docs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])   # (3, 2)
query = np.array([1.0, 0.05])                           # (2,)
print(cosine_rank(query, docs))                         # [0, 1, 2] — the near-duplicate ranks second
```

TypeScript, for the service that ranks candidates returned by a vector index. Note the
explicit normalisation and the single metric choice, which is where these bugs live.

```ts
export function cosineRank(query: number[], docs: number[][]): number[] {
  const norm = (v: number[]): number => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const qn = norm(query) || 1e-12;
  const scores = docs.map((d, i) => {
    const dn = norm(d) || 1e-12;
    let dot = 0;
    for (let j = 0; j < d.length; j += 1) dot += d[j] * query[j];
    return { i, score: dot / (dn * qn) };
  });
  return scores.sort((a, b) => b.score - a.score).map((s) => s.i);
}
```

## Real use case

An internal knowledge assistant let employees search policy documents. The first version
stored vectors with an inner-product index and queried with cosine similarity — a
mismatch nobody noticed because most scores were close. Short documents, whose vectors
had smaller norms, were systematically ranked below long ones, so one-line policies were
effectively invisible.

The fix was two lines: L2-normalise vectors before insert, and use the same metric at
query time. Because normalised vectors make inner product equal to cosine, the existing
index needed no rebuild — only a re-ingest. The lesson generalises: an embedding is only
useful if index and query agree on what distance means.

## Common mistakes

- **Mistake:** comparing raw dot products across documents of different length — **Instead:** normalise, or use cosine; unnormalised dot product rewards long vectors.
- **Mistake:** mixing metrics between index build and query — **Instead:** pick one metric, store it with the index, and assert it in code.
- **Mistake:** treating similarity as relatedness of meaning in all cases — **Instead:** remember negation; "not approved" and "approved" are near-neighbours.
- **Mistake:** assuming a word has one vector — **Instead:** in a transformer the vector is contextual; only static tables give one row per word.
- **Mistake:** a vocabulary so large that rare rows never train — **Instead:** use subword tokenisation so every row gets updates.
- **Mistake:** comparing embeddings from two different models — **Instead:** they live in unrelated spaces; vectors from model A are meaningless to model B.

## Checkpoint

1. Train the toy embedding above, then print the full `(vocab, vocab)` similarity matrix as a table. Do the pet words cluster separately from the infrastructure words?
2. Change the objective to predict the center from the context (reverse the direction) and retrain. Does the clustering survive?
3. Write a function that takes a query vector and a matrix of document vectors and returns the top `k` indices by both cosine and Euclidean distance. Construct an example where the two rankings differ, and explain why.

**Self-check:** pet words should be mutually more similar than to the infrastructure words, because the training pairs never mix the two topics, so their rows are pulled in different directions.

## Go deeper

- [Embeddings in Vector Databases](../09-vector-databases/01-embeddings.md) — the production version of this table.
- [Similarity Metrics](../09-vector-databases/02-similarity-metrics.md) — cosine, dot product and Euclidean in depth.
- [Tokenization](../06-llm-fundamentals/02-tokenization.md) — how the vocabulary these rows index is built.
- [Transformers](07-transformers.md) — where the embedding table is the first layer.

---
[← Previous: Transformers](07-transformers.md) · [Stage index](README.md) · [Next: Transfer Learning](09-transfer-learning.md)
