# 1. Embeddings: Text as Points in Space

> **Stage 09 · Vector Databases** — turns text into a fixed-length vector so that "meaning" becomes arithmetic a machine can do.
>
> **You'll be able to:** embed documents and queries, and explain why a similarity score between two embeddings is meaningful at all.
>
> **Prerequisites:** [Stage 06 — LLM Fundamentals](../06-llm-fundamentals/README.md) · [Self-Attention](../06-llm-fundamentals/04-self-attention.md)

## Why this exists

Keyword search fails on the obvious case. A user types "how do I reset my password" and the help article is titled "Account recovery steps". The word *password* appears in neither place in a matching form, so `WHERE body LIKE '%password%'` returns nothing useful. Before embeddings, the only fixes were hand-built synonym lists and stemming rules — brittle, language-specific, and always one phrasing behind your users.

An embedding fixes this by mapping text to a point in a few hundred or few thousand dimensions, trained so that texts used in similar contexts land near each other. "Reset my password" and "account recovery" end up close even though they share no tokens.

The mental model in one sentence: **an embedding turns meaning into a direction, and similarity into the angle between two directions.**

## The mental model

```text
 "reset my password"  ──┐
                        ├─► embed model ──► [0.021, -0.118, 0.004, ... ]  (1536 numbers)
 "account recovery"   ──┘

  2-D shadow of that space (illustrative):

        ▲ dim 2
        │        ● "reset my password"
        │      ●   "account recovery"
        │
        │                     ● "invoice for March"
        │                   ●  "billing statement"
        └──────────────────────────────────────► dim 1

  close together = similar meaning. Distance is the whole product.
```

## Core concepts

### A vector is a list of floats, and its length is arbitrary

An embedding is `list[float]` of a fixed size decided by the model — 384, 768, 1536 and 3072 are common. The model is trained with a contrastive objective: pull paraphrases together, push unrelated text apart. You do not interpret individual dimensions; no dimension means "formality". Only relative positions matter.

Crucially, the **magnitude** of the vector is mostly an artefact of the text's length and token frequency, not its meaning. Two texts about the same thing can have very different lengths.

### Direction carries meaning, length does not

If length is noise, the useful part is the direction. That is why nearly every retrieval system normalises embeddings to unit length before storing them: once every vector has length 1, all points live on a sphere and only angle remains.

Normalising is one line: divide by the L2 norm.

```text
v = [3, 4]        ||v|| = sqrt(9 + 16) = 5
v̂ = [0.6, 0.8]    ||v̂|| = 1
```

After this, the dot product of two vectors *is* their cosine similarity. Keep that fact; the next file builds on it.

### One model, one space — and you cannot mix models

Vectors from model A and model B are not comparable. Dimension counts may even match (two different 768-dim models), and the numbers still mean nothing across the boundary. The practical rule: **the model id is part of your index's schema.** Store it as metadata on every vector, and if you change models you must re-embed the whole corpus. This is the single most expensive operational fact in this stage, and it is why [Embedding and Storage Design](../10-rag/04-embedding-storage.md) treats re-indexing as a first-class design concern.

### Queries and documents can use different instructions

Many embedding models are asymmetric: documents are embedded as-is, while queries get a short instruction prefix, or the model has separate `embed_documents` and `embed_query` methods. If you skip the prefix, recall drops quietly — results still come back, they are just worse. Check the model card for whether it expects one.

## Code

This embeds three sentences with a sentence-transformers model and prints a small similarity matrix, so you can see related sentences scoring higher than unrelated ones. Model ids change; check the model card on the hub for the current name and dimension.

```python
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims; check the hub for current ids

model = SentenceTransformer(MODEL_ID)

texts = [
    "How do I reset my password?",
    "Steps to recover access to your account.",
    "What is the refund policy for annual plans?",
]

# normalise_embeddings=True makes every row unit length, so dot == cosine
emb: np.ndarray = model.encode(texts, normalize_embeddings=True)
print(emb.shape)          # (3, 384)
print(np.linalg.norm(emb[0]))  # ~1.0

sim = emb @ emb.T          # 3x3 cosine matrix, because rows are unit length
for i, row in enumerate(sim):
    print(texts[i][:34].ljust(36), " ".join(f"{v:+.3f}" for v in row))
```

You should see the (0,1) and (1,0) cells clearly above the (0,2) and (1,2) cells: the two password sentences agree, the refund sentence does not. Exact numbers depend on the model build.

The second snippet shows the asymmetry rule and a batching detail that matters in production: embedding a 50,000-chunk corpus one string at a time is mostly overhead.

```python
from __future__ import annotations

import numpy as np

def embed_corpus(model, chunks: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed in batches, return float32 unit-length matrix (n, dim)."""
    out: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        out.append(model.encode(batch, normalize_embeddings=True, batch_size=batch_size))
    return np.vstack(out).astype(np.float32)   # float32 halves memory vs float64

def embed_query(model, query: str) -> np.ndarray:
    # If the model card specifies a query prefix, prepend it here:
    # query = "query: " + query
    return model.encode([query], normalize_embeddings=True)[0]
```

## Real use case

A 40-person B2B SaaS company had a support portal with 1,200 help articles, searched by a Postgres `ILIKE` query. Their analytics showed a large share of searches ended in a zero-result page, and the top zero-result queries were paraphrases of existing articles — customers wrote "can't log in" while the article said "authentication troubleshooting".

They added one table: article id, the article's embedding, and the model id used. Search became: embed the query, cosine-similarity against all 1,200 rows, return the top 5. No index, no service — a sequential scan over 1,200 × 384 floats is trivially fast.

Two design choices from this file made it work. First, they normalised at write time so the query path was a plain dot product, which kept the endpoint simple. Second, they stored `model_id` next to every vector; six months later they switched embedding models, and the migration script simply selected rows whose `model_id` differed and re-embedded only those, instead of the whole table blindly.

The failure mode before: synonym gaps. Zero-result searches were the visible symptom, but the deeper cost was that customers had to guess the article's vocabulary before they could read it.

## Common mistakes

- **Mistake:** comparing embeddings from two different models or versions — **Instead:** store the model id with every vector and refuse to mix; re-embed on change.
- **Mistake:** skipping normalisation and then using a dot-product index — **Instead:** normalise at write time (or pick an index that uses cosine) so scores are comparable across document lengths.
- **Mistake:** forgetting the query prefix the model card asks for — **Instead:** read the card and keep `embed_query` and `embed_documents` as separate functions from day one.
- **Mistake:** treating the raw similarity number as a probability — **Instead:** treat it as a ranking score only; thresholds must be calibrated per model and per corpus.
- **Mistake:** embedding a huge corpus in a single call — **Instead:** batch it and persist to disk as you go, so a crash does not lose an hour of work.

## Checkpoint

1. Compute the unit-length version of `[3, 4]` and then of `[6, 8]`. What does the answer tell you about the relationship between magnitude and meaning?
2. Embed ten sentences from one document and ten from a different document. Print the full 20×20 similarity matrix. Which cells look like noise, and why?
3. Take a corpus of 200 short texts, embed them with two different models, and check whether the top-5 neighbours of one query agree. Write two sentences on what you would have to do to switch models safely in production.

**Self-check:** `[3, 4]` and `[6, 8]` both normalise to `[0.6, 0.8]` — they point in the same direction, so they are identical in meaning-space even though one is twice as long. Magnitude is not meaning.

## Go deeper

- [Similarity Metrics](02-similarity-metrics.md) — what exactly you are measuring between two of these points.
- [Stage 04 — Embeddings](../04-deep-learning/08-embeddings.md) — how these vectors are trained, if you want the model side.
- [Chunking](../10-rag/03-chunking.md) — what you should embed, once you know how embedding works.

---
[← Previous: Stage 08 — Prompt and Context](../08-prompt-and-context/README.md) · [Stage index](README.md) · [Next: 2. Similarity Metrics](02-similarity-metrics.md)
