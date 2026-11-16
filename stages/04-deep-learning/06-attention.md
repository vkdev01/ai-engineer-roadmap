# 6. Attention

> **Stage 04 · Deep Learning** — soft lookup: how a model decides which parts of its input matter right now
>
> **You'll be able to:** implement scaled dot-product attention in NumPy from the formula, and explain every shape in it.
>
> **Prerequisites:** [LSTMs and GRUs](05-lstms-grus.md) · [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md)

## Why this exists

A recurrent model compresses everything it has seen into one fixed vector. If the
answer depends on one detail from forty steps ago, that detail competes with everything
else for space in the vector. And because the model processes one step at a time, it
cannot be parallelised across the sequence.

Attention replaces "compress the past into a vector" with "keep the past, and look it
up". Each position produces a **query** describing what it is looking for, and every
position produces a **key** describing what it offers. Comparing the query against all
keys gives a score per position; those scores become weights; the weighted average of
the **values** is what the position receives. No bottleneck, no sequential dependency,
and the lookup is learned end to end.

Mental model: **attention is a soft dictionary lookup — the query is your search term,
the keys are the index, and you get back a blend of the values.**

## The mental model

```text
  Q (seq_q, d_k)   K (seq_k, d_k)   V (seq_k, d_v)

  1. scores   S = Q @ K.T                (seq_q, seq_k)     "how well does each query match each key"
  2. scale    S = S / sqrt(d_k)          (seq_q, seq_k)     keeps scores from growing with d_k
  3. mask     S[where masked] = -inf     (seq_q, seq_k)     causal: no looking ahead
  4. weights  A = softmax(S, axis=-1)    (seq_q, seq_k)     each row sums to 1
  5. output   O = A @ V                  (seq_q, d_v)       blend of values

  one row of A:
     key1  key2  key3  key4
    [ 0.05  0.70  0.20  0.05 ]  ->  O_row = 0.05*V1 + 0.70*V2 + 0.20*V3 + 0.05*V4
```

## Core concepts

### Q, K, V are three projections of the same input

In self-attention the input `X` of shape `(seq, d_model)` is projected three ways:
`Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v`, with `W_q, W_k` of shape
`(d_model, d_k)` and `W_v` of shape `(d_model, d_v)`. The projections are what the model
learns. In cross-attention — a decoder attending to an encoder — queries come from one
sequence and keys and values from another.

### The dot product is the similarity

`Q @ K.T` gives `(seq_q, seq_k)`: entry `(i, j)` is how much position `i`'s query
aligns with position `j`'s key. This is a learned similarity, not a fixed metric —
which is why it can capture relations that cosine similarity cannot.

### Why divide by the square root of d_k

A dot product of two `d_k`-dimensional vectors with roughly unit-variance entries has
variance around `d_k`. With `d_k = 64`, scores can be large enough that softmax
saturates: one weight becomes almost 1 and the gradient through softmax vanishes.
Dividing by `sqrt(d_k)` restores unit variance, keeping the softmax in a responsive
range. It is a numerical stability fix, not a modelling choice.

### Softmax turns scores into a convex combination

Softmax exponentiates and normalises along the key axis, so each query's weights are
non-negative and sum to 1. The output row is therefore a weighted average of value
vectors — an interpolation, not a selection. Nothing is discarded; things are
down-weighted.

### Masks encode what is allowed

Causal (autoregressive) attention sets scores above the diagonal to `-infinity` before
softmax, so position `i` can only see positions `<= i`. Padding masks do the same for
padding tokens so they contribute nothing. The mask must be applied *before* softmax,
and with `-1e9` or `-inf` rather than 0 — a score of 0 is a legitimate score.

## Code

Scaled dot-product attention in NumPy, with every shape annotated and a causal mask.
You should see an output of `(3, 4)` and attention rows that sum to 1.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)      # numerical stability, same shape
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)   # (seq_q, d_k) @ (d_k, seq_k) -> (seq_q, seq_k)

    if causal:
        seq_q, seq_k = scores.shape[-2:]
        # Upper triangle (j > i) is the future: forbid it.
        future = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
        scores = np.where(future, -np.inf, scores)

    weights = softmax(scores, axis=-1)               # (seq_q, seq_k), rows sum to 1
    return weights @ V, weights                      # (seq_q, seq_k)@(seq_k, d_v) -> (seq_q, d_v)


seq_q, seq_k, d_k, d_v = 3, 4, 8, 6
Q = rng.normal(size=(seq_q, d_k))     # (3, 8)
K = rng.normal(size=(seq_k, d_k))     # (4, 8)
V = rng.normal(size=(seq_k, d_v))     # (4, 6)

out, w = scaled_dot_product_attention(Q, K, V)
print(out.shape, w.shape)             # (3, 6) (3, 4)
print(w.sum(axis=-1).round(6))        # [1. 1. 1.]

# Causal: position 0 attends only to key 0, position 1 to keys 0-1, and so on.
out_c, w_c = scaled_dot_product_attention(Q[:, :4], K, V, causal=True)
print(w_c.round(2))
```

The batched, multi-head version — the shape you will meet in every real implementation.
Batches and heads are just extra leading axes; nothing about the maths changes.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(1)
batch, heads, seq, d_model = 2, 4, 5, 32
d_head = d_model // heads                      # 8

X = rng.normal(size=(batch, seq, d_model))     # (2, 5, 32)
W_qkv = rng.normal(size=(d_model, 3 * d_model)) * 0.02
qkv = X @ W_qkv                                # (2, 5, 96)
Q, K, V = np.split(qkv, 3, axis=-1)            # three (2, 5, 32)


def split_heads(t: np.ndarray) -> np.ndarray:
    # (batch, seq, d_model) -> (batch, heads, seq, d_head)
    b, s, _ = t.shape
    return t.reshape(b, s, heads, d_head).transpose(0, 2, 1, 3)


Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
print(Qh.shape)                                # (2, 4, 5, 8)

scores = Qh @ Kh.swapaxes(-1, -2) / np.sqrt(d_head)   # (2, 4, 5, 5)
w = np.exp(scores - scores.max(-1, keepdims=True))
w = w / w.sum(-1, keepdims=True)
attn = w @ Vh                                  # (2, 4, 5, 8)
merged = attn.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
print(merged.shape)                            # (2, 5, 32) — back to d_model
```

## Real use case

A contract-review tool highlights the clause a question refers to. The first version
encoded the whole contract into one vector and compared it with the question — a
single-vector similarity that worked for short contracts and failed on 40-page ones,
because the clause that mattered contributed a tiny fraction of one embedding.

Attention changed the shape of the problem: the question's query attends over every
clause's key, and the attention weights themselves became the highlight. The product
gained a feature for free — showing the weights tells the user *why* a clause was
chosen, which a single embedding could never expose. The design choice that made it
work was cross-attention rather than self-attention: queries from the question, keys
and values from the document.

## Common mistakes

- **Mistake:** forgetting the `1/sqrt(d_k)` scale — **Instead:** always scale; without it, softmax saturates as `d_k` grows and gradients vanish.
- **Mistake:** masking after softmax — **Instead:** set masked scores to `-inf` before softmax, so those positions get exactly zero weight.
- **Mistake:** masking with 0 instead of `-inf` — **Instead:** 0 is a valid score; only `-inf` guarantees zero weight after softmax.
- **Mistake:** mixing up the axes in `Q @ K.T` — **Instead:** the result is `(seq_q, seq_k)`; if you get `(seq_k, seq_q)` you transposed the wrong one.
- **Mistake:** forgetting the head reshape before the attention maths — **Instead:** reshape to `(batch, heads, seq, d_head)` first, then attend per head.
- **Mistake:** using a fixed similarity instead of learned projections — **Instead:** keep `W_q`, `W_k`, `W_v` as parameters; that is what makes attention expressive.

## Checkpoint

1. Implement the function above without looking, then verify `w.sum(axis=-1)` is all ones for random inputs of three different shapes.
2. Remove the `1/sqrt(d_k)` scale, set `d_k=512`, and print the max attention weight and the softmax gradient magnitude. Describe what happens.
3. Apply a padding mask that zeroes the last two keys, and confirm those columns of the attention matrix are exactly zero while each row still sums to 1.

**Self-check:** without scaling, dot products grow with `d_k` so softmax pushes almost all weight onto one key, giving a near-one-hot matrix whose gradient is close to zero — which is exactly why the `1/sqrt(d_k)` term exists.

## Go deeper

- [Transformers](07-transformers.md) — attention wrapped in residuals, layer norm and a feedforward block.
- [Self-Attention](../06-llm-fundamentals/04-self-attention.md) — the same mechanism viewed from the LLM side.
- [LSTMs and GRUs](05-lstms-grus.md) — the bottleneck attention removed.
- [KV Cache](../16-inference/02-kv-cache.md) — what inference systems store so attention does not recompute keys and values.

---
[← Previous: LSTMs and GRUs](05-lstms-grus.md) · [Stage index](README.md) · [Next: Transformers](07-transformers.md)
