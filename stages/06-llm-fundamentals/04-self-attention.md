# 4. Self-Attention From Scratch

> **Stage 06 · LLM Fundamentals** — Queries, keys and values: how every token decides which other tokens to read, written out with shapes at every step.
>
> **You'll be able to:** implement scaled dot-product attention and multi-head attention from memory, and explain each term in the formula.
>
> **Prerequisites:** [6. Attention](../04-deep-learning/06-attention.md) · [1. GPT and BERT: Decoder-Only vs Encoder-Only](01-gpt-and-bert.md)

## Why this exists

You have used attention as a layer in [7. Transformers](../04-deep-learning/07-transformers.md). This file opens it up, because attention is where most shape bugs and most "why is my model slow" questions live. It is also the mechanism whose cost grows with context length, which is why long contexts are expensive and why the KV cache exists.

The mental model: *each token asks a question (query), every token advertises what it contains (key), and the match score decides how much of each token's content (value) to mix in.*

## The mental model

```text
        x: (B, T, d_model)
             │  three linear projections
   ┌─────────┼─────────┐
   Q         K         V            each (B, T, d_head)
   │         │         │
   └─ Q @ Kᵀ ─┐        │           scores (B, T, T)
              │        │
        / sqrt(d_head) │           scale: keeps variance ~1
              │        │
        + mask (causal)│           -inf where the future is
              │        │
          softmax(dim=-1)          rows sum to 1: attention weights
              │        │
              └─ @ V ──┘           (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
```

Read the last row as: for each output position, a weighted average of all value vectors, where the weights come from how well that position's query matched each key.

## Core concepts

### Q, K, V are three views of the same token

Each token's vector is projected three ways. The **query** is what this token is looking for, the **key** is what it offers as a match target, and the **value** is the content it contributes once matched. They are separate projections so the model can learn "match on this, but copy that" — a distinction a single vector could not make.

### The dot product is the score, and `sqrt(d_head)` is not cosmetic

`Q @ Kᵀ` gives one number per (query, key) pair. Without scaling, the variance of that dot product grows with `d_head`, so with large head sizes the softmax saturates: one weight becomes ~1 and the rest ~0, and gradients vanish. Dividing by `sqrt(d_head)` restores unit variance. This is the single most-missed detail when reimplementing attention.

### Softmax over the key axis makes weights a distribution

`softmax(dim=-1)` normalises across keys, so each query's weights sum to 1. Softmax over the wrong axis is a common bug: it still runs, still produces plausible numbers, and trains badly.

### The causal mask is additive `-inf`

For a decoder, positions after `t` are set to `-inf` before the softmax so their weight becomes exactly zero. It must be applied before softmax, not after — masking after normalisation leaves the remaining weights not summing to 1.

### Multi-head is one matmul plus a reshape

Instead of one attention over `d_model = 512`, run `h = 8` attentions over `d_head = 64` in parallel, then concatenate and project. Total compute is the same; the model gets eight independent "matching rules". The implementation trick is to compute all heads in one batched matmul by reshaping `(B, T, d_model)` into `(B, h, T, d_head)`.

### Cost is quadratic in sequence length

Scores are `(T, T)`. Doubling context quadruples both the score matrix and the attention compute. That is why long contexts are expensive, why flash-attention kernels matter, and why inference keeps a KV cache instead of recomputing keys and values for the whole prefix.

## Code

Scaled dot-product attention with a causal mask, every shape annotated. This runs as written.

```python
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    """q,k,v: (B, T, d_head) -> (B, T, d_head)."""
    d_head = q.size(-1)
    # (B, T, d) @ (B, d, T) -> (B, T, T): one score per (query, key) pair
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
    if causal:
        T = q.size(-2)
        # upper triangle (excluding diagonal) is forbidden: token t cannot see t+1
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))     # additive -inf
    weights = scores.softmax(dim=-1)                         # over KEYS -> rows sum to 1
    return weights @ v                                       # (B,T,T) @ (B,T,d) -> (B,T,d)


torch.manual_seed(0)
B, T, d_head = 2, 4, 8
q, k, v = (torch.randn(B, T, d_head) for _ in range(3))
out = attention(q, k, v)
print(out.shape)                       # torch.Size([2, 4, 8])

# sanity checks worth keeping in a test
scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)
w = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
print(w[0, 0])                         # only position 0 is visible: weight 1.0
print(w.sum(-1))                       # all ones: each query distributes full mass
```

Now multi-head attention as a module: one projection to QKV, a reshape into heads, attention, then concatenation.

```python
class MultiHeadAttention(nn.Module):
    """d_model split into h heads of size d_head = d_model // h."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide by n_heads"
        self.h = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # one fused projection
        self.proj = nn.Linear(d_model, d_model)                 # output projection

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape                       # (B, T, d_model)
        qkv = self.qkv(x)                       # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=-1)           # three (B, T, d_model)
        # (B, T, h, d_head) -> (B, h, T, d_head): heads become a batch dimension
        q, k, v = (t.view(B, T, self.h, self.d_head).transpose(1, 2)
                   for t in (q, k, v))
        out = attention(q, k, v, causal=True)   # (B, h, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)   # (B, T, d_model) concatenated
        return self.proj(out)


mha = MultiHeadAttention(d_model=64, n_heads=8)
print(mha(torch.randn(2, 5, 64)).shape)         # torch.Size([2, 5, 64])
```

## Real use case

An engineer ports a small transformer to a new runtime and the model outputs gibberish — not random, but repetitive. Weights load fine, shapes match, no errors.

Two bugs interact. First, the port omitted the `1/sqrt(d_head)` scaling; scores grow large, softmax saturates to one-hot, and the model becomes a copy-the-previous-token machine, which reads as repetition. Second, the causal mask was applied *after* softmax, so the remaining weights no longer sum to one and the output magnitudes drift with sequence length — subtle on short prompts, severe on long ones.

Both are caught by the two checks in the snippet above: rows of the weight matrix must sum to 1, and row 0 of a causal mask must be `[1, 0, 0, 0]`. Those two assertions, run on a 4-token toy input, would have found both bugs in a minute. This is why every serious implementation of attention ships with a shape-and-sum test.

## Common mistakes

- **Mistake:** forgetting the `1/sqrt(d_head)` scale. — **Instead:** always scale before the softmax; without it attention saturates.
- **Mistake:** masking after softmax. — **Instead:** add `-inf` to scores, then softmax.
- **Mistake:** `softmax(dim=-2)`. — **Instead:** normalise over the key axis, `dim=-1`.
- **Mistake:** reshaping heads without `transpose(1, 2)`. — **Instead:** move heads to the batch axis before the matmul, then transpose back.
- **Mistake:** assuming attention output needs no projection. — **Instead:** concatenated heads must be projected back to `d_model` for the residual stream to line up.

## Checkpoint

1. Remove the scaling factor and print the softmax weights for `d_head = 64`. How many weights are non-negligible, and what does that mean for learning?
2. Implement non-causal attention (a bidirectional encoder) by passing `causal=False`, and confirm that row 0 now attends to every position.
3. Add a padding mask on top of the causal mask so padded positions receive zero weight. Which operation combines the two masks, and why is addition the right one?

**Self-check:** Without scaling, the largest weight dominates and the rest collapse toward zero — effectively hard attention, with gradients through the softmax close to zero, so most positions stop learning.

## Go deeper

- [5. Pretraining: Objectives and Scale](05-pretraining.md) — what this mechanism is trained to do.
- [2. KV Cache](../16-inference/02-kv-cache.md) — how inference avoids recomputing K and V.
- [6. Attention](../04-deep-learning/06-attention.md) — the intuition-first version, if the maths felt sudden.
- [10. Ship: A Mini Transformer](../04-deep-learning/10-ship-mini-transformer.md) — this module inside a full model.

---
[← Previous: Context Windows](03-context-windows.md) · [Stage index](README.md) · [Next: Pretraining: Objectives and Scale](05-pretraining.md)
