# 7. Transformers

> **Stage 04 · Deep Learning** — assembling attention into the block that runs the modern AI stack
>
> **You'll be able to:** build a transformer block in PyTorch, annotate every tensor shape in it, and explain what each sub-layer contributes.
>
> **Prerequisites:** [Attention](06-attention.md) · [Optimizers](03-optimizers.md)

## Why this exists

Attention alone is a lookup. On its own it has no notion of position, no capacity to
transform what it retrieves, and no stable way to stack twenty of them. The transformer
block is the engineering around attention that makes depth possible: a residual
connection so gradients reach the bottom, a normalisation so activations stay in range,
and a position-wise feedforward network so each position can actually compute with what
it retrieved.

That block, repeated, is GPT and BERT. Understanding it is not academic — it is what
lets you reason about context length, why flash attention helps, what a KV cache
stores, and why a 70B model costs what it does.

Mental model: **a block is `x + attn(norm(x))`, then `x + mlp(norm(x))` — two residual
branches, each normalised on the way in.**

## The mental model

```text
  x  (batch, seq, d_model)
  |
  +--> LayerNorm --> MultiHeadSelfAttention --> Dropout --+   (residual)
  |                                                       |
  +-----------------------------------------------(add)---+
  |
  +--> LayerNorm --> FeedForward (d_model -> 4*d_model -> d_model) --+  (residual)
  |                                                                  |
  +----------------------------------------------------------(add)---+
  |
  v
  x'  (batch, seq, d_model)     — same shape in, same shape out, so you can stack N

  attention:     mixes information ACROSS positions
  feedforward:   transforms information WITHIN each position
  residual:      keeps the gradient path short, lets a block learn "no change"
  layernorm:     normalises over the feature axis, per position
```

## Core concepts

### The residual connection is why depth works

`x + f(x)` means the block only has to learn a correction to the identity. Gradients
flow straight back through the `+`, so a twenty-layer stack trains even though each
layer's function is complicated. Remove the residuals and a deep transformer fails to
train at all — this is not a small optimisation, it is structural.

### LayerNorm normalises per position, over features

For each position, subtract the mean and divide by the standard deviation across the
`d_model` features, then scale and shift with two learned vectors. It does not mix
across the batch or across positions, so it behaves the same at training and inference.
Pre-norm (normalise before the sub-layer, as above) is the modern default because it
trains more stably at depth; post-norm was the original arrangement.

### The feedforward network is where the parameters are

Two linear layers with a nonlinearity: `d_model -> 4*d_model -> d_model`, typically
GELU or ReLU in between. About two thirds of a transformer block's parameters live here.
Attention decides *what to look at*; the feedforward layer decides *what to compute
about it*. It is applied identically to each position, so it adds no cross-position
mixing.

### Multi-head attention runs several lookups in parallel

Instead of one attention over `d_model` dimensions, split into `h` heads of size
`d_model / h`, attend in each, concatenate, and project back with `W_o`. Each head can
specialise — one tracks syntax, another tracks a repeated entity. The output projection
is what merges them.

### Position information must be injected

Self-attention is permutation-invariant: shuffle the tokens and the output is shuffled
the same way. So position has to be added explicitly — learned position embeddings, or
rotary position embeddings (RoPE) applied to queries and keys. Without it, "dog bites
man" and "man bites dog" are the same input.

## Code

A complete pre-norm transformer block in PyTorch, shapes annotated. Run it and confirm
the output shape equals the input shape — that is the invariant that lets you stack it.

```python
from __future__ import annotations

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must split evenly across heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # fused Q, K, V
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape                              # (batch, seq, d_model)
        q, k, v = self.qkv(x).split(C, dim=-1)          # three (B, T, C)
        # (B, T, C) -> (B, heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)   # (B, heads, T, T)
        if causal:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))           # rows sum to 1
        y = attn @ v                                              # (B, heads, T, d_head)
        y = y.transpose(1, 2).reshape(B, T, C)                     # merge heads
        return self.out(y)                                        # (B, T, C)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))     # residual: attention across positions
        x = x + self.mlp(self.norm2(x))      # residual: transform per position
        return x


block = Block(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)                   # (batch=2, seq=10, d_model=64)
print(block(x).shape)                        # (2, 10, 64) — unchanged
print(sum(p.numel() for p in block.parameters()), "parameters")
```

The full model skeleton — embedding, positional information, N blocks, language-model
head. This is the shape of a decoder-only model in about thirty lines.

```python
from __future__ import annotations

import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 128) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)        # (vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)      # (max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)   # (d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape                               # (batch, seq) of token ids
        pos = torch.arange(T, device=idx.device)       # (T,)
        x = self.tok(idx) + self.pos(pos)              # (B, T, d_model) + (T, d_model)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))                 # (B, T, vocab) — logits per position


model = TinyLM(vocab=1000)
logits = model(torch.randint(0, 1000, (2, 16)))
print(logits.shape)     # (2, 16, 1000)
```

## Real use case

A team serving an on-device autocomplete needed a model small enough to run on a phone
keyboard. Their first attempt stacked eight attention layers with post-norm and no
residuals on the feedforward branch; training loss stalled and the deeper they went the
worse it got. The diagnosis was structural, not a hyperparameter: without residuals the
gradient had to pass through eight blocks of nonlinearity, and with post-norm the
activations at depth were unstable.

Switching to pre-norm blocks with both residuals, reducing to four layers and shrinking
`d_model` while keeping four heads fixed the run — and the model fitted the memory
budget. The shape invariant from this file (a block returns what it receives) is what
made the layer count a tunable number rather than a rewrite.

| Change | Effect |
|---|---|
| Post-norm to pre-norm | Stable training at depth |
| Adding feedforward residual | Gradient reaches the embedding |
| 8 layers to 4 with smaller `d_model` | Fits the on-device budget |
| Fused QKV projection | One matmul instead of three |

## Common mistakes

- **Mistake:** dropping the residual connections — **Instead:** keep both `x + attn(...)` and `x + mlp(...)`; without them deep stacks do not train.
- **Mistake:** normalising over the wrong axis — **Instead:** LayerNorm normalises over features per position, not across the batch.
- **Mistake:** forgetting position embeddings — **Instead:** add them before the blocks, or use RoPE inside attention; otherwise the model is order-blind.
- **Mistake:** a `d_model` that does not divide by `n_heads` — **Instead:** assert it, as the code does; a silent reshape will scramble head contents.
- **Mistake:** using the padded positions in the loss — **Instead:** ignore the padding index when computing cross-entropy, or the model learns to predict padding.
- **Mistake:** `view` after a `transpose` without care — **Instead:** use `reshape` (or `contiguous()` then `view`) when merging heads back.

## Checkpoint

1. Build a `Block`, pass a `(2, 10, 64)` tensor through it, and assert the output shape matches. Then pass `(1, 1, 64)` and explain why it still works.
2. Set `causal=False` and compare the attention matrix to the causal version. Describe which entries changed and why.
3. Add a padding mask so positions beyond a given length contribute nothing, and verify that changing a padded token does not change the output at real positions.

**Self-check:** the block is shape-preserving because both sub-layers are `(B, T, C) -> (B, T, C)` and the residual add requires matching shapes; a `(1, 1, 64)` input works because attention over a single position is just its own value vector.

## Go deeper

- [Attention](06-attention.md) — the mechanism inside the block, derived from scratch.
- [Ship: A Miniature Transformer](10-ship-mini-transformer.md) — train this architecture on a toy task.
- [GPT and BERT](../06-llm-fundamentals/01-gpt-and-bert.md) — how the same block becomes an encoder or a decoder.
- [Prefill and Decode](../16-inference/01-prefill-decode.md) — what the block's shapes mean for serving cost.

---
[← Previous: Attention](06-attention.md) · [Stage index](README.md) · [Next: Embeddings](08-embeddings.md)
