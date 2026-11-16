# 2. The KV Cache

> **Stage 16 · Inference** — the memory structure that makes generation possible and usually decides your batch size.
>
> **You'll be able to:** compute KV cache bytes per token from a model config, and use that number to size a deployment.
>
> **Prerequisites:** [Self-Attention](../06-llm-fundamentals/04-self-attention.md) · [Prefill and Decode](01-prefill-decode.md)

## Why this exists

Attention needs, for every token, a key vector and a value vector to compare against. When
token 500 is generated, it attends to keys and values of tokens 1 through 499. Recomputing
those from scratch each step would mean re-running the whole network for the whole prefix,
500 times.

The KV cache is the obvious fix: store the keys and values you already computed and append
to the store each step. Generation becomes O(n) work per token instead of O(n²).

The cost is memory, and it is not small. The cache grows linearly with context length and
with the number of concurrent requests, and it lives on the GPU alongside the weights. On a
24 GB card, weights are usually the smaller half of the bill. The mental model: **the KV
cache is the reason a long-context model needs more hardware than its parameter count
suggests.**

## The mental model

```text
per token, per layer:      [ K vector ]  [ V vector ]
                           head_dim        head_dim
                           x kv_heads      x kv_heads

KV bytes per token = 2 x n_layers x n_kv_heads x head_dim x bytes_per_element

one request, 8k context, 128 KiB/token  ->  1 GiB
32 requests, 8k context                 ->  32 GiB   (bigger than the weights)
```

## Core concepts

### The formula, term by term

```text
KV bytes/token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
```

- `2` — one key vector and one value vector per layer.
- `n_layers` — every transformer block has its own cache.
- `n_kv_heads` — note **kv** heads, not attention heads. Under grouped-query attention (GQA)
  several query heads share one key/value head, which is exactly why GQA exists.
- `head_dim` — the width of each head, typically `hidden_size / n_heads`, often 128.
- `dtype_bytes` — 2 for fp16 or bf16, 1 for fp8 or int8.

### A worked example

Take a typical 8B-class open model: 32 layers, 32 attention heads, 8 KV heads, head_dim 128,
bf16 weights and cache.

```text
per layer, per token:  2 * 8 * 128 * 2 bytes = 4096 bytes = 4 KiB
all 32 layers:         32 * 4 KiB           = 128 KiB per token
8192-token sequence:   8192 * 128 KiB       = 1,048,576 KiB = 1 GiB
```

So one full 8k conversation occupies 1 GiB of cache. Sixteen of those fill 16 GiB — which is
why an 8B model on a 24 GB card does not serve sixteen long conversations, no matter how
small the weights look.

Now remove GQA and use 32 KV heads instead of 8. Per token: `2 * 32 * 128 * 2 = 16 KiB` per
layer, `512 KiB` per token, `4 GiB` for the same 8k sequence. Four times the memory for the
same model size. GQA is not a quality trick, it is a memory trick.

### Cache dtype is a separate dial from weight dtype

You can store weights in int4 and keep the cache in fp16, or store the cache in fp8. Because
the cache often exceeds the weights at long context, quantizing the cache is frequently the
biggest single memory win. Halving cache dtype from 2 bytes to 1 byte halves the term above:
8 GiB of cache becomes 4 GiB. Quality effects are real but usually milder than quantizing
weights, because the cache is recomputed per conversation rather than fixed at export time.

### Cache grows during the conversation, which is a scheduling problem

You cannot know in advance how long an answer will be, so you cannot know the final cache
size. Reserve the maximum and you waste memory on every short request; grow on demand and you
risk running out mid-generation. Serving engines solve this with paged allocation, covered in
[PagedAttention](04-pagedattention.md).

## Code

This computes the cache for any Hugging Face config and prints a table you can paste into a
capacity plan. It reads the config from disk or a dict, so it works offline.

```python
from __future__ import annotations

from dataclasses import dataclass

DTYPE_BYTES: dict[str, float] = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0,
                                 "fp8": 1.0, "int8": 1.0}


@dataclass(frozen=True)
class KVBudget:
    bytes_per_token: int
    per_sequence_gib: float
    batch_gib: float


def kv_cache_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, dtype: str = "bfloat16"
) -> int:
    """2 tensors (K and V) x layers x kv heads x head dim x element size."""
    return 2 * n_layers * n_kv_heads * head_dim * int(DTYPE_BYTES[dtype])


def plan(
    n_layers: int,
    n_kv_heads: int,
    hidden_size: int,
    n_heads: int,
    context: int,
    batch: int,
    dtype: str = "bfloat16",
) -> KVBudget:
    # head_dim is conventionally hidden_size / n_heads, not a separate config field.
    head_dim = hidden_size // n_heads
    per_token = kv_cache_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype)
    per_seq = per_token * context / (1024**3)
    return KVBudget(per_token, round(per_seq, 3), round(per_seq * batch, 3))


if __name__ == "__main__":
    # Illustrative configs; read the real values from each model's config.json.
    configs = {
        "8B, GQA 8 kv heads, 32 layers": dict(
            n_layers=32, n_kv_heads=8, hidden_size=4096, n_heads=32),
        "8B, no GQA (32 kv heads)": dict(
            n_layers=32, n_kv_heads=32, hidden_size=4096, n_heads=32),
        "70B, GQA 8 kv heads, 80 layers": dict(
            n_layers=80, n_kv_heads=8, hidden_size=8192, n_heads=64),
    }
    for name, cfg in configs.items():
        b = plan(context=8192, batch=8, dtype="bfloat16", **cfg)
        print(f"{name:34s} {b.bytes_per_token/1024:7.0f} KiB/token  "
              f"{b.per_sequence_gib:6.2f} GiB/seq  {b.batch_gib:7.2f} GiB @ batch 8")
```

You should see the no-GQA variant land at four times the per-token cost of the GQA variant,
and the 70B config come out at 320 KiB per token — a full 8k sequence costing 2.5 GiB
on its own.

## Real use case

A support-automation startup served an 8B model on a single 24 GB GPU and sized the machine
by weights: 16 GB of fp16 weights "fits in 24 GB", so they expected room to spare. In
production the server held only two or three concurrent chats before it started evicting
requests, and long conversations failed outright.

The cause was the term they had not computed. At 8k context with 8 KV heads the cache is
128 KiB per token: 1 GiB per conversation, and their p95 conversation was 6,000 tokens, so
about 0.73 GiB each. Weights 16 GB plus overhead left roughly 6 GB usable — about eight
conversations in theory, three in practice once fragmentation and peak activation memory
were counted. They had planned for weights and forgotten the cache.

Two changes fixed it without new hardware. They exported an int4 weight quant, cutting
weights to about 4.5 GB, and kept the cache in fp16 — which moved the constraint entirely
onto the cache, where they could see it. Then they capped generation length and started
reusing a shared prefix. Concurrency went up roughly threefold, and the team could predict
concurrency from the formula instead of discovering it in an incident.

## Common mistakes

- **Mistake:** using `n_heads` instead of `n_kv_heads` in the formula — **Instead:** read `num_key_value_heads` from the config; on GQA models it is much smaller.
- **Mistake:** sizing a deployment on weights alone — **Instead:** compute cache at your p95 context length times your target concurrency, then add both.
- **Mistake:** forgetting the cache is per layer — **Instead:** multiply by `n_layers`; a single layer's cache looks harmless.
- **Mistake:** assuming cache dtype must match weight dtype — **Instead:** treat them as independent dials and measure quality on each.
- **Mistake:** computing with KiB/MiB loosely and landing 7 percent off — **Instead:** use 1024-based units consistently, as the code above does.

## Checkpoint

1. A model has 40 layers, 8 KV heads, hidden size 5120, 40 attention heads, bf16 cache. How many KiB per token, and how much for a 16k-token sequence?
2. Your card has 80 GB. Weights take 40 GB and you reserve 6 GB for activations and overhead. At 128 KiB per token and 8k context, how many concurrent requests fit?
3. Extend the script to take a Hugging Face model id, download only `config.json`, and print the plan automatically. Then compare a GQA model against a multi-head model of similar size.

**Self-check:** head_dim is `5120 / 40 = 128`, so per layer per token is `2 x 8 x 128 x 2 = 4096` bytes; across 40 layers that is 160 KiB per token, and 16,384 tokens costs `16384 x 160 KiB = 2.5 GiB`.

## Go deeper

- [PagedAttention](04-pagedattention.md) — how engines allocate this cache without wasting it.
- [GPU Memory Budgeting](10-gpu-memory.md) — put weights, cache, activations and overhead in one budget.
- [Quantization](06-quantization.md) — the weight-side lever that frees room for the cache.
- [Context Windows](../06-llm-fundamentals/03-context-windows.md) — why context length is a cost decision, not just a feature.

---
[← Previous: Prefill and Decode](01-prefill-decode.md) · [Stage index](README.md) · [Next: Continuous Batching](03-continuous-batching.md)
