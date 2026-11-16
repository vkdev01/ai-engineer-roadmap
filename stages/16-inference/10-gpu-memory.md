# 10. GPU Memory Budgeting

> **Stage 16 · Inference** — adding up weights, cache, activations and overhead before you rent the card.
>
> **You'll be able to:** produce a memory budget for a model and card that predicts peak usage within a small margin.
>
> **Prerequisites:** [The KV Cache](02-kv-cache.md) · [Quantization](06-quantization.md)

## Why this exists

Every failed deployment starts with the same sentence: "the model fits, the weights are only
16 GB on a 24 GB card." Weights are one line of four. The other three are the KV cache,
activation peaks during prefill, and the framework's own overhead. Underestimating any of them
turns a capacity plan into an out-of-memory crash under load, usually on a long prompt, usually
at the worst time.

The mental model: **the card is a fixed budget, and concurrency is what is left after
everything else is paid for.**

## The mental model

```text
+--------------------------------------------------+  card capacity, e.g. 24 GiB
|  weights            (from bit-width)             |  4.25 GiB  (8B int4)
+--------------------------------------------------+
|  KV cache           (concurrency x tokens x 128K)|  budget = whatever remains
+--------------------------------------------------+
|  activation peaks   (grows with prompt length)   |  1-3 GiB, transient
+--------------------------------------------------+
|  framework overhead (CUDA context, kernels,      |  1-2 GiB, fixed
|  fragmentation, allocator slack)                 |
+--------------------------------------------------+
```

| Line | Depends on | Scales with |
|---|---|---|
| Weights | parameter count, bit width | nothing at runtime |
| KV cache | layers, kv heads, head_dim, dtype | concurrency x context |
| Activations | batch, prompt length, hidden size | peak, not average |
| Overhead | engine, CUDA, allocator | roughly fixed per process |

## Core concepts

### Weights first, and be honest about the units

`weights_GiB = params x bits_per_weight / 8 / 1024^3`. An 8B model at 4.25 effective bits is
`8e9 x 4.25 / 8 = 4.25e9 bytes = 3.96 GiB`. Note the gap between GB and GiB: card capacity is
quoted in GiB by `nvidia-smi` (a "24 GB" card shows about 24,564 MiB), while model sizes are
often quoted in decimal GB. Mixing the two is a routine source of 7 percent errors that matter
when you are near the edge.

### Activations are a peak, not a steady state

During prefill, the model holds intermediate tensors for every token in the prompt at once.
The largest of these scales roughly with `batch x prompt_tokens x hidden_size x bytes`. For an
8k prompt and a hidden size of 4,096 in bf16, one such tensor alone is
`8192 x 4096 x 2 = 64 MiB`; the network holds several, plus attention score buffers, so a peak
of a couple of gigabytes on a long prompt is normal.

This is why engines profile memory at startup with a maximum-length dummy request, and why a
server that runs fine at 2k prompts can crash at 16k with the same weights and the same
concurrency.

### Overhead is the line people forget

The CUDA context, the engine's own kernels, the communication buffers for tensor parallelism,
and the allocator's fragmentation all take space. A fixed 1 to 2 GiB per process is a
reasonable planning figure for a single-GPU deployment, and more with tensor parallelism
because each rank holds its own context.

### Solve for concurrency, not for "does it fit"

The useful question is not "does the model fit" but "how many concurrent requests at my p95
context length". Rearranged:

```text
free_GiB      = capacity x util_fraction - weights - activations - overhead
max_tokens    = free_GiB x 1024^3 / bytes_per_token
concurrency   = max_tokens / p95_context_tokens
```

Worked example on a 24 GiB card, util 0.92, 8B int4 (3.96 GiB), 1.5 GiB activations,
1.5 GiB overhead, 128 KiB per token, p95 context 3,000 tokens:

```text
free      = 24 x 0.92 - 3.96 - 1.5 - 1.5 = 22.08 - 6.96 = 15.12 GiB
max_tokens= 15.12 x 1,073,741,824 / 131,072 = 123,863 tokens
concurrency = 123,863 / 3,000 = 41 requests
```

That is a prediction you can check against the engine's own reported cache capacity. If the
engine says 100,000 tokens instead of 123,000, your overhead estimate was low — find the
missing gigabyte before you promise a concurrency number to anyone.

## Code

This solves the budget in both directions: given a card, report concurrency; given a
concurrency target, report the card you need. It also flags the unit traps explicitly.

```python
from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class Budget:
    capacity_gib: float
    util: float
    weights_gib: float
    activations_gib: float
    overhead_gib: float
    kv_bytes_per_token: int
    context_tokens: int

    @property
    def free_gib(self) -> float:
        return (self.capacity_gib * self.util - self.weights_gib
                - self.activations_gib - self.overhead_gib)

    @property
    def max_tokens(self) -> int:
        return max(0, int(self.free_gib * GIB // self.kv_bytes_per_token))

    @property
    def concurrency(self) -> int:
        return self.max_tokens // max(self.context_tokens, 1)

    def report(self, label: str) -> str:
        return (f"{label}: free {self.free_gib:6.2f} GiB, "
                f"cache {self.max_tokens:>8d} tokens, "
                f"concurrency {self.concurrency:3d} @ {self.context_tokens} tok")


def weights_gib(params: int, bits: float, group_size: int | None = 128) -> float:
    bpw = bits + (32.0 / group_size if group_size and bits < 16 else 0.0)
    return params * bpw / 8 / GIB


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    return 2 * layers * kv_heads * head_dim * dtype_bytes


if __name__ == "__main__":
    # 8B-class config: 32 layers, 8 kv heads, head_dim 128 -> 128 KiB per token
    per_token = kv_bytes_per_token(32, 8, 128)
    print(f"KV per token: {per_token / 1024:.0f} KiB")

    for label, cap, params, bits in [
        ("24 GiB card, 8B int4", 24.0, 8_000_000_000, 4),
        ("48 GiB card, 8B bf16", 48.0, 8_000_000_000, 16),
        ("80 GiB card, 70B int4", 80.0, 70_000_000_000, 4),
    ]:
        b = Budget(cap, 0.92, weights_gib(params, bits), 1.5, 1.5, per_token, 3000)
        print(b.report(label))

    # Reverse direction: what card do I need for 64 concurrent 8k requests?
    need_tokens = 64 * 8192
    needed = (weights_gib(8_000_000_000, 4) + 1.5 + 1.5
              + need_tokens * per_token / GIB) / 0.92
    print(f"capacity needed for 64 x 8k: {needed:.1f} GiB")
```

You should see the 24 GiB row land near 41 concurrent 3k-token requests, and the reverse
calculation ask for about 77 GiB — 64 sequences of 8k tokens is 524,288 tokens of cache, which
alone is 64 GiB, so the answer is an 80 GiB card, not a 48 GiB one.

## Real use case

An internal knowledge assistant was specced on a 24 GiB card using an 8B model at bf16. The
engineer checked the obvious thing: 16 GiB of weights on a 24 GiB card, so 8 GiB spare. The
deployment served one or two users fine and fell over in a team demo of eight people asking
questions at once, with an out-of-memory error that pointed at KV cache allocation.

Running the budget showed why. `24 x 0.92 = 22.08 GiB`, minus 14.9 GiB of weights, minus 1.5
GiB activations, minus 1.5 GiB overhead, left 4.18 GiB of cache — about 34,000 tokens, which is
four concurrent 8k conversations, not eight, and only if nobody used a long document.

They had three options and chose the arithmetic-driven one: int4 weights dropped weights to
3.96 GiB and raised the cache budget to 15.12 GiB, roughly 41 concurrent 3k conversations. They
validated the prediction by asking the engine to report its own cache capacity in tokens and
comparing it to the calculation. The numbers agreed within a few percent, which meant the team
could size the next deployment without renting hardware to find out.

## Common mistakes

- **Mistake:** comparing decimal GB model sizes to GiB card capacity — **Instead:** convert everything to GiB and label the unit.
- **Mistake:** budgeting activations as zero because inference is "just a forward pass" — **Instead:** reserve 1 to 3 GiB; long prompts produce large transient tensors.
- **Mistake:** planning concurrency at 8k when your p95 context is 3k — **Instead:** use p95, and cap `max_model_len` so one outlier cannot take the whole pool.
- **Mistake:** running `gpu_memory_utilization` at 1.0 — **Instead:** leave 5 to 10 percent for peaks and fragmentation.
- **Mistake:** trusting the budget without checking the engine's reported cache size — **Instead:** compare, and treat a mismatch as a missing line in your model.

## Checkpoint

1. A 48 GiB card, util 0.90, 8B bf16 weights (14.9 GiB), 2 GiB activations, 2 GiB overhead, 128 KiB per token. How many concurrent 4k-token requests fit?
2. Using the script, find the smallest card that serves 32 concurrent 16k requests with an 8B int4 model. Then compute the same for bf16 and explain the gap.
3. Instrument a real deployment: log `nvidia-smi` memory every second during a load test and plot it against your predicted line. Identify which of the four lines you underestimated.

**Self-check:** `48 x 0.90 = 43.2`, minus 14.9, 2 and 2, leaves 24.3 GiB; at 131,072 bytes per token that is about 199,000 tokens, so roughly 48 concurrent 4k requests.

## Go deeper

- [Ship: A vLLM Server](11-ship-vllm-server.md) — measure peak memory against this budget on real hardware.
- [The KV Cache](02-kv-cache.md) — the term that usually decides the answer.
- [Quantization](06-quantization.md) — the lever that frees the most room.
- [GPU Cloud](../19-cloud-devops/08-gpu-cloud.md) — turning a GiB requirement into a rental decision.

---
[← Previous: Ollama and TensorRT-LLM](09-ollama-tensorrt.md) · [Stage index](README.md) · [Next: Ship: A vLLM Server](11-ship-vllm-server.md)
