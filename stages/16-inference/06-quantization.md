# 6. Quantization

> **Stage 16 · Inference** — storing weights in fewer bits so they move faster and fit smaller cards.
>
> **You'll be able to:** compute weight memory for any bit width, choose between GPTQ, AWQ, GGUF and fp8, and measure the quality you gave up.
>
> **Prerequisites:** [The KV Cache](02-kv-cache.md) · [Open vs Closed Models](../06-llm-fundamentals/08-open-vs-closed.md)

## Why this exists

Decode reads every weight once per token. If weights occupy 16 GB, each token costs 16 GB of
memory traffic; if they occupy 4.5 GB, each token costs 4.5 GB. On a bandwidth-bound step, that
is close to a direct speed multiplier — and it also decides whether the model fits on the card
you can afford.

Quantization is the practice of storing weights (and sometimes activations or the KV cache) in
lower precision: fp16 or bf16 at 16 bits, int8 or fp8 at 8 bits, int4 at 4 bits. The mental
model: **quantization trades a small amount of numerical fidelity for a large amount of
bandwidth and capacity.**

## The mental model

```text
8B parameters:
  fp16 / bf16   8e9 x 16 bits / 8 = 16.0 GB
  int8 / fp8    8e9 x  8 bits / 8 =  8.0 GB
  int4          8e9 x  4 bits / 8 =  4.0 GB   (+ scale overhead, see below)

decode ceiling at 2 TB/s of bandwidth:
  16.0 GB -> 8.0 ms/token      8.0 GB -> 4.0 ms/token      4.25 GB -> 2.1 ms/token
```

| Format | Bits/weight | Where you see it |
|---|---|---|
| bf16 / fp16 | 16 | default for training and unquantized serving |
| fp8 | 8 | datacentre GPUs with native fp8 kernels |
| int8 | 8 | older CPU and GPU paths, still useful on edge |
| int4 group-wise | ~4.25 | GPTQ, AWQ, GGUF Q4 — the common serving choice |
| NF4 | ~4 | QLoRA fine-tuning base weights |

## Core concepts

### Group-wise quantization has real overhead

You cannot store a 4-bit integer without knowing the scale it should be multiplied by. The
usual scheme is groups of 128 weights sharing one fp16 scale, or a scale and a zero-point for
asymmetric ranges.

```text
symmetric,  128-weight group: 4 bits + 16 bits/128 = 4.125 bits/weight
asymmetric, 128-weight group: 4 bits + 32 bits/128 = 4.25  bits/weight
```

So an "int4" 8B model is not 4.0 GB. At 4.25 bits per weight it is
`8e9 x 4.25 / 8 = 4.25 GB`. Small groups improve quality and increase overhead; group size 128
is a common default, and smaller groups (64, 32) help the smallest models most.

### What quality loss actually looks like

Quantization rarely produces obvious garbage at 8 bits, and at 4 bits the failure is subtle
and task-dependent. Watch for these patterns rather than a single average score:

- **Long-tail knowledge degrades first** — rare entities, exact numbers, unusual names.
- **Instruction following slips slightly** — the model ignores a constraint it used to respect.
- **Structured output breaks** — a JSON schema violated on a fraction of calls.
- **Reasoning chains shorten** — the model stops verifying its own work.
- **Repeat and loop behaviour appears** at aggressive settings, especially with small models.

The practical rule: a 4-bit version of a large model is usually better than an fp16 version of
a small model at the same memory, but you must confirm that on *your* task. A 7B model at 4
bits is noticeably more fragile than a 70B model at 4 bits, because there is less redundancy
per weight to absorb the error.

### The main formats and where they come from

- **GPTQ** — post-training, layer-by-layer, uses a calibration set of a few hundred samples.
  Weights-only, GPU-oriented, mature and widely supported.
- **AWQ** — activation-aware: it protects the small fraction of weights that matter most for
  large activations. Often a good default for 4-bit GPU serving.
- **GGUF** — the llama.cpp format, a family of block-quantized types (Q4_K_M and friends) with
  mixed precision per tensor. The standard for CPU, Apple Silicon and local runners.
- **fp8** — 8-bit float with a wider dynamic range than int8; needs hardware support and is
  usually near-lossless for inference.
- **NF4 / bitsandbytes** — a 4-bit normal-float format used mainly for QLoRA fine-tuning, not
  for high-throughput serving.

Names and defaults change; check the current documentation of the tool you use for exact
quant types and calibration recipes.

### KV cache quantization is a separate decision

As [The KV Cache](02-kv-cache.md) shows, the cache often exceeds the weights at long context.
Quantizing the cache from 2 bytes to 1 byte halves that term. Quality effects are usually
milder than weight quantization because the cache is per-conversation, but verify it on long
inputs specifically — errors accumulate over the whole context.

## Code

First, a memory calculator that makes the overhead explicit, so your capacity plan does not
come out a gigabyte short.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantPlan:
    bits_per_weight: float
    weight_gib: float
    ms_per_token: float | None


def plan(params: int, bits: int, group_size: int | None = None,
         bandwidth_gbs: float | None = None) -> QuantPlan:
    bpw = float(bits)
    if group_size and bits < 16:
        # fp16 scale (16 bits) + fp16 zero point (16 bits) per group, for int4-style schemes
        bpw += 32.0 / group_size
    gib = params * bpw / 8 / (1024**3)
    ms = (gib * 1024**3 / (bandwidth_gbs * 1e9) * 1000) if bandwidth_gbs else None
    return QuantPlan(round(bpw, 3), round(gib, 2), round(ms, 2) if ms else None)


if __name__ == "__main__":
    bw = 2000.0  # GB/s, illustrative; use your GPU's spec
    print(f"{'config':22s} {'bits/wt':>8s} {'GiB':>7s} {'ms/tok':>7s}")
    for label, bits, group in [
        ("8B bf16", 16, None),
        ("8B int8", 8, 128),
        ("8B int4 g128", 4, 128),
        ("8B int4 g64", 4, 64),
        ("70B bf16", 16, None),
        ("70B int4 g128", 4, 128),
    ]:
        params = 70_000_000_000 if label.startswith("70B") else 8_000_000_000
        p = plan(params, bits, group, bw)
        print(f"{label:22s} {p.bits_per_weight:8.3f} {p.weight_gib:7.2f} {p.ms_per_token:7.2f}")
```

You should see int4 with group size 64 cost slightly more memory than group size 128 — the
price of better fidelity — and the 70B int4 row land near 35 GiB, which is the difference
between needing two cards and one.

Second, the part most teams skip: measuring what the quant broke. This compares a quantized
model against its fp16 baseline on a small golden set and reports agreement, using the pattern
from [Golden Datasets](../15-evals/03-golden-datasets.md).

```python
from __future__ import annotations

import json
from pathlib import Path


def agreement(golden_path: Path, baseline_fn, candidate_fn) -> dict[str, float]:
    """Exact-match and JSON-schema agreement between two generators."""
    items = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    exact = schema_ok = 0
    for item in items:
        a = baseline_fn(item["prompt"])
        b = candidate_fn(item["prompt"])
        exact += int(a.strip() == b.strip())
        try:
            json.loads(b)
            schema_ok += 1
        except json.JSONDecodeError:
            pass
    n = max(len(items), 1)
    return {"n": len(items), "exact_match": exact / n, "valid_json": schema_ok / n}
```

Run it against your baseline before you ship. If `valid_json` drops, the quant is not
acceptable for a structured-output endpoint no matter how fast it is.

## Real use case

A document-extraction product ran a 70B model in bf16 across two 80 GB GPUs. Serving cost was
the dominant line in the budget, and a second GPU per replica also doubled the failure
surface: any single-card fault took a replica down.

They moved to a 4-bit AWQ export with group size 128. Weight memory fell from about 130 GiB
to about 35 GiB, which fit on one 80 GB card with room for the KV cache. Decode speed rose
because each token now moved a quarter of the bytes, and the team measured a large drop in
cost per thousand documents.

Quality was the risk, so they gated the change on their own golden set of 400 invoices. Two
regressions showed up that an average score would have hidden: date formats occasionally
flipped to a locale-specific form, and one long tail of vendor names lost exact spelling.
They fixed both in post-processing with a schema validator and a lookup against the vendor
master list. Had they only tracked an average score, they would have shipped a model that
looked fine and failed on real invoices.

## Common mistakes

- **Mistake:** computing int4 as exactly 4 bits per weight — **Instead:** add scale and zero-point overhead; group size 128 makes it about 4.25.
- **Mistake:** judging a quant by one average benchmark score — **Instead:** check exact-match, schema validity and long-tail entity accuracy separately.
- **Mistake:** quantizing the KV cache without testing long contexts — **Instead:** evaluate at your p95 context length; errors compound over tokens.
- **Mistake:** using a 4-bit quant of a small model where a bf16 larger model fits — **Instead:** compare total memory, not bits; a bigger model at lower precision often wins.
- **Mistake:** assuming a quantized file is portable between engines — **Instead:** GPTQ, AWQ and GGUF are different formats; export per engine.

## Checkpoint

1. A 13B model is exported at 4 bits with group size 128 and asymmetric scales. What is its weight memory, in GB?
2. Your card has 48 GB, weights must fit in 60 percent of it, and you need at least 8k context. Which bit width can you use for an 8B model at 128 KiB per token of cache?
3. Build the agreement harness above, run it on 100 prompts with an fp16 model and its 4-bit export, and report exact-match, valid-JSON rate and the three worst disagreements by category.

**Self-check:** `4 + 32/128 = 4.25` bits per weight, so `13e9 x 4.25 / 8 = 6.9 GB` — noticeably more than the 6.5 GB the naive 4-bit figure suggests.

## Go deeper

- [Distillation](07-distillation.md) — the other way to make a model cheaper: train a smaller one.
- [GPU Memory Budgeting](10-gpu-memory.md) — place these weights in a full card budget.
- [Finetune vs RAG](../13-fine-tuning/07-finetune-vs-rag.md) — before you reach for quantization, check you need a model at all.
- [QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) — where 4-bit base weights are used for training rather than serving.

---
[← Previous: Speculative Decoding](05-speculative-decoding.md) · [Stage index](README.md) · [Next: Distillation](07-distillation.md)
