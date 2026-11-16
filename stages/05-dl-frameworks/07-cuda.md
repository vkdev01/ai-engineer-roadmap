# 7. CUDA, Devices and Mixed Precision

> **Stage 05 · DL Frameworks** — Getting computation onto the GPU on purpose, reading an out-of-memory error instead of guessing, and using half precision without producing NaNs.
>
> **You'll be able to:** place models and data deliberately, triage any OOM in four steps, and switch on bf16/fp16 autocast safely.
>
> **Prerequisites:** [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md) · [1. PyTorch: Tensors and Modules](01-pytorch.md)

## Why this exists

CUDA is NVIDIA's platform for running general computation on a GPU. You will never write CUDA C in this roadmap, but you will spend real time on its three visible surfaces: **device placement** (which memory a tensor lives in), **memory limits** (the OOM error), and **numeric precision** (fp32 vs fp16 vs bf16).

Each of those has a failure mode that produces a confusing error or, worse, no error at all. A model silently left on CPU trains a hundred times slower. An OOM message names a byte count that does not match your model size. fp16 without scaling turns your loss into `nan` after a few hundred steps.

The mental model: *the GPU is a separate machine with its own memory; every tensor lives on exactly one of them, and moving data between them is the thing you are trying to minimise.*

## The mental model

```text
        CPU RAM                          GPU VRAM (e.g. 16 GB)
   ┌──────────────┐                  ┌───────────────────────┐
   │ dataset      │  ── .to(cuda) ──►│ model weights         │
   │ DataLoader   │  ── per batch ──►│ activations           │
   │ (workers)    │                  │ gradients, opt state  │
   └──────────────┘                  └───────────────────────┘
     cheap, big                        fast, small, one owner
```

Rules that follow from the picture: the model and every batch must be on the *same* device; transfers are the bottleneck, so move batches, not individual samples; and VRAM is finite, so anything that grows with batch or sequence length is the first suspect when it fills.

## Core concepts

### Choosing a device, once, at the top

`torch.cuda.is_available()` is the only safe check. Write `device = "cuda" if torch.cuda.is_available() else "cpu"` and thread that single variable through the whole program. `cuda:0` and `cuda:1` select a specific GPU on a multi-GPU box; `torch.cuda.set_device(i)` changes the default. Never hardcode `"cuda"` in library code that others might run on a laptop.

### Non-blocking transfers and pinned memory

`.to(device, non_blocking=True)` combined with a `DataLoader(pin_memory=True)` lets the copy overlap with computation. It only helps if you are actually transfer-bound, which you check before optimising — measure first.

### Reading an OOM error properly

The message reports the allocation that failed, total capacity, and how much is already allocated *and* reserved. Two numbers matter: `allocated` (live tensors) and `reserved` (the caching allocator's pool). If allocated is much smaller than reserved, the pool is fragmented or holding freed blocks — that is a different problem from a genuinely too-large tensor. Then apply the four levers from [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md) in order: batch size, autocast, gradient accumulation, activation checkpointing.

### Mixed precision: bf16 by default, fp16 with care

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` runs matrix multiplications in 16-bit while keeping master weights in fp32. It roughly halves activation memory and is usually faster on modern GPUs.

bf16 has the same exponent range as fp32, so it rarely overflows — prefer it when your hardware supports it. fp16 has more mantissa bits but a narrow range, so `GradScaler` is required: it multiplies the loss by a large factor before backward, then unscales, keeping small gradients out of the subnormal range where they become zero.

### Numeric stability is not optional at 16-bit

Softmax over long sequences, layernorm, and any `exp` of a large number can overflow in fp16. That is why autocast keeps those operations in fp32 automatically — and why you should not disable autocast around them "for speed". If you see `nan` loss with fp16, the first fix is bf16 or a `GradScaler`, and the second is checking for a `log(0)` or a division by a near-zero.

## Code

A minimal, correct autocast training step with a scaler, plus the checks that tell you which precision path you are on. The shape comments show that precision changes dtype, not shape.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, torch.cuda.get_device_name(0) if device == "cuda" else "")

# bf16 needs Ampere or newer; fp16 is the fallback
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
print("amp dtype", amp_dtype)

model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 4)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
# GradScaler is a no-op for bf16 and essential for fp16
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))

x: Tensor = torch.randn(16, 256, device=device)     # (B=16, 256)
y: Tensor = torch.randint(0, 4, (16,), device=device)

opt.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
    logits = model(x)                               # (16, 4), bf16 inside
    loss = nn.functional.cross_entropy(logits.float(), y)   # reduce in fp32
scaler.scale(loss).backward()                       # scale, then backward
scaler.unscale_(opt)                                # undo scaling before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)                                    # skips the step if grads inf/nan
scaler.update()                                     # adjusts the scale factor
print(round(float(loss), 4))
```

Now the OOM triage snippet. Run it at the moment of failure and it tells you whether you are out of real memory or out of pool.

```python
def memory_report() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    gib = 1024 ** 3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
        "free_gib": round(
            (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.memory_reserved()) / gib, 2),
    }

# wrap the step that fails
try:
    logits = model(x)
except torch.cuda.OutOfMemoryError:
    print(memory_report())          # allocated << reserved => fragmentation, not size
    torch.cuda.empty_cache()        # releases unused cached blocks only
    raise
```

## Real use case

A team trains a vision model on a rented A10 and hits OOM at batch size 64, even though batch 48 ran for hours. They try `empty_cache()` in the training loop — it makes each step slower and does not fix the OOM, because the tensors are genuinely live.

The report shows `allocated` close to `reserved` and `peak` at capacity: real memory pressure, not fragmentation. They switch on bf16 autocast, which halves the activation term, and keep batch size 64. Training gets faster as well as fitting, because the GPU's tensor cores prefer 16-bit inputs.

A second problem appears: occasional `nan` losses. bf16 was not the cause — a log-probability computed on a zero-probability prediction was. The fix is a small epsilon in the log and gradient clipping. Without the memory report, they would have spent the day tuning batch size and never learned that the real bug was numeric.

## Common mistakes

- **Mistake:** hardcoding `"cuda"` in a library. — **Instead:** accept a `device` argument with a CPU-safe default.
- **Mistake:** calling `empty_cache()` inside the training loop. — **Instead:** use it only as triage after an OOM; it slows steady-state training.
- **Mistake:** fp16 without `GradScaler`. — **Instead:** add the scaler, or switch to bf16 where supported.
- **Mistake:** computing the loss outside the autocast block. — **Instead:** keep the loss inside, and cast logits to fp32 if you sum or exponentiate them.
- **Mistake:** reading `total_memory` as "what I can use". — **Instead:** subtract what is reserved and remember other processes on the same GPU.

## Checkpoint

1. Print `allocated`, `reserved` and `peak` before and after one training step, and explain each change.
2. Train the same model for 50 steps with and without autocast and compare peak memory. Then explain why the speed difference depends on the hardware.
3. Deliberately trigger an OOM with a huge batch, catch it, print the report, and recover by halving the batch — all inside one script.

**Self-check:** `allocated` rises with weights and gradients; `reserved` rises in coarser chunks because the caching allocator grabs blocks from the driver; `peak` only ever increases until you reset it with `torch.cuda.reset_peak_memory_stats()`.

## Go deeper

- [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md) — the full memory budget, including optimizer state.
- [6. Quantization](../16-inference/06-quantization.md) — going below 16 bits.
- [10. GPU Memory](../16-inference/10-gpu-memory.md) — the inference-time version of this accounting.
- [8. Ship: A Reusable Training Loop](08-ship-training-loop.md) — device and precision as config fields.

---
[← Previous: Hugging Face Transformers and Datasets](06-hugging-face.md) · [Stage index](README.md) · [Next: Ship: A Reusable Training Loop](08-ship-training-loop.md)
