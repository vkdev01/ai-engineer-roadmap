# 4. GPU Memory and Checkpoints

> **Stage 05 · DL Frameworks** — Why your 8 GB card reports 8 GB used by a 2 GB model, and how to save and resume training without losing a week of compute.
>
> **You'll be able to:** account for where GPU memory goes, shrink it with a deliberate lever, and write a checkpoint that resumes exactly where it left off.
>
> **Prerequisites:** [3. Anatomy of a Training Loop](03-training-loops.md) · [7. CUDA, Devices and Mixed Precision](07-cuda.md)

## Why this exists

Two failures end more training runs than anything else. The first is `CUDA out of memory` — which almost never means "the model is too big", it means one of five specific things is bigger than you think. The second is a checkpoint that loads but does not *resume*: the loss curve restarts, or worse, silently jumps, because you saved the weights and forgot the optimizer state.

Neither is hard once you can count. This file gives you the count.

The mental model: *GPU memory = weights + gradients + optimizer state + activations, and only the last one depends on batch size.*

## The mental model

For a model with `P` parameters trained with Adam in fp32:

| What | Size | For P = 1e9 (1 B params) |
|------|------|--------------------------|
| Weights | `4P` bytes | 4 GB |
| Gradients | `4P` bytes | 4 GB |
| Adam moments (`m`, `v`) | `8P` bytes | 8 GB |
| **Subtotal, batch-independent** | `16P` bytes | **16 GB** |
| Activations | grows with batch × sequence length | often the largest term |

That table is why "the model is only 4 GB" is not a memory plan: the training footprint of a 1 B parameter model in fp32 with Adam is roughly 16 GB before a single activation is stored. Mixed precision and 8-bit optimizers attack this table; batch size and checkpointing of activations attack the last row.

## Core concepts

### Activations dominate, and they scale with batch

Every tensor a forward pass produces that `backward` will need must be kept alive. For a transformer, that is roughly proportional to `batch × sequence_length × hidden_size × layers`. Doubling sequence length roughly doubles attention memory; doubling batch doubles everything. This is why OOM usually appears when you increase batch size or context length, not when you load the model.

### The five levers, cheapest first

1. **Lower batch size** — free, no code change, always try first.
2. **Mixed precision (`autocast`)** — halves activation and weight memory for the parts that run in bf16, and is usually faster. See [7. CUDA, Devices and Mixed Precision](07-cuda.md).
3. **Gradient accumulation** — get a large effective batch with a small real one; memory stays at the small size.
4. **Gradient checkpointing** (`model.gradient_checkpointing_enable()`) — recompute activations in the backward pass instead of storing them. Trades roughly 20-30% more compute for a large memory saving.
5. **Shrink the model** — quantisation, LoRA, or a smaller base. The last resort, and the right one when the first four fail.

### `state_dict` is the only portable format

`torch.save(model, path)` pickles the class, which breaks the moment you refactor the module or move the code. `torch.save(model.state_dict(), path)` saves tensors keyed by name and loads into any object with the same architecture. Same for the optimizer: `optimizer.state_dict()` holds the Adam moments and step counts. Forgetting it is the classic "resume" bug — the model restarts with fresh momentum and the loss jumps.

### A checkpoint is a dict, not a file of weights

A resumable checkpoint carries: model state, optimizer state, scheduler state, the epoch and global step, the best metric so far, and the config or a hash of it. Save the RNG state too if you care about exact reproducibility. Everything you need to answer "what was I doing" belongs in it.

### Saving safely: write, then rename

A checkpoint written directly to its final name is corrupt if the process dies mid-write — and long jobs do die. Write to `path.tmp`, `flush`, then `os.replace`, which is atomic on POSIX. Keep the last N checkpoints and delete older ones so a long run cannot fill the disk.

## Code

This snippet reports the real memory footprint of a model in training, which is the count from the table above measured rather than guessed.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(4096, 4096), nn.GELU(), nn.Linear(4096, 4096)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

def gib(n: int) -> float:
    return n / 1024 ** 3

params = sum(p.numel() for p in model.parameters())
print("params", params, "weights GiB", round(gib(params * 4), 3))
# gradients: same size as weights, allocated on the first backward
x: Tensor = torch.randn(8, 4096, device=device)     # (B=8, 4096)
loss = model(x).sum()
loss.backward()
print("grad GiB", round(gib(params * 4), 3))
# Adam state exists only after the first step
opt.step()
state_bytes = sum(v.numel() * v.element_size() for s in opt.state.values()
                  for v in s.values() if torch.is_tensor(v))
print("optimizer GiB", round(gib(state_bytes), 3))
if device == "cuda":
    print("reserved GiB", round(gib(torch.cuda.memory_reserved()), 3))
```

Now a checkpoint that actually resumes. The `save` half is five lines; the `load` half is the part people get wrong.

```python
from pathlib import Path
import os

def save_ckpt(path: Path, model, opt, sched, epoch: int, step: int, best: float) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),          # Adam moments + step count
        "scheduler": sched.state_dict(),        # where the LR curve is
        "epoch": epoch, "global_step": step, "best_metric": best,
        "rng": torch.get_rng_state(),           # reproducibility of shuffling
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)                       # atomic: never a half-written file

def load_ckpt(path: Path, model, opt, sched) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])        # shapes must match exactly
    opt.load_state_dict(ckpt["optimizer"])      # skip this and loss jumps
    sched.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["rng"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_metric"]
```

## Real use case

A two-person team rents a single GPU by the hour to fine-tune a 7 B model. Their first run dies at hour nine of an estimated twelve with `CUDA out of memory`, on a batch they had already run successfully for eight hours.

The trigger is a long example: activations scale with sequence length, and one batch contained a document long enough to blow past the remaining headroom. The fix is not a smaller batch everywhere — it is `model.gradient_checkpointing_enable()` plus truncating inputs to a fixed maximum length, both decided before the run starts and recorded in the config.

The second run hits an unrelated problem: the machine is preempted, and the resume loads weights only. Loss restarts at a high value and takes hours to recover, because the Adam moments were discarded. Once the checkpoint dict includes the optimizer state, resume is invisible in the loss curve — which is exactly the test in [8. Ship: A Reusable Training Loop](08-ship-training-loop.md).

## Common mistakes

- **Mistake:** `torch.save(model, path)`. — **Instead:** save `state_dict()`; it survives refactors and loads across code versions.
- **Mistake:** saving weights only and calling it a resume. — **Instead:** include optimizer, scheduler, step and epoch; otherwise the run restarts in disguise.
- **Mistake:** calling `torch.cuda.empty_cache()` to fix OOM. — **Instead:** it only releases unused cached blocks; find the tensor that is actually too big.
- **Mistake:** writing checkpoints straight to their final filename. — **Instead:** write to `.tmp` and `os.replace` so a crash cannot corrupt the file.
- **Mistake:** loading a checkpoint with `map_location` omitted on a CPU-only machine. — **Instead:** pass `map_location="cpu"`, then move the model to the device you want.

## Checkpoint

1. For the two-layer model above, compute the expected total training memory for 1 M parameters with Adam in fp32, and confirm against the printed numbers.
2. Add gradient checkpointing to a model with at least four layers and measure the memory difference on one batch.
3. Make the checkpoint keep only the three most recent files, and prove it by saving ten and listing the directory.

**Self-check:** `16P` bytes — 4P weights, 4P gradients, 8P for two Adam moments — so 1 M parameters is about 16 MB before activations, matching the three printed figures summed.

## Go deeper

- [7. CUDA, Devices and Mixed Precision](07-cuda.md) — the memory levers in detail.
- [8. Ship: A Reusable Training Loop](08-ship-training-loop.md) — checkpointing wired into a real trainer.
- [10. GPU Memory](../16-inference/10-gpu-memory.md) — the same accounting at inference time, with a KV cache.
- [2. LoRA, QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) — shrinking the weights term itself.

---
[← Previous: Anatomy of a Training Loop](03-training-loops.md) · [Stage index](README.md) · [Next: Keras and the High-Level Path](05-keras.md)
