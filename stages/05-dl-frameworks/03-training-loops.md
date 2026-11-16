# 3. Anatomy of a Training Loop

> **Stage 05 · DL Frameworks** — The seven lines that must happen in the right order, and what breaks in each when they do not.
>
> **You'll be able to:** write a training loop from memory, add validation without leaking gradients, and use a learning-rate scheduler correctly.
>
> **Prerequisites:** [2. Autograd From Scratch](02-autograd.md) · [3. Optimizers](../04-deep-learning/03-optimizers.md)

## Why this exists

A training loop is short enough to memorise and subtle enough to be wrong in four different ways that all look like "the model doesn't learn". Nothing crashes. Loss goes down a bit, then plateaus, or oscillates, or the validation score is suspiciously good.

The reason is that the loop mixes two modes — training and evaluation — with completely different requirements, and the switch is a single method call that people forget. This file is the reference version you will copy into every project from here on.

The mental model: *one step is forward, loss, clear, backward, step, repeat — and the mode you are in decides whether the graph is built and whether dropout and batch-norm behave normally.*

## The mental model

```text
for each epoch
  └─ model.train()                        ← dropout ON, batchnorm uses batch stats
     for each batch
        optimizer.zero_grad(set_to_none=True)   ← 1. clear old grads
        logits = model(x)                       ← 2. forward (graph recorded)
        loss = criterion(logits, y)             ← 3. scalar loss
        loss.backward()                         ← 4. fill .grad for all params
        optimizer.step()                        ← 5. update weights
     scheduler.step()                           ← 6. once per epoch, not per batch
  └─ model.eval(); with torch.no_grad():   ← 7. validate with dropout OFF
        measure, never update
```

The order of steps 1 and 4 is the one people get wrong: gradients accumulate, so clearing must happen before the backward pass of the *next* step, not after the step of the previous one. Both placements "work" only if you never forget — which you will.

## Core concepts

### `model.train()` and `model.eval()` change layer behaviour, not gradients

Two things respond to the flag: dropout (drops units only in train mode) and batch normalisation (uses batch statistics in train mode, running buffers in eval mode). If you validate without `eval()`, your validation loss is computed with dropout active — it looks worse than reality and, worse, is noisy enough to hide real regressions. `eval()` does *not* stop gradients; `torch.no_grad()` does. You want both.

### `zero_grad(set_to_none=True)` is the modern default

Clearing to `None` rather than zero lets the backward pass allocate fresh gradient buffers, which is marginally faster and makes it easy to tell "no gradient ever arrived" from "gradient is exactly zero" — a distinction that matters when debugging frozen layers.

### Loss must be a scalar, and it must be the mean

`backward()` needs a single number. If you accidentally keep per-sample losses and sum them, your effective learning rate scales with batch size — so changing batch size silently changes training dynamics. Use the criterion's default reduction (`mean`) and know that `CrossEntropyLoss` expects raw logits, not softmax outputs. Applying softmax first is the most common cause of a model that trains but converges slowly and poorly.

### Schedulers: step where the docs say, not where it feels right

Most PyTorch schedulers (for example `CosineAnnealingLR`, `StepLR`) are designed to be stepped **once per epoch**, after the optimizer. `ReduceLROnPlateau` is different: it takes the validation metric and steps per epoch as well. Stepping per batch with a per-epoch scheduler decays the learning rate thousands of times too fast, and the run dies quietly. Read the scheduler's docs and note whether its `step()` takes an argument.

### Gradient clipping belongs between backward and step

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` rescales gradients so their total norm is bounded. It is standard for transformers and RNNs, where occasional large gradients destabilise a run. It must be called after `backward()` (gradients exist) and before `step()` (they are used).

### Gradient accumulation simulates a bigger batch

If memory limits you to 8 samples but you want the dynamics of 32, run four forward/backward passes and call `optimizer.step()` only every fourth. Divide the loss by the accumulation count so the scale stays the same, and remember to clear gradients only at the start of each real step.

## Code

A complete, correct loop. It trains a small classifier on random data so it runs anywhere; swap the loader for your own and nothing else changes.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 512 examples, 20 features, 3 classes
ds = TensorDataset(torch.randn(512, 20), torch.randint(0, 3, (512,)))
train_dl = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 3))
model.to(device)
criterion = nn.CrossEntropyLoss()                       # expects raw logits
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)


def evaluate() -> float:
    model.eval()                                        # dropout off, BN uses buffers
    total, correct = 0, 0
    with torch.no_grad():                               # no graph, less memory
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred: Tensor = model(xb).argmax(dim=-1)     # (B,) from (B, 3)
            correct += int((pred == yb).sum())
            total += yb.numel()
    return correct / total


for epoch in range(5):
    model.train()                                       # dropout on
    running = 0.0
    for step, (xb, yb) in enumerate(train_dl):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)           # 1
        logits: Tensor = model(xb)                      # 2  (32, 3)
        loss: Tensor = criterion(logits, yb)            # 3  scalar
        loss.backward()                                 # 4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                                # 5
        running += float(loss.detach())
    scheduler.step()                                    # 6  once per epoch
    acc = evaluate()                                    # 7
    print(f"epoch {epoch} loss {running / len(train_dl):.4f} acc {acc:.3f} "
          f"lr {scheduler.get_last_lr()[0]:.2e}")
```

Here is the same step with gradient accumulation, which is what you reach for when the batch you want does not fit in memory.

```python
accum_steps = 4                       # effective batch = 32 * 4 = 128
model.train()
optimizer.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(train_dl):
    xb, yb = xb.to(device), yb.to(device)
    loss = criterion(model(xb), yb) / accum_steps     # keep the scale constant
    loss.backward()                                   # grads add up across micro-batches
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)         # clear only at real steps
```

## Real use case

A team fine-tunes a model for classifying support emails. Their training loss falls smoothly to near zero while validation accuracy stays at the majority-class baseline. They conclude the data is unlearnable and start relabelling — expensive, and wrong.

The loop is the problem. They call `model.train()` once, before the epoch loop, and never call `model.eval()` around validation. With dropout at 0.3 active during validation, predictions are noisy enough to collapse to the majority class, while the training number looks healthy because it is measured in the same noisy mode.

The fix is two lines: `model.eval()` and `torch.no_grad()` around the evaluation function, restored with `model.train()` afterwards. Accuracy jumps to a level that actually reflects the model. The lesson they keep: a training loop bug is a *measurement* bug until proven otherwise — check what mode you measured in before you blame the data.

## Common mistakes

- **Mistake:** validating without `model.eval()`. — **Instead:** wrap every evaluation in `eval()` plus `no_grad()`, and switch back with `train()`.
- **Mistake:** applying `softmax` before `CrossEntropyLoss`. — **Instead:** pass raw logits; the loss does the log-softmax internally.
- **Mistake:** stepping a per-epoch scheduler per batch. — **Instead:** `scheduler.step()` once per epoch, after the batch loop.
- **Mistake:** logging `loss` without `.item()` or `.detach()`. — **Instead:** accumulate `float(loss.detach())` so the graph is not held alive.
- **Mistake:** changing batch size and wondering why the learning rate "changed". — **Instead:** with summed losses it did; use mean reduction or scale the learning rate deliberately.

## Checkpoint

1. Add gradient clipping and print the gradient norm before clipping for one epoch. What does the distribution tell you about whether clipping is doing anything?
2. Add early stopping on validation loss with a patience of three epochs, restoring the best weights. Which part of the loop do you need to save?
3. Convert the loop to accumulate over four micro-batches and verify that the weights after one real step match a loop with batch size 128 — they should be close, not identical, because of dropout.

**Self-check:** The clipped norms should mostly sit below your `max_norm`; if nearly all batches are clipped you are cutting real signal, and the fix is a lower learning rate or warmup rather than a bigger threshold.

## Go deeper

- [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md) — save the state this loop produces, and resume it.
- [8. Ship: A Reusable Training Loop](08-ship-training-loop.md) — turn this into a package with config and resume.
- [5. Loss Functions](../02-math-and-ml-basics/05-loss-functions.md) — why cross-entropy expects logits.
- [1. SFT](../13-fine-tuning/01-sft.md) — the same loop, applied to a language model.

---
[← Previous: Autograd From Scratch](02-autograd.md) · [Stage index](README.md) · [Next: GPU Memory and Checkpoints](04-gpu-checkpoints.md)
