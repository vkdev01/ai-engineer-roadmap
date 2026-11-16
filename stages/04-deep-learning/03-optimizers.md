# 3. Optimizers

> **Stage 04 · Deep Learning** — the part that decides how the weights actually move
>
> **You'll be able to:** implement SGD, momentum and Adam, and read a loss curve to say which one a given run needs.
>
> **Prerequisites:** [Feedforward Networks](02-feedforward.md) · [Calculus](../02-math-and-ml-basics/02-calculus.md)

## Why this exists

Backpropagation gives you a gradient: the direction of steepest increase. It does not
tell you how far to step, and for a deep network the raw gradient is a poor guide.
Loss surfaces are not bowls — they are long narrow valleys where the gradient points
across the valley rather than along it, so plain gradient descent zig-zags and takes
forever. Different features also need different step sizes, and a single global
learning rate cannot serve them all.

Optimizers are the accumulated engineering answer. Momentum remembers the direction it
has been travelling and dampens the zig-zag. Adam keeps a per-parameter step size based
on how noisy that parameter's gradient has been. Both are the same idea: **the
gradient is a suggestion, and the optimizer decides how much history to mix in.**

Mental model: **SGD follows the gradient; momentum follows a smoothed gradient; Adam
follows a per-parameter normalised gradient.**

## The mental model

```text
  plain SGD          momentum                Adam
  w -= lr * g        v = b*v + g             m = b1*m + (1-b1)*g
                     w -= lr * v             v = b2*v + (1-b2)*g^2
                                             w -= lr * m_hat / (sqrt(v_hat)+eps)

  narrow valley, top view:

  SGD zig-zags          momentum goes straight      Adam: per-axis steps
   \/\/\/\/\             ------------->              so the steep axis is
   slow across           fast across                 damped automatically
```

## Core concepts

### Plain SGD needs a good learning rate and nothing else

`w -= lr * g` is the baseline, and it is not obsolete — with a well-tuned schedule it
is often what large vision training uses. Its weakness is that one `lr` applies to all
parameters: too large for the steep directions, too small for the flat ones. It is
also sensitive to feature scaling, which is why normalised inputs matter so much.

### Momentum is an exponential moving average of gradients

`v = beta * v + g` then `w -= lr * v`, with `beta` typically around 0.9. On a
consistent slope the steps accumulate and you move faster. On a zig-zag, alternating
gradients cancel in the average. The effect is that the optimiser coasts through small
dips and noisy batches — it has inertia, which is both the benefit and, if `beta` is
too high, the reason it overshoots.

### Adam normalises per parameter

Adam tracks a first moment (mean gradient, like momentum) and a second moment (mean
squared gradient, i.e. a per-parameter scale). Dividing by the square root of the
second moment means a parameter with consistently large gradients takes smaller steps,
and a rarely-moving parameter takes bigger ones. The bias-correction terms
(`m_hat`, `v_hat`) fix the fact that both averages start at zero, which would otherwise
make the first steps far too large.

### The learning rate still matters most

Adam reduces how much you have to tune, but a bad `lr` still ruins the run. Typical
practice: a small default, a warmup for the first few hundred steps (the moments are
noisy early), then decay. If the loss spikes and recovers, the rate is a bit high. If
it goes to `nan`, it is much too high, or there is a `log(0)` in your loss.

### Read the loss curve like a diagnostic

Flat from the start: rate too small, or a broken gradient (a detached tensor is the
usual culprit). Smooth but slow: rate slightly small. Oscillating: rate too high or
batch too small. Falls then rises: overfitting. Spikes: a bad batch or a rate peak. You
can diagnose most training failures from a single plot of loss against step.

## Code

All three optimizers on the same one-dimensional problem, so the difference is the only
thing that varies. You should see momentum reach the smallest gradient norm in these 60 steps, Adam
close behind, and plain SGD lagging badly — on a toy quadratic Adam's advantage over
momentum does not show, and its real value is on noisy, badly scaled problems.

```python
from __future__ import annotations

import numpy as np

# A narrow valley: curvature 40x steeper in one direction than the other.
A = np.array([[40.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 1.0])


def grad(w: np.ndarray) -> np.ndarray:
    return A @ w - b          # gradient of 0.5 * w^T A w - b^T w


def sgd(steps: int = 60, lr: float = 0.02) -> float:
    w = np.zeros(2)
    for _ in range(steps):
        w = w - lr * grad(w)
    return float(np.linalg.norm(grad(w)))


def momentum(steps: int = 60, lr: float = 0.02, beta: float = 0.9) -> float:
    w, v = np.zeros(2), np.zeros(2)
    for _ in range(steps):
        v = beta * v + grad(w)          # accumulated direction, shape (2,)
        w = w - lr * v
    return float(np.linalg.norm(grad(w)))


def adam(steps: int = 60, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999,
         eps: float = 1e-8) -> float:
    w, m, v = np.zeros(2), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(w)
        m = b1 * m + (1 - b1) * g            # first moment
        v = b2 * v + (1 - b2) * g ** 2       # second moment
        m_hat = m / (1 - b1 ** t)            # bias correction
        v_hat = v / (1 - b2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return float(np.linalg.norm(grad(w)))


print(f"grad norm after 60 steps — sgd {sgd():.4f}  momentum {momentum():.4f}  adam {adam():.4f}")
```

The same optimizers in PyTorch, where you would actually use them. Note the parameter
group: in a real model you often give the bias and normalisation weights no weight
decay, because decaying them hurts.

```python
from __future__ import annotations

import torch
from torch import nn

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)   # biases and norms: no decay

opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)

# Warmup then cosine decay: warmup stabilises the early moments, decay settles it.
warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=100)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=900)
sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[100])

x = torch.randn(32, 64)                 # (batch=32, features=64)
y = torch.randint(0, 10, (32,))         # (32,)
for step in range(200):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    sched.step()
    if step % 50 == 0:
        print(step, round(loss.item(), 4), opt.param_groups[0]["lr"])
```

## Real use case

A team fine-tuned a small language model on domain data and got a loss curve that
looked fine but produced a model that answered everything in one flat sentence. The
cause was the optimizer setup, not the data: plain SGD at a rate copied from a vision
recipe. The embedding gradients were tiny (most tokens appear rarely) while the
classifier gradients were large, so one learning rate either moved the classifier too
fast or the embeddings too slowly.

Switching to AdamW with a short warmup fixed it — the per-parameter second moment gives
rare tokens a larger effective step. The team also removed weight decay from the bias
and layer-norm parameters, which stopped the model drifting toward degenerate outputs
late in training. Same data, same architecture, different optimizer, different model.

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss flat from step 0 | Rate too small, or detached gradient | Raise rate, check `.backward()` reaches the parameter |
| Loss oscillates | Rate too high | Lower rate, raise batch size |
| Loss spikes late | Rate still high, or bad batch | Add decay, clip gradients |
| Rare tokens learn slowly | One global rate | Adam/AdamW |

## Common mistakes

- **Mistake:** assuming Adam removes the need to tune the rate — **Instead:** treat `lr` as the first knob; Adam makes it more forgiving, not irrelevant.
- **Mistake:** applying weight decay to biases and norm weights — **Instead:** split parameter groups, as the PyTorch snippet does.
- **Mistake:** skipping warmup on transformers — **Instead:** warm up over the first few hundred steps; the second-moment estimate is noisy early.
- **Mistake:** using `Adam` where `AdamW` is intended — **Instead:** note they differ in how decay is applied; `AdamW` is the usual default for transformers.
- **Mistake:** judging a run by its final loss only — **Instead:** plot loss against step; the shape tells you what to change.
- **Mistake:** changing three hyperparameters at once — **Instead:** change one, re-run, compare curves.

## Checkpoint

1. Run the NumPy comparison with `beta=0.0` for momentum. Confirm it reduces to plain SGD, and explain why.
2. Sweep the learning rate for plain SGD on the narrow-valley problem (try 0.001, 0.01, 0.03, 0.1) and plot gradient norm against step. Find where it diverges.
3. Set Adam's `b1=0` and `b2=0`. Show that the update becomes `lr * g / (|g| + eps)` and describe in one sentence what kind of optimizer that is.

**Self-check:** with `beta=0` the momentum buffer becomes just the current gradient each step, so `w -= lr * v` is exactly `w -= lr * g`, which is plain SGD.

## Go deeper

- [Feedforward Networks](02-feedforward.md) — where these gradients come from.
- [Training Loops](../05-dl-frameworks/03-training-loops.md) — optimizers, schedulers and checkpointing in one reusable loop.
- [Transformers](07-transformers.md) — the architecture whose training recipes these defaults were tuned for.
- [GPU and Checkpoints](../05-dl-frameworks/04-gpu-checkpoints.md) — what to save so a run can resume.

---
[← Previous: Feedforward Networks](02-feedforward.md) · [Stage index](README.md) · [Next: CNNs and RNNs](04-cnns-rnns.md)
