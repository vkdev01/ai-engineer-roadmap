# 7. Ship Gradient Descent

> **Stage 02 · Math and ML Basics** — you build the training loop yourself, once, so that every framework afterwards is familiar rather than magic.
>
> **You'll be able to:** fit a two-layer network to a toy dataset with only NumPy, add momentum and a learning-rate schedule, and prove the training is correct with a gradient check and a loss curve.
>
> **Prerequisites:** [Backpropagation and Gradients](06-backprop-gradients.md) · [Loss Functions](05-loss-functions.md)

## Why this exists

You can call `model.fit()` and get a trained model without understanding any of this. But when fine-tuning goes wrong — loss flat, loss `nan`, loss falling while quality drops — the fix comes from knowing what the loop does at each step. Building it once in NumPy is the fastest way to own that knowledge permanently.

The project, `micrograd-numpy`, is small on purpose: a two-layer classifier on a synthetic two-moons-style dataset, trained by hand. No autograd, no PyTorch, no hidden steps.

The mental model: **training is a loop of forward, backward, update — and everything else is decoration around that loop.**

## The mental model

```text
for each epoch:
    for each batch:
        forward  -> logits -> loss                (cache activations)
        backward -> grads for W1,b1,W2,b2          (chain rule, right to left)
        update   -> w -= lr * (momentum-adjusted grad)
    evaluate on validation -> record loss/accuracy
    adjust lr by schedule
```

## Core concepts

### Layout: separate the maths from the loop

The pure functions (forward, backward, losses) live apart from the loop, so they can be tested alone. This is the same structure you will see in every serious training codebase.

```text
micrograd-numpy/
├── pyproject.toml
├── README.md
├── src/micrograd_numpy/
│   ├── __init__.py
│   ├── data.py        # make_moons-like toy dataset
│   ├── net.py         # init, forward, backward
│   ├── optim.py       # SGD with momentum, lr schedule
│   └── train.py       # the loop, CLI entry point
└── tests/
    ├── test_gradcheck.py
    └── test_optim.py
```

### Momentum smooths the path

Plain SGD steps along `-grad`. Momentum keeps a running average `v = β·v + grad` and steps along `-lr·v`, which carries motion through flat regions and damps oscillation in narrow valleys. `β = 0.9` is the usual starting point.

### A learning-rate schedule usually beats a fixed rate

High early (make progress), low late (settle precisely). Cosine decay is one line: `lr_t = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))`. Keep the schedule a function of step, so you can log and reproduce it.

### Shuffling and batching are part of correctness

If your data is ordered by class, minibatches are unrepresentative and training lurches. Shuffle the index array each epoch with a seeded generator so runs are reproducible.

### Stop on validation, not on train

Record validation loss every epoch and keep the parameters from the best epoch. That single habit prevents most overfitting in small projects.

### Prove it three ways

A gradient check (maths is right), a loss curve that falls below the baseline (learning happens), and a held-out accuracy that holds up (it generalises). Any two of three is not enough.

## Code

The optimiser and the training loop. Everything else is the forward/backward pass from the previous file.

```python
# src/micrograd_numpy/optim.py
from __future__ import annotations

import math

import numpy as np


class MomentumSGD:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 0.1, beta: float = 0.9) -> None:
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}

    def schedule(self, step: int, total: int, lr_min: float = 0.001) -> float:
        # Cosine decay from self.lr down to lr_min over the run.
        return lr_min + 0.5 * (self.lr - lr_min) * (1 + math.cos(math.pi * step / total))

    def step(self, grads: dict[str, np.ndarray], step: int, total: int) -> None:
        lr = self.schedule(step, total)
        for name, grad in grads.items():
            self.velocity[name] = self.beta * self.velocity[name] + grad
            self.params[name] -= lr * self.velocity[name]
```

```python
# src/micrograd_numpy/train.py
from __future__ import annotations

import numpy as np

from .data import make_toy
from .net import backward, forward, init
from .optim import MomentumSGD


def accuracy(params, X: np.ndarray, y: np.ndarray) -> float:
    probs, _ = forward(params, X)
    return float(np.mean(probs.argmax(axis=1) == y))


def train(epochs: int = 300, batch_size: int = 32, lr: float = 0.3, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_val, y_val = make_toy(seed=seed)
    params = init(rng, n_in=X_tr.shape[1], n_hidden=16, n_out=2)
    optim = MomentumSGD(params, lr=lr)

    steps_per_epoch = max(1, len(X_tr) // batch_size)
    total = epochs * steps_per_epoch
    history: list[dict[str, float]] = []
    best = {"val_loss": float("inf"), "params": {k: v.copy() for k, v in params.items()}}

    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(X_tr))          # shuffle every epoch
        for start in range(0, len(X_tr), batch_size):
            idx = order[start:start + batch_size]
            _, cache = forward(params, X_tr[idx])
            _, grads = backward(params, cache, y_tr[idx])
            optim.step(grads, step, total)
            step += 1

        val_loss = backward(params, forward(params, X_val)[1], y_val)[0]
        history.append({"epoch": epoch, "val_loss": val_loss, "val_acc": accuracy(params, X_val, y_val)})
        if val_loss < best["val_loss"]:              # early stopping on validation
            best = {"val_loss": val_loss, "params": {k: v.copy() for k, v in params.items()}}

    return {"history": history, "best": best,
            "train_acc": accuracy(best["params"], X_tr, y_tr),
            "val_acc": accuracy(best["params"], X_val, y_val)}


if __name__ == "__main__":
    result = train()
    print(f"train acc {result['train_acc']:.3f}  val acc {result['val_acc']:.3f}")
```

## How to prove it works

Three checks, run in order. Each one tests a different claim.

```bash
# 1. The maths is right: analytic gradients match numerical ones
pytest -q tests/test_gradcheck.py     # passes, max relative error < 1e-6

# 2. Learning happens: loss falls well below the "guessing" baseline
python -m micrograd_numpy.train       # val acc well above 0.5, val_loss below ln(2)
```

The gradient check test, which is the one that catches real bugs:

```python
# tests/test_gradcheck.py
import numpy as np

from micrograd_numpy.net import backward, forward, init


def test_gradients_match_numeric() -> None:
    rng = np.random.default_rng(0)
    params = init(rng, n_in=3, n_hidden=4, n_out=2)
    X, y = rng.normal(size=(6, 3)), np.array([0, 1, 0, 1, 1, 0])
    _, analytic = backward(params, forward(params, X)[1], y)
    for name, value in params.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + 1e-6
            plus = backward(params, forward(params, X)[1], y)[0]
            value[index] = original - 1e-6
            minus = backward(params, forward(params, X)[1], y)[0]
            value[index] = original
            numeric = (plus - minus) / 2e-6
            assert abs(numeric - analytic[name][index]) < 1e-5
```

Third check: the learning curve. Plot `history` val loss against epoch and confirm it drops fast, then flattens. A curve that never falls means a broken backward pass or a learning rate near zero; a curve that spikes to `nan` means the learning rate is too high.

What "done" looks like: `train acc` and `val acc` are both clearly above 0.5 and close to each other (no big train/val gap), and re-running with the same `seed` gives identical numbers.

## Real use case

A team needed a tiny intent classifier that had to run on a CPU-only edge box, so a transformer was out of scope. They wrote a NumPy MLP with this exact structure, trained on a few thousand labelled utterances, and shipped it as a 40 KB weight file behind a FastAPI endpoint.

Two things made the project trustworthy rather than a prototype. First, the gradient-check test in CI meant any refactor of the backward pass that broke the maths failed before review — the check caught a transposed matrix during a "harmless" cleanup. Second, the training script kept the best-validation parameters and logged the curve to JSON, so when a later dataset update halved accuracy, comparing curves showed the new labels had a much higher noise floor rather than the code being wrong.

The design choice that mattered was separating the maths (testable pure functions) from the loop (CLI and logging), which is why the same code survived a dataset change and a refactor without a rewrite.

## Common mistakes

- **Mistake:** training on the whole dataset and reporting training accuracy as quality. — **Instead:** hold out a validation split and report that.
- **Mistake:** a fixed learning rate so high the loss oscillates, then blaming the architecture. — **Instead:** try 10x smaller before changing anything structural.
- **Mistake:** forgetting to shuffle, so batches are class-ordered. — **Instead:** permute indices each epoch with a seeded RNG.
- **Mistake:** momentum without resetting velocity when the learning rate changes drastically. — **Instead:** rebuild the optimiser state when you change regimes.
- **Mistake:** no seed, so results are not reproducible and every comparison is noise. — **Instead:** seed the generator and log it with the run.

## Checkpoint

1. Add L2 regularisation (`loss += λ·Σ‖W‖²`) with its gradient, and show the train/val gap shrink on a deliberately over-capacity network.
2. Swap momentum for plain SGD and compare epochs-to-convergence on the same seed.
3. Log the history to JSON, plot it, and identify the epoch where validation loss is minimised — then confirm it matches the early-stopping choice.

**Self-check:** the L2 gradient is `2λW`, added to `grads["W1"]` and `grads["W2"]` only — not to biases, which have no effect on a weight-norm penalty.

## Go deeper

- [Stage 03 — Data Engineering](../03-data-engineering/README.md) — next stage; getting real data into shape.
- [Autograd](../05-dl-frameworks/02-autograd.md) — the same loop, with the graph built for you.
- [Optimizers](../04-deep-learning/03-optimizers.md) — Adam, schedules and what they change about convergence.

---
[← Previous: Backpropagation and Gradients](06-backprop-gradients.md) · [Stage index](README.md) · [Next: Stage 03 — Data Engineering](../03-data-engineering/README.md)
