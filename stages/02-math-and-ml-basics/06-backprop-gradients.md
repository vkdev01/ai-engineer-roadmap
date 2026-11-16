# 6. Backpropagation and Gradients

> **Stage 02 · Math and ML Basics** — how a network finds out which of its millions of numbers caused the error.
>
> **You'll be able to:** write the forward and backward pass of a two-layer network by hand, and verify your gradients numerically.
>
> **Prerequisites:** [Loss Functions](05-loss-functions.md) · [Calculus](02-calculus.md)

## Why this exists

A two-layer network on small data has hundreds of weights. A transformer has billions. After a forward pass you get one loss number; you need a derivative for every single weight. Doing that one weight at a time would cost a forward pass each — hopeless.

Backpropagation is the trick that makes it cheap: run forward once, run backward once, and get every gradient in the process. It is the chain rule applied right-to-left, with each layer's derivative cached during the forward pass.

The mental model: **forward computes values and remembers them; backward passes blame leftwards, one layer at a time.**

## The mental model

```text
FORWARD (left to right)                 BACKWARD (right to left)
x -> [W1,b1] -> h -> [relu] -> a        dL/dW2 = a.T @ dz2
       |          |        [W2,b2] -> z2 -> softmax -> L
       |          |                     dz2 = (p - y)          <- from the loss
   cache x     cache a                   dL/dW1 = x.T @ dz1
                                         dz1 = (dz2 @ W2.T) * (a > 0)
```

Every arrow is a local derivative. Backward multiplies the incoming gradient by each local derivative — nothing more.

## Core concepts

### The forward pass must cache what backward needs

Backward needs the inputs and outputs of each layer, so forward stores them. This is why training uses more memory than inference: activations for the whole batch stay alive. It is also the reason `torch.no_grad()` exists for inference.

### Gradients flow backwards through the transposed weight matrix

If forward is `z = a @ W`, then the gradient with respect to `a` is `dz @ W.T` and with respect to `W` is `a.T @ dz`. The transpose is not a convention; it comes from the shapes. Writing these down and checking shapes is how you avoid the classic `(batch, out)` versus `(out, batch)` bug.

### The activation derivative is applied elementwise

`relu'(x)` is 1 where `x > 0` and 0 elsewhere, so backward multiplies the incoming gradient by a mask. Sigmoid's derivative is `σ(x)(1-σ(x))`, at most 0.25 — which is the vanishing-gradient story from [Calculus](02-calculus.md) in one line of code.

### Softmax plus cross-entropy collapses to `p - y`

Differentiating the combination gives `softmax(logits) - one_hot(label)`. This is why every framework fuses them: it is simpler and numerically stabler than composing two derivatives. It is also the single most useful formula to know by heart.

### Gradient checking is not optional

For each parameter, nudge it by `h`, recompute the loss, and compare `(L+ - L-)/2h` with your analytic gradient. If the relative error is above about `1e-6` in float64, your backward pass has a bug. Do this on a tiny network before training anything real.

### Accumulate, then update

Backward computes gradients; a separate step applies them. Keeping them apart is what lets you accumulate over a batch, clip gradients, and swap optimisers without touching the backward code.

## Code

A complete two-layer network, forward and backward, in NumPy. Read the shapes in the comments — they are the whole story.

```python
from __future__ import annotations

import numpy as np


def init(rng: np.random.Generator, n_in: int, n_hidden: int, n_out: int) -> dict[str, np.ndarray]:
    # Small random init: scaling by 1/sqrt(n_in) keeps activations in range.
    return {
        "W1": rng.normal(scale=1 / np.sqrt(n_in), size=(n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(scale=1 / np.sqrt(n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def forward(params: dict[str, np.ndarray], X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = X @ params["W1"] + params["b1"]        # (N, in)  @ (in, H)  -> (N, H)
    a1 = np.maximum(z1, 0.0)                    # relu
    z2 = a1 @ params["W2"] + params["b2"]       # (N, H)  @ (H, out) -> (N, out)
    shifted = z2 - z2.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def backward(params: dict[str, np.ndarray], cache: dict[str, np.ndarray],
             y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    N = y.size
    probs = cache["probs"]
    loss = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    # dL/dlogits = softmax - one_hot, averaged over the batch
    dz2 = probs.copy()
    dz2[np.arange(N), y] -= 1.0
    dz2 /= N                                     # (N, out)

    grads: dict[str, np.ndarray] = {}
    grads["W2"] = cache["a1"].T @ dz2            # (H, N) @ (N, out) -> (H, out)
    grads["b2"] = dz2.sum(axis=0)                # (out,)

    da1 = dz2 @ params["W2"].T                   # (N, out) @ (out, H) -> (N, H)
    dz1 = da1 * (cache["z1"] > 0)                # relu mask, elementwise

    grads["W1"] = cache["X"].T @ dz1             # (in, N) @ (N, H) -> (in, H)
    grads["b1"] = dz1.sum(axis=0)                # (H,)
    return loss, grads
```

The gradient check that proves the backward pass is correct:

```python
from __future__ import annotations

import numpy as np


def grad_check(params, X, y, h: float = 1e-6) -> dict[str, float]:
    _, analytic = backward(params, forward(params, X)[1], y)
    report: dict[str, float] = {}
    for name, value in params.items():
        numeric = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + h
            plus = backward(params, forward(params, X)[1], y)[0]
            value[index] = original - h
            minus = backward(params, forward(params, X)[1], y)[0]
            value[index] = original
            numeric[index] = (plus - minus) / (2 * h)
        # Relative error, safe when gradients are tiny.
        denom = np.maximum(np.abs(analytic[name]) + np.abs(numeric), 1e-8)
        report[name] = float(np.max(np.abs(analytic[name] - numeric) / denom))
    return report


rng = np.random.default_rng(0)
params = init(rng, n_in=3, n_hidden=4, n_out=2)
X = rng.normal(size=(5, 3))
y = np.array([0, 1, 1, 0, 1])
print(grad_check(params, X, y))   # every value should be around 1e-7 or smaller
```

## Real use case

A team implementing a small model from scratch to understand transformers kept getting loss stuck near `ln(2)` on a binary task — exactly the value for guessing. The forward pass was obviously fine, since outputs were valid probabilities summing to one.

The bug was one transposed matrix in the backward pass: `dz2 @ W2.T` had been written as `W2.T @ dz2`. Because the toy batch size happened to equal the hidden size, the shapes still multiplied, so nothing raised — the gradient was simply wrong, and the weights drifted in a direction that cancelled out. Adding `grad_check` before any training run found it in one print: relative error of `1.7` on `W1`, versus `1e-8` on everything else.

After the fix, the same code trained to a loss well below the guessing baseline. The design choice that mattered was checking gradients against a numerical reference instead of trusting that shapes lined up.

## Common mistakes

- **Mistake:** forgetting `dz2 /= N`, so the gradient scales with batch size. — **Instead:** average the gradient exactly as you averaged the loss.
- **Mistake:** using `z1 > 0` versus `a1 > 0` inconsistently in the ReLU mask. — **Instead:** they agree except at exactly zero; be consistent and document it.
- **Mistake:** computing gradients but never zeroing them between steps. — **Instead:** reset or overwrite; in frameworks, call `zero_grad()`.
- **Mistake:** skipping the numerical check on a "simple" network. — **Instead:** check every new layer, once, forever.
- **Mistake:** storing activations for the whole dataset instead of the batch. — **Instead:** backward per batch; memory is the constraint, not elegance.

## Checkpoint

1. Add a third hidden layer and write the backward pass for it without looking at the pattern above.
2. Replace ReLU with sigmoid and observe the gradient magnitudes on the first layer; explain the drop.
3. Break the backward pass deliberately (remove the ReLU mask) and show that `grad_check` catches it.

**Self-check:** `grads["W2"] = a1.T @ dz2` — the transpose puts the hidden dimension first so the result matches `W2`'s shape `(H, out)`.

## Go deeper

- [Ship Gradient Descent](07-ship-gradient-descent.md) — put this backward pass into a trainer that runs.
- [Autograd](../05-dl-frameworks/02-autograd.md) — how PyTorch builds this graph automatically.
- [Attention](../04-deep-learning/06-attention.md) — the same rules, applied to a much larger graph.

---
[← Previous: Loss Functions](05-loss-functions.md) · [Stage index](README.md) · [Next: Ship Gradient Descent](07-ship-gradient-descent.md)
