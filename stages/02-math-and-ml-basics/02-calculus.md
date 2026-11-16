# 2. Calculus

> **Stage 02 · Math and ML Basics** — derivatives tell you which way to nudge a number to make the error smaller.
>
> **You'll be able to:** compute derivatives and partial derivatives of the functions in a model, apply the chain rule through nested functions, and read a gradient as "a direction to step".
>
> **Prerequisites:** [Linear Algebra](01-linear-algebra.md) · [Stage 02 — Math and ML Basics](README.md)

## Why this exists

Training a model is a search. You have knobs (weights) and a number that says how wrong you are (the loss). You need to know, for each knob, whether turning it up or down reduces the loss — and roughly how much. That is precisely what a derivative gives you.

Without this, "the model is not learning" is a mystery. With it, you can say: the gradient is zero here, or the learning rate overshoots the minimum, or one layer's gradient is vanishing.

The mental model: **a derivative is a sensitivity — change the input a little, see how much the output moves.**

## The mental model

```text
loss
  |            *
  |         *     *
  |      *           *
  |   *                 *
  |__________________________ weight
       ^        ^
       |        |
   slope<0    slope=0    slope>0
   increase w  minimum   decrease w
```

| Concept | Question it answers | In a model |
|---|---|---|
| Derivative | how does output move per unit input | one weight |
| Partial derivative | same, holding others fixed | one weight among millions |
| Gradient | vector of all partials | `∇L` for all weights |
| Chain rule | derivative through composition | layer after layer |
| Learning rate | how far to step along `-∇L` | a hyperparameter |

## Core concepts

### The derivative is a slope, and a slope is a limit

For `f(x) = x²`, `f'(x) = 2x`. At `x = 3` the slope is 6: nudge `x` up by 0.001 and `f` rises by about 0.006. You rarely differentiate from the definition, but you should know that the derivative is what a tiny finite difference converges to — because that is exactly how you will numerically check your code.

### Useful rules, memorised, and that is enough

| `f(x)` | `f'(x)` |
|---|---|
| `c` | `0` |
| `x` | `1` |
| `xⁿ` | `n·xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `ln x` | `1/x` |
| `max(0, x)` | `1` if `x > 0`, else `0` |

### Partial derivatives: one knob at a time

For `f(w, b) = (wx + b - y)²`, the partial with respect to `w` treats `b` as a constant: `2(wx + b - y)·x`. The extra `x` is the chain rule quietly appearing. The gradient stacks every partial into one vector so a single update can move all knobs at once.

### The chain rule is the whole of backpropagation

If `y = g(u)` and `u = f(x)`, then `dy/dx = g'(u) · f'(x)`. Derivatives multiply along a chain. A neural network is a long chain, so backpropagation is just the chain rule applied right to left, reusing each intermediate derivative once.

### Gradient descent steps against the gradient

`w ← w - lr · ∂L/∂w`. The minus sign is because the gradient points uphill; you want downhill. The learning rate `lr` decides step size: too small and training crawls, too large and the loss bounces or diverges.

### Numerical gradients are your safety net

`(f(x+h) - f(x-h)) / (2h)` approximates the derivative to good accuracy for small `h`. Every hand-derived gradient in the next few files gets checked this way — if analytic and numerical disagree, the analytic one is wrong.

## Code

Derivatives of the functions you will actually meet, each checked against a central difference.

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numeric_grad(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    # Central difference: more accurate than a one-sided difference.
    return (f(x + h) - f(x - h)) / (2 * h)


cases: list[tuple[str, Callable[[float], float], Callable[[float], float]]] = [
    ("x^2", lambda x: x**2, lambda x: 2 * x),
    ("exp", lambda x: np.exp(x), lambda x: np.exp(x)),
    ("relu", lambda x: max(0.0, x), lambda x: 1.0 if x > 0 else 0.0),
]

for name, f, df in cases:
    for x in (-2.0, 0.5, 3.0):
        print(name, x, round(numeric_grad(f, x), 5), round(df(x), 5))
```

The chain rule on a real loss: mean squared error of a single linear unit, gradients for both parameters.

```python
from __future__ import annotations

import numpy as np

x, y = 2.0, 10.0            # one example: input 2.0, target 10.0
w, b = 0.5, 0.0


def loss(w: float, b: float) -> float:
    pred = w * x + b
    return (pred - y) ** 2   # squared error, not yet averaged


# dL/dpred = 2(pred - y); dpred/dw = x; dpred/db = 1  -> chain rule
for step in range(50):
    pred = w * x + b
    d_pred = 2 * (pred - y)
    d_w = d_pred * x
    d_b = d_pred * 1.0
    w -= 0.01 * d_w
    b -= 0.01 * d_b

print(round(w, 3), round(b, 3), round(loss(w, b), 5))   # w -> 4.08, b -> 1.79 after 50 steps
```

Checking that by hand-derived gradient numerically, which is the habit to build:

```python
from __future__ import annotations

import numpy as np


def grad_check(f, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
    g = np.zeros_like(params)
    for i in range(params.size):
        original = params[i]
        params[i] = original + h
        plus = f(params)
        params[i] = original - h
        minus = f(params)
        params[i] = original
        g[i] = (plus - minus) / (2 * h)
    return g


def f(p: np.ndarray) -> float:
    return float((p[0] * 2.0 + p[1] - 10.0) ** 2)


print(grad_check(f, np.array([0.5, 0.0])))   # [-36.0, -18.0]
```

## Real use case

A team training a small classifier from scratch found the loss flat at exactly `0.693` — the value you get from predicting 0.5 for every example. Nothing was crashing, so the instinct was "we need more data". Adding data changed nothing.

Printing gradients for each layer showed the first layer's gradients at `1e-9` while the last layer's were normal. The cause was a `sigmoid` activation stacked three deep with weights initialised small: the derivative of sigmoid is at most 0.25, so three layers multiplied it to under 0.02, and with small weights the chain shrank to nothing. The gradient was not wrong, it was vanishing — a calculus fact, visible in the numbers.

Switching the hidden activations to `relu` (derivative exactly 1 for positive inputs) and scaling the initial weights brought the first layer's gradients back to a comparable order and training started moving within a few dozen steps.

The design choice that mattered was instrumenting per-layer gradient magnitudes: a number derived directly from the chain rule told them where to look in minutes.

## Common mistakes

- **Mistake:** forgetting the minus sign and stepping uphill. — **Instead:** `w -= lr * grad`; if the loss rises, check the sign first.
- **Mistake:** a learning rate so large the loss oscillates or becomes `nan`. — **Instead:** start small, plot the loss, and reduce by 10x when it diverges.
- **Mistake:** checking gradients with `h` too small. — **Instead:** `h` around `1e-5` for float64; below that, rounding error dominates.
- **Mistake:** treating a zero gradient as "converged" without checking. — **Instead:** it can mean a dead ReLU or a saturated sigmoid.
- **Mistake:** differentiating a mean over a batch and forgetting the `1/N`. — **Instead:** average the gradient too, or the effective learning rate scales with batch size.

## Checkpoint

1. Derive `∂L/∂b` for `L = (wx + b - y)²` on paper, then confirm it with `grad_check`.
2. Fit `w` for `y = wx` with no bias using gradient descent, and find the largest learning rate that still converges.
3. Write `f(w, b) = (w·x₁ + b - y₁)² + (w·x₂ + b - y₂)²` for two examples and derive both partials; verify numerically.

**Self-check:** `∂L/∂b = 2(pred - y)`, and `grad_check` at `(w=0.5, b=0)` returns `[-36.0, -18.0]` because `x = 2` doubles the `w` gradient (the raw error there is -9).

## Go deeper

- [Backpropagation and Gradients](06-backprop-gradients.md) — the chain rule applied layer by layer.
- [Loss Functions](05-loss-functions.md) — what function you are differentiating in the first place.
- [Optimizers](../04-deep-learning/03-optimizers.md) — what people build once plain gradient descent is too slow.

---
[← Previous: Linear Algebra](01-linear-algebra.md) · [Stage index](README.md) · [Next: Probability and Bayes](03-probability-bayes.md)
