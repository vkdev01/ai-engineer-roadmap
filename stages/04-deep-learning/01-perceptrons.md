# 1. Perceptrons

> **Stage 04 · Deep Learning** — the smallest unit that learns, and the reason neural networks went quiet for a decade
>
> **You'll be able to:** implement a perceptron, train it on separable data, and demonstrate numerically the problem it cannot solve.
>
> **Prerequisites:** [Stage 03 — Data Engineering](../03-data-engineering/README.md) · [ML Basics](../02-math-and-ml-basics/04-ml-basics.md)

## Why this exists

Before 1958, "learning from data" meant fitting a curve by hand. The perceptron made
it mechanical: take a weighted sum of inputs, compare it to a threshold, and if the
answer is wrong, nudge the weights in the direction that would have been right. Repeat
until no answer is wrong. That loop is the ancestor of every training run you will ever
launch.

It is also the cautionary tale. The perceptron can only draw a straight line. A single
perceptron cannot learn XOR — a fact that stalled the field for years and that you
should be able to prove yourself, in ten lines, before trusting anything built on top.

Mental model: **a perceptron is a dot product plus a step — a learned straight line
through your data.**

## The mental model

```text
  x1 --w1--\
  x2 --w2---(+ b) --> z = w·x + b --> step(z) --> ŷ  in {0, 1}
  x3 --w3--/                          |
                                      +--> update: w += lr * (y - ŷ) * x

  geometry, 2 inputs:               separable          not separable (XOR)
        x2                              /                  x2
        |                        o    /   x                |   o       x
        |                          o /                    |       o
        |   x   x     x          /                        |   x       o
        +-------------------- x1                        +-------------------- x1
           one line splits them                    no line splits them
```

## Core concepts

### The unit is a weighted sum and a threshold

`z = w·x + b` is a dot product: it measures how much the input points in the
direction the weights point. The step function turns that score into a decision. The
bias `b` shifts the line; without it every boundary must pass through the origin,
which is a surprisingly common bug when accuracy plateaus at an odd value.

### Learning is a correction, not a calculation

There is no formula for the right weights. The perceptron rule is: if the prediction is
correct, do nothing; if it predicted 0 and should be 1, add the input to the weights;
if it predicted 1 and should be 0, subtract it. That is exactly
`w += lr * (y - ŷ) * x`. It converges — provably — if the data is linearly separable.

### The decision boundary is a hyperplane

`w·x + b = 0` is a line in 2-D, a plane in 3-D, a hyperplane in general. The weights
*are* the normal vector to that surface. This is why scaling a feature changes which
boundary you get: a large-magnitude feature dominates the dot product. Normalising
inputs is not cosmetic.

### The step function blocks the gradient

The perceptron rule works, but it is not gradient descent — the step function's
derivative is zero almost everywhere, so there is no gradient to follow. Replace the
step with a sigmoid or a ReLU and you get a differentiable unit, which is the entire
difference between a perceptron and a neuron in a modern network.

### XOR is the boundary you cannot draw

Map the four XOR points: `(0,0)->0`, `(1,1)->0`, `(0,1)->1`, `(1,0)->1`. The two 1s sit
on opposite corners, the two 0s on the other opposite corners. No single straight line
separates the corners. This is not a training problem — no weights exist. The fix is a
second layer, which is the next file.

## Code

A perceptron in NumPy, trained on an AND gate (separable) and then on XOR (not). You
should see accuracy reach 1.0 on AND. XOR never converges: from zero-initialised
weights it collapses to predicting one class for every input, so accuracy lands at
0.0 — change the initial weights and you will get 0.5 instead, which is the same
failure wearing a different mask.

```python
from __future__ import annotations

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)   # shape (4, 2)
y_and = np.array([0, 0, 0, 1], dtype=np.float64)                   # shape (4,)
y_xor = np.array([0, 1, 1, 0], dtype=np.float64)                   # shape (4,)


def train_perceptron(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                     epochs: int = 20) -> tuple[np.ndarray, float, list[float]]:
    n_features = X.shape[1]
    w = np.zeros(n_features)      # shape (2,)
    b = 0.0
    history: list[float] = []
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):                 # shape (2,), scalar
            z = float(w @ xi + b)                # dot product -> scalar
            y_hat = 1.0 if z >= 0 else 0.0
            update = lr * (yi - y_hat)           # 0, +lr or -lr
            w += update * xi                     # both (2,)
            b += update
            errors += int(y_hat != yi)
        history.append(errors / len(y))
    return w, b, history


w, b, hist = train_perceptron(X, y_and)
print("AND  acc:", 1 - hist[-1], "w:", w.round(2), "b:", round(b, 2))
w, b, hist = train_perceptron(X, y_xor, epochs=200)
print("XOR  acc:", 1 - hist[-1], "w:", w.round(2), "b:", round(b, 2))
```

The proof that XOR has no solution: search the whole weight space on a grid and show
that no `(w1, w2, b)` gets all four points right.

```python
from __future__ import annotations

import itertools

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([0, 1, 1, 0], dtype=np.float64)

grid = np.linspace(-3, 3, 61)
best = 0.0
for w1, w2, b in itertools.product(grid, grid, grid):
    preds = ((X @ np.array([w1, w2]) + b) >= 0).astype(np.float64)
    best = max(best, float((preds == y).mean()))

print("best accuracy over 226,981 weight settings:", best)   # 0.75 — never 1.0
```

## Real use case

A payments team wanted a fraud flag: block a transaction if a weighted sum of three
signals crosses a threshold. A single perceptron was a good fit — the signals were
roughly monotone in risk, the data was separable, and the model was a number the risk
team could read and adjust.

It shipped, and it failed in one specific way: fraud that combined two individually
mild signals (a slightly unusual amount *and* a slightly unusual hour) scored low,
because a weighted sum cannot express "both, together". That is XOR in production.
The team kept the perceptron as a fast pre-filter and added a two-layer network for
the combination cases — the fix that the next file in this stage builds.

| Case | Single perceptron | Two-layer network |
|---|---|---|
| One strong signal | Works | Works |
| Two mild signals together | Cannot represent | Learns it |
| Explaining a decision | Weights are readable | Needs attribution methods |

## Common mistakes

- **Mistake:** forgetting the bias term — **Instead:** always keep `b`; without it the boundary is forced through the origin and some data can never be separated.
- **Mistake:** unnormalised features — **Instead:** standardise inputs; a feature with 1000x the scale dominates the dot product and the learning.
- **Mistake:** expecting a perceptron to fit XOR — **Instead:** add a hidden layer (see [Feedforward Networks](02-feedforward.md)); no weights exist for the single-unit version.
- **Mistake:** reading the perceptron rule as gradient descent — **Instead:** note the step function has no useful derivative; the rule is a correction heuristic that happens to converge when separable.
- **Mistake:** tuning `lr` without checking separability — **Instead:** first plot the data; if classes overlap, no learning rate will fix it.

## Checkpoint

1. Train the perceptron on an OR gate and on NAND, and record how many epochs each needs. Both are separable — do they need the same number?
2. Modify `train_perceptron` to track the weight vector after every epoch and plot the trajectory in weight space. Describe what the path looks like.
3. Add a third input to XOR (`x3 = x1*x2`) and retrain. Show that accuracy reaches 1.0, and explain in one sentence why this is the hidden layer in disguise.

**Self-check:** AND is separable and converges to accuracy 1.0; XOR never converges — with zero-initialised weights it collapses to one class, and with other initialisations it stalls at 0.5 — because its two positive and two negative points sit on opposite corners, so no single line splits them.

## Go deeper

- [Feedforward Networks](02-feedforward.md) — the second layer that solves what a perceptron cannot.
- [Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md) — how a differentiable unit gets trained.
- [Loss Functions](../02-math-and-ml-basics/05-loss-functions.md) — replacing the error count with a differentiable loss.
- [PyTorch](../05-dl-frameworks/01-pytorch.md) — the same unit, as a library layer.

---
[← Previous: Stage 03 — Data Engineering](../03-data-engineering/README.md) · [Stage index](README.md) · [Next: Feedforward Networks](02-feedforward.md)
