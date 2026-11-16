# 2. Feedforward Networks

> **Stage 04 · Deep Learning** — stacking neurons, and the one element that makes the stack worth anything
>
> **You'll be able to:** build and train a two-layer network in NumPy, and prove numerically that without a nonlinearity the whole stack is one linear map.
>
> **Prerequisites:** [Perceptrons](01-perceptrons.md) · [Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md)

## Why this exists

A perceptron draws one line. Real problems need curves, regions and combinations —
"unusual amount *and* unusual hour", not one or the other. Stacking neurons should
give you that, and it does, but only because of one small function between the layers.

Remove the nonlinearity and the whole network collapses. Multiply two matrices and you
get a third matrix; a hundred layers of `Wx + b` is still a single `W'x + b'`. The
network has the capacity of one linear model and cannot learn XOR no matter how wide
you make it. So this file is really about one idea: **the activation function is what
buys you depth.**

Mental model: **a feedforward network is alternating linear maps and squashing
functions — linear to mix features, nonlinear to bend space.**

## The mental model

```text
  x (batch, 2)
   |  W1 (2, 8) + b1
   v
  z1 = x @ W1 + b1          (batch, 8)
   |  relu
   v
  h1 = relu(z1)             (batch, 8)      <- the bend
   |  W2 (8, 1) + b2
   v
  z2 = h1 @ W2 + b2         (batch, 1)
   |  sigmoid
   v
  p  (batch, 1)  -> loss vs y

  shapes are (batch, in) @ (in, out) + (out,) -> (batch, out)
  every layer: rows = examples, columns = features
```

## Core concepts

### A layer is a matrix multiply plus a bias

`x @ W + b` with `x` of shape `(batch, in)` and `W` of shape `(in, out)` produces
`(batch, out)`. Every example is one row; every output feature is one column. If you
can say those three shapes out loud, you can debug almost any network. The most common
error in practice is a transposed weight, which either crashes on the multiply or —
worse — silently produces a plausible shape with meaningless numbers.

### Nonlinearity is the whole point

ReLU (`max(0, z)`) is the default: cheap, and its gradient is 0 or 1, so it does not
saturate for positive inputs. Sigmoid squashes to `(0, 1)` and is used on an output
when you want a probability. Tanh squashes to `(-1, 1)`. The choice matters less than
having one — but a ReLU unit that only ever sees negative inputs outputs 0 forever and
its gradient dies, which is why initialisation matters.

### Depth beats width for composition

One hidden layer can approximate a wide class of functions in theory, but it may need
an unreasonable number of units. Depth lets the network build intermediate features and
reuse them: layer 1 detects edges, layer 2 combines edges into shapes, layer 3 combines
shapes. That compositional structure is why deep networks are efficient, and it is why
the same block repeated is the design pattern everywhere.

### Backprop is the chain rule, run backwards

The forward pass caches intermediates. The backward pass starts from the loss and
multiplies local derivatives back through each layer, reusing the cached values. For
`z2 = h1 @ W2 + b2` with a sigmoid and binary cross-entropy, `dz2 = p - y`, and then
`dW2 = h1.T @ dz2` — shapes `(8, batch) @ (batch, 1) = (8, 1)`, matching `W2`. Every
gradient has the same shape as the thing it updates; that is your check.

### Overfitting is the default, not the exception

A network with enough parameters will memorise your training set. Watch validation
loss, not training loss. When training loss falls and validation loss rises, you are
past the useful point — regularise (weight decay, dropout), or get more data. This is
the same bias-variance story from [ML Basics](../02-math-and-ml-basics/04-ml-basics.md),
now with more knobs.

## Code

A two-layer network in NumPy with manual gradients, trained on XOR. You should see the
loss fall below 0.1 and all four predictions correct.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)  # (4, 2)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)              # (4, 1)

H = 8
W1 = rng.normal(0, 1.0, size=(2, H))    # (2, 8)
b1 = np.zeros(H)                        # (8,)
W2 = rng.normal(0, 1.0, size=(H, 1))    # (8, 1)
b2 = np.zeros(1)                        # (1,)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def forward(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ W1 + b1          # (4, 2) @ (2, 8) -> (4, 8)
    h1 = np.maximum(0.0, z1)  # relu, same shape
    p = sigmoid(h1 @ W2 + b2) # (4, 8) @ (8, 1) -> (4, 1)
    return z1, h1, p


lr = 0.5
for step in range(4001):
    z1, h1, p = forward(X)
    loss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    dz2 = (p - y) / len(X)              # (4, 1) — BCE + sigmoid derivative
    dW2 = h1.T @ dz2                    # (8, 4) @ (4, 1) -> (8, 1), matches W2
    db2 = dz2.sum(axis=0)               # (1,)
    dh1 = dz2 @ W2.T                    # (4, 1) @ (1, 8) -> (4, 8)
    dz1 = dh1 * (z1 > 0)                # relu derivative, same shape as z1
    dW1 = X.T @ dz1                     # (2, 4) @ (4, 8) -> (2, 8), matches W1
    db1 = dz1.sum(axis=0)               # (8,)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if step % 1000 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

_, _, p = forward(X)
print("predictions:", (p > 0.5).astype(int).ravel(), "expected:", y.ravel().astype(int))
```

The collapse, demonstrated: two linear layers with no activation produce exactly the
same output as one linear layer with the composed weight matrix. You should see a
difference of about zero.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(1)
x = rng.normal(size=(5, 4))        # (5, 4)
W1 = rng.normal(size=(4, 16))      # (4, 16)
W2 = rng.normal(size=(16, 3))      # (16, 3)

two_linear = x @ W1 @ W2           # (5, 4) -> (5, 16) -> (5, 3)
one_linear = x @ (W1 @ W2)         # W1 @ W2 is (4, 3)
print("max abs difference:", float(np.abs(two_linear - one_linear).max()))  # ~1e-15

# With a nonlinearity between them, the two are genuinely different.
relu_two = np.maximum(0.0, x @ W1) @ W2
print("with relu, difference:", float(np.abs(relu_two - one_linear).max()))
```

## Real use case

A resume-screening tool started as logistic regression on five hand-built features. It
worked until the customer asked it to catch "senior title but junior years of
experience" — a combination, not a single signal. A linear model cannot express the
interaction, and adding the product by hand worked for that one case and not the next
twenty.

The replacement was a two-hidden-layer feedforward network over the same features plus
a few text-derived scores. The design choice that mattered was the ReLU between layers,
which let the network carve the feature space into regions rather than slicing it with
one hyperplane. The team kept the linear model in production as a fallback and
monitored both; the network's decisions are explained with feature-attribution tooling
because, unlike the perceptron, its weights are not readable on their own.

## Common mistakes

- **Mistake:** omitting the activation between layers — **Instead:** always place one; without it the stack is a single linear map, as the second snippet shows.
- **Mistake:** forgetting to divide `dz2` by the batch size — **Instead:** average the loss and the gradient, or your effective learning rate scales with batch size.
- **Mistake:** transposing a weight matrix — **Instead:** write the shapes in a comment above every multiply; `(batch, in) @ (in, out)` is the invariant.
- **Mistake:** initialising all weights to zero — **Instead:** use random small values; identical units receive identical gradients and never differentiate.
- **Mistake:** watching training loss only — **Instead:** hold out a validation set and stop when its loss turns upward.
- **Mistake:** a very large learning rate to "converge faster" — **Instead:** start small and check the loss falls smoothly; divergence to `nan` is almost always the learning rate.

## Checkpoint

1. Train the XOR network with `H=1` and with `H=8`. Record the final loss for each and explain the difference.
2. Replace ReLU with tanh and retrain. Does it still learn XOR? How does the loss curve differ?
3. Remove the `/ len(X)` from `dz2` and retrain with the same `lr`. Describe what happens to the loss and explain why in one sentence.

**Self-check:** with `H=1` the network has one hidden unit and effectively one bend — not enough to separate XOR's corners, so the loss stalls high; `H=8` gives enough units to build the two regions XOR needs.

## Go deeper

- [Optimizers](03-optimizers.md) — what to use instead of the plain update loop here.
- [Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md) — the chain rule that produced `dW1` and `dW2`.
- [Training Loops](../05-dl-frameworks/03-training-loops.md) — the same loop, written once and reused.
- [Transformers](07-transformers.md) — the same block idea, with attention inside instead of a plain matrix.

---
[← Previous: Perceptrons](01-perceptrons.md) · [Stage index](README.md) · [Next: Optimizers](03-optimizers.md)
