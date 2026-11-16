# 2. Autograd From Scratch

> **Stage 05 · DL Frameworks** — What `.backward()` really does: a graph of small operations, each knowing its own local derivative, replayed in reverse.
>
> **You'll be able to:** build a working autograd engine in about forty lines, then explain any gradient bug in terms of the graph it came from.
>
> **Prerequisites:** [6. Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md) · [1. PyTorch: Tensors and Modules](01-pytorch.md)

## Why this exists

Backpropagation is the chain rule applied to a graph. You have already seen the maths in stage 02. The trouble is that in real code the graph is invisible: you write `loss = criterion(model(x), y)` and a number appears in `param.grad`. When that number is wrong, or zero, or `None`, you have nothing to look at.

Building a tiny engine fixes this permanently. Once you have written a `Value` class that records its parents and multiplies local gradients, PyTorch's behaviour stops being magic: `zero_grad`, `detach`, `no_grad`, `retain_graph` and the classic "gradient is None" bug all become obvious consequences of how the graph is stored.

The mental model: *each operation remembers its inputs and its local derivative; `backward` walks the graph from the output to the leaves, multiplying local derivatives along the way.*

## The mental model

```text
 forward (record)                 backward (replay)
 ────────────────                 ─────────────────
 a ──┐                            d = 1.0  (seed)
     ├─► c = a*b   c.parents=(a,b)   │
 b ──┘                               ▼
     ├─► d = c + a   d.parents=(c,a)  c.grad += d.grad * b   (∂d/∂c)
                                     a.grad += d.grad * c   (∂d/∂c)
                                     a.grad += d.grad * 1   (∂d/∂a)
                                     b.grad += c.grad * a   (∂c/∂b)
```

Two rules make the whole thing work. Gradients **accumulate** (`+=`, never `=`), because a value used twice receives a contribution from each use. And the walk is in **reverse topological order**, so every node's gradient is complete before it is used to compute its parents'.

## Core concepts

### A node stores value, gradient, and how it was made

Each `Value` holds a number, a `.grad` starting at zero, a list of the nodes that produced it, and a `_backward` closure that knows the local derivative. Addition's closure pushes the incoming gradient to both parents unchanged; multiplication's multiplies by the *other* operand.

### Accumulation is why `zero_grad()` exists

If you call `loss.backward()` twice without clearing, gradients double. That is not a bug in the engine — it is the feature that makes gradient accumulation over micro-batches possible. PyTorch therefore leaves clearing to you, and forgetting `optimizer.zero_grad()` is the single most common silent training bug.

### Only leaves get gradients

In PyTorch, `x.grad` is populated for tensors with `requires_grad=True` that are leaves. Intermediate results also receive gradients during the walk but are freed once consumed — unless you call `.retain_graph(True)`, which is what you need when you want two backward passes over one forward pass (as in some policy-gradient setups).

### `detach` cuts the graph

`tensor.detach()` returns a tensor sharing the same storage but with no history. It is how you stop a target from receiving gradients (labels), and how you avoid backpropagating into a frozen model. `with torch.no_grad():` is the same idea applied to a block, and is why inference uses far less memory than training.

## Code

Here is the engine. Read it as three parts: the node, the two operations, and the backward pass.

```python
from __future__ import annotations

import math


class Value:
    """A scalar with a gradient and a record of how it was produced."""

    def __init__(self, data: float, parents: tuple[Value, ...] = ()) -> None:
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None      # replaced by each operation

    def __add__(self, other: Value) -> Value:
        out = Value(self.data + other.data, (self, other))
        def _backward() -> None:
            # d(a+b)/da = 1, so both parents receive out.grad unchanged
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Value) -> Value:
        out = Value(self.data * other.data, (self, other))
        def _backward() -> None:
            # d(ab)/da = b, d(ab)/db = a
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward() -> None:
            # d(tanh x)/dx = 1 - tanh(x)^2
            self.grad += out.grad * (1 - t * t)
        out._backward = _backward
        return out

    def backward(self) -> None:
        order, seen = [], set()
        def visit(v: Value) -> None:        # post-order DFS = topological order
            if v in seen:
                return
            seen.add(v)
            for p in v._parents:
                visit(p)
            order.append(v)
        visit(self)
        self.grad = 1.0                     # seed: dL/dL = 1
        for v in reversed(order):           # children before parents
            v._backward()
```

Now use it. This is one neuron with two inputs and a tanh, trained on one example by hand — the same three lines a real training loop performs.

```python
# forward: x1*w1 + x2*w2 + b, then tanh, target is 1.0
x1, x2 = Value(2.0), Value(0.0)
w1, w2 = Value(-3.0), Value(1.0)
b = Value(6.8813735870195432)

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
loss = (o + Value(-1.0)) * (o + Value(-1.0))    # squared error to target 1.0

loss.backward()
print(round(loss.data, 6), round(w1.grad, 6), round(b.grad, 6))
# 0.085786 -0.585786 -0.292893

# verify against PyTorch: the numbers must match to float precision
import torch
t = torch.tensor
tw1 = t(-3.0, requires_grad=True)
tout = (t(2.0) * tw1 + t(0.0) * t(1.0) + t(6.8813735870195432)).tanh()
((tout - 1.0) ** 2).backward()
print(round(float(tw1.grad), 6))                # -0.585786  (same)
```

## Real use case

A team fine-tunes a small model for classification and freezes the encoder to save memory. They set `encoder.requires_grad_(False)`, train, and find the encoder's weights have *changed* anyway — accuracy collapses after a few epochs.

The cause is that they created the optimizer before freezing, so the frozen parameters were still in the optimizer's parameter group; with `.grad` left over from earlier steps, the optimizer keeps applying stale updates. Seeing the engine's mechanics makes the fix obvious: check that `param.grad is None` after a backward pass for frozen params, and rebuild the optimizer from `[p for p in model.parameters() if p.requires_grad]`.

Before they understood the graph, the bug looked like a data problem and cost a week. After writing a forty-line engine, it took an afternoon.

## Common mistakes

- **Mistake:** assigning `grad =` instead of `grad +=` in a backward closure. — **Instead:** accumulate; a value used twice gets two contributions.
- **Mistake:** calling `backward()` twice on one graph without `retain_graph=True`. — **Instead:** the graph is freed after the first pass; either retain it or restructure.
- **Mistake:** forgetting `optimizer.zero_grad()`. — **Instead:** clear at the top of every step, before `backward()`.
- **Mistake:** wrapping parameters in `.detach()` and expecting them to learn. — **Instead:** detach labels and frozen activations, never learnable weights.
- **Mistake:** treating a `None` gradient as a framework bug. — **Instead:** it means no path from the loss to that tensor; check `requires_grad` and the graph.

## Checkpoint

1. Add a `__pow__` to the engine so `x ** 2` works, and check the gradient against the expanded form `x * x`.
2. Add `relu()` and confirm that for a negative input the gradient is exactly zero — then explain why that can stall training.
3. Extend the engine to work on lists of floats instead of scalars, i.e. a tiny tensor. What breaks first, and why is that the reason real frameworks use vectorised kernels?

**Self-check:** `x ** 2` has local derivative `2x`, so the closure adds `out.grad * 2 * self.data`; for `x = 3` a gradient of 1.0 arrives as 6.0, which matches `x * x` where each factor receives `1.0 * 3`.

## Go deeper

- [3. Anatomy of a Training Loop](03-training-loops.md) — where `zero_grad` and `backward` sit in the real loop.
- [6. Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md) — the calculus underneath this code.
- [4. Self-Attention From Scratch](../06-llm-fundamentals/04-self-attention.md) — the same chain rule, applied to a transformer block.
- [3. Optimizers](../04-deep-learning/03-optimizers.md) — what happens to these gradients next.

---
[← Previous: PyTorch: Tensors and Modules](01-pytorch.md) · [Stage index](README.md) · [Next: Anatomy of a Training Loop](03-training-loops.md)
