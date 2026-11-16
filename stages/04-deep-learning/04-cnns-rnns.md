# 4. CNNs and RNNs

> **Stage 04 · Deep Learning** — two ways to share weights, over space and over time
>
> **You'll be able to:** explain and code a convolution and a recurrent step, with shapes annotated, and say which structure fits which kind of data.
>
> **Prerequisites:** [Feedforward Networks](02-feedforward.md) · [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md)

## Why this exists

A dense layer treats its input as a bag of independent numbers. That is wrong for
almost every kind of real data. An image has structure: a cat's ear in the top-left is
the same feature as an ear in the bottom-right, and a dense layer needs separate weights
for each position. A sentence has order: "dog bites man" and "man bites dog" have the
same words.

Both CNNs and RNNs fix this with the same trick — **weight sharing**. A convolution
slides one small filter across all positions, so a feature detector is reused
everywhere. A recurrent network applies one cell at every timestep, so the same
transformation is reused across time. Fewer parameters, and a built-in assumption about
the data's structure.

Mental model: **a CNN shares weights across space; an RNN shares weights across time.**

## The mental model

```text
CONVOLUTION (1-D, kernel 3, stride 1)          RECURRENCE (unrolled over 3 steps)
input   [a  b  c  d  e]                        x1     x2     x3
kernel  [w1 w2 w3]  slides ->                  |      |      |
        [a b c]·w = y1                          v      v      v
         [b c d]·w = y2                       [h1] -> [h2] -> [h3]
          [c d e]·w = y3                        |      |      |
output  [y1 y2 y3]                              y1     y2     y3
                                               
same w at every position ->                   same cell W_hh at every step ->
translation equivariant                       order-aware, variable length
```

## Core concepts

### Convolution: a learned sliding filter

A 2-D convolution with `C_in` input channels, `C_out` filters and kernel `K` has weight
shape `(C_out, C_in, K, K)`. Output height and width for a square input of size `H`
with padding `P` and stride `S` is `(H + 2P - K) / S + 1`. In words: each output value
is a dot product between one filter and a small patch of the input. The filter learns
to detect one pattern — an edge, a colour blob — and the same detector applies
everywhere, which is what translation invariance means in practice.

### Pooling shrinks and adds tolerance

Max pooling takes the largest value in each small window, halving the resolution. It
reduces computation and makes the representation slightly insensitive to exact
position. Modern architectures often replace pooling with a strided convolution, but
the idea — reduce spatial size as depth grows — is unchanged.

### Recurrence: one cell, applied repeatedly, with state

At each step the cell computes `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)`. Shapes:
`x_t` is `(batch, input_dim)`, `W_xh` is `(input_dim, hidden)`, `W_hh` is
`(hidden, hidden)`, so `h_t` is `(batch, hidden)`. The same matrices are used at every
step, so the parameter count does not grow with sequence length. That is the point, and
also the problem: to backpropagate through 500 steps you multiply the same matrix 500
times, and its eigenvalues decide whether the gradient vanishes or explodes.

### Both are priors, and priors can be wrong

A convolution assumes locality and translation symmetry. A recurrence assumes that a
single fixed-size state is enough to summarise everything so far. That second
assumption is the one that broke: compressing a whole document into one vector loses
detail, and doing it sequentially means step `t` waits for step `t-1`, so training
cannot parallelise over the sequence. Attention drops both assumptions — every position
looks at every other position, in parallel — and that is the subject of the next two
files.

### 1-D convolutions are still useful for text

A 1-D convolution over token embeddings with kernel 3 to 5 learns local n-gram
detectors, and stacks with dilation to widen the receptive field. It is fast, it
parallelises, and for classification tasks with local cues it remains a strong
baseline. What it cannot do is relate a pronoun at position 5 to a noun at position
400.

## Code

A 2-D convolution from scratch with NumPy, to make the shapes concrete. You should see
an output of shape `(1, 2, 3, 3)` — one image, two filters, three by three.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

x = rng.normal(size=(1, 3, 5, 5))       # (batch, C_in, H, W)
W = rng.normal(size=(2, 3, 3, 3))       # (C_out, C_in, K, K)
b = np.zeros(2)                          # (C_out,)


def conv2d_valid(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, c_in, h, w = x.shape
    c_out, _, k, _ = W.shape
    out_h, out_w = h - k + 1, w - k + 1          # "valid" convolution, no padding
    out = np.zeros((n, c_out, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, :, i:i + k, j:j + k]    # (n, c_in, k, k)
            # einsum: sum over c_in, k, k for every (n, c_out)
            out[:, :, i, j] = np.einsum("nckl,ockl->no", patch, W) + b
    return out


y = conv2d_valid(x, W, b)
print(y.shape)      # (1, 2, 3, 3)
```

A recurrent cell unrolled over a sequence, showing the state carrying forward. You
should see the hidden state change shape never — `(batch, hidden)` at every step.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
batch, seq, input_dim, hidden = 4, 6, 8, 16

X = rng.normal(size=(batch, seq, input_dim))       # (4, 6, 8)
W_xh = rng.normal(size=(input_dim, hidden)) * 0.1  # (8, 16)
W_hh = rng.normal(size=(hidden, hidden)) * 0.1     # (16, 16)
b = np.zeros(hidden)                               # (16,)

h = np.zeros((batch, hidden))                      # (4, 16) — initial state
outputs = []
for t in range(seq):
    x_t = X[:, t, :]                               # (4, 8)
    h = np.tanh(x_t @ W_xh + h @ W_hh + b)         # (4, 8)@(8,16) + (4,16)@(16,16) -> (4,16)
    outputs.append(h)

H = np.stack(outputs, axis=1)                      # (4, 6, 16) — one state per step
print(H.shape, "final state:", h.shape)
```

## Real use case

An invoice-processing product needed to find the "total amount" region on scanned
invoices. The first version fed the whole page image into a dense network, which
required a fixed image size, needed thousands of parameters per pixel position, and
broke whenever the layout shifted a few centimetres — the same total box in a different
place was, to a dense layer, a completely different input.

A small convolutional network with two convolutional layers and pooling handled it: the
filter that detects a rupee symbol is the same filter at any position, so layout
changes cost nothing. The team's second problem — invoices that span two pages — was
solved separately with a recurrent layer over page-level features, because "page 2
continues page 1" is a sequence relationship, not a spatial one.

| Property | Dense | CNN | RNN |
|---|---|---|---|
| Params scale with position | Yes | No | No |
| Order matters | No | Local only | Yes |
| Parallel over positions | Yes | Yes | No |
| Handles variable length | No | Yes | Yes |

## Common mistakes

- **Mistake:** computing output size by guesswork — **Instead:** use `(H + 2P - K) / S + 1` and write it in a comment.
- **Mistake:** using a dense layer on raw images — **Instead:** start with convolutions; the parameter count and the translation assumption both favour them.
- **Mistake:** initialising `W_hh` with large values — **Instead:** scale it small; large recurrent weights make gradients explode within a few dozen steps.
- **Mistake:** forgetting to reset the hidden state between sequences — **Instead:** initialise `h` per batch; carrying state across unrelated examples leaks information.
- **Mistake:** assuming an RNN can remember 500 steps — **Instead:** expect a useful window of tens of steps, and reach for attention when you need more.

## Checkpoint

1. Change the convolution above to use padding 1 and stride 2 and predict the output shape before running it. Confirm with the formula.
2. Train the recurrent cell above to predict the next value in a simple sequence (for example, a shifted copy of the input) using plain SGD. Note how many steps it takes and where it struggles.
3. Stack two recurrent layers and describe what the second layer's input shape is, and why it is not `(batch, input_dim)`.

**Self-check:** with `H=5`, `K=3`, `P=0`, `S=1` the output is `(5 + 0 - 3)/1 + 1 = 3` per spatial axis, so the output is `(1, 2, 3, 3)`.

## Go deeper

- [LSTMs and GRUs](05-lstms-grus.md) — gating, the fix for the vanishing gradient in a plain RNN.
- [Attention](06-attention.md) — what replaced recurrence for sequence modelling.
- [Vision-Language Models](../14-multimodal/01-vision-language.md) — where convolutions still appear in modern multimodal stacks.
- [Linear Algebra](../02-math-and-ml-basics/01-linear-algebra.md) — the matrix products underneath both structures.

---
[← Previous: Optimizers](03-optimizers.md) · [Stage index](README.md) · [Next: LSTMs and GRUs](05-lstms-grus.md)
