# 5. LSTMs and GRUs

> **Stage 04 · Deep Learning** — gating: how recurrence bought itself a longer memory, and why it still lost
>
> **You'll be able to:** implement an LSTM and a GRU cell, explain what each gate controls, and say precisely what attention does better.
>
> **Prerequisites:** [CNNs and RNNs](04-cnns-rnns.md) · [Backpropagation and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md)

## Why this exists

A plain RNN forgets. Its hidden state is rewritten at every step, and backpropagating
through many steps multiplies the same `W_hh` over and over — if its largest singular
value is below 1, gradients shrink toward zero and the early steps learn nothing; above
1, they explode. In practice a vanilla RNN carries useful information for a few dozen
steps at most.

The LSTM's answer was a separate, additive memory channel with gates: instead of
overwriting the state, add to it, and let learned gates decide how much to add, how much
to keep and how much to expose. Addition has a derivative of 1, so the gradient flows
back through hundreds of steps without shrinking. That worked, and LSTMs ran sequence
modelling for years.

Mental model: **an LSTM is a conveyor belt (the cell state) with three valves (gates);
a GRU is the same idea with two valves and no separate belt.**

## The mental model

```text
LSTM cell at one timestep

  c_{t-1} ----------[x]--------[+]------------------> c_t
                     |  forget  |  (add)
                     |  gate    |
  h_{t-1}, x_t --> [sigmoid]    |
        |                       |
        +--> [sigmoid] -> input gate --[x]--> (candidate, tanh)
        |                                        |
        +--> [sigmoid] -> output gate --> [x] --> h_t

  f_t = sigmoid(W_f [h,x] + b_f)     keep fraction of old memory
  i_t = sigmoid(W_i [h,x] + b_i)     how much new info to write
  o_t = sigmoid(W_o [h,x] + b_o)     how much memory to reveal

GRU:  two gates (update z, reset r), one state h, no separate c
```

## Core concepts

### The cell state is the memory, the hidden state is the output

An LSTM keeps two vectors of the same size: `c` (cell state, the long-term memory) and
`h` (hidden state, what the rest of the network sees this step). `c` is updated almost
entirely by addition: `c_t = f_t * c_{t-1} + i_t * candidate`. Because the update is
additive, the gradient path back through time is a product of the forget gates rather
than a product of matrices — and a forget gate near 1 passes the gradient almost
unchanged. That single change is the whole reason LSTMs work.

### Three gates, three questions

The forget gate asks "how much of the past do I keep?". The input gate asks "how much
of this new candidate do I write?". The output gate asks "how much of what I remember
do I show right now?". Each is a sigmoid, so each output is in `(0, 1)` and acts as a
soft switch. A candidate vector, produced by `tanh`, is the new information on offer.

### A GRU merges the belt and the output

The GRU drops the separate cell state and uses two gates. The update gate `z` trades
off old state against new state directly: `h_t = (1 - z) * h_{t-1} + z * h_tilde`.
The reset gate `r` decides how much past state the candidate sees. Fewer parameters,
similar quality on most tasks, and faster to train — which is why GRUs are the practical
default when you must use recurrence at all.

### Shapes are stable, which is the point

All gates are computed from the concatenation of `h_{t-1}` and `x_t`, so with `hidden =
H` and `input = D`, the weight matrices are `(D + H, H)` each and every gate vector is
`(batch, H)`. Stack four of them for an LSTM and you get `(D + H, 4H)`, usually computed
as one matrix multiply and split. That single fused multiply is why real LSTM
implementations look different from the textbook diagram.

### Why they were replaced

Two reasons, and neither is about accuracy alone. **Sequential dependency:** step `t`
needs step `t-1`, so you cannot parallelise across a sequence, and GPUs sit idle. **A
fixed-size bottleneck:** everything the model knows about the past must fit into `h`,
so a 2,000-token document is compressed into one vector. Attention fixes both: every
position attends to every other in one parallel matrix operation, and no single vector
has to hold the whole past. LSTMs survive where sequences are short, latency matters and
the model must be tiny.

## Code

An LSTM cell in NumPy with the fused gate matrix, so the shapes are visible. You should
see `(batch, hidden)` for both outputs and a comment-matching shape on every line.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
batch, D, H = 4, 8, 16                 # batch, input dim, hidden dim

x_t = rng.normal(size=(batch, D))       # (4, 8)
h_prev = np.zeros((batch, H))           # (4, 16)
c_prev = np.zeros((batch, H))           # (4, 16)

# One fused matrix for all four gates: (D+H, 4H).
W = rng.normal(size=(D + H, 4 * H)) * 0.1
b = np.zeros(4 * H)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def lstm_step(x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    z = np.concatenate([x_t, h_prev], axis=1) @ W + b     # (4, 24)@(24, 64) -> (4, 64)
    i, f, o, g = np.split(z, 4, axis=1)                   # four (4, 16) blocks
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = np.tanh(g)                                        # candidate
    c_t = f * c_prev + i * g                              # additive memory update
    h_t = o * np.tanh(c_t)                                # what we expose
    return h_t, c_t


h, c = lstm_step(x_t, h_prev, c_prev)
print(h.shape, c.shape)     # (4, 16) (4, 16)
```

A GRU cell, for comparison — two gates and one state. Note that the reset gate is
applied to `h_prev` *inside* the candidate, not to the final state.

```python
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(1)
batch, D, H = 4, 8, 16
x_t = rng.normal(size=(batch, D))     # (4, 8)
h_prev = np.zeros((batch, H))         # (4, 16)

W_z = rng.normal(size=(D + H, H)) * 0.1    # update gate
W_r = rng.normal(size=(D + H, H)) * 0.1    # reset gate
W_n = rng.normal(size=(D + H, H)) * 0.1    # candidate
b_z = np.zeros(H); b_r = np.zeros(H); b_n = np.zeros(H)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def gru_step(x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
    xh = np.concatenate([x_t, h_prev], axis=1)        # (4, 24)
    z = sigmoid(xh @ W_z + b_z)                       # (4, 16) update gate
    r = sigmoid(xh @ W_r + b_r)                       # (4, 16) reset gate
    xh_r = np.concatenate([x_t, r * h_prev], axis=1)  # reset applied to past state
    n = np.tanh(xh_r @ W_n + b_n)                     # (4, 16) candidate
    return (1 - z) * h_prev + z * n                   # convex blend, not an add


print(gru_step(x_t, h_prev).shape)   # (4, 16)
```

## Real use case

A telemedicine triage tool summarised each patient's symptom history before the visit.
The history is a sequence of free-text notes, and the first version used a plain RNN
over note embeddings. It worked for recent notes and ignored anything older than about
ten notes — a chronic condition mentioned at the start of a long history simply did not
influence the output, and clinicians noticed.

Replacing the RNN with a two-layer LSTM extended the useful window enough to fix most
cases, because the cell state carried the early notes forward additively. The remaining
failures were documents where the relevant note sat in the middle of a very long history,
and the team eventually moved to an attention-based encoder — not because the LSTM was
wrong, but because a single hidden vector cannot hold a long history without loss, and
attention lets the model look back at the specific note it needs.

| Requirement | LSTM/GRU | Attention |
|---|---|---|
| Short sequences, tiny model | Good fit | Overkill |
| 2,000-token context | Degrades | Designed for it |
| Parallel training | No | Yes |
| Streaming, one token at a time | Natural | Needs a cache |

## Common mistakes

- **Mistake:** concatenating in the wrong order — **Instead:** pick `[x_t, h_prev]` and keep it consistent; the weight matrix's row order must match.
- **Mistake:** using `tanh` for the gates — **Instead:** gates are sigmoids (values in `(0,1)`); `tanh` gives `(-1, 1)` and breaks the "fraction to keep" reading.
- **Mistake:** resetting `c` and `h` between unrelated sequences — **Instead:** zero them per batch; leaking state across examples silently corrupts training.
- **Mistake:** expecting an LSTM to solve arbitrarily long dependencies — **Instead:** measure the useful window; if it is not enough, use attention.
- **Mistake:** initialising the forget gate bias to zero — **Instead:** a small positive initial bias (around 1) helps the memory persist early in training.
- **Mistake:** comparing LSTM and transformer parameter counts without noting the sequence-length term — **Instead:** note that attention cost grows with sequence length squared; the comparison is not a single number.

## Checkpoint

1. Run the LSTM cell for ten timesteps with random inputs and print the norm of `c_t` at each step. Does it grow, shrink or stay stable? Explain using the forget gate.
2. Set the forget gate output to a constant 0.9 (by overriding `f`) and repeat. Compare the cell-state norms to exercise 1.
3. Implement a `GRU` over a full sequence and compare its parameter count to the LSTM's for `D=8, H=16`. Confirm the ratio matches `3H` versus `4H` gate matrices.

**Self-check:** the cell state stays roughly stable because `c_t = f*c_{t-1} + i*g` is additive and the forget gate near 1 passes most of the previous state through, so the norm neither collapses nor explodes.

## Go deeper

- [Attention](06-attention.md) — the mechanism that replaced recurrence.
- [Transformers](07-transformers.md) — attention assembled into a block.
- [CNNs and RNNs](04-cnns-rnns.md) — the plain recurrence these gates were added to.
- [Context Windows](../06-llm-fundamentals/03-context-windows.md) — the modern answer to "how much history can the model use".

---
[← Previous: CNNs and RNNs](04-cnns-rnns.md) · [Stage index](README.md) · [Next: Attention](06-attention.md)
