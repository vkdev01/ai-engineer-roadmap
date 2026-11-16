# 1. PyTorch: Tensors and Modules

> **Stage 05 · DL Frameworks** — The two objects every PyTorch program is built from: the array that carries data, and the class that carries weights.
>
> **You'll be able to:** build, reshape and move tensors on purpose, and wrap any computation in an `nn.Module` that reports its own parameters.
>
> **Prerequisites:** [Stage 04 — Deep Learning](../04-deep-learning/README.md) · [2. Calculus](../02-math-and-ml-basics/02-calculus.md)

## Why this exists

Before frameworks, writing a network meant writing the forward pass *and* the backward pass by hand for every layer. Change one line in the forward pass and your hand-derived gradient was silently wrong — the model trained, just badly, and nothing told you.

PyTorch removed that class of bug by making arrays that remember how they were computed. A **tensor** is a NumPy array that can live on a GPU and can record its own history. A **module** is a class that owns parameters and knows how to run them forward. Almost everything in this stage, and in every fine-tuning and inference script later, is these two ideas composed.

The one-sentence mental model: *a tensor is data with a shape and a device; a module is a shape-declaring container for tensors that learn.*

## The mental model

```text
        TENSOR                              MODULE
  ┌──────────────────────┐          ┌──────────────────────────┐
  │ data: numbers        │          │ __init__: declares params│
  │ shape: (B, T, C)     │          │ forward(x): runs maths   │
  │ dtype: float32       │          │ parameters(): learnables │
  │ device: cuda:0       │          │ to(device): moves all    │
  └──────────────────────┘          └──────────────────────────┘
             │                                    │
             └──────────── both carry ────────────┘
                    requires_grad flag
```

Reading a shape out loud is a skill. `(32, 128, 768)` in a transformer means *32 sequences, 128 tokens each, 768 numbers describing each token*. When an error says `Expected 3D input`, it is telling you which of those three axes you lost.

## Core concepts

### Tensors are arrays with three extra properties

Beyond shape and dtype, PyTorch tensors carry a **device** (CPU or a GPU index) and a **`requires_grad`** flag. Operations only work between tensors on the same device; mixing CPU and CUDA tensors raises an error rather than quietly copying, which is deliberate — silent copies are how training slows to a crawl.

`torch.zeros((2, 3))` gives a float32 tensor on CPU. `torch.zeros((2, 3), device="cuda")` puts it on the GPU. `.to("cuda")` returns a *new* tensor; it does not modify in place, which is why `model.to(device)` is reassignment-safe but `tensor.to(device)` on a local variable does nothing unless you keep the result.

### Reshaping: `view` versus `permute`

`view`/`reshape` change how the same numbers are grouped. `permute` reorders axes and is what you use when a layer expects `(batch, channels, length)` and you have `(batch, length, channels)`. `permute` changes the memory layout, so it is usually followed by `.contiguous()` before a `view`. Getting this wrong produces no error, just a model that learns nothing — the numbers are right, the axes are not.

### `nn.Module` is a shape declaration, not a function

A module's `__init__` registers parameters (anything assigned as an `nn.Parameter` or another `nn.Module` is auto-registered); its `forward` defines the computation. You call `model(x)`, never `model.forward(x)` — the `__call__` wrapper runs hooks that the library depends on.

Registration is why `model.parameters()` returns exactly the learnables, and why `model.to(device)` moves everything at once. If you store a weight in a plain Python list, the optimizer will never see it.

### Parameters, buffers and plain attributes

Learnables are parameters (updated by the optimizer). Running statistics — batch-norm means, for example — are **buffers**: saved in `state_dict` but not updated by gradients. Anything else is a plain attribute and is lost when you save. That distinction matters in [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md).

## Code

A quick tour of the tensor operations you will use every day, with the shape written at each step.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

# (B=2, T=4, C=3): 2 sequences, 4 tokens each, 3 features per token
x: Tensor = torch.randn(2, 4, 3)
print(x.shape, x.dtype, x.device)          # torch.Size([2, 4, 3]) float32 cpu

# permute moves the feature axis to the middle: (B, C, T)
# needed when a 1-D conv expects channels before length
y: Tensor = x.permute(0, 2, 1).contiguous()
print(y.shape)                             # torch.Size([2, 3, 4])

# broadcasting: (2, 4, 3) + (3,) adds a bias per feature
bias: Tensor = torch.arange(3, dtype=torch.float32)
print((x + bias).shape)                    # torch.Size([2, 4, 3])

# matmul over the last two axes: (2, 4, 3) @ (3, 8) -> (2, 4, 8)
w: Tensor = torch.randn(3, 8)
print((x @ w).shape)                       # torch.Size([2, 4, 8])

# reductions keepdim=True keeps the axis so broadcasting still lines up
mean = x.mean(dim=1, keepdim=True)
print(mean.shape, (x - mean).shape)        # [2, 1, 3] [2, 4, 3]
```

Now the same computation as a module. This is a single linear layer plus a non-linearity, written the way you will write every layer after this.

```python
class TinyBlock(nn.Module):
    """Linear (C_in -> C_out) followed by GELU, applied per token."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        # assigning an nn.Module auto-registers its parameters
        self.proj = nn.Linear(c_in, c_out)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, c_in) -> (B, T, c_out); Linear acts on the last axis only
        return self.act(self.proj(x))


block = TinyBlock(3, 8)
print(sum(p.numel() for p in block.parameters()))   # 3*8 + 8 = 32
print(block(x).shape)                               # torch.Size([2, 4, 8])

# state_dict holds parameters AND buffers, keyed by dotted attribute path
print(list(block.state_dict().keys()))              # ['proj.weight', 'proj.bias']
```

## Real use case

A team building a document classifier for support tickets starts with a scikit-learn model that plateaus. They switch to a small transformer, and the first version trains but scores barely above chance. Nothing crashes; the loss wobbles near a flat value.

The bug is an axis. Their data loader returns `(batch, length, features)` but the convolutional stem expects `(batch, features, length)`, so the model is convolving along the wrong axis — every token's 3 features are being treated as a 3-step sequence. The fix is one `permute(0, 2, 1).contiguous()`.

What made it findable is that the code asserts shapes rather than trusting them: the block declares `c_in` and `c_out` in its constructor, and a two-line check prints the shape before and after each stage of the forward pass on the first batch only. Before that discipline, the same bug took two days and a rewritten data pipeline to find.

## Common mistakes

- **Mistake:** calling `model.forward(x)` directly. — **Instead:** call `model(x)`; hooks and `__call__` bookkeeping are skipped otherwise.
- **Mistake:** `x.to("cuda")` on a local and then using the old name. — **Instead:** reassign — `x = x.to(device)`.
- **Mistake:** storing a learnable weight in a plain list or attribute. — **Instead:** wrap it in `nn.Parameter` so `parameters()` and the optimizer see it.
- **Mistake:** `permute` then `view` without `.contiguous()`. — **Instead:** call `.contiguous()` between them, or use `reshape`.
- **Mistake:** reading a shape error as "my data is broken". — **Instead:** print `x.shape` at each stage of `forward` for one batch; the axis you lost is almost always obvious.

## Checkpoint

1. Create a tensor of shape `(8, 16, 32)`, take the mean over the last axis with `keepdim=True`, and subtract it from the original. What are both shapes, and why does the subtraction work?
2. Write a `nn.Module` called `MLP` with two linear layers and a ReLU, taking 32 features to 1. Print its parameter count and verify the number by hand.
3. Build a module that keeps a running mean of its inputs as a **buffer**, then confirm the buffer appears in `state_dict()` but not in `parameters()`.

**Self-check:** The shapes are `(8, 16, 1)` and `(8, 16, 32)`. Subtraction works because broadcasting stretches the size-1 axis across the 32 features, so each feature is centred by its own mean.

## Go deeper

- [2. Autograd From Scratch](02-autograd.md) — what `requires_grad` actually switches on.
- [3. Anatomy of a Training Loop](03-training-loops.md) — put modules and tensors into a loop that learns.
- [7. Transformers](../04-deep-learning/07-transformers.md) — the shapes you just learned, assembled into a real architecture.
- [1. NumPy and Pandas](../03-data-engineering/01-numpy-pandas.md) — the array model PyTorch borrows from.

---
[← Previous: Stage 04 — Deep Learning](../04-deep-learning/README.md) · [Stage index](README.md) · [Next: Autograd From Scratch](02-autograd.md)
