# 10. Ship: A Miniature Transformer

> **Stage 04 · Deep Learning** — the stage project: a character-level transformer you wrote, trained and sampled from
>
> **You'll be able to:** implement a full decoder-only transformer forward pass and a training loop, and prove it learned something.
>
> **Prerequisites:** [Transformers](07-transformers.md) · [Attention](06-attention.md)

## Why this exists

You have seen every piece of a transformer separately. Assembling them yourself — the
embedding table, the positional information, the block, the language-model head, the
shifted-target loss — is the difference between recognising a diagram and being able to
debug one. Everything you will later read about KV caches, context windows, prefill and
decode assumes you know what these tensors are.

The project is deliberately small: a character-level model on a text file of a few
kilobytes, trained on a CPU in minutes. Small does not mean toy — every shape and every
operation is exactly what a large model does, at a scale you can print.

Mental model: **a language model is a function from `(batch, seq)` of ids to
`(batch, seq, vocab)` of scores, trained by predicting the next id at every position.**

## The mental model

```text
  text: "the cat sat"
        |  char vocab
        v
  ids  [3, 8, 1, 5, 9, 1, 7]           (T,)
        |  embed + position
        v
  x    (B, T, d_model)
        |  N x Block   (attention across positions, mlp within each)
        v
  h    (B, T, d_model)
        |  head: d_model -> vocab
        v
  logits (B, T, vocab)
        |
        |  shift by one: input [t0..t_{n-1}] predicts [t1..t_n]
        v
  cross_entropy(logits[:, :-1], ids[:, 1:])
```

## Core concepts

### One sequence, shifted, is the whole training set

Given ids `[3, 8, 1, 5]`, the input is `[3, 8, 1]` and the target is `[8, 1, 5]`. Every
position predicts the next character, so a single sequence of length `T` gives `T-1`
training signals. No labels are needed beyond the text itself — that is why
self-supervised training scales.

### The causal mask is what makes it a language model

Without a causal mask, position 3 could attend to position 4 and simply copy the answer.
The mask sets the upper triangle of the score matrix to `-inf` before softmax, so each
position sees only itself and the past. Get this wrong and the loss goes to nearly zero
during training and sampling produces nonsense — the model learnt to copy, not predict.

### The file layout

```text
04-mini-transformer/
├── data/
│   └── corpus.txt          # any small text file; public-domain prose works well
├── mini_transformer/
│   ├── __init__.py
│   ├── model.py            # attention, Block, MiniGPT
│   ├── data.py             # char vocab, encode/decode, batch sampler
│   └── train.py            # the training loop and sampling
├── out/
│   ├── model.pt            # checkpoint: weights + vocab + config
│   └── loss.json           # loss per step, for the plot
└── README.md               # what you built, how to run it, what you observed
```

### Sample greedily first, then with temperature

Greedy sampling (`argmax`) is the easiest to reason about and shows whether the model
has learned anything at all — if greedy output is random characters, training has not
worked. Temperature divides the logits before softmax: below 1 sharpens toward the most
likely token, above 1 flattens toward randomness. Sample with temperature once greedy
output is sensible, not before.

## Code

`model.py` — attention, a pre-norm block, and the model. This is the whole
architecture; every shape is annotated.

```python
from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                                   # (B, T, C)
        q, k, v = self.qkv(x).split(C, dim=-1)              # three (B, T, C)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)    # (B, h, T, d)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        s = q @ k.transpose(-2, -1) / self.d ** 0.5         # (B, h, T, T)
        # Causal mask: position i may only see j <= i. This is the language-model part.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        s = s.masked_fill(mask, float("-inf"))
        a = torch.softmax(s, dim=-1)                        # rows sum to 1
        y = a @ v                                           # (B, h, T, d)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))     # residual branch 1
        return x + self.mlp(self.n2(x))   # residual branch 2


class MiniGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 3, max_len: int = 128) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)          # (vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)        # (max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)  # (d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape                                  # (B, T)
        pos = torch.arange(T, device=idx.device)           # (T,)
        x = self.tok(idx) + self.pos(pos)                  # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))                     # (B, T, vocab)
```

`train.py` — the loop and sampling. Run it; the loss should fall steadily and the
samples should move from noise to something resembling the corpus.

```python
from __future__ import annotations

import torch
from torch import nn

from .model import MiniGPT

text = open("data/corpus.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])              # (N,)

model = MiniGPT(vocab=len(chars))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
T, B = 64, 16                                             # context length, batch size

for step in range(1500):
    ix = torch.randint(0, len(data) - T - 1, (B,))         # (B,) start positions
    xb = torch.stack([data[i:i + T] for i in ix])          # (B, T)
    yb = torch.stack([data[i + 1:i + T + 1] for i in ix])  # (B, T) — shifted by one

    logits = model(xb)                                     # (B, T, vocab)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), yb.reshape(-1)  # (B*T, vocab) vs (B*T,)
    )
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")


@torch.no_grad()
def sample(n: int = 200, temperature: float = 0.8) -> str:
    model.eval()
    idx = torch.tensor([[stoi[text[0]]]])                 # (1, 1) prompt
    for _ in range(n):
        logits = model(idx[:, -T:])[:, -1, :] / temperature  # (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                  # (1, 1)
        idx = torch.cat([idx, nxt], dim=1)                 # grow along seq
    return "".join(itos[i] for i in idx[0].tolist())


print(sample())
torch.save({"model": model.state_dict(), "stoi": stoi,
            "config": {"d_model": 96, "n_heads": 4, "n_layers": 3}}, "out/model.pt")
```

### How to prove it works

Four checks, in order. Each produces evidence you can put in the project README.

1. **Shapes hold.** Before training, pass a `(2, 16)` batch and assert
   `logits.shape == (2, 16, vocab)`. Then assert that the model's parameter count matches
   your hand calculation, with `d = d_model`: `vocab*d + max_len*d + d*vocab` for the
   embeddings and head, `n_layers * (12*d*d + 10*d)` for the blocks (three `d*d`
   projections in attention, one `d*d` output projection, `8*d*d` in the MLP, plus
   biases and two layer norms), and `2*d` for the final norm. If it does not match, a
   layer is missing.
2. **The mask is real.** Feed a sequence, then change the *last* character only, and
   confirm the logits at earlier positions are unchanged. If they change, information is
   leaking from the future and the mask is wrong.
3. **Loss falls.** Log loss every step to `out/loss.json`. A working run shows a steady
   decline; a flat curve means a bug, usually the shift (`yb` equal to `xb`) or a
   learning rate of zero after a typo.
4. **Sampling is recognisable.** After training, greedy sampling should produce text with
   the corpus's character statistics — spaces roughly where words end, common letters
   common. It will not be grammatical with this budget, and it should not be.

## Real use case

A developer-education team wanted to explain tokenisation and context windows to new
hires without hand-waving. They built this project — a 96-dimensional, three-layer,
four-head character model trained on a few kilobytes of documentation — and used it in
onboarding.

The value was diagnostic. When a hire asked "why does the model repeat itself?", the
answer came from changing one line: raising the temperature in `sample()`, and watching
repetition turn into noise and then back. When someone asked why context length costs
memory, they printed the attention matrix for `T=64` and saw the `(T, T)` shape. And
when a hire set `n_layers=12` on a CPU and training slowed to a crawl, the shape of the
cost stopped being abstract.

## Common mistakes

- **Mistake:** forgetting to shift the target — **Instead:** use `xb` and `yb = xb` shifted by one; identical input and target teaches copying.
- **Mistake:** no causal mask — **Instead:** apply the triangular mask before softmax, then verify with check 2 above.
- **Mistake:** sampling before greedy output is sensible — **Instead:** check greedy first; temperature hides whether the model learned anything.
- **Mistake:** a batch drawn from one contiguous block — **Instead:** sample `B` random start positions so the batch is not one repeated context.
- **Mistake:** `T` larger than `max_len` — **Instead:** assert `T <= max_len`; the positional table has exactly `max_len` rows.
- **Mistake:** forgetting `model.eval()` when sampling — **Instead:** switch modes; it matters the moment you add dropout.

## Checkpoint

1. Train the model and save the loss curve. Report the loss at step 0 and at step 1500, and sample text at both.
2. Remove the causal mask and retrain. Compare the final loss and the sample. Explain the difference in one sentence.
3. Add a `n_heads=1` configuration and compare parameter counts and samples against `n_heads=4` at the same `d_model`. Which one produces more word-like output on your corpus?

**Self-check:** the forward pass maps `(B, T)` ids to `(B, T, vocab)` logits because the embedding produces `(B, T, d_model)` and the head maps the last axis from `d_model` to `vocab`, leaving batch and sequence untouched.

## Go deeper

- [Stage 05 — DL Frameworks](../05-dl-frameworks/README.md) — the same model with real training infrastructure.
- [Self-Attention](../06-llm-fundamentals/04-self-attention.md) — how this mechanism appears inside production LLMs.
- [KV Cache](../16-inference/02-kv-cache.md) — why your `sample()` loop gets slower without one.
- [Tokenization](../06-llm-fundamentals/02-tokenization.md) — replacing the character vocabulary with a subword one.

---
[← Previous: Transfer Learning](09-transfer-learning.md) · [Stage index](README.md) · [Next: Stage 05 — DL Frameworks](../05-dl-frameworks/README.md)
