# 9. Transfer Learning

> **Stage 04 · Deep Learning** — standing on a model someone already paid to train
>
> **You'll be able to:** freeze, unfreeze and fine-tune a pretrained model deliberately, and choose between feature extraction and full fine-tuning.
>
> **Prerequisites:** [Embeddings](08-embeddings.md) · [Optimizers](03-optimizers.md)

## Why this exists

Training a model from scratch means learning everything at once: what a word is, what
grammar is, what the task is. That requires data and compute most teams do not have.
Transfer learning separates those: a pretrained model already knows language (or
images), and you teach it only the part it does not know — your task.

The evidence is practical rather than theoretical: a small dataset that trains a
from-scratch model to nothing will fine-tune a pretrained model to something useful.
The pretrained weights are a starting point on the loss surface that is already in a
good neighbourhood, so the same few hundred steps go much further.

Mental model: **pretraining learns features, fine-tuning learns a decision on top of
them — and sometimes adjusts the features slightly.**

## The mental model

```text
  pretrained model                        your data
  (millions of examples)                  (hundreds)
        |                                     |
        v                                     v
  +----------------------+            +------------------+
  |  frozen body         |  features  |  small head      | -> your labels
  |  (do not update)     | ---------> |  (train this)    |
  +----------------------+            +------------------+

  three strategies, in increasing cost:

  1. feature extraction   freeze all, train head only      tiny data
  2. partial fine-tune    unfreeze top N layers            medium data
  3. full fine-tune       update everything, small lr      lots of data

  the smaller the dataset, the less you unfreeze, and the lower the lr
```

## Core concepts

### Frozen means no gradient, not "small gradient"

`param.requires_grad = False` stops PyTorch from computing and storing gradients for
that parameter. It also means the optimizer must not receive it — passing a frozen
parameter to an optimizer wastes memory and can silently update it if you forget the
flag. The pattern is: freeze the body, build the optimizer over
`[p for p in model.parameters() if p.requires_grad]`.

### The head is a new randomly-initialised layer

You replace the final layer with one sized to your label space. That layer starts
random, so it produces large gradients at first — which is exactly why the body must be
frozen during the first phase. Warm up the head, then unfreeze.

### Fine-tuning needs a much smaller learning rate

Pretrained weights are already good; a large learning rate destroys them. Typical
practice is one to two orders of magnitude below pretraining rates, with warmup, and
often a lower rate for the body than for the head (discriminative learning rates). The
symptom of too high a rate is a model that gets worse before it gets better, or never
recovers.

### Unfreeze gradually, and watch validation

The standard recipe: train the head until validation loss plateaus, then unfreeze the
top few layers at a low rate, then optionally the whole body. Stop when validation loss
turns upward — with a few hundred examples that happens fast. Early stopping is not
optional here, it is the main regulariser.

### The same idea, three surfaces

Vision models: freeze a pretrained backbone, train a classifier. Language: fine-tune a
small model on instruction data (Stage 13). Embeddings: use a sentence-embedding model
as a frozen encoder and never train it at all — the most common and cheapest form of
transfer in the LLM stack. Recognising which of the three you are doing keeps the
decision clear.

## Code

The full recipe in PyTorch: freeze, replace the head, train it, then unfreeze the top
layers at a lower rate. The data here is random, so the loss barely moves — what you
should check is the final line: only 4 parameter tensors are trainable after phase 2,
because the rest of the body is still frozen.

```python
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Stand-in for a pretrained body: in real code, load from a hub or a checkpoint.
body = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())

# Phase 1: freeze the body, train a fresh head.
for p in body.parameters():
    p.requires_grad = False

model = nn.Sequential(body, nn.Linear(256, 3))     # 3 classes
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=1e-3)

X = torch.randn(512, 128)                          # (n=512, features=128)
y = torch.randint(0, 3, (512,))                    # (512,)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(3):
    for xb, yb in loader:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xb), yb)   # (32, 3) vs (32,)
        loss.backward()
        opt.step()
    print("phase 1 epoch", epoch, round(loss.item(), 4))

# Phase 2: unfreeze the last body block, lower lr for it than for the head.
for p in body[-2].parameters():                     # last Linear + GELU block
    p.requires_grad = True

opt = torch.optim.AdamW([
    {"params": body[-2].parameters(), "lr": 1e-5},  # body: slow
    {"params": model[-1].parameters(), "lr": 1e-4}, # head: faster
])
print("trainable tensors:", sum(1 for p in model.parameters() if p.requires_grad))
```

Using a frozen encoder as a feature extractor — the pattern you will meet constantly in
retrieval systems. No training at all; the model produces vectors.

```python
from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

# Check the model id against the current docs; ids change.
name = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(name)
enc = AutoModel.from_pretrained(name).eval()

texts = ["how do I reset my password", "password reset instructions"]
batch = tok(texts, padding=True, truncation=True, return_tensors="pt")  # ids (2, T)

with torch.no_grad():                        # no graph: this is inference, not training
    out = enc(**batch).last_hidden_state     # (2, T, hidden)

mask = batch["attention_mask"].unsqueeze(-1).float()          # (2, T, 1)
pooled = (out * mask).sum(1) / mask.sum(1)                    # masked mean -> (2, hidden)
pooled = torch.nn.functional.normalize(pooled, dim=-1)         # L2 -> cosine == dot
print(pooled.shape, float(pooled[0] @ pooled[1]))             # high similarity
```

## Real use case

A hospital's radiology group wanted to flag chest X-rays with a pneumothorax. They had
about 900 labelled images — far too few to train a convolutional network from scratch,
which their first attempt confirmed: validation accuracy barely moved off chance.

They started from a model pretrained on a large public image corpus, replaced the
classifier head with a two-class head, and trained only that for a few epochs. That
alone produced a usable model. Unfreezing the final convolutional block at a tenth of
the head's learning rate improved it further; unfreezing everything did not, and made
training unstable, which is the expected result at 900 examples.

The team's deployment choice came from the same file: the frozen encoder runs once per
image and its features are cached, so re-training a new head on newly labelled data
takes minutes rather than hours.

| Data size | Strategy | Risk |
|---|---|---|
| Under a few hundred | Head only | Underfits the task |
| Hundreds to a few thousand | Unfreeze top layers | Overfits without early stopping |
| Tens of thousands and up | Full fine-tune, small lr | Catastrophic forgetting |

## Common mistakes

- **Mistake:** passing frozen parameters to the optimizer — **Instead:** build the optimizer over `[p for p in model.parameters() if p.requires_grad]`.
- **Mistake:** fine-tuning at the pretraining learning rate — **Instead:** drop one to two orders of magnitude and add warmup.
- **Mistake:** unfreezing everything on day one — **Instead:** train the head first, then unfreeze gradually and watch validation.
- **Mistake:** forgetting `model.eval()` and `torch.no_grad()` for a frozen encoder — **Instead:** set both; otherwise dropout stays active and you allocate graphs you never use.
- **Mistake:** no early stopping on a small dataset — **Instead:** stop when validation loss turns; the best checkpoint is rarely the last.
- **Mistake:** mean-pooling without the attention mask — **Instead:** mask before averaging, or padding tokens leak into your embedding.

## Checkpoint

1. Run phase 1 above and count trainable parameters with `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Confirm it equals only the head's parameters.
2. Unfreeze the whole body at `lr=1e-3` instead of `1e-5` and record what happens to the loss. Explain it in one sentence.
3. Take any sentence-embedding model, embed five short texts, and check that two paraphrases score higher than an unrelated pair. Report the three similarities.

**Self-check:** with the body frozen, only the final `Linear(256, 3)` has `requires_grad=True`, so the count is `256*3 + 3 = 771` — the body contributes nothing.

## Go deeper

- [Embeddings](08-embeddings.md) — the representation you are transferring.
- [LoRA, QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) — the parameter-efficient version of full fine-tuning.
- [Hugging Face](../05-dl-frameworks/06-hugging-face.md) — where pretrained models come from.
- [Fine-tune vs RAG](../13-fine-tuning/07-finetune-vs-rag.md) — when to adapt a model at all.

---
[← Previous: Embeddings](08-embeddings.md) · [Stage index](README.md) · [Next: Ship: A Miniature Transformer](10-ship-mini-transformer.md)
