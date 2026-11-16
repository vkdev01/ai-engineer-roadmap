# 5. Pretraining: Objectives and Scale

> **Stage 06 · LLM Fundamentals** — What a base model actually learns from trillions of tokens of raw text, and why that leaves it fluent but unhelpful.
>
> **You'll be able to:** read a pretraining objective as a loss function, explain what scale buys, and predict what a base model will and will not do.
>
> **Prerequisites:** [4. Self-Attention From Scratch](04-self-attention.md) · [3. Anatomy of a Training Loop](../05-dl-frameworks/03-training-loops.md)

## Why this exists

A base model is the output of one training run on one objective over an enormous text corpus. Understanding that objective explains most of the model's strengths and its most annoying weaknesses: it will continue your text rather than answer your question, it will confidently produce false facts, and it has no idea when to stop.

This matters to you even if you never pretrain, because everything in the next four files — instruction tuning, preference optimisation, prompting, evaluation — is a correction applied on top of a base model. If you do not know what the base learned, the corrections look arbitrary.

The mental model: *pretraining compresses the statistics of text into weights; the only thing it optimises is "predict the next token well".*

## The mental model

```text
 raw text (web, books, code, papers)
        │  tokenize -> fixed-length blocks of ids
        ▼
 ┌─────────────────────────────────────────┐
 │ objective: next-token prediction        │
 │ input:  t0 t1 t2 t3 t4                  │
 │ target: t1 t2 t3 t4 t5   (shifted by 1) │
 │ loss:   cross-entropy, averaged per token│
 └─────────────────────────────────────────┘
        │  billions of steps, many GPUs
        ▼
 base model  -> fluent continuation, no instruction-following
        │  + instruction data (file 6)
        ▼
 instruct model -> answers requests
```

## Core concepts

### The objective is cross-entropy over the next token

For each position, the model outputs a distribution over the vocabulary. The loss compares it to the actual next token: `-log p(correct_token)`. Averaged over all positions and all sequences, that single number is what gets minimised. Every capability you observe — grammar, arithmetic, translation, code — is a side effect of getting very good at this.

`exp(loss)` is **perplexity**: roughly, how many equally likely tokens the model is choosing between. Lower is better, and it is only comparable between models with the same tokenizer.

### The label shift is the implementation detail people get wrong

Inputs and labels are the *same* token sequence, offset by one. `logits[:, :-1]` predicts `labels[:, 1:]`. Get the shift wrong — forget it, or shift the wrong way — and the model is asked to predict the token it was just given. It learns to copy and the loss looks fantastic while the model is useless.

### Data quality beats data quantity past a point

Pretraining corpora are filtered, deduplicated and mixed by domain. Duplication is actively harmful: repeated documents are memorised, which inflates benchmark scores while degrading generalisation. The practical lesson for you is that the same is true at small scale — fine-tuning on near-duplicate examples produces a model that recites rather than generalises.

### Scale improves loss smoothly, which is why the recipe is stable

Loss falls predictably as you increase parameters, data and compute together. That predictability is why pretraining is an industrial process rather than an experiment: you can estimate the outcome before spending the money. It also means a base model's knowledge is bounded by its corpus — no objective can teach it a fact that is not in the data.

### What a base model cannot do

It has no notion of a request. Ask it a question and the statistically likely continuation is *another question*, because that is what follows a question in text. It also has no calibrated uncertainty: it produces the most likely token, not the most likely truth. And it does not stop — end of text is just another token to predict.

## Code

The loss, written out. This is the entire pretraining objective in fifteen lines, and running it on a toy model shows the shift concretely.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

torch.manual_seed(0)
V, d_model, T = 50, 32, 8                      # vocab, width, sequence length

emb = nn.Embedding(V, d_model)                 # toy: no attention, just a lookup
head = nn.Linear(d_model, V)                   # per-position vocabulary scores

ids: Tensor = torch.randint(0, V, (2, T + 1))  # (B=2, T+1): one extra token for the shift
x, y = ids[:, :-1], ids[:, 1:]                 # (2, 8) inputs, (2, 8) targets
print(x.shape, y.shape)

logits: Tensor = head(emb(x))                  # (B,T,d) -> (B, T, V) = (2, 8, 50)
# CrossEntropyLoss wants (N, C) logits and (N,) targets: flatten the time axis
loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
print("loss", round(float(loss), 3), "perplexity", round(float(loss.exp()), 1))

# the wrong version: no shift, so each token predicts itself
wrong = nn.functional.cross_entropy(logits.reshape(-1, V), x.reshape(-1))
print("no-shift loss", round(float(wrong), 3))   # lower, and meaningless
```

Turning a corpus into fixed-length training blocks, which is the data half of pretraining:

```python
from itertools import islice

def blocks(token_ids: list[int], block: int) -> list[list[int]]:
    """Chop a flat token stream into fixed-length windows, dropping the tail."""
    n = len(token_ids) // block
    return [token_ids[i * block:(i + 1) * block] for i in range(n)]

# documents are concatenated with an end-of-text id between them, then chopped,
# so a block may span two documents — normal, and the model learns the boundary
stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(blocks(stream, 4))     # [[1,2,3,4], [5,6,7,8]]  (9,10 dropped: partial block)

# sanity check that catches the classic bug
assert all(len(b) == 4 for b in blocks(stream, 4)), "ragged batch"
```

## Real use case

A team wants a domain model for legal text and decides to pretrain a small transformer from scratch on their contract corpus, rather than fine-tuning an existing model. Six weeks later they have a model that continues contracts convincingly and cannot answer a single question about one.

Nothing was broken. The objective did exactly what it was asked: predict the next token. Their corpus was also far too small for the model to acquire legal reasoning — pretraining buys general competence, and general competence needs volume they did not have.

The rebuild keeps the corpus but changes the target: start from an existing base model, then instruction-tune on a few thousand question/answer pairs written by their lawyers, and evaluate on held-out questions. That works because it uses pretraining for what it is good at — general language and reasoning — and adds the domain-specific *behaviour* with far less data. The team's real asset was the labelled questions, not the raw contracts.

## Common mistakes

- **Mistake:** forgetting the one-token label shift. — **Instead:** `logits[:, :-1]` against `labels[:, 1:]`, and test that loss is not suspiciously low.
- **Mistake:** comparing perplexity across models with different tokenizers. — **Instead:** only compare within one vocabulary; otherwise use task metrics.
- **Mistake:** pretraining from scratch on a small corpus and expecting reasoning. — **Instead:** fine-tune a pretrained base unless you have web-scale data and compute.
- **Mistake:** keeping duplicate documents in the corpus. — **Instead:** deduplicate; duplicates cause memorisation and inflate scores.
- **Mistake:** expecting a base model to follow instructions. — **Instead:** that is instruction tuning, covered next.

## Checkpoint

1. Run the loss snippet with and without the shift and explain the gap. Which one would you trust to detect a broken data pipeline?
2. Compute perplexity for a model that assigns uniform probability over a 50-token vocabulary. What loss value does that correspond to, and why is it a useful baseline?
3. Take a paragraph of text, tokenize it with any tokenizer, chop it into blocks of 16, and verify that concatenating the blocks reproduces the original stream exactly.

**Self-check:** The shifted version asks the model to predict the next token, which is a real task; the unshifted version asks it to reproduce the input, which is trivially learnable and produces an artificially low loss.

## Go deeper

- [6. Instruction Tuning and Chat Templates](06-instruction-tuning.md) — turning this base model into an assistant.
- [7. RLHF and DPO](07-rlhf-dpo.md) — the stage after instruction tuning.
- [1. SFT](../13-fine-tuning/01-sft.md) — the same loss, on a small curated dataset.
- [3. Optimizers](../04-deep-learning/03-optimizers.md) — what is used to minimise this loss at scale.

---
[← Previous: Self-Attention From Scratch](04-self-attention.md) · [Stage index](README.md) · [Next: Instruction Tuning and Chat Templates](06-instruction-tuning.md)
