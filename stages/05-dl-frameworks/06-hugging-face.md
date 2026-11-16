# 6. Hugging Face Transformers and Datasets

> **Stage 05 · DL Frameworks** — The default toolchain for every LLM task after this stage: load a pretrained model and its tokenizer, run it, and fine-tune it on your own data.
>
> **You'll be able to:** load any public model and tokenizer correctly, batch inputs with padding and attention masks, and run a training job without writing a loop.
>
> **Prerequisites:** [1. PyTorch: Tensors and Modules](01-pytorch.md) · [7. Transformers](../04-deep-learning/07-transformers.md)

## Why this exists

Before Hugging Face, using a pretrained transformer meant finding the authors' repository, their idiosyncratic data format, and their slightly different layer names. Comparing two models meant two codebases. Fine-tuning meant reimplementing the training recipe from a paper.

`transformers` standardised the interface: every model loads through `from_pretrained`, every tokenizer through `from_pretrained`, every output is a `ModelOutput` object with named fields, and `Trainer` runs the loop. `datasets` standardised data the same way — memory-mapped arrow tables with `map`, `filter` and streaming, so a 50 GB dataset does not have to fit in RAM.

The mental model: *a checkpoint id gives you three things — a config, a tokenizer, and weights — and all three must agree.*

## The mental model

```text
 checkpoint id  ("org/model-name")
        │
        ├── config.json      architecture, hidden size, layer count, vocab size
        ├── tokenizer.json   how text <-> integer ids
        └── weights          .safetensors (or .bin)

AutoTokenizer.from_pretrained(id) ──► text  -> input_ids, attention_mask
AutoModelForSequenceClassification.from_pretrained(id)
        │
        └── logits  (B, num_labels)
```

Two families of class you will use constantly: `AutoModel*` for the task (sequence classification, causal LM, token classification) and `AutoTokenizer`. `Auto` means "read the config and pick the right class", which is why the same two lines work for hundreds of architectures.

## Core concepts

### The tokenizer defines the vocabulary, so it must match the weights

Every model was trained on integer ids from one specific vocabulary. Feed it ids from a different tokenizer and you get fluent nonsense — no error, just bad output. Always load the tokenizer from the *same* checkpoint id as the model, never from a different one that looks similar.

### Padding, truncation and the attention mask

A batch is a rectangle, but your texts have different lengths. `padding=True` pads to the longest sequence in the batch; `truncation=True, max_length=N` cuts longer ones. Both are required in practice, and `padding="max_length"` with a fixed length is what you want on GPU so shapes do not change per batch.

The `attention_mask` marks real tokens (1) versus padding (0). The model uses it so padded positions do not affect attention. If you build tensors by hand and forget the mask, padded tokens silently influence predictions — the model does not crash, it just becomes worse.

### `model.eval()` plus `no_grad`, as always

Inference through a loaded model still records gradients unless you say otherwise. `with torch.no_grad():` halves memory and speeds things up. The PyTorch rules from [3. Anatomy of a Training Loop](03-training-loops.md) apply unchanged — Hugging Face does not replace them, it wraps them.

### `Trainer` is the loop you already know

`TrainingArguments` carries the settings (epochs, batch size, learning rate, evaluation strategy, output directory), and `Trainer` runs train/eval/checkpoint/save. It handles the details that are easy to get wrong for transformers: weight decay exclusion for bias and norm parameters, linear warmup and decay, fp16/bf16, gradient accumulation, and distributed launch. Reach for a manual loop when you need custom logic; start with `Trainer`.

### Datasets are lazy, and `map` is where preprocessing goes

`load_dataset` returns a memory-mapped table. `.map(fn, batched=True)` applies your tokenizer to many rows at once, which is far faster than per-row. `.set_format("torch", columns=[...])` converts columns to tensors on access. `.filter` and `.train_test_split` complete the toolkit. For datasets larger than disk-comfortable, `streaming=True` gives an iterator instead of a table.

## Code

Load a model and tokenizer, run a batched inference with correct padding and masking, and read the output as named fields.

```python
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt = "distilbert-base-uncased-finetuned-sst-2-english"   # check current ids in the docs
tok = AutoTokenizer.from_pretrained(ckpt)                  # must match the model
model = AutoModelForSequenceClassification.from_pretrained(ckpt)
model.eval()

texts = ["the parcel arrived a day early", "the parcel never arrived at all"]
batch = tok(
    texts,
    padding=True,          # pad to the longest sequence in this batch
    truncation=True,       # respect the model's max length
    max_length=128,
    return_tensors="pt",   # PyTorch tensors, not lists
)
print(batch["input_ids"].shape)      # (2, T) — T is the longest text, here ~9
print(batch["attention_mask"][0])    # 1s for real tokens, 0s for padding

with torch.no_grad():
    out = model(**batch)             # named fields, unpacked directly
probs = out.logits.softmax(dim=-1)   # (2, num_labels) -> (2, 2)
print(probs.argmax(dim=-1).tolist(), model.config.id2label)
```

Fine-tuning with `Trainer` on your own small dataset. The tokenizer call is the same; only the plumbing around it is new.

```python
import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

ckpt = "distilbert-base-uncased"        # base model; check current ids in the docs
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

rows = {"text": ["late again", "arrived early", "still waiting", "perfect"],
        "label": [1, 0, 1, 0]}
ds = Dataset.from_dict(rows).train_test_split(test_size=0.25, seed=0)

def tokenize(batch: dict[str, list]) -> dict:
    # batched: the tokenizer sees many rows at once, which is much faster
    return tok(batch["text"], truncation=True, max_length=64)

ds = ds.map(tokenize, batched=True).rename_column("label", "labels")
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

args = TrainingArguments(
    output_dir="out/distilbert-late",       # checkpoints + logs land here
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,                     # small for fine-tuning
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], tokenizer=tok).train()
```

## Real use case

A support team wants to route incoming tickets into three queues automatically. They have 4,000 historical tickets already labelled by agents — small for training from scratch, plenty for fine-tuning.

They fine-tune a small pretrained encoder with `Trainer` on a single GPU in well under an hour, and measure accuracy on a held-out split they labelled separately. The first version scores poorly on short tickets. The cause is truncation combined with a tokenizer call that dropped `attention_mask` when they batched manually for the eval pass, so padded positions leaked into the logits.

Switching the evaluation to the same `tok(...)` call as training, with `padding=True` and the mask passed through, fixes it. The design choice that made this project viable at all was using `Trainer` with a pretrained checkpoint: the team wrote no training loop, no warmup schedule and no checkpoint logic, and shipped in a week.

## Common mistakes

- **Mistake:** loading a tokenizer from a different checkpoint than the model. — **Instead:** one `ckpt` variable feeds both.
- **Mistake:** dropping `attention_mask` when batching by hand. — **Instead:** pass the whole tokenizer dict to the model with `model(**batch)`.
- **Mistake:** forgetting `truncation=True`, so one long document raises an index error. — **Instead:** always set `truncation=True` with an explicit `max_length`.
- **Mistake:** using a large learning rate like `1e-3` for fine-tuning. — **Instead:** start around `2e-5` to `5e-5` for full fine-tuning of a pretrained encoder.
- **Mistake:** running inference without `no_grad()`. — **Instead:** wrap it; memory and speed both improve.

## Checkpoint

1. Load the sentiment checkpoint above and print the tokenizer's vocabulary size and the model's `max_position_embeddings`. Explain why the two numbers are unrelated.
2. Tokenize a paragraph of 500 words with `max_length=128` and `truncation=True`, then report how many tokens were dropped and what text was lost.
3. Swap the base model for a different encoder architecture in the fine-tuning snippet and note what breaks first. That tells you which assumptions were architecture-specific.

**Self-check:** The vocabulary size is how many distinct token ids exist; `max_position_embeddings` is how many positions the model can attend over. A model can have a large vocabulary and a short context window — the two are independent.

## Go deeper

- [7. CUDA, Devices and Mixed Precision](07-cuda.md) — making these runs fit and go faster.
- [2. Tokenization](../06-llm-fundamentals/02-tokenization.md) — what the tokenizer is actually doing.
- [1. SFT](../13-fine-tuning/01-sft.md) — fine-tuning a generative model, not a classifier.
- [8. Embeddings](../04-deep-learning/08-embeddings.md) — the same encoder, used for retrieval.

---
[← Previous: Keras and the High-Level Path](05-keras.md) · [Stage index](README.md) · [Next: CUDA, Devices and Mixed Precision](07-cuda.md)
