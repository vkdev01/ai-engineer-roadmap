# 6. Instruction Tuning and Chat Templates

> **Stage 06 · LLM Fundamentals** — How a text-continuation model becomes an assistant: a small set of demonstrations, and one exact string format that must match at inference.
>
> **You'll be able to:** build an SFT dataset that masks prompt tokens, apply the correct chat template, and explain why a mismatched template ruins a model.
>
> **Prerequisites:** [5. Pretraining: Objectives and Scale](05-pretraining.md) · [6. Hugging Face Transformers and Datasets](../05-dl-frameworks/06-hugging-face.md)

## Why this exists

A pretrained model asked "What is the capital of France?" is likely to reply with another question, because in web text a question is often followed by another question. It has the knowledge — it does not have the *behaviour*.

Instruction tuning, also called supervised fine-tuning or SFT, fixes this with a few thousand to a few hundred thousand examples of the behaviour you want: a prompt and a good response. The loss is the same next-token loss from [5. Pretraining](05-pretraining.md); only the data changes. But the data carries a format, and the format must be identical at training and at inference.

The mental model: *SFT teaches a format plus a behaviour, using the same objective, on a small curated dataset.*

## The mental model

```text
 one training row, rendered through a chat template:

  <|system|>You are a support agent.<|end|>
  <|user|>Where is my order?#123?<|end|>
  <|assistant|>It shipped yesterday and arrives Friday.<|end|>
  └──────────── labels = -100 ─────────────┘└───── labels = token ids ─────┘
                                              (only the answer is scored)

 at inference, the SAME template is rendered, then generation starts
 after the <|assistant|> marker.
```

The two things to get right: only the assistant's tokens contribute to the loss, and the template at inference is byte-identical to the one used in training.

## Core concepts

### The dataset is prompt/response pairs, not documents

Each row is a conversation: a system instruction, a user turn, and the desired assistant reply. Quality dominates quantity — a few thousand careful examples beat a hundred thousand scraped ones, because the model imitates what it sees, including sloppiness and hedging.

### Masking the prompt is what makes SFT work

If you compute loss over the whole sequence, the model spends most of its gradient learning to predict the *user's* text — which it will never need to do. Set the labels for prompt tokens to `-100`, which `CrossEntropyLoss` ignores by default, so only response tokens are scored. This is the difference between a model that answers and a model that rambles.

### The chat template is part of the model, not a style choice

Chat models ship a template — Jinja, in the Hugging Face convention — that turns a list of messages into the exact token sequence the model was trained on. `tokenizer.apply_chat_template(messages, tokenize=False)` renders it. Use the tokenizer's own template; do not hand-write `<|user|>` markers unless you know the model's exact special tokens, because a wrong marker is an unknown token and the model sees garbage where the turn boundary should be.

### The failure is silent

A mismatched template does not raise. The model receives a well-formed token sequence that does not look like its training distribution, so it degrades: it may answer in the wrong voice, forget the system prompt, or refuse. This is the single most common bug when serving an open-weight chat model yourself.

### Loss on the response only means variable-length rows

Padding is needed, but padded positions must also be `-100`, or the model learns to predict padding. Set both the attention mask (1 for real tokens, 0 for padding) and the labels (`-100` for prompt and padding).

## Code

Building a masked SFT example by hand first, so the mechanics are visible, then with the tokenizer's template.

```python
from __future__ import annotations

import torch
from transformers import AutoTokenizer

ckpt = "Qwen/Qwen2.5-0.5B-Instruct"     # small instruct model; check current ids
tok = AutoTokenizer.from_pretrained(ckpt)

messages = [
    {"role": "system", "content": "You are a support agent. Be brief."},
    {"role": "user", "content": "Where is order 123?"},
    {"role": "assistant", "content": "It shipped yesterday and arrives Friday."},
]

# render the model's own template, with the assistant turn still open
prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True,
                                     add_generation_prompt=True)
full_ids = tok.apply_chat_template(messages, tokenize=True)
print(len(prompt_ids), len(full_ids))          # full is longer by the reply

# labels: -100 for everything the model should not be scored on
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
assert len(labels) == len(full_ids)
print(sum(1 for x in labels if x != -100), "tokens contribute to the loss")

batch = tok.pad({"input_ids": [full_ids], "labels": [labels]},
                padding=True, return_tensors="pt")
print(batch["input_ids"].shape, batch["attention_mask"].sum().item())
```

Now the same thing over a dataset, batched, which is the shape a real SFT run needs. Note the two separate settings that must both be applied.

```python
from datasets import Dataset


def to_supervised(rows: dict[str, list]) -> dict[str, list]:
    """Turn question/answer pairs into masked, padded training tensors."""
    out_ids, out_labels, out_mask = [], [], []
    for q, a in zip(rows["question"], rows["answer"]):
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "content": a}]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=True,
                                         add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=True)
        labels = [-100] * len(prompt) + full[len(prompt):]
        out_ids.append(full)
        out_labels.append(labels)
        out_mask.append([1] * len(full))

    # pad on the RIGHT for training; pad positions are masked out of the loss
    width = max(len(ids) for ids in out_ids)
    pad = tok.pad_token_id
    out_ids = [ids + [pad] * (width - len(ids)) for ids in out_ids]
    out_mask = [m + [0] * (width - len(m)) for m in out_mask]
    out_labels = [l + [-100] * (width - len(l)) for l in out_labels]
    return {"input_ids": out_ids, "attention_mask": out_mask, "labels": out_labels}


ds = Dataset.from_dict({"question": ["Where is order 123?", "Can I get a refund?"],
                        "answer": ["Shipped yesterday, arrives Friday.",
                                   "Yes, within 30 days of delivery."]})
ds = ds.map(to_supervised, batched=True, remove_columns=ds.column_names)
sample = ds[0]
print(len(sample["input_ids"]), sum(1 for x in sample["labels"] if x != -100))
```

## Real use case

A company self-hosts a small open-weight instruct model for an internal assistant. In the vendor's playground the model is sharp and follows the system prompt. Behind their own API it ignores the system prompt entirely and answers in a chatty, generic voice.

They are building the prompt string by hand: `"System: ...\nUser: ...\nAssistant:"`. The model was trained on its own template with special tokens around each turn, so the hand-built string is out of distribution — every turn boundary is wrong at once.

The fix is one call: `tok.apply_chat_template(messages, add_generation_prompt=True)`, using the tokenizer that ships with the checkpoint. Behaviour returns to the playground's. The same bug had also corrupted their fine-tuning data, so the model they had trained was learning a format it would never see in production. Switching both sides to the template — training data and inference — fixed both at once.

## Common mistakes

- **Mistake:** computing loss over prompt tokens. — **Instead:** set them to `-100`; only the response should be scored.
- **Mistake:** hand-writing turn markers. — **Instead:** `apply_chat_template`, always, from the checkpoint's own tokenizer.
- **Mistake:** forgetting `add_generation_prompt=True` at inference. — **Instead:** render the open assistant turn, or the model continues the user's turn instead of starting its own.
- **Mistake:** padding labels with the pad id. — **Instead:** pad labels with `-100`, and the attention mask with 0.
- **Mistake:** training on a huge, noisy scrape because "more data is better". — **Instead:** curate; the model imitates tone and errors as faithfully as content.

## Checkpoint

1. Render a three-turn conversation with your tokenizer's template and print the raw string. Identify the special tokens and the exact place generation would start.
2. Compute the fraction of scored tokens for a long prompt with a short answer, and explain why that fraction affects how much the model learns per step.
3. Deliberately use a different model's template at inference and describe the degradation. Then fix it and compare.

**Self-check:** Generation starts immediately after the assistant-turn header token, with no closing marker yet — the model must produce that marker itself, which is why it appears in the training labels for the reply.

## Go deeper

- [7. RLHF and DPO](07-rlhf-dpo.md) — the stage after SFT.
- [1. SFT](../13-fine-tuning/01-sft.md) — running this at scale with LoRA.
- [2. LoRA, QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) — fine-tuning without a full GPU.
- [3. Synthetic Data](../13-fine-tuning/03-synthetic-data.md) — where to get the examples.

---
[← Previous: Pretraining: Objectives and Scale](05-pretraining.md) · [Stage index](README.md) · [Next: RLHF and DPO](07-rlhf-dpo.md)
