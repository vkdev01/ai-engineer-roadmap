# 1. Supervised Fine-Tuning (SFT)

> **Stage 13 · Fine-Tuning** — turns a base model that completes text into a model that answers requests.
>
> **You'll be able to:** build a masked, chat-formatted training set and run a correct SFT loop whose loss reflects only the tokens you want the model to produce.
>
> **Prerequisites:** [2. Tokenization](../06-llm-fundamentals/02-tokenization.md) · [6. Instruction Tuning](../06-llm-fundamentals/06-instruction-tuning.md)

## Why this exists

A base model predicts the next token of whatever text it is given. Give it `What is the capital of France?` and it may continue with `What is the capital of Germany? What is the capital of Spain?` — a perfectly reasonable continuation of a list of questions, and a useless answer.

SFT fixes this by training on pairs: a prompt in the exact format users will send, and the answer you want. The model already knows facts and grammar; you are teaching it a *shape* — "when a user message looks like this, produce an answer in this style, this length, this format."

The mental model: **SFT is a format and behaviour transplant, not a knowledge transplant.** If your model does not know something, fine-tuning on a hundred examples will teach it to sound confident about it, not to know it. Knowledge belongs in retrieval or pretraining; SFT moves behaviour.

## The mental model

```text
tokens:  <|user|> Refund a duplicate charge <|end|> <|assistant|> I can help ...
labels:    -100      -100  -100   -100  -100      -100       I can help ...
           ^--------------------------- masked -----------------^  ^-- loss here --^
```

| Hyperparameter | What it controls | Symptom when wrong |
|---|---|---|
| learning rate | step size | too high: model rambles or forgets English |
| epochs | passes over data | too many: memorises the eval set |
| batch size | gradient noise | too small: unstable; too large: needs more LR |
| max sequence length | truncation | too short: answers get cut mid-sentence |
| prompt masking | where loss lands | unmasked: model learns to write your prompts |

## Core concepts

### The loss is next-token cross-entropy, applied where you point it

At every position the model outputs a probability for each token in the vocabulary. The loss is the negative log probability assigned to the token that actually came next, averaged over positions. In SFT you set the label to `-100` on every prompt token; PyTorch's `CrossEntropyLoss` ignores `-100` by default. So a 400-token example where 60 tokens are the answer contributes 60 loss terms, not 400.

Why this matters: if you skip masking, roughly 85% of your gradient signal teaches the model to generate user questions. It learns to answer its own prompts and to ramble after finishing. Masking is one argument (`train_on_inputs=False`) in most libraries, and the single most common silent bug in hand-rolled SFT.

### The chat template must match inference exactly

A chat template is a string pattern that marks roles and turn boundaries, for example `<|im_start|>user ... <|im_end|>`. Every model family ships its own. You must use the *same* template in training and at inference, because the model learns the delimiters as strongly as the content. Training with one template and serving with another produces a model that ignores instructions — it has never seen your delimiters.

Use the tokenizer's own template rather than string-building it: `tokenizer.apply_chat_template(messages, tokenize=False)` returns the exact string the model expects.

### Data volume, and what "enough" looks like

There is no universal number. The honest rule: start with a few hundred high-quality examples that cover every behaviour you need, measure, then add. Quality dominates quantity — a few hundred clean, diverse examples usually beat tens of thousands of templated near-duplicates, because duplicates teach memorisation and waste epochs. If your examples differ only in a noun, you have one example.

### Hyperparameters that actually move the needle

Rank the knobs by effect: learning rate, then epochs, then sequence length, then batch size. Typical full fine-tuning learning rates are in the `1e-5` to `5e-5` range; LoRA tolerates `1e-4` to `3e-4` because it updates far fewer parameters. Two or three epochs is the usual starting point; watch eval loss and stop when it turns up. Treat all these as starting points to tune, not constants.

## Code

First, the data side: build a chat-formatted example and mask the prompt tokens. Print `sum(labels != -100)` to confirm only the answer is supervised.

```python
from __future__ import annotations

from transformers import AutoTokenizer

MODEL_ID = "some-small-open-model"  # check the model card for the current id

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def build_example(system: str, user: str, answer: str) -> dict[str, list[int]]:
    """Tokenise a single turn and mask everything before the assistant reply."""
    prompt_msgs = [{"role": "system", "content": system},
                   {"role": "user", "content": user}]
    # Tokenise prompt and answer separately, then concatenate. This is the
    # clearest way to get the mask right; it avoids fragile offset arithmetic.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids   # -100 = ignored by the loss
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

ex = build_example(
    system="You are a support agent. Answer in one short paragraph.",
    user="My card was charged twice for order 4411.",
    answer="I can see two charges on order 4411. I have started a refund for the duplicate; it lands in 5-7 working days.",
)
print(len(ex["input_ids"]), sum(t != -100 for t in ex["labels"]))
# prompt tokens are NOT supervised; only the answer tokens are.
```

Then the training call. `SFTTrainer` from TRL is the standard wrapper; `DataCollatorForCompletionOnlyLM` is the alternative if you use plain `Trainer`. Note `train_on_inputs=False`, which is the masking switch.

```python
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

rows = [
    {"messages": [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is order 4411?"},
        {"role": "assistant", "content": "It left the warehouse yesterday and is due Thursday."},
    ]},
    # ... a few hundred more, covering every behaviour you need
]

cfg = SFTConfig(
    output_dir="out/sft-run1",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch = 16
    learning_rate=1e-4,                 # LoRA-scale LR; lower for full FT
    max_length=1024,
    train_on_inputs=False,              # mask the prompt -> loss on answers only
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to=[],
)

trainer = SFTTrainer(
    model=MODEL_ID,
    args=cfg,
    train_dataset=Dataset.from_list(rows[:400]),
    eval_dataset=Dataset.from_list(rows[400:]),
)
trainer.train()
```

## Real use case

A logistics company runs a support inbox. Agents paste customer emails into a general chat model and get long, hedged answers in a house style nobody recognises; the model invents policy details such as refund windows. Prompting with a 900-word system prompt helps, but it costs tokens on every request and the model still drifts when an email is long.

They collect 600 real agent replies that were approved by a team lead, strip names and order ids into placeholders, and SFT a small open model with masking and the model's own chat template. The system prompt shrinks to one line because the style now lives in the weights. Measured on a held-out 100-email set, the fine-tune matches the house style far more often than the prompted base model in a blind pairwise review, and — the part that matters — it stops inventing refund windows, because the 600 examples all state the real policy. The failure before was not "the model was too small"; it was that policy and tone lived in a prompt that competed with a long email for attention.

## Common mistakes

- **Mistake:** not masking prompt tokens — **Instead:** verify with `sum(label != -100 for label in labels)` on one example before training; it should equal the answer length only.
- **Mistake:** hand-writing the chat template — **Instead:** use `tokenizer.apply_chat_template` in both training and inference, from the same tokenizer revision.
- **Mistake:** training on the same examples you report metrics on — **Instead:** split before you look at the data, and keep a locked held-out set.
- **Mistake:** ten epochs "to be safe" — **Instead:** two or three, and stop when eval loss rises.
- **Mistake:** expecting SFT to add facts — **Instead:** put facts in retrieval; use SFT for format, tone and task behaviour.

## Checkpoint

1. Take one training example and print, token by token, which tokens carry loss. Confirm the count equals the answer length.
2. Train the same 300 examples twice, once with `train_on_inputs=True`, and compare outputs on five prompts. Write down what changes.
3. Build a 40-example held-out set with at least four distinct behaviours, then check your training data covers each behaviour at least ten times.

**Self-check:** only the answer tokens are supervised because every prompt token gets label `-100`, which `CrossEntropyLoss` ignores; the loss is averaged over the remaining positions, so the model is never rewarded for reproducing your prompt.

## Go deeper

- Next: [2. LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) — the same loop, at a fraction of the memory.
- Background: [6. Instruction Tuning](../06-llm-fundamentals/06-instruction-tuning.md) and [5. Pretraining](../06-llm-fundamentals/05-pretraining.md).
- Build the eval before the trainer: [3. Golden Datasets](../15-evals/03-golden-datasets.md).

---
[← Previous: Stage 12 — Orchestration](../12-orchestration/README.md) · [Stage index](README.md) · [Next: 2. LoRA, QLoRA and PEFT](02-lora-qlora-peft.md)
