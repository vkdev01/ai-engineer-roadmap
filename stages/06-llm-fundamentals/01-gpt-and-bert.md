# 1. GPT and BERT: Decoder-Only vs Encoder-Only

> **Stage 06 · LLM Fundamentals** — Two ways to point the same transformer at text: read it all at once, or predict the next piece.
>
> **You'll be able to:** name the architectural difference in one sentence, and choose the right family for a task instead of defaulting to whatever is popular.
>
> **Prerequisites:** [7. Transformers](../04-deep-learning/07-transformers.md) · [6. Hugging Face Transformers and Datasets](../05-dl-frameworks/06-hugging-face.md)

## Why this exists

Both GPT and BERT are stacks of the same transformer block. The difference is one mask and one training objective, and from that difference comes everything else: what the model is good at, how you prompt it, whether you can stream its output, and whether you fine-tune it or call it.

This matters practically because teams regularly reach for a chat model to do a classification job an encoder would do better, cheaper and with far less latency — or they try to generate text with an encoder and get confused by the output shape.

The mental model: *an encoder sees the whole sentence at once and produces a representation of it; a decoder sees only the past and produces a probability for what comes next.*

## The mental model

| | Encoder-only (BERT family) | Decoder-only (GPT family) |
|---|---|---|
| Attention | bidirectional — every token sees every token | causal — token `t` sees only `≤ t` |
| Trained on | masked language modelling: fill in blanks | next-token prediction |
| Output | one vector per token, or one for the whole input | a distribution over the vocabulary, per position |
| Good at | classification, NER, retrieval embeddings, reranking | generation, chat, code, reasoning, tool calls |
| Cost shape | one forward pass, fixed length | one pass per generated token |
| Typical use | `AutoModelForSequenceClassification` | `AutoModelForCausalLM` |

Encoder-decoder models (the T5 family) are a third option: encode the input bidirectionally, then decode the output causally. Use them for translation and summarisation where the input and output are clearly separate sequences.

## Core concepts

### The causal mask is the whole trick

A decoder applies a mask so position `t` cannot attend to `t+1`. Without it, next-token prediction would be trivially solvable by copying the answer. The mask is an upper-triangular matrix of `-inf` added to the attention scores before softmax, so forbidden positions get probability zero.

This is also why generation is sequential: each new token needs a forward pass, because it could not have been seen earlier. That is the origin of everything in [Stage 16 — Inference](../16-inference/README.md).

### Masked language modelling makes the encoder bidirectional

BERT is trained by hiding about 15% of tokens and predicting them from both sides. Predicting a blank requires the left *and* right context, so no mask is needed. The side effect is a model that produces excellent sentence representations but cannot generate coherently — ask it to continue text and you get noise.

### `AutoModel` picks the head, not the body

The transformer body is the same; the *head* on top decides the task. `...ForSequenceClassification` puts a linear layer on the pooled output; `...ForCausalLM` puts a linear layer over the vocabulary at every position and shifts the labels by one. Loading the wrong head is a common cause of shape errors — `logits` of `(B, T, V)` where you expected `(B, num_labels)`.

### Choosing, in one table you can act on

- Need a label for a piece of text? Encoder.
- Need an embedding for search? Encoder (see [1. Embeddings](../09-vector-databases/01-embeddings.md)).
- Need text out? Decoder.
- Need a fixed transformation of one sequence into another? Encoder-decoder.
- Need it hosted behind an API and nothing else? It is almost certainly a decoder.

### The practical cost difference

An encoder does one forward pass and returns. A decoder does one forward pass per output token, each re-reading the whole context (mitigated by the KV cache, covered in [2. KV Cache](../16-inference/02-kv-cache.md)). For a classification job over 10,000 documents, that is the difference between 10,000 forward passes and 10,000 × output-length forward passes.

## Code

Load both families and observe the shape difference directly. This is the fastest way to make the table above concrete.

```python
from __future__ import annotations

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

text = "the invoice was paid on time"

# ---- encoder-only: one label per input -------------------------------
enc_id = "distilbert-base-uncased-finetuned-sst-2-english"   # check current ids
enc_tok = AutoTokenizer.from_pretrained(enc_id)
enc = AutoModelForSequenceClassification.from_pretrained(enc_id).eval()
enc_batch = enc_tok(text, return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    enc_out = enc(**enc_batch)
print("encoder logits", enc_out.logits.shape)        # (1, 2): one label per INPUT
print("last_hidden", enc_out.logits.shape, enc_batch["input_ids"].shape)  # (1, T)

# ---- decoder-only: one distribution per position ---------------------
dec_id = "gpt2"                                        # small; check current ids
dec_tok = AutoTokenizer.from_pretrained(dec_id)
dec = AutoModelForCausalLM.from_pretrained(dec_id).eval()
dec_batch = dec_tok(text, return_tensors="pt")
with torch.no_grad():
    dec_out = dec(**dec_batch)
V = dec.config.vocab_size
print("decoder logits", dec_out.logits.shape)        # (1, T, V): one per POSITION

# next-token distribution from the final position only
next_id = int(dec_out.logits[0, -1].argmax())
print("next token:", dec_tok.decode([next_id]))       # a plausible continuation
```

Now the mask itself, in ten lines. Seeing the matrix makes the causal claim obvious.

```python
import torch

T = 4
# additive causal mask: 0 where allowed, -inf where forbidden
mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
print(mask)
# tensor([[0., -inf, -inf, -inf],
#         [0.,   0., -inf, -inf],
#         [0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.]])

scores = torch.randn(T, T)          # stand-in for Q @ K^T / sqrt(d)
masked = scores + mask              # (T, T); row t only sees columns <= t
print(masked.softmax(dim=-1)[0])    # row 0 puts all mass on position 0
```

## Real use case

A bank needs to triage 40,000 customer complaints a month into twelve categories, with an audit trail for each decision. The first version sends each complaint to a hosted chat model with a prompt asking for the category.

It works, but latency is seconds per ticket, cost scales with output length, and — worse for audit — the model occasionally writes an explanation instead of a label, so the parser fails on roughly one ticket in fifty.

The rebuild uses a small encoder fine-tuned on 3,000 labelled complaints. One forward pass per ticket, logits over twelve classes, `argmax` for the label and `softmax` for a confidence score. Latency drops to milliseconds, cost becomes predictable, and the output shape is fixed by the architecture rather than by prompt discipline. Complaints below a confidence threshold still go to a human.

The chat model was not wrong — it was being used to compute a fixed function with a tool designed for open-ended generation.

## Common mistakes

- **Mistake:** using a chat model for classification when an encoder would do. — **Instead:** fine-tune a small encoder; fixed output shape, far lower latency.
- **Mistake:** expecting a BERT-family model to continue text. — **Instead:** it was never trained to; use a decoder for generation.
- **Mistake:** removing the causal mask "to see more context" in a decoder. — **Instead:** you break the training objective; the model will produce degenerate text.
- **Mistake:** reading `(B, T, V)` logits as `(B, num_labels)`. — **Instead:** check `model.config.architectures` and the head you loaded.
- **Mistake:** assuming bidirectional attention means a decoder has no context limit. — **Instead:** the window is a separate, fixed property; see [3. Context Windows](03-context-windows.md).

## Checkpoint

1. Run the encoder and decoder snippets above and write down both output shapes with the axis meanings.
2. Take the causal mask code and confirm that position 2's attention weights sum to 1 while ignoring positions 3 and beyond. Then explain what the softmax does with the `-inf` entries.
3. Load an encoder-decoder model and run the same sentence through it. What is the output shape, and why does it not match either family above?

**Self-check:** The encoder returns `(1, 2)` — one logit pair for the whole input — while the decoder returns `(1, T, V)` — a distribution at every position. The extra axis exists because each position predicts its own next token.

## Go deeper

- [2. Tokenization: BPE and Byte-Level](02-tokenization.md) — the units behind that vocabulary axis.
- [4. Self-Attention From Scratch](04-self-attention.md) — the mechanism both families share.
- [5. Pretraining: Objectives and Scale](05-pretraining.md) — what these objectives actually teach.
- [1. Embeddings](../09-vector-databases/01-embeddings.md) — where encoder models earn their keep in RAG.

---
[← Previous: Stage 05 — DL Frameworks](../05-dl-frameworks/README.md) · [Stage index](README.md) · [Next: Tokenization: BPE and Byte-Level](02-tokenization.md)
