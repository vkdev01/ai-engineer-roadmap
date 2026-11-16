# Stage 06 — LLM Fundamentals

Stage 05 taught you to run models. This stage explains the thing you are running. Everything that later feels like unpredictable behaviour — why the price is what it is, why the model forgets the middle of your document, why the same prompt gives different answers, why a licence forbids your use case — is decided by five mechanics covered here: architecture, tokenizer, context window, training objective, and alignment.

**Time:** 2-3 weeks · **You need first:** [Stage 05 — DL Frameworks](../05-dl-frameworks/README.md) · **You leave with:** your own tokenizer plus a token-budget analyser for prompts

## What you will be able to do

- Say what a decoder-only model does differently from an encoder-only one, and pick the right family for a task.
- Implement byte-level BPE, and explain why an ordinary word can arrive as several tokens.
- Measure a prompt in tokens, predict what will be truncated, and design around lost-in-the-middle.
- Write self-attention with the Q/K/V maths and correct shapes, without a library.
- Explain pretraining, instruction tuning and preference optimisation as three distinct stages with different data.
- Read a model licence and decide between open-weight and closed for a given product.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [GPT and BERT: Decoder-Only vs Encoder-Only](01-gpt-and-bert.md) | Which family to reach for, and why |
| 2 | [Tokenization: BPE and Byte-Level](02-tokenization.md) | The unit everything is counted and priced in |
| 3 | [Context Windows](03-context-windows.md) | What fills them and how to budget them |
| 4 | [Self-Attention From Scratch](04-self-attention.md) | The Q/K/V mechanism, written out |
| 5 | [Pretraining: Objectives and Scale](05-pretraining.md) | What the base model actually learned |
| 6 | [Instruction Tuning and Chat Templates](06-instruction-tuning.md) | How a base model becomes an assistant |
| 7 | [RLHF and DPO](07-rlhf-dpo.md) | How preferences shape behaviour |
| 8 | [Open-Weight vs Closed Models](08-open-vs-closed.md) | Licences, hosting and the real trade-offs |
| 9 | [Ship: Tokenizer and Token-Budget Analyser](09-ship-tokenizer.md) | A tokenizer you built, plus a budget tool |

## The shipping project

[09-ship-tokenizer.md](09-ship-tokenizer.md) — you train a byte-level BPE tokenizer on a corpus, then build a token-budget analyser (Python and a Node service) that reports tokens per section, finds what would be truncated, and estimates cost. You prove it works by round-tripping text losslessly and matching counts against a reference tokenizer on the same strings.

## How to know you are done

- You can look at a task and say, in one sentence, whether it is an encoder job, a decoder job, or an embedding job.
- Your own BPE tokenizer encodes and decodes any UTF-8 string back to the identical bytes.
- You can predict, before running it, roughly how many tokens a paragraph of English will cost.
- You can write scaled dot-product attention from memory with shapes annotated at every step.
- You can explain the difference between SFT and DPO data in terms of what a row contains.
- You have read at least one real model licence and can name what it restricts.

---
[← Back to roadmap](../../README.md) · [Next stage: LLM APIs](../07-llm-apis/README.md)
