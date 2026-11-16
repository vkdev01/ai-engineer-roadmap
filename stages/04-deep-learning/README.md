# Stage 04 — Deep Learning

Before you can reason about an LLM, you need to have built something small that learns. This stage goes bottom-up: a perceptron that cannot solve XOR, a feedforward net that can, optimisers, the sequence models that came before transformers, attention implemented by hand, a transformer block, embeddings, and transfer learning.

The goal is not to train big models. It is to hold the shapes in your head — because when a real model misbehaves, the answer is almost always in the shapes, the loss curve or the mask.

**Time:** 3-4 weeks · **You need first:** [Stage 02 — Math and ML Basics](../02-math-and-ml-basics/README.md) · **You leave with:** a miniature transformer you wrote yourself, trained on a toy task

## What you will be able to do

- Explain why a single layer cannot learn XOR, and show it with code
- Build a feedforward network, read its loss curve and diagnose underfitting versus overfitting
- Choose an optimiser and a learning rate for a small training run
- Explain what a convolution and a recurrence each buy, and why recurrence lost
- Implement scaled dot-product attention and the causal mask from scratch
- Assemble a transformer block: attention, residual, norm, feedforward
- Reason about embeddings as geometry, and use transfer learning instead of training from zero
- Train a small character-level transformer and sample from it

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Perceptrons](01-perceptrons.md) | The linear threshold unit, and the limit that broke it |
| 2 | [Feedforward Networks](02-feedforward.md) | Hidden layers, nonlinearity, and what depth buys |
| 3 | [Optimizers](03-optimizers.md) | SGD, momentum, Adam, and picking a learning rate |
| 4 | [CNNs and RNNs](04-cnns-rnns.md) | Weight sharing over space and over time |
| 5 | [LSTMs and GRUs](05-lstms-grus.md) | Gates, vanishing gradients, and why they were replaced |
| 6 | [Attention](06-attention.md) | Soft lookup, implemented with real shapes |
| 7 | [Transformers](07-transformers.md) | The block: attention, residual, norm, MLP |
| 8 | [Embeddings](08-embeddings.md) | Meaning as geometry, and what neighbours mean |
| 9 | [Transfer Learning](09-transfer-learning.md) | Standing on a pretrained model instead of starting at zero |
| 10 | [Ship: A Miniature Transformer](10-ship-mini-transformer.md) | A working character-level model in ~100 lines |

## The shipping project

[Ship: A Miniature Transformer](10-ship-mini-transformer.md) — you implement a causal self-attention block and a two-layer transformer, train it on a small text file until the loss drops, and sample from it. It will produce almost-words, and that moment is the point: you have seen the machinery that the rest of this roadmap calls an API.

## How to know you are done

- You can state the tensor shape at every step of attention without looking
- You can look at a loss curve and say whether the model is underfitting, overfitting, or misconfigured
- You have trained the mini transformer and sampled from it
- You can explain, in one paragraph, why the causal mask exists
- You can explain why attention replaced recurrence, in terms of parallelism rather than fashion

---
[← Back to roadmap](../../README.md) · [Next stage: DL Frameworks](../05-dl-frameworks/README.md)
