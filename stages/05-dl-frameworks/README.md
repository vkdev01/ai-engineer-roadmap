# Stage 05 — DL Frameworks

Stage 04 taught you what a neural network *is* on paper. This stage gives you the tools that people actually type into an editor: PyTorch for the mechanics, Keras when you want less ceremony, and Hugging Face when the model is an LLM rather than something you invented. It sits here because every later stage — fine-tuning, inference, RAG, agents — is PyTorch and Hugging Face underneath, and you cannot debug what you cannot run.

**Time:** 2-3 weeks · **You need first:** [Stage 04 — Deep Learning](../04-deep-learning/README.md) · **You leave with:** a small, reusable training-loop package with checkpoint resume and metrics logging

## What you will be able to do

- Read and write PyTorch confidently: tensors, shapes, `nn.Module`, devices.
- Explain what `loss.backward()` does by building a tiny autograd engine yourself.
- Write a correct training loop from memory — and spot the four mistakes that make one silently wrong.
- Move a model to GPU, read an out-of-memory error, and cut its memory use without guessing.
- Load a pretrained transformer and its tokenizer from Hugging Face in under ten lines.
- Ship a config-driven trainer that resumes from a checkpoint and logs metrics you can plot.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [PyTorch: Tensors and Modules](01-pytorch.md) | The two objects every PyTorch program is made of |
| 2 | [Autograd From Scratch](02-autograd.md) | How gradients are recorded and replayed, by building it |
| 3 | [Anatomy of a Training Loop](03-training-loops.md) | The seven steps that make a loop correct |
| 4 | [GPU Memory and Checkpoints](04-gpu-checkpoints.md) | Where memory goes and how to save state safely |
| 5 | [Keras and the High-Level Path](05-keras.md) | The same model in far fewer lines, and when that is right |
| 6 | [Hugging Face Transformers and Datasets](06-hugging-face.md) | The default toolchain for every LLM task after this |
| 7 | [CUDA, Devices and Mixed Precision](07-cuda.md) | Device placement, OOM triage, fp16/bf16 autocast |
| 8 | [Ship: A Reusable Training Loop](08-ship-training-loop.md) | A config-driven trainer you can reuse all roadmap |

## The shipping project

[08-ship-training-loop.md](08-ship-training-loop.md) — you build a `trainer/` package with a YAML config, a model registry, checkpoint resume and JSONL metrics. You prove it works by killing the process mid-run, restarting it, and showing that the loss curve continues from the same step rather than restarting at one.

## How to know you are done

- You can write a training loop with `model.train()`, `optimizer.zero_grad()`, `loss.backward()` and `optimizer.step()` in the right order without looking it up, and say what each one breaks if omitted.
- You can look at a tensor shape `(32, 128, 768)` and say what each axis means in a transformer batch.
- You can load a model on GPU, hit an OOM, and reduce memory by at least one deliberate lever (batch size, autocast, gradient accumulation) rather than by luck.
- Your shipped trainer resumes from a checkpoint and its metrics file has one line per step with no gaps.
- You can explain, in one paragraph, what `.backward()` walks over and why the graph is freed afterwards.

---
[← Back to roadmap](../../README.md) · [Next stage: LLM Fundamentals](../06-llm-fundamentals/README.md)
