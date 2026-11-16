# Stage 13 — Fine-Tuning

Prompting and retrieval get you most of the way. This stage covers the last mile: changing the weights. You will learn what supervised fine-tuning actually optimises, how LoRA makes it cheap enough to run on one rented GPU, how to build the data, and — most importantly — how to prove the result is better than what you started with. It sits after Stage 6 (how models are trained) and Stage 10 (RAG, the usual alternative), because you should know both before you spend money on GPUs.

**Time:** roughly 12-18 hours · **You need first:** Stage 6 — LLM Fundamentals, and Stage 5 — DL Frameworks · **You leave with:** a LoRA adapter, a dataset card, a model card and a before/after eval report

## What you will be able to do

- Format a dataset into chat-template turns and mask the prompt tokens so the loss only lands on the answer.
- Explain rank, alpha and target modules, and estimate adapter and optimiser memory before you rent a GPU.
- Generate synthetic training data with quality filters that remove the junk instead of teaching it.
- Decide between prompting, RAG and fine-tuning using a cost, latency and data-freshness table.
- Run a DPO pass on preference pairs without needing a reward model.
- Evaluate a fine-tune honestly: held-out set, general-ability regression, judge-based A/B.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Supervised Fine-Tuning (SFT)](01-sft.md) | The loss, the chat template, prompt masking, hyperparameters |
| 2 | [LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) | Low-rank maths, rank and alpha, memory budgets |
| 3 | [Synthetic Data for Fine-Tuning](03-synthetic-data.md) | Generation prompts, dedup, filters, leakage checks |
| 4 | [Instruction Tuning vs Continued Pretraining](04-instruction-tuning.md) | Which of the two you actually want |
| 5 | [DPO: Preference Tuning Without RLHF](05-dpo.md) | The DPO loss in plain words, pair construction |
| 6 | [Evaluating a Fine-Tune Honestly](06-finetune-evals.md) | Held-out sets, regression checks, judge comparisons |
| 7 | [Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md) | A decision table on cost, latency, freshness |
| 8 | [Ship: A LoRA Fine-Tune of a Small Open Model](08-ship-lora-model.md) | The full artifact: data, config, evals, model card |

## The shipping project

[08-ship-lora-model.md](08-ship-lora-model.md) — you fine-tune a small open-weight model with LoRA on a task you define, then prove it works with a held-out eval, a general-ability regression check and a written model card that states what it does not do.

## How to know you are done

- You can point at one line of your training script and say which tokens contribute to the loss, and why.
- Your training config lists rank, alpha, learning rate and target modules, and you can justify each number.
- You have a held-out split that was never used for training or for prompt tuning.
- Your before/after table shows the base model, your fine-tune and one general benchmark, not just your fine-tune.
- Your model card names the intended use, the data, and at least two known failure modes.

---
[← Back to roadmap](../../README.md) · [Next stage: Multimodal](../14-multimodal/README.md)
