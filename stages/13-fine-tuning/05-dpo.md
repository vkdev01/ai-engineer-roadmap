# 5. DPO: Preference Tuning Without RLHF

> **Stage 13 · Fine-Tuning** — uses "this answer is better than that one" instead of "this is the correct answer".
>
> **You'll be able to:** build a preference dataset, explain the DPO loss in plain words, and run a DPO pass without the model collapsing into short, bland replies.
>
> **Prerequisites:** [1. Supervised Fine-Tuning (SFT)](01-sft.md) · [7. RLHF and DPO](../06-llm-fundamentals/07-rlhf-dpo.md)

## Why this exists

SFT says "produce this". Many qualities you care about are comparative, not absolute: answer A is more helpful than answer B, both are grammatical, neither is wrong. Writing a single gold answer for "explain a deductible" is impossible; ranking two attempts is easy for a human reviewer.

Classic RLHF turns rankings into a reward model and then optimises the policy against it with reinforcement learning — powerful, and fiddly: reward hacking, unstable training, several models in memory. DPO removes the RL loop. It takes pairs `(prompt, chosen, rejected)` and optimises a simple classification-style loss that makes the chosen answer more likely and the rejected one less likely, relative to a frozen reference copy of the model.

The mental model: **DPO is SFT where the label is "more like the chosen one, less like the rejected one".**

## The mental model

```text
prompt ──┬──► chosen   (preferred by a human or a stronger model)
         └──► rejected (the one we want to move away from)

loss ≈ -log sigmoid( beta * [ (logp_model(chosen) - logp_ref(chosen))
                            - (logp_model(rejected) - logp_ref(rejected)) ] )
```

| Symbol | Meaning | Practical effect |
|---|---|---|
| `logp_model` | model's log-prob of the answer tokens | what training moves |
| `logp_ref` | same, from the frozen SFT model | the anchor that stops drift |
| `beta` | how hard to push away from the reference | small: timid, large: collapse risk |

## Core concepts

### The loss in plain words

For each pair, compute how much more likely the model finds the chosen answer than the reference model does, and the same for the rejected answer. The loss is small when the model has raised the chosen answer's relative probability and lowered the rejected one's. The reference model appears because "more likely" is meaningless in absolute terms — a model can raise every probability by getting more confident overall. Subtracting the reference measures *change in preference*, not raw confidence. That subtraction is what makes DPO stable without a reward model.

### `beta` controls how far you are allowed to drift

`beta` scales the log-ratio inside the sigmoid. Large `beta` (illustratively 0.5) means small differences already saturate the sigmoid, so the model pushes hard and risks collapsing into a narrow style. Small `beta` (0.01-0.1 range is common as a starting point) means gentler movement. If your DPO model starts producing very short, safe, repetitive answers, `beta` is usually too high or you trained too long — the fastest fix is a smaller `beta` and fewer steps.

### Pairs are the whole game

Three sources, in descending order of trust: human preference on real traffic; a stronger model judging two sampled outputs with a rubric; and edits — take the model's output and the human-corrected version as `(rejected, chosen)`. That third source is the cheapest and the most aligned with what you actually want, because the correction is exactly the delta you care about. Whatever the source, the two answers should differ in the quality dimension you care about and not in length alone — otherwise you train "longer is better", which is a real and common failure.

### Keep the reference model frozen, and evaluate against it

DPO needs the SFT model twice: once as the trainable policy, once frozen as the reference. With LoRA you can keep one base in memory and treat "adapter off" as the reference, which halves memory. Always evaluate the DPO model against the SFT model on a held-out set — DPO can improve the pairwise preference while making absolute quality worse, for example by making answers terser. Check both.

## Code

Build the preference dataset and run DPO with TRL. `beta` and `num_train_epochs` are the two knobs to start with; the reference model is the same checkpoint you DPO from.

```python
from __future__ import annotations

from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "your-sft-checkpoint"   # DPO starts from YOUR SFT model, not the base
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
policy = AutoModelForCausalLM.from_pretrained(MODEL_ID)
reference = AutoModelForCausalLM.from_pretrained(MODEL_ID)  # frozen copy
reference.eval()

pairs = [
    {
        "prompt": "Explain what a deductible is to a first-time buyer.",
        "chosen": "A deductible is the amount you pay yourself before insurance "
                  "starts covering the rest. With a 10,000 deductible, a 45,000 "
                  "claim means you pay 10,000 and the insurer pays 35,000.",
        "rejected": "A deductible is a complex contractual construct involving "
                    "risk allocation and actuarial considerations.",
    },
    # ... hundreds more, differing in helpfulness, not in length alone
]

cfg = DPOConfig(
    output_dir="out/dpo-run1",
    beta=0.1,                     # start gentle; raise only if too timid
    num_train_epochs=1,           # DPO overfits fast; one pass is normal
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,           # much lower than SFT
    max_length=1024,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=policy,
    ref_model=reference,
    args=cfg,
    train_dataset=Dataset.from_list(pairs),
    tokenizer=tokenizer,
)
trainer.train()
```

Then measure what changed, not just the loss. This scores the model's preference for chosen over rejected on a held-out pair set, which is the number DPO is actually optimising.

```python
import torch

@torch.no_grad()
def pair_accuracy(model, tok, held_out: list[dict]) -> float:
    wins = 0
    for ex in held_out:
        scores = []
        for answer in (ex["chosen"], ex["rejected"]):
            ids = tok(ex["prompt"] + answer, return_tensors="pt")["input_ids"]
            logits = model(ids).logits[:, :-1]
            target = ids[:, 1:]
            lp = torch.log_softmax(logits, -1).gather(-1, target.unsqueeze(-1)).sum()
            scores.append(float(lp))
        wins += scores[0] > scores[1]          # chosen must beat rejected
    return wins / len(held_out)
```

## Real use case

A tax-filing assistant ships SFT answers that are accurate but wordy: users abandon the flow on the explanation screens. The team has three months of support tickets where an agent edited the model's draft before sending. Each edited pair is free preference data — the model's draft is `rejected`, the agent's send is `chosen` — and it encodes exactly the quality dimension that matters: shorter, more direct, no restating the question.

They run one DPO epoch on 1,800 edit pairs with `beta=0.1` and a low learning rate. Their first run at `beta=0.5` produced two-sentence answers that dropped the caveats the legal team requires, so they lowered `beta` and added a rule to the pair filter: reject any pair where the chosen answer is more than 30% shorter than the rejected one *and* drops a required disclaimer. After that, held-out pair accuracy rises and average answer length falls while the disclaimer rate stays flat. The design choice that made it work was using human edits rather than model-vs-model comparisons: the pairs carried the reviewer's intent, not a generator's guess.

## Common mistakes

- **Mistake:** running DPO on a base model with no SFT first — **Instead:** DPO from your SFT checkpoint; it sharpens a behaviour that already exists.
- **Mistake:** pairs that differ only in length — **Instead:** filter pairs so length is not a confound, or you train "longer wins".
- **Mistake:** `beta` too high and too many epochs — **Instead:** start around 0.1 and one epoch; watch for terse, repetitive answers.
- **Mistake:** reporting only DPO loss — **Instead:** report held-out pair accuracy against the SFT model plus absolute quality.
- **Mistake:** forgetting the frozen reference — **Instead:** keep a reference copy (or adapter-off) or the objective is undefined.

## Checkpoint

1. Take 20 real model outputs you have edited by hand and turn them into `(prompt, chosen, rejected)` rows. Check the length distribution of each side.
2. Compute held-out pair accuracy for your SFT model before any DPO. That is your baseline.
3. Run one DPO epoch at `beta=0.1` and `beta=0.5`, then compare answer length and pair accuracy. Write down which you would ship.

**Self-check:** the reference model is subtracted so the loss measures a change in relative preference rather than an increase in overall confidence — without it, a model could satisfy the objective by becoming uniformly more confident.

## Go deeper

- Next: [6. Evaluating a Fine-Tune Honestly](06-finetune-evals.md).
- Background: [7. RLHF and DPO](../06-llm-fundamentals/07-rlhf-dpo.md).
- Judge-based comparison: [2. LLM-as-Judge, Done Properly](../15-evals/02-llm-as-judge.md).

---
[← Previous: 4. Instruction Tuning vs Continued Pretraining](04-instruction-tuning.md) · [Stage index](README.md) · [Next: 6. Evaluating a Fine-Tune Honestly](06-finetune-evals.md)
