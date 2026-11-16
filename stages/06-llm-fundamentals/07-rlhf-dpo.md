# 7. RLHF and DPO

> **Stage 06 · LLM Fundamentals** — Teaching a model which of two answers people prefer, without needing a human in the loop for every update.
>
> **You'll be able to:** explain the RLHF pipeline end to end, implement the DPO loss, and build a preference dataset with the right fields.
>
> **Prerequisites:** [6. Instruction Tuning and Chat Templates](06-instruction-tuning.md) · [5. Loss Functions](../02-math-and-ml-basics/05-loss-functions.md)

## Why this exists

SFT teaches a model to imitate good answers. It cannot teach the difference between two *plausible* answers where one is better — more helpful, less preachy, better formatted, less likely to invent a citation. Writing the ideal answer by hand is expensive and often impossible; ranking two attempts is cheap and fast.

Preference learning turns rankings into a training signal. The original pipeline, RLHF, trains a reward model on human comparisons and then optimises the language model against that reward with reinforcement learning. DPO reaches a similar place with a simpler objective: a classification loss directly on pairs, no reward model, no RL loop.

The mental model: *SFT says "do this"; preference optimisation says "of these two, do more of the one people liked".*

## The mental model

```text
 RLHF (three models, two stages)          DPO (two models, one stage)
 ───────────────────────────────          ────────────────────────────
 pairs (chosen, rejected)                 pairs (chosen, rejected)
        │                                        │
        ▼                                        ▼
 train reward model r(x, y)               policy  π_θ   (trainable)
        │                                 reference π_ref (frozen copy)
        ▼                                        │
 PPO: maximise r − β·KL(π ‖ π_ref)               ▼
                                          loss: push log-ratio gap up
        ▲                                        │
        └── needs sampling, rollouts,            └── plain supervised step
            reward hacking risk
```

Both include a **reference model** for the same reason: an unconstrained optimiser will happily drift into degenerate text that scores well. The KL term (RLHF) and the reference log-ratio (DPO) both keep the model close to where it started.

## Core concepts

### The data is triples, and that is the whole input

One row: a prompt, a `chosen` response, a `rejected` response. Nothing else. Annotators compare, they do not write. This is why preference data scales better than SFT data — the task is easier, so quality is more consistent and disagreements are visible.

### RLHF needs a reward model because PPO optimises a number

A reward model is trained on the same pairs with a Bradley-Terry style objective: the score of `chosen` should exceed the score of `rejected`. Then PPO maximises that score while a KL penalty keeps the policy near the reference. It works, and it is operationally heavy: reward model training, sampling, four models in memory, and a real risk of **reward hacking** — the policy finds inputs where the reward model is wrong.

### DPO removes the reward model with an algebraic trick

DPO shows that the optimal policy under that KL-constrained objective can be written in terms of the policy and reference log-probabilities. Substituting it into the preference likelihood gives a loss you can compute with two forward passes:

```text
loss = -log sigmoid( β · [ (log π(chosen) - log π_ref(chosen))
                         - (log π(rejected) - log π_ref(rejected)) ] )
```

Read the inner bracket as: *how much more did training make the chosen answer likely, relative to how much it made the rejected one likely?* DPO increases that gap. `β` controls how far the model may drift from the reference — small `β` allows more drift.

### Log-probabilities are summed over response tokens only

The score of a response is the sum of per-token log-probabilities of its tokens, given the prompt. Prompt tokens are not scored, exactly as in SFT. Averaging instead of summing, or including prompt tokens, changes the scale of the loss and silently changes what `β` means.

### What preference tuning does and does not fix

It reliably changes *style and preference*: verbosity, formatting, refusal calibration, tone. It does not add knowledge. If the model does not know a fact, no preference pair will teach it — that is a data problem, and the answer is retrieval or fine-tuning, not DPO.

## Code

The DPO loss, written from the formula. Every tensor is annotated; the important detail is the masked sum over response tokens.

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_logprob(logits: Tensor, labels: Tensor) -> Tensor:
    """logits: (B, T, V); labels: (B, T) with -100 at prompt/pad. -> (B,)"""
    # shift: position t predicts token t+1
    logits = logits[:, :-1, :]                    # (B, T-1, V)
    targets = labels[:, 1:]                       # (B, T-1)
    logp = F.log_softmax(logits.float(), dim=-1)  # fp32 for stable log-probs
    # gather the log-prob of the actual token at each position
    tok_logp = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (targets != -100).float()              # (B, T-1): score responses only
    return (tok_logp * mask).sum(dim=-1)          # (B,)


def dpo_loss(policy_chosen: Tensor, policy_rejected: Tensor,
             ref_chosen: Tensor, ref_rejected: Tensor, beta: float = 0.1) -> Tensor:
    """All four inputs are (B,) sequence log-probs."""
    # how much training raised the chosen answer's likelihood over the reference
    chosen_gap = policy_chosen - ref_chosen
    rejected_gap = policy_rejected - ref_rejected
    logits = beta * (chosen_gap - rejected_gap)   # (B,)
    return -F.logsigmoid(logits).mean()           # minimise


B, T, V = 2, 6, 100
policy_chosen = torch.tensor([-8.0, -12.0])
policy_rejected = torch.tensor([-9.0, -11.5])
ref_chosen = torch.tensor([-8.5, -12.5])
ref_rejected = torch.tensor([-8.6, -12.2])
print(round(float(dpo_loss(policy_chosen, policy_rejected,
                          ref_chosen, ref_rejected)), 4))     # a small positive loss

# a policy that already prefers the rejected answer has a larger loss
print(round(float(dpo_loss(policy_rejected, policy_chosen,
                          ref_chosen, ref_rejected)), 4))
```

One training step, showing the four forward passes and which model gets gradients.

```python
def dpo_step(policy, ref, batch: dict[str, Tensor], beta: float = 0.1) -> Tensor:
    # two passes through the policy (chosen, rejected), two through the frozen ref
    p_chosen = sequence_logprob(policy(**batch["chosen"]).logits, batch["chosen"]["labels"])
    p_rejected = sequence_logprob(policy(**batch["rejected"]).logits, batch["rejected"]["labels"])
    with torch.no_grad():                       # reference never updates
        r_chosen = sequence_logprob(ref(**batch["chosen"]).logits, batch["chosen"]["labels"])
        r_rejected = sequence_logprob(ref(**batch["rejected"]).logits, batch["rejected"]["labels"])
    return dpo_loss(p_chosen, p_rejected, r_chosen, r_rejected, beta)


# training loop shape:
#   loss = dpo_step(policy, ref, batch)
#   opt.zero_grad(); loss.backward(); opt.step()
#   every N steps: check the implicit reward margin on a held-out pair set
```

## Real use case

A team ships a coding assistant and gets consistent feedback that answers are correct but exhausting: three paragraphs of preamble before the code. SFT on "good answers" does not fix it, because their good answers already contained the preamble — the annotators wrote in that voice.

They collect 3,000 preference pairs where the chosen answer is the same content with the preamble removed and the code first, and run DPO with a small `β` against the SFT model as reference. Within one epoch the preamble shrinks; the code quality is unchanged, which they verify by running the generated code against unit tests rather than by reading it.

Their first attempt failed because they scored the whole sequence, including the prompt, so long prompts dominated the loss and the model drifted on unrelated behaviour. Masking to response tokens only, as above, fixed it. The lesson: preference tuning is a scalpel for style, and its precision depends entirely on what you score.

## Common mistakes

- **Mistake:** including prompt tokens in the log-probability. — **Instead:** mask to response tokens, as in SFT.
- **Mistake:** averaging token log-probs instead of summing. — **Instead:** sum, so length differences are reflected; otherwise long answers are systematically favoured.
- **Mistake:** forgetting the frozen reference model. — **Instead:** keep a copy with `requires_grad=False`; the loss is defined relative to it.
- **Mistake:** using DPO to teach facts. — **Instead:** add the knowledge with retrieval or SFT; preference pairs only shift behaviour.
- **Mistake:** tuning `β` blindly. — **Instead:** smaller `β` means more drift from the reference; watch held-out quality as you lower it.

## Checkpoint

1. Compute the DPO loss by hand for a pair where the policy and reference are identical. What value do you get, and why is that the correct starting point?
2. Modify `sequence_logprob` to average instead of sum, then explain which kind of answer the loss now prefers.
3. Build ten preference pairs from your own prompts, write them as JSONL with `prompt`, `chosen`, `rejected`, and write a loader that tokenises them into the batch shape `dpo_step` expects.

**Self-check:** When policy equals reference, both gaps are zero, so the logit is zero, `logsigmoid(0)` is `-log 2`, and the loss is about 0.693 — exactly the loss of a coin flip, which is what an untrained preference model is.

## Go deeper

- [6. Instruction Tuning and Chat Templates](06-instruction-tuning.md) — the stage this builds on.
- [5. DPO](../13-fine-tuning/05-dpo.md) — running this on a real model with LoRA.
- [2. LLM as Judge](../15-evals/02-llm-as-judge.md) — how to evaluate style changes at scale.
- [8. Open-Weight vs Closed Models](08-open-vs-closed.md) — whether you can do this at all with your chosen model.

---
[← Previous: Instruction Tuning and Chat Templates](06-instruction-tuning.md) · [Stage index](README.md) · [Next: Open-Weight vs Closed Models](08-open-vs-closed.md)
