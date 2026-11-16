# 6. Evaluating a Fine-Tune Honestly

> **Stage 13 · Fine-Tuning** — the part that decides whether the GPU money bought anything.
>
> **You'll be able to:** design a held-out evaluation for a fine-tune that resists self-deception, measure regression on general ability, and run a judge-based A/B that you can defend.
>
> **Prerequisites:** [1. Supervised Fine-Tuning (SFT)](01-sft.md) · [2. LLM-as-Judge, Done Properly](../15-evals/02-llm-as-judge.md)

## Why this exists

Fine-tuning produces the most persuasive false positives in machine learning. You read twenty outputs, they look great, you ship. What you did not check: whether they looked better than the base model's outputs on the same twenty (they might be equally good), whether the model got worse at everything else, or whether your "held-out" set shares sentences with training data.

Honest evaluation of a fine-tune answers three separate questions. **Did it learn the target task?** (task metric on a locked held-out set.) **Did it lose anything?** (general-ability regression check.) **Is it better than the alternative?** (head-to-head against the base model and against your prompt-only baseline.) A fine-tune that wins the first and fails the other two is a regression you have not noticed yet.

The mental model: **you are not measuring the model, you are measuring a delta** — always against a baseline, on data the model has never influenced.

## The mental model

```text
locked held-out set (never used for training, prompt tuning, or threshold picking)
        │
        ├── task metric      → did it learn the task?
        ├── general battery  → did it lose anything?
        └── pairwise A/B     → is it better than baseline?

baselines you must report: base model · prompt-only · your fine-tune
```

| Metric type | Example | When to prefer it |
|---|---|---|
| Exact / field match | JSON fields equal gold | structured outputs |
| Rule check | arithmetic reconciles | anything verifiable |
| Judge rubric score | 1-5 on 3 named criteria | open-ended text |
| Pairwise preference | chosen / rejected / tie | comparing two models |

## Core concepts

### The held-out set must be locked before training starts

Write it to a file, hash it, and do not touch it. Every temptation to "just fix one bad example" is prompt tuning on the test set. Split by source — by document, by customer, by seed prompt — never by row, or paraphrases of training data leak in. Size it by the decisions you need to make: a few hundred examples is usually enough to separate two clearly different models, and it is not enough to detect a one-point difference. Say that out loud in your report instead of pretending the number is precise.

### Measure regression, not just improvement

A fine-tune on a narrow task can quietly damage general instruction-following, multilingual ability, or long-context handling. Keep a small battery of general tasks you care about — twenty to fifty items across four or five capabilities, written by you, not scraped. Run it on base and fine-tune and report the delta. A drop of a few items is noise; a drop across a whole capability is a finding. This battery takes an afternoon to write and it is the single highest-value eval asset you will own.

### Pairwise judging beats absolute scoring

Asking a judge for "a score out of 10" gives you numbers that drift between runs and cluster in the middle. Asking "which of these two answers better satisfies the criterion, A, B, or tie?" is more stable and directly answers the question you have. Run both orders (A/B and B/A) and count a win only when the judge is consistent across orders — that cancels position bias. Report ties explicitly; a fine-tune that wins 40%, ties 55% and loses 5% is a different story from one that wins 40% and loses 60%.

### Numbers you can compute from your own run

Report counts, not just percentages, and always with the comparison. Instead of "field accuracy 0.81", write "field accuracy 0.81 (162/200) versus 0.62 (124/200) for the prompted base model, same 200 rows". Then run a quick significance check — for paired binary outcomes, McNemar's test on the discordant pairs is the right tool and it is a dozen lines. If the confidence interval crosses zero, say "no measurable difference" rather than claiming a win.

## Code

A paired comparison with a bootstrap confidence interval on the difference. This answers "is my fine-tune actually better, or is this 200-row noise?" without any statistics library beyond numpy.

```python
from __future__ import annotations

import numpy as np

def paired_delta_ci(
    base: list[int], tuned: list[int], n_boot: int = 10_000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap CI for (tuned - base) on paired 0/1 outcomes, same rows."""
    assert len(base) == len(tuned)
    b = np.asarray(base, dtype=float)
    t = np.asarray(tuned, dtype=float)
    d = t - b                                   # per-row difference: -1, 0, +1
    idx = np.random.default_rng(0).integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)

base_correct  = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 20
tuned_correct = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1] * 20
delta, lo, hi = paired_delta_ci(base_correct, tuned_correct)
print(f"delta={delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
# If the interval excludes 0 you can claim a difference; otherwise say "no clear effect".
```

The pairwise judge, run in both orders, counting only consistent wins. Pass the criterion explicitly; a vague criterion produces a vague verdict.

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
MODEL = "check-current-model-id"   # verify against provider docs

async def prefer(prompt: str, a: str, b: str, criterion: str) -> str:
    body = (
        f"Criterion: {criterion}\n\nRequest:\n{prompt}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
        'Reply JSON: {"winner": "A"|"B"|"tie", "reason": "one sentence"}'
    )
    r = await client.messages.create(model=MODEL, max_tokens=150, temperature=0,
                                     messages=[{"role": "user", "content": body}])
    import json
    return json.loads("".join(x.text for x in r.content if x.type == "text"))["winner"]

async def ab_consistent(prompt: str, base: str, tuned: str, criterion: str) -> str:
    fwd, rev = await asyncio.gather(
        prefer(prompt, base, tuned, criterion),      # A=base, B=tuned
        prefer(prompt, tuned, base, criterion),      # A=tuned, B=base
    )
    # Consistent only when both orders name the same model.
    if fwd == "B" and rev == "A":
        return "tuned"
    if fwd == "A" and rev == "B":
        return "base"
    return "tie"
```

## Real use case

A voice-of-customer team fine-tunes a model to write one-line summaries of app reviews, grouped by theme. Their first report says "much better than base" — because they compared against the base model with no system prompt, on 30 rows they had read during training.

They rebuild the eval. Locked 250-row held-out set split by app version, so no review appears twice across a release. Three metrics: theme-label exact match, a rule that the summary must not exceed 25 words, and a pairwise judge on "does this capture the specific complaint". Baselines: base model with the production prompt, base model with no prompt, and the fine-tune. Result: the fine-tune beats the prompted base on pairwise preference with a CI that excludes zero, but the theme label accuracy difference is inside the noise band, so they report it as "no measurable change". They also run the general battery, find a small drop in Hindi summaries, and add 80 Hindi examples before shipping. The finding that changed the decision was the regression check — the summary quality win was real, the Hindi loss would have been invisible in their original process.

## Common mistakes

- **Mistake:** evaluating on rows you read while writing prompts — **Instead:** lock the held-out file, hash it, and split by source not row.
- **Mistake:** reporting only the fine-tune's score — **Instead:** always report base, prompt-only baseline and fine-tune on the same rows.
- **Mistake:** a single judge run without order swapping — **Instead:** judge both orders and count only consistent verdicts; report ties.
- **Mistake:** skipping the general-ability battery — **Instead:** keep 20-50 items across the capabilities you care about and report the delta.
- **Mistake:** quoting a percentage difference with no interval — **Instead:** bootstrap the paired difference and say "no measurable difference" when it crosses zero.

## Checkpoint

1. Write a 30-item general-ability battery across four capabilities and run it on a base model. Save the scores as your baseline.
2. Take your task metric and compute a bootstrap CI on the paired difference between base and fine-tune.
3. Run a pairwise judge on 50 rows in both orders, then hand-check 10 verdicts. Report your judge's agreement with yourself.

**Self-check:** you lock the held-out set before training, split it by source (document, customer, seed) rather than by row, and never use it to tune prompts or thresholds — otherwise the score measures your tuning, not the model.

## Go deeper

- Next: [7. Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md) — whether to do this at all.
- Judge calibration: [7. Human Eval Design](../15-evals/07-human-evals.md).
- Wire it into CI: [8. Regression Tests and CI Gates](../15-evals/08-regression-tests.md).

---
[← Previous: 5. DPO: Preference Tuning Without RLHF](05-dpo.md) · [Stage index](README.md) · [Next: 7. Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md)
