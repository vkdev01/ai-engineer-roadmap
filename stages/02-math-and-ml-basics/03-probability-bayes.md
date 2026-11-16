# 3. Probability and Bayes

> **Stage 02 · Math and ML Basics** — the language for uncertainty, which is what every model output really is.
>
> **You'll be able to:** read a probability distribution, condition on evidence, apply Bayes' rule to update a belief, and explain why a softmax output is not a calibrated truth.
>
> **Prerequisites:** [Calculus](02-calculus.md) · [Linear Algebra](01-linear-algebra.md)

## Why this exists

A classifier does not say "this is a bug report". It says "0.72 bug, 0.21 feature, 0.07 billing" — and you have to decide what to do with that. Do you route it, ask a human, or re-run with a better prompt? That decision needs probability, not just the top label.

The same maths shows up in retrieval scores, hallucination estimates, sampling temperature, evaluation confidence intervals and Bayesian A/B tests on prompt variants. This file gives you the working vocabulary.

The mental model: **probability is a number between 0 and 1 that you update when you learn something.**

## The mental model

```text
prior belief          evidence                posterior belief
P(sick) = 0.01   +    test positive      ->   P(sick | positive)
                        sensitivity 0.99
                        false alarm 0.05
```

| Term | Means | Example |
|---|---|---|
| `P(A)` | prior | 1% of tickets are urgent |
| `P(A \| B)` | conditional | given it says "down", chance it is urgent |
| Independence | `P(A∩B) = P(A)P(B)` | two unrelated tickets |
| Expectation | long-run average | average tokens per request |
| Variance | spread | how much latency varies |

## Core concepts

### Probabilities are counts in the limit, and they must add to 1

Roll a die a million times and each face appears about a sixth of the time. A distribution over labels assigns each label a number and the numbers sum to one. When your numbers do not sum to one, something in your pipeline is wrong — a missing class, or a logit passed through the wrong transform.

### Conditional probability is where the information is

`P(A|B) = P(A∩B)/P(B)`. "Chance of urgent given the word 'down'" is different from "chance of the word 'down' given urgent". Confusing the two is the single most common reasoning error in ML, and it is the thing Bayes' rule fixes.

### Bayes' rule turns evidence into a belief

`P(A|B) = P(B|A)·P(A) / P(B)`. In words: posterior ∝ likelihood × prior. With a 1% base rate and a test that is 99% sensitive but 5% false-positive, a positive result gives only about 17% chance of the condition — because false positives vastly outnumber true positives in a rare population. Base rates matter more than most people expect.

### Softmax turns scores into a distribution

`softmax(z)ᵢ = exp(zᵢ) / Σ exp(zⱼ)`. It exponentiates (so all values are positive) and normalises (so they sum to one). Temperature divides the logits before the softmax: low temperature sharpens toward the argmax, high temperature flattens toward uniform. That is literally what the `temperature` parameter does.

### Expected value is the right way to compare options

`E[X] = Σ x·P(x)`. "Model A is cheaper but fails 3% more often" is an expectation question once you price a failure. Same tool for latency budgets and for choosing between a large model and a small one with retries.

### Independence is an assumption, not a fact

`P(A∩B) = P(A)P(B)` only when `A` and `B` do not influence each other. Two chunks from the same document are not independent, which is why naive assumptions about retrieval diversity break. When in doubt, do not assume independence.

## Code

Bayes' rule as a function, then applied to the base-rate trap.

```python
from __future__ import annotations


def bayes(prior: float, sensitivity: float, false_positive: float) -> float:
    """P(condition | positive test) from the base rate and test quality."""
    p_pos = sensitivity * prior + false_positive * (1 - prior)   # total probability
    return sensitivity * prior / p_pos


print(round(bayes(prior=0.01, sensitivity=0.99, false_positive=0.05), 4))  # ~0.1667
print(round(bayes(prior=0.50, sensitivity=0.99, false_positive=0.05), 4))  # ~0.9519
```

Softmax with temperature, and the numbers that explain the sampling slider.

```python
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)        # shift for numerical stability
    exps = np.exp(scaled)
    return exps / exps.sum()


logits = np.array([2.0, 1.0, 0.1])
for t in (0.5, 1.0, 2.0):
    print(t, np.round(softmax(logits, t), 3))

# 0.5 -> [0.864 0.117 0.019]   sharper, closer to always picking class 0
# 2.0 -> [0.502 0.304 0.194]   flatter, more variety
```

A tiny Bayesian update loop — the shape of "we started with a prior, and each evaluation batch moved it".

```python
from __future__ import annotations


def update(prior: float, successes: int, trials: int) -> float:
    """Beta-Binomial posterior mean for a success rate, starting from `prior`."""
    alpha, beta = prior * 10, (1 - prior) * 10   # weak prior worth 10 observations
    return (alpha + successes) / (alpha + beta + trials)


belief = 0.5
for good, total in [(7, 10), (18, 20), (45, 50)]:
    belief = update(belief, good, total)
    print(round(belief, 3))    # climbs, then settles near the observed rate
```

## Real use case

A hallucination filter flagged answers whose model confidence was below a threshold, routing them to a human. It was tuned on a test set where half the answers were wrong, and at that base rate a threshold of 0.6 looked excellent. In production, flagged rate collapsed and bad answers reached users.

The reason is the base-rate effect: in production only a small fraction of answers are wrong, so almost all low-confidence answers were actually fine, and the threshold caught little. The team re-derived the threshold using `bayes` with the real production prior, which showed that a 0.6 confidence cut had a precision far below what the test set implied. They lowered the threshold, added a second signal (retrieval score), and combined the two as independent-ish evidence rather than trusting confidence alone.

The design choice that mattered was treating confidence as a likelihood and the production wrong-rate as a prior, instead of reading confidence as a probability of being wrong.

## Common mistakes

- **Mistake:** reading a softmax score as "72% chance this is correct". — **Instead:** it is a normalised score, often poorly calibrated; measure calibration before relying on it.
- **Mistake:** confusing `P(symptom|disease)` with `P(disease|symptom)`. — **Instead:** write both out and apply Bayes' rule.
- **Mistake:** tuning a threshold on a balanced test set and deploying to an unbalanced population. — **Instead:** recompute with the real prior.
- **Mistake:** assuming two retrieval chunks are independent when scoring combined evidence. — **Instead:** treat overlap as redundancy and discount it.
- **Mistake:** averaging probabilities of different events and calling it "confidence". — **Instead:** decide what the average is supposed to mean first.

## Checkpoint

1. Compute the posterior for a base rate of 0.05 with a test at 90% sensitivity and 10% false positive, and explain the result in one sentence.
2. Plot `softmax([3, 1, 0])` for temperatures from 0.1 to 3 and describe what happens at each end.
3. Simulate 1,000 draws from a distribution you define, then compare the empirical frequencies to the theoretical ones.

**Self-check:** `bayes(0.05, 0.90, 0.10) ≈ 0.321` — even with a good test, a rare condition means most positives are false alarms.

## Go deeper

- [Loss Functions](05-loss-functions.md) — cross-entropy is built from exactly these log-probabilities.
- [LLM as Judge](../15-evals/02-llm-as-judge.md) — where base rates quietly ruin evaluation.
- [Hallucinations](../15-evals/05-hallucinations.md) — uncertainty in practice rather than in theory.

---
[← Previous: Calculus](02-calculus.md) · [Stage index](README.md) · [Next: ML Basics](04-ml-basics.md)
