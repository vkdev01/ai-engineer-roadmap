# 5. Loss Functions

> **Stage 02 · Math and ML Basics** — the single number that defines what "good" means to your model.
>
> **You'll be able to:** choose mean squared error or cross-entropy to match a task, explain why the wrong loss produces a technically-working but useless model, and read a loss curve.
>
> **Prerequisites:** [ML Basics](04-ml-basics.md) · [Probability and Bayes](03-probability-bayes.md)

## Why this exists

A model does not learn what you meant, it learns what you measured. Say "predict the price" and the model minimises average squared error — which it can do by predicting the average price for everything, because that minimises the metric while being useless. The loss is the specification, and vague specifications produce confidently wrong models.

Getting this right is the cheapest improvement available: change the loss to match the decision, and the same architecture behaves differently.

The mental model: **the loss is a scalar scoreboard — lower is better, and the gradient only knows about the scoreboard.**

## The mental model

```text
prediction  y_hat
    |
    +--> loss function  --> one number  --> gradient --> weight update
              ^
        what you told it to care about
```

| Loss | For | Formula | Bad behaviour when misused |
|---|---|---|---|
| MSE | regression | `mean((ŷ - y)²)` | sensitive to outliers |
| MAE | regression with outliers | `mean(\|ŷ - y\|)` | zero gradient at the optimum is fine, but slow |
| Binary cross-entropy | two classes | `-(y log p + (1-y) log(1-p))` | explodes if `p` is 0 or 1 exactly |
| Categorical cross-entropy | many classes | `-log p[correct]` | punishes confident mistakes hard |
| Contrastive / triplet | similarity | `max(0, margin + d(anchor,pos) - d(anchor,neg))` | collapses if margin is wrong |

## Core concepts

### Squared error is for predicting a quantity

`MSE = (1/N) Σ (ŷᵢ - yᵢ)²`. Squaring makes errors positive and punishes large ones more — good when a 10-unit error is much worse than two 5-unit errors, bad when one outlier dominates the sum. Its gradient is `2(ŷ - y)`, which shrinks as you approach the target, so learning slows near the optimum.

### Cross-entropy is for predicting a distribution

`CE = -Σ yᵢ log pᵢ`. With a one-hot label this is just `-log p[correct]`: if you assign 0.9 to the right class the loss is about 0.105; if you assign 0.01 it is about 4.6. The penalty for confident wrongness is severe, which is exactly what you want in classification. In one sentence: **cross-entropy measures surprise.**

### Use logits with cross-entropy, not probabilities

PyTorch's `cross_entropy` and the usual NumPy recipes take raw scores and apply log-softmax internally. Feeding in softmax output and taking `log` yourself invites `log(0)` and loses numerical stability. If you are hand-rolling, subtract the max before exponentiating.

### Averaging versus summing changes the learning rate

`mean` over a batch keeps the gradient scale independent of batch size; `sum` makes it grow with the batch. If you switch from one to the other and keep the same learning rate, training behaviour changes. Pick `mean` and be consistent.

### The loss must match the decision, not just the data type

If a late answer costs ten times as much as a slightly wrong one, plain MSE is the wrong scoreboard. Weighted losses, quantile losses and asymmetric costs exist for this reason. Write down the cost of each kind of error first, then pick a loss that ranks them the same way.

### A loss curve is a diagnostic, not decoration

Both falling: learning. Train falling, validation rising: overfitting. Flat at the start: learning rate too small or gradients dead. Spikes: learning rate too large. `nan`: a log of zero, or a division by zero.

## Code

MSE and cross-entropy from first principles, with gradients — you will reuse these in the next file.

```python
from __future__ import annotations

import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size      # mean -> divide by N


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)   # stability
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, correct: int) -> float:
    return float(-log_softmax(logits)[correct])


def cross_entropy_grad(logits: np.ndarray, correct: int) -> np.ndarray:
    # d(CE)/d(logits) = softmax - one_hot : beautifully simple
    probs = np.exp(log_softmax(logits))
    probs[correct] -= 1.0
    return probs


logits = np.array([2.0, 1.0, 0.1])
print(round(cross_entropy(logits, 0), 4))    # low: right answer, high score
print(round(cross_entropy(logits, 2), 4))    # high: right answer scored badly
print(np.round(cross_entropy_grad(logits, 0), 3))   # negative on correct class
```

Why the choice matters: the same regression data with MSE and with MAE, and a single outlier.

```python
from __future__ import annotations

import numpy as np

y_true = np.array([10.0, 12.0, 11.0, 13.0, 100.0])   # last value is an outlier
y_pred = np.array([10.5, 11.5, 11.0, 12.5, 40.0])

print("MSE:", round(mse(y_pred, y_true), 2))                 # dominated by the outlier
print("MAE:", round(float(np.mean(np.abs(y_pred - y_true))), 2))   # outlier matters linearly

# The minimiser of MSE is the mean; of MAE it is the median.
print("mean:", np.mean(y_true), "median:", np.median(y_true))
```

## Real use case

A pricing assistant predicted delivery times. It was trained with MSE, and users complained that the estimate was "always about twenty minutes late". The model had learned the conditional mean, and because a small number of deliveries went very wrong, the mean sat well above the typical case — the loss rewarded being right on average rather than right for most people.

Changing the loss to the median (MAE, equivalently a 0.5 quantile loss) moved the prediction onto the typical case, which is what a customer reads. When the operations team said late promises cost more than early ones, they moved again to a 0.7 quantile loss so the estimate was wrong in the cheaper direction more often.

Nothing about the network changed. The design choice that mattered was treating the loss as the product decision it is, rather than a default.

## Common mistakes

- **Mistake:** MSE on probabilities, which flattens confident learning. — **Instead:** cross-entropy for anything that outputs a distribution.
- **Mistake:** taking `log(softmax(x))` yourself and hitting `log(0)`. — **Instead:** use log-softmax, or the library's fused loss.
- **Mistake:** reporting loss without a baseline loss. — **Instead:** cross-entropy of uniform guessing is `ln(n_classes)`; report that.
- **Mistake:** switching between `sum` and `mean` reductions while keeping the learning rate. — **Instead:** fix the reduction and say which it is.
- **Mistake:** ignoring outliers in regression because MSE "handles noise". — **Instead:** plot the residuals; consider MAE or Huber.

## Checkpoint

1. Compute cross-entropy for a five-class problem at uniform predictions and confirm it equals `ln(5)`.
2. Implement Huber loss and show how it behaves differently from MSE on the outlier example.
3. Take the loss curve description above and construct a small training run for each of the four symptoms — flat, spiky, `nan`, diverging — by only changing the learning rate.

**Self-check:** with five classes each at probability 0.2, `-ln(0.2) ≈ 1.609`, which is `ln(5)` — that is the "knows nothing" reference point for your loss.

## Go deeper

- [Backpropagation and Gradients](06-backprop-gradients.md) — differentiating these losses by hand.
- [Ship Gradient Descent](07-ship-gradient-descent.md) — using them in a working trainer.
- [Fine-Tune Evaluations](../13-fine-tuning/06-finetune-evals.md) — why a falling loss is not the same as a better product.

---
[← Previous: ML Basics](04-ml-basics.md) · [Stage index](README.md) · [Next: Backpropagation and Gradients](06-backprop-gradients.md)
