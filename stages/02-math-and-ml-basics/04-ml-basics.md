# 4. ML Basics

> **Stage 02 · Math and ML Basics** — the framing that turns a vague wish into a measurable learning problem.
>
> **You'll be able to:** state a problem as features, labels and a metric; split data so your measurement is honest; and recognise overfitting before it embarrasses you.
>
> **Prerequisites:** [Probability and Bayes](03-probability-bayes.md) · [Type Hints and Pydantic v2](../01-programming-basics/02-type-hints-pydantic.md)

## Why this exists

Most failed ML projects failed before any code was written: nobody defined what a correct answer looks like, so nobody could tell whether the model improved. The discipline that prevents this is unglamorous — name the features, name the label, name the metric, hold out data, and always compare against a dumb baseline.

This framing is not only for training. When you evaluate an LLM feature, you are doing exactly the same thing: inputs, expected outputs, a split, and a score.

The mental model: **supervised learning is "find a function that maps features to labels, and generalises to inputs you have not seen".**

## The mental model

```text
                 all labelled data
                        |
        +---------------+---------------+
        |               |               |
     train            validation       test
   (fit weights)   (tune choices)   (report once)
        |               |               |
      seen            seen            never seen
```

| Term | Means | Failure if ignored |
|---|---|---|
| Feature | the input numbers | model learns the wrong signal |
| Label | the target | you cannot measure anything |
| Baseline | the dumbest reasonable guess | no idea if the model is good |
| Overfitting | great on train, worse on val | shipped model that disappoints |
| Leakage | test information in training | flattering, meaningless scores |

## Core concepts

### Features are decisions, and they are most of the quality

A feature is any number you can compute from the input: word counts, an embedding, a retrieval score, the hour of day. Good features make simple models work. Poor features make sophisticated models fail. Say out loud what each feature would let a human guess, because if a human could not use it, neither can the model.

### Labels are the ground truth, so their quality caps everything

A label is what you want predicted, recorded consistently. If two annotators disagree on 20% of examples, no model can exceed roughly 80% agreement with either — the ceiling is the label noise, not the architecture. Measure inter-annotator agreement before training.

### Three splits, three jobs

Train fits the weights. Validation chooses between options: which model, which features, when to stop. Test is touched once, at the end, and reports the number you tell people. Tuning on test is the most common way to fool yourself, because you have quietly fitted your choices to it.

### Overfitting is memorising the training set

The classic signature: training loss keeps falling while validation loss turns upward. It happens when the model has more capacity than the data can constrain. Fixes in order of usefulness: more data, better features, simpler model, regularisation, early stopping at the validation minimum.

### Always beat a baseline

Predict the majority class, or the mean, or the last value. If your model does not clearly beat that, you have a data problem, not a model problem. A baseline also gives you the scale: going from 62% to 71% means something different from 94% to 95%.

### Class imbalance changes what accuracy means

With 2% positives, a model that always says "no" is 98% accurate and useless. Report precision and recall, or use a metric that cares about the minority class, and look at the confusion matrix rather than one number.

## Code

A full honest loop on a toy dataset: split, baseline, fit, and compare validation scores. No ML library, so nothing is hidden.

```python
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def make_data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Two features that carry signal, plus one pure-noise feature.
    signal = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 1))
    X = np.hstack([signal, noise])
    y = (signal[:, 0] + 0.5 * signal[:, 1] > 0).astype(int)   # deterministic rule
    return X, y


X, y = make_data()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)

# Baseline: always predict the majority class.
majority = int(np.bincount(y_tr).argmax())
print("baseline val accuracy:", round(accuracy_score(y_val, np.full_like(y_val, majority)), 3))

model = LogisticRegression().fit(X_tr, y_tr)
val_pred = model.predict(X_val)
print("model val accuracy:", round(accuracy_score(y_val, val_pred), 3))
print("precision/recall:", round(precision_score(y_val, val_pred), 3), round(recall_score(y_val, val_pred), 3))
print("weights (noise feature should be near zero):", np.round(model.coef_[0], 3))
```

Watching overfitting happen, using a deliberately over-capacity model:

```python
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def loss_curve(max_depth: int) -> None:
    X, y = make_data()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_tr, y_tr)
    train = clf.score(X_tr, y_tr)
    val = clf.score(X_val, y_val)
    print(f"depth={max_depth:<3} train={train:.3f} val={val:.3f} gap={train - val:+.3f}")


for depth in (1, 3, 10, None):   # None = grow until pure leaves
    loss_curve(depth)
```

## Real use case

A team building an "is this ticket urgent?" router started with a transformer fine-tune and got 0.93 accuracy on their test set. In production, urgent tickets were missed constantly.

Two framing errors were behind it. First, the test set had been reused for weeks of prompt and threshold tuning, so 0.93 measured their memory of the test set, not generalisation. Second, urgent tickets were 4% of the data, so 0.93 accuracy was barely above the "always not urgent" baseline of 0.96 — the metric was hiding the failure that mattered.

They rebuilt the framing: a fresh held-out set touched once, a majority-class baseline reported first, and recall on the urgent class as the primary metric with precision as the guardrail. The same model, re-measured honestly, showed recall 0.61 at the threshold in use. A threshold chosen on validation data lifted recall to 0.88 while precision stayed acceptable to the triage team.

The design choice that mattered was naming the baseline and the minority-class metric before looking at any model score.

## Common mistakes

- **Mistake:** tuning on the test set. — **Instead:** tune on validation, report test once.
- **Mistake:** reporting accuracy on imbalanced data. — **Instead:** precision, recall, and a confusion matrix.
- **Mistake:** no baseline, so "0.93" means nothing. — **Instead:** compute majority-class or mean-prediction first.
- **Mistake:** leakage — normalising or imputing before splitting. — **Instead:** fit every transform on train only, then apply to val and test.
- **Mistake:** adding features until the validation score stops improving, then stopping. — **Instead:** that is fitting the validation set; keep a third split or accept the noise.

## Checkpoint

1. Compute the majority-class baseline for the toy data and compare it to the logistic model; explain the gap.
2. Run `loss_curve` for depths 1, 3, 10 and `None` and identify where overfitting starts.
3. Introduce a leaky feature — one computed from the label — and show how validation accuracy becomes misleading.

**Self-check:** the baseline sits near 0.5 because the toy labels are balanced by construction; the logistic model should reach the high 0.8s to 0.9s because it can represent the true rule.

## Go deeper

- [Loss Functions](05-loss-functions.md) — the number that training actually minimises.
- [EDA](../03-data-engineering/03-eda.md) — where you find out whether your features exist at all.
- [Golden Datasets](../15-evals/03-golden-datasets.md) — the same split discipline applied to LLM features.

---
[← Previous: Probability and Bayes](03-probability-bayes.md) · [Stage index](README.md) · [Next: Loss Functions](05-loss-functions.md)
