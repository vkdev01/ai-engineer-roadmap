# 8. Dataset Preparation

> **Stage 03 · Data Engineering** — turning validated records into a dataset you can train and evaluate on
>
> **You'll be able to:** deduplicate near-identical examples, split without leakage, check for contamination, and balance a dataset deliberately.
>
> **Prerequisites:** [Schema Validation with Pydantic](07-validation.md) · [ML Basics](../02-math-and-ml-basics/04-ml-basics.md)

## Why this exists

You have clean, validated records. That is not yet a dataset. Three specific things
still break a fine-tune or an eval, and none of them show up as an error:

- **Near-duplicates.** The same question phrased twice, or a document whose
  boilerplate repeats 400 times. The model memorises the boilerplate and you cannot
  tell from the loss curve.
- **Split leakage.** A near-duplicate of a test example sitting in the training set.
  Your eval score is high and production quality is not.
- **Contamination.** Your eval questions appear in the pretraining corpus of the model
  you are evaluating, so you are measuring recall of the internet, not skill.

Mental model: **preparation is dedup, then split by group, then measure what is left.**

## The mental model

```text
validated records
      |
      v
 1. normalise text for comparison   (lowercase, collapse whitespace)
      |
      v
 2. near-dup cluster  --(shingle + hash + similarity)-->  keep 1 per cluster
      |
      v
 3. group split       --(split by group key, not by row)-->  train / val / test
      |
      v
 4. contamination     --(n-gram overlap with eval set)-->  drop or flag
      |
      v
 5. balance + report  (length buckets, label mix, token totals)
      |
      v
 train.jsonl  val.jsonl  test.jsonl  +  stats.json
```

## Core concepts

### Deduplicate by near-identity, not exact match

Exact dedup (`set` of lines) catches nothing useful in real text. Near-duplicate
detection hashes overlapping word shingles: turn each document into the set of its
5-word sequences, hash each sequence, keep a fixed-size sketch (MinHash), and compare
sketches. Documents sharing most shingles are the same document with an edit. Keep one
representative per cluster — and keep the *earliest* or the *longest*, but decide the
rule before you look at results.

### Split by group, never by row

If two near-duplicates land on opposite sides of a split, your test set is a copy of
your training set. The fix is to cluster first, then assign whole clusters to a split.
The same applies to any natural grouping — all examples from one customer, one
document, one source must stay together. A random row split over grouped data is the
single most common reason an eval looks great and the model disappoints.

### Contamination checks are cheap and worth running

For each eval example, check whether a long n-gram from it appears in the training
data. A 13-gram match is a strong signal. Report the overlap rate; you do not always
have to remove it, but you must know the number before you quote an eval score.

### Balance is a decision, not a discovery

Class imbalance, length imbalance and language imbalance all distort training. Measure
the distribution, then decide: oversample the rare class, cap the long examples, or
stratify the split so each side has the same mix. Whatever you choose, record it — the
next person needs to know whether the 3% of long examples were dropped or weighted.

### Report the numbers, not the vibes

End with a `stats.json`: counts per split, duplicates removed, contamination rate,
median and p95 token length per split, label distribution. That file is the evidence
that the dataset is what you claim, and it is the first thing a reviewer asks for.

## Code

Near-duplicate clustering with MinHash-style shingles, then a group split. You should
see a cluster count, then split sizes that sum to the cluster count.

```python
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

SHINGLE = 5
SKETCH = 64


@dataclass
class Doc:
    doc_id: str
    text: str
    group: str          # e.g. source document id — keeps related rows together


def sketch(text: str, k: int = SHINGLE, size: int = SKETCH) -> set[int]:
    """Bottom-k sketch of hashed k-word shingles: a cheap Jaccard proxy."""
    words = text.lower().split()
    if len(words) < k:
        k = max(1, len(words))
    shingles = {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    hashes = sorted(int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)
                    for s in shingles)
    return set(hashes[:size])


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def cluster(docs: list[Doc], threshold: float = 0.8) -> list[list[Doc]]:
    sketches = [(d, sketch(d.text)) for d in docs]
    clusters: list[list[Doc]] = []
    for doc, sk in sketches:
        for group in clusters:
            if jaccard(sk, sketch(group[0].text)) >= threshold:
                group.append(doc)
                break
        else:
            clusters.append([doc])
    return clusters


def group_split(docs: list[Doc], clusters: list[list[Doc]],
                ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, list[Doc]]:
    # Split CLUSTERS, not rows: a near-duplicate can never cross the boundary.
    by_group: dict[str, list[Doc]] = defaultdict(list)
    for group in clusters:
        by_group[group[0].group].extend(group)
    groups = sorted(by_group)                       # deterministic order
    n = len(groups)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    return {
        "train": [d for g in groups[:n_train] for d in by_group[g]],
        "val":   [d for g in groups[n_train:n_train + n_val] for d in by_group[g]],
        "test":  [d for g in groups[n_train + n_val:] for d in by_group[g]],
    }
```

The measurement half: contamination and a stats report. You should see a contamination
rate near zero and a JSON summary printed to stdout.

```python
from __future__ import annotations

import json
from statistics import median


def ngrams(text: str, n: int = 13) -> set[str]:
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def contamination_rate(train_texts: list[str], eval_texts: list[str]) -> float:
    train_grams: set[str] = set()
    for t in train_texts:
        train_grams |= ngrams(t)
    hits = sum(1 for e in eval_texts if ngrams(e) & train_grams)
    return hits / len(eval_texts) if eval_texts else 0.0


def report(splits: dict[str, list[str]]) -> dict[str, object]:
    out: dict[str, object] = {
        "contamination_test": round(
            contamination_rate(splits["train"], splits["test"]), 4),
    }
    for name, texts in splits.items():
        lens = sorted(len(t.split()) for t in texts) or [0]
        out[name] = {
            "count": len(texts),
            "median_tokens": int(median(lens)),
            "p95_tokens": int(lens[int(len(lens) * 0.95)]),
            "total_tokens": int(sum(lens)),
        }
    return out


if __name__ == "__main__":
    # splits: dict[str, list[str]] built from group_split above
    print(json.dumps(report({"train": ["a b c " * 40], "val": ["x"], "test": ["y z"]}), indent=2))
```

## Real use case

A team built a customer-support eval set of 300 questions by sampling tickets. Their
model scored well on it and users still complained. Running the dedup and group split
from this file showed two problems at once: 41% of the eval questions had a
near-duplicate in the training set (the same complaint, retyped by different agents),
and the 300 questions came from only 60 customers, so the eval measured six people's
phrasing.

The fix was to cluster first, split by cluster, and cap examples per customer. The
eval score dropped noticeably — and became predictive. The team's next release shipped
with a smaller, harder eval set and a `stats.json` recording the contamination rate,
which is now the number they quote when someone asks whether a score is real.

## Common mistakes

- **Mistake:** `train_test_split(df, test_size=0.1)` on text data — **Instead:** cluster near-duplicates, then split clusters; a row split leaks.
- **Mistake:** exact-string dedup only — **Instead:** use shingle sketches and a similarity threshold; real duplicates differ by a word or a date.
- **Mistake:** quoting an eval score without a contamination check — **Instead:** run the n-gram overlap and report the rate next to the score.
- **Mistake:** letting one customer or one document dominate — **Instead:** cap examples per group and report the distribution.
- **Mistake:** choosing the threshold after seeing the results — **Instead:** fix the rule in advance and record it in `stats.json`.
- **Mistake:** dropping long examples silently — **Instead:** cap or weight them, and say which in the report.

## Checkpoint

1. Take a text dataset and cluster it with the shingle sketch above at thresholds 0.6, 0.8 and 0.95. Plot or tabulate cluster count against threshold and pick one with a written justification.
2. Split your clusters into train/val/test by group and prove no cluster spans two splits — write the assertion.
3. Compute the 13-gram contamination rate between your train and test splits, then investigate the highest-overlap pairs by hand and decide whether to drop them.

**Self-check:** clustering before splitting means a near-duplicate pair is always on the same side of the boundary, so the test split measures generalisation rather than recall of the training text.

## Go deeper

- [Ship: A Clean Dataset with a Data Card](09-ship-clean-dataset.md) — this file's steps as a complete project.
- [Schema Validation with Pydantic](07-validation.md) — the gate that runs before dedup.
- [Instruction Tuning](../13-fine-tuning/04-instruction-tuning.md) — where the prepared splits get used.
- [Golden Datasets](../15-evals/03-golden-datasets.md) — how an eval set like this one is maintained over time.

---
[← Previous: Schema Validation with Pydantic](07-validation.md) · [Stage index](README.md) · [Next: Ship: A Clean Dataset with a Data Card](09-ship-clean-dataset.md)
