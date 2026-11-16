# 7. Human Evals

> **Stage 15 · Evals and Testing** — designing annotation that produces a number you can trust, with an agreement statistic attached.
>
> **You'll be able to:** design an annotation task, write guidelines that two people can apply the same way, compute inter-annotator agreement, and use human labels to calibrate your judges.
>
> **Prerequisites:** [LLM as Judge](02-llm-as-judge.md) · [Golden Datasets](03-golden-datasets.md) · [Checkpoint](../11-agents/12-ship-harness-agent.md)

## Why this exists

Every automated measure in this stage is calibrated against human judgement somewhere, and human judgement is noisy in specific, measurable ways. Two annotators reading the same answer often disagree — not because one is careless, but because the guideline left a gap.

Human evals are therefore not a fallback for "real quality". They are a measurement instrument too, and like any instrument they need calibration: clear guidelines, independent labelling, and an agreement number reported alongside the result.

The mental model: **if two careful people cannot agree using your guideline, the guideline is the bug.**

## The mental model

```text
   guideline ──► annotator A ──┐
        │                      ├──► agreement (kappa) ──► fix guideline if low
        │      ──► annotator B ─┘                │
        │                                        ▼
        └──────────────────────────────► labels ──► calibrate the judge
                                              │
                                              ▼
                                    trustworthy automated score
```

## Core concepts

### Write the guideline before the task

A usable guideline has four parts: the decision (what exactly is being judged), the levels with observable boundaries, worked examples for each level including the hard ones, and a rule for what to do when the case is ambiguous ("if the source is silent, mark UNSUPPORTED, not FAIL").

The test of a guideline is not whether it reads well; it is whether two annotators reach the same verdict on twenty cases. Run that pilot before paying for volume.

### Independence, then discussion

Annotators label independently first. Then a short adjudication session resolves disagreements — and every disagreement resolved by discussion is a guideline gap you should write down. Never let one annotator see the other's labels first; agreement measured that way is theatre.

### Agreement: kappa, not percentage

Percentage agreement is inflated on skewed data. Report Cohen's kappa for two annotators and categorical labels, or Krippendorff's alpha for more than two or ordinal scales. Interpret roughly: below 0.4 means the guideline is unusable; 0.4-0.6 is tentative; above 0.6 is usable for decisions; above 0.8 is strong.

```python
def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 0.0 if expected == 1 else (observed - expected) / (1 - expected)
```

### Sample size, honestly

Twenty cases cannot distinguish 0.80 from 0.85. If you need that resolution, you need a few hundred. For calibration of a judge, 40-60 well-chosen cases is enough to detect a badly broken rubric. Say which of the two jobs you are doing, because they have different budgets.

### Where humans are genuinely needed

Use people for: calibrating judges, adjudicating contested cases, judging tone and policy compliance, reviewing safety incidents, and any decision with legal consequence. Do not use people for: volume screening, regression detection on 300 cases per commit, or anything a mechanical check can settle.

## Code

An annotation round end to end: sheet generation, agreement, and adjudication queue.

```python
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass

@dataclass
class Item:
    iid: str
    question: str
    answer: str
    sources: list[dict]

def make_sheets(items: list[Item], annotators: list[str], seed: int = 11) -> None:
    """One CSV per annotator, shuffled independently so order effects do not correlate."""
    for name in annotators:
        rows = items[:]
        random.Random(seed + hash(name) % 1000).shuffle(rows)
        with open(f"annotation/{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iid", "question", "answer", "label", "note"])
            for it in rows:
                w.writerow([it.iid, it.question, it.answer, "", ""])


def load_labels(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["label"].strip():
                out[row["iid"]] = row["label"].strip().upper()
    return out


def agreement(a: dict[str, str], b: dict[str, str]) -> dict:
    shared = sorted(set(a) & set(b))
    xs = [a[i] for i in shared]
    ys = [b[i] for i in shared]
    kappa = cohens_kappa(xs, ys)
    return {
        "n": len(shared),
        "raw_agreement": round(sum(1 for x, y in zip(xs, ys) if x == y) / max(len(shared), 1), 3),
        "kappa": round(kappa, 3),
        "usable": kappa >= 0.6,
        "disagreements": [i for i in shared if a[i] != b[i]],
    }


def adjudication_queue(disagreements: list[str], a: dict[str, str], b: dict[str, str]) -> list[dict]:
    return [{"iid": i, "a": a[i], "b": b[i]} for i in disagreements]
```

The output of a round is three artefacts: the agreement number, the adjudicated labels (the ground truth you will calibrate against), and a list of guideline clarifications. That third artefact is the one teams forget, and it is the one that makes the next round cheaper.

TypeScript, for an annotation UI that prevents skipped fields:

```ts
type Label = "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED";

export function validateRow(row: { iid: string; label?: string; note?: string }): string[] {
  const problems: string[] = [];
  const allowed: Label[] = ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"];
  if (!row.label || !allowed.includes(row.label as Label)) problems.push(`${row.iid}: invalid label`);
  if (row.label === "CONTRADICTED" && !row.note?.trim()) {
    problems.push(`${row.iid}: contradiction requires a note quoting the source`);
  }
  return problems;
}
```

## Real use case

A team building an answer-quality judge needs 60 labelled cases. They write a four-page guideline, run a 20-case pilot with two annotators, and get kappa 0.31 — barely above chance.

The disagreements are all one pattern: answers that are correct but uncited. One annotator counts them as a failure of "grounded", the other as fine because "the answer is right". The guideline gets one new line — "an uncited answer that cannot be traced to a source scores UNSUPPORTED regardless of correctness" — plus two worked examples. Second pilot: kappa 0.71.

Only then do they label 60 cases and calibrate the judge, which lands at kappa 0.64 against the adjudicated labels. The entire sequence took two days and produced a judge they can gate releases on. Skipping the pilot would have produced 60 labels of noise and a judge optimised against a contradiction.

## Common mistakes

- **Mistake:** a guideline that describes the task instead of the decision — **Instead:** name each level with observable boundaries and give a worked example per level.
- **Mistake:** skipping the pilot — **Instead:** 20 cases, two annotators, kappa; rewrite before scaling.
- **Mistake:** annotators seeing each other's labels — **Instead:** independent labelling, then adjudication.
- **Mistake:** reporting percentage agreement — **Instead:** kappa or alpha, plus the label distribution.
- **Mistake:** no record of why disagreements were resolved a particular way — **Instead:** keep a guideline changelog; it is the artefact that improves over time.
- **Mistake:** using humans for regression screening — **Instead:** humans calibrate; machines gate.

## Checkpoint

1. Write a one-page guideline for a decision you actually make about your system, with three levels and a worked example each.
2. Pilot it with a colleague on 20 cases, independently, and compute kappa. If below 0.5, find the pattern in the disagreements and fix the guideline.
3. Use the adjudicated labels to calibrate your judge and report both numbers in your README.

**Self-check:** if kappa is low but both annotators feel confident, the disagreement is definitional — read the three worst cases together before changing anything else.

## Go deeper

- [LLM as Judge](02-llm-as-judge.md) — what these labels are for
- [Golden Datasets](03-golden-datasets.md) — where the cases come from
- [Online and Offline Evals](09-online-offline-evals.md) — the loop that keeps human review targeted

---
[← Previous: Agent Evals](06-agent-evals.md) · [Stage index](README.md) · [Next: Regression Tests](08-regression-tests.md)
