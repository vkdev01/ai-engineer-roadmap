# 3. Golden Datasets

> **Stage 15 · Evals and Testing** — the frozen set of cases that every score is measured against, built from real usage and versioned like code.
>
> **You'll be able to:** build a golden dataset that catches real regressions, keep it honest as the product changes, and avoid the two ways datasets rot.
>
> **Prerequisites:** [LLM as Judge](02-llm-as-judge.md) · [Dataset Preparation](../03-data-engineering/08-dataset-prep.md) · [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md)

## Why this exists

A score means nothing without a fixed thing to score against. Change the test set and every historical number becomes fiction; that is why teams quietly stop comparing across weeks and start saying "it feels better".

A golden dataset fixes that. It is small enough to inspect, frozen enough to compare, and representative enough that improving on it improves the product. Building one well is mostly curation, and curation is the part people skip.

The mental model: **a golden dataset is a regression test suite where the expected output is a property, not a string.**

## The mental model

```text
   real traffic ──► sample ──► label ──► freeze ──► gate every change
        ▲             │         │         │              │
        │             │         │         │              ▼
        │             │         │         │        regression caught
        │             │         │         │              │
        └─────────────┴─────────┴──── add a case ◄────────┘
                          (the loop that keeps it alive)
```

Two failure modes to design against: a dataset that never grows (it stops representing users) and a dataset that is edited whenever a test fails (it stops measuring anything).

## Core concepts

### Composition beats size

Forty well-chosen cases catch more than four hundred random ones. A workable composition for a support or RAG system:

| Category | Share | Why |
|---|---|---|
| Straightforward, answerable | 40% | the baseline; if these fail, something is badly wrong |
| Multi-step or multi-document | 20% | the failures users actually complain about |
| Vocabulary mismatch | 10% | user words differ from document words |
| Unanswerable | 15% | refusal is a behaviour and needs testing |
| Adversarial | 10% | injection attempts, prompt leakage, off-scope |
| Previously failed | 5% | every production incident becomes a case |

The "previously failed" row is the mechanism that keeps a dataset alive: when something breaks in production, the case goes in, permanently.

### Expected output is a property

Store what must be true, not one blessed string. A reference answer helps a judge, but the assertions are what make the test deterministic.

```python
from pydantic import BaseModel, Field

class Case(BaseModel):
    qid: str
    question: str
    category: str
    reference: str | None = None            # None => must refuse
    must_cite: list[str] = Field(default_factory=list)      # passage ids that must appear
    must_contain: list[str] = Field(default_factory=list)   # substrings, case-insensitive
    must_not_contain: list[str] = Field(default_factory=list)
    max_words: int | None = None
    notes: str = ""
```

### Version it, and never edit history

The dataset lives in git as JSONL, one case per line, with a version tag in a header file. Adding cases is a commit. Changing a case is a commit with a reason. Rewriting the file to make a score improve is how teams lose the ability to trust their own numbers.

```bash
evals/
├── dataset.v3.jsonl      # current
├── dataset.v2.jsonl      # kept, for comparing old reports
├── baselines/
│   ├── prompt-2026-10-14.json
│   └── prompt-2026-11-02.json
└── CHANGELOG.md          # why each case was added
```

### Contamination is real

If you use a public benchmark, assume it is in the model's training data. Your own dataset, built from your own traffic, is the only one that measures your product. And keep a held-out slice you never look at while iterating, so you can find out whether you overfitted the visible part.

## Code

Building the dataset from logs, and validating it before it is used.

```python
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

def sample_from_logs(path: str, n: int = 60, seed: int = 7) -> list[dict]:
    """Stratified sample of real questions, deduplicated and trimmed."""
    seen: set[str] = set()
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        q = rec["question"].strip()
        key = " ".join(q.lower().split())
        if key in seen or len(q) < 8:
            continue
        seen.add(key)
        rows.append({"question": q, "category": rec.get("intent", "unknown")})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def check_composition(cases: list[dict], target: dict[str, float]) -> dict:
    total = len(cases)
    got = Counter(c["category"] for c in cases)
    report = {}
    for cat, want in target.items():
        have = got.get(cat, 0) / total if total else 0.0
        report[cat] = {"want": want, "have": round(have, 2), "ok": abs(have - want) <= 0.08}
    return report


def validate_dataset(path: str) -> list[str]:
    """Fail loudly: a broken case is worse than a missing one."""
    problems: list[str] = []
    ids: set[str] = set()
    for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
        try:
            case = Case.model_validate_json(line)
        except Exception as exc:                       # noqa: BLE001 - report, do not crash
            problems.append(f"line {i}: {exc}")
            continue
        if case.qid in ids:
            problems.append(f"line {i}: duplicate qid {case.qid}")
        ids.add(case.qid)
        if case.reference is None and not case.must_not_contain:
            problems.append(f"line {i}: unanswerable case needs a must_not_contain assertion")
        if case.category not in {"straight", "multi", "vocab", "unanswerable", "adversarial", "regression"}:
            problems.append(f"line {i}: unknown category {case.category}")
    return problems


if __name__ == "__main__":
    issues = validate_dataset("evals/dataset.v3.jsonl")
    print("\n".join(issues) if issues else "dataset valid")
```

TypeScript, if your eval runner is in Node:

```ts
import { z } from "zod";

export const CaseSchema = z.object({
  qid: z.string(),
  question: z.string().min(8),
  category: z.enum(["straight", "multi", "vocab", "unanswerable", "adversarial", "regression"]),
  reference: z.string().nullable().default(null),
  must_cite: z.array(z.string()).default([]),
  must_contain: z.array(z.string()).default([]),
  must_not_contain: z.array(z.string()).default([]),
});

export type Case = z.infer<typeof CaseSchema>;

export function parseDataset(jsonl: string): Case[] {
  return jsonl
    .split("\n")
    .filter((l) => l.trim())
    .map((l, i) => {
      const parsed = CaseSchema.safeParse(JSON.parse(l));
      if (!parsed.success) throw new Error(`bad case on line ${i + 1}: ${parsed.error.message}`);
      return parsed.data;
    });
}
```

## Real use case

A document-Q&A product has 30 eval cases, all written by the engineer who built the system, all straightforward. The suite is green for months while support tickets keep arriving about "wrong policy version" answers.

The team replaces the dataset: 60 cases sampled from actual questions, categorised, with the five most common complaint shapes added as regression cases. On first run the new suite scores noticeably worse than the old one — which is the correct outcome, because the old suite was measuring the engineer's imagination. Within two iterations the team finds the bug the old suite could never see: the ingestion pipeline was indexing superseded policy versions alongside current ones.

The lesson generalises. **A dataset written by the builder measures the builder's mental model; a dataset sampled from users measures the product.**

## Common mistakes

- **Mistake:** cases written only by the person who wrote the system — **Instead:** sample from real logs; include the badly phrased ones.
- **Mistake:** no unanswerable cases — **Instead:** 10-15% of the set, each with an assertion that the answer must not contain invented specifics.
- **Mistake:** editing cases to make a failing test pass — **Instead:** if a case is genuinely wrong, change it in a commit that says why, and re-baseline.
- **Mistake:** one blessed string as the expected output — **Instead:** assertions plus a reference for the judge.
- **Mistake:** a dataset that never grows — **Instead:** every production incident becomes a case, same day.
- **Mistake:** using a public benchmark as your main measure — **Instead:** use it for sanity, build your own for decisions.

## Checkpoint

1. Sample 60 real questions from your logs, deduplicate, and categorise them. Compare the composition against the table above and fill the gaps.
2. Write ten cases with assertions only (no reference answers) and confirm your runner can grade them deterministically.
3. Add the five most recent complaints as regression cases, then run the suite and record the score — that is your new baseline.

**Self-check:** if your suite passes on a day when users are complaining, the dataset is the bug — go back to the logs before touching the prompt.

## Go deeper

- [Regression Tests](08-regression-tests.md) — wiring the dataset into CI
- [Online and Offline Evals](09-online-offline-evals.md) — the loop that keeps it fresh
- [Dataset Preparation](../03-data-engineering/08-dataset-prep.md) — dedup and split discipline

---
[← Previous: LLM as Judge](02-llm-as-judge.md) · [Stage index](README.md) · [Next: Faithfulness](04-faithfulness.md)
