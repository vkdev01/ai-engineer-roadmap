# 3. Synthetic Data for Fine-Tuning

> **Stage 13 · Fine-Tuning** — manufactures training examples when you do not have enough real ones, without teaching the model your generator's mistakes.
>
> **You'll be able to:** generate task data with a strong model, filter it for correctness and diversity, and prove your synthetic set does not leak into your eval.
>
> **Prerequisites:** [1. Supervised Fine-Tuning (SFT)](01-sft.md) · [2. Few-Shot and Chain of Thought](../08-prompt-and-context/02-few-shot-cot.md)

## Why this exists

Fine-tuning needs examples in a format you control, with answers you trust. Real data is scarce: 600 approved support replies took months to accumulate, and you need coverage of edge cases that have not happened yet. Meanwhile a strong model can write plausible examples by the thousand in an hour.

The trap is that plausible is not correct. Synthetic data amplifies whatever the generator does — its tone, its mistakes, its blind spots. A model trained on unfiltered synthetic output learns to be confidently wrong in a specific, repeatable way, which is worse than being uncertain. The whole discipline is therefore: **generate wide, filter hard, and never let the generator grade its own homework unsupervised.**

## The mental model

```text
seed taxonomy ──► generate (many, varied) ──► filter ──► dedup ──► split
   (topics,            (strong model,        (rules +    (near-      (train /
    intents,            high temperature,     a judge)    duplicate   held-out,
    edge cases)         many prompts)                     removal)    by seed)
```

| Filter | Catches | Cost |
|---|---|---|
| Schema/rule check | malformed JSON, wrong enum, wrong length | free |
| Executable check | wrong code, wrong arithmetic | cheap |
| Judge model | off-topic, unhelpful, wrong tone | model call |
| Embedding dedup | templated near-duplicates | cheap |
| Leakage check | eval overlap | cheap |

## Core concepts

### Start from a taxonomy, not from "write 5,000 examples"

Ask for volume and you get the model's average case five thousand times. Instead, enumerate the space you need covered: intents × input shapes × difficulty × edge cases. Then generate a handful per cell. A 30-cell taxonomy with 15 examples each gives 450 examples with deliberate coverage, and you can name the cell an eval failure came from. Keep the taxonomy in a YAML file so it is reviewable and versioned.

### Diversity comes from the generation prompt, not the temperature

Temperature alone produces stylistic noise, not semantic variety. Vary the seed content: persona of the asker, length, language mix, whether the request is polite, terse or confused, and whether the answer should refuse. Ask the generator for a *reason* field as well as the answer — examples with reasoning are easier to audit and better training signal than bare labels.

### Filtering is a funnel, and the cheapest stage goes first

Run deterministic checks before any model call: JSON parses, required keys present, enums valid, length in range, no placeholder text like `[insert name]`, answer not a restatement of the question. This typically removes the obvious junk for free. Then a judge pass with an explicit rubric — one question, one verdict, and a `reason` field so you can sample-audit the judge itself. Then dedup by embedding similarity: if two examples sit above a high cosine threshold, keep the shorter or the one from the rarer taxonomy cell. Report the count at each stage; a funnel that keeps 8% of 5,000 is normal and worth knowing.

### Leakage is the failure that flatters you

If the generator saw your eval questions — because you described the task using eval examples, or because the generator's own training data overlaps your eval — your held-out score is meaningless. Split by *seed*, not by row: all examples generated from one seed prompt go to the same side of the split. Then check overlap explicitly by comparing normalised text and embeddings between splits, and drop collisions.

## Code

A generation step with a strict schema. The generator returns a list of examples per taxonomy cell; anything that fails validation never reaches disk.

```python
from __future__ import annotations

import asyncio
import json

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError

client = AsyncAnthropic()
MODEL = "check-current-model-id"   # verify against provider docs

class Example(BaseModel):
    user: str = Field(min_length=10, max_length=1200)
    answer: str = Field(min_length=10, max_length=2000)
    reason: str = Field(min_length=5)          # why the answer is correct
    intent: str

async def generate_cell(intent: str, n: int, examples: list[Example]) -> list[Example]:
    prompt = (
        f"Generate {n} distinct customer-support examples for the intent '{intent}'.\n"
        "Vary the asker's tone, length and language mix. Some requests should be\n"
        "unanswerable and get a refusal. Return JSON: a list of objects with keys\n"
        "user, answer, reason, intent. No commentary."
    )
    resp = await client.messages.create(
        model=MODEL, max_tokens=4000, temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    out: list[Example] = []
    for item in json.loads(raw):                 # raises if the model rambled
        try:
            out.append(Example(**item))
        except ValidationError:
            continue                             # drop, never repair silently
    return out
```

Then the funnel: rule checks, then a judge verdict, then embedding dedup. This prints how many examples survive each gate — read those numbers before you train.

```python
import hashlib
import numpy as np

def rule_ok(ex: Example) -> bool:
    text = (ex.user + ex.answer).lower()
    return "[insert" not in text and ex.answer.lower() != ex.user.lower()

async def judged_ok(ex: Example) -> bool:
    rubric = (
        "Does the answer correctly and completely handle the user request, "
        "in the voice of a professional support agent? Reply with JSON "
        '{"verdict": "pass"|"fail", "reason": "..."}'
    )
    resp = await client.messages.create(
        model=MODEL, max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"{rubric}\n\nUSER: {ex.user}\nANSWER: {ex.answer}"}],
    )
    body = json.loads("".join(b.text for b in resp.content if b.type == "text"))
    return body["verdict"] == "pass"

def dedup(examples: list[Example], emb: dict[str, np.ndarray], threshold: float = 0.95) -> list[Example]:
    kept: list[Example] = []
    kept_vecs: list[np.ndarray] = []
    for ex in examples:
        v = emb[hashlib.sha1(ex.user.encode()).hexdigest()]
        v = v / (np.linalg.norm(v) + 1e-9)
        if all(float(v @ k) < threshold for k in kept_vecs):   # no near-duplicate
            kept.append(ex)
            kept_vecs.append(v)
    return kept
```

## Real use case

A B2B invoicing product needs to extract line items from messy invoice text. They have 80 real invoices — far too few to fine-tune, and too sensitive to send to a hosted model as training data. They write a taxonomy: 6 supplier styles × 4 layout quirks × 5 edge cases (credit notes, partial payments, foreign currency, missing GSTIN, multi-page). They generate 40 examples per cell with a strong model, then filter: schema validity drops malformed JSON; an arithmetic check confirms `quantity × rate == amount`; a judge pass removes examples whose totals do not reconcile; embedding dedup removes the templated clones. Roughly a tenth survives. They hold out two entire supplier styles.

Before, a prompted model extracted 60-70% of line items correctly and silently invented totals that did not reconcile. After fine-tuning on the filtered set, extraction on the held-out supplier styles is materially better and — the reason the project was approved — the model now leaves a field blank rather than fabricating a total, because the retained examples all demonstrate "unknown means empty". The filter that mattered most was the arithmetic check: it caught generator mistakes that read perfectly.

## Common mistakes

- **Mistake:** training on raw generator output — **Instead:** funnel it through rules, a judge and dedup, and log survival counts per stage.
- **Mistake:** using the same model to generate and to judge with no rubric — **Instead:** write an explicit rubric, sample-audit 30 verdicts by hand, and keep a human-labelled slice to calibrate against.
- **Mistake:** splitting rows randomly — **Instead:** split by seed prompt or taxonomy cell so a paraphrase of a training example cannot appear in eval.
- **Mistake:** generating 5,000 examples before checking 20 — **Instead:** validate the pipeline on a small batch, hand-read every example, then scale.
- **Mistake:** letting synthetic data drift the tone — **Instead:** include real examples in every batch and compare style metrics against them.

## Checkpoint

1. Write a 12-cell taxonomy for a task you know, generate 5 examples per cell, and hand-audit all 60. Record your own pass rate.
2. Add an executable or arithmetic check to your funnel and report how many examples it removes.
3. Compute pairwise embedding similarity within your set and plot the distribution; pick a dedup threshold and justify it.

**Self-check:** your own audit pass rate on the first 60 generated examples is the honest ceiling for the set — if you reject a fifth by eye, a judge with the same rubric should be rejecting a similar share, or the rubric is too loose.

## Go deeper

- Next: [4. Instruction Tuning vs Continued Pretraining](04-instruction-tuning.md).
- Judge design in depth: [2. LLM-as-Judge, Done Properly](../15-evals/02-llm-as-judge.md).
- Data hygiene: [2. Data Cleaning](../03-data-engineering/02-data-cleaning.md) and [7. Validation](../03-data-engineering/07-validation.md).

---
[← Previous: 2. LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) · [Stage index](README.md) · [Next: 4. Instruction Tuning vs Continued Pretraining](04-instruction-tuning.md)
