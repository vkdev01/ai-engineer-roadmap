# 2. LLM as Judge

> **Stage 15 · Evals and Testing** — using a model to grade outputs at scale, with a rubric, a calibration number and an honest account of its bias.
>
> **You'll be able to:** write a grading rubric, build a judge that returns structured verdicts, measure its agreement with your own labels, and know when not to trust it.
>
> **Prerequisites:** [Unit Tests for LLM Code](01-unit-tests.md) · [Structured Outputs](../07-llm-apis/04-structured-outputs.md) · [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md)

## Why this exists

You cannot read ten thousand answers. A judge model can, cheaply, and that is the only reason LLM evaluation scales past a few dozen cases.

The danger is treating the judge as ground truth. It has known biases — it prefers longer answers, it prefers answers that look like its own output, and it is sensitive to the order in which it sees options. A judge you have not calibrated is a random number generator with good manners.

The mental model: **a judge is a measuring instrument; calibrate it before you trust a reading.**

## The mental model

```text
   your labels (50 cases, human)  ──┐
                                    ├──► agreement rate  ──► "can I trust it?"
   judge verdicts (same 50 cases) ──┘

   then, and only then:
   judge over 10,000 cases  ──► score with error bars  ──► CI gate
```

If agreement is below roughly 80% on a binary decision, fix the rubric before scaling up. A vague rubric produces a confident judge that disagrees with you, and you will spend weeks optimising against noise.

## Core concepts

### Rubrics, not adjectives

"Rate the answer 1-5 for quality" is not a rubric; it is an invitation to hallucinate a number. A rubric names the dimensions, defines each score level in observable terms, and forbids partial credit where it does not exist.

```text
Score 5: answers the question, every claim is supported by the provided context,
         cites at least one source id, no invented specifics.
Score 4: answers the question, all claims supported, but no citations.
Score 3: answers the question with one unsupported claim.
Score 2: partially answers; or answers a related but different question.
Score 1: contradicts the context, or invents facts.
```

Note that 5 and 4 differ on a checkable property (citations), not on taste. Every boundary in your rubric should be checkable by a second person reading the same text.

### Structured verdicts, validated

Ask for JSON, parse it with a schema, and treat a parse failure as a failed grade rather than a silent default. The judge's output is data; validate it like any other data.

```python
from pydantic import BaseModel, Field, ValidationError

class Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = Field(max_length=600)
```

### Position and length bias, and the cheapest fixes

For pairwise comparison ("which answer is better"), swap the order and require consistency: if the judge prefers A-then-B and also B-then-A, that comparison is void. For absolute scoring, control for length by including length in the rubric, or by scoring short and long answers separately and reporting both.

### Calibration is a number you publish

Take 40-60 cases, label them yourself (or with a colleague, independently), run the judge, and compute agreement. Report it next to every score you produce from that judge. A scorecard that says "faithfulness 0.93 (judge agreement 0.87 vs 50 human labels)" is a different artefact from one that says "faithfulness 0.93".

```python
def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Agreement corrected for chance. Use this, not raw accuracy."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 0.0 if expected == 1 else (observed - expected) / (1 - expected)
```

Raw agreement flatters a judge on skewed data: if 90% of answers are good, a judge that always says "good" scores 0.90 and has learned nothing. Kappa subtracts that.

### Cost control

Grading is a model call per case. Ten thousand cases at a large model is real money. Two practical moves: grade with a small model and calibrate it (often agreement is within a few points of the big one), and grade a stratified sample rather than everything — 300 well-chosen cases beat 10,000 random ones for detecting regressions.

## Code

A complete judge with schema validation, order swapping and calibration.

```python
from __future__ import annotations

import json
from statistics import mean

from openai import OpenAI

client = OpenAI()

RUBRIC = """You are grading a support answer against the sources it was given.

Score 5: fully answers the question; every claim is supported by the sources; cites at least one source id.
Score 4: fully answers; all claims supported; no citations.
Score 3: answers, but contains exactly one claim not supported by the sources.
Score 2: answers a related but different question, or answers only partially.
Score 1: contradicts the sources, or invents facts.

Return JSON only: {"score": int, "unsupported_claims": [str], "reasoning": str}
Reasoning must be under 400 characters and must cite the source ids you checked."""


def judge_answer(question: str, answer: str, sources: list[dict]) -> dict:
    src = "\n\n".join(f"[{s['id']}] {s['text'][:1200]}" for s in sources)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                    # check the provider's current model list
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nSOURCES:\n{src}\n\nANSWER:\n{answer}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        verdict = Verdict.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError):
        return {"score": None, "error": "unparseable_verdict", "raw": raw[:200]}
    return verdict.model_dump()


def compare(a: str, b: str, question: str) -> str | None:
    """Pairwise with order swap. Returns 'a', 'b' or None if the judge contradicts itself."""
    first = _prefer(question, a, b)
    second = _prefer(question, b, a)
    if first is None or second is None:
        return None
    if first == "first" and second == "second":
        return "a"
    if first == "second" and second == "first":
        return "b"
    return None                                  # inconsistent: void this comparison


def calibrate(cases: list[dict], human: list[int]) -> dict:
    judge_scores = [judge_answer(c["question"], c["answer"], c["sources"])["score"] for c in cases]
    pairs = [(j, h) for j, h in zip(judge_scores, human) if j is not None]
    js = [j for j, _ in pairs]
    hs = [h for _, h in pairs]
    return {
        "n": len(pairs),
        "mean_judge": round(mean(js), 2),
        "mean_human": round(mean(hs), 2),
        "kappa": round(cohens_kappa(js, hs), 3),
        "within_one_point": round(mean(1.0 if abs(j - h) <= 1 else 0.0 for j, h in pairs), 3),
    }
```

The `within_one_point` number is often more useful than exact agreement, because adjacent rubric levels are genuinely arguable. Publish both and say which you gate on.

TypeScript, when the scorecard lives in a dashboard:

```ts
type Calibration = { n: number; kappa: number; within_one_point: number };

export function trustLabel(c: Calibration): string {
  if (c.n < 30) return "insufficient labels — do not gate on this judge";
  if (c.kappa >= 0.6) return "usable for gating";
  if (c.kappa >= 0.4) return "usable for monitoring only";
  return "rubric too vague — rewrite it before scaling up";
}
```

## Real use case

A team building a contract-review assistant wants to gate releases on "answer quality". Their first judge uses a 1-5 scale with a one-line instruction; it reports 0.91 mean quality, which is suspiciously stable across every prompt change. Nobody believes it.

They label 50 cases themselves, compute kappa against the judge, and get 0.22 — the judge is scoring fluency, not correctness. They rewrite the rubric to name observable failures (a claim not in the contract, a missing clause reference, a wrong party name), re-calibrate, and land at kappa 0.64 with `within_one_point` at 0.88. Only then do they trust the number, and the first thing it reveals is that their prompt change from two weeks earlier had made answers 8 points less faithful — a regression nobody had noticed.

That is the entire value proposition of this file: **a calibrated judge finds regressions you would otherwise ship.**

## Common mistakes

- **Mistake:** shipping a judge without calibration — **Instead:** 40-60 human labels and a kappa number before any score is quoted.
- **Mistake:** a rubric of adjectives ("clear", "helpful") — **Instead:** observable boundaries; every level checkable by a second person.
- **Mistake:** reporting raw agreement on skewed data — **Instead:** kappa, plus the class distribution.
- **Mistake:** trusting a judge that cannot parse its own output — **Instead:** count parse failures; above a few percent, the rubric is too long or the schema too strict.
- **Mistake:** using the same model as judge and generator — **Instead:** prefer a different family, and check for self-preference on your own data.
- **Mistake:** grading all 10,000 cases every commit — **Instead:** gate on a stratified 300, run the full set nightly.

## Checkpoint

1. Write a rubric for a task you own, with five levels and observable boundaries. Grade ten answers by hand first.
2. Build the judge, run it on the same ten, and compute kappa. If it is below 0.4, rewrite the rubric and try again.
3. Add order-swapped pairwise comparison for two prompt variants and report how many comparisons were voided by inconsistency.

**Self-check:** if your kappa is high but your judge's mean score never moves when you break the prompt on purpose, the rubric is measuring style — add a failure mode that a broken prompt must trip.

## Go deeper

- [Golden Datasets](03-golden-datasets.md) — the labelled set calibration depends on
- [Faithfulness](04-faithfulness.md) — a measure worth judging
- [Regression Tests](08-regression-tests.md) — putting a calibrated judge behind a CI gate

---
[← Previous: Unit Tests for LLM Code](01-unit-tests.md) · [Stage index](README.md) · [Next: Golden Datasets](03-golden-datasets.md)
