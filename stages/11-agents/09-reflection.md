# 9. Reflection and Self-Critique

> **Stage 11 · AI Agents** — asking the model to check its own work: what a critique loop genuinely fixes, what it only appears to fix, and how to make it earn its tokens.
>
> **You'll be able to:** add a bounded critique-and-revise step that improves measurable quality, and recognise the cases where it just adds words.
>
> **Prerequisites:** [Reasoning and Planning](03-reasoning-planning.md) · [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md)

## Why this exists

Reflection is the most over-sold idea in agent design. The appealing story is that the model reviews its answer, spots the mistake, and fixes it. Sometimes that happens. Often the model rewrites the same content with more confident phrasing, and you have paid twice for the same answer.

It works when the critique has **external evidence** — a test that fails, a schema that rejects, a document that contradicts the claim, a tool that returns "no such order". It fails when the critique is the model marking its own homework with no new information, because the model's judgement is the very thing that was wrong.

The mental model: **reflection is worth it only when the second pass sees something the first pass did not.**

## The mental model

```text
 weak: answer ──► same model reads it ──► "looks good" ──► same answer
                                       └─► "improved" ──► reworded answer

 strong: answer ──► check against EVIDENCE ──► specific defect ──► targeted fix
                    (tests, schema, docs,            │
                     tool result, rubric)            └─► re-run only the failed part
```

| Critique source | Fixes | Typical failure |
|---|---|---|
| Model re-reading its own text | Format, obvious gaps | Confident rewording, no gain |
| Rubric with named criteria | Missing sections, tone | Rubric gaming |
| Unit tests or schema | Real correctness | Only where tests exist |
| Retrieved source passages | Unsupported claims | Retrieval misses the source |
| Tool result | Wrong facts about the world | Extra step cost |

## Core concepts

### Critique against evidence, not opinion

Write the critique prompt so the model must point at something external: a failing assertion, a passage that contradicts the claim, a required field that is absent. Then require the output to be a list of **specific defects with locations** — "paragraph 3 states the refund window is 30 days; the policy document says 14" — rather than a verdict. A verdict invites agreement; a located defect forces work.

### Bound the loop

Two passes maximum in most systems: one to produce, one to revise. A third pass rarely changes the answer and always changes the bill. Track revision count in the trace and stop on "no defects found", on a step budget, or when the diff between passes is trivial.

### Check the things code can check

Before spending a model call, run the cheap checks: does the JSON parse, does the schema validate, do all cited ids exist in the retrieved set, are the numbers internally consistent, does the length fit. These catch a large share of real defects for free. Reflection is what you do after the mechanical checks pass.

### Separate the critic from the producer when it matters

Same-model self-critique has a known blind spot: it defends its own reasoning. Two cheap mitigations — a different prompt and a different context (the critic sees only the answer plus evidence, not the reasoning that produced it), or a different model for the critique. The second costs more and helps most on high-stakes output.

### Measure, do not assume

Add the critique step behind a flag and run your eval set with it off and on. If the score does not move, delete it — a step that does not help is a step that adds latency and a new failure mode. This single habit separates engineers from enthusiasts.

## Code

A bounded critique loop that returns located defects, and a mechanical checker that runs first. The `critique` function refuses to run when there is nothing to check against.

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

@dataclass
class Defect:
    location: str        # "paragraph 3" | "field: total" | "claim 2"
    problem: str         # what is wrong
    evidence: str        # the passage, test or tool result that proves it
    fix: str             # the smallest change that resolves it

def mechanical_checks(answer: str, schema: type | None, known_ids: set[str]) -> list[Defect]:
    defects: list[Defect] = []
    if schema is not None:
        try:
            json.loads(answer)
        except json.JSONDecodeError as exc:
            defects.append(Defect("whole answer", f"invalid JSON: {exc.msg}", "json.loads", "return valid JSON"))
    for token in _extract_ids(answer):                 # your own id extractor
        if token not in known_ids:
            defects.append(Defect(f"citation {token}", "id not in retrieved set",
                                  f"known ids: {sorted(known_ids)[:5]}", "drop or replace the claim"))
    return defects

CRITIQUE_PROMPT = """You are reviewing a draft against evidence.
Rules:
- Report only defects you can prove from the evidence below.
- Each defect: location, problem, evidence quote, smallest fix.
- If you cannot point at evidence, do not report it.
Return JSON: {"defects": [{"location": str, "problem": str, "evidence": str, "fix": str}]}"""

async def critique(draft: str, evidence: str, model: Any) -> list[Defect]:
    if not evidence.strip():
        return []                                       # nothing to check against: skip
    reply = await model.complete(
        [{"role": "system", "content": CRITIQUE_PROMPT},
         {"role": "user", "content": f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{draft}"}], [])
    raw = json.loads(reply["text"])
    return [Defect(**d) for d in raw.get("defects", [])][:5]   # cap: force prioritisation

async def revise_until_clean(draft: str, evidence: str, model: Any,
                             max_passes: int = 2) -> tuple[str, list[Defect]]:
    current, last = draft, []
    for _ in range(max_passes):
        last = await critique(current, evidence, model)
        if not last:
            break                                       # exit 1: no provable defect
        fixes = "\n".join(f"- at {d.location}: {d.fix}" for d in last)
        reply = await model.complete(
            [{"role": "user",
              "content": f"Apply only these fixes. Do not rewrite anything else.\n"
                         f"{fixes}\n\nDRAFT:\n{current}"}], [])
        current = reply["text"]
    return current, last
```

The TypeScript version is the one that decides whether to spend the extra call at all — a cost gate in the request path.

```ts
export type Defect = { location: string; problem: string; fix: string };

export async function maybeReflect(
  draft: string,
  defects: Defect[],
  revise: (draft: string, defects: Defect[]) => Promise<string>,
  maxPasses = 2,
): Promise<{ text: string; passes: number }> {
  let text = draft;
  let passes = 0;
  while (defects.length > 0 && passes < maxPasses) {
    text = await revise(text, defects);
    passes += 1;
    if (defects.length === 0) break;
  }
  return { text, passes };      // passes === 0 means the draft was already clean
}
```

## Real use case

A legal-tech team generated contract summaries with a single call. Summaries read well but occasionally cited a clause number that did not exist. A first fix asked the model to "review your summary for accuracy" — the summaries got longer and the invented clause numbers stayed, because the model had no clause text to check against.

The second fix attached the parsed clause list to the critique step and demanded located defects with quotes. Invented clause numbers dropped to zero on their regression set, because the critic could point at the clause table. Rewriting was also cheaper: the revise prompt applies named fixes only, so the draft changes by a few sentences instead of a full regeneration.

## Common mistakes

- **Mistake:** "review your answer for errors" with no evidence attached — **Instead:** give the critic the sources, the tests or the tool result.
- **Mistake:** accepting a verdict instead of located defects — **Instead:** require location, problem, evidence and a minimal fix.
- **Mistake:** three or more revision passes — **Instead:** cap at two and log when the cap is hit.
- **Mistake:** letting the revision pass rewrite the whole answer — **Instead:** instruct it to apply only the listed fixes.
- **Mistake:** shipping reflection without an A/B on your eval set — **Instead:** measure score, latency and cost with it on and off.

## Checkpoint

1. Add a mechanical check that every number in the draft also appears in the evidence, and see how many defects it catches before any model call.
2. Make `critique` return an empty list when all defects are stylistic, and confirm the revision loop exits immediately.
3. A/B the loop on your eval set: report score, latency and token cost for draft-only versus draft-plus-revision.

**Self-check:** Exercise 1 — extract numbers with a regex from both strings, subtract the evidence set from the draft set, and report the difference as a located defect per number.

## Go deeper

- [Loop Engineering](10-loop-engineering.md) — budgets and termination for critique loops.
- [LLM as Judge](../15-evals/02-llm-as-judge.md) — turning a critic into a measured scorer.
- [Faithfulness](../15-evals/04-faithfulness.md) — checking claims against retrieved evidence.
- [Agent Evals](../15-evals/06-agent-evals.md) — evaluating whole runs, not single answers.

---

[← Previous: Human in the Loop](08-human-in-the-loop.md) · [Stage index](README.md) · [Next: Loop Engineering](10-loop-engineering.md)
