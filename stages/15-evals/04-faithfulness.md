# 4. Faithfulness

> **Stage 15 · Evals and Testing** — measuring whether the answer is supported by the context it was given, which is not the same as whether the answer is true.
>
> **You'll be able to:** decompose an answer into claims, grade each against the sources, and use the resulting number to distinguish a retrieval problem from a generation problem.
>
> **Prerequisites:** [LLM as Judge](02-llm-as-judge.md) · [RAG Evaluation](../10-rag/12-rag-evals.md) · [System Instructions](../08-prompt-and-context/01-system-instructions.md)

## Why this exists

A model can produce an answer that is true, useful and completely unsupported by the documents you gave it — because it remembered the fact from pretraining. That answer is a liability: you cannot cite it, you cannot audit it, and when the policy changes, it stays confidently wrong.

Faithfulness measures the property you can control: **is every claim in the answer traceable to the provided context?** It is the strictest useful measure in a grounded system, and the one that separates a RAG product from a chat interface with documents in the prompt.

The mental model: **faithfulness is a claim-by-claim audit of the answer against its sources.**

## The mental model

```text
   answer: "Refunds are allowed within 30 days [p3]. Damaged goods
            are covered for 90 days [p7]. Shipping is free." 
                     │                    │              │
                     ▼                    ▼              ▼
              supported by p3      supported by p7    NOT in context
                                                        (UNSUPPORTED)

   faithfulness = supported claims / total claims = 2/3
```

Score per claim, not per answer. The fraction is what you optimise; the list of unsupported claims is what you fix.

## Core concepts

### Three verdicts, not two

A claim can be supported, contradicted, or absent. Collapsing contradictions into "unsupported" hides the worst failure: an answer that says the opposite of the source. Keep them separate — a contradiction is a bug, an unsupported claim is a discipline problem.

### Decomposition is where the work is

Long sentences hide several claims. "Refunds are available within 30 days except for custom items, which are non-refundable" contains three. A judge that grades the sentence as a whole will mark it supported if two of three hold.

Ask for atomic claims explicitly, and cap the count (eight is a practical ceiling) so a rambling answer does not produce a 40-claim audit nobody reads.

### Faithfulness is not relevance, and not correctness

Keep the three separate in your scorecard:

| Measure | Question | Failure it catches |
|---|---|---|
| Faithfulness | Is each claim supported by the context? | model answering from memory |
| Correctness | Is the answer true? | wrong retrieval, wrong reasoning |
| Relevance | Does it answer what was asked? | grounded non-answer |

A system can be perfectly faithful and useless (it answers the wrong question from the right document). Reporting only faithfulness lets you ship that.

### The citation check is a cheap proxy

Before any judge runs, check mechanically: does every cited id exist in the retrieved set, and does the text near the citation actually contain the claim? The first is exact and free; it catches the most common failure, where a model cites a plausible-looking id.

## Code

A claim-level faithfulness grader with a mechanical pre-check.

```python
from __future__ import annotations

import re

from openai import OpenAI

client = OpenAI()

DECOMPOSE = """Break the ANSWER into atomic factual claims. Each claim must be a single
verifiable statement, rewritten to stand alone (resolve pronouns). Maximum 8 claims.
Return JSON: {"claims": [str]}"""

GRADE = """You are auditing an answer against its sources.

For each claim, return one verdict:
- SUPPORTED: the sources state it (quote the span).
- CONTRADICTED: the sources state the opposite.
- UNSUPPORTED: the sources say nothing about it.

Return JSON: {"verdicts": [{"claim": str, "verdict": str, "evidence": str}]}
Evidence must be a verbatim span from the sources, or "" if none."""


def decompose(answer: str) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                 # check current model ids in provider docs
        messages=[{"role": "system", "content": DECOMPOSE},
                  {"role": "user", "content": answer}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    import json
    return json.loads(resp.choices[0].message.content or "{}").get("claims", [])[:8]


def grade(claims: list[str], sources: list[dict]) -> list[dict]:
    ctx = "\n\n".join(f"[{s['id']}] {s['text'][:1500]}" for s in sources)
    import json
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": GRADE},
                  {"role": "user", "content": f"SOURCES:\n{ctx}\n\nCLAIMS:\n" + "\n".join(f"- {c}" for c in claims)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}").get("verdicts", [])


def citation_check(answer: str, allowed_ids: set[str]) -> dict:
    cited = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", answer))
    return {"cited": sorted(cited), "invalid": sorted(cited - allowed_ids), "ok": cited.issubset(allowed_ids)}


def faithfulness(answer: str, sources: list[dict]) -> dict:
    mech = citation_check(answer, {s["id"] for s in sources})
    claims = decompose(answer)
    verdicts = grade(claims, sources)
    n = len(verdicts) or 1
    counts = {k: sum(1 for v in verdicts if v.get("verdict") == k) for k in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")}
    return {
        "score": round(counts["SUPPORTED"] / n, 3),
        "contradicted": counts["CONTRADICTED"],
        "unsupported": [v["claim"] for v in verdicts if v.get("verdict") == "UNSUPPORTED"],
        "citations_ok": mech["ok"],
        "invalid_citations": mech["invalid"],
    }
```

A scorecard that reads well:

```text
faithfulness            0.94   (target >= 0.90)
contradictions          0      (must be 0)
citations valid         1.00
worst case              q17: 2 unsupported claims about 2023 policy
```

TypeScript, for a release gate:

```ts
type Faith = { score: number; contradicted: number; citations_ok: boolean };

export function gate(f: Faith, min = 0.9): { pass: boolean; reason?: string } {
  if (f.contradicted > 0) return { pass: false, reason: `${f.contradicted} contradicted claims` };
  if (!f.citations_ok) return { pass: false, reason: "invalid citation id" };
  if (f.score < min) return { pass: false, reason: `faithfulness ${f.score} < ${min}` };
  return { pass: true };
}
```

## Real use case

An insurance assistant answers coverage questions from policy documents. Its answers are accurate — audited by humans, mostly true — but a compliance review flags that a third of them contain at least one sentence with no basis in the cited document. The model is filling gaps from general knowledge about insurance.

The team adds faithfulness grading to the release gate with a hard rule: contradictions must be zero, faithfulness must be at or above 0.90. Two changes follow. The prompt is rewritten to require a citation per sentence and to permit "the documents do not state this". And the ingestion pipeline is fixed, because the unsupported claims cluster around a document that failed to parse.

The second finding is the important one. **Faithfulness failures are often ingestion failures wearing a costume** — the model was not inventing, it was answering from a document that never made it into the index.

## Common mistakes

- **Mistake:** grading the whole answer with one score — **Instead:** decompose into claims; the per-claim list is your debugging queue.
- **Mistake:** treating contradictions as unsupported claims — **Instead:** separate verdicts; a contradiction is a bug, not a style issue.
- **Mistake:** no mechanical citation check — **Instead:** assert every cited id is in the retrieved set; it is free and catches the most common error.
- **Mistake:** optimising faithfulness alone — **Instead:** report relevance alongside; a faithful non-answer is still a failure.
- **Mistake:** letting "not stated in the documents" count as a low score — **Instead:** count it as correct behaviour when the corpus genuinely lacks the fact.
- **Mistake:** grading against the model's own output as the source — **Instead:** grade against retrieved passages only; self-agreement is not faithfulness.

## Checkpoint

1. Take five answers from your system, decompose each into claims by hand, and grade them yourself. Compare with the grader's verdicts and count disagreements.
2. Add the citation pre-check to your test suite and confirm it catches an answer with a fabricated id.
3. Fix one unsupported claim you found — usually by improving retrieval or adding an instruction — and re-run to show the score move.

**Self-check:** if faithfulness is high but users still report invented details, your claims are too coarse — tighten decomposition before touching the prompt.

## Go deeper

- [Hallucinations](05-hallucinations.md) — the broader failure this measure narrows
- [RAG Evaluation](../10-rag/12-rag-evals.md) — the four numbers this belongs with
- [Guardrails](../20-ai-security/09-guardrails.md) — enforcing grounded output at request time

---
[← Previous: Golden Datasets](03-golden-datasets.md) · [Stage index](README.md) · [Next: Hallucinations](05-hallucinations.md)
