# 5. Hallucinations

> **Stage 15 · Evals and Testing** — detecting, classifying and reducing invented content, with the recognition that "hallucination" names several different bugs.
>
> **You'll be able to:** classify a hallucination by cause, build detection that fits the cause, and choose a mitigation that addresses it instead of adding another disclaimer.
>
> **Prerequisites:** [Faithfulness](04-faithfulness.md) · [RAG Architecture](../10-rag/01-rag-architecture.md) · [Context Engineering](../08-prompt-and-context/08-context-engineering.md)

## Why this exists

"Hallucination" is used as one word for at least five different failures, and each has a different fix. Teams that treat it as one problem add "do not hallucinate" to the system prompt, which does approximately nothing, and then conclude the model is unfixable.

The productive move is to stop asking "why does it make things up" and start asking "which of these five things just happened". Classification is the whole job; the mitigation follows mechanically.

The mental model: **a hallucination is a symptom; the cause is in the pipeline, and it is usually retrieval.**

## The mental model

```text
   user complaint: "it made something up"
                    │
     ┌──────────────┼───────────────┬────────────────┬─────────────────┐
     ▼              ▼               ▼                ▼                 ▼
 1 context      2 context       3 blending        4 over-          5 confident
   missing        wrong           sources           generalisation    gap-filling
     │              │               │                │                 │
 retrieval      ranking          citation        prompt scope      "say I don't know"
   miss           error           discipline      + examples        + refusal tests
```

In practice, causes 1 and 2 account for most reported hallucinations in RAG systems. Cause 5 is what people mean when they say "the model just makes things up", and it is the easiest to fix.

## Core concepts

### The five causes, with the tell

| Cause | Tell in the trace | Fix |
|---|---|---|
| Missing context | retrieved set has no passage with the answer | fix retrieval, chunking, coverage |
| Wrong context | passage retrieved is topically similar, not answer-bearing | reranker, better query |
| Blending | answer mixes two passages into one wrong claim | citation per sentence, tighter context |
| Over-generalisation | answer states a rule more broadly than the source | scope in the prompt, examples of narrow answers |
| Gap-filling | no context needed; model answers from pretraining | require citations, allow refusal, test refusals |

### Detection that fits the cause

For causes 1-3 you need the retrieved context, so detection is a faithfulness check. For cause 5 you need a different signal: an answer containing specifics (numbers, dates, names) that appear nowhere in the context. That is a cheap regex-plus-context search, and it catches most gap-filling without a judge call.

```python
import re

SPECIFIC = re.compile(r"\b(\d{1,4}(?:[.,]\d+)?%?|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ (?:Ltd|Inc|Pvt|LLC))\b")

def unsupported_specifics(answer: str, context: str) -> list[str]:
    """Numbers, dates and entity names in the answer that never appear in the context."""
    ctx = context.lower()
    out = []
    for match in SPECIFIC.findall(answer):
        token = match if isinstance(match, str) else match[0]
        if token.lower() not in ctx:
            out.append(token)
    return out
```

This is deliberately dumb and it works: a claim about "90 days" when the context never says 90 is either a retrieval problem or an invention, and either way you want to know before the user does.

### Refusal must be a tested behaviour

The single most effective mitigation is making "the documents do not state this" a first-class, rewarded output. That means an instruction, an example, and — critically — test cases where refusal is the only correct answer. If refusal is never tested, it will never be reliable.

### Do not promise zero hallucinations

You cannot verify every claim at runtime for every answer, and claiming otherwise is a compliance risk. What you can promise: grounded answers with citations, mechanical checks on specifics, a measured faithfulness score, and a feedback path. Write that in your README instead of "no hallucinations".

## Code

A classifier that routes a failing case to the right fix.

```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Diagnosis:
    cause: str
    evidence: str
    fix: str

FIXES = {
    "missing_context": "improve retrieval coverage: chunking, hybrid search, ingestion",
    "wrong_context": "add a reranker; tighten the query; check metadata filters",
    "blending": "require a citation per sentence; reduce context to the top 3 passages",
    "over_generalisation": "narrow the prompt scope; add examples of precise answers",
    "gap_filling": "require citations; add refusal instruction and refusal test cases",
}


def diagnose(answer: str, context: list[dict], faithful: dict) -> list[Diagnosis]:
    """Classify each unsupported claim into one of the five causes."""
    ctx_text = "\n".join(c["text"] for c in context).lower()
    out: list[Diagnosis] = []

    for claim in faithful.get("unsupported", []):
        specifics = unsupported_specifics(claim, ctx_text)
        if not context:
            out.append(Diagnosis("missing_context", "no passages retrieved", FIXES["missing_context"]))
        elif specifics:
            # a specific that appears nowhere: either it was never retrieved or it was invented
            out.append(Diagnosis("gap_filling", f"unsupported specifics: {specifics}", FIXES["gap_filling"]))
        elif any(word in claim.lower() for word in ("all ", "any ", "always", "never", "every ")):
            out.append(Diagnosis("over_generalisation", f"absolute claim: {claim[:90]}", FIXES["over_generalisation"]))
        else:
            out.append(Diagnosis("blending", f"claim not traceable to one passage: {claim[:90]}", FIXES["blending"]))

    return out


def summary(diagnoses: list[Diagnosis]) -> dict:
    counts: dict[str, int] = {}
    for d in diagnoses:
        counts[d.cause] = counts.get(d.cause, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
```

TypeScript, for a monitoring dashboard:

```ts
type Diag = { cause: string; count: number };

export function topFix(diags: Diag[]): string {
  const top = [...diags].sort((a, b) => b.count - a.count)[0];
  if (!top) return "no failures recorded";
  switch (top.cause) {
    case "missing_context":
      return "retrieval coverage — re-ingest and check chunk overlap";
    case "wrong_context":
      return "ranking — add or tune the reranker";
    case "gap_filling":
      return "prompt — require citations, add refusal cases";
    default:
      return "review the top failing cases by hand";
  }
}
```

## Real use case

A pharma medical-information team runs a grounded assistant over approved product documentation. Clinicians report "invented dosages". The first instinct — a stronger model — changes nothing.

They add `unsupported_specifics` and run it over 200 real answers. Of 31 answers with unsupported numbers, 24 trace back to a document section that the parser had dropped because it lived in a table; the retrieval was fine, the ingestion was not. Six were gap-filling about a contraindication the corpus did not cover; the fix is a refusal instruction plus three test cases. One was a genuine blending error.

That split — 24 ingestion, 6 prompt, 1 model — is typical, and it is only visible if you classify. The team's subsequent rule is written in their runbook: **before changing the model or the prompt, count the causes.**

## Common mistakes

- **Mistake:** adding "do not hallucinate" to the system prompt — **Instead:** classify the failure; the instruction addresses at most one of five causes.
- **Mistake:** upgrading the model to fix invented details — **Instead:** check ingestion and retrieval first; the model is usually faithful to bad context.
- **Mistake:** no refusal test cases — **Instead:** make refusal a scored behaviour with its own cases.
- **Mistake:** treating every unsupported claim as invention — **Instead:** separate missing context from gap-filling; the fixes are in different parts of the system.
- **Mistake:** promising zero hallucinations in a README or a contract — **Instead:** promise citations, mechanical checks, a measured score and a feedback path.
- **Mistake:** fixing cases one by one from user reports — **Instead:** count causes weekly and fix the biggest bucket.

## Checkpoint

1. Run `unsupported_specifics` over twenty of your system's answers and list every number or date that is not in the context.
2. Classify ten failing cases with `diagnose` and count the causes. Write one sentence on which bucket you will fix first.
3. Add three refusal test cases and confirm your prompt passes them. If it does not, fix the instruction before anything else.

**Self-check:** if most of your failures classify as `missing_context`, you do not have a hallucination problem — you have a retrieval problem, and no prompt change will fix it.

## Go deeper

- [Faithfulness](04-faithfulness.md) — the measure this classification depends on
- [Ingestion and Parsing](../10-rag/02-ingestion-parsing.md) — where most of these bugs start
- [Guardrails](../20-ai-security/09-guardrails.md) — runtime checks on output

---
[← Previous: Faithfulness](04-faithfulness.md) · [Stage index](README.md) · [Next: Agent Evals](06-agent-evals.md)
