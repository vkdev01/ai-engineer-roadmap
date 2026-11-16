# 12. RAG Evaluation

> **Stage 10 · RAG** — measuring the two halves separately, so you know whether the retriever or the generator is losing your answers.
>
> **You'll be able to:** build a RAG eval set, compute context recall, context precision, faithfulness and answer relevance, and read the four numbers together as a diagnosis.
>
> **Prerequisites:** [RAG Architecture](01-rag-architecture.md) · [Golden Datasets](../15-evals/03-golden-datasets.md) · [LLM as Judge](../15-evals/02-llm-as-judge.md)

## Why this exists

A RAG system has one output and two failure sources. When an answer is wrong, you cannot tell from the answer alone whether the right passage was never retrieved, or was retrieved and ignored. Teams that measure only answer quality spend weeks rewriting prompts to fix a chunking problem.

The fix is to score the halves separately. Retrieval gets a recall number and a precision number. Generation gets a faithfulness number and a relevance number. Four numbers, four different fixes.

The mental model: **did we fetch the right pages, and did the model use them?**

## The mental model

```text
  question ──► retriever ──► context ──► generator ──► answer
                  │                        │
                  ▼                        ▼
        context recall            faithfulness
        context precision         answer relevance

  low recall      → fix retrieval (chunking, hybrid, reranker)
  low precision   → fix ranking (too much noise in the window)
  low faithful    → fix the prompt / force citations
  low relevant    → fix the question framing or the answer contract
```

Read them as a diagnosis, not a score. High recall with low faithfulness is a prompt problem. Low recall with high faithfulness is a retrieval problem. Both low: your corpus does not contain the answer, and no prompt will save you.

## Core concepts

### Context recall: did the answer's evidence get retrieved?

You need a ground-truth answer that is decomposable into claims, and a check of whether each claim's supporting passage is in the retrieved context. Two practical implementations: human-labelled "required passage ids" per question (most reliable, most work), or a judge that reads the reference answer, the retrieved context, and asks which claims are supported.

```python
RECALL_PROMPT = """You are grading retrieval.
Reference answer: {reference}
Retrieved context: {context}

List each factual claim in the reference answer. For each, say SUPPORTED if the retrieved
context contains it, else MISSING. Return JSON: {{"claims":[{{"claim":..,"status":..}}]}}"""


def context_recall(reference: str, context: str, judge) -> float:
    out = judge(RECALL_PROMPT.format(reference=reference, context=context))
    claims = out["claims"]
    if not claims:
        return 0.0
    return sum(1 for c in claims if c["status"] == "SUPPORTED") / len(claims)
```

### Context precision: how much of what you retrieved was useful

Recall alone rewards stuffing the window. Precision punishes it: of the passages you sent, what fraction actually mattered? The cheap version is a judge that marks each passage relevant or not, and reports the proportion — ideally rank-weighted, so that relevant passages at the top count more.

### Faithfulness: is the answer supported by the context?

This is the anti-hallucination measure, and it is stricter than correctness: an answer can be true and unfaithful. Decompose the answer into claims and check each against the retrieved context only.

```python
FAITHFUL_PROMPT = """Context: {context}
Answer: {answer}

Break the answer into atomic claims. For each claim, mark SUPPORTED (the context states it),
CONTRADICTED, or UNSUPPORTED (context says nothing). Return JSON."""
```

### Answer relevance: does it answer the question asked?

The failure this catches is the well-grounded non-answer: three accurate sentences about the wrong thing. Judge by generating the question the answer implies and comparing it to the original, or by direct rubric grading on a 1-5 scale with an explicit rubric.

### Build the set before you optimise

Thirty to fifty questions is enough to start, if they are the right thirty. Include: questions answerable by one chunk; questions needing two chunks; questions with no answer in the corpus (should return "not found"); questions using vocabulary that differs from the documents; and one question per document type.

```python
import json
from dataclasses import dataclass

@dataclass
class Case:
    qid: str
    question: str
    reference: str | None      # None means "not answerable from the corpus"
    must_cite: list[str]       # passage ids that must appear in context

def load_cases(path: str = "evals/rag_cases.jsonl") -> list[Case]:
    with open(path) as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]
```

## Code

A runner that produces the four numbers plus the split you actually need.

```python
from __future__ import annotations

import json
from statistics import mean

def evaluate_rag(pipeline, cases, judge) -> dict:
    rows = []
    for case in cases:
        result = pipeline(case.question)            # {"context": [...], "answer": str, "cited": [...]}

        if case.reference is None:
            # unanswerable: the only correct behaviour is a refusal
            refused = "not found" in result["answer"].lower()
            rows.append({"qid": case.qid, "refused": refused})
            continue

        context = "\n\n".join(result["context"])
        recall = context_recall(case.reference, context, judge)
        precision = context_precision(case.question, result["context"], judge)
        faithful = faithfulness(result["answer"], context, judge)
        relevant = answer_relevance(case.question, result["answer"], judge)
        cited_ok = all(c in {p["id"] for p in result["context"]} for c in result["cited"])

        rows.append({
            "qid": case.qid, "recall": recall, "precision": precision,
            "faithfulness": faithful, "relevance": relevant, "citations_valid": cited_ok,
        })

    answered = [r for r in rows if "recall" in r]
    refusals = [r for r in rows if "refused" in r]
    return {
        "n": len(rows),
        "context_recall": round(mean(r["recall"] for r in answered), 3),
        "context_precision": round(mean(r["precision"] for r in answered), 3),
        "faithfulness": round(mean(r["faithfulness"] for r in answered), 3),
        "answer_relevance": round(mean(r["relevance"] for r in answered), 3),
        "citations_valid": round(mean(1.0 if r["citations_valid"] else 0.0 for r in answered), 3),
        "refusal_rate_on_unanswerable": round(mean(1.0 if r["refused"] else 0.0 for r in refusals), 3),
        "worst": sorted(answered, key=lambda r: r["recall"])[:5],
    }


if __name__ == "__main__":
    report = evaluate_rag(pipeline, load_cases(), judge)
    print(json.dumps(report, indent=2))
```

The `worst` list matters more than the averages. Five low-recall questions are a debugging queue; a mean of 0.78 is a status bar.

TypeScript, if your eval report is rendered in a dashboard:

```ts
export type RagReport = {
  n: number;
  context_recall: number;
  context_precision: number;
  faithfulness: number;
  answer_relevance: number;
  citations_valid: number;
};

export function verdict(r: RagReport): string {
  if (r.context_recall < 0.7) return "retrieval: recall too low — fix chunking or add hybrid search";
  if (r.context_precision < 0.5) return "retrieval: too much noise — add a reranker";
  if (r.faithfulness < 0.85) return "generation: prompt not forcing citations";
  if (r.answer_relevance < 0.8) return "generation: answer contract unclear";
  return "healthy";
}
```

## Real use case

A support-engineering team ships RAG over product documentation and gets complaints about "confidently wrong" answers. Their first instinct is a better prompt; it does not help.

They build forty cases and discover the split: context recall sits near 0.55 while faithfulness is 0.93. The model is being faithful to context that does not contain the answer — so it does its best with what it has. The fix is retrieval, not prompting: hybrid search plus a reranker lifts recall, and complaint volume drops without a single prompt change.

The second finding is subtler. After recall improves, faithfulness dips, because the model now sees more passages and occasionally blends two of them. Adding an explicit "cite the id for each sentence" instruction restores it. That is the normal shape of this work: fix one half, watch the other, measure again.

## Common mistakes

- **Mistake:** one overall score with no split — **Instead:** always report recall and faithfulness separately; they have different fixes.
- **Mistake:** evaluating on questions you wrote while looking at the documents — **Instead:** collect questions from real users or support tickets, including the badly phrased ones.
- **Mistake:** no unanswerable cases in the set — **Instead:** add at least five; refusal behaviour is a feature and needs a test.
- **Mistake:** using the same model as judge and generator with no calibration — **Instead:** spot-check 20 judge decisions against your own labels and report agreement.
- **Mistake:** reporting a mean and hiding the distribution — **Instead:** always print the worst five cases alongside the mean.
- **Mistake:** re-running evals on a changed dataset — **Instead:** freeze the dataset, version it, and diff the scores against the same file.

## Checkpoint

1. Write fifteen cases for your corpus: five single-chunk, five multi-chunk, five unanswerable. Include the exact passage ids for the multi-chunk ones.
2. Run the evaluator and record all four numbers plus the refusal rate. Save the report as JSON in your repo.
3. Make one retrieval change (add a reranker or hybrid search) and re-run. Write two sentences on which number moved and why.

**Self-check:** if recall and faithfulness are both low, check whether the corpus actually contains the answer before touching the pipeline — a missing document looks exactly like a broken retriever.

## Go deeper

- [Ship: A RAG Service with Evals](13-ship-rag-with-evals.md) — wiring this into the shipped service
- [Faithfulness](../15-evals/04-faithfulness.md) — the general version of this measure
- [Golden Datasets](../15-evals/03-golden-datasets.md) — building the set properly

---
[← Previous: Agentic RAG](11-agentic-rag.md) · [Stage index](README.md) · [Next: Ship: A RAG Service with Evals](13-ship-rag-with-evals.md)
