# 6. Reranking

> **Stage 10 · RAG** — a second, slower model that reads the query and each candidate together and fixes the ordering.
>
> **You'll be able to:** rerank the top 50 candidates down to 5 with a cross-encoder, and justify the added latency with measured numbers.
>
> **Prerequisites:** [Query Transformation](05-query-transform.md) · [Similarity Metrics](../09-vector-databases/02-similarity-metrics.md)

## Why this exists

Vector search is fast because it never looks at the query and the document together. Each is embedded independently, then compared by arithmetic. That independence is what makes it scale — and what makes it approximate in a way that matters: the query's embedding does not know which parts of the document are relevant to *this* question.

A reranker does look at both together. It takes the query and a candidate passage as one input and outputs a relevance score. That is far more accurate and far more expensive, so you run it on 50 candidates, not 5 million.

The mental model: **retrieval is a cheap filter, reranking is an expensive reader.** Retrieve wide, read narrow.

## The mental model

```text
question ──► vector search ──► 50 candidates (fast, approximate, recall-oriented)
                                   │
                                   ▼
                        cross-encoder: score (question, passage) 50 times
                                   │
                                   ▼
                              top 5 (slow, accurate, precision-oriented)
                                   │
                                   ▼
                          prompt → answer

  bi-encoder:   embed(q) · embed(d)          independent, cacheable, O(1) per doc
  cross-encoder: model(q, d) → score         joint, not cacheable, O(n) per query
```

## Core concepts

### Why a cross-encoder is more accurate

A bi-encoder must compress a document into a vector *before knowing the question*. Anything the question cares about that the vector did not capture is lost. A cross-encoder sees the pair and can attend to the exact sentence that answers the question — including negation, numbers and named entities, which are precisely where embeddings are weakest.

The cost structure is why you cannot use it for retrieval: scores cannot be precomputed, so every candidate needs a forward pass. Fifty candidates is fine; fifty thousand is not.

### Choose the cut from the numbers, not from habit

The pattern that works: retrieve `k=50`, rerank, keep 5. The 50 is generous because the reranker can only reorder what it is given — retrieval recall at 50 sets the ceiling. The 5 is small because prompt tokens are the expensive part of generation.

If your retrieval recall@50 is poor, reranking cannot save you. That is why [RAG Evaluation](12-rag-evals.md) measures context recall before anything else: reranking improves precision, not recall.

### Score thresholds beat fixed counts for refusal

Reranker scores are comparable within a query, and with cross-encoders they are often used with a threshold: if the best score is below it, return "not in the documents" instead of five irrelevant passages. This is a much better refusal signal than a vector similarity threshold, because the score reflects joint relevance.

Calibrate the threshold on labelled data — a set of answerable and unanswerable questions — and store it in config with the model id. It is model-specific and will change when you change rerankers.

### Order matters to the model, so order your context deliberately

Models attend more reliably to the beginning and end of a long context than to the middle. With five passages, the effect is modest; with twenty it is real. Sort reranked passages by score descending — the strongest first — and keep the count low enough that the middle is not a dead zone.

### Latency is predictable and worth paying

A cross-encoder pass over 50 candidates is 50 short forward passes, batchable on a GPU and roughly linear in candidate count on CPU. Measure it: time a batch of 50 on your hardware, and compare with the retrieval time. If reranking adds more latency than your budget allows, cut candidates to 30 or use a smaller reranker — do not drop reranking, which is usually the single biggest quality win per line of code in RAG.

## Code

Rerank with a cross-encoder from `sentence-transformers`, batching candidates and keeping a score threshold for refusal.

```python
from __future__ import annotations

import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # check the hub for current ids

@dataclass
class Candidate:
    id: str
    text: str
    source: str
    page: int
    retrieve_score: float

class Reranker:
    def __init__(self, model_id: str = RERANKER_ID, threshold: float = 0.0) -> None:
        self.model = CrossEncoder(model_id)
        self.threshold = threshold       # calibrate per model on labelled data

    def rerank(self, question: str, candidates: list[Candidate],
               keep: int = 5) -> tuple[list[Candidate], float]:
        if not candidates:
            return [], 0.0
        t0 = time.perf_counter()
        pairs = [(question, c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=32)      # one score per pair
        ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(scores, candidates), key=lambda p: -p[0])
        kept = [c for s, c in ranked if s >= self.threshold][:keep]
        return kept, round(ms, 2)
```

Using it in the pipeline: retrieve wide, rerank, and refuse when nothing clears the threshold. Note that the refusal is decided here, before any generation call — which saves money and prevents a hallucinated answer.

```python
from __future__ import annotations

def answer_with_rerank(question: str, retriever, reranker: Reranker, generate) -> dict:
    candidates = retriever.search(question, k=50)          # recall-oriented
    kept, rerank_ms = reranker.rerank(question, candidates, keep=5)

    if not kept:
        return {"answer": "Not found in the provided documents.",
                "sources": [], "rerank_ms": rerank_ms, "n_candidates": len(candidates)}

    text, used = generate(question, kept)                  # generate from the top 5 only
    return {"answer": text, "sources": used,
            "rerank_ms": rerank_ms, "n_candidates": len(candidates),
            "top_score": kept[0].retrieve_score}
```

## Real use case

A financial-research assistant answered questions over quarterly filings. Vector-only retrieval put the right filing in the top 50 reliably, but the top 5 was noisy: chunks about the same company and quarter, with the wrong metric. "What was the gross margin in Q2" returned chunks mentioning margin in a risk-factor discussion rather than the income statement.

They added a cross-encoder reranker over the same 50 candidates. Because the reranker reads the question and the passage together, the chunk containing "Gross margin 41.2%" alongside the question's words outranked the risk-factor paragraph that merely mentioned margins. They measured context precision before and after on a 50-question set with labelled answer chunks, and the improvement was large enough that the added latency was an easy trade.

They also set a reranker threshold calibrated on 20 unanswerable questions, so questions about periods the corpus did not cover returned a refusal rather than the nearest plausible filing. That change reduced the most damaging error class — a confident answer citing a filing from the wrong quarter.

The failure mode before: retrieval was good and answers were still wrong, because the model was handed five passages that shared a topic but not the fact.

## Common mistakes

- **Mistake:** reranking only 5 candidates — **Instead:** rerank 30–50; the reranker cannot reorder what it never sees.
- **Mistake:** expecting reranking to fix missing content — **Instead:** measure retrieval recall@50 first; reranking improves precision only.
- **Mistake:** passing whole documents to the reranker — **Instead:** pass chunks; cross-encoder input length is limited and long inputs dilute the signal.
- **Mistake:** using one threshold across rerankers — **Instead:** calibrate per model on labelled answerable/unanswerable questions and store it in config.
- **Mistake:** reranking on CPU in a latency-critical path without measuring — **Instead:** time a batch of 50 first, then choose candidate count and model size.
- **Mistake:** keeping all 50 reranked passages in the prompt — **Instead:** keep 3–5; more context is not better context.

## Checkpoint

1. Rerank a set of 50 candidates for one question with a cross-encoder and print the top 10 with scores. Compare the ordering with the retrieval order and note where they disagree.
2. Time the rerank of 20, 50 and 100 candidates on your hardware and plot the relationship. Decide your candidate count from the numbers.
3. Calibrate a threshold: score 20 answerable and 20 unanswerable questions, plot the best-score distributions, and pick a threshold that keeps most answerable cases while rejecting most unanswerable ones.

**Self-check:** the cross-encoder usually promotes the passage containing the literal answer and demotes passages that merely share topic vocabulary — if it does not, check that you are passing chunk text and not the full document.

## Go deeper

- [Hybrid Retrieval](07-hybrid-retrieval.md) — improving the candidate list before it reaches the reranker.
- [RAG Evaluation](12-rag-evals.md) — measuring context precision, the metric reranking moves.
- [LLM-as-Judge](../15-evals/02-llm-as-judge.md) — the alternative when you need a reranker for reasoning-heavy relevance.

---
[← Previous: 5. Query Transformation](05-query-transform.md) · [Stage index](README.md) · [Next: 7. Hybrid Retrieval](07-hybrid-retrieval.md)
