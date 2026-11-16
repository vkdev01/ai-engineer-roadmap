# 13. Ship: A RAG Service with Evals

> **Stage 10 · RAG** — the whole pipeline, served behind an API, with citations, an eval harness and a README that reports every step's score.
>
> **You'll be able to:** ship a RAG service that answers with citations, measure it against a frozen dataset, and improve it in three measured steps.
>
> **Prerequisites:** [RAG Evaluation](12-rag-evals.md) · [Hybrid Retrieval](07-hybrid-retrieval.md) · [Ship: A Streaming API](../17-ai-backend/09-ship-streaming-api.md)

## Why this exists

Every file in this stage taught one improvement. This project proves you can sequence them: build the naive version, measure it, add one thing, measure again. That sequence — not the final architecture — is the skill interviewers and reviewers actually test, because it is the difference between a system you can improve and a system you can only hope about.

The deliverable is small on purpose: one corpus, one endpoint, one eval file, four numbers before and after.

The mental model: **a RAG service is a retrieval pipeline plus a citation contract plus a score you can reproduce.**

## The mental model

```text
 ingest/          serve/                    eval/
 -------          ------                    -----
 parse ─► chunk   POST /answer              cases.jsonl (40)
   │        │        │                        │
 embed ─► store   hybrid search ─► rerank     run.py ─► report.json
                    │                          │
              parent section ─► prompt ─► answer + [citations]
                                               │
                                    recall · precision · faithfulness · relevance
```

Four artefacts, one command each: `ingest`, `serve`, `eval`. If you cannot run those three commands from a clean checkout, the project is not finished.

## Core concepts

### The citation contract

Decide before you write code: every sentence in the answer either cites a passage id from the retrieved set, or is explicitly a summary with no citation. Validate that mechanically on every response, not just in evals — a response that cites an id it never retrieved is a bug, not a style issue.

```python
import re

def enforce_citations(answer: str, allowed_ids: set[str]) -> tuple[str, bool]:
    cited = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", answer))
    bad = cited - allowed_ids
    if bad:
        # do not silently strip: log, and return a refusal the caller can act on
        return ("I could not ground this answer in the retrieved sources.", False)
    return (answer, True)
```

### Parent-child, because chunk size is a compromise

Index small chunks (good for matching) and hand the model the parent section (good for answering). This one change fixes the most common "the answer was there but the model did not have enough of it" failure.

```python
def build_context(hits: list[dict], max_tokens: int = 3000) -> list[dict]:
    """Swap child chunks for their parent sections, dedup, then budget."""
    seen: set[str] = set()
    out: list[dict] = []
    used = 0
    for h in hits:
        parent = h["parent_id"]
        if parent in seen:
            continue
        seen.add(parent)
        text = PARENTS[parent]
        cost = len(text) // 4                     # rough token estimate
        if used + cost > max_tokens:
            continue
        out.append({"id": h["id"], "parent": parent, "text": text})
        used += cost
    return out
```

### Budgets, in three places

Retrieval budget (how many chunks), context budget (how many tokens reach the model), answer budget (max output tokens). Write all three as constants, log them per request, and let the eval harness vary them. Cost regressions hide in unstated defaults.

## Code

The service, minimal but complete.

```text
rag-service/
├── pyproject.toml
├── README.md                 <- the score table lives here
├── src/rag/
│   ├── ingest.py             parse -> chunk -> embed -> upsert (idempotent by doc hash)
│   ├── retrieve.py           hybrid (BM25 + vector, RRF) -> rerank -> parent expansion
│   ├── answer.py             prompt + citation enforcement
│   ├── api.py                POST /answer, POST /search
│   └── config.py             budgets, model ids, index name
├── evals/cases.jsonl         40 cases incl. 5 unanswerable
├── evals/run.py              prints the four numbers, exits non-zero below thresholds
└── tests/test_citations.py   the contract test
```

```python
# src/rag/retrieve.py
from __future__ import annotations

from .config import K_VECTOR, K_BM25, K_RERANK


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: sum 1/(k + rank) across lists. No score calibration needed."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def retrieve(question: str) -> list[dict]:
    dense = vector_search(question, k=K_VECTOR)          # list[str] ids
    lexical = bm25_search(question, k=K_BM25)            # list[str] ids
    fused = rrf([dense, lexical])
    shortlist = fused[: max(K_RERANK * 3, K_RERANK)]
    reranked = rerank(question, shortlist)[:K_RERANK]    # cross-encoder
    return [{"id": doc_id, "parent_id": PARENT_OF[doc_id]} for doc_id in reranked]
```

```python
# evals/run.py
import json
import sys

from rag.answer import answer_question
from rag.eval import evaluate_rag, load_cases, make_judge

THRESHOLDS = {"context_recall": 0.75, "faithfulness": 0.85, "answer_relevance": 0.80, "citations_valid": 1.0}


def main() -> int:
    cases = load_cases("evals/cases.jsonl")
    report = evaluate_rag(answer_question, cases, make_judge())
    print(json.dumps({k: report[k] for k in THRESHOLDS}, indent=2))
    failed = {k: (report[k], v) for k, v in THRESHOLDS.items() if report[k] < v}
    if failed:
        print("BELOW THRESHOLD:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### The score table in the README

| Step | Change | Recall | Precision | Faithfulness | Relevance | Cost / 1k answers |
|---|---|---|---|---|---|---|
| 0 | Fixed 400-token chunks, dense only, top 6 | 0.58 | 0.61 | 0.90 | 0.84 | baseline |
| 1 | + hybrid retrieval (BM25 + RRF) | 0.71 | 0.58 | 0.89 | 0.85 | +embedding only |
| 2 | + cross-encoder rerank, top 20 → 5 | 0.78 | 0.79 | 0.91 | 0.87 | +rerank calls |
| 3 | + parent-child context expansion | 0.83 | 0.77 | 0.93 | 0.90 | +input tokens |

The numbers are illustrative — yours will differ, and that is fine. What must not differ is the shape: one row per change, one column per measure, and the cost column filled in so the trade is visible.

## Real use case

A legal-ops team ships this for questions over a 4,000-document contract library. The naive pipeline answered confidently and wrongly on questions whose answer spanned a definitions section and a clause — the classic two-chunk failure. Parent-child expansion fixed those; hybrid retrieval fixed the other cluster, questions containing contract numbers that dense embeddings could not match.

The step that mattered most was not technical. Because every change was scored against a frozen 40-case set, they could reject a tempting idea — a much larger model — when the eval showed faithfulness already at 0.93 and recall unchanged. They spent the money on a reranker instead, and the numbers moved.

## Common mistakes

- **Mistake:** shipping without an eval file — **Instead:** the 40 cases are part of the deliverable; a RAG service without them is a demo.
- **Mistake:** changing three things at once and re-measuring — **Instead:** one change, one row in the table, one re-run.
- **Mistake:** stripping invalid citations and returning the answer anyway — **Instead:** refuse and log; silent stripping hides a real retrieval problem.
- **Mistake:** no unanswerable cases — **Instead:** five of your forty should have no answer in the corpus, and refusal is a scored behaviour.
- **Mistake:** ingesting without idempotency, so re-runs duplicate documents — **Instead:** key on a content hash and upsert.
- **Mistake:** a README that describes the architecture but not the results — **Instead:** put the score table first; it is the only part a reviewer trusts.

## Checkpoint

1. Build the naive pipeline (step 0) and record all four numbers plus the refusal rate on unanswerable cases.
2. Add hybrid retrieval, re-run, and add the row. Then add reranking and add that row. Stop when a change does not move recall.
3. Break something on purpose — halve the chunk size — and confirm your eval catches the regression and exits non-zero.

**Self-check:** if your table has a row where recall improved but faithfulness fell, you have found the normal tension in RAG work; write one sentence in the README explaining which you preferred and why.

## Go deeper

- [RAG Evaluation](12-rag-evals.md) — the measures behind the table
- [Ship: A Streaming API](../17-ai-backend/09-ship-streaming-api.md) — serving this with streaming
- [Production RAG](../22-real-projects/03-production-rag.md) — the multi-tenant version of this project

---
[← Previous: RAG Evaluation](12-rag-evals.md) · [Stage index](README.md) · [Next stage: AI Agents](../11-agents/README.md)
