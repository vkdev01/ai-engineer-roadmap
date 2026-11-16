# 1. RAG Architecture

> **Stage 10 · RAG** — the whole pipeline in one picture, plus the failure map that tells you which stage to blame.
>
> **You'll be able to:** draw the RAG pipeline from memory, and diagnose a bad answer by locating its stage instead of guessing.
>
> **Prerequisites:** [Stage 09 — Vector Databases](../09-vector-databases/README.md) · [Semantic Search End to End](../09-vector-databases/03-semantic-search.md)

## Why this exists

A language model only knows what it was trained on, up to a cutoff, with no access to your documents and no idea what happened last week. Ask it about your company's refund policy and it will either say it does not know or invent something plausible. The second outcome is the dangerous one.

RAG (retrieval-augmented generation) fixes this the obvious way: before answering, look things up, put the passages in the prompt, and instruct the model to answer from them. The model stops being a knowledge store and becomes a reasoning layer over text you control.

The mental model: **RAG is an open-book exam.** The retrieval step finds the right pages; the model writes the answer while looking at them. Every failure is either "the right page was not found" or "the model did not use the page it was given".

## The mental model

```text
OFFLINE (per corpus change)                            ONLINE (per question)
──────────────────────────                             ─────────────────────
documents ─► parse ─► chunk ─► embed ─► store          question
                                          │               │
                                     metadata,            ├─► transform (rewrite / HyDE)
                                     filters              ├─► retrieve  (vector + BM25)
                                                          ├─► rerank    (top 50 → top 5)
                                                          ├─► build prompt with citations
                                                          └─► generate ─► answer + sources

  failure map:
  parse      → content missing entirely         ("we never had that page")
  chunk      → answer split across two chunks   (retrieval half-right)
  embed      → paraphrase missed                (nothing relevant returned)
  filter     → right doc, wrong tenant          (or zero results)
  rerank     → right doc at rank 40, cut at 5   (lost in the middle)
  prompt     → model ignores context            (hallucinates anyway)
  generate   → correct context, wrong answer    (reasoning failure)
```

## Core concepts

### Two phases, and only one of them is per-query

The **indexing phase** runs when documents change: parse, chunk, embed, store. It is batch, it is slow, it can take hours, and it is where most quality is decided. The **query phase** runs per request and must be fast: transform, retrieve, rerank, prompt, generate.

Beginners optimise the query phase because it is the part they can see. In practice, a broken parser or a 2,000-token chunk caps your quality no matter how clever the prompt is.

### Retrieval is a recall problem; generation is a precision problem

Retrieval must not miss the passage that contains the answer — that is recall. Generation must not include claims the passage does not support — that is faithfulness. They fail differently and need different metrics, which is why [RAG Evaluation](12-rag-evals.md) keeps four numbers instead of one.

The practical consequence: **retrieve more than you need, then narrow**. Fetch 50 candidates, rerank to 5. Retrieval is cheap; a wrong answer is expensive.

### The prompt is a contract, not a suggestion

The generation prompt has four jobs: put the context in a clearly delimited block, number the passages, instruct the model to answer only from them, and require a citation per claim with a defined behaviour when the context is insufficient. That last part — "if the context does not contain the answer, say so" — is what turns a hallucination into an honest refusal.

Vague prompts like "use the following context to answer" leave the model free to blend its priors with your documents, which produces answers that are 90% right and impossible to audit.

### Every answer must carry its sources

Return the chunk ids used, and render them as citations in the UI. Two benefits: users can verify, and you can debug. When a user reports a wrong answer, the cited chunk tells you instantly whether retrieval or generation failed. Without citations you are reading tea leaves.

### Cost and latency live in the context window

Prompt tokens scale with the number of chunks × chunk size. Five chunks of 400 tokens is roughly 2,000 tokens of context per request — fine. Twenty chunks of 1,000 tokens is 20,000, which is slower, more expensive, and measurably harder for models to use well. Reranking to a small, high-precision set is the cheapest optimisation in the whole pipeline.

## Code

A complete minimal pipeline: retrieve, build a numbered context block, call the model, return the answer with the sources it cited. This is the skeleton every later file in this stage refines.

```python
from __future__ import annotations

import re
from dataclasses import dataclass

from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-sonnet-4-5"   # check the provider's docs for the current model id

SYSTEM = """Answer using only the numbered context passages.
Cite the passage number after each claim, like [2].
If the context does not contain the answer, reply exactly: NOT_IN_CONTEXT."""

@dataclass
class Passage:
    id: str
    text: str
    source: str

def build_context(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{i}] ({p.source})\n{p.text}" for i, p in enumerate(passages, start=1))

def answer(question: str, passages: list[Passage]) -> tuple[str, list[str]]:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{build_context(passages)}\n\nQuestion: {question}"}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    used = [passages[i - 1].id for i in cited if 1 <= i <= len(passages)]
    return text, used
```

Two behaviours to check before building anything else: a question whose answer is in the passages returns a cited answer, and a question whose answer is not there returns `NOT_IN_CONTEXT` rather than a guess. The second is the one that usually fails first, and it fails because the prompt allowed it.

The second snippet makes the failure map measurable: instrument each stage so a slow or empty result is attributable.

```python
from __future__ import annotations

import time
from contextlib import contextmanager

@contextmanager
def stage(name: str, timings: dict[str, float]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - t0) * 1000, 2)

def run_pipeline(question: str, retriever, reranker, generator) -> dict:
    timings: dict[str, float] = {}
    with stage("retrieve", timings):
        candidates = retriever.search(question, k=50)
    with stage("rerank", timings):
        top = reranker.rerank(question, candidates, keep=5)
    with stage("generate", timings):
        text, used = generator(question, top)
    return {"answer": text, "sources": used, "n_candidates": len(candidates),
            "timings_ms": timings, "empty": len(candidates) == 0}
```

Log that dict for every request. When quality drops, `n_candidates == 0` versus "50 candidates but the answer is wrong" splits your debugging in half.

## Real use case

A bank's internal policy assistant answered questions from 800 HR and compliance PDFs. Early users reported confident, wrong answers about leave encashment. The team's first instinct was to rewrite the system prompt, which changed nothing.

Adding the instrumentation above settled it in one afternoon. Retrieval was returning 50 candidates every time — not empty — but the answer-bearing passage was a table inside a PDF, and their parser had dropped tables entirely. The model was faithfully summarising the prose around the table, which discussed encashment without giving the numbers.

The fix was in the indexing phase: a parser that preserved tables as markdown. Same prompt, same model, same retrieval settings, and the answers became correct. The team's own note afterwards was that they had spent three days tuning a generation prompt for a parsing bug.

The design choice that made the diagnosis possible: `n_candidates` and per-stage timings logged for every request, plus citations returned to the UI so a user could point at the exact passage that misled them.

## Common mistakes

- **Mistake:** tuning the generation prompt to fix missing content — **Instead:** check whether retrieval returned the passage at all before touching the prompt.
- **Mistake:** instructing the model to "use the context" without a refusal path — **Instead:** define an explicit insufficient-context response and test it.
- **Mistake:** stuffing 20 chunks into the prompt — **Instead:** retrieve wide, rerank narrow, and keep context small and precise.
- **Mistake:** returning an answer with no source ids — **Instead:** require citations in the prompt and return the used chunk ids with the response.
- **Mistake:** rebuilding the index on every request — **Instead:** separate the batch indexing job from the query path, with a manifest.
- **Mistake:** assuming a higher temperature makes answers "more natural" — **Instead:** keep temperature low for grounded answers; fluency is not the goal, accuracy is.

## Checkpoint

1. Build the minimal pipeline above over ten documents and ask three questions: one answerable, one answerable only by paraphrase, one not answerable. Record which behaviour each produced.
2. Add the `run_pipeline` instrumentation and log ten requests. Identify which stage dominates latency.
3. Deliberately break one stage — drop tables at parse time, or truncate chunks mid-sentence — and observe how the failure presents in the logs. Write down the symptom-to-stage mapping you find.

**Self-check:** the answerable question returns a cited answer, the paraphrase question usually returns the right passage but may cite the wrong number, and the unanswerable one must return `NOT_IN_CONTEXT` — if it invents an answer, the refusal path is not being enforced.

## Go deeper

- [Ingestion and Parsing](02-ingestion-parsing.md) — the stage that decides your quality ceiling.
- [RAG Evaluation](12-rag-evals.md) — turning "looks better" into four numbers.
- [RAG Design](../21-system-design/02-rag-design.md) — the architecture decisions at production scale.

---
[← Previous: Stage 09 — Vector Databases](../09-vector-databases/README.md) · [Stage index](README.md) · [Next: 2. Ingestion and Parsing](02-ingestion-parsing.md)
