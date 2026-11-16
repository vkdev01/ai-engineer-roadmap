# 5. Query Transformation

> **Stage 10 · RAG** — rewriting the user's words into something retrieval can actually match, before the search runs.
>
> **You'll be able to:** condense follow-up questions, generate hypothetical answers for HyDE, and decompose multi-part questions.
>
> **Prerequisites:** [RAG Architecture](01-rag-architecture.md) · [Chunking](03-chunking.md)

## Why this exists

Users do not write retrieval queries. They write "and the second one?" after a previous turn, or "why is my bill wrong", or a question that spans two documents. The embedding model sees a short, context-free string and does its best, which is often not enough.

Query transformation inserts a cheap LLM step between the user and the index: turn the messy input into one or more well-formed retrieval queries. It is the highest-leverage per-request improvement in RAG, because a retrieval miss cannot be fixed downstream.

The mental model: **you are translating from conversational language into index language.**

## The mental model

```text
raw input                              transform                        retrieval
─────────────────────────────────      ─────────────────────────       ─────────
"and the second one?"         ──►  condense with history  ──►  "What is the
                     (turn 2 of a thread)                       notice period for
                                                                contractors?"

"why is my bill wrong"        ──►  HyDE: write a fake answer ──►  embed the fake
                                   then embed that answer        answer, not the question

"compare our SLA with the     ──►  decompose into sub-queries ──►  q1: "our SLA
vendor's"                          then retrieve for each        uptime commitment"
                                                                 q2: "vendor SLA
                                                                 uptime commitment"

  cost: +1 LLM call per transform. latency: +0.3–2 s. only use what you measured.
```

## Core concepts

### Condensation fixes the multi-turn problem, and it is non-optional

In a chat interface, the second question is almost always elliptical. "And for contractors?" has no subject. Embedding it retrieves nothing useful, because the vector has no topic.

The fix is a rewrite against the conversation history: a small, fast model call that outputs a standalone question. It is a mechanical task — a few hundred tokens in, one sentence out — and it should be the first transformation you add if your product has a chat UI.

Two rules: preserve the user's terminology where it is domain-specific (do not let the model "improve" a part number), and never let the rewrite introduce facts that were not in the history.

### HyDE: search with a hypothetical answer

Embeddings match *answers to answers* better than *questions to answers*. HyDE (hypothetical document embeddings) exploits this: ask the model to write a short, plausible answer to the question — without retrieval — then embed that generated passage instead of the question, and search with it.

The generated text may be factually wrong, and that is fine: you are only using it as a retrieval probe. Its vocabulary and shape resemble the real answer, so it lands closer in embedding space.

The costs are honest: one extra generation call per query, added latency, and a failure mode where the model's wrong prior pulls retrieval toward the wrong region. Measure it; on well-formed corpora with technical vocabulary it often helps, on short factual lookups it can hurt.

### Decomposition handles multi-hop questions

"Compare our SLA with the vendor's" needs two retrievals: one for our document, one for theirs. A single embedding of the compound question sits between the two topics and matches neither well.

Ask a small model to split the question into independent sub-questions, retrieve for each, then fuse the results — [Multi-Query Retrieval](09-multi-query.md) covers the fusion. Cap the number of sub-queries at three or four; beyond that, latency and context bloat outweigh the gain.

### Every transform is a latency and cost decision

Each transformation is one more LLM call on the critical path. A 400 ms rewrite on a 200 ms retrieval budget doubles your response time. Three habits keep this honest:

- Use the **smallest model that does the job** — rewriting is not a reasoning task.
- **Route**: skip transformation for queries that are already well-formed (long, self-contained), and apply it to short or context-dependent ones.
- **Log** the original and transformed query, so you can see what the rewrite did when a result looks wrong.

## Code

Condensation against history, with a strict output contract so the rewrite cannot ramble. The prompt forbids new facts and asks for one line back.

```python
from __future__ import annotations

from anthropic import Anthropic

client = Anthropic()
SMALL = "claude-haiku-4-5"   # check the provider's docs for the current model id

CONDENSE = """Rewrite the user's latest message as a standalone search query.
Rules:
- Use the conversation history only to resolve references (it, that, the second one).
- Do not add facts, do not answer the question, do not add commentary.
- Keep domain terms, codes and product names exactly as written.
- Output the query only, one line."""

def condense(history: list[dict[str, str]], question: str) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])  # last few turns
    msg = client.messages.create(
        model=SMALL, max_tokens=120, system=CONDENSE,
        messages=[{"role": "user", "content": f"History:\n{convo}\n\nLatest message: {question}"}],
    )
    out = "".join(b.text for b in msg.content if b.type == "text").strip()
    return out or question          # never let a failed rewrite blank the query
```

HyDE, where the generated passage is used only as a retrieval probe — the caller still answers from real chunks.

```python
from __future__ import annotations

HYDE = """Write a short factual paragraph that would answer this question,
as it might appear in a technical document. Do not hedge, do not explain.
Two to four sentences."""

def hyde_probe(question: str, model: str = SMALL) -> str:
    msg = client.messages.create(
        model=model, max_tokens=200, system=HYDE,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()

def retrieve_with_hyde(question: str, embed, search) -> list[dict]:
    probe = hyde_probe(question)
    vector = embed(probe)            # embed the fake answer, not the question
    return search(vector, k=50)      # same retrieval path as before
```

Decomposition returns a list, and the cap is enforced in code rather than trusted to the prompt.

```python
from __future__ import annotations

DECOMPOSE = """Split the question into at most 3 independent sub-questions that can each
be answered from one document. One per line. If the question needs no splitting,
return it unchanged as a single line."""

def decompose(question: str, max_parts: int = 3) -> list[str]:
    msg = client.messages.create(
        model=SMALL, max_tokens=200, system=DECOMPOSE,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    parts = [line.strip(" -0123456789.") for line in text.splitlines() if line.strip()]
    return (parts or [question])[:max_parts]      # hard cap, regardless of what the model says
```

## Real use case

A B2B analytics product shipped a chat assistant over their own documentation. Single-question sessions worked well; sessions with follow-ups failed almost every time. Their logs showed queries like "how do I set that up" reaching the index, which of course returned generic onboarding pages.

Adding condensation against the last six turns fixed the follow-up class entirely, and their eval set — which had no multi-turn cases — was extended with 20 follow-up threads so the improvement was measurable rather than anecdotal.

They then tried HyDE on the same set. It helped on questions phrased in user vocabulary ("why is my dashboard slow") and hurt on precise lookups ("what is the maximum rows per export"), where the generated passage invented a plausible-sounding limit that pulled retrieval toward the wrong page. They kept HyDE only for queries under a length threshold and routed precise-looking queries to plain retrieval.

The design choice from this file: **log the original and the transformed query together.** Without that, a bad answer looks like a retrieval bug when it is actually a rewrite bug.

## Common mistakes

- **Mistake:** embedding the raw follow-up in a chat product — **Instead:** condense against history first; it is the cheapest fix available.
- **Mistake:** letting the rewrite answer the question — **Instead:** forbid answers in the prompt and validate that the output is short.
- **Mistake:** using a large model for mechanical rewriting — **Instead:** use a small, fast model and spend the budget on reranking.
- **Mistake:** applying HyDE everywhere — **Instead:** route by query type and measure both paths separately.
- **Mistake:** trusting the model to respect a sub-query cap — **Instead:** truncate in code.
- **Mistake:** failing open by sending an empty query when the rewrite errors — **Instead:** fall back to the original question.

## Checkpoint

1. Implement `condense` and test it on a three-turn conversation where the final message is "and for contractors?". Confirm the output is standalone and preserves any domain terms.
2. Implement HyDE and compare retrieval for ten questions with and without it. Record how often the gold chunk appears in the top 5 in each mode.
3. Add a router: condense always, HyDE only when the query is under 40 characters and contains no digits. Measure the latency difference over 20 queries.

**Self-check:** the condensed query should read as a complete question with the topic restored, and any product or part names must appear exactly as the user typed them — if the model "cleaned them up", tighten the prompt.

## Go deeper

- [Multi-Query Retrieval](09-multi-query.md) — fusing results from several transformed queries.
- [Reranking](06-reranking.md) — the step that usually pays for itself before HyDE does.
- [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md) — testing a rewrite prompt properly.

---
[← Previous: 4. Embedding and Storage Design](04-embedding-storage.md) · [Stage index](README.md) · [Next: 6. Reranking](06-reranking.md)
