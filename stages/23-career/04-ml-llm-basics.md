# 4. ML and LLM Basics

> **Stage 23 · Career** — the fundamentals round: six questions you will certainly be asked, with answers short enough to say out loud and precise enough to survive a follow-up.
>
> **You'll be able to:** explain attention, tokenisation, RAG versus fine-tuning, temperature, embeddings and evaluation in ninety seconds each, with the trade-off named.
>
> **Prerequisites:** [Self-Attention](../06-llm-fundamentals/04-self-attention.md) · [Tokenization](../06-llm-fundamentals/02-tokenization.md)

## Why this exists

Interviewers ask fundamentals not to test memorisation but to find the edge of your understanding. The follow-up is where the signal is: you say "attention lets tokens look at each other", they ask "why divide by the square root of the dimension", and either you know or you do not. Rambling for four minutes without a mechanism reads worse than a thirty-second answer plus "I would check the exact formulation".

The other trap is over-claiming. Saying "RAG eliminates hallucinations" is a claim a good interviewer will dismantle, and then your credibility on everything else drops.

Mental model: answer in three beats — what it is, why it exists, what it costs. Then stop and let them steer.

## The mental model

| Question | The one-sentence core | The follow-up they will ask |
|---|---|---|
| Attention | every token builds a weighted average of the others, weighted by learned similarity | why the scaling factor, why multiple heads |
| Tokenisation | text becomes subword ids from a fixed vocabulary | why not words, why not characters |
| RAG vs fine-tuning | retrieval adds knowledge, tuning changes behaviour | what breaks when the corpus changes hourly |
| Temperature | a divisor on logits before sampling | why 0 is not fully deterministic |
| Embeddings | a vector space where distance means relatedness | what they are bad at |
| Evaluation | a frozen dataset, a grader, and a comparison with uncertainty | how you stop tuning to the test |

## Core concepts

### Attention: weighted averaging by learned similarity

Each token produces a query, a key and a value. Score every query against every key with a dot product, scale, softmax to get weights, and sum the values. That is it — the mechanism is a differentiable lookup where the weights are content-dependent instead of positional.

Why scale by the square root of the head dimension: dot products of high-dimensional vectors grow in magnitude with dimension, which pushes softmax into saturation where gradients vanish; dividing by the square root keeps the variance roughly constant. Why multiple heads: each head can specialise in a different relation (previous token, matching bracket, coreferent entity), and concatenating them gives the model several views at the same cost.

Cost: attention is quadratic in sequence length in both time and memory, which is why long contexts are expensive and why KV caching and efficient attention variants matter. Say the word "quadratic" — it is the answer to half the follow-ups.

### Tokenisation: subwords because of the long tail

Word vocabularies are unbounded and blow up on typos, morphology and code. Character vocabularies make sequences long and lose meaning per token. Subword schemes (byte-pair encoding and its relatives) sit in between: frequent strings get one token, rare strings get split into pieces, and a byte-level fallback means nothing is ever out of vocabulary.

Consequences you should be able to state: token counts, not word counts, drive cost and context limits; non-English text and code often tokenise less efficiently, so the same sentence costs more tokens; and counting tokens for a prompt means using the model's own tokeniser, not `len(text.split())`.

### RAG versus fine-tuning

Retrieval changes what the model can *see* at inference time; fine-tuning changes what the model *does* by default. Use retrieval when the knowledge changes, must be cited, or must respect permissions. Use fine-tuning when the behaviour is stable but the format is hard to prompt — a fixed taxonomy, a house style, a strict output schema — or when you need a smaller model to imitate a larger one's output on a narrow task.

The honest answer includes the third option: better prompting and a stricter output schema fix more problems than either. And the failure mode of each: retrieval fails when the answer requires synthesis across many documents; tuning fails when the data is thin, and it can quietly degrade capabilities you did not test.

### Temperature and sampling

Temperature divides the logits before softmax: low temperature sharpens the distribution toward the top token, high temperature flattens it. Top-p truncates the distribution to the smallest set of tokens whose probability sums to p. Temperature 0 with greedy decoding is deterministic *for a fixed model, prompt and hardware*, but in practice batching, floating-point non-determinism and provider-side changes make "always identical" a false promise — say that, it is a strong answer.

Practical guidance: extraction and classification want low temperature; brainstorming wants higher; anything with a schema wants low temperature plus validation.

### Embeddings: distance as meaning

An embedding model maps text to a vector such that related texts are near each other under a chosen metric — cosine similarity is the default because it ignores magnitude. They are trained with contrastive objectives on pairs that should and should not be close.

What they are bad at: exact identifiers, negation, numbers, and long documents (one vector cannot hold a chapter). That is why hybrid retrieval with keyword matching exists, why chunking matters, and why a reranker over the top candidates fixes what a single vector cannot express. Also: you cannot mix embeddings from two models in one index — the spaces are unrelated.

### Evaluation: the answer that separates candidates

Say the structure: a frozen dataset with expected outputs or rubrics, graders that are deterministic where possible and model-based only where necessary, a comparison against a baseline with an interval, and per-slice reporting so an average cannot hide a broken case type. Then name the failure you have seen: tuning on the test set until it stops measuring anything.

## Code

Attention in a dozen lines, which is worth being able to write from memory on a whiteboard:

```python
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True) -> torch.Tensor:
    """q,k,v: (batch, heads, seq, dim). Returns (batch, heads, seq, dim)."""
    dim = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)      # (b, h, seq, seq)
    if causal:
        seq = q.size(-2)
        # mask future positions so token t cannot attend to t+1..
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)                     # rows sum to 1
    return weights @ v
```

And the token-counting habit, because it is the practical version of the tokenisation answer:

```python
def budget(prompt: str, tokenizer, max_out: int = 500, context_limit: int = 8000) -> dict:
    """Never estimate cost from word counts; use the model's own tokeniser."""
    n_in = len(tokenizer.encode(prompt))
    if n_in + max_out > context_limit:
        raise ValueError(f"prompt {n_in} + output {max_out} exceeds {context_limit}")
    return {"tokens_in": n_in, "tokens_out": max_out}
```

The evaluation answer in code: a frozen set, a retrieval metric, and a comparison that reports which case type fails — the structure an interviewer is listening for.

```python
from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(queries: list[dict], embed, k: int = 5) -> dict[str, float]:
    """queries: [{text, gold_doc_id, slice}]. Reports overall and per slice."""
    hits: dict[str, list[int]] = {}
    for q in queries:
        qv = embed(q["text"])
        ranked = sorted(q["corpus"], key=lambda d: cosine(qv, embed(d["text"])), reverse=True)
        top_ids = [d["doc_id"] for d in ranked[:k]]
        hits.setdefault(q["slice"], []).append(int(q["gold_doc_id"] in top_ids))
    out = {s: sum(v) / len(v) for s, v in hits.items()}
    out["overall"] = sum(sum(v) for v in hits.values()) / sum(len(v) for v in hits.values())
    return out
```

Report the per-slice numbers, not just `overall` — an average over an easy majority slice hides the case type that fails.

## Real use case

In a mock loop, a candidate answered "how would you stop hallucinations" with "use RAG". The interviewer asked what RAG does when the retrieved passage does not contain the answer. The candidate had no answer; the stronger answer is that retrieval reduces unsupported claims but does not eliminate them, so you verify claims against retrieved spans and add an explicit "not found" path.

The design choice that makes such answers strong is naming the residual failure: an answer that includes its own limits is harder to dismantle than a confident one.

## Common mistakes

**Mistake:** reciting formulas with no mechanism — **Instead:** one sentence of mechanism, then the trade-off.
**Mistake:** claiming temperature 0 gives identical output — **Instead:** say it is deterministic in principle, with batching and provider changes as caveats.
**Mistake:** "RAG eliminates hallucinations" — **Instead:** retrieval reduces unsupported claims; verification and a not-found path are separate work.
**Mistake:** treating fine-tuning as a knowledge store — **Instead:** retrieval for changing knowledge, tuning for behaviour and format.
**Mistake:** estimating tokens from words — **Instead:** the model's tokeniser; non-English text and code differ.
**Mistake:** talking past the question into a project story — **Instead:** answer, then offer one sentence of example if invited.

## Checkpoint

1. Explain attention to a colleague in ninety seconds without a diagram, then have them ask the scaling question. If you cannot answer it, re-read the section and try again tomorrow.
2. Compute token counts for the same paragraph in English and in your first language using one tokeniser. Note the ratio and why it matters for cost.
3. Take a task you solved with prompting and argue both sides: when would fine-tuning beat it, and what data would you need?

**Self-check:** Exercise 1's answer is variance control — unscaled dot products grow with dimension, saturating softmax and vanishing gradients.

## Go deeper

- [Transformers](../04-deep-learning/07-transformers.md) — the architecture around the attention block
- [RAG Architecture](../10-rag/01-rag-architecture.md) — retrieval as a system, not a trick
- [Fine-Tune vs RAG](../13-fine-tuning/07-finetune-vs-rag.md) — the decision framed as a table
- [RAG and Agent Patterns](06-rag-agent-patterns.md) — the next interview round

---

[← Previous: Python and SQL Interviews](03-python-sql-interviews.md) · [Stage index](README.md) · [Next: System Design Practice](05-system-design-practice.md)
