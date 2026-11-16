# 9. Multi-Query Retrieval

> **Stage 10 · RAG** — asking the same question several ways and fusing what comes back, so one unlucky phrasing cannot lose the answer.
>
> **You'll be able to:** generate query variants, fuse their results with RRF, and measure whether the extra calls earned their cost.
>
> **Prerequisites:** [Query Transformation](05-query-transform.md) · [Hybrid Retrieval](07-hybrid-retrieval.md)

## Why this exists

A single embedding of a question is one point in space. If the corpus expresses the answer using different vocabulary, that point can sit just outside the region where the answer lives — and one point gives you no second chance.

Multi-query retrieval hedges. Generate three or four paraphrases, embed each, retrieve for each, and fuse the ranked lists. A document that appears in two of four lists is very likely relevant, even if it ranked poorly for any single phrasing.

The mental model: **one query is a single draw from a noisy process; several queries average the noise out.**

## The mental model

```text
question ──► small model ──► ["How do I reset my password?",
                              "steps to recover a forgotten password",
                              "account login recovery process",
                              "regain access to my account"]
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
          retrieve(q1)        retrieve(q2)   ...   retrieve(q4)     each top 20
              └────────────────────┼────────────────────┘
                                   ▼
                        RRF fuse (k=60) ──► dedupe ──► rerank ──► top 5
```

## Core concepts

### Variants, not sub-questions

This is different from decomposition. Decomposition splits a compound question into parts, each answering a different aspect. Multi-query generates *alternatives* for the same information need — same answer, different wording. Mixing the two up produces prompts that generate sub-questions and then fuse them as if they were paraphrases, which loses both benefits.

Keep the prompts distinct: "write N different ways a user might ask this" versus "split this into independent sub-questions".

### Three or four variants is the useful range

Returns flatten quickly. Going from one query to three usually helps materially; three to six helps marginally and triples embedding cost, retrieval cost and latency. The embedding calls are cheap and parallel; the fusion and rerank costs are linear in the candidate set, and a bigger candidate set dilutes the reranker.

Set a cap in code, and prefer four. If your variant generator returns eight, take four.

### Fusion needs deduplication by chunk id, and RRF does it for free

Fusing by chunk id means the same document found by three variants accumulates three contributions, which is exactly the agreement signal you want. Do not deduplicate before fusion — that discards the signal. Deduplicate after, when you build the prompt.

Use RRF rather than averaging similarity scores: the variants' scores are not comparable across queries, since a variant that is a poor paraphrase will produce systematically lower similarities across the board.

### Multi-query raises recall, and recall is what reranking cannot fix

The reason this technique is worth its latency is that it moves the ceiling. Reranking reorders candidates; multi-query produces candidates that single-query retrieval never surfaced. Measure it as recall@20 of a gold chunk before and after — that is the number this technique should move. If your context precision improves but recall does not, the variants are not adding anything new.

### The failure case: variants that drift

Left unconstrained, a model will produce variants that answer a *different* question — narrower, broader, or shifted to an adjacent topic. The broader variant drags in generic chunks; the narrower one misses. Two controls: instruct the model to preserve entities and constraints exactly, and reject variants that share almost no tokens with the original (a cheap guard against topic drift).

## Code

Generate variants with a strict contract, guard against drift, and fuse with the same RRF used elsewhere in this stage.

```python
from __future__ import annotations

from anthropic import Anthropic

client = Anthropic()
SMALL = "claude-haiku-4-5"   # check the provider's docs for the current model id

VARIANTS = """Write {n} alternative phrasings of the user's search query.
Rules:
- Same information need, different wording. Not broader, not narrower.
- Keep names, numbers, product codes and constraints exactly as written.
- One per line, no numbering, no explanation, no quotes."""

def generate_variants(question: str, n: int = 4) -> list[str]:
    msg = client.messages.create(
        model=SMALL, max_tokens=300, system=VARIANTS.format(n=n),
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    lines = [l.strip(" -0123456789.\t") for l in text.splitlines() if l.strip()]
    return [l for l in lines if l][:n] or [question]
```

The guard and the fusion, kept as small pure functions so they are testable without a model.

```python
from __future__ import annotations

import re

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s.lower()))

def keep_on_topic(original: str, variants: list[str], min_overlap: float = 0.34) -> list[str]:
    """Drop variants that share too few tokens with the original (topic drift)."""
    base = _tokens(original)
    if not base:
        return variants
    kept = []
    for v in variants:
        overlap = len(base & _tokens(v)) / len(base)
        if overlap >= min_overlap:
            kept.append(v)
    return kept or [original]

def rrf_fuse(ranked_lists: list[list[str]], k: int = 60, limit: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:limit]

def multi_query_retrieve(question: str, embed, search, reranker,
                         n_variants: int = 4, depth: int = 20, keep: int = 5) -> dict:
    variants = keep_on_topic(question, generate_variants(question, n_variants))
    queries = [question] + variants                  # always include the original

    lists = [search(embed(q), depth) for q in queries]   # parallelise these in production
    fused = rrf_fuse(lists, k=60, limit=60)

    candidates = [cid for cid, _ in fused]
    top = reranker(question, candidates, keep)       # rerank with the ORIGINAL question
    return {"hits": top, "variants": variants, "n_queries": len(queries),
            "n_candidates": len(candidates)}
```

One detail worth noticing: the reranker scores against the **original** question, not the variants. The variants exist to widen retrieval; the user's actual wording is the best relevance signal for the final ordering.

## Real use case

A telecom operator's support assistant answered from tariff documents and troubleshooting guides. Users wrote in mixed vocabulary — "my net is not working", "data not coming", "internet issue" — and each phrasing landed in a slightly different region, so the correct troubleshooting page was sometimes at rank 12 and sometimes absent entirely.

Adding four variants per query, fused with RRF over depth 20, moved the correct page into the top 20 much more consistently. Their measurement was recall@20 of a gold chunk over 80 real logged questions, split into "symptom phrasing" and "product-name phrasing" groups.

The numbers showed something worth knowing: symptom-phrased queries improved substantially, while product-name queries did not move — because those were already unambiguous. So they routed: multi-query only for queries without a recognised product name. That kept the latency cost where it bought something.

The failure mode before: support agents learned to translate customer phrasing into document phrasing before searching. Again, the workaround in the logs was the defect report.

## Common mistakes

- **Mistake:** conflating multi-query with decomposition — **Instead:** keep two prompts: paraphrases for one need, sub-questions for multiple needs.
- **Mistake:** generating ten variants — **Instead:** cap at three or four; returns flatten and costs do not.
- **Mistake:** dropping the original question from the query list — **Instead:** always include it first; the user's own wording is a strong signal.
- **Mistake:** reranking against a variant — **Instead:** rerank against the original question.
- **Mistake:** averaging similarity scores across variants — **Instead:** fuse ranks with RRF; scores are not comparable across queries.
- **Mistake:** not guarding against topic drift — **Instead:** require entity preservation and reject low-overlap variants.

## Checkpoint

1. Generate four variants for ten questions and inspect them. How many drifted broader or narrower? Tighten the prompt and re-check.
2. Measure recall@20 of a gold chunk with one query versus four fused variants on 40 questions. Report both numbers with the query count.
3. Add the routing rule (multi-query only when no product name is present) and measure latency over 20 queries with and without it.

**Self-check:** variants should differ in wording only — same entities, same constraints, same specificity. A variant that asks about a broader topic, or drops a number, is drift and should be rejected by the overlap guard.

## Go deeper

- [Graph RAG](10-graph-rag.md) — when the answer needs relationships rather than passages.
- [RAG Evaluation](12-rag-evals.md) — measuring the recall this technique is supposed to move.
- [Context Compression](../08-prompt-and-context/04-context-compression.md) — fitting a larger candidate set into the prompt.

---
[← Previous: 8. Parent-Child Retrieval](08-parent-child.md) · [Stage index](README.md) · [Next: 10. Graph RAG](10-graph-rag.md)
