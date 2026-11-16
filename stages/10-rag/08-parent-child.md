# 8. Parent-Child Retrieval

> **Stage 10 · RAG** — search over small chunks for precision, then hand the model the larger section they came from for context.
>
> **You'll be able to:** build a two-level chunk structure, link children to parents, and return parents after a child-level search.
>
> **Prerequisites:** [Chunking](03-chunking.md) · [Reranking](06-reranking.md)

## Why this exists

Chunking forces a compromise you do not have to make. Small chunks embed precisely — one idea, one vector — but they strip context, so the model receives a fragment that says "the limit is 10 days" without saying of what. Large chunks carry context but embed vaguely, so retrieval misses them.

Parent-child retrieval refuses the compromise. Embed small children; store their parents. Search children; return parents. You get small-chunk precision in the ranking and large-chunk context in the prompt.

The mental model: **children are for finding, parents are for reading.**

## The mental model

```text
parent chunk (1,200 tokens) ─────────────────────────────────────────┐
  "Leave policy > Carryover                                            │
   Employees may carry over unused annual leave…                       │  returned to
   [child 1: 200 tok]  [child 2: 200 tok]  [child 3: 200 tok] …        │  the model
   …the cap is 10 days per year…"                                      │
└──────────────────────────────────────────────────────────────────────┘
        ▲                 ▲                ▲
        │  embedded       │  embedded      │  embedded        ← indexed
     child 1           child 2          child 3

  query → child 2 wins → fetch parent of child 2 → prompt gets the parent
```

## Core concepts

### Two levels, two purposes

A **child** is a small passage — a sentence or two, or a paragraph, roughly 100–250 tokens — that gets embedded. A **parent** is the surrounding section — 800–1,500 tokens — that gets returned. Each child stores a `parent_id`; each parent stores its full text.

The retrieval flow is three steps: search children, take the distinct parent ids of the top-k children, fetch those parents, and pass the parents (deduplicated, ordered by best child score) to the prompt. A parent hit by three different children is a strong signal and should rank first.

### This is a different problem from "return the whole document"

The parent must be small enough to fit several in a prompt. A whole document is not a parent — it is the thing you were trying to avoid. The parent's size should be bounded by a budget: if you plan to send 4 passages, each parent can be at most a quarter of your context budget.

### Deduplicate parents, and count the hits

If you fetch the parent of each of the top 10 children, you may get 3 distinct parents. That is the point — it means several children agreed. Aggregate: for each parent, keep the max child score and the number of children that hit it, then sort by max score (or a small combination). Report the distinct-parent count in your logs; it tells you whether your children are too small (many hits per parent) or too large (one hit per parent, and you have gained nothing).

### Sentence-window is the same idea at a smaller scale

Instead of parent sections, index single sentences and return a window of `w` sentences around the matched one. It is cheaper to build and gives tighter context, at the cost of losing document structure. Use it when your documents are unstructured prose; use section parents when headings exist.

### The failure mode this fixes, and the one it does not

It fixes "the right sentence was retrieved but the model could not tell what it referred to". It does not fix "no child matched", because retrieval recall is still set by the children. If your children are too small and context-free, their embeddings can be ambiguous and recall drops — which is why you measure recall@k of a gold child before and after the change.

## Code

Build the two levels from structured sections, keeping the parent text once and the children as the indexed unit.

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel

class Child(BaseModel):
    id: str
    parent_id: str
    text: str
    embed_text: str          # heading path + text, so the child embeds with context
    ordinal: int

class Parent(BaseModel):
    id: str
    heading_path: str
    text: str                # the full section, returned to the model

def build_tree(doc_id: str, heading_path: str, body: str,
               child_tokens: int = 200) -> tuple[Parent, list[Child]]:
    parent_id = f"{doc_id}::{hashlib.sha256((heading_path + body).encode()).hexdigest()[:12]}"
    parent = Parent(id=parent_id, heading_path=heading_path, text=body)

    sentences = [s.strip() for s in body.replace("\n", " ").split(". ") if s.strip()]
    children: list[Child] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush(ordinal: int) -> None:
        if not buf:
            return
        text = ". ".join(buf) + "."
        children.append(Child(
            id=f"{parent_id}::c{ordinal:03d}",
            parent_id=parent_id,
            text=text,
            embed_text=f"{heading_path}\n\n{text}" if heading_path else text,
            ordinal=ordinal,
        ))

    for s in sentences:
        est = len(s.split()) * 4 // 3          # rough tokens; use a real tokenizer in production
        if buf_tokens + est > child_tokens and buf:
            flush(len(children))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += est
    flush(len(children))

    return parent, children
```

Retrieval: search children, group by parent, keep the best score per parent, then fetch parent text for the prompt.

```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ParentHit:
    parent_id: str
    score: float
    child_hits: int
    text: str
    heading_path: str

def search_parents(
    query_vector,
    child_search,             # (vector, k) -> list[(child_id, parent_id, score)]
    fetch_parents,            # (parent_ids) -> dict[str, Parent]
    k_children: int = 20,
    keep_parents: int = 4,
) -> list[ParentHit]:
    children = child_search(query_vector, k_children)

    agg: dict[str, tuple[float, int]] = {}
    for _child_id, parent_id, score in children:
        best, count = agg.get(parent_id, (float("-inf"), 0))
        agg[parent_id] = (max(best, score), count + 1)     # max score, hit count

    ranked = sorted(agg.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:keep_parents]
    parents = fetch_parents([pid for pid, _ in ranked])

    return [
        ParentHit(parent_id=pid, score=round(score, 4), child_hits=count,
                  text=parents[pid].text, heading_path=parents[pid].heading_path)
        for pid, (score, count) in ranked if pid in parents
    ]
```

## Real use case

A clinical-trials documentation assistant answered protocol questions from 3,000 pages of study documents. With 400-token chunks, retrieval found the right chunks, but the model often answered with a caveat missing — because the caveat was in the preceding paragraph, in a different chunk. Increasing chunk size to 1,200 fixed that and broke retrieval: recall dropped, since each large chunk averaged several eligibility criteria.

Parent-child solved both. They embedded 180-token children and returned the 1,200-token parent section. A question about an eligibility criterion now matched a precise child — "Participants with eGFR below 30 are excluded" — and the model received the full inclusion/exclusion section, including the exception clause two paragraphs later.

Two numbers they reported honestly: recall@20 of a gold child was unchanged from the small-chunk baseline (as expected, since the indexed units were the same size), and answer correctness on their 40-question set improved, measured by a human rater because automated metrics were unreliable on clinical wording.

The failure mode before: correct retrieval, incomplete answers — the hardest class to notice, because nothing looked broken.

## Common mistakes

- **Mistake:** making the parent the whole document — **Instead:** bound the parent by your context budget divided by the number of passages you send.
- **Mistake:** embedding the child without its heading path — **Instead:** prepend the heading path to the embedded text so a bare sentence is still locatable.
- **Mistake:** passing duplicate parents to the prompt — **Instead:** deduplicate by `parent_id` and rank by max child score.
- **Mistake:** ignoring `child_hits` — **Instead:** log it; many hits per parent means children are too small, one hit each means the structure buys nothing.
- **Mistake:** assuming this improves recall — **Instead:** measure recall@k of a gold child; parent-child improves the context you generate from, not what you can find.
- **Mistake:** returning parents that exceed the prompt budget after dedup — **Instead:** cap total parent tokens and drop the lowest-scoring parents.

## Checkpoint

1. Build the two levels for one long document and print three children with their parent ids. Confirm each child's text appears verbatim inside its parent.
2. Implement `search_parents` and compare prompts from child-only versus parent-return for five questions. Note where the parent adds a caveat or definition the child lacked.
3. Tune child size at 120, 200 and 350 tokens. Report recall@20 of a gold child and average distinct parents returned per query, then pick a size.

**Self-check:** every child's text is a contiguous substring of its parent, and a parent hit by several children should rank above one hit by a single child at a similar score — that agreement is the signal the aggregation is designed to capture.

## Go deeper

- [Multi-Query Retrieval](09-multi-query.md) — another way to raise recall before aggregation.
- [Chunking](03-chunking.md) — the sizes you are now choosing at two levels.
- [Context Compression](../08-prompt-and-context/04-context-compression.md) — trimming parents once you have them.

---
[← Previous: 7. Hybrid Retrieval](07-hybrid-retrieval.md) · [Stage index](README.md) · [Next: 9. Multi-Query Retrieval](09-multi-query.md)
