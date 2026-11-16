# 3. Chunking

> **Stage 10 · RAG** — choosing what unit of text gets embedded, which quietly sets the ceiling on retrieval quality.
>
> **You'll be able to:** implement fixed, recursive and structural chunking, and run an experiment that picks size and overlap from evidence.
>
> **Prerequisites:** [Ingestion and Parsing](02-ingestion-parsing.md) · [Embeddings: Text as Points in Space](../09-vector-databases/01-embeddings.md)

## Why this exists

An embedding compresses a passage into one vector. That vector is a summary — so it can only be as precise as the passage is focused. Chunk too large and the vector averages several topics, matching none of them well. Chunk too small and you strip the context that made the words meaningful: a chunk reading "The limit is 10 days." does not say of what, for whom, or per year.

Chunking is the most-tuned parameter in RAG and the least glamorous. It is also the cheapest to experiment with, because it requires no model change — just re-chunking and re-embedding.

The mental model: **a chunk should be the smallest passage that still makes sense on its own.**

## The mental model

```text
fixed      [400 tok][400 tok][400 tok]        simple, cuts mid-sentence
recursive  split on \n\n → \n → ". " → " "     respects natural boundaries
           until each piece fits the budget
semantic   embed sentences, split where        slower, data-dependent
           adjacent similarity drops
structural split on the document's own          best when headings exist
           headings / sections, then to size

overlap:  ...[---- chunk 1 ----]
                    [---- chunk 2 ----]...    repeats the tail so a boundary
                                              sentence appears in one piece
```

## Core concepts

### The arithmetic of size and overlap

Number of chunks for a document of `N` tokens, chunk size `S`, overlap `O`:

```text
stride = S − O
chunks = ceil((N − S) / stride) + 1
```

For a 10,000-token document at `S = 400`, `O = 50`: stride 350, `ceil(9600/350) + 1 = 28 + 1 = 29` chunks. Overlap costs you `O / stride ≈ 14%` extra embeddings — a real but modest price for not cutting answers in half.

A short 800-token document gives `ceil(400/350) + 1 = 3` chunks, so overlap matters more on short documents than on long ones.

### Fixed chunking is the baseline, and it is fine to start with

Split every `S` tokens with `O` overlap, ignoring content. It is trivial to implement, deterministic, and gives you a number to beat. Its weakness is obvious: it cuts mid-sentence and mid-table.

Always implement fixed first. It is your control in every later experiment, and any "smart" chunker that does not beat it on your corpus is complexity you should not ship.

### Recursive chunking respects structure cheaply

Try the largest separator first — paragraph breaks — and recurse into a smaller one only if a piece still exceeds the budget: paragraph, then line, then sentence, then word. This produces chunks that almost always end at a natural boundary, with a fraction of the complexity of semantic chunking.

It is the sensible default for prose. Two implementation details matter: keep a running character budget rather than tokenising every candidate (tokenise once at the end), and carry the overlap by prepending the last `O` tokens of the previous chunk.

### Structural chunking wins when headings exist

If your documents have real headings, use them. Split at heading boundaries and keep the heading path with each chunk — `"Leave policy > Carryover > Annual leave"`. Two payoffs: chunks become coherent by construction, and you can prepend the heading path to the chunk text before embedding, which is often worth more than any size tuning.

The rule: a chunk should never span two sections, even if that makes it short.

### Semantic chunking is a real option with real costs

Embed each sentence, compute similarity between consecutive sentences, and start a new chunk where similarity drops below a threshold (a percentile of the observed drops is more stable than a fixed number). It finds topic shifts that punctuation misses.

The costs are honest: one embedding call per sentence at index time, a threshold that needs tuning per corpus, and chunks of unpredictable size — which complicates batching and can produce 30-token chunks that retrieve badly. Use it when your documents are long, unstructured prose, and measure it against recursive.

### Run the experiment, and keep it small

Change one variable at a time, on a fixed query set, and measure retrieval quality (not answer quality) first. A workable protocol: 40–60 questions with known answer-bearing chunks, four chunk sizes, and recall@5 of "is a gold chunk retrieved". Re-embedding is the expensive step, so keep the corpus to a few hundred documents while you tune, then apply the winner to the full set.

## Code

Fixed and recursive chunkers with token-based sizing, plus the chunk-count arithmetic so you can predict cost before you run anything.

```python
from __future__ import annotations

import math
import re

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")   # check the encoding for your model

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def predict_chunks(n_tokens: int, size: int, overlap: int) -> int:
    stride = size - overlap
    return max(1, math.ceil((n_tokens - size) / stride) + 1)

def fixed_chunks(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    ids = ENC.encode(text)
    stride = size - overlap
    return [ENC.decode(ids[i : i + size]) for i in range(0, max(len(ids) - overlap, 1), stride)]

def recursive_chunks(text: str, size: int = 400, overlap: int = 50,
                     seps: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    """Split on the coarsest separator that fits, recursing only when needed."""
    def split(block: str, seps: tuple[str, ...]) -> list[str]:
        if count_tokens(block) <= size or not seps:
            return [block] if block.strip() else []
        sep, rest = seps[0], seps[1:]
        pieces = block.split(sep) if sep in block else [block]
        out: list[str] = []
        for p in pieces:
            out.extend(split(p + sep if sep != " " else p, rest))
        return out

    parts, merged, buf = split(text, seps), [], ""
    for p in parts:                       # merge small pieces up to the budget
        if count_tokens(buf + p) <= size:
            buf += p
        else:
            if buf.strip():
                merged.append(buf)
            buf = p[-overlap:] + p if overlap else p
    if buf.strip():
        merged.append(buf)
    return merged
```

Structural chunking from a heading path, which is the highest-value variant when your documents have headings at all.

```python
from __future__ import annotations

import re

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

def structural_chunks(markdown: str, size: int = 400) -> list[dict]:
    """Split on headings; keep the heading path; then size-cap each section."""
    chunks: list[dict] = []
    stack: list[str] = []
    section: list[str] = []

    def flush() -> None:
        body = "\n".join(section).strip()
        if not body:
            return
        path = " > ".join(stack)
        for piece in recursive_chunks(body, size=size):
            chunks.append({"heading_path": path, "text": piece,
                           # prefixing the path helps the embedding understand context
                           "embed_text": f"{path}\n\n{piece}" if path else piece})

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            section = []
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
        else:
            section.append(line)
    flush()
    return chunks
```

## Real use case

A legal team's assistant answered questions from a 900-page employee handbook. With 1,500-token chunks, recall was poor on specific questions: a chunk covering an entire "Leave" section averaged holiday, sick leave and parental leave into one vector, so "how many sick days carry over" matched the section but the answer was buried among three policies.

They re-chunked structurally — split at headings, then capped at 400 tokens, with the heading path prepended to the embedded text. The handbook's own structure did the work: "Leave policy > Sick leave > Carryover" became a chunk that is unmistakably about one thing.

They ran the size experiment on a 40-question set with known answer sections and compared recall@5 of a gold chunk: fixed 1,500, fixed 400 with 50 overlap, and structural 400. Structural won; fixed 400 with overlap was close and much simpler, which they noted honestly rather than claiming a landslide.

The failure mode before: the right section was retrieved, so the answer *looked* grounded, but it cited the wrong sub-policy — the worst kind of error, because it is plausible.

## Common mistakes

- **Mistake:** picking a chunk size from a blog post — **Instead:** run the four-size experiment on your own 40-question set.
- **Mistake:** no overlap, then wondering why answers are cut in half — **Instead:** use a modest overlap, around 10–15% of chunk size.
- **Mistake:** enormous overlap "for safety" — **Instead:** keep it small; large overlap multiplies embedding cost and duplicates results.
- **Mistake:** letting a chunk span two headings — **Instead:** split structurally first, then size-cap within sections.
- **Mistake:** chunking by characters while measuring by tokens — **Instead:** count tokens with the model's tokenizer, since limits and costs are in tokens.
- **Mistake:** evaluating chunking by reading five chunks — **Instead:** measure recall@5 of a gold chunk over dozens of questions.

## Checkpoint

1. Take a 5,000-token document, predict the chunk count at `S = 400, O = 50` with `predict_chunks`, then verify against `fixed_chunks`. Do they agree?
2. Implement all three chunkers and print the first 200 characters of ten chunks from each. Note which produce mid-sentence starts.
3. Run the size experiment: fixed at 200, 400, 800 and 1,500 tokens, 40 questions, recall@5 of a gold chunk. Tabulate and pick a winner with a one-sentence justification.

**Self-check:** `ceil((5000 − 400)/350) + 1 = ceil(13.14) + 1 = 15` chunks, and `fixed_chunks` should return 15 — if it returns a different number, the range step is off by one.

## Go deeper

- [Embedding and Storage Design](04-embedding-storage.md) — what to store alongside each chunk.
- [Parent-Child Retrieval](08-parent-child.md) — a way to get small-chunk precision and large-chunk context at once.
- [Context Compression](../08-prompt-and-context/04-context-compression.md) — reducing what you send to the model after retrieval.

---
[← Previous: 2. Ingestion and Parsing](02-ingestion-parsing.md) · [Stage index](README.md) · [Next: 4. Embedding and Storage Design](04-embedding-storage.md)
