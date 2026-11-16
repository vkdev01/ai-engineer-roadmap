# 10. Technical Articles

> **Stage 23 · Career** — writing one idea per post, with a reproduction, so people finish it and remember who wrote it.
>
> **You'll be able to:** choose a topic you can actually own, structure a post that gets read to the end, and make every claim reproducible.
>
> **Prerequisites:** [Production Case Studies](08-production-case-studies.md) · [Ship: An Eval Suite](../15-evals/10-ship-eval-suite.md) · [Building in Public](11-build-in-public.md)

## Why this exists

Writing is the cheapest way to be found. A post that solves one specific problem — "why my RAG missed the last paragraph of every document" — keeps bringing you readers for years, because people search for their bug, not for your career.

It is also the fastest way to find out whether you understand something. Most of what feels like knowledge is recognition; writing forces recall.

The mental model: **one problem, one reproduction, one lesson — and a title someone would type into a search box.**

## The mental model

```text
   pick a bug you actually fixed
              │
              ├── reproduce it in <20 lines        ← makes it credible
              ├── show the wrong mental model      ← makes it interesting
              ├── show the fix, with the number    ← makes it useful
              └── state the general lesson         ← makes it memorable

   title = the search query, not the topic
   "RAG missed the end of long documents"   beats   "Thoughts on chunking"
```

## Core concepts

### Own a narrow problem

"Introduction to RAG" competes with a thousand posts and cannot be won. "Why parent-child retrieval fixed our two-chunk failures, with numbers" has almost no competition and the exact reader you want. Narrow topics also let you be authoritative without being comprehensive, which is the only realistic bar for a first post.

Good sources of topics: a bug you fixed (with the reproduction), a benchmark you ran on your own data, a comparison you actually tested, or an explanation of a paper's mechanism with working code.

### The structure that gets finished

Readers leave at the first sign of padding. A shape that works:

1. **The symptom** — what went wrong, in one paragraph, in the reader's language.
2. **The wrong explanation** — what you first believed (most readers believe this too).
3. **The reproduction** — minimal code, actual output.
4. **The real cause** — the mechanism, with a diagram or a shape annotation.
5. **The fix** — code plus the measurement before and after.
6. **The general lesson** — two or three sentences, no moralising.

Under 1,500 words for a technical post is not a limitation; it is the reason people finish.

### Reproducibility is the differentiator

Every number in the post should come from a script in a repository you link. If you cannot reproduce it, do not print it — say "in our workload" and describe the shape of the observation instead. Readers forgive approximations and never forgive invented benchmarks.

```python
# figures.py — the script behind every number in the post.
# Committed alongside the post, so a reader can re-run it.

RESULTS = {
    "dense_only":   {"recall": 0.58, "n": 40},
    "hybrid":       {"recall": 0.71, "n": 40},
    "hybrid+rerank": {"recall": 0.78, "n": 40},
}

def delta(a: str, b: str) -> float:
    return round(RESULTS[b]["recall"] - RESULTS[a]["recall"], 3)

if __name__ == "__main__":
    for pair in [("dense_only", "hybrid"), ("hybrid", "hybrid+rerank")]:
        print(f"{pair[0]} -> {pair[1]}: +{delta(*pair)} recall (n={RESULTS[pair[1]]['n']})")
```

### Title and first line do the work

Write the title as the query a stuck engineer would type. Write the first line as the symptom, not as context about your journey. "Our RAG silently dropped the last 40 tokens of every long document" beats "Recently I have been exploring retrieval systems".

### Publishing mechanics

One canonical location (your own site or a dev-blog platform), syndicated elsewhere with a canonical link. Include the repository, the script, and the versions you tested with. Reply to comments for the first week — that is where the next post's topic usually comes from.

## Code

The post as a markdown file in a repo, so it is versioned with the code it describes.

````markdown
---
title: "Why your RAG silently drops the end of long documents"
canonical: https://example.dev/posts/rag-tail-loss
repo: https://github.com/you/rag-tail-repro
tested: chunking-lib 0.4.1, python 3.11
---

## The symptom

Answers about the final section of a long policy document were consistently wrong,
while everything else worked. Retrieval scores looked healthy.

## What I first believed

That the embedding model was weak on that vocabulary. I spent two days on query
rewriting. It changed nothing.

## Reproducing it in fifteen lines

```python
from chunking import split_by_tokens

text = " ".join(f"w{i}" for i in range(120))
chunks = split_by_tokens(text, max_tokens=50, overlap=10)
print(text.endswith(chunks[-1]))     # False
```

The last window started 10 tokens before the end, so the final 10 tokens were
never emitted. With `overlap=200` on a short document, the entire tail vanished.

## The real cause

[diagram: window arithmetic, stride = max_tokens - overlap]

## The fix and the number

[code + recall 0.58 -> 0.83 on our 40-case set, n=40]

## The lesson

Chunking bugs look like retrieval bugs. Before tuning anything, assert that
concatenating your chunks reproduces the input.
````

Note the shape: symptom, wrong belief, fifteen-line reproduction, mechanism, fix with a number, lesson. That is the whole template, and it works for almost any technical post.

TypeScript, if the post has an interactive element:

```ts
// A live chunker in the browser, so readers can see the bug themselves.
export function chunkRanges(len: number, maxTokens: number, overlap: number): [number, number][] {
  const stride = Math.max(1, maxTokens - overlap);
  const out: [number, number][] = [];
  for (let start = 0; start < len; start += stride) {
    const end = Math.min(start + maxTokens, len);
    out.push([start, end]);
    if (end === len) break;          // the fix: emit the tail even if short
  }
  return out;
}
```

## Real use case

An engineer publishes one post a month for four months: a chunking bug, a KV-cache sizing walkthrough with a calculator, a comparison of two rerankers on their own data, and an honest account of an agent that looped in production.

By the third post, recruiters are referencing the articles in first messages. The fourth — the failure story — gets the most engagement, because everyone has had that incident and nobody writes about it. The practical outcome is not virality; it is that interviews start with "I read your post on chunking" instead of "tell me about yourself", which is a much better room to be in.

## Common mistakes

- **Mistake:** a broad topic competing with established content — **Instead:** one narrow problem you have personally fixed.
- **Mistake:** a personal-journey opening — **Instead:** the symptom in the first line.
- **Mistake:** numbers with no script behind them — **Instead:** link the repository; make every figure reproducible.
- **Mistake:** code that only works on your machine — **Instead:** pin versions, use a small sample dataset, include a `run.sh`.
- **Mistake:** publishing and disappearing — **Instead:** answer comments for a week; the questions become your next post.
- **Mistake:** writing about something you have not built — **Instead:** write the post after the project; the specificity is the value.

## Checkpoint

1. Pick one bug from your last month of work and write the fifteen-line reproduction. If you cannot reproduce it, that is not yet a post.
2. Draft the post with the six-part structure and check that the first line is a symptom, not an introduction.
3. Publish with the repository linked, then note the first three questions you receive — they are your next three topics.

**Self-check:** if a reader cannot run your reproduction in under two minutes, the post will be read but not trusted.

## Go deeper

- [Building in Public](11-build-in-public.md) — the cadence around publishing
- [Open Source](09-open-source.md) — where the material comes from
- [Production Case Studies](08-production-case-studies.md) — the same stories, in interview form

---
[← Previous: Open Source](09-open-source.md) · [Stage index](README.md) · [Next: Building in Public](11-build-in-public.md)
