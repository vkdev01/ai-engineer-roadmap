# 4. Core Algorithms

> **Stage 01 · Programming Basics** — gives you the small set of patterns that decide whether a script finishes in a second or a week.
>
> **You'll be able to:** read a loop and say what it costs as the input grows, then replace the slow version with hashing, sorting or a two-pointer pass.
>
> **Prerequisites:** [Generators](03-generators.md) · [Type Hints and Pydantic v2](02-type-hints-pydantic.md)

## Why this exists

You will not be asked to invert a binary tree to work on retrieval pipelines. You will be asked why deduplicating 200,000 documents takes four hours. The answer is almost always the same: a loop inside a loop, or a list membership test inside a loop, when a dictionary would have made it free.

Algorithmic thinking is not about memorising recipes. It is the habit of asking "how does the cost grow when the input gets ten times bigger?" — and knowing the three or four patterns that fix the common cases.

The mental model: **the shape of your loops determines the shape of your bill.**

## The mental model

Costs you will actually meet, with a rough feel for a million items:

| Notation | Name | A million items feels like | Typical cause |
|---|---|---|---|
| O(1) | constant | instant | dict lookup, set membership |
| O(log n) | logarithmic | instant | binary search on sorted data |
| O(n) | linear | fast | one pass, one file read |
| O(n log n) | sorting | quick | `sorted()` |
| O(n²) | quadratic | hours | loop inside a loop over the same list |

```text
n = 1,000        n = 10,000       n = 100,000
O(n)     1x            10x              100x
O(n²)  1x           100x           10,000x
```

## Core concepts

### Hashing is the cheapest trick in the book

`x in my_list` scans. `x in my_set` computes a hash and jumps. Deduplication, joining, counting, "have I seen this before" — all become one pass plus O(1) lookups. Whenever you see a membership test inside a loop, ask whether a `set` or `dict` can be built first.

### Sorting buys you everything after it

Once data is sorted you get binary search, two-pointer scans, deduplication of adjacent items, and grouping with `itertools.groupby`. Sorting costs O(n log n) once; a thousand linear scans over unsorted data cost more. For top-k, `heapq.nlargest` beats a full sort when k is small.

### Two pointers and sliding windows replace nested loops

If the data is sorted and you need a pair or a range, one pointer from each end (or a window that grows and shrinks) turns O(n²) into O(n). This is the pattern behind "find the most similar pair", "longest run meeting a constraint", and merge steps in deduplication.

### Recursion is a stack you write with function calls

Divide-and-conquer — merge sort, binary search, tree walks — is natural recursively, but Python's default recursion limit (about a thousand frames) makes deep recursion a crash. Convert to a loop with an explicit list as your stack when depth is unbounded.

### Space is a cost too

A dict of two million embeddings is memory you may not have. Say the space complexity out loud as well as the time: `sorted()` needs O(n) extra, an in-place pass needs O(1), a cache costs as much as you let it grow.

## Code

The three rewrites that cover most real slowdowns: dedupe with a set, top-k with a heap, and a sliding window.

```python
from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence


def dedupe_slow(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:        # O(n) scan on every iteration -> O(n^2)
            out.append(item)
    return out


def dedupe_fast(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:           # O(1) average
            continue
        seen.add(item)
        out.append(item)
    return out                     # order preserved, one pass


def top_k(scores: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    # O(n log k) instead of sorting everything: matters when n is large.
    return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
```

A sliding window that finds the longest stretch of a document under a character budget — the same logic as packing chunks for an embedding pipeline.

```python
from __future__ import annotations


def longest_window(text: str, max_chars: int) -> tuple[int, int]:
    """Return (start, end) of the longest window with len <= max_chars."""
    best = (0, 0)
    start = 0
    for end in range(len(text)):
        # Shrink from the left until the window fits again: each index
        # enters and leaves the window at most once, so this is O(n).
        while end - start + 1 > max_chars:
            start += 1
        if end - start + 1 > best[1] - best[0]:
            best = (start, end + 1)
    return best


print(longest_window("abcdefghij", 4))   # (0, 4)
```

Binary search by hand once, so library calls are not magic:

```python
from __future__ import annotations


def bisect_left_sorted(xs: list[int], target: int) -> int:
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo      # insert position; xs[:lo] is always < target
```

## Real use case

A retrieval service needed to drop near-duplicate support articles before indexing. The first implementation compared every article against every other with a string similarity function: for 40,000 articles that is roughly 800 million comparisons, and the nightly job never finished inside its window.

The rewrite sorted articles by a cheap normalised key, grouped adjacent items, and compared only within groups — collapsing the comparisons by orders of magnitude. A `set` of normalised text caught exact duplicates for free, and a heap kept only the top few candidate pairs per group for the expensive similarity check.

Nothing about the similarity function changed. The design choice that made it work was refusing to compare pairs that sorting had already proved could not be close.

## Common mistakes

- **Mistake:** `if item in results:` inside a loop that appends to `results`. — **Instead:** keep a parallel `set` and test against that.
- **Mistake:** sorting the whole list to get the top five. — **Instead:** `heapq.nlargest(5, ...)`.
- **Mistake:** building the full cartesian product because it is easier to read. — **Instead:** filter with a cheap key first, then compare survivors.
- **Mistake:** deep recursion over a long chain of documents. — **Instead:** an explicit stack, or `sys.setrecursionlimit` plus a rethink.
- **Mistake:** measuring with a tiny input and concluding it is fast. — **Instead:** time it at 10x and 100x; the curve tells you the complexity.

## Checkpoint

1. Time `dedupe_slow` and `dedupe_fast` on a list of 20,000 strings with many repeats, and explain the gap in one sentence.
2. Rewrite `top_k` using a full `sorted()` and compare on a 500,000-item dict.
3. Extend `longest_window` to return the window with the most whole words under `max_chars`, and state the time and space cost of your version.

**Self-check:** `dedupe_slow` does a linear scan per item, so total work grows with the square of the list length; the set version does one constant-time lookup per item.

## Go deeper

- [Generators](03-generators.md) — the algorithms above work on streams, not just lists.
- [Python and SQL Interviews](../23-career/03-python-sql-interviews.md) — these patterns in interview form.
- [Similarity Metrics](../09-vector-databases/02-similarity-metrics.md) — what "compare every pair" costs once vectors are involved.

---
[← Previous: Generators](03-generators.md) · [Stage index](README.md) · [Next: Git and GitHub](05-git-github.md)
