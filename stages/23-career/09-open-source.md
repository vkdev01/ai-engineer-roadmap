# 9. Open Source

> **Stage 23 · Career** — making a contribution that a maintainer accepts, without wasting their time or yours.
>
> **You'll be able to:** find an issue worth doing, reproduce a bug properly, write a pull request maintainers want to merge, and build a contribution record that reads as real experience.
>
> **Prerequisites:** [Git and GitHub](../01-programming-basics/05-git-github.md) · [Unit Tests for LLM Code](../15-evals/01-unit-tests.md) · [Technical Articles](10-technical-articles.md)

## Why this exists

Open source is the cheapest credibility available to an engineer with no brand: a merged pull request in a library people use is third-party evidence that you can read unfamiliar code and work to someone else's standards.

It is also a place where enthusiasm gets wasted. Drive-by pull requests on stylistic issues, unreproducible bug reports and "any update?" comments are how newcomers get ignored. The mechanics of being useful are learnable in an afternoon.

The mental model: **a contribution is a small, tested, reviewable change that solves someone else's stated problem.**

## The mental model

```text
   pick ──► reproduce ──► discuss ──► patch ──► test ──► PR ──► follow up
     │         │            │          │        │        │        │
   small     minimal      ask first   one      add a    small    respond
   scope     failing      on big      idea     failing  diff     within
             example      changes     per PR   test              a day
```

The step people skip is *discuss*. On anything larger than a typo, an issue comment saying "here is my reproduction and my plan — does that fit your direction?" saves a week.

## Core concepts

### Choosing the right first issue

Good first issues share three properties: a maintainer has confirmed it, the scope is a single file or function, and there is a way to test it. In practice that means label filters (`good first issue`, `help wanted`, `bug`), plus a check that the repository has merged a community PR in the last few months — an unmaintained repo will not review your work.

For AI engineering specifically, the highest-value contributions are usually not features: they are documentation fixes where the docs are wrong, small bugs in retry or parsing paths, and test coverage for edge cases maintainers have not hit.

### Reproductions are the currency

A bug report with a minimal reproduction is worth ten without. The minimal reproduction is a small script that fails, with the version pinned and the output pasted. Learn to cut your failing case down: remove the framework, remove your data, keep removing until removing anything more makes the bug disappear. That process often reveals the bug yourself.

### The pull request itself

Small diff, one concern, tests included, commit message in the project's style, and a description that says what changed, why, and how you verified it. If the repo has a `CONTRIBUTING.md` and a linter, run both before opening — most first-time rejections are mechanical.

```markdown
## What
Fix chunk boundary loss in `split_by_tokens` when the last chunk is shorter than
`overlap`, which dropped trailing text.

## Why
Closes #412. Reproduced with the script below; the final 40 tokens of the input
were never emitted, so retrieval silently missed the end of long documents.

## How
Adjust the final window to include the tail when `len(text) - start < overlap`,
and add a regression test covering a 3-token tail.

## Verification
- `pytest tests/test_chunking.py -q` → 14 passed
- New test fails on `main`, passes on this branch
- Ran the repo's `make lint`
```

### Review etiquette that gets you merged

Respond to review comments within a day or two; a stale pull request is a closed pull request. Push fixes as new commits during review (do not force-push mid-review — reviewers lose their place), then squash if the project asks. If a maintainer asks for a change you disagree with, say so once, with reasoning, and then do it their way — it is their repository.

### What it is actually worth

Be precise about this in interviews: a merged fix in a library you use is evidence of code-reading and standards, not proof of seniority. Describe it exactly as it was — "I fixed a chunking bug and added the regression test" — and let the reviewer draw conclusions. Overclaiming open-source work is one of the easiest things for an experienced interviewer to detect.

## Code

The reproduction and the regression test are the two artefacts that get a fix accepted. Here is the shape of both.

```python
# repro.py — minimal, pinned, no framework, prints the failure
from chunking import split_by_tokens          # replace with the real import

text = " ".join(f"w{i}" for i in range(120))

chunks = split_by_tokens(text, max_tokens=50, overlap=10)
recovered = " ".join(chunks)

print("chunks:", len(chunks))
print("last chunk:", chunks[-1][-60:])
print("tail preserved:", text.endswith(chunks[-1]))
# observed on 0.4.1: tail preserved: False
```

```python
# tests/test_chunking.py — the test that makes the fix permanent
import pytest

from chunking import split_by_tokens


def test_tail_shorter_than_overlap_is_preserved() -> None:
    """Regression for #412: a final chunk shorter than the overlap was dropped."""
    text = " ".join(f"w{i}" for i in range(120))
    chunks = split_by_tokens(text, max_tokens=50, overlap=10)
    assert text.endswith(chunks[-1]), "trailing text was lost"


@pytest.mark.parametrize("overlap", [0, 1, 10, 49])
def test_no_duplicate_or_missing_tokens(overlap: int) -> None:
    text = " ".join(f"w{i}" for i in range(200))
    chunks = split_by_tokens(text, max_tokens=60, overlap=overlap)
    seen = set()
    for chunk in chunks:
        for token in chunk.split():
            if token in seen and overlap == 0:
                pytest.fail("duplicate token with zero overlap")
            seen.add(token)
    assert len(seen) == 200
```

TypeScript, if the library is JS — same discipline, different syntax:

```ts
import { describe, expect, it } from "vitest";
import { splitByTokens } from "../src/chunking.js";

describe("splitByTokens", () => {
  it("preserves a tail shorter than the overlap (regression #412)", () => {
    const text = Array.from({ length: 120 }, (_, i) => `w${i}`).join(" ");
    const chunks = splitByTokens(text, 50, 10);
    expect(text.endsWith(chunks[chunks.length - 1])).toBe(true);
  });
});
```

## Real use case

An engineer applying for AI roles has no public work. Over six weeks they make four small contributions: a docs correction in an evaluation library, a retry-path bug fix with a test, a chunking edge case (the one above), and a reproduction that helps a maintainer fix a tokeniser bug without them writing the patch.

None of it is impressive in isolation. In an interview, the effect is different: the chunking fix lets them talk about boundary conditions in retrieval with a concrete story, and the reproduction shows they can isolate a problem. Two of the four contributions came from bugs they hit while doing Stage 10 of this roadmap — which is the practical advice: **contribute to what breaks while you are building.**

## Common mistakes

- **Mistake:** a pull request that reformats a whole file — **Instead:** one concern per PR, minimal diff, no incidental changes.
- **Mistake:** opening a PR for a large feature without asking — **Instead:** comment on an issue with your plan first; most maintainers will say yes or redirect you.
- **Mistake:** a bug report with no reproduction — **Instead:** a pinned, minimal script that fails, plus the actual output.
- **Mistake:** no test with the fix — **Instead:** add the failing test; it is what makes the fix reviewable in seconds.
- **Mistake:** force-pushing during review — **Instead:** add commits; squash only if asked.
- **Mistake:** overclaiming in the interview — **Instead:** describe the change exactly; the size of the claim should match the size of the diff.

## Checkpoint

1. Find three repositories you actually use, filter for `good first issue` or `bug`, and pick one where a community PR merged in the last three months.
2. Reproduce one issue with a minimal script under twenty lines. Post it as an issue comment with your environment and the output.
3. Open one pull request with a test. If it is rejected, ask why and record the answer — that answer is more valuable than the merge.

**Self-check:** if you cannot explain your change in two sentences, the diff is too big for a first contribution.

## Go deeper

- [Technical Articles](10-technical-articles.md) — writing up what you learned
- [GitHub Portfolio](01-github-portfolio.md) — how contributions appear on your profile
- [Unit Tests for LLM Code](../15-evals/01-unit-tests.md) — the testing discipline PRs require

---
[← Previous: Production Case Studies](08-production-case-studies.md) · [Stage index](README.md) · [Next: Technical Articles](10-technical-articles.md)
