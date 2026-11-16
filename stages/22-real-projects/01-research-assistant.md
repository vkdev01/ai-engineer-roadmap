# 1. Research Assistant

> **Stage 22 · Real Projects** — a search-read-synthesise system that answers a research question with numbered citations you can click and check.
>
> **You'll be able to:** build and evaluate a research assistant whose every sentence traces back to a verbatim span it retrieved, with a cost and latency budget you can defend.
>
> **Prerequisites:** [RAG with Evals](../10-rag/13-ship-rag-with-evals.md) · [Harness Engineering](../11-agents/01-harness-engineering.md)

## Why this exists

Ask a general assistant a research question — "how are states handling AI procurement rules this year" — and you get confident, well-organised prose with no way to check it. You cannot tell whether a sentence came from a real document, from a half-remembered training example, or from the model's sense of what the answer probably looks like. So you go back to the search engine and do the reading yourself.

The people who pay for this are the ones whose job is that reading: research desks, consulting teams, paralegals, policy analysts. Their problem is not writing. It is trust and coverage — did I find the relevant documents, and is what I wrote actually in them. A research assistant is worth money only if it shortens the reading while keeping the checking cheap.

The mental model: the model is a *writer*, never a *source*. Sources come from search and fetch; every claim carries a pointer to a span of retrieved text; a deterministic verifier checks the pointer before the claim reaches the report.

## The mental model

```text
question
   |
   v
[planner] --sub-questions--> [search] --> [fetcher] --> [reader]
    ^                                                    |
    |                                        extract claims + verbatim quotes
    |                                                    v
[stop rule] <---- [coverage check] <---------------- [claim store]
    |                                                    |
    |                                                    v
    |                                     [verifier: quote found in passage?]
    |                                                    |
    +--------------------> [writer] --> report with [1][2][3]
```

The loop, not the prompt, is the design. The writer only ever sees claims that already passed verification.

## Core concepts

### Scope and non-goals

Scope: one question in, one markdown report out, with a claim table attached. In scope — web search, page fetch and text extraction, claim extraction, span verification, coverage-driven looping, a cost ceiling per run.

Non-goals, written down before you start: no authentication, no long-term memory across runs, no PDF chart understanding, no multilingual sources, no "chat with the report" follow-up. Each of these is a week of work that will not make the demo better. Say no in the README so a reviewer sees you can scope.

### Claims are the unit of work, not documents

A document is too coarse to verify and too coarse to deduplicate. A claim is one assertion, one passage id, and one verbatim quote copied out of that passage. Because the quote is verbatim, you can check it with `str.find` — no model, no judge, no flakiness. A report sentence with no verified claim behind it does not get written.

This one decision gives you three things at once: a citation pipeline you can unit-test, a precision metric that is deterministic, and a writer prompt that cannot hallucinate a source because it never sees the open web.

### Data model

| table | key fields |
|-------|-----------|
| `run` | id, question, budget_usd, steps_used, status |
| `source` | id, run_id, url, canonical_url, title, fetched_at, sha256 |
| `passage` | id, source_id, char_start, char_end, text |
| `claim` | id, run_id, text, passage_id, quote, verdict |
| `report` | id, run_id, markdown, claim_ids[] |

`canonical_url` plus `sha256` is how you avoid citing the same page five times under five tracking parameters. `passage` keeps character offsets so a click can scroll to the exact span.

### Tech stack, kept loose

Python 3.11+, `httpx` for fetch, `trafilatura` or `readability` for main-text extraction, `pydantic` v2 for the models above, SQLite for the store (Postgres only when you add users), any hosted search API for discovery, and one LLM SDK for the planner, extractor and writer. Pin nothing to a model id in the README — say which family you used and note that ids and prices change; check the provider's docs for current ones.

### Milestones: four weeks

- Week 1 — fetch and read. Given a URL, store cleaned passages with offsets. Success: 20 URLs, no empty extracts, dedupe works.
- Week 2 — claims. Extractor returns claims with quotes; verifier rejects quotes not found. Success: citation precision above your floor on a hand-labelled sample.
- Week 3 — the loop. Planner, coverage check, stop rule, cost ceiling. Success: a run finishes under budget and covers most sub-questions.
- Week 4 — writer, evals, demo. Success: the eval script runs on a frozen question set and prints a results table.

### Eval plan and the budget arithmetic

What "works" means, numerically, on your own frozen set of questions (write the set before you tune anything):

- **Citation precision** — share of claims whose quote is found in the cited passage. Deterministic, target near-total; anything below it is a bug, not a quality issue.
- **Coverage** — share of sub-questions with at least one supported claim.
- **Faithfulness** — share of report sentences entailed by their cited claims, judged by an LLM judge on a sample you also read by hand. Read [Faithfulness](../15-evals/04-faithfulness.md) for the rubric shape.
- **p95 latency** and **cost per run**, both from your own traces.

Budget arithmetic, illustrative for a typical setup: assume a run makes 12 searches, fetches 30 pages of roughly 3,000 tokens each, and reads each with a 1,000-token output of claims. That is about 90k input tokens for reading, plus roughly 15k for planning and 8k for writing. Multiply those three numbers by your provider's per-million input and output rates — prices change, so read them off the pricing page rather than from any tutorial. Then set `budget_usd` to about twice the result and hard-stop the loop at 80% of it. If your p95 target is 45 seconds, cap fetched pages at 30 and run the fetches concurrently; reading is what you cut, never verification.

## Code

The genuinely hard part is not the prompt, it is making the citation check impossible to skip. This snippet is the verifier: it locates the quote inside the passage, tolerating whitespace differences from extraction but nothing else, and it returns offsets so the UI can highlight.

```python
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WS.sub(" ", text).strip()


class Passage(BaseModel):
    id: str
    source_id: str
    url: str
    text: str
    char_start: int
    char_end: int


class Claim(BaseModel):
    """One assertion, one passage, one verbatim quote."""

    text: str
    passage_id: str
    quote: str = Field(description="copied verbatim from the passage")
    verdict: Literal["supported", "unsupported"] = "unsupported"
    span: tuple[int, int] | None = None


def locate(passage: Passage, quote: str) -> tuple[int, int] | None:
    """Return (start, end) of the quote in the passage, or None."""
    raw = quote.strip()
    if not raw:
        return None
    idx = passage.text.find(raw)          # exact match first: cheapest
    if idx >= 0:
        return (idx, idx + len(raw))
    squashed_quote = _squash(raw)         # extraction reflows newlines; retry once
    idx = _squash(passage.text).find(squashed_quote)
    return None if idx < 0 else (idx, idx + len(squashed_quote))


def verify(claims: list[Claim], passages: dict[str, Passage]) -> list[Claim]:
    for claim in claims:
        passage = passages.get(claim.passage_id)
        if passage is None:               # model invented an id
            claim.verdict, claim.span = "unsupported", None
            continue
        span = locate(passage, claim.quote)
        claim.span = span
        claim.verdict = "supported" if span else "unsupported"
    return claims
```

The loop around it is small and boring on purpose: plan, search, fetch, extract, verify, check coverage, repeat until covered or out of budget, then write from verified claims only.

```python
async def research(question: str, budget_usd: float, max_steps: int = 8) -> str:
    state = RunState(question=question, budget_usd=budget_usd)
    for step in range(max_steps):
        if state.spend_usd > 0.8 * budget_usd:      # leave room for the writer
            break
        sub_questions = await plan_missing(state)   # only what is not yet covered
        if not sub_questions:                       # coverage check passed
            break
        pages = await fetch_all(await search_all(sub_questions))
        state.add_passages(extract_passages(pages))
        state.add_claims(verify(await extract_claims(state, sub_questions), state.passages))
    return await write_report(state.supported_claims())
```

## Real use case

A four-analyst desk covering a regulated industry needs a background note every Monday on a moving topic. Before: one analyst spends most of a day reading, and the note still cannot be checked quickly because citations point at homepages rather than sentences. After: the analyst writes the question, the assistant returns a draft with 30-odd claims, each carrying a passage and a highlightable span; the analyst reads only the claims, rejects three, and rewrites the framing — which is the part that actually needs a human.

The design choice that made it work is span-level verification: the note is reviewable in ten minutes because a reviewer clicks a citation and lands on the exact sentence, instead of a page they must search.

Demo script, two minutes: ask a question with a known-good answer; open the claim table and show a rejected claim; click one citation and show the highlight; show the cost line and the p95 from your last eval run. Talking points for the interview: why claims instead of documents, why the verifier is deterministic, what you cut to hit the latency target, and what still fails (paywalled pages, and pages that render text with JavaScript — you skip them and say so).

## Common mistakes

**Mistake:** citing the search-result snippet instead of the fetched page — **Instead:** extract claims only from stored `passage` rows, and let the verifier fail anything else.
**Mistake:** letting the writer see raw passages "for context" — **Instead:** the writer sees verified claims plus their quotes, nothing else; unverifiable prose is the whole problem you are solving.
**Mistake:** looping on a step count alone — **Instead:** loop on coverage, and stop on cost; step counts neither bound spend nor tell you when you are done.
**Mistake:** treating near-duplicate URLs as separate sources — **Instead:** canonicalise and hash before inserting, or your precision metric rewards repetition.
**Mistake:** measuring quality only with an LLM judge — **Instead:** keep citation precision deterministic and unit-tested; judges drift, `str.find` does not.
**Mistake:** fetching pages serially — **Instead:** run fetches concurrently with a small semaphore and per-host politeness; latency here is wall-clock waiting, not compute.

## Checkpoint

1. Add a `verdict == "unsupported"` counter to the report header and make the writer refuse to include those claims. Confirm the report shrinks rather than silently keeping the sentence.
2. Add a second verifier rule: a claim is unsupported if its quote appears in a *different* passage than the one cited. Why does this matter for multi-source pages?
3. Give the planner the list of already-covered sub-questions and measure whether total steps drop on your frozen question set without losing coverage.

**Self-check:** Exercise 1 works because the writer's input is the claim list — filter it before rendering and the sentence has nothing to stand on.

## Go deeper

- [Production RAG](03-production-rag.md) — take this pipeline's retrieval layer to multi-tenant with freshness and CI evals
- [Multi-Agent Research](04-multi-agent-research.md) — split the planner into specialists when one loop stops covering the question
- [Hallucinations](../15-evals/05-hallucinations.md) — the failure taxonomy behind the verifier rules
- [Cost and Latency](../21-system-design/08-cost-latency.md) — turn your budget arithmetic into a capacity plan

---

[← Previous: Stage 21 — System Design](../21-system-design/README.md) · [Stage index](README.md) · [Next: Document Intelligence](02-document-intelligence.md)
