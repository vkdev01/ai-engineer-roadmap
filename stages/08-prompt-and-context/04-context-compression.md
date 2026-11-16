# 4. Context Compression

> **Stage 08 · Prompt and Context** — fitting the tokens that matter into a window that is too small.
>
> **You'll be able to:** compress long context with summarisation, extraction, dedup and ranking instead of truncation.
>
> **Prerequisites:** [Prompt Templates with Variables and Validation](03-templates.md) · [Context Windows](../06-llm-fundamentals/03-context-windows.md)

## Why this exists

A context window is a hard boundary with a soft edge: output quality degrades as the window fills, and irrelevant tokens compete with relevant ones for the model's attention. When your input is a 60-page contract, a 200-message support thread, or twenty retrieved chunks, you cannot simply send everything.

The lazy fix is slicing — `text[:4000]` — which is a random lossy filter that keeps the beginning of the document and drops the end, where the signature, total and conclusion usually are. Compression is the disciplined version: decide what the model needs, keep that, and discard the rest with a documented rule.

The mental model: **compression is a relevance filter with a budget, not a truncation.**

## The mental model

```text
raw input (tokens: 40k, budget: 6k)
   |
   +-- dedup        drop repeated boilerplate, headers, quoted replies
   +-- extract      pull only fields/sentences that answer the question
   +-- rank         score units by relevance, keep the top ones
   +-- summarise    condense the remainder into a short digest
   |
   v
assembled context (tokens: ~6k) with provenance kept
```

| Strategy | Keeps | Loses | Best for |
|---|---|---|---|
| Dedup | everything unique | repetition | threads, scraped pages |
| Extraction | matching spans | narrative | question answering over long docs |
| Ranking | top-k units | long tail | retrieved chunks |
| Summarisation | gist | exact wording, numbers | history, background |
| Truncation | the prefix | everything after | almost never correct |

## Core concepts

### Budget before you compress

Decide the token budget for context first: window size minus system prompt, minus expected output, minus a safety margin. Without a number, "compress" means "make it shorter", and you cannot tell when you are done.

### Extract before you summarise

Summarisation with an LLM costs a call and can lose exact figures. If the question is "what is the notice period", sentence-level extraction — pick the sentences containing "notice" and "termination" — keeps the exact wording and costs nothing. Reach for summarisation when you need the gist of material that has no obvious keywords, such as long conversation history.

### Summarisation is itself a model call with its own cost

Rolling summaries of a conversation mean one extra call per turn, or one per N turns. Decide the cadence and cache the result. A summary regenerated on every request is a cost bug, not a feature.

### Preserve provenance

When you compress, keep a pointer back to the source unit — chunk id, message index, page number. The model may cite it, and when it answers wrongly you need to know what it actually saw. Compressed context with no provenance is unauditable.

### Recency and position interact with compression

If you keep only the top-ranked chunks, you may drop the immediately preceding turn, which usually carries the most conversational relevance. Always reserve a slice of the budget for recent turns before ranking the rest. Position effects are covered in [Context Engineering](08-context-engineering.md).

## Code

A budgeted compression pipeline: dedup, rank by overlap, extract matching sentences, then optionally summarise the remainder.

```python
from __future__ import annotations

import re
from dataclasses import dataclass

WORD = re.compile(r"[a-z0-9]+")


def tokens_approx(text: str) -> int:
    """Cheap estimate for budgeting. Use a real tokeniser when precision matters."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Unit:
    id: str          # provenance: chunk id, message index, page number
    text: str
    recency: int = 0  # higher = more recent


@dataclass(frozen=True)
class Budget:
    total_tokens: int
    recent_reserve: int = 400   # always keep the newest turns


def _terms(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def dedup(units: list[Unit]) -> list[Unit]:
    seen: set[str] = set()
    out: list[Unit] = []
    for u in units:
        key = " ".join(sorted(_terms(u.text)))[:200]  # coarse signature
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def rank(units: list[Unit], query: str) -> list[Unit]:
    q = _terms(query)
    # simple overlap score; swap in an embedding score when you have one
    return sorted(units, key=lambda u: len(q & _terms(u.text)), reverse=True)


def extract_sentences(units: list[Unit], query: str, limit: int) -> list[Unit]:
    q = _terms(query)
    picked: list[Unit] = []
    for u in units:
        for sent in re.split(r"(?<=[.!?])\s+", u.text):
            if q & _terms(sent):
                picked.append(Unit(id=u.id, text=sent.strip(), recency=u.recency))
            if len(picked) >= limit:
                return picked
    return picked


def compress(
    units: list[Unit], query: str, budget: Budget
) -> tuple[list[Unit], int]:
    """Returns the kept units and the token count used."""
    units = dedup(units)
    used = 0
    kept: list[Unit] = []

    # 1. recent turns always get their slice first
    recent = sorted(units, key=lambda u: u.recency, reverse=True)
    for u in recent:
        cost = tokens_approx(u.text)
        if used + cost > budget.recent_reserve:
            break
        kept.append(u)
        used += cost

    # 2. remaining budget goes to ranked units, extracted to sentences
    remaining = [u for u in units if u not in kept]
    for u in extract_sentences(rank(remaining, query), query, limit=40):
        cost = tokens_approx(u.text)
        if used + cost > budget.total_tokens:
            break
        kept.append(u)
        used += cost

    return kept, used


def assemble(kept: list[Unit]) -> str:
    # keep provenance visible: the model can cite it, you can audit it
    return "\n".join(f"[{u.id}] {u.text}" for u in kept)
```

For the case where extraction is not enough — a long support thread with no keyword to latch onto — a single summarisation call over the dropped material keeps the gist without the tokens.

```python
async def summarise_dropped(dropped: list[Unit], client, model: str) -> Unit:
    body = "\n".join(f"[{u.id}] {u.text}" for u in dropped)
    resp = await client.chat.completions.create(
        model=model,                       # a small model is fine for this
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Summarise the transcript in under 120 words. "
                           "Keep names, numbers and decisions exactly. "
                           "Cite source ids in square brackets.",
            },
            {"role": "user", "content": body},
        ],
    )
    return Unit(id="summary", text=resp.choices[0].message.content or "")
```

In TypeScript the same pipeline fits in a few functions; note the explicit budget object, which is the part that makes compression testable.

```ts
export type Unit = { id: string; text: string; recency?: number };

const approxTokens = (s: string): number => Math.max(1, Math.ceil(s.length / 4));

export function compress(
  units: Unit[],
  query: string,
  totalTokens: number,
): { kept: Unit[]; usedTokens: number } {
  const q = new Set(query.toLowerCase().match(/[a-z0-9]+/g) ?? []);
  const score = (u: Unit): number =>
    (u.text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => q.has(w)).length;

  const kept: Unit[] = [];
  let usedTokens = 0;

  for (const u of [...units].sort((a, b) => score(b) - score(a))) {
    const cost = approxTokens(u.text);
    if (usedTokens + cost > totalTokens) continue; // skip, keep trying smaller
    kept.push(u);
    usedTokens += cost;
  }
  return { kept, usedTokens };
}

export const assemble = (kept: Unit[]): string =>
  kept.map((u) => `[${u.id}] ${u.text}`).join("\n");
```

## Real use case

A contract-review tool answered questions like "what is the liability cap?" over agreements that were often 40-80 pages. The first version sent the first 30,000 characters of extracted text, which silently dropped the schedules at the end where caps and termination terms usually live — and the model answered from the wrong clause with total confidence.

The rewrite reserved a recency slice for the user's last two questions, ranked the remaining chunks against the question, extracted only sentences containing query terms, and summarised the rest into a short digest with source ids. Questions about schedules now found the right clause because ranking was by relevance rather than position. The failure mode before was truncation dressed up as compression: the pipeline kept the beginning of the document and the beginning had nothing to do with the question.

## Common mistakes

- **Mistake:** slicing on character count. — **Instead:** rank and extract by relevance against the actual question.
- **Mistake:** summarising everything with an LLM on every request. — **Instead:** extract first, summarise only the residue, and cache summaries.
- **Mistake:** dropping provenance when compressing. — **Instead:** keep source ids in the assembled text so answers are auditable.
- **Mistake:** ranking out the most recent conversation turn. — **Instead:** reserve a fixed slice of the budget for recency before ranking.
- **Mistake:** compressing without a token budget. — **Instead:** compute the budget from window size minus system prompt, output allowance and margin.

## Checkpoint

1. Add a dedup step that catches near-duplicates (not just exact matches) and measure how many units it removes on a real thread.
2. Make the pipeline return a report: units in, units kept, tokens used, tokens dropped by strategy. Log it per request.
3. Replace the overlap ranker with an embedding similarity ranker and compare which clause gets selected for five questions.

**Self-check:** you compute a token budget, keep a recency slice first, then rank and extract by relevance against the query while preserving source ids — never a blind prefix slice.

## Go deeper

- [Context Engineering](08-context-engineering.md) — where the compressed pieces are placed in the window.
- [Chunking](../10-rag/03-chunking.md) — the unit granularity that compression works over.
- [Reranking](../10-rag/06-reranking.md) — a learned relevance score for the ranking step.
- [Prompt Evals](07-prompt-evals.md) — proving compression did not lose the answer.

---
[← Previous: Prompt Templates with Variables and Validation](03-templates.md) · [Stage index](README.md) · [Next: Prompt Versioning as Code](05-prompt-versioning.md)
