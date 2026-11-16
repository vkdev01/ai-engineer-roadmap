# 3. Context Windows

> **Stage 06 · LLM Fundamentals** — The model's working memory: a hard token limit, a known blind spot in the middle, and the strategies that decide what gets left out.
>
> **You'll be able to:** budget a prompt in tokens, choose a truncation strategy deliberately, and place the information that matters where the model actually reads it.
>
> **Prerequisites:** [2. Tokenization: BPE and Byte-Level](02-tokenization.md) · [1. System Instructions](../08-prompt-and-context/01-system-instructions.md)

## Why this exists

Every request has a fixed budget: system prompt, conversation history, retrieved documents, tool outputs, the user's question, and the space reserved for the answer. When the total exceeds the window, something is silently dropped — usually the oldest turns or the end of a document — and the model answers confidently from incomplete input.

A larger window is not a fix. Models attend less reliably to material in the middle of a long context than to the beginning and end, a pattern commonly called lost-in-the-middle. So the design question is not "how much fits" but "what deserves the space, and where should it sit".

The mental model: *a context window is a fixed-size buffer that you are responsible for filling, in priority order, with the highest-value tokens.*

## The mental model

```text
 ┌───────────────────────────────────────────┐
 │ system prompt            ~ fixed, always  │  ← strongest attention
 ├───────────────────────────────────────────┤
 │ retrieved documents      ~ the bulk       │  ← middle: weakest recall
 │ (put the best chunk FIRST and LAST)       │
 ├───────────────────────────────────────────┤
 │ recent turns             ~ recency wins   │
 ├───────────────────────────────────────────┤
 │ current question         ~ always kept    │  ← strongest attention
 ├───────────────────────────────────────────┤
 │ reserved for output      ~ max_tokens     │  ← not yours to spend
 └───────────────────────────────────────────┘
        total ≤ context window
```

Two numbers to hold separately: the **window** (input + output combined, in most APIs) and your **output reservation**. If you spend the whole window on input, the model has no room to answer and either errors or gets cut off mid-sentence.

## Core concepts

### The window is shared between input and output

Most APIs count `input_tokens + max_output_tokens` against the limit. Always subtract your answer reservation before budgeting input. A prompt that fits at 100% leaves zero tokens for a reply — a failure that looks like an API error and is really arithmetic.

### Lost in the middle: position changes recall

Recall of a fact placed in the middle of a long context is measurably worse than the same fact at the start or the end. The practical consequences: put instructions at the top, put the question at the bottom, and when you retrieve several chunks, order them so the two most relevant sit at the edges rather than in the middle.

### Four truncation strategies, in order of quality

1. **Budget by priority** — allocate tokens per section (system: 500, history: 2000, documents: 6000) and drop from the lowest-value section first.
2. **Drop oldest turns** — keep the system prompt, the last N turns, and a one-line summary of what was dropped.
3. **Summarise, then drop** — compress old turns into a paragraph before removing them; the basis of context compression in [4. Context Compression](../08-prompt-and-context/04-context-compression.md).
4. **Truncate the tail of a document** — the worst option, because the end of a document often holds the conclusion.

### Reserve, then measure, then send

Counting tokens after building the request is too late if you have already exceeded the limit. The order is: reserve output tokens, compute the remaining input budget, fill sections in priority order, measure the total, and only then send. If the total still exceeds the budget, fail loudly in development rather than letting the provider truncate silently.

### Chunk size should be a token decision, not a character decision

Retrieval that returns "five chunks" returns wildly varying token counts, because chunk sizes in characters map to unpredictable token counts. Return "at most 6,000 tokens of chunks" and let the count decide how many chunks that is.

## Code

A Python budget allocator that fills sections in priority order and reports exactly what was dropped. Run it and the printed report tells you which section gave up tokens.

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Section:
    name: str
    text: str
    priority: int                     # 1 = keep first, higher = drop first
    tokens: int = 0
    dropped_chars: int = 0


def count_tokens(text: str) -> int:
    """Stand-in for the real tokenizer. Swap in tiktoken or the provider's counter."""
    return max(1, len(text) // 4)


def fill(sections: list[Section], input_budget: int) -> tuple[list[Section], int]:
    """Keep sections in priority order until the budget is gone."""
    ordered = sorted(sections, key=lambda s: s.priority)
    kept: list[Section] = []
    remaining = input_budget
    for s in ordered:
        s.tokens = count_tokens(s.text)
        if s.tokens <= remaining:
            kept.append(s)
            remaining -= s.tokens
        elif remaining > 50:
            # partial keep: cut to the remaining budget, note what was lost
            cut_chars = remaining * 4
            kept.append(Section(s.name, s.text[:cut_chars], s.priority,
                                tokens=remaining, dropped_chars=len(s.text) - cut_chars))
            remaining = 0
        else:
            s.dropped_chars = len(s.text)
    return kept, remaining


WINDOW = 16_000
MAX_OUTPUT = 2_000
INPUT_BUDGET = WINDOW - MAX_OUTPUT          # 14_000 tokens for input

sections = [
    Section("system", "You answer only from the provided context. " * 10, priority=1),
    Section("question", "What is the refund window for enterprise plans?", priority=1),
    Section("docs", "Refund policy text. " * 900, priority=3),
    Section("history", "Earlier turns. " * 1200, priority=2),
]

kept, left = fill(sections, INPUT_BUDGET)
for s in kept:
    print(f"{s.name:9s} tokens={s.tokens:6d} dropped_chars={s.dropped_chars}")
print("unused input tokens:", left)
```

The same budgeting problem appears on the server side of a chat product, where the request is assembled in Node. This one enforces the budget before the call is made.

```ts
// contextBudget.ts — Node 20+, ESM
export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export type TokenCounter = (text: string) => number;

export interface BudgetResult {
  messages: Message[];
  droppedTurns: number;
  inputTokens: number;
}

/**
 * Keep the system message and the most recent turns that fit.
 * The counter is injected so tests can use a deterministic fake.
 */
export function fitToBudget(
  messages: Message[],
  countTokens: TokenCounter,
  inputBudget: number,
): BudgetResult {
  const system = messages.filter((m) => m.role === "system");
  const turns = messages.filter((m) => m.role !== "system");

  let used = system.reduce((n, m) => n + countTokens(m.content), 0);
  const kept: Message[] = [];

  // walk backwards: recency wins, so newest turns are kept first
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const cost = countTokens(turns[i].content);
    if (used + cost > inputBudget) break;
    kept.unshift(turns[i]);
    used += cost;
  }

  if (used > inputBudget) {
    // the system prompt alone overflows: fail loudly instead of sending
    throw new Error(`system prompt exceeds budget: ${used} > ${inputBudget}`);
  }
  return { messages: [...system, ...kept], droppedTurns: turns.length - kept.length, inputTokens: used };
}
```

## Real use case

A support assistant answers questions from a help centre. Agents complain that it "ignores the article" when a ticket is long, but works fine on short ones. The team assumes the retrieval is bad and rebuilds the index twice.

The real cause is ordering. The pipeline puts the system prompt, then six retrieved articles, then the full ticket history, then the question. On long tickets, the history pushes the articles into the middle of the window and the model's recall of them drops; on short tickets everything is near an edge, so it works.

The fix is a token budget with priorities: system 800, question always last, articles capped at 6,000 tokens with the highest-scoring article first and second-highest last, history capped at 2,000 tokens and dropped oldest-first with a one-line summary. Nothing about the index changed, and the "ignores the article" complaints stop.

## Common mistakes

- **Mistake:** budgeting input without reserving output tokens. — **Instead:** subtract `max_tokens` from the window first.
- **Mistake:** burying the important document in the middle of many. — **Instead:** put the two most relevant chunks at the start and end of the context block.
- **Mistake:** truncating by characters. — **Instead:** count tokens with the real tokenizer and cut on a section boundary.
- **Mistake:** returning a fixed number of chunks from retrieval. — **Instead:** cap by tokens; chunk counts vary wildly in size.
- **Mistake:** letting the provider truncate silently. — **Instead:** enforce the budget yourself and log what was dropped.

## Checkpoint

1. Change `MAX_OUTPUT` to 8,000 in the Python snippet and describe which sections get cut and why.
2. Add a fifth section with priority 0 that can never be dropped, and make `fill` raise if it cannot fit.
3. Write a test for `fitToBudget` with a fake counter (`text.length`) that proves the newest turn is always kept and the oldest is dropped first.

**Self-check:** With an 8,000-token output reservation the input budget halves to 8,000, so the lowest-priority section (`docs`, priority 3) is partially cut or dropped and `history` is cut next, while `system` and `question` survive intact.

## Go deeper

- [4. Context Compression](../08-prompt-and-context/04-context-compression.md) — summarise instead of dropping.
- [9. Ship: Tokenizer and Token-Budget Analyser](09-ship-tokenizer.md) — build the measurement tool.
- [3. Chunking](../10-rag/03-chunking.md) — deciding chunk size in tokens.
- [8. Cost Tracking](../07-llm-apis/08-cost-tracking.md) — turning the budget into a cost estimate.

---
[← Previous: Tokenization: BPE and Byte-Level](02-tokenization.md) · [Stage index](README.md) · [Next: Self-Attention From Scratch](04-self-attention.md)
