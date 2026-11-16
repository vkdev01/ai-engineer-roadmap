# 8. Context Engineering

> **Stage 08 · Prompt and Context** — assembling the right tokens, in the right order, inside a fixed budget.
>
> **You'll be able to:** budget a context window, order sections for attention, and place critical instructions where the model actually reads them.
>
> **Prerequisites:** [Context Compression](04-context-compression.md) · [Context Windows](../06-llm-fundamentals/03-context-windows.md)

## Why this exists

Prompting is what you say. Context engineering is what you put around it: which documents, which history, which tool results, in which order, within which budget. Two systems with identical prompts and identical models can differ enormously in quality purely because one assembles its context well and the other dumps everything it has.

The reason order matters is empirical and well known in practice: models attend unevenly across a long input. Material at the very beginning and very end is used more reliably than material buried in the middle — the "lost in the middle" effect. A fact at position 40 of 80 chunks may as well not be there.

The mental model: **the context window is a budget you allocate deliberately, and position is a resource as scarce as tokens.**

## The mental model

```text
+--------------------------------------------------+
| system rules            (fixed, top)             |  always first
+--------------------------------------------------+
| task + output contract  (fixed, near top)        |  what to do
+--------------------------------------------------+
| high-relevance evidence (ranked)                 |  the payload
| ...                                              |
| weak / background material  (middle = weakest)   |
+--------------------------------------------------+
| the actual question     (last)                   |  recency slot
+--------------------------------------------------+
```

| Budget line | Typical share | Notes |
|---|---|---|
| System rules + contract | fixed, small | do not let it grow unbounded |
| Few-shot examples | 10-25% | cut first when tight |
| Retrieved evidence | 40-60% | ranked, deduped |
| Conversation history | 10-20% | recent turns reserved |
| Output allowance | reserve | max output tokens + margin |

## Core concepts

### Budget the window in named lines

Write down the allocation before you write code: system, examples, evidence, history, output reserve, safety margin. When a request grows, you can see which line to shrink. Without named lines, every change is a guess and the prompt silently crowds out evidence.

### Put the question last, the rules first

The two positions with the most reliable attention are the start and the end. Put standing rules and the output contract at the start; put the actual question at the end, after the evidence. Do not bury the question in the middle of retrieved chunks — a model can lose track of what it was asked.

### Rank, do not just filter

Compression decides what fits; ranking decides what goes first. Sort evidence by relevance score descending so the strongest material sits adjacent to the question. If you must include weak material, put it in the middle, where attention is weakest, not at the end where it competes with the question.

### Reserve output space explicitly

A context that fills the window leaves no room to answer, and some APIs error rather than truncate. Reserve `max_output_tokens` plus a margin. For reasoning models that emit long traces, reserve more.

### Keep structure stable between calls

Changing section order between calls makes behaviour non-reproducible and your evals noisy. Fix the order in the template, and make the amount of each section the only variable.

### More context is not more information

Every irrelevant token is a competitor for attention and a cost. Adding "just in case" material measurably reduces quality on tasks that need precision. The best context is the smallest one that contains the answer.

## Code

An assembler that enforces a named budget, orders sections deliberately, and reports what it used.

```python
from __future__ import annotations

from dataclasses import dataclass, field


def approx_tokens(text: str) -> int:
    # rough: ~4 characters per token for English. Use a real tokeniser if you
    # need precision against a hard limit.
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Budget:
    window: int = 32_000
    system: int = 900
    examples: int = 1_500
    history: int = 2_500
    output_reserve: int = 2_000
    margin: int = 1_000

    @property
    def evidence(self) -> int:
        used = self.system + self.examples + self.history + self.output_reserve
        return max(0, self.window - used - self.margin)


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    score: float          # higher = more relevant; from your ranker


@dataclass
class Assembled:
    text: str
    used_tokens: dict[str, int] = field(default_factory=dict)
    included_ids: list[str] = field(default_factory=list)


def assemble(
    *,
    system: str,
    task_and_contract: str,
    question: str,
    evidence: list[Evidence],
    history: list[str],
    budget: Budget = Budget(),
) -> Assembled:
    """Order: rules, contract, ranked evidence, weak evidence, question last."""
    used = {"system": approx_tokens(system) + approx_tokens(task_and_contract)}

    # 1. history: keep the most recent turns that fit, oldest dropped first
    kept_history: list[str] = []
    for turn in reversed(history):
        cost = approx_tokens(turn)
        if used["history"] + cost > budget.history:
            break
        kept_history.insert(0, turn)
        used["history"] = used.get("history", 0) + cost

    # 2. evidence: rank descending, fill until the evidence line is spent
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    kept: list[Evidence] = []
    ev_tokens = 0
    for ev in ranked:
        cost = approx_tokens(ev.text)
        if ev_tokens + cost > budget.evidence:
            continue                     # skip this one, try a smaller one
        kept.append(ev)
        ev_tokens += cost
    used["evidence"] = ev_tokens

    # 3. strongest evidence adjacent to the question (end of the evidence block)
    parts = [
        system.strip(),
        task_and_contract.strip(),
    ]
    if kept_history:
        parts.append("Conversation so far:\n" + "\n".join(kept_history))
    parts.append(
        "Evidence (most relevant last):\n"
        + "\n\n".join(f"[{e.id}] {e.text}" for e in kept)
    )
    # 4. the question goes last: the recency slot
    parts.append(f"Question:\n{question}")

    text = "\n\n".join(parts)
    used["total"] = approx_tokens(text)
    return Assembled(
        text=text,
        used_tokens=used,
        included_ids=[e.id for e in kept],
    )
```

Note the two decisions encoded there: evidence is sorted ascending in presentation (weakest first, strongest last) even though it is selected by descending score, and the question is the final block.

The TypeScript version makes the budget explicit in the function signature so a reviewer can see the numbers at the call site.

```ts
export type Evidence = { id: string; text: string; score: number };

export type ContextBudget = {
  window: number;
  system: number;
  history: number;
  outputReserve: number;
  margin: number;
};

const approx = (s: string): number => Math.max(1, Math.ceil(s.length / 4));

export function assemble(opts: {
  system: string;
  contract: string;
  question: string;
  evidence: Evidence[];
  history: string[];
  budget?: Partial<ContextBudget>;
}): { text: string; evidenceTokens: number; ids: string[] } {
  const b: ContextBudget = {
    window: 32_000,
    system: 900,
    history: 2_500,
    outputReserve: 2_000,
    margin: 1_000,
    ...opts.budget,
  };
  const evidenceBudget =
    b.window - b.system - b.history - b.outputReserve - b.margin;

  const ranked = [...opts.evidence].sort((x, y) => y.score - x.score);
  const kept: Evidence[] = [];
  let evidenceTokens = 0;
  for (const e of ranked) {
    const cost = approx(e.text);
    if (evidenceTokens + cost > evidenceBudget) continue;
    kept.push(e);
    evidenceTokens += cost;
  }

  // present weakest first, strongest last, so the best sits next to the question
  const presented = [...kept].reverse();

  const text = [
    opts.system.trim(),
    opts.contract.trim(),
    opts.history.length ? `Conversation so far:\n${opts.history.join("\n")}` : "",
    `Evidence (most relevant last):\n${presented
      .map((e) => `[${e.id}] ${e.text}`)
      .join("\n\n")}`,
    `Question:\n${opts.question}`,
  ]
    .filter(Boolean)
    .join("\n\n");

  return { text, evidenceTokens, ids: kept.map((e) => e.id) };
}
```

## Real use case

A legal-research assistant retrieved 30 clauses per question and pasted them in retrieval order. Answers were good on questions whose supporting clause happened to be retrieved first and poor otherwise, and the team suspected the retriever.

Instrumenting position fixed the diagnosis: the supporting clause was usually present in the top 30, but often around position 15-20, which is the middle of a long block. Reordering evidence by score ascending so the strongest clause sat immediately before the question, trimming to the top 12, and moving the question to the last block moved quality without touching the retriever or the prompt text. The failure mode before was treating retrieval as a set — position was never considered a variable.

## Common mistakes

- **Mistake:** filling the window with everything available. — **Instead:** allocate named budget lines and keep only what the task needs.
- **Mistake:** burying the question in the middle of evidence. — **Instead:** put the question in the final block, after the evidence.
- **Mistake:** presenting evidence in retrieval order. — **Instead:** rank by relevance and place the strongest closest to the question.
- **Mistake:** forgetting the output reserve. — **Instead:** subtract `max_output_tokens` plus a margin from the window before allocating input.
- **Mistake:** changing section order between runs. — **Instead:** fix the order in the template so evals measure content, not layout noise.

## Checkpoint

1. Add a per-section token report to the assembler and print it for three real requests; identify which line is closest to its cap.
2. Run the same ten fixtures with evidence ordered best-first versus best-last and record the difference in your pass rate.
3. Shrink the evidence budget by 50% and see which fixtures fail — that tells you which facts were load-bearing.

**Self-check:** you subtract the system prompt, history, output reserve and margin from the window, then fill the remaining evidence budget by descending relevance, presenting the strongest material immediately before the question.

## Go deeper

- [Context Compression](04-context-compression.md) — how the evidence gets small enough to fit.
- [Context Windows](../06-llm-fundamentals/03-context-windows.md) — what the limit physically is.
- [Reranking](../10-rag/06-reranking.md) — a better relevance score for the ordering step.
- [Ship: A Prompt Test Suite](09-ship-prompt-test-suite.md) — proving ordering changes help.

---
[← Previous: Prompt Evals](07-prompt-evals.md) · [Stage index](README.md) · [Next: Ship: A Prompt Test Suite](09-ship-prompt-test-suite.md)
