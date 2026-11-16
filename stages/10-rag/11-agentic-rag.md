# 11. Agentic RAG

> **Stage 10 · RAG** — moving retrieval inside the agent loop, so the model decides what to search, when to search again, and when it has enough.
>
> **You'll be able to:** build a retrieval agent with a bounded search loop, decide when a fixed pipeline is the better answer, and measure whether the agent actually retrieves better.
>
> **Prerequisites:** [RAG Architecture](01-rag-architecture.md) · [Agent Architecture](../11-agents/02-agent-architecture.md) · [Loop Engineering](../11-agents/10-loop-engineering.md)

## Why this exists

A fixed RAG pipeline makes every decision before it sees the question: one rewrite, one search, top-k, rerank, answer. That is fast, cheap and predictable — and it fails on questions that need a second look. *"Compare our 2023 and 2024 refund policy"* needs two searches, and the first result tells you what the second search should be.

Agentic RAG hands that decision to the model: give it a `search` tool and a budget, and let it iterate. The cost is latency, money and unpredictability. The gain is that multi-part and ambiguous questions stop being edge cases.

The mental model: **fixed RAG is a pipeline; agentic RAG is a research assistant with a search box.**

## The mental model

```text
   fixed RAG                          agentic RAG

   question                           question
      │                                  │
   rewrite                            ┌──▼──────────────┐
      │                               │  think: what do │
   search ──► top-k                   │  I still need?  │
      │                               └──┬───────────▲──┘
   rerank                                 │ search    │ observe
      │                                ┌──▼───────┐   │
   answer                              │ retrieve │───┘
                                       └──────────┘
   one shot, predictable                  │ enough?
                                       ┌──▼──────┐
                                       │ answer  │   with citations,
                                       └─────────┘   or "not found"
```

The box that decides "enough?" is where all the engineering lives: a step budget, a no-progress check, and a rule that refuses to answer without evidence.

## Core concepts

### The search tool is the product

Everything the agent can do, it does through one tool. Its description determines whether the agent searches well, so write it like documentation for a new colleague: what the index covers, what a good query looks like, and what it returns.

```python
SEARCH_TOOL = {
    "name": "search",
    "description": (
        "Search the policy corpus. Use short keyword queries (3-8 words), not full sentences. "
        "Returns up to 6 passages with ids and scores. Call again with different terms if the "
        "first results do not contain the answer. Never answer from memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 6},
        },
        "required": ["query"],
    },
}
```

### Bounded iteration, not open-ended browsing

Three limits make an agentic retriever shippable: a step budget (usually 3-5 searches), a total token budget, and a rule that stops when two consecutive searches return the same document ids — the agent is looping, not learning.

```python
def is_stuck(seen: list[set[str]], min_overlap: float = 0.8) -> bool:
    if len(seen) < 2:
        return False
    a, b = seen[-2], seen[-1]
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= min_overlap
```

### Citations are the safety net

An agent that can search freely can also answer freely. Force the final answer to name the passage ids it used, and validate that those ids appeared in the tool results. If the model cites a passage it never retrieved, you have caught a hallucination mechanically, not by vibes.

### The honest comparison

| | Fixed RAG | Agentic RAG |
|---|---|---|
| Latency | one pass, ~1-3 s | 2-6 model calls, 4-15 s |
| Cost | low and stable | 3-8× a fixed pass |
| Multi-part questions | fails | handles |
| Predictability | high | medium — bound it |
| Debugging | trace the pipeline | trace the trajectory |
| Best fit | high-volume FAQ | research, comparison, compliance |

Most production systems end up hybrid: a router sends simple questions to the fixed pipeline and complex ones to the agent. That is the design this file recommends.

## Code

A bounded retrieval agent with citation validation.

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

client = OpenAI()

@dataclass
class Trace:
    searches: list[dict] = field(default_factory=list)
    seen: list[set[str]] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""


def agentic_answer(question: str, search_fn, max_searches: int = 4) -> Trace:
    trace = Trace()
    messages = [
        {"role": "system", "content": (
            "You answer questions about company policy. You must call search before answering. "
            "Cite passage ids as [id]. If the corpus does not contain the answer, say 'not found'."
        )},
        {"role": "user", "content": question},
    ]

    for _ in range(max_searches):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",                # check current model ids in provider docs
            messages=messages,
            tools=[{"type": "function", "function": SEARCH_TOOL}],
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            trace.answer = msg.content or ""
            trace.stop_reason = "answered"
            return trace

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            hits = search_fn(args["query"], args.get("k", 6))
            ids = {h["id"] for h in hits}
            trace.searches.append({"query": args["query"], "ids": sorted(ids)})
            trace.seen.append(ids)
            if is_stuck(trace.seen):
                trace.stop_reason = "no_progress"
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": "Same passages as the previous search. Answer with what you have, "
                                            "or say 'not found'."})
            else:
                payload = [{"id": h["id"], "text": h["text"][:800]} for h in hits]
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)})

    # budget exhausted: force an answer from what we have, and be explicit about it
    messages.append({"role": "user", "content": "Search budget exhausted. Answer from the passages already retrieved, "
                                                "or reply 'not found'. Cite ids."})
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    trace.answer = resp.choices[0].message.content or ""
    trace.stop_reason = "budget_exhausted"
    return trace


def validate_citations(trace: Trace) -> bool:
    """Every [id] in the answer must have come from a real search result."""
    import re
    cited = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", trace.answer))
    retrieved = set().union(*trace.seen) if trace.seen else set()
    return cited.issubset(retrieved)
```

TypeScript, streaming the trajectory to a UI so users can watch it search:

```ts
type Step = { type: "search"; query: string } | { type: "answer"; text: string };

export async function* streamAgent(question: string): AsyncGenerator<Step> {
  const res = await fetch(`${process.env.RAG_URL}/agent/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.body) throw new Error("no stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const part of parts) {
      if (part.startsWith("data: ")) yield JSON.parse(part.slice(6)) as Step;
    }
  }
}
```

## Real use case

An insurer deploys a fixed RAG pipeline for policy questions. It handles roughly four in five questions well; the failures cluster into two shapes — comparisons ("what changed between the 2023 and 2024 wording") and questions whose first phrasing does not match the document vocabulary.

They add a router: a cheap classifier sends comparison-shaped and low-confidence questions to a bounded agent (max four searches), everything else stays on the fixed path. The agent resolves most of the comparison cases in two searches. Because the agent path costs several times more per question, they cap it per user per day and log the trajectory of every agent run.

The failure they had to design around was citation drift: the agent, having seen a passage three searches ago, cited an id from a *different* search. Validating that every cited id appeared in the union of results caught it; showing the model the retrieved passages with their ids in the tool message, rather than a wall of text, fixed it.

## Common mistakes

- **Mistake:** an unbounded `while True` agent — **Instead:** hard search budget, token budget and no-progress detection, all three.
- **Mistake:** letting the agent answer from memory when retrieval fails — **Instead:** make "not found" an acceptable, tested output, and score it in your evals.
- **Mistake:** returning whole documents to the tool result — **Instead:** return short passages with ids; long tool results crowd out the reasoning and inflate cost.
- **Mistake:** not validating citations — **Instead:** assert every cited id was retrieved; this is your cheapest hallucination detector.
- **Mistake:** routing every question to the agent because it is better on hard cases — **Instead:** route on question shape; the fixed path is 3-8× cheaper and usually right.
- **Mistake:** logging only the final answer — **Instead:** log the trajectory (queries, ids, stop reason); without it, a bad answer is undebuggable.

## Checkpoint

1. Build the bounded agent above with a fake `search_fn` over five documents. Ask a comparison question and count the searches it takes.
2. Add `is_stuck` and confirm that a deliberately impossible question ends with `no_progress` or `budget_exhausted` rather than a fabricated answer.
3. Take ten questions from your own eval set, run them through both fixed RAG and agentic RAG, and tabulate accuracy against cost and latency.

**Self-check:** if the agent's accuracy gain does not justify its cost multiple on your own data, keep the fixed pipeline and add a reranker instead.

## Go deeper

- [Graph RAG](10-graph-rag.md) — the other answer to multi-hop questions
- [RAG Evaluation](12-rag-evals.md) — how to score a trajectory, not just an answer
- [Agent Evals](../15-evals/06-agent-evals.md) — trajectory versus outcome grading

---
[← Previous: Graph RAG](10-graph-rag.md) · [Stage index](README.md) · [Next: RAG Evaluation](12-rag-evals.md)
