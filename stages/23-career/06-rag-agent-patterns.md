# 6. RAG and Agent Patterns

> **Stage 23 · Career** — the retrieval and agent questions that decide AI-engineer interviews, answered as patterns with the trade-off attached.
>
> **You'll be able to:** answer chunking, reranking, agent-loop and failure-handling questions with a concrete pattern, a number and a known limitation.
>
> **Prerequisites:** [Chunking](../10-rag/03-chunking.md) · [Agent Architecture](../11-agents/02-agent-architecture.md)

## Why this exists

Almost every AI-engineer interview reaches the same territory: how would you chunk, why rerank, when does an agent loop stop, and what happens when a tool fails. The questions are pattern questions — there is a small set of defensible answers, and the interviewer is checking whether you have run one, because the difference between a worked and unworked answer is visible in the details.

Candidates usually answer with a technology ("we used LangChain") instead of a mechanism ("we split on headings, kept a parent, and reranked 40 candidates down to 6"). The second answer is the one that gets a follow-up about measurement, which is where you want the conversation.

Mental model: every answer has four parts — the pattern, why it beats the naive option, how you measure it, and where it fails.

## The mental model

```text
RAG question          -> chunking / retrieval / rerank / evaluation
   pattern            -> what you actually do
   why not naive      -> the failure it removes
   measured by        -> recall@k, citation support, faithfulness
   fails when         -> the corpus is adversarial / answer spans documents

Agent question        -> loop / tools / memory / termination / safety
   pattern            -> the loop shape and the guard
   why not naive      -> runaway cost, wrong tool, unverified claim
   measured by        -> task success, steps per task, cost per task
   fails when         -> tool semantics are ambiguous or effects repeat
```

## Core concepts

### Chunking: split on structure, retrieve small, read large

The naive answer is "512 tokens with 50 overlap". The stronger answer: split on document structure first — headings, list boundaries, table rows — because a chunk that ends mid-table or mid-clause retrieves badly and reads worse. Then use parent-child: embed small child chunks (a few sentences) for precise matching, and return the parent section to the model for context. Add the heading path to each chunk so the model knows where it is.

Measure with recall@k on a frozen question set: does the chunk containing the answer appear in the top k. That number, for two or three chunking configurations, is the whole answer.

### Hybrid retrieval and reranking: two different jobs

Keyword search (BM25-style) is unbeatable for identifiers, product codes and rare names; dense retrieval is better for paraphrase. Hybrid combines both and fuses the rankings. A cross-encoder reranker then re-scores the top candidates by reading query and chunk together, which is much more accurate than vector distance and much slower — so you rerank a small candidate set (tens), not the corpus.

Say the pipeline concretely: retrieve 40 by hybrid, rerank to 6, then send those to the model. And say the cost: reranking adds latency proportional to candidates times chunk length, so the candidate count is a latency lever.

### Agent loops: the guard is the design

The pattern is a loop: model proposes a tool call, runtime executes, result is appended, repeat. What interviewers want is the guard set:

- A step cap, so a confused model cannot loop forever
- A budget cap in tokens or currency, checked before each call
- Tool schemas that are narrow and typed, so the model cannot pass junk
- A termination condition that is not "the model says done" — for example, a required artefact exists (a claim list, a passing test, a filled schema)
- Structured errors returned to the model so it can recover, plus a retry classification that stops retrying permanent failures

Then the answer to "what does it do when it cannot finish": it returns partial results with a reason, or it escalates with a packet. Never a silent failure.

### Failure handling: name the four classes

Say this list every time: provider failure (timeout, rate limit, 5xx) → retry with backoff and jitter, then fall back to a second model or a cached answer; quality failure (retrieval empty, answer unsupported) → explicit not-found path and verification; tool failure (permanent error, invalid arguments) → return the error to the model once, then stop; effect failure (the side effect may or may not have happened) → reconcile against the system of record, never blind-retry.

That last one is the answer that makes an interviewer lean forward, because it is the one people learn from an incident.

### Memory and state: say what is durable

Short-term state is the conversation; durable state is a store you can resume from. In an agent, keep the plan, the tool results and the budget in persisted state so a crash does not restart the whole run. For retrieval, memory means conversation-aware query rewriting, not stuffing the whole history into the prompt.

## Code

A chunker that respects structure and keeps parents, which is the answer most candidates cannot write:

```python
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    heading_path: str
    parent_id: str


def chunk_markdown(doc: str, doc_id: str, target_chars: int = 900) -> list[Chunk]:
    """Split on headings; inside a section, split on paragraph boundaries.

    Small children for matching, stable parent id so the retriever can fetch
    the whole section for the prompt.
    """
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []          # (level, title)
    section: list[str] = []
    parent_id = f"{doc_id}:root"

    def flush() -> None:
        body = "\n".join(section).strip()
        if not body:
            return
        path = " > ".join(t for _, t in stack) or "root"
        for part in split_paragraphs(body, target_chars):
            chunks.append(Chunk(text=part, heading_path=path, parent_id=parent_id))

    for line in doc.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            section = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            parent_id = f"{doc_id}:{'/'.join(t for _, t in stack)}"
        else:
            section.append(line)
    flush()
    return chunks


def split_paragraphs(body: str, target: int) -> list[str]:
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > target:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out
```

The retrieval pipeline with an explicit candidate budget, which is how you answer the latency follow-up:

```python
async def retrieve(query: str, k_final: int = 6, k_candidates: int = 40) -> list[Chunk]:
    dense = await vector_search(query, limit=k_candidates)
    sparse = await bm25_search(query, limit=k_candidates)
    fused = reciprocal_rank_fusion(dense, sparse)      # rank-based, no score scaling needed
    top = fused[:k_candidates]
    reranked = await rerank(query, top)                # cross-encoder: query+chunk jointly
    return reranked[:k_final]
```

And the loop with guards, which is the shape to write on a whiteboard:

```python
async def run(task: str, tools: ToolRegistry, max_steps: int = 8, budget_usd: float = 0.50) -> dict:
    state = {"task": task, "history": [], "spent": 0.0}
    for step in range(max_steps):
        if state["spent"] > budget_usd:
            return {"status": "budget_exhausted", "partial": state["history"]}
        action = await propose(state)                  # model returns tool name + args
        if action["kind"] == "finish":
            return {"status": "done", "result": action["result"]}
        try:
            result = await tools.execute(action["name"], action["args"])
        except PermanentError as exc:
            state["history"].append({"tool": action["name"], "error": str(exc)})
            continue                                    # let the model see the error once
        state["spent"] += result.cost_usd
        state["history"].append({"tool": action["name"], "output": result.data})
    return {"status": "step_limit", "partial": state["history"]}
```

## Real use case

In an interview, a candidate was asked "your agent booked the same meeting twice — what happened?" The strong answer names the class: the tool call succeeded but the process died before recording it, so the retry repeated the effect. The fix is a write-ahead intent keyed by the conversation and normalised arguments, and reconciliation against the calendar API when the outcome is unknown. Then the measurement: a fault-injection test that kills the worker between effect and record, asserting zero duplicates.

The pattern that makes the answer strong is treating effect failure as its own class rather than lumping it under "errors".

## Common mistakes

**Mistake:** answering with a framework name — **Instead:** describe the mechanism and the numbers.
**Mistake:** fixed-size chunking with no structure — **Instead:** split on headings and rows, then size.
**Mistake:** reranking the whole candidate set — **Instead:** rerank tens, not thousands; the candidate count is your latency lever.
**Mistake:** a loop that ends when the model says so — **Instead:** a required artefact plus step and budget caps.
**Mistake:** blind retries after a timeout on a write — **Instead:** reconcile against the system of record.
**Mistake:** no not-found path — **Instead:** an explicit "I could not find this" with what would help.

## Checkpoint

1. Write the chunker above for your own corpus and check that no chunk ends mid-table. Then measure recall@5 with child chunks versus whole sections.
2. Add reciprocal rank fusion to your retrieval and compare recall@5 with dense-only on the same frozen set.
3. Add fault injection to an agent that performs a write, and prove that a crash between effect and record produces no duplicate.

**Self-check:** Exercise 1 usually shows child chunks improve recall while parents improve answer quality, which is why the parent-child pair exists.

## Go deeper

- [Parent-Child Retrieval](../10-rag/08-parent-child.md) — the mechanism in depth
- [Reranking](../10-rag/06-reranking.md) — cross-encoders and the latency trade
- [Loop Engineering](../11-agents/10-loop-engineering.md) — termination rules that hold up
- [Support Agent](../22-real-projects/05-support-agent.md) — idempotency in a working system

---

[← Previous: System Design Practice](05-system-design-practice.md) · [Stage index](README.md) · [Next: Inference Optimization](07-inference-optimization.md)
