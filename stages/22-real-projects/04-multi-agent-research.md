# 4. Multi-Agent Research

> **Stage 22 · Real Projects** — a supervisor that splits a broad question across specialist workers, merges what they find into one evidence board, and knows when to stop.
>
> **You'll be able to:** build a multi-agent system whose parts share state instead of chatting, whose budget is split explicitly, and whose output is better than a single agent on the same question.
>
> **Prerequisites:** [Multi-Agent Systems](../11-agents/07-multi-agent.md) · [Graph Engineering](../12-orchestration/03-graph-engineering.md)

## Why this exists

Broad questions have parts that need different tools: a market question needs filings and news, a technical question needs source code and papers, a policy question needs statutes. One agent with one toolset does all of it badly — its context fills with material it does not need, and its plan drifts as it reads.

The naive fix, agents talking to each other in free-form text, is where most multi-agent projects die. Transcripts grow, the same fact gets rediscovered three times, one worker's confident wrong claim propagates because another agent "heard it", and cost doubles for output nobody can attribute.

The buyers are analyst teams and consultancies doing wide, shallow coverage on a deadline. What they need is coverage, not conversation.

Mental model: agents are *workers on a shared board*, not participants in a chat. They write findings into typed state; the supervisor reads the board and assigns more work; nothing is true until it is on the board with a source.

## The mental model

```text
                    [supervisor: plan + budget + stop rule]
                        |            |             |
          +-------------+            |             +--------------+
          v                          v                            v
  [filing worker]            [web/news worker]            [code/papers worker]
          |                          |                            |
          +----------> [shared evidence board] <------------------+
                       claim, source, span,
                       worker, cost, confidence
                                |
                    [dedupe + contradiction check]
                                |
                        [writer: one voice]
```

Every worker returns the same typed object. The board is the only channel — workers never talk to each other.

## Core concepts

### Scope and non-goals

In scope: a question decomposed into facets, one worker per facet with a restricted toolset, a shared evidence board with deduplication, explicit per-worker budget, contradiction surfacing, and a single writer.

Non-goals: open-ended agent-to-agent conversation, dynamic agent creation, long-lived memory across runs, and human-in-the-loop mid-run. Each of these multiplies failure modes; add one at a time and only with an eval that proves it helped.

### The board beats conversation

Give every worker the same output schema — claims with a source and a verbatim span, the same object as [Research Assistant](01-research-assistant.md). Then "communication" is a database insert, not a message. This buys you: deduplication (two workers finding the same filing collapse to one claim), attribution (each claim knows which worker and which cost produced it), and determinism (the writer reads a table, not a transcript).

### Budget is a first-class field

Split the run budget across workers *before* they start, weighted by facet difficulty, and let a worker hard-stop when it hits its share. Without this, one worker with a hard question consumes everything and the run ends with three empty facets. Record spend per claim so you can answer "what did this paragraph cost" — an interview question you should be able to answer with a query, not an estimate.

### Contradictions are output, not noise

When two workers return claims that conflict (same entity, different number, same period), do not silently pick one. Store both, mark the pair, and let the writer report the disagreement with both sources. A research product that says "two sources disagree, here they are" is more valuable than one that averages them into a confident lie.

### Milestones: four weeks

- Week 1 — single worker, board schema, writer. Success: the single-agent baseline is reproducible and measured.
- Week 2 — supervisor with facets and per-worker toolsets. Success: each facet gets relevant sources, not the whole web.
- Week 3 — budget split, dedupe, contradiction pairs. Success: no worker exceeds its share and duplicates collapse.
- Week 4 — the comparison that matters: multi-agent vs single-agent on the same frozen question set, on coverage and cost.

### Eval plan and budget arithmetic

The only reason to build this is if it beats the single agent. Measure both on the same questions:

- **Facet coverage** — share of planned facets with at least one supported claim.
- **Claim precision** — spans verified (deterministic, from the shared verifier).
- **Duplication rate** — distinct claims over total claims; low means workers are wasting money on the same pages.
- **Cost and wall-clock per run**, compared to the single agent. Report the ratio honestly, including when it is worse.
- **Faithfulness** of the final report by judge on a hand-read sample.

Budget, illustrative: five workers at 20k input tokens each is 100k tokens for reading, versus roughly 40k for a single agent doing the same breadth. That gap is the price of parallelism; it buys wall-clock time, since the workers run concurrently, and coverage, since each worker's context stays small. If your wall-clock p95 target is 90 seconds, run workers concurrently with a per-worker timeout, and let the writer start once the board stops growing rather than when the slowest worker finishes.

### Tech stack and data model

LangGraph or a hand-written supervisor loop (see [Tradeoffs](../12-orchestration/10-tradeoffs.md)); Postgres for the board; the same fetch and extract stack as the research assistant.

| table | key fields |
|-------|-----------|
| `run` | id, question, facets_json, budget_usd, status |
| `task` | id, run_id, facet, worker, budget_usd, spent_usd, status |
| `claim` | id, run_id, task_id, text, source_url, span, verdict, dup_of |
| `conflict` | id, run_id, claim_a, claim_b, reason |

`dup_of` and `conflict` are the two columns that make the board useful.

## Code

The supervisor's genuinely hard job is deciding what still needs doing. This snippet assigns work from board state, enforces per-worker budget, and terminates on coverage rather than on a step count.

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class Facet:
    name: str
    question: str
    weight: float = 1.0


@dataclass
class Board:
    claims: dict[str, Claim] = field(default_factory=dict)   # claim_id -> claim
    covered: set[str] = field(default_factory=set)           # facet names with >=1 claim

    def add(self, facet: str, claims: list[Claim]) -> None:
        for c in claims:
            if c.dup_key not in {x.dup_key for x in self.claims.values()}:
                self.claims[c.id] = c
                self.covered.add(facet)


async def run_supervisor(question: str, facets: list[Facet], budget_usd: float) -> Board:
    board = Board()
    total_weight = sum(f.weight for f in facets)
    shares = {f.name: budget_usd * f.weight / total_weight for f in facets}
    spent: dict[str, float] = {f.name: 0.0 for f in facets}
    pending = list(facets)
    rounds = 0

    while pending and rounds < 3:
        rounds += 1
        # only dispatch facets that still have budget and no coverage
        batch = [f for f in pending if spent[f.name] < shares[f.name] and f.name not in board.covered]
        if not batch:
            break
        results = await asyncio.gather(
            *[run_worker(f, budget_usd=shares[f.name] - spent[f.name]) for f in batch],
            return_exceptions=True,      # one dead worker must not kill the run
        )
        for facet, result in zip(batch, results):
            if isinstance(result, Exception):
                continue
            spent[facet.name] += result.cost_usd
            board.add(facet.name, result.claims)

    board.claims.update(find_conflicts(board.claims))
    return board
```

Workers share one schema, so the merge is a dictionary operation rather than a negotiation:

```python
from pydantic import BaseModel

class Claim(BaseModel):
    id: str
    facet: str
    text: str
    source_url: str
    span: str
    dup_key: str          # normalised (entity, attribute, period) for dedupe

class WorkerResult(BaseModel):
    facet: str
    claims: list[Claim]
    cost_usd: float
    notes: str
```

## Real use case

A consultancy's research team needs a same-day briefing on a sector question: pricing moves, regulatory changes, competitor announcements. Before: three analysts each read one slice and merged in a document, losing track of who found what, and a single-agent attempt produced a report that was strong on news and empty on filings because the one toolset was dominated by web search.

After: three workers with different toolsets write to one board; the writer produces one voice with numbered citations and a "sources disagree" paragraph the analysts actually use in client calls.

The design choice that made it work is the typed board: merging became dedupe plus conflict detection, and the cost per paragraph became a query.

Demo script, two minutes: run a question, show the board filling per facet, open the duplicate that collapsed, show the conflict pair, then show the single-agent baseline side by side on coverage and cost. Talking points: why workers do not talk, how you split budget, why termination is coverage-based, and the honest answer to "was multi-agent worth it here" — sometimes it is not.

## Common mistakes

**Mistake:** letting workers message each other — **Instead:** one typed board; transcripts are unmeasurable and unverifiable.
**Mistake:** equal budgets for unequal facets — **Instead:** weight by expected difficulty and let a worker hard-stop.
**Mistake:** terminating on step count — **Instead:** terminate on coverage plus budget, and log why you stopped.
**Mistake:** picking a winner when sources conflict — **Instead:** report both with sources; disagreement is the finding.
**Mistake:** no single-agent baseline — **Instead:** measure it first, or you cannot claim the architecture helped.
**Mistake:** `gather` without `return_exceptions` — **Instead:** catch per worker; one timeout should degrade coverage, not the run.

## Checkpoint

1. Add a `stop_reason` column and record which condition ended each run. Inspect ten runs: how many stopped for coverage, how many for budget?
2. Add a per-worker timeout and measure coverage loss against the timeout value. Pick a value and justify it in the README.
3. Compare against the single agent on the same ten questions and write the results table, including the runs where multi-agent lost.

**Self-check:** Exercise 1 usually shows most runs stopping on budget or rounds, which is the signal to reweight facets rather than add workers.

## Go deeper

- [Research Assistant](01-research-assistant.md) — the single-agent baseline you must beat
- [Agent Evals](../15-evals/06-agent-evals.md) — measuring loops, not just final answers
- [Loop Engineering](../11-agents/10-loop-engineering.md) — termination rules that survive real questions
- [Agent Design](../21-system-design/03-agent-design.md) — turn this into a design-doc answer

---

[← Previous: Production RAG](03-production-rag.md) · [Stage index](README.md) · [Next: Support Agent](05-support-agent.md)
