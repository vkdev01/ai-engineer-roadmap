# 9. Scaling Trade-offs

> **Stage 21 · System Design** — the recurring decisions in AI system design, each with both sides, so you can defend a choice instead of defending a habit.
>
> **You'll be able to:** name the trade-off behind each common design decision, state the conditions that flip it, and answer "why not the simpler option" in a review.
>
> **Prerequisites:** [Scalable AI Apps](01-scalable-ai-apps.md) · [Model Routing](04-model-routing.md)

## Why this exists

Design reviews are not quizzes about technology. They are tests of whether you know what you gave up. "We chose a managed vector database" is not an answer; "we chose it because our index is under ten million chunks, we have no on-call capacity for a self-hosted index, and we accept the per-query price until query volume makes it the dominant cost line" is.

Every decision in this stage has a cost on the other side, and the cost is usually one of five currencies: **money, latency, operational effort, quality, or complexity**. Knowing which currency you are spending, and the threshold at which the trade flips, is the whole skill.

The mental model: **a design choice is a bet on your future traffic, your future team and your future prices — name the bet.**

## The mental model

```text
   decision ──► spend one of:
                  money · latency · ops effort · quality · complexity
                          │
                          ▼
              "we flip this when ___ crosses ___"
```

| Decision | Option A | Option B | Flips when |
|---|---|---|---|
| Retrieval store | managed vector DB | self-hosted (pgvector, Qdrant) | query volume makes per-query price dominate, or residency requires it |
| Isolation | shared index + filter | index per tenant | a contract requires physical separation, or one tenant's volume distorts recall |
| Agent shape | free loop | explicit state machine | any run must be resumable, budgeted or approved |
| Model choice | one frontier model | tiered routing | request mix is lopsided and quality is measured per route |
| Retrieval | always retrieve | conditional retrieval | many requests need no documents, and a classifier is accurate enough |
| Freshness | nightly re-index | incremental sync | users complain about stale answers, or a policy change must land in minutes |
| Serving | provider API | self-hosted inference | sustained volume, data residency, or a fine-tuned model with no API |
| Context | long context window | RAG | cost per request matters, or the corpus exceeds what you can send |
| Consistency | cached/stale allowed | always fresh | a wrong stale answer has real cost (money, safety, compliance) |
| Failure | fail the request | degrade with a caveat | partial answers are useful and the caveat is honest |

## Core concepts

### Long context versus retrieval

A large context window tempts you to skip retrieval: put everything in and let the model find it. It works in a demo and breaks on three axes. **Cost** scales linearly with tokens, and input tokens are most of your bill. **Latency** grows with prompt length, both time-to-first-token and total. **Quality** does not grow monotonically — models attend unevenly across very long inputs, and a needle in a large haystack can be missed while a small, well-chosen context is read carefully.

Retrieval wins when the corpus is large, changes often, or is shared across users. Long context wins when the document set is small and fixed, when the task needs cross-document reasoning, or when you need an answer *now* and retrieval quality is not yet trustworthy. The pragmatic default: retrieve, then expand the top hits to their parent sections (see [Parent-Child](../10-rag/08-parent-child.md)) — a middle path that gives the model coherent context without sending the corpus.

### Managed versus self-hosted

Managed buys you operational simplicity and costs money per query and per GB. Self-hosted buys you unit-cost control, residency, and the ability to run a custom model, and costs you on-call, upgrades, index rebuilds and capacity planning. The honest threshold: self-hosting becomes worth it when the saved per-query cost exceeds the fully loaded cost of the engineers and machines who run it — and when you actually have those engineers. A small team self-hosting a vector database to save a modest monthly bill is usually a bad trade; a large team self-hosting to meet a residency requirement is not.

### Consistency, freshness and the cost of being right

Every cache and every index is a bet that slightly old data is acceptable. That bet is fine for a product FAQ and unacceptable for a bank balance. So make freshness per-intent: "account balance" reads the source of truth every time; "how do I reset my password" may be served from a cache for an hour. Write the freshness contract down per intent, and let it drive cache TTLs and index sync frequency rather than picking one global number.

### Quality versus latency versus cost: pick two, per request class

You cannot maximise all three. The resolution is to segment: high-stakes requests get the frontier model, no cache, full retrieval and a human review option; routine requests get a small model, aggressive caching and streamed answers. Segment by *consequence*, not by user tier — a free user asking a medical question deserves the careful path more than an enterprise user asking about office hours.

### Simplicity is a feature with a maintenance budget

Each component you add — a second model provider, a semantic cache, an agent loop, a reranker — is a thing that can break at 2 a.m. and a thing someone must understand. The reviewer's question "why not the simpler option" is not hostility; it is asking you to price your complexity. A good answer names the specific failure the complexity prevents and the volume at which it matters.

## Code

A small decision record that makes trade-offs explicit and reviewable — the artefact that turns "we chose X" into a defensible sentence. Keep these next to the design doc.

```python
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    chosen: str
    alternatives: list[str]
    currency_spent: str = Field(pattern="^(money|latency|ops|quality|complexity)$")
    why_now: str                    # the conditions that make this right today
    flip_when: str                  # the measurable threshold that reverses it
    owner: str


def record(decisions: list[Decision], path: Path) -> None:
    path.write_text(
        json.dumps([d.model_dump() for d in decisions], indent=2), encoding="utf-8"
    )


DECISIONS = [
    Decision(
        id="ADR-001",
        question="Shared vector index or one per tenant?",
        chosen="Shared index with a mandatory tenant filter inside the query",
        alternatives=["collection per tenant", "database per tenant"],
        currency_spent="complexity",  # we pay in review discipline, not money
        why_now="Largest tenant is well under the size where filtering hurts recall",
        flip_when="Any tenant exceeds ~2M chunks, or a contract requires separation",
        owner="platform",
    ),
    Decision(
        id="ADR-002",
        question="Always retrieve, or only when the intent needs documents?",
        chosen="Conditional retrieval behind an intent classifier",
        alternatives=["always retrieve", "never retrieve"],
        currency_spent="latency",  # we pay a classifier call to save retrieval
        why_now="Roughly half of traffic is small talk and account actions",
        flip_when="Classifier precision on the 'needs documents' class drops below 0.9",
        owner="app",
    ),
]
```

The capacity sanity check you should be able to run in your head during a review — three numbers, labelled assumptions, no provider prices.

```python
from __future__ import annotations

# Illustrative assumptions — replace with your own measurements.
PEAK_RPS = 40                 # requests per second at peak
CONCURRENCY_PER_REQ_S = 3.0   # seconds a request occupies a worker
WORKERS_NEEDED = PEAK_RPS * CONCURRENCY_PER_REQ_S
TOKENS_IN = 3_000
TOKENS_OUT = 400
REQ_PER_MONTH = 20_000_000

print(f"in-flight requests at peak : {WORKERS_NEEDED:,.0f}")
print(f"input tokens / month (M)   : {REQ_PER_MONTH * TOKENS_IN / 1e6:,.0f}")
print(f"output tokens / month (M)  : {REQ_PER_MONTH * TOKENS_OUT / 1e6:,.0f}")
# Cost = (input_M * P_in) + (output_M * P_out). Substitute current prices.
```

## Real use case

**A design review for a knowledge assistant serving an enterprise customer.** The first proposal was a single frontier model, retrieval on every request, no cache, and a self-hosted vector database — the team's reasoning being "best quality, full control". A reviewer asked three questions and the design changed shape.

"Why self-host the index?" — the customer required data residency, so a managed option in-region was acceptable and self-hosting was not needed. That removed a component and its on-call burden. "Why retrieve for every request?" — about half the traffic was navigation and account questions needing no documents; a classifier plus a fallback to retrieval cut the retrieval hop from most requests. "Why the frontier model everywhere?" — measurement on a golden set showed the mid tier matched on the routine intents, so routing went in with a per-route quality gate.

Each change spent a different currency: residency spent money (in-region managed pricing), conditional retrieval spent latency (a classifier call) and complexity (a new classifier to monitor), routing spent quality risk (bounded by measurement). Written as decision records with flip thresholds, the design survived the next review because the team could say what would reverse each choice.

## Common mistakes

- **Mistake:** "we chose the managed service because it is easier" — **Instead:** name the currency you spent and the threshold at which you would move.
- **Mistake:** sending the whole corpus in a long context to avoid building retrieval — **Instead:** retrieve and expand to parent sections; long context is a tool for small fixed corpora.
- **Mistake:** one global freshness number for every intent — **Instead:** set freshness per intent, by consequence.
- **Mistake:** optimising quality, latency and cost simultaneously — **Instead:** segment by consequence and pick two per segment.
- **Mistake:** adding a component without naming the failure it prevents — **Instead:** every component gets a sentence: what breaks without it, at what volume it matters.

## Checkpoint

1. Write three decision records for a system you have built, each with a `flip_when` threshold you could actually measure.
2. Compute the in-flight concurrency and monthly token volume for 40 requests per second at peak, using your own average request duration. State what changes at 4× that traffic.
3. Take the long-context-versus-RAG trade-off and write the paragraph you would say in a review for a 200-document internal wiki. Then write the paragraph for a two-million-document corpus.

**Self-check:** Each record names the currency spent, the conditions that justify the choice today, and a measurable threshold that reverses it — the threshold is what turns an opinion into a decision someone can revisit.

## Go deeper

- [Ship: Design Document](10-ship-design-doc.md) — where these records live.
- [Fault Tolerance](07-fault-tolerance.md) — the reliability side of the same trade-offs.
- [Cost and Latency Engineering](08-cost-latency.md) — the arithmetic behind the money and latency currencies.
- [System Design Practice](../23-career/05-system-design-practice.md) — rehearsing these answers out loud.

---
[← Previous: Cost and Latency Engineering](08-cost-latency.md) · [Stage index](README.md) · [Next: Ship: Design Document](10-ship-design-doc.md)
