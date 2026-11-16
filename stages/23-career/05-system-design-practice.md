# 5. System Design Practice

> **Stage 23 · Career** — a repeatable thirty-minute structure for design rounds, plus two worked examples you can adapt to whatever prompt you are given.
>
> **You'll be able to:** run a design interview from requirements to trade-offs without rambling, do the arithmetic out loud, and land two full examples end to end.
>
> **Prerequisites:** [Scalable AI Apps](../21-system-design/01-scalable-ai-apps.md) · [Design Doc](../21-system-design/10-ship-design-doc.md)

## Why this exists

Design rounds are graded on structure, not on the diagram. Most candidates lose marks in the first five minutes: they start drawing boxes before knowing the scale, so the design cannot be evaluated. Or they spend twenty minutes on retrieval internals and never reach cost, failure handling, or what they would measure.

The interviewer is looking for someone they would trust with a vague brief. That means: ask what matters, commit to a design, name where it breaks, and say what you would instrument.

Mental model: requirements → success metric → data flow → components and data model → bottlenecks → failure modes → trade-offs → next steps. Eight boxes, each with a time budget.

## The mental model

```text
0-3   clarify: users, scale, latency target, budget, quality bar, constraints
3-5   success metric: what number proves this works
5-10  data flow: the happy path, left to right
10-20 components + data model: tables, indexes, services, queues
20-25 bottlenecks: cost, latency, quality -- with arithmetic
25-28 failure modes: what breaks, what degrades, what alerts
28-30 trade-offs: what you would do differently at 10x, and why not now
```

Say the time budget out loud at the start. Interviewers relax when they can see you are managing the clock.

## Core concepts

### Requirements come first, and they are numbers

Ask for: who the user is, how many of them, request rate, latency expectation, cost ceiling, and what "good" means. Convert to numbers on the spot. "Ten thousand questions a day" is roughly one every nine seconds on average — trivially low QPS, so the design is not about throughput at all; it is about quality and cost. Saying that out loud demonstrates judgement better than any diagram.

### Pick one metric and defend it

For an assistant: task success on a frozen set, or citation support rate, or deflection. For a pipeline: field accuracy at auto-accept. For a platform: p95 latency at a fixed cost. A design with no metric cannot be evaluated, and interviewers notice.

### Draw data flow, not architecture astronauts

Left to right: source → ingestion → storage → retrieval/agent → model → response → logging → eval. Then add the asynchronous parts (queues, workers) and the control parts (auth, quotas, rate limits) only where they change behaviour. Keep the diagram small enough to read while you talk.

### Do the arithmetic out loud

Every AI design has three numbers that decide it: tokens per request, cost per 1,000 requests, and p95 latency. Compute them from stated assumptions, say they are illustrative, and note where you would verify them. An interviewer would rather hear "assume 4k input tokens, so roughly X per thousand requests at current rates — I would check the pricing page" than a confident wrong figure.

### Failure modes are a checklist, not inspiration

Walk the same list every time: provider outage or rate limit, retrieval returning nothing relevant, model returning invalid JSON, one tenant flooding the system, stale index, cost spike from a loop, prompt injection from retrieved content, and data deletion not propagating. For each: what degrades, what the user sees, what alerts. This section is where senior candidates separate themselves.

### Trade-offs at 10x

Finish with what changes at ten times the load or a tenth of the budget: shard the index, add a cache, route cheap models for easy queries, batch offline, precompute embeddings. Naming the trigger ("when p95 exceeds the target") is stronger than naming the technology.

## Code

A capacity worksheet you can run in your head or in a notebook during the interview. Doing this in code at home is how the numbers become reflexive.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Assumptions:
    requests_per_day: int
    tokens_in: int
    tokens_out: int
    usd_per_mtok_in: float      # illustrative: read current rates from the provider
    usd_per_mtok_out: float
    peak_factor: float = 3.0    # peak QPS vs average
    seconds_per_request: float = 3.0


def worksheet(a: Assumptions) -> dict[str, float]:
    per_req_cost = (
        a.tokens_in / 1_000_000 * a.usd_per_mtok_in
        + a.tokens_out / 1_000_000 * a.usd_per_mtok_out
    )
    avg_qps = a.requests_per_day / 86_400
    peak_qps = avg_qps * a.peak_factor
    return {
        "cost_per_request": per_req_cost,
        "cost_per_1k": per_req_cost * 1_000,
        "cost_per_day": per_req_cost * a.requests_per_day,
        "avg_qps": avg_qps,
        "peak_qps": peak_qps,
        # concurrency = arrival rate x time in system (Little's law)
        "concurrent_requests_at_peak": peak_qps * a.seconds_per_request,
        "tokens_per_day": a.requests_per_day * (a.tokens_in + a.tokens_out),
    }


if __name__ == "__main__":
    a = Assumptions(
        requests_per_day=10_000, tokens_in=4_000, tokens_out=400,
        usd_per_mtok_in=0.0, usd_per_mtok_out=0.0,   # fill from current pricing docs
    )
    for k, v in worksheet(a).items():
        print(f"{k:32} {v:,.4f}")
```

Fill the two rate fields from the provider's pricing page before you trust any output — they change, and quoting a stale rate in an interview is worse than saying you would look it up.

## Real use case

Two worked examples, each in the shape you would say them.

**Example A — internal knowledge assistant for a 500-person company.** Requirements: 500 users, maybe 2,000 questions a day at peak, sub-5-second answers, strict permissions, answers must cite a page. Success metric: citation support rate plus time-to-answer versus search. Data flow: wiki connector on a schedule and webhooks → chunker with heading paths → vector store with `acl_groups` and `updated_at` payloads → query-time group expansion → filtered retrieval → rerank → answer with page, owner and date. Bottlenecks: sync cost dominates (hash and re-embed only changed pages); latency is rerank plus generation, so cap candidates. Failure modes: ACL drift after a reorg (mitigate by expanding groups at query time), stale pages (show dates and warn), injection in page text (treat as data). At 10x: precompute embeddings offline, add a semantic cache, shard by space.

**Example B — invoice processing at volume.** Requirements: 20,000 documents a month, results within an hour of upload, 99% field accuracy at auto-accept for header fields, human review for the rest. Success metric: auto-accept precision and review load per document. Data flow: upload → object storage → OCR worker queue → layout parse → rules extractor → vision fallback for missing fields → merge and confidence → auto-accept or review queue → ledger. Bottlenecks: OCR and vision calls are the cost; run rules first to avoid most model calls. Failure modes: unreadable scans (reject early with a clear message), a new supplier template (fallback extractor handles it, and the review queue reveals the pattern), duplicate uploads (hash and dedupe). At 10x: batch the vision calls, cache by page hash, add per-supplier rules.

## Common mistakes

**Mistake:** drawing before asking — **Instead:** three minutes of requirements, then draw.
**Mistake:** no numbers — **Instead:** state assumptions and do the arithmetic out loud, flagged as illustrative.
**Mistake:** designing for imaginary scale — **Instead:** compute QPS first; most internal tools are low-QPS problems about quality and cost.
**Mistake:** ignoring permissions and deletion — **Instead:** put them in the data model, not in a later phase.
**Mistake:** a design with no failure section — **Instead:** walk the checklist every time.
**Mistake:** ending without next steps — **Instead:** name what you would measure first and what would change at 10x.

## Checkpoint

1. Take Example A and redo it for a company where the wiki is 50 times larger and questions arrive in bursts. What changes first, and what stays the same?
2. Run the worksheet for Example B with your own assumptions about documents per month and pages per document. Which line item dominates?
3. Time yourself: requirements in three minutes, full design in thirty. Record yourself and count how many minutes you spent on retrieval internals.

**Self-check:** Exercise 1 changes the sync and storage strategy (incremental sync, sharding) while the permission and freshness design stays identical, because those are correctness concerns rather than scale concerns.

## Go deeper

- [Multi-Tenant](../21-system-design/05-multi-tenant.md) and [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — the two sections interviewers probe hardest
- [Knowledge Assistant](../22-real-projects/09-knowledge-assistant.md) — Example A as a build brief
- [Document Intelligence](../22-real-projects/02-document-intelligence.md) — Example B as a build brief
- [Inference Optimization](07-inference-optimization.md) — the arithmetic round that follows design

---

[← Previous: ML and LLM Basics](04-ml-llm-basics.md) · [Stage index](README.md) · [Next: RAG and Agent Patterns](06-rag-agent-patterns.md)
