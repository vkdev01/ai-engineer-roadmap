# 10. Ship: A Design Document

> **Stage 21 · System Design** — a ten-page design document for a real AI product, with capacity arithmetic, failure modes and a cost model a reviewer can check.
>
> **You'll be able to:** write a design document that gets approved, and defend every number in it.
>
> **Prerequisites:** [Scalable AI Apps](01-scalable-ai-apps.md) · [Cost and Latency](08-cost-latency.md) · [Scaling Trade-offs](09-scaling-tradeoffs.md)

## Why this exists

Engineers lose more opportunities to a badly argued design than to bad code. A design review is not a formality; it is where an organisation decides whether to spend money on your idea. The document is the artefact that carries the decision, and its quality is judged in minutes.

The trap is writing architecture without arithmetic. "It scales horizontally" is not a design; "at 40 requests per second with a 2.1k-token average prompt, we need N replicas and it costs X per thousand requests" is.

The mental model: **a design document is a set of claims, each with the number that supports it.**

## The mental model

```text
   requirements ──► assumptions ──► arithmetic ──► decision
        │                │               │            │
   who, how many,    tokens/request   capacity,    the chosen
   how fast, how     users, growth    cost, p95    architecture
   wrong can it be                                   │
        └────────────── risk ────── failure modes ───┘

   every arrow is a paragraph a reviewer can argue with.
   If they cannot argue with it, it is not a claim, it is a vibe.
```

## Core concepts

### Structure that survives review

Ten sections, in this order, each short. Reviewers read the first two and the numbers; the rest is for the implementer.

1. **Problem and users** — who, what they do today, what breaks.
2. **Requirements** — functional, then non-functional with numbers: latency, throughput, accuracy, availability, cost ceiling.
3. **Non-goals** — what you are deliberately not building. This section buys credibility.
4. **Assumptions** — the numbers everything else derives from, each labelled "verify".
5. **Architecture** — one diagram, components, data flow, and the async seams.
6. **Capacity** — arithmetic for storage, compute, concurrency.
7. **Cost model** — per request, per thousand, per month at assumed volume.
8. **Failure modes** — what breaks, what the system does, what the user sees.
9. **Rollout** — phases, flags, migration, rollback.
10. **Open questions** — the ones you want the review to settle.

### Assumptions are the load-bearing wall

Every downstream number derives from a handful of inputs. Write them in a table, mark each one's confidence, and say what happens if it is wrong by 3×. A design that breaks at 3× the assumed load is a different design, and the reviewer needs to know that before approving.

| Assumption | Value | Confidence | If it is 3× higher |
|---|---|---|---|
| Daily active users | 2,000 | medium | need 3× replicas, cache hit rate improves |
| Requests per user per day | 8 | medium | cost line triples |
| Prompt tokens per request | 1,800 | high | KV cache pressure, cost triples |
| Output tokens per request | 320 | high | decode-bound, p95 latency rises |
| Corpus size | 40,000 docs | high | re-embedding cost, index memory |
| Peak-to-average ratio | 4× | low | concurrency sizing, need burst capacity |

### Capacity arithmetic, shown

Reviewers trust arithmetic they can redo. Write the formula, then the substitution, then the result.

```python
# capacity.py — every number in the design doc comes from here, not from memory.

ASSUMPTIONS = {
    "dau": 2_000,
    "requests_per_user_day": 8,
    "prompt_tokens": 1_800,
    "output_tokens": 320,
    "peak_to_average": 4,
    "seconds_per_request": 4.5,
    "price_per_mtok_input": 0.15,     # substitute the provider's current published price
    "price_per_mtok_output": 0.60,    # substitute the provider's current published price
}

A = ASSUMPTIONS
daily_requests = A["dau"] * A["requests_per_user_day"]
peak_rps = daily_requests / 86_400 * A["peak_to_average"]
concurrency = peak_rps * A["seconds_per_request"]

in_tok = daily_requests * A["prompt_tokens"]
out_tok = daily_requests * A["output_tokens"]
daily_cost = (in_tok / 1e6) * A["price_per_mtok_input"] + (out_tok / 1e6) * A["price_per_mtok_output"]

print(f"daily requests      {daily_requests:,}")
print(f"peak rps            {peak_rps:.2f}")
print(f"concurrency needed  {concurrency:.1f} in-flight")
print(f"input tokens/day    {in_tok:,}")
print(f"output tokens/day   {out_tok:,}")
print(f"cost/day            ${daily_cost:,.2f}")
print(f"cost per 1k reqs    ${daily_cost / daily_requests * 1000:,.2f}")
print(f"cost/month          ${daily_cost * 30:,.2f}")
```

Run it, paste the output into the document, and note which assumption each line is sensitive to. That last part is what separates a design document from a spreadsheet.

### Failure modes are a table, not a paragraph

For each dependency: what fails, the detector, the automatic response, and what the user sees. The user-visible column is the one that gets you approval, because it is the one product managers care about.

| Failure | Detector | Automatic response | User sees |
|---|---|---|---|
| Provider 429 | status code | backoff, retry ×3, then fallback model | slight delay, same answer |
| Provider 5xx | status code | circuit breaker opens 60 s | "temporarily degraded", cached answer if any |
| Provider timeout | 20 s deadline | cancel, serve degraded template | partial answer + retry button |
| Vector store down | health check | keyword-only retrieval | answer with fewer citations |
| Redis down | connection error | skip cache, log | slower, no correctness change |
| Worker backlog | queue depth metric | alert, autoscale, shed low priority | long jobs queue, chat unaffected |

## Code

The document is markdown in the repo, so it is reviewable in a pull request and diffable over time.

```text
design/
├── 001-support-assistant.md     <- the document
├── capacity.py                  <- every number, reproducible
├── cost_model.py                <- sensitivity analysis
└── diagrams/
    └── architecture.txt         <- ASCII, so it diffs
```

```python
# cost_model.py — sensitivity: what happens if the assumptions are wrong?
from capacity import ASSUMPTIONS, daily_cost

def with_multiplier(key: str, factor: float) -> float:
    a = dict(ASSUMPTIONS)
    a[key] = a[key] * factor
    return daily_cost(**{k: a[k] for k in a})

for key in ("prompt_tokens", "output_tokens", "dau"):
    for factor in (0.5, 3.0):
        print(f"{key} x{factor}: ${with_multiplier(key, factor):,.2f}/day")
```

### The one-paragraph summary that opens the document

> We are building a support assistant over 40,000 help-centre documents for 2,000 daily users at 8 questions each. At an assumed 1,800 input and 320 output tokens per request, that is 16,000 requests a day, a peak of 0.74 requests per second and roughly 3.3 concurrent model calls. Daily model cost is approximately $8.60, or $0.54 per thousand requests, before caching; exact-match caching of repeat questions should reduce that by a quarter to a third. We accept up to 4.5 seconds p95 because answers stream. The system degrades to keyword retrieval and a template answer if the vector store or provider fails; chat remains available. We are not building multi-language support or agentic tool use in this phase.

Every clause in that paragraph is a claim with a number or a boundary. That is the standard.

## Real use case

A payments platform's AI team proposes a document Q&A assistant. The first design document is rejected in review — not for the architecture, which was sound, but because the cost section said "roughly affordable" and the failure section said "we will handle errors". The reviewer's note was one line: "I cannot approve a budget I cannot compute."

The rewrite adds the assumptions table, the arithmetic script and the failure matrix. It also adds a non-goals section that removes two features the reviewer was worried about, which shortens the meeting. Approval comes in the second review, and — the useful part — the numbers become the project's operating constraints: when the team later proposes a bigger model, the same document shows the cost line tripling, and the decision takes five minutes instead of a week.

## Common mistakes

- **Mistake:** architecture with no arithmetic — **Instead:** a script that reproduces every number; paste its output.
- **Mistake:** assumptions buried in prose — **Instead:** a table with a confidence column and a sensitivity note.
- **Mistake:** omitting non-goals — **Instead:** list three things you are not building; it prevents scope arguments later.
- **Mistake:** failure handling described as "we will retry" — **Instead:** a table with detector, response and user-visible behaviour.
- **Mistake:** quoting a provider price as a permanent fact — **Instead:** name the price as an input with a date and a note to re-check it.
- **Mistake:** a document that describes the final state with no rollout — **Instead:** phases, feature flags, migration and rollback; reviewers approve transitions, not endpoints.

## Checkpoint

1. Run `capacity.py` for a product of your choice and paste the output into a one-page summary. Label every assumption.
2. Write the failure matrix for that system: six rows minimum, including one user-visible degraded mode.
3. Do the sensitivity pass: which single assumption, if 3× wrong, breaks the design? Write one paragraph on the mitigation.

**Self-check:** if a reviewer cannot recompute your cost line from the document alone, the document is not finished — move the arithmetic into a script and link it.

## Go deeper

- [Cost and Latency](08-cost-latency.md) — the budgets this document must respect
- [Production Case Studies](../23-career/08-production-case-studies.md) — turning this document into an interview story
- [System Design Practice](../23-career/05-system-design-practice.md) — the interview version of this exercise

---
[← Previous: Scaling Trade-offs](09-scaling-tradeoffs.md) · [Stage index](README.md) · [Next stage: Real Projects](../22-real-projects/README.md)
