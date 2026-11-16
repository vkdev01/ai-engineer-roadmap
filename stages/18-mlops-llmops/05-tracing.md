# 5. Tracing Design

> **Stage 18 · MLOps and LLMOps** — the design rules that decide whether your traces answer questions or just fill a database.
>
> **You'll be able to:** choose span boundaries, pick the attributes worth capturing, and sample traffic so that a slow request is never the one you dropped.
>
> **Prerequisites:** [LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) · [Stage 17 — AI Backend](../17-ai-backend/README.md)

## Why this exists

Instrumentation is easy to add and hard to add well. Two teams can both "have tracing" and get very different value. Team A's traces are one flat span per HTTP request: they can see the request took 2.4 seconds but not which of five model calls ate it. Team B has a hundred spans per request because every helper function got decorated: the tree is unreadable and the storage bill is absurd. Neither can answer "why was this answer expensive".

Good tracing is a design decision made once, before the code, and it is mostly about **where you draw boundaries** and **what you put on the nodes**. This file is those decisions written down.

The mental model: **a span exists if and only if you would ever want its latency, its cost, or its failure counted separately.**

## The mental model

```text
one user request
`-- trace: POST /chat                       <- entry point, has tenant + release
    |-- span: guardrail.check                <- separate: cheap, but can reject early
    |-- span: retrieve                       <- separate: latency varies, k is tunable
    |   `-- span: vector.search              <- separate only if you may swap the index
    |-- span: llm.plan          (generation) <- model call: tokens + cost
    |-- span: tool.order_lookup               <- separate: external, can time out
    `-- span: llm.answer        (generation) <- model call: tokens + cost

rule of thumb: 3-8 spans per trace. 1 is too few. 40 is a smell.
```

## Core concepts

### One trace per request, not per function call

The trace is the unit a user experiences. If your service handles one user message with three internal model calls, that is one trace with three generation spans. If a background job processes a batch of 50 documents, that is one trace per document, plus optionally a parent trace for the batch — because you will want to find the one document that failed.

The test: can you point at a single user-visible outcome and say "this trace"? If a trace spans two users' requests, you cannot answer questions about either.

### Span boundaries that survive refactoring

Put spans around **steps whose cost or failure you would want to attribute**, not around code structure. Retrieval, each model call, each external tool, and any guardrail that can short-circuit. Do not span pure functions, serialisation, or a retry loop's individual attempts — put the attempt count on the parent span instead.

Naming matters more than people expect. Use `noun.verb` (`retrieve`, `llm.answer`, `tool.order_lookup`) so traces group in dashboards. Inconsistent names (`retrieveDocs`, `retrieval_step`, `Retrieve`) make per-step latency charts useless.

### Attributes worth capturing

Every span: name, start time, duration, status (`ok`/`error`), and a `error.type` when it fails. Generations additionally: model id, input and output token counts, the rendered prompt (or its hash), and cost. Tools additionally: a redacted argument summary and whether the call was retried.

On the root trace only: `tenant_id`, `user_id` (hashed), `feature`, `prompt_version`, `release_sha`, `env`. These are the join keys for every dashboard you will build. Decide them once, in a shared constant module, and refuse PRs that invent new spellings.

What not to capture: full customer documents, unredacted PII, raw auth headers, and entire retrieved corpora. Capture ids and counts; fetch the content from the system of record when you actually need it.

### Sampling that does not lie

Naive random sampling at 5% will, on a good day, drop 95% of your errors. Three rules fix most of it:

- **Always keep errors.** If the SDK supports it, sample on the trace's outcome, not its start (tail sampling).
- **Always keep negative feedback.** When a thumbs-down arrives, you may not be able to retroactively keep the trace, so keep a short buffer of recent traces per user, or capture 100% for users on paid plans.
- **Sample the boring successes hard.** A 2% sample of healthy traffic is plenty for latency percentiles; you need volume for p99, so compute percentiles from the sample and say so.

Write the policy down. "Sampled at 2%, errors 100%" belongs in a README, because otherwise the next person reads a p95 from a 2% sample as if it were the whole population.

### Cardinality is the enemy of a good attribute

Do not put unbounded strings in attributes you intend to filter or chart on: user ids, search queries, error messages with ids in them. They explode the number of distinct values and make grouping slow and expensive. Bound them: `feature` from an enum, `model_id` from a known list, `error.type` from a small set of categories. Put the unbounded detail in the span's body, where it is read, not indexed.

## Code

A small tracing helper that enforces the rules above rather than trusting every developer to remember them. It centralises the attribute keys, times the block, records exceptions with a bounded error type, and honours the sampling policy.

```python
from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from langfuse import Langfuse

lf = Langfuse()
SUCCESS_SAMPLE_RATE = 0.02  # keep 2% of healthy traces; errors are always kept


@dataclass
class TraceContext:
    trace_id: str
    tenant_id: str
    feature: str
    prompt_version: str
    release_sha: str
    _spans: list[dict[str, object]] = field(default_factory=list)

    def add(self, record: dict[str, object]) -> None:
        self._spans.append(record)


def start_trace(
    *, tenant_id: str, feature: str, prompt_version: str, release_sha: str
) -> TraceContext:
    trace = lf.trace(
        name=f"{feature}.request",
        metadata={
            "tenant_id": tenant_id,
            "feature": feature,
            "prompt_version": prompt_version,
            "release_sha": release_sha,
            "env": "prod",
            "sample_rate": SUCCESS_SAMPLE_RATE,
        },
    )
    return TraceContext(trace.id, tenant_id, feature, prompt_version, release_sha)


@contextlib.contextmanager
def span(ctx: TraceContext, name: str, **attrs: object) -> Iterator[dict[str, object]]:
    """Timed, error-aware span. Bounded attributes only."""
    started = time.perf_counter()
    record: dict[str, object] = {"name": name, "status": "ok", **attrs}
    try:
        yield record
    except Exception as exc:
        # Bound the error type so it is safe to group on.
        record["status"] = "error"
        record["error.type"] = type(exc).__name__
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        ctx.add(record)
        lf.span(trace_id=ctx.trace_id, name=name, metadata=record)
```

Using it: two spans and a generation, with the error path kept automatically. Note the explicit `error=True` marker so tail sampling can decide to keep the trace.

```python
from __future__ import annotations

from app.tracing import span, start_trace
from app.clients import embed, search, chat


def answer(question: str, tenant_id: str) -> dict[str, object]:
    ctx = start_trace(
        tenant_id=tenant_id, feature="support", prompt_version="answer.v3", release_sha="9c2e"
    )
    with span(ctx, "retrieve", k=5) as rec:
        vector = embed(question)
        chunks = search(vector, k=5)
        rec["chunk_count"] = len(chunks)

    with span(ctx, "llm.answer", model="gpt-4.1-mini") as rec:
        resp = chat(question=question, context=chunks, prompt_version="answer.v3")
        rec["tokens_in"] = resp.usage.input_tokens
        rec["tokens_out"] = resp.usage.output_tokens
        rec["cost_usd"] = resp.cost_usd  # computed from a price table, not guessed

    return {"answer": resp.text, "trace_id": ctx.trace_id}
```

## Real use case

A voice-of-customer tool summarises call transcripts. Users complain that some summaries take eight seconds. The team's first instrumentation was one span per request, which confirmed the eight seconds and nothing else. They redesigned to four spans: `transcribe.fetch`, `chunk.summarise` (per chunk), `llm.merge`, `store.write`.

The redesign immediately localised the problem: p95 on `chunk.summarise` was fine, but traces with twelve or more chunks were serialised, so total time scaled linearly with call length. The fix was batching chunks into groups of four, which cut the worst case roughly in half without changing the model.

The design choice that made it findable was spanning per chunk group rather than per HTTP request, plus recording `chunk_count` as a span attribute so they could filter "traces where chunk_count > 10" and compare. The earlier flat tracing could only tell them the app was slow.

## Common mistakes

**Mistake:** one span per HTTP handler and nothing inside — **Instead:** span the retrieval, model and tool steps; that is where the variance and cost live.

**Mistake:** decorating every function for "coverage" — **Instead:** 3-8 spans per trace; if a span never appears in a chart or a diagnosis, delete it.

**Mistake:** random head sampling with no error exception — **Instead:** always retain errors and negative feedback, and state the sample rate in the trace metadata.

**Mistake:** unbounded values like user queries or raw error strings as filterable attributes — **Instead:** bounded enums for grouping, free text in the body only.

**Mistake:** different attribute key spellings across services — **Instead:** one shared constants module imported everywhere, and a review rule against inline literals.

**Mistake:** logging full prompts containing customer data — **Instead:** log the prompt hash and a redacted sample; see [PII Redaction](../20-ai-security/04-pii-redaction.md).

## Checkpoint

1. Take your Stage 17 endpoint and give it three to five spans using the helper above. Trigger one success and one forced failure, and confirm the failure's span has `status=error` and an `error.type`.
2. Add a `chunk_count`-style attribute that predicts latency, then use it to filter traces and produce two p95 numbers: small inputs versus large inputs.
3. Write a one-page sampling policy for your app: which traces are always kept, which are sampled at what rate, and how percentiles are computed from a sample. Implement the "always keep errors" rule in code.

**Self-check:** exercise 1 is done when the trace view shows a tree with three to five named spans, the failing one marked error, and the root carrying `tenant_id`, `feature`, `prompt_version` and `release_sha`.

## Go deeper

- [Cost Monitoring and Budgets](06-cost-monitoring.md) — turning per-span token counts into per-feature spend.
- [LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) — the platform concepts these rules sit on.
- [Ship: A Trace Dashboard](08-ship-trace-dashboard.md) — consuming traces designed this way.
- [Agent Evaluations](../15-evals/06-agent-evals.md) — why per-tool spans matter more for agents than for single calls.

---

[← Previous: DVC for Data and Models](04-dvc.md) · [Stage index](README.md) · [Next: Cost Monitoring and Budgets](06-cost-monitoring.md)
