# 8. Cost and Latency Engineering

> **Stage 21 · System Design** — the two numbers that decide whether an AI product works commercially and feels usable, and the levers that move them.
>
> **You'll be able to:** compute unit economics per request with labelled assumptions, set token budgets that hold, and choose latency techniques by their measured effect rather than by fashion.
>
> **Prerequisites:** [Cost Tracking](../07-llm-apis/08-cost-tracking.md) · [Streaming](../07-llm-apis/03-streaming.md)

## Why this exists

Two numbers decide whether your AI product exists next quarter. **Cost per request** against revenue per request tells you if the business works. **Time to first useful output** tells you if people use it. Both are engineerable — they respond to specific, measurable changes — and both are usually treated as weather.

The reason they are usually neglected is that neither is visible by default. A model call's cost is hidden in token counts you did not log; the latency is a distribution you only see if you measure p95 and not the average. Once you measure them, the levers are concrete: fewer input tokens, less output, a cheaper tier, a cache, streaming, batching.

The mental model: **cost and latency are both integrals over tokens — so every technique is either "send fewer tokens", "send them to a cheaper place", or "show something sooner".**

## The mental model

```text
                       cost per request
                              │
     ┌────────────────┬───────┴────────┬──────────────────┐
     ▼                ▼                ▼                  ▼
 input tokens    output tokens    price per token     non-token cost
     │                │                │                  │
 compress        constrain        route to a        retrieval, tools,
 context,        length,          cheaper tier,     storage, egress,
 cache prefix    schema-first     cache             observability

                       latency to first token
                              │
     ┌────────────────┬───────┴────────┬──────────────────┐
     ▼                ▼                ▼                  ▼
 retrieval time   model TTFT      streaming         speculative UX
 cache, prefilter  smaller tier,  show tokens       show progress,
                   shorter prompt as they come      skeleton, optimistic
```

| Lever | Cost effect | Latency effect | Quality risk |
|---|---|---|---|
| Shorter prompt / compression | large (input dominates) | medium | medium — can drop facts |
| Output cap + schema-first | large | large | low if designed |
| Cheaper tier / routing | large | large | measurable, per route |
| Prefix caching | large on repeat | medium | none |
| Exact / semantic cache | large on repeats | large | medium (staleness) |
| Streaming | none | large perceived | none |
| Batching | medium (throughput) | worse per request | none |

## Core concepts

### Unit economics: one formula, then the arithmetic

Cost per request is a sum of token costs plus a share of non-token costs:

```text
cost = (input_tokens  × P_in  + output_tokens × P_out) / 1_000_000
     + embedding_tokens × P_embed / 1_000_000
     + (tool_calls × P_tool) + (retrieval_queries × P_query)
     + fixed_monthly / requests_per_month
```

Labelled example — **prices and volumes are placeholders; substitute current published values and your own measurements**:

| Assumption | Value |
|---|---|
| Input tokens per request (system + context + question) | 3,000 |
| Output tokens per request | 400 |
| Input price (placeholder) | `$P_in` per 1M tokens |
| Output price (placeholder) | `$P_out` per 1M tokens, with `$P_out ≈ 3 × $P_in` for illustration |
| Retrieval queries per request | 1 |
| Fixed infra per month (placeholder) | `$FIXED` |
| Requests per month | 1,000,000 |
| Price to customer per request (placeholder) | `$R` |

Token cost per request: `(3,000 × $P_in + 400 × $P_out) / 1,000,000`. With `$P_in = 1.0` and `$P_out = 3.0` (illustrative only), that is `(3,000 + 1,200) / 1,000,000 = $0.0042`. Add fixed cost per request `$FIXED / 1,000,000` — if `$FIXED` were `$2,000`, that is `$0.002` — so roughly `$0.0062` all-in. If the customer pays `$R = $0.02` per request, gross margin is about `(0.02 − 0.0062) / 0.02 ≈ 69%`.

Now the useful part: the sensitivity. Output tokens cost three times as much per token as input here, so **capping output from 400 to 250 saves `150 × 3.0 / 1e6 = $0.00045`**, about 7% of total cost. Halving the input prompt saves `1,500 × 1.0 / 1e6 = $0.0015`, about 24%. A 30% cache hit rate removes 30% of both. That ranking — input first, cache second, output third — is typical, and it is why prompt compression and prefix caching are the first things to try.

Do the same sum per tenant (see [Multi-Tenant Design](05-multi-tenant.md)); an average hides the tenants who lose you money.

### Token budgets that actually hold

A token budget is a design constraint, not a wish. For each request: a cap on retrieved context (drop lowest-scoring chunks, never truncate mid-chunk), a cap on conversation history (summarise beyond a window — see [Context Compression](../08-prompt-and-context/04-context-compression.md)), a cap on output (and a schema that makes long output unnecessary), and a hard check before the call that refuses an oversized prompt with a clear error rather than sending it.

Measure the distribution, not the mean: track p50, p95 and max input tokens per intent. The max is what blows your budget in a month.

### Latency: optimise the perceived number

Users do not experience total latency; they experience **time to first useful output**. So:

- **Stream** the answer. Time to first token is a fraction of total time, and a streaming answer feels responsive even when total time is unchanged.
- **Show progress** for multi-step work: which document is being read, which tool is running. A skeleton with a status line beats a spinner.
- **Do retrieval and prompt assembly in parallel** where they are independent, and cache retrieval so the second question in a session is faster.
- **Front-load** what the user can act on: a summary sentence first, detail after.
- **Pre-warm** the expensive path (cache, connection pool) for known high-traffic intents.

Then measure the right percentiles. Report p50 and p95 for time-to-first-token and for completion separately, per route. An average hides the tail, and the tail is where users churn.

### Batching and offline work

Anything that does not need a synchronous answer should be batched or moved offline: embeddings for ingestion, classification of historical tickets, nightly re-scoring, evaluation runs. Batch APIs and larger batch sizes improve throughput and often price, at the cost of latency — which is fine when nobody is waiting. The design rule: **if a human is not waiting for it, it should not be in the request path.**

## Code

A token budget enforcer that runs before the model call, plus a cost meter that records per-request economics. Both are cheap; both are the difference between knowing and guessing.

```python
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

# PLACEHOLDER prices: substitute the provider's current published rates.
P_IN_PER_MTOK = 1.0
P_OUT_PER_MTOK = 3.0
P_EMBED_PER_MTOK = 0.02


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int
    output_tokens: int
    embedding_tokens: int = 0
    tool_calls: int = 0


def cost_usd(u: Usage) -> float:
    return (
        u.input_tokens * P_IN_PER_MTOK / 1e6
        + u.output_tokens * P_OUT_PER_MTOK / 1e6
        + u.embedding_tokens * P_EMBED_PER_MTOK / 1e6
    )


@dataclass(frozen=True)
class TokenBudget:
    max_input: int = 6_000
    max_output: int = 500
    max_cost_usd: float = 0.02

    def check(self, input_tokens: int, planned_output: int) -> None:
        if input_tokens > self.max_input:
            raise ValueError(f"prompt too large: {input_tokens} > {self.max_input}")
        if planned_output > self.max_output:
            raise ValueError("requested output exceeds budget")
        projected = cost_usd(
            Usage(input_tokens=input_tokens, output_tokens=planned_output)
        )
        if projected > self.max_cost_usd:
            raise ValueError(f"projected cost {projected:.4f} over budget")


def fit_chunks(chunks: list[tuple[str, float]], max_tokens: int) -> list[str]:
    """Keep the highest-scoring chunks that fit. Never truncate mid-chunk."""
    kept, used = [], 0
    for text, _score in sorted(chunks, key=lambda c: c[1], reverse=True):
        cost = len(text) // 4 + 1  # rough; use the real tokenizer
        if used + cost > max_tokens:
            break
        kept.append(text)
        used += cost
    return kept
```

A streaming handler with an idle timeout, so a stalled stream does not hang a request. Note that it streams to the client and counts tokens as they arrive.

```python
from __future__ import annotations

import asyncio


async def stream_answer(client, prompt: str, send, idle_timeout_s: float = 15.0) -> int:
    """Stream tokens to `send`; return output token count. Aborts on idle."""
    tokens = 0
    stream = await client.messages.stream(model="<current-model-id>", prompt=prompt)
    try:
        while True:
            try:
                event = await asyncio.wait_for(anext(stream), timeout=idle_timeout_s)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                # A stream that stops producing is a failure, not a slow answer.
                await send({"type": "error", "message": "response stalled"})
                break
            if event.delta_text:
                tokens += 1
                await send({"type": "delta", "text": event.delta_text})
    finally:
        await stream.close()
    await send({"type": "done", "output_tokens": tokens})
    return tokens
```

## Real use case

**A legal-research assistant for a mid-sized firm, priced per seat.** Each query sent a 9,000-token prompt: a long system prompt, six retrieved judgments, and the conversation so far. Answers averaged 900 output tokens because the prompt asked for "a full analysis". Cost per query was roughly triple the team's estimate, and p95 latency was around nine seconds, which lawyers worked around by opening a second tab.

Four changes, measured one at a time. The system prompt was compressed and made a stable prefix, so prefix caching applied. Retrieved judgments were capped by score to fit a 4,000-token budget instead of including six regardless of relevance. The output contract became schema-first — a summary, then key holdings as bullets, then citations — which cut output to about 400 tokens without losing substance. Streaming was enabled so the summary appeared in about a second even though total time was still several seconds.

Cost per query fell by roughly half and time to first useful output dropped from many seconds to about one. The team kept a dashboard of p50/p95 time-to-first-token per intent and cost per query per tenant, which is what let them catch a regression three weeks later when a new prompt version quietly doubled the input tokens.

## Common mistakes

- **Mistake:** measuring average latency — **Instead:** report p50 and p95 for time-to-first-token and completion, per route and per intent.
- **Mistake:** an uncapped output length because the prompt says "be concise" — **Instead:** cap `max_tokens` and use a schema that makes long output unnecessary.
- **Mistake:** including every retrieved chunk regardless of score — **Instead:** fit chunks to a token budget by score, never truncating mid-chunk.
- **Mistake:** quoting a model price as a design fact — **Instead:** keep prices in one config with a comment to substitute the current published rate, and recompute the arithmetic.
- **Mistake:** batching things a user is waiting for — **Instead:** keep the request path synchronous and short; move everything else to a queue or a batch job.

## Checkpoint

1. Compute cost per request and per month for your app using the formula, then compute the effect of a 30% cache hit rate. Show the arithmetic for both.
2. Add a token budget that rejects oversized prompts, and log p50/p95 input tokens per intent for a week. Name the intent with the worst tail.
3. Add streaming with an idle timeout and measure time-to-first-token before and after. Then measure what a 40% output cap does to your quality metric on a golden set.

**Self-check:** The formula is `(input × P_in + output × P_out) / 1e6` plus non-token costs, and the sensitivity usually ranks input tokens first, caching second, output third — so compression and caching are the levers to try before switching models.

## Go deeper

- [Caching Layers](06-caching.md) — the lever with the best cost-to-risk ratio.
- [Model Routing](04-model-routing.md) — matching tier to request.
- [Context Compression](../08-prompt-and-context/04-context-compression.md) — sending fewer input tokens without losing facts.
- [Cost Monitoring](../18-mlops-llmops/06-cost-monitoring.md) — the dashboards these numbers feed.

---
[← Previous: Fault Tolerance](07-fault-tolerance.md) · [Stage index](README.md) · [Next: Scaling Trade-offs](09-scaling-tradeoffs.md)
