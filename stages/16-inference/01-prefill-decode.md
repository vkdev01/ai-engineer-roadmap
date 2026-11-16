# 1. Prefill and Decode

> **Stage 16 · Inference** — the two phases every request goes through, and why they have different bottlenecks.
>
> **You'll be able to:** predict time-to-first-token and tokens-per-second from prompt length, model size and GPU bandwidth.
>
> **Prerequisites:** [Self-Attention](../06-llm-fundamentals/04-self-attention.md) · [Context Windows](../06-llm-fundamentals/03-context-windows.md)

## Why this exists

A user types a question and waits. Two things happen before they see a finished answer:
the model reads the question, then it writes the answer one token at a time. Those are not
the same operation and they do not cost the same thing.

Reading is parallel. The model has the whole prompt, so it can push all of those tokens
through the network at once — a big matrix multiplication that keeps the GPU busy. Writing
is sequential. Token 50 cannot be computed until token 49 exists, so the GPU does enormous
work to produce one number, then waits to do it again.

The mental model: **prefill is compute-bound, decode is memory-bandwidth-bound.** Almost
every serving decision follows from that one sentence. Long prompts hurt the first phase;
long answers hurt the second; batching helps the second and barely helps the first.

## The mental model

```text
 prompt: "Summarise this contract ..."        answer: "The contract ..."
 |----------------------------------|         |--|--|--|--|--|--|--|
            PREFILL                                  DECODE
       all tokens in parallel                  one token per step
       big matmuls, GPU busy                   reads all weights per step
       cost ~ 2 x params x prompt_tokens       cost ~ 2 x params per token
       sets TTFT                               sets TPOT -> tokens/sec
```

| | Prefill | Decode |
|---|---|---|
| Tokens per step | all prompt tokens | 1 |
| Bottleneck | FLOPs | memory bandwidth |
| Scales with | prompt length | output length |
| User-visible number | TTFT | tokens/sec |
| Helped by batching | little | a lot |

## Core concepts

### Time-to-first-token is prefill plus queue

TTFT is the wall-clock delay before the first streamed token. It has three parts: queueing
while the scheduler picks you up, the prefill pass over your whole prompt, and one decode
step to emit the first token. If your product feels slow to start, look at prompt length and
queue depth, not at model quality.

Prefill cost is roughly `2 x params x prompt_tokens` floating point operations, because each
weight participates in one multiply and one add per token. Take an 8-billion-parameter model
and a 1000-token prompt: `2 x 8e9 x 1000 = 1.6e13` FLOPs, i.e. 16 TFLOP. On a card that
sustains 150 TFLOP/s of useful fp16 matmul throughput (illustrative), that is
`16 / 150 ≈ 0.107 s`, so about 107 ms of pure prefill.

### Tokens per second is bandwidth divided by weight bytes

In decode, every step must read the entire weight matrix to produce a single token, because
each weight is used exactly once per token. So the ceiling on tokens per second at batch size
one is:

```text
tokens/sec (batch 1)  <=  memory bandwidth  /  bytes of weights
```

An 8B model in fp16 is `8e9 x 2 = 16 GB` of weights. On a card with about 2 TB/s of HBM
bandwidth (check your GPU's spec sheet for the real figure):

```text
2000 GB/s / 16 GB = 125 tokens/sec
```

That is the hard ceiling for one stream. Measured throughput is lower, because of kernel
launch overhead, attention work and the KV cache reads — but you will never beat it.

### Batching breaks the bandwidth wall

The same 16 GB of weights can serve 32 sequences in one step, because they are read once and
applied to 32 query vectors. Decode becomes compute-bound again somewhere around the point
where the arithmetic intensity of the batch exceeds the card's FLOP-to-byte ratio. This is
why serving engines batch aggressively and why your tokens-per-second *per user* can fall
while total throughput rises. Stage file [Continuous Batching](03-continuous-batching.md)
takes this further.

### The decode step is not free even when it is short

`TPOT` (time per output token) is what users feel as "typing speed". A useful product rule:
keep TTFT under about a second and TPOT under about 50 ms for chat. Given the ceiling above,
a 70B fp16 model at 140 GB of weights needs at least 70 ms per token on a 2 TB/s card before
any overhead — so on one GPU, a 70B fp16 model simply cannot feel snappy at batch one. That
is the argument for quantization, which you meet in [Quantization](06-quantization.md).

## Code

This script measures both phases from a streaming API. It records the arrival time of the
first token and of the last, so TTFT and TPOT come from real timestamps rather than from
the server's self-report. Point `base_url` at any OpenAI-compatible endpoint.

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class Measurement:
    ttft_ms: float
    tpot_ms: float
    output_tokens: int

    @property
    def tokens_per_sec(self) -> float:
        # Decode-only rate: excludes the prefill wait from the denominator.
        if self.tpot_ms <= 0:
            return 0.0
        return 1000.0 / self.tpot_ms


async def measure(prompt: str, max_tokens: int = 200) -> Measurement:
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    start = time.perf_counter()
    first: float | None = None
    chunks = 0

    # The model id must match what your server exposes; check its /v1/models route.
    stream = await client.chat.completions.create(
        model="my-served-model",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    last = start
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if first is None:
                first = now
            chunks += 1
            last = now

    if first is None or chunks == 0:
        raise RuntimeError("no tokens arrived")

    # chunks counts streamed deltas, a good proxy for tokens; the final usage
    # block gives the exact completion_tokens if your server sends one.
    decode_span = last - first
    tpot = (decode_span * 1000.0) / max(chunks - 1, 1)
    return Measurement(
        ttft_ms=(first - start) * 1000.0,
        tpot_ms=tpot,
        output_tokens=chunks,
    )


async def main() -> None:
    # Same output length, very different prompt length: TTFT moves, TPOT barely does.
    short = await measure("Name three uses of a queue.", max_tokens=150)
    long_prompt = "Here is a specification. " * 400 + " Summarise it in 150 tokens."
    long = await measure(long_prompt, max_tokens=150)
    for label, m in (("short prompt", short), ("long prompt", long)):
        print(f"{label}: TTFT {m.ttft_ms:.0f} ms, TPOT {m.tpot_ms:.1f} ms, "
              f"{m.tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
```

You should see the long-prompt run report a much larger TTFT with a similar TPOT. That
split is the whole point of this file: the two numbers move independently, so optimise them
with different levers.

## Real use case

A document-review tool at a mid-sized legal firm let a user select a 40-page contract and ask
questions about it. The team had tuned for "tokens per second" and were proud of the number,
but users complained the tool was unusable. The measured split was the diagnosis: with a
30,000-token prompt, prefill alone was several seconds, so TTFT dominated, while decode was
comfortably fast. They had optimised the phase that was not the problem.

The fix had three parts. They cached the prefilled prefix for the document using the
engine's prefix cache (see [vLLM and SGLang](08-vllm-sglang.md)), so the second question
about the same contract skipped most of prefill. They moved retrieval earlier so the prompt
carried 6 relevant pages instead of 40. And they streamed a short "reading the document"
status so the wait was honest. Users stopped complaining about speed without any change to
the model.

## Common mistakes

- **Mistake:** treating "tokens/sec" as one number — **Instead:** report TTFT and TPOT separately; they have different causes and different fixes.
- **Mistake:** benchmarking with a 20-token prompt and shipping with a 20,000-token one — **Instead:** measure at your real p95 prompt length, not the demo length.
- **Mistake:** assuming a faster GPU with more FLOPs will speed up decode — **Instead:** check memory bandwidth first; decode at batch one is a bandwidth problem.
- **Mistake:** counting prefill time inside your tokens-per-second figure — **Instead:** compute decode rate over the decode span only, or the number flatters you.
- **Mistake:** comparing your server's throughput to a vendor blog post — **Instead:** compare against the bandwidth ceiling you can compute yourself.

## Checkpoint

1. An 8B fp16 model runs on a card with 1.6 TB/s of memory bandwidth. What is the theoretical maximum tokens per second at batch size one?
2. You need TTFT under 800 ms on a card sustaining 200 TFLOP/s with an 8B model. Roughly what is the largest prompt (in tokens) that fits, ignoring queue time?
3. Write a small script that sweeps prompt lengths from 500 to 8000 tokens against your server, then plot TTFT against prompt length. Confirm the relationship is roughly linear and estimate the per-token prefill cost from the slope.

**Self-check:** 1600 GB/s divided by 16 GB of fp16 weights is 100 tokens per second — a ceiling, not a promise.

## Go deeper

- [The KV Cache](02-kv-cache.md) — the other half of the memory story, and why decode slows as context grows.
- [Continuous Batching](03-continuous-batching.md) — how serving engines keep the card busy across many users.
- [GPU Memory Budgeting](10-gpu-memory.md) — turn these per-token costs into a card-size decision.
- [Streaming](../07-llm-apis/03-streaming.md) — the client side of the numbers you just measured.

---
[← Previous: Stage 15 — Evals](../15-evals/README.md) · [Stage index](README.md) · [Next: The KV Cache](02-kv-cache.md)
