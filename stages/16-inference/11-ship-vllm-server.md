# 11. Ship: A vLLM Server

> **Stage 16 · Inference** — serve an open-weight model behind an OpenAI-compatible endpoint and prove its numbers with your own load test.
>
> **You'll be able to:** deploy, load test and report on a real inference server using predictions you made before measuring.
>
> **Prerequisites:** [vLLM and SGLang](08-vllm-sglang.md) · [GPU Memory Budgeting](10-gpu-memory.md)

## Why this exists

Everything in this stage so far is arithmetic and explanation. The skill that gets hired is
the one where you predict a number, measure it, and explain the gap. A candidate who says "it
handles about 40 concurrent users" is guessing. A candidate who says "the KV cache budget is
15.1 GiB, which is 41 concurrent 3k-token conversations, and the load test peaked at 22.4 of
24 GiB with p95 TTFT at 780 ms" is doing engineering.

This project is deliberately small: one model, one card, one endpoint, one load test. The
deliverable is a short report with three measured numbers and your predicted values beside
them.

## The mental model

```text
   your laptop                    GPU host
+---------------+          +---------------------------+
| load_test.py  | -------> | vllm serve (OpenAI /v1)   |
|  N concurrent |          |  weights + KV cache       |
|  streams      | <------- |  continuous batching      |
+---------------+  tokens  +---------------------------+
        |                          |
        |  records TTFT, TPOT      |  nvidia-smi sampled every 0.5s
        v                          v
   p50/p95 report  <---- peak memory, cache capacity in tokens
```

| Step | Output |
|---|---|
| 1. Predict | GiB needed, cache tokens, expected concurrency |
| 2. Serve | a running endpoint on `/v1/chat/completions` |
| 3. Load test | p50 and p95 TTFT, aggregate tokens/sec, peak memory |
| 4. Reconcile | a table of predicted versus measured, with a cause for each gap |

## Core concepts

### Write the prediction down before you run anything

Take an 8B-class open model at int4 with 32 layers, 8 KV heads and head_dim 128. From
[The KV Cache](02-kv-cache.md), the cache is 128 KiB per token. On a 24 GiB card:

```text
weights (int4, g128)      3.96 GiB
activations (reserve)     1.50 GiB
framework overhead        1.50 GiB
util 0.92 of 24 GiB      22.08 GiB
free for KV cache        15.12 GiB  ->  123,863 tokens
concurrency at 3k tokens          41
```

Write those four numbers down. They are the target your measurement is judged against, and
writing them first is what stops you from reverse-engineering a story after the fact.

### The load test must open and close streams concurrently

Sequential requests measure nothing about batching. You need `N` tasks in flight, each reading
its own stream, each recording the timestamp of its first token. That is the only way to see
TTFT under contention, which is what users actually experience.

Two numbers matter, and they are not the same:
- **Aggregate output tokens per second** across all requests — your cost efficiency.
- **p50 and p95 TTFT** — the user experience. p95 is the one that generates complaints.

### Sample memory from outside the process

Do not trust the server's self-report alone. Poll `nvidia-smi` on the host during the run. The
peak is what matters, because a single long prompt can spike activation memory above the steady
state you designed for.

### Sweep concurrency, do not test one point

Run the same workload at concurrency 1, 8 and 32. You should see aggregate throughput rise,
p50 TPOT rise slightly, and p95 TTFT rise sharply as queueing begins. That curve is the
artefact worth showing; a single number tells nobody anything.

## Code

The server, started with the flags that make your budget explicit. Flag names change between
releases — check `vllm serve --help` for your installed version.

```bash
# Serve the model. --max-model-len caps the cache per request; --max-num-seqs caps
# concurrency so one burst cannot exhaust the pool.
vllm serve <hf-repo-id> \
  --host 0.0.0.0 --port 8000 \
  --served-model-name my-served-model \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92

# Sanity check: the model name clients must send, plus the reported cache capacity.
curl -s localhost:8000/v1/models
curl -s localhost:8000/metrics | grep -i -E "cache|kv" | head -20
```

The load test. It runs `concurrency` streams in parallel, records per-request TTFT and TPOT,
samples GPU memory in the background, and prints the report you will paste into your README.

```python
from __future__ import annotations

import asyncio
import statistics
import subprocess
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "my-served-model"
PROMPT = "Explain the difference between prefill and decode in two short paragraphs."


@dataclass
class Sample:
    ttft_ms: float
    tpot_ms: float
    tokens: int


async def one_request(client: AsyncOpenAI, max_tokens: int) -> Sample:
    start = time.perf_counter()
    first: float | None = None
    last = start
    tokens = 0
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            first = first or now
            last = now
            tokens += 1
    if first is None or tokens < 2:
        raise RuntimeError("stream produced too few tokens")
    return Sample(
        ttft_ms=(first - start) * 1000.0,
        tpot_ms=((last - first) * 1000.0) / (tokens - 1),
        tokens=tokens,
    )


def gpu_used_mib() -> float:
    """Total memory used on GPU 0, in MiB. Returns 0 if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", "0"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        return float(out)
    except (subprocess.SubprocessError, IndexError, ValueError):
        return 0.0


async def sample_memory(peak: list[float], stop: asyncio.Event) -> None:
    while not stop.is_set():
        peak[0] = max(peak[0], gpu_used_mib())
        await asyncio.sleep(0.5)


async def run(concurrency: int, max_tokens: int = 200) -> dict[str, float]:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="unused")
    peak = [gpu_used_mib()]
    stop = asyncio.Event()
    monitor = asyncio.create_task(sample_memory(peak, stop))

    start = time.perf_counter()
    samples = await asyncio.gather(
        *[one_request(client, max_tokens) for _ in range(concurrency)],
        return_exceptions=True,
    )
    elapsed = time.perf_counter() - start
    stop.set()
    await monitor

    ok = [s for s in samples if isinstance(s, Sample)]
    if not ok:
        raise RuntimeError(f"all requests failed: {samples[0]!r}")

    ttfts = sorted(s.ttft_ms for s in ok)
    tpot = statistics.median(s.tpot_ms for s in ok)
    total_tokens = sum(s.tokens for s in ok)

    def pct(xs: list[float], q: float) -> float:
        idx = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
        return xs[idx]

    return {
        "concurrency": float(concurrency),
        "ok": float(len(ok)),
        "ttft_p50_ms": pct(ttfts, 0.50),
        "ttft_p95_ms": pct(ttfts, 0.95),
        "tpot_median_ms": tpot,
        "aggregate_tok_s": total_tokens / elapsed,
        "peak_gpu_mib": peak[0],
    }


async def main() -> None:
    for c in (1, 8, 32):
        r = await run(c)
        print(f"c={int(r['concurrency']):2d} ok={int(r['ok']):2d} "
              f"TTFT p50 {r['ttft_p50_ms']:6.0f} ms  p95 {r['ttft_p95_ms']:6.0f} ms  "
              f"TPOT {r['tpot_median_ms']:5.1f} ms  "
              f"{r['aggregate_tok_s']:6.1f} tok/s  "
              f"peak {r['peak_gpu_mib']:6.0f} MiB")


if __name__ == "__main__":
    asyncio.run(main())
```

You should see aggregate tokens per second climb steeply from concurrency 1 to 8, then flatten
as the card saturates, while p95 TTFT grows because later requests wait in the queue. Peak
memory should be flat across concurrency levels — the cache pool is pre-allocated — and slightly
higher than the steady state during the longest prompt.

## Real use case

A four-person team was about to commit to a second GPU for a customer-support assistant. Their
justification was a demo day where the assistant "got slow with everyone trying it at once".
Before spending, they ran this project on the card they already had.

The prediction said 41 concurrent 3k-token conversations. The load test at concurrency 32
showed p95 TTFT of just under a second and aggregate throughput that was still climbing, with
peak memory about 1.5 GiB below the card's capacity. That combination said the card was not
memory-bound; the latency came from queueing behind long prompts, and the fix was scheduling,
not hardware.

Two changes followed. They set `--max-num-seqs` to a value below the memory limit so the
scheduler queued predictably instead of thrashing, and they capped `--max-model-len` at a level
their real conversations never exceeded, which freed cache for more concurrent requests. They
also moved a long system prompt to the front of the message array unchanged, so prefix caching
applied to every request from the same tenant. The second GPU was not purchased, and the report
that showed why — predicted versus measured, at three concurrency levels — became the template
the team used for every later capacity decision.

## Common mistakes

- **Mistake:** measuring with sequential requests — **Instead:** run concurrent streams, or you will never observe batching or queueing.
- **Mistake:** reporting only a mean — **Instead:** report p50 and p95; the tail is the user complaint.
- **Mistake:** trusting the server's own memory number — **Instead:** poll `nvidia-smi` externally and compare both.
- **Mistake:** testing at one concurrency level — **Instead:** sweep at least three so the throughput and latency curves are visible.
- **Mistake:** quoting a throughput figure without the prompt and output length that produced it — **Instead:** publish the workload with the number, since tokens/sec means nothing alone.

## Checkpoint

1. Run the load test at concurrency 1 and compute the implied memory bandwidth from your measured TPOT and the model's weight size. Compare with the card's spec sheet.
2. Extend the script to vary prompt length (500, 2,000, 8,000 tokens) at fixed concurrency, and plot TTFT against prompt length. Report the slope in milliseconds per token.
3. Add a failure path: send a request with `max_tokens` above `--max-model-len` and confirm your client reports a clear error rather than hanging. Then write the one-paragraph capacity report with predicted versus measured numbers.

**Self-check:** `bandwidth_implied = weight_bytes / (tpot_seconds)`. For a 3.96 GiB int4 model at 30 ms per token that is `4.25e9 / 0.030 ≈ 142 GB/s` — far below the card's peak, which tells you decode is dominated by overhead at batch one, not by raw bandwidth.

## Go deeper

- [GPU Memory Budgeting](10-gpu-memory.md) — the prediction side of this project in full.
- [Continuous Batching](03-continuous-batching.md) — read the curve you just plotted as a scheduling story.
- [Cost and Latency](../21-system-design/08-cost-latency.md) — turn these measurements into a product decision.
- [Streaming](../07-llm-apis/03-streaming.md) — what the client does with the tokens your server emits.

---
[← Previous: GPU Memory Budgeting](10-gpu-memory.md) · [Stage index](README.md) · [Next: Stage 17 — AI Backend](../17-ai-backend/README.md)
