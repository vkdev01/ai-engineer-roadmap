# 3. Continuous Batching

> **Stage 16 · Inference** — how a serving engine keeps the GPU busy without making every user wait for the slowest one.
>
> **You'll be able to:** explain throughput versus latency in batching terms and compute the waste in a static batch.
>
> **Prerequisites:** [Prefill and Decode](01-prefill-decode.md) · [The KV Cache](02-kv-cache.md)

## Why this exists

Decode is bandwidth-bound, so the same weight read can serve many sequences at once. Batching
is free throughput — up to the point where the GPU becomes compute-bound or memory runs out.

The naive way to batch is to collect a group of requests, run them together until all finish,
then start the next group. That is **static batching**, and it wastes the card badly. If one
sequence in the group produces 400 tokens and another produces 20, the short one holds its
slot idle for 380 steps. Every user in the batch waits for the longest answer, and new
arrivals wait for the whole group.

**Continuous batching** (also called iteration-level scheduling) fixes both problems: the
scheduler makes a decision every decode step, admitting new requests as slots free up and
retiring finished ones immediately. The mental model: **a batch is not a group of requests,
it is a group of one-step jobs.**

## The mental model

```text
static batching                  continuous batching
step: 1 2 3 ... 20 ... 400       step: 1  2  3  4  5  6  7
A ####################          A ####
B ####                          B ####
C ############                  C ##########        <- D admitted at step 4
D waits ........................  D      ######
                                E            ####  <- E admitted at step 8
GPU idle slots: many            GPU idle slots: few
```

| | Static batching | Continuous batching |
|---|---|---|
| Scheduling unit | request | one decode step |
| Slot freed when | all requests finish | that request finishes |
| Tail latency | worst in batch | own length only |
| New request waits | for whole batch | for next step |
| Throughput | low at mixed lengths | high |

## Core concepts

### Padding waste is arithmetic you can do

Suppose a static batch of four requests generates 20, 40, 100 and 200 tokens. The batch runs
for 200 steps because the longest sequence dictates it. Slot-steps consumed: `4 x 200 = 800`.
Useful slot-steps: `20 + 40 + 100 + 200 = 360`. Efficiency is `360 / 800 = 45 percent`, so
more than half the card's work is thrown away.

With continuous batching, a finished sequence's slot is handed to a waiting request on the
next step. In the ideal case where there is always a queued request to fill a slot, the
wasted slot-steps approach zero and you serve `800 / 360 ≈ 2.2 times` more work for the same
GPU time. That factor is illustrative and depends entirely on your arrival pattern — if
traffic is bursty and the queue empties, continuous batching has nothing to backfill with.

### Prefill and decode compete for the same card

A new request needs a prefill pass over its prompt, which is compute-heavy and stalls the
decode steps of everyone else. Engines make a policy choice here. Some chunk prefill into
pieces and interleave them with decode steps so TTFT for one user does not become jitter for
everyone else. Others prioritise prefill and accept a small decode hiccup. When you tune a
server, this is the knob behind phrases like "chunked prefill" — see
[vLLM and SGLang](08-vllm-sglang.md).

### Concurrency is capped by KV cache, not by willingness

You can only add a sequence to the batch if there is cache memory for it. So maximum batch
size is a memory calculation: free bytes divided by cache bytes per token at the sequence's
current length. Raising `max_num_seqs` in a config does nothing if the cache is full; the
scheduler will simply queue. This is why [PagedAttention](04-pagedattention.md) and
continuous batching are always discussed together — one is the policy, the other is the
allocator that makes the policy pay off.

### Throughput and per-user latency trade against each other

A bigger batch raises total tokens per second and usually raises per-user TPOT, because each
step now does more work before it returns. Two levers move in opposite directions:

- **More concurrency** — better cost per token, worse tail latency.
- **Fewer, larger prefill chunks** — better throughput on long prompts, worse TTFT jitter.

The right setting is a product decision. Interactive chat wants TPOT under about 50 ms;
batch summarisation wants tokens per rupee and does not care about 300 ms per token.

## Code

This simulator is small enough to run in a second and shows the effect concretely. It models
one decode step per unit of time and compares static against continuous scheduling on the same
arrival pattern.

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    id: str
    tokens: int          # how many output tokens this request will generate
    arrived: int         # step index at which it is queued
    done: int | None = None


def simulate_static(reqs: list[Request], batch_size: int) -> int:
    """Run fixed groups to completion; no refill until the group is empty."""
    step, slot_steps, i = 0, 0, 0
    while i < len(reqs):
        group = reqs[i:i + batch_size]
        longest = max(r.tokens for r in group)
        slot_steps += len(group) * longest     # short requests idle the whole time
        for r in group:
            r.done = step + r.tokens
        step += longest
        i += batch_size
    return slot_steps


def simulate_continuous(reqs: list[Request], batch_size: int) -> int:
    """Refill free slots every step; a slot is only occupied while generating."""
    running: list[Request] = []
    queued = sorted(reqs, key=lambda r: r.arrived)
    remaining = {r.id: r.tokens for r in reqs}
    slot_steps, step = 0, 0
    while queued or running:
        while len(running) < batch_size and queued and queued[0].arrived <= step:
            running.append(queued.pop(0))
        if not running:
            step += 1
            continue
        for r in list(running):
            remaining[r.id] -= 1
            slot_steps += 1
            if remaining[r.id] == 0:
                r.done = step + 1
                running.remove(r)
        step += 1
    return slot_steps


if __name__ == "__main__":
    lengths = [20, 40, 100, 200, 60, 80, 30, 120]
    reqs = [Request(f"r{i}", t, arrived=i) for i, t in enumerate(lengths)]
    useful = sum(lengths)
    static = simulate_static([Request(**vars(r)) for r in reqs], batch_size=4)
    cont = simulate_continuous([Request(**vars(r)) for r in reqs], batch_size=4)
    print(f"useful slot-steps {useful}")
    print(f"static     {static:5d}  efficiency {useful/static:.0%}")
    print(f"continuous {cont:5d}  efficiency {useful/cont:.0%}")
```

You should see static efficiency at about 51 percent overall (45 percent for the first group
of four alone) and continuous efficiency at 100 percent, because no slot ever idles while
work is queued.

## Real use case

A team ran an internal coding assistant on a single GPU behind a plain request queue with a
fixed batch of eight. It worked in testing and collapsed in the morning rush. The cause was
exactly the padding waste above: a one-line autocomplete answer of 8 tokens sat in a batch
behind a 700-token explanation, holding a slot idle and delaying every other request in that
group. Users learned to hit the button twice because the first press "did nothing".

Switching to an engine with continuous batching (see [vLLM and SGLang](08-vllm-sglang.md))
changed two things at once. Finished short answers freed their slots within one step, so
p95 latency stopped being dictated by the longest request in a group. And short requests were
admitted mid-flight, which mattered because autocomplete traffic is dominated by them. The
team also split long-form "explain this file" requests onto a separate queue with a lower
priority, so the cheap interactive path never queued behind expensive ones.

## Common mistakes

- **Mistake:** setting a large `max_num_seqs` and assuming you get that concurrency — **Instead:** check free KV cache first; the scheduler is memory-limited.
- **Mistake:** mixing interactive and batch workloads on one queue — **Instead:** separate queues or priority levels, so long jobs cannot starve chat.
- **Mistake:** measuring throughput with a single request — **Instead:** load test with realistic concurrency, or you will never see batching effects at all.
- **Mistake:** assuming continuous batching reduces TTFT for a single isolated user — **Instead:** it mainly raises throughput and fixes tail latency; single-user TTFT is still prefill.
- **Mistake:** ignoring prefill interference — **Instead:** enable chunked prefill when long prompts cause jitter in decode.

## Checkpoint

1. A static batch of six requests generates 10, 15, 15, 20, 20, 400 tokens. What is the slot-step efficiency?
2. Change `batch_size` to 2 and 8 in the simulator and describe what happens to total GPU time and to the finish time of the last request.
3. Extend the simulator with a second request class that has a priority weight, and implement "high priority is admitted first when a slot frees". Report what happens to the tail latency of low-priority requests.

**Self-check:** the batch runs for 400 steps with 6 slots, so 2,400 slot-steps against 480 useful ones — 20 percent efficiency, and five-sixths of the card's work is wasted.

## Go deeper

- [PagedAttention](04-pagedattention.md) — the allocator that makes large dynamic batches affordable.
- [Speculative Decoding](05-speculative-decoding.md) — a different way to get more tokens per step.
- [Scaling Trade-offs](../21-system-design/09-scaling-tradeoffs.md) — the same throughput-versus-latency reasoning at system level.
- [Ship: A vLLM Server](11-ship-vllm-server.md) — measure these effects on your own traffic.

---
[← Previous: The KV Cache](02-kv-cache.md) · [Stage index](README.md) · [Next: PagedAttention](04-pagedattention.md)
