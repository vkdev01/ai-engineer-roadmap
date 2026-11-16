# 7. Inference Optimisation

> **Stage 23 · Career** — the interview answers about serving models, with the arithmetic an interviewer is actually listening for.
>
> **You'll be able to:** answer KV-cache, batching, quantization and latency questions with numbers rather than definitions, and diagnose a slow endpoint out loud.
>
> **Prerequisites:** [KV Cache](../16-inference/02-kv-cache.md) · [Continuous Batching](../16-inference/03-continuous-batching.md) · [GPU Memory](../16-inference/10-gpu-memory.md)

## Why this exists

Most candidates can define a KV cache. Very few can compute its size, or explain why doubling batch size does not halve throughput. The interviewer's question is rarely "do you know the term" — it is "can you reason about this under a memory budget".

This file is a rehearsal sheet: the eight questions that come up repeatedly, the answer that lands, and the arithmetic to have memorised.

The mental model: **inference is a memory-bandwidth problem wearing a compute problem's clothes.**

## The mental model

```text
   prefill (compute-bound)          decode (memory-bound)
   ─────────────────────            ─────────────────────
   all prompt tokens in one pass    one token at a time,
   builds the KV cache              re-reading weights each step

   TTFT  ← queue + prefill          tokens/sec ← bandwidth / bytes read per step

   two numbers, two different fixes:
   slow TTFT      → shorter prompts, more replicas, prefix caching
   slow tok/sec   → smaller model, quantization, batching, speculative decoding
```

## Core concepts

### Question 1: "How big is the KV cache?"

The formula, and it is worth memorising:

$$
\text{KV bytes} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes per element} \times \text{tokens}
$$

Worked example — a 70B-class model with 80 layers, 8 KV heads (grouped-query attention), head dim 128, fp16 (2 bytes), 8k context:

```python
layers, kv_heads, head_dim, dtype_bytes, tokens = 80, 8, 128, 2, 8192
per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
print(per_token)                      # 327,680 bytes  ≈ 320 KiB per token
print(per_token * tokens / 2**30)     # ≈ 2.5 GiB for one 8k sequence
```

Say the sentence that follows: **"so with 80 GB of weights I have room for roughly a dozen concurrent 8k sequences, which is why concurrency collapses when context grows."** That is the answer they are looking for.

### Question 2: "Why does batching help, and why not linearly?"

Decode reads the whole weight matrix for every token regardless of batch size — it is bandwidth-bound, so the marginal cost of another sequence is small until you run out of memory or saturate compute. Beyond that, throughput gains flatten and per-request latency rises. Continuous batching helps because it keeps the batch full instead of waiting for the longest sequence.

### Question 3: "What does quantization cost you?"

Fewer bytes per weight means fewer bytes to read per token, so decode speeds up roughly proportionally to the reduction. int8 is usually near-lossless for quality-sensitive work; 4-bit helps most on memory-bound decode and can be acceptable with good methods, but validate on your own evals. The interview-safe answer names the trade: memory and speed up, quality risk small but task-dependent, and always measured, never assumed.

### Question 4: "How would you cut p95 latency by half?"

Never answer with one lever. Answer with a diagnosis first: split TTFT from decode time from queueing. Then name the lever per cause — prompt caching and shorter prompts for TTFT; a smaller or quantized model, speculative decoding for decode; more replicas or better scheduling for queueing.

### Question 5: "Why is my endpoint slow at 3pm?"

Look at queue depth before model speed. A rising queue with flat per-token latency means you are underscaled, not slow; a flat queue with rising latency means the provider or the GPU is the problem.

### Question 6: "How do you choose between a hosted API and self-hosting?"

Volume, control and privacy. Below a certain volume, hosted is cheaper and vastly less work. Self-hosting wins on predictable high volume, data residency, and fine-grained control of the model version — and costs you GPU operations, on-call and utilisation risk.

### Question 7: "What is speculative decoding?"

A small draft model proposes k tokens; the large model verifies them in one forward pass; accepted tokens are free. Throughput improves when acceptance is high, which depends on how predictable the text is. It changes nothing about output distribution in the ideal case, and adds complexity.

### Question 8: "Your GPU OOMs at 32k context. What now?"

Budget it out loud: weights + KV cache + activations + fragmentation overhead. Then the levers: fewer concurrent sequences, smaller KV dtype, grouped-query attention, paged attention, shorter context, or a smaller model. Show the arithmetic for at least one of them.

## Code

The two scripts worth having in your head, because they turn a vague answer into a checkable one.

```python
def kv_cache_gib(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int,
                 tokens: int, batch: int = 1) -> float:
    per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return per_token * tokens * batch / 2**30


def max_concurrency(gpu_gib: float, weights_gib: float, overhead_gib: float,
                    layers: int, kv_heads: int, head_dim: int, dtype_bytes: int,
                    context: int) -> int:
    free = gpu_gib - weights_gib - overhead_gib
    per_seq = kv_cache_gib(layers, kv_heads, head_dim, dtype_bytes, context)
    return max(0, int(free // per_seq))


print(max_concurrency(80, 40, 8, 80, 8, 128, 2, 8192))    # concurrency at 8k
print(max_concurrency(80, 40, 8, 80, 8, 128, 2, 32768))   # same GPU, 32k context
```

Run it once and you will never again be surprised by a context-length change collapsing throughput. That single experience is what makes the interview answer credible.

TypeScript has nothing to add here except the client-side lesson: measure what the user experiences, and separate it into the two phases.

```ts
export type Latency = { ttftMs: number; totalMs: number; tokens: number };

export function diagnose(l: Latency): string {
  const decodeMs = l.totalMs - l.ttftMs;
  const perToken = decodeMs / Math.max(l.tokens - 1, 1);
  if (l.ttftMs > 1500) return `prefill/queue bound (ttft ${l.ttftMs}ms) — shorten prompt or scale out`;
  if (perToken > 60) return `decode bound (${perToken.toFixed(0)}ms/token) — smaller/quantized model`;
  return "healthy";
}
```

## Real use case

A candidate is asked to explain why their RAG service handles twenty concurrent users but falls to four when documents get longer. They sketch the budget: 40 GB of weights, 8 GB of overhead, 32 GB left; at 2k context the KV cache allows many sequences, at 32k it allows a handful — and the service holds one document per request, so context grew with document size.

Then they name the fix they actually shipped: cap retrieved context, and page long documents through a summarisation step rather than one huge prompt. That answer — arithmetic, then diagnosis, then the shipped fix — is what turns a "no" into an offer.

## Common mistakes

- **Mistake:** defining terms instead of computing numbers — **Instead:** always answer with one calculation, even rough.
- **Mistake:** blaming the model for a queueing problem — **Instead:** check queue depth and TTFT before touching the model.
- **Mistake:** promising linear batching gains — **Instead:** say where it flattens and why (memory and compute limits).
- **Mistake:** claiming quantization is free — **Instead:** say what you would measure, and on which eval set.
- **Mistake:** answering "use vLLM" to every latency question — **Instead:** name the bottleneck first; the tool follows the diagnosis.
- **Mistake:** no memory for the KV formula — **Instead:** memorise it; it is the single most reused piece of arithmetic in this field.

## Checkpoint

1. Compute KV cache size per token for two models you know, and concurrency at 4k and 32k context on a 24 GB and an 80 GB GPU.
2. Write the sixty-second answer to "cut p95 latency in half" for a system you have built, with a diagnosis step and three levers.
3. Explain speculative decoding to a non-specialist in four sentences, then to an interviewer in two with one number.

**Self-check:** if your answer to a latency question does not start with "which part of the latency", you are guessing.

## Go deeper

- [KV Cache](../16-inference/02-kv-cache.md) — the mechanism behind the formula
- [GPU Memory](../16-inference/10-gpu-memory.md) — the full budget
- [Production Case Studies](08-production-case-studies.md) — turning this into a story

---
[← Previous: RAG and Agent Patterns](06-rag-agent-patterns.md) · [Stage index](README.md) · [Next: Production Case Studies](08-production-case-studies.md)
