# 8. vLLM and SGLang

> **Stage 16 · Inference** — the two open serving engines you will actually deploy, and how to choose between them.
>
> **You'll be able to:** stand up either engine behind an OpenAI-compatible endpoint and tune its main memory and scheduling knobs.
>
> **Prerequisites:** [PagedAttention](04-pagedattention.md) · [Continuous Batching](03-continuous-batching.md)

## Why this exists

You could write a serving loop yourself: batch requests, manage a KV cache, handle streaming,
implement sampling. You would then spend months rediscovering the problems that two projects
have already solved. Both vLLM and SGLang give you continuous batching, a paged KV cache,
prefix caching, tensor parallelism and an OpenAI-compatible HTTP surface.

They are close cousins with different emphases. vLLM is the broader, more widely deployed
engine with the largest model and hardware coverage. SGLang grew out of a research agenda
around structured generation and aggressive prefix reuse, and its frontend lets you write
multi-step LLM programs. The mental model: **vLLM optimises the serving loop; SGLang
optimises the serving loop *and* the repeated-prefix patterns typical of agents.**

## The mental model

```text
             your app (OpenAI SDK, any language)
                          |
                  /v1/chat/completions
                          |
        +-----------------+-----------------+
        |                                   |
     vLLM                               SGLang
  PagedAttention KV cache            RadixAttention prefix tree
  continuous batching                continuous batching
  prefix caching (opt-in/auto)       prefix reuse by default
  broad model + quant coverage       fast constrained decoding
  OpenAI server built in             OpenAI-compatible server built in
```

| Question | Leans vLLM | Leans SGLang |
|---|---|---|
| Very new or unusual architecture | yes | check support |
| Many different quant formats | yes | good coverage |
| Agents with huge shared system prompts | good | excellent |
| Strict JSON or grammar output at speed | good | excellent |
| Team wants one familiar default | yes | if already in the ecosystem |

## Core concepts

### Both engines are configured by one memory knob

The critical setting is how much of the GPU the engine may use for weights plus KV cache.
In vLLM this is `--gpu-memory-utilization`, a fraction of total card memory, defaulting to a
high value. The engine then profiles the model, subtracts weights and activation peaks, and
gives the remainder to KV cache blocks. That remainder — divided by your cache bytes per token
from [The KV Cache](02-kv-cache.md) — is your real maximum token capacity.

Set it too high and you get out-of-memory crashes when activation peaks spike on a long
prompt. Set it too low and you waste card. Leaving a gigabyte or two of headroom is normal.

### Prefix caching is the highest-leverage feature for chat and agents

If your requests begin with the same long text — a system prompt, tool definitions, a retrieved
document — the engine can reuse the computed KV blocks for that prefix instead of re-running
prefill. With 128 KiB per token, a shared 3,000-token prefix is 375 MiB of cache computed once
and referenced thereafter.

The saving is not just memory, it is TTFT. Prefill over 3,000 tokens is real work; skipping it
moves the first token out by that whole amount. Both engines support this — vLLM through
prefix caching, SGLang through its radix tree — but caching only helps when the prefix is
byte-identical. A timestamp or a session id at the top of your system prompt destroys it. Put
anything variable at the *end* of the prompt, not the start.

### Tensor parallelism has a cost you can predict

To serve a model that does not fit on one card, both engines split layers across GPUs. Every
layer then needs an all-reduce, which is a network round trip per layer per token. So:

```text
TP=2 gives you 2x the memory and roughly 2x the compute,
but decode latency per token is higher than a single card,
because each step now waits on inter-GPU communication.
```

Prefer the fewest GPUs that fit, and prefer quantizing onto one card over splitting across two
if latency matters. Splitting is for capacity, not for speed.

### Choose the model id from what the server actually serves

Engines accept a Hugging Face repo id or a local path. Whatever you pass becomes the model name
clients must send. Query the server's `/v1/models` route rather than assuming a friendly name
like `gpt-4` will work.

## Code

First, launching. These are CLI commands; flag names do change between releases, so run
`vllm serve --help` on your installed version rather than copying flags from any blog post,
including this one.

```bash
# vLLM: OpenAI-compatible server on :8000
vllm serve <hf-repo-id> \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --served-model-name my-served-model

# SGLang: same OpenAI surface, its own flags
python -m sglang.launch_server \
  --model-path <hf-repo-id> \
  --host 0.0.0.0 --port 8000 \
  --context-length 8192 \
  --mem-fraction-static 0.85 \
  --served-model-name my-served-model
```

Second, driving either engine in-process from Python, which is how you benchmark or script
batch work without HTTP overhead. The `LLM` class is synchronous batching; for a server use the
CLI above.

```python
from __future__ import annotations

import time
from vllm import LLM, SamplingParams


def main() -> None:
    # Check the current docs for argument names; they evolve between releases.
    llm = LLM(
        model="<hf-repo-id>",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    params = SamplingParams(temperature=0.2, max_tokens=128)

    shared_prefix = "You are a support agent. Policy: " + ("rule; " * 800)
    prompts = [shared_prefix + f"\nQuestion {i}: how do I return an item?"
               for i in range(32)]

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    produced = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"{len(outputs)} requests, {produced} tokens in {elapsed:.2f}s "
          f"= {produced / elapsed:.1f} tok/s aggregate")
    # Compare this run with prefix caching disabled to see the TTFT benefit.
    print(outputs[0].outputs[0].text[:200])


if __name__ == "__main__":
    main()
```

You should see aggregate throughput far above the single-stream ceiling you computed in
[Prefill and Decode](01-prefill-decode.md), because thirty-two sequences share each weight
read. Turn prefix caching off and the same run takes measurably longer on prefill.

## Real use case

A team built a coding agent that ran up to twenty tool-calling steps per task. Every step sent
the whole conversation plus a 4,000-token tool schema and system prompt. On a naive server the
agent was slow in a way that looked like model latency but was not: each of the twenty steps
re-prefilled a prompt that was 95 percent identical to the previous one.

They moved to SGLang, whose radix prefix cache matches the longest shared prefix automatically,
and restructured the prompt so the invariant part — system prompt and tool definitions — came
first and byte-identical, with the growing conversation after it. Step one still paid full
prefill; steps two through twenty hit the cache for the shared portion and only prefilled the
new tokens.

The secondary win was memory. With cache blocks shared across the twenty steps of a task, and
across the many tasks running concurrently, the same card held more concurrent agents. The
team's capacity plan, which they had built from the KV cache formula, held up because they
could now see how many *distinct* prefixes were live rather than counting every request as
independent.

## Common mistakes

- **Mistake:** putting a timestamp or request id at the top of the system prompt — **Instead:** keep the prefix byte-identical and move variable content to the end.
- **Mistake:** copying engine flags from a tutorial — **Instead:** check `--help` for your installed version; these change often.
- **Mistake:** setting `gpu_memory_utilization` to 0.99 — **Instead:** leave headroom for activation peaks or long prompts will OOM you.
- **Mistake:** splitting a model across two GPUs to make it faster — **Instead:** quantize onto one card when latency matters; tensor parallelism buys capacity, not speed.
- **Mistake:** assuming a client can send any model name — **Instead:** read `/v1/models` and use the served name.

## Checkpoint

1. Your card is 80 GB, the engine is allowed 0.90, weights are 16 GB, and activation peaks plus framework overhead are about 6 GB. At 128 KiB per token, how many 8k-token sequences fit?
2. Run the same batch with prefix caching enabled and disabled, and report the difference in both wall-clock time and peak memory.
3. Set `--max-num-seqs` to 1, then 8, then 64 on a fixed workload of 64 short requests. Record p50 TTFT, p95 TTFT and aggregate tokens per second for each, and describe the trade-off curve.

**Self-check:** `80 x 0.90 = 72 GB`; minus 16 GB weights and 6 GB overhead leaves 50 GB; an 8k sequence costs `8192 x 128 KiB = 1 GiB`, so about 50 sequences fit, minus fragmentation and block rounding.

## Go deeper

- [Ollama and TensorRT-LLM](09-ollama-tensorrt.md) — the local-development and maximum-speed ends of the spectrum.
- [GPU Memory Budgeting](10-gpu-memory.md) — turn these knobs into a sizing decision.
- [Ship: A vLLM Server](11-ship-vllm-server.md) — the project that measures all of this on your hardware.
- [LangGraph](../12-orchestration/02-langgraph.md) — the multi-step agent loops that make prefix reuse so valuable.

---
[← Previous: Distillation](07-distillation.md) · [Stage index](README.md) · [Next: Ollama and TensorRT-LLM](09-ollama-tensorrt.md)
