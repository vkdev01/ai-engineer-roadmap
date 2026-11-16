# 9. Ollama and TensorRT-LLM

> **Stage 16 · Inference** — the two ends of the runner spectrum: one command on a laptop, versus a compiled engine for maximum speed.
>
> **You'll be able to:** run a model locally in minutes and explain when a compiled engine earns its build complexity.
>
> **Prerequisites:** [Quantization](06-quantization.md) · [vLLM and SGLang](08-vllm-sglang.md)

## Why this exists

The same weights can be served by wildly different software. On one end, Ollama downloads a
model and starts an HTTP server with no configuration, on a CPU if necessary. On the other,
TensorRT-LLM compiles a model into a GPU-specific engine with fused kernels and in-flight
batching, and can be meaningfully faster than a generic runtime — at the cost of a build step
and a tighter coupling to one GPU family and one model version.

Between them sits everything else. Knowing where a runner sits tells you what you give up.
The mental model: **Ollama optimises for time-to-first-token-of-your-project; TensorRT-LLM
optimises for tokens-per-second-per-card in production.**

## The mental model

```text
laptop / dev box                     production GPU node
+----------------------+             +---------------------------+
| Ollama               |             | TensorRT-LLM              |
| GGUF weights         |             | compiled engine per GPU   |
| CPU or GPU           |             | CUDA kernels, fp8/int4    |
| simple HTTP API      |             | in-flight batching        |
| one command to run   |             | build step, version-locked|
+----------------------+             +---------------------------+
        fast to start                        fast to run
```

| | Ollama | TensorRT-LLM |
|---|---|---|
| Setup | one binary, one pull | build engine, container, matching driver |
| Hardware | CPU, Apple Silicon, consumer GPUs | specific NVIDIA GPU families |
| Weights | GGUF quant types | engine built with a quant recipe |
| Portability | high | low, rebuild per GPU/model/version |
| Best use | development, demos, offline apps | high-volume serving on owned hardware |
| Batching | modest | in-flight batching, high concurrency |

## Core concepts

### Local runners exist so you can iterate without a GPU cluster

Ollama pulls a GGUF quant of a model, keeps it in a local store, and serves a simple HTTP API
plus an OpenAI-compatible route. That means you can develop an application against a real
local model, with no API key, no per-token cost, and no data leaving the machine. For a
beginner this is the fastest way to see how a model behaves; for a team it is the fastest way
to test a prompt change without spending money.

The trade is throughput. GGUF quant types are designed to be efficient on CPUs and mixed
hardware, not to saturate a datacentre GPU. Do not benchmark your production plan on a laptop.

### A Modelfile is a tiny, versionable model configuration

Ollama's `Modelfile` pins the base weights, the quantization, the context length, the system
prompt and sampling defaults. Committing it to your repo gives you reproducible local
behaviour — the same discipline you apply to prompts in
[Prompt Versioning](../08-prompt-and-context/05-prompt-versioning.md).

### TensorRT-LLM trades flexibility for kernel-level speed

TensorRT-LLM compiles your model into an engine. Compilation lets it fuse operations, pick
kernel implementations for the exact GPU, and specialise for a maximum batch size and sequence
length. That is why it can beat a generic runtime — and why the engine is not portable: change
the GPU model, the library version, or the maximum sequence length, and you rebuild.

The practical consequence: keep the build in CI as an artifact, version it, and treat engine
builds like container images. A team that rebuilds engines by hand on a shared box will
eventually ship a mismatched pair of engine and driver and lose a day.

### Quantization is how these runners fit on the hardware you have

Both lean on the bit-width arithmetic from [Quantization](06-quantization.md). A 70B model at
about 4.25 bits per weight is roughly 35 GiB, which is the difference between a consumer
32 GB card being useless and a workstation GPU being enough. On the compiled side, fp8 and
int4 weight-only recipes are the common production choices.

### Choose by constraint, not by reputation

- **Iterating on a prompt on your laptop:** Ollama.
- **Serving a small model inside a desktop app, offline:** Ollama or an embedded GGUF runtime.
- **High volume on GPUs you own, latency critical:** TensorRT-LLM, or vLLM if you value
  flexibility more than the last few percent.
- **A model that changes weekly:** vLLM or SGLang; a compile step per change is friction.

## Code

First, a `Modelfile` that pins everything that matters, then the two commands to build and run
it. The API call afterwards is plain HTTP, so it works from any language.

```bash
# Modelfile
# FROM a GGUF quant tag published for the model you want
FROM <gguf-model-tag>
PARAMETER num_ctx 8192
PARAMETER temperature 0.2
SYSTEM "You are a terse assistant. Answer in at most three sentences."

# build a named local model from it, then serve
ollama create support-bot -f Modelfile
ollama run support-bot
```

Second, calling the local server from Python. Ollama exposes both its own `/api/chat` shape and
an OpenAI-compatible `/v1` route; use whichever fits your existing client code.

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def stream_chat(prompt: str, model: str = "support-bot") -> AsyncIterator[str]:
    """Stream tokens from a local Ollama server over its native API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"num_ctx": 8192, "temperature": 0.2},
    }
    async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=120.0) as client:
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)          # one JSON object per line
                if text := chunk.get("message", {}).get("content"):
                    yield text
                if chunk.get("done"):
                    # final object carries timing counters, useful for local profiling
                    print(f"eval_count={chunk.get('eval_count')} "
                          f"eval_duration_ns={chunk.get('eval_duration')}")
                    break
```

You should see tokens arrive incrementally, and a final object whose `eval_count` divided by
`eval_duration` gives you your machine's real decode rate — the number to compare against the
bandwidth ceiling from [Prefill and Decode](01-prefill-decode.md).

For TensorRT-LLM, the build step looks roughly like this. Exact flags and the supported
quantization recipes depend on your installed version; check that version's documentation.

```bash
# Convert the checkpoint to the engine's format, then build an engine for your GPU.
python convert_checkpoint.py --model_dir ./hf-model \
  --output_dir ./ckpt --dtype float16 --use_weight_only --weight_only_precision int4_awq

trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine \
  --max_batch_size 64 --max_input_len 4096 --max_seq_len 8192

# Serve the compiled engine with the bundled OpenAI-compatible server.
trtllm-serve ./engine --host 0.0.0.0 --port 8000
```

Note the `--max_batch_size` and `--max_seq_len`: they are compiled in. Choosing them is a
capacity decision, not a runtime one, which is the core difference from vLLM.

## Real use case

A company shipped a desktop research assistant that had to work on a plane. Their prototype
called a hosted API, which failed the requirement outright: no network, and confidential
documents could not leave the machine. They rebuilt the model layer around a local GGUF quant
served by Ollama, with the model bundled at install time and a pinned `Modelfile` so behaviour
did not drift between releases.

Two problems surfaced that only appear offline. First, the model's context length defaulted
lower than the app assumed, so long documents were silently truncated; pinning `num_ctx` in the
`Modelfile` fixed it. Second, on machines without a capable GPU, decode was slow enough that
streaming became essential — they had been building responses as a whole string. Switching to
the streaming API meant the UI showed progress from the first token.

The same team later built a server-side batch summariser for the same documents. There they used
TensorRT-LLM on owned GPUs, because the workload was high volume, the model was fixed, and the
hardware fleet was homogeneous — exactly the conditions where a compile step pays for itself.
Two runners, two constraints, one shared prompt layer.

## Common mistakes

- **Mistake:** benchmarking a local GGUF runner and using it to size production GPUs — **Instead:** size production on the engine you will deploy.
- **Mistake:** leaving context length at the runner's default — **Instead:** pin `num_ctx`; silent truncation looks like a model quality problem.
- **Mistake:** building a TensorRT engine by hand on one machine — **Instead:** build in CI, version the artifact, pin the driver and library version.
- **Mistake:** choosing TensorRT-LLM for a model you swap weekly — **Instead:** use vLLM or SGLang until the model stabilises.
- **Mistake:** assuming an offline app can build a response in one piece — **Instead:** stream, so slow hardware still feels responsive.

## Checkpoint

1. Write a `Modelfile` for a local model with an 8k context, temperature 0, and a system prompt that forbids markdown. Build it and confirm the system prompt is honoured.
2. Measure decode rate on your own machine using the `eval_count` and `eval_duration` fields, then compute the implied bandwidth from the model's file size. Explain any gap.
3. For a fixed model and a fixed GPU, write down the three things that would force you to rebuild a TensorRT-LLM engine, and estimate how often each changes in your project.

**Self-check:** `FROM` sets the weights, `PARAMETER num_ctx 8192` sets context, `PARAMETER temperature 0` makes output deterministic, and a `SYSTEM` line enforces the no-markdown rule; rebuild with `ollama create` after editing.

## Go deeper

- [vLLM and SGLang](08-vllm-sglang.md) — the middle ground you will probably deploy first.
- [Quantization](06-quantization.md) — the bit-width decisions both runners depend on.
- [Docker](../19-cloud-devops/04-docker.md) — packaging a compiled engine reproducibly.
- [GPU Cloud](../19-cloud-devops/08-gpu-cloud.md) — renting the hardware these engines are built for.

---
[← Previous: vLLM and SGLang](08-vllm-sglang.md) · [Stage index](README.md) · [Next: GPU Memory Budgeting](10-gpu-memory.md)
