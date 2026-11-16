# Stage 16 — Inference

Inference is where your model meets a user and the bill. Every stage before this one
produced a model or an application; this stage explains what actually happens inside the
serving stack when a request arrives, so you can choose an engine, size a GPU, and explain
your numbers. It sits after evals because you should only optimise a model you can measure.

**Time:** 2-3 weeks · **You need first:** Stage 06 (LLM Fundamentals) and Stage 15 (Evals) · **You leave with:** a vLLM server behind an OpenAI-compatible endpoint, with a load test that reports TTFT, tokens per second and GPU memory

## What you will be able to do

- Explain the prefill and decode phases and why time-to-first-token and tokens-per-second are different problems.
- Compute KV cache size from a model config by hand, and use that number to budget a GPU.
- Reason about batching, paged memory and speculative decoding in throughput-versus-latency terms.
- Pick between vLLM, SGLang, Ollama and TensorRT-LLM for a given constraint.
- Serve an open-weight model with real measurements instead of guessed benchmark numbers.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Prefill and Decode](01-prefill-decode.md) | The two phases and the two latency numbers |
| 2 | [The KV Cache](02-kv-cache.md) | The memory formula that decides everything |
| 3 | [Continuous Batching](03-continuous-batching.md) | Throughput without wrecking latency |
| 4 | [PagedAttention](04-pagedattention.md) | Why memory fragmentation caps your batch |
| 5 | [Speculative Decoding](05-speculative-decoding.md) | Trading spare compute for fewer sequential steps |
| 6 | [Quantization](06-quantization.md) | fp16, int8, int4 and what quality loss looks like |
| 7 | [Distillation](07-distillation.md) | A smaller model that keeps the teacher's behaviour |
| 8 | [vLLM and SGLang](08-vllm-sglang.md) | The two engines you will actually deploy |
| 9 | [Ollama and TensorRT-LLM](09-ollama-tensorrt.md) | Local development versus maximum speed |
| 10 | [GPU Memory Budgeting](10-gpu-memory.md) | Weights plus KV plus activations plus overhead |
| 11 | [Ship: A vLLM Server](11-ship-vllm-server.md) | A measured endpoint, not a claim |

## The shipping project

[11-ship-vllm-server.md](11-ship-vllm-server.md) — you serve an open-weight model with vLLM on an OpenAI-compatible route, then run a load test that reports TTFT percentiles, output tokens per second and peak GPU memory. You prove it works by predicting the numbers from arithmetic first, then comparing your prediction to the measurement.

## How to know you are done

- You can compute KV cache bytes per token for a model you have never seen, from its config.
- Your load test output includes p50 and p95 TTFT, aggregate tokens per second, and peak memory.
- Your measured peak memory is within roughly 15 percent of your own hand calculation.
- You can say, in one sentence each, when you would pick each of the four engines.
- No number in your write-up is a copied benchmark you cannot reproduce.

---
[← Back to roadmap](../../README.md) · [Next stage: AI Backend](../17-ai-backend/README.md)
