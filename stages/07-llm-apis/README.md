# Stage 07 — LLM APIs

Models do not appear in your program by magic. You send tokens over HTTP, you get tokens back, and every production problem — streaming, retries, cost, fallback — lives in that boundary. This stage teaches you to write that boundary yourself, so that the SDK is a detail and not a dependency.

It sits after Stage 06 because you now know what a token, a context window and a chat template are. It sits before RAG and agents because both are, at the code level, loops that call the API you build here.

**Time:** 3-4 weeks · **You need first:** Stage 06 · **You leave with:** a multi-model client library with typed responses, retries, a cost ledger and provider fallback

## What you will be able to do

- Call OpenAI-, Anthropic- and Gemini-style chat APIs and read the response objects without guessing at field names.
- Stream tokens to a terminal with Python generators and to a browser with Server-Sent Events from a Node service.
- Force a JSON shape out of a model with schema-constrained output, parse it with pydantic or zod, and repair it when it breaks.
- Run a full tool-calling loop, including feeding `tool_result` messages back to the model.
- Handle 429s and 5xx with exponential backoff plus jitter, and keep a per-request and per-tenant cost ledger.
- Route between models and fall back on error, timeout or bad output without the caller noticing.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [The OpenAI and Anthropic Chat APIs](01-openai-anthropic.md) | The request and response shapes you will reuse everywhere |
| 2 | [Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md) | A third provider shape, and how to call open models |
| 3 | [Streaming Responses](03-streaming.md) | Token-by-token output with SSE and async generators |
| 4 | [Structured Outputs](04-structured-outputs.md) | Schemas in, validated objects out, with repair loops |
| 5 | [Tool Calling](05-tool-calling.md) | The full request, execute, feed-back loop |
| 6 | [Multimodal APIs](06-multimodal-apis.md) | Images, PDFs and audio as request content |
| 7 | [Retries and Rate Limits](07-retries-rate-limits.md) | Backoff, jitter and what not to retry |
| 8 | [Token Accounting and Cost Tracking](08-cost-tracking.md) | Counting tokens and attributing cost per request |
| 9 | [Routing and Fallbacks](09-routing-fallbacks.md) | Choosing a model at runtime and degrading gracefully |
| 10 | [Ship: A Multi-Model Client](10-ship-multi-model-client.md) | One interface over several providers, with a ledger |

## The shipping project

[10-ship-multi-model-client.md](10-ship-multi-model-client.md) — you build a client library with one interface over at least two providers, typed responses, retries with jitter, a cost ledger and automatic fallback. You prove it works by running the same call against two providers, forcing a failure in the primary, and showing the fallback answer plus a correct cost row.

## How to know you are done

- You can switch provider with one line of config and the calling code does not change.
- A forced 429 and a forced timeout both recover without an exception reaching the caller.
- Every call writes a ledger row with input tokens, output tokens and an estimated cost.
- Your streaming endpoint delivers first token to a browser before the full response exists on the server.
- A malformed JSON answer from the model is repaired or rejected, never passed upward unparsed.

---
[← Back to roadmap](../../README.md) · [Next stage: Prompt and Context](../08-prompt-and-context/README.md)
