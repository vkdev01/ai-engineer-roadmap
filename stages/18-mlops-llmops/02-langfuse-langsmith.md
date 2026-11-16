# 2. LLM Observability with Langfuse and LangSmith

> **Stage 18 · MLOps and LLMOps** — the production lens that shows you what users actually sent, what the model actually did, and what it cost.
>
> **You'll be able to:** instrument an LLM service so every request produces a trace you can read, group traces into user sessions, and attach thumbs-up/down feedback to the exact generation it judges.
>
> **Prerequisites:** [Experiment Tracking with MLflow and W&B](01-mlflow-wandb.md) · [Stage 17 — AI Backend](../17-ai-backend/README.md)

## Why this exists

Offline evals tell you how your system behaves on 240 questions you wrote. Production tells you about the other 40,000 questions you did not think of, asked in Tamil-English, in a hurry, with typos, pasted into a chat box at 2 a.m. Your eval suite will never cover those.

When a user says "it gave me a wrong answer yesterday", the useful questions are: what exactly did they type, what did retrieval return, which prompt version rendered, which tool calls ran, how many tokens, and what did the model say. Application logs give you a line of text. A trace gives you the tree.

The mental model: **a trace is one request's family tree.** The root is the user request; children are the retrieval call, each model call, each tool call; each node carries inputs, outputs, timings, tokens and metadata.

## The mental model

```text
trace  7f21  (user u_88, session s_4, release 2026-03-05)
|  input: "how do I cancel an order that already shipped?"
|-- span  retrieve         42ms   k=5, index=faq-v3
|-- span  rerank          180ms   model=bge-reranker
|-- span  llm.answer     1240ms   gpt-4.1-mini  in 812 / out 141 tok
|     metadata: prompt_version=answer.v4, prompt_hash=c40f
|     score   : thumbs_down (from UI)  -> linked to this generation
`-- span  tool.order_lookup  310ms  order_id=OR-99x
```

Langfuse and LangSmith both store this shape and both render it as a timeline plus a tree. Langfuse is open source and self-hostable; LangSmith is the hosted companion to the LangChain ecosystem with strong dataset and eval integration. The concepts below transfer between them, and both accept OpenTelemetry-style spans if you prefer a vendor-neutral emitter.

## Core concepts

### Trace, span, generation, session

A **trace** is one end-to-end request. A **span** is any timed step inside it. A **generation** is a span that is specifically a model call, and it carries the extra fields that matter for LLMs: model id, prompt messages, completion, token counts, cost. A **session** groups traces that belong to one conversation, so you can see a user's third turn in the context of their first two.

Get these four right and everything else is UI. Most instrumentation bugs are one of two kinds: everything flattened into a single span (no tree, no per-step latency), or every internal helper function made a span (a hundred-node tree nobody reads).

### Attach the metadata you will filter on

Traces are only useful if you can query them. Before you write any code, decide the metadata keys: `tenant_id`, `feature`, `prompt_version`, `release_sha`, `model_id`, `env`. Put them on the root trace so a filter finds the whole tree. These same keys are what your cost dashboard groups by later, so agree on the names once and use them everywhere. Inconsistent keys (`tenant` in one service, `tenant_id` in another) is the most common reason a dashboard comes out blank.

### User feedback is a first-class signal

A thumbs-down is worth more than a hundred synthetic eval rows, but only if it is attached to the generation that earned it, not to a support ticket. Both platforms let you write a score against a trace id from your own backend: the browser sends `{trace_id, value}` and your API records it. Store the trace id in your frontend state next to the answer so the button knows what it is judging.

### Sampling: full capture is a bill, zero capture is blind

Traces cost storage and, in hosted tools, money. Sample deliberately: capture 100% of errors and of thumbs-down-linked traces, plus a small random share of successes. If your SDK supports head sampling, set it low and use tail sampling where the platform offers it, so a slow or failing request is kept even when its head was not sampled.

## Code

A FastAPI endpoint that creates one trace per request, nests a retrieval span and a generation, and tags the trace with the metadata you will filter on. The SDK call shapes differ slightly between Langfuse versions and between Langfuse and LangSmith; check the current docs for the exact import path and decorator signature.

```python
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI
from langfuse import Langfuse
from pydantic import BaseModel

app = FastAPI()
# Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from the environment.
lf = Langfuse()


class AskRequest(BaseModel):
    question: str
    tenant_id: str
    feature: str = "support"
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    trace_id: str  # the frontend echoes this back when the user clicks thumbs


def retrieve(question: str, k: int = 5) -> list[str]:
    # Stand-in for your vector search; returns chunk ids.
    return [f"chunk-{i}" for i in range(k)]


def call_model(messages: list[dict[str, str]]) -> tuple[str, int, int]:
    # Replace with a real client call; check the provider docs for the model id.
    return "You can cancel an in-transit order from Orders > Cancel.", 812, 141


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    trace = lf.trace(
        name="support.ask",
        input=req.question,
        session_id=req.session_id or str(uuid.uuid4()),
        metadata={
            "tenant_id": req.tenant_id,
            "feature": req.feature,
            "prompt_version": "answer.v4",
            "env": "prod",
        },
    )

    started = time.perf_counter()
    chunks = retrieve(req.question, k=5)
    trace.span(name="retrieve", input=req.question, output=chunks,
               metadata={"elapsed_ms": round((time.perf_counter() - started) * 1000)})

    messages = [
        {"role": "system", "content": "Answer only from the provided context."},
        {"role": "user", "content": f"Context: {chunks}\n\nQ: {req.question}"},
    ]
    answer, tok_in, tok_out = call_model(messages)

    # A generation, not a plain span: it carries the LLM-specific fields.
    trace.generation(
        name="llm.answer",
        model="gpt-4.1-mini",
        input=messages,
        output=answer,
        usage={"input": tok_in, "output": tok_out},
        metadata={"prompt_version": "answer.v4"},
    )

    trace.update(output=answer)
    lf.flush()  # do not rely on process exit in a serverless or container deploy
    return AskResponse(answer=answer, trace_id=trace.id)
```

Second snippet: recording user feedback against that trace id. Note that this endpoint must validate the trace id belongs to the caller's tenant, or one tenant can score another's traces.

```python
from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.main import lf  # same Langfuse client


class Feedback(BaseModel):
    trace_id: str
    value: int = Field(ge=-1, le=1)  # -1 thumbs down, 1 thumbs up
    comment: str | None = None


@app.post("/feedback")
async def feedback(body: Feedback, tenant_id: str) -> dict[str, str]:
    trace = lf.get_trace(body.trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="unknown trace")
    if trace.metadata.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="not your trace")
    lf.score(trace_id=body.trace_id, name="user_thumbs", value=body.value,
             comment=body.comment)
    lf.flush()
    return {"status": "recorded"}
```

## Real use case

A B2B document assistant answers questions over customer contracts. Support gets a complaint: "it invented a termination clause". Before tracing, the engineer reads the app log, sees the answer text, and cannot tell whether retrieval missed the clause, the prompt ignored it, or the model hallucinated. With traces, the specific request is found by filtering `feature=contracts` and `user_thumbs=-1` in the last 24 hours. The trace shows retrieval returned five chunks, none containing "termination", because the upload pipeline had indexed the pre-signature draft. That is a data bug, not a prompt bug, and it took four minutes to identify instead of a day of reproduction attempts.

The design choice that made it work was putting `tenant_id` and `feature` on the root trace and attaching feedback to the generation. Without the tenant key the complaint could not be located at all, because the customer had not shared a request id.

## Common mistakes

**Mistake:** one span for the whole request — **Instead:** span per model call and per tool call, so latency and token cost attribute to the step that caused them.

**Mistake:** logging raw prompts and completions that contain customer PII — **Instead:** redact at the SDK boundary with a hook, or mask fields before sending; see [PII Redaction](../20-ai-security/04-pii-redaction.md).

**Mistake:** tracing in dev but not in prod, or in prod but never sampling — **Instead:** always-on for errors and negative feedback, sampled for the rest, so cost stays bounded without going blind.

**Mistake:** fire-and-forget export with no flush — **Instead:** call `flush()` before returning from a serverless handler or on shutdown, or you lose the traces of the requests that crashed.

**Mistake:** a feedback endpoint that trusts the trace id blindly — **Instead:** verify tenant ownership before writing the score.

## Checkpoint

1. Instrument one endpoint from your Stage 17 service so it emits a root trace, a retrieval span and a generation span, with `feature` and `prompt_version` metadata. Trigger it three times and confirm three trees appear.
2. Add `/feedback` and wire one button in a minimal HTML page that posts the trace id. Confirm the score appears attached to the generation, not to the trace root.
3. Turn on sampling at 10% and add a rule that keeps 100% of traces where `error=true`. Replay 50 requests, half of them forced failures, and verify all 25 failures were captured.

**Self-check:** exercise 1 is done when your trace view shows one root with two children and the generation lists input tokens, output tokens and model id.

## Go deeper

- [Tracing Design](05-tracing.md) — the design rules for span boundaries and attributes, once the basics work.
- [Cost Monitoring and Budgets](06-cost-monitoring.md) — the token counts you just logged become per-tenant spend here.
- [Evals](../15-evals/09-online-offline-evals.md) — how online traces feed back into offline test sets.
- [Prompt Versioning and Registries](03-prompt-versioning.md) — making the `prompt_version` you tag traces with a real, resolvable thing.

---

[← Previous: Experiment Tracking with MLflow and W&B](01-mlflow-wandb.md) · [Stage index](README.md) · [Next: Prompt Versioning and Registries](03-prompt-versioning.md)
