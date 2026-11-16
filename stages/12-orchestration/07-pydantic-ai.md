# 7. Pydantic AI

> **Stage 12 · Orchestration** — typed agents: the return type is the contract, dependencies are injected, and validation failures become retries you did not have to write.
>
> **You'll be able to:** define an agent whose output is a validated Pydantic model, wire typed dependencies, and test it without a network call.
>
> **Prerequisites:** [Semantic Kernel](06-semantic-kernel.md) · [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md)

## Why this exists

You already used Pydantic for tool schemas. Pydantic AI takes the next step: the agent's **final output is also a Pydantic model**, and if the model returns something that does not validate, the framework feeds the validation error back and retries. That one idea removes a whole class of glue code — the parse, the try/except, the "please return valid JSON" follow-up prompt.

The second idea is dependency injection. Instead of tools reaching into globals for a database or an HTTP client, you declare a dependency type and the framework passes it in. That makes agents testable in the way ordinary functions are testable: give a fake dependency, assert on a typed result.

The mental model: **an agent is a typed function from input plus dependencies to a validated output.**

## The mental model

```text
   @agent ... output_type=Report        ┌─── deps: Deps (db, clock, user) ───┐
                                        │                                     │
  question ──► run(question, deps=d) ──►│  model ⇄ tools (typed args)        │
                                        │    │                                │
                                        └────┼────────────────────────────────┘
                                             ▼
                                  validate output against Report
                                     │ ok            │ invalid
                                     ▼               ▼
                               Report object    retry with the error text
```

| Idea | What it replaces |
|---|---|
| `output_type` | Hand-written parsing and retry prompts |
| `deps_type` | Globals and hidden singletons in tools |
| `RunContext[Deps]` | Passing a context dict through every function |
| `TestModel` | Mocks for unit tests |
| Usage limits | Your own counter plumbing |

## Core concepts

### The output type is the specification

Declare a Pydantic model with the fields you actually need and constraints you actually enforce: `confidence: float = Field(ge=0, le=1)`, `citations: list[str] = Field(min_length=1)`. The schema goes to the model, the validation runs on the reply, and a mismatch triggers a bounded retry. This is the difference between "the agent usually returns JSON" and "the agent cannot return anything else".

### Dependencies are explicit, so tests are trivial

Declare `deps_type=Deps` where `Deps` holds the database handle, the clock and the current user. Tools take `ctx: RunContext[Deps]` and read from it. In tests you construct a `Deps` with a fake and run the agent with `TestModel`, which generates a valid structured response without a network call. Fast, deterministic, and it catches schema mistakes.

### Tools are still typed functions

Registering a tool is a decorator on a typed function; the signature becomes the schema. Docstrings become descriptions, so write them for the model. The rule from [Tool Calling](../11-agents/04-tool-calling.md) holds: name the when, the when-not and the return shape.

### Usage limits are built in, not bolted on

Requests and tool calls per run can be capped by the framework, which is the loop-engineering work from [Loop Engineering](../11-agents/10-loop-engineering.md) in one parameter. Combine it with your own trace; the limit stops the run, the trace explains it.

### Where it fits, and where it does not

It fits services where correctness of shape matters: extraction, classification with reasons, report generation with citations, anything feeding another system. It is not a graph engine — for branching, parallel branches and persistence you want [LangGraph](02-langgraph.md). Many teams use both: Pydantic AI for the typed inner step, LangGraph for the outer flow.

## Code

A typed agent with injected dependencies, two tools and a validated output. The model id must be checked against the provider's current docs.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.test import TestModel

@dataclass
class Deps:
    db: object                     # read-only repository in prod, a fake in tests
    today: str                     # injected clock: makes runs reproducible

class Claim(BaseModel):
    claim: str = Field(description="One factual statement from the report")
    source_id: str = Field(pattern=r"^doc-[0-9]+$", description="Passage id that supports it")

class Report(BaseModel):
    title: str = Field(max_length=120)
    summary: str = Field(min_length=50, max_length=600)
    claims: list[Claim] = Field(min_length=1, max_length=8)
    verdict: Literal["supported", "unsupported", "mixed"]

agent: Agent[Deps, Report] = Agent(
    "<current-chat-model-id>",
    deps_type=Deps,
    output_type=Report,                 # the contract: validated, with bounded retries
    system_prompt=("You write short evidence-backed reports. Every claim needs a source id "
                   "from the passages you retrieved. Never invent an id."),
)

@agent.tool
async def search(ctx: RunContext[Deps], query: str) -> str:
    """Search the document store. Use for factual questions; returns passages with ids."""
    rows = ctx.deps.db.search(query, k=5)
    if not rows:
        return f'no results for query="{query}"'
    return "\n".join(f"[{r.id}] {r.text[:240]}" for r in rows)

@agent.tool
async def doc_date(ctx: RunContext[Deps], doc_id: str) -> str:
    """Return the effective date of a document, to check whether it is still current."""
    return ctx.deps.db.effective_date(doc_id) or "unknown"

async def write_report(question: str, deps: Deps) -> Report:
    # usage_limits caps model requests and tool calls: the budget lives here.
    result = await agent.run(question, deps=deps, usage=UsageLimits(request_limit=6))
    return result.output

def test_agent_shape() -> None:
    # TestModel generates a schema-valid response with no network call.
    fake = Deps(db=object(), today="2026-01-01")
    result = agent.run_sync("summarise the refund policy", deps=fake,
                            model=TestModel())
    assert isinstance(result.output, Report)
```

Second, a typed tool that returns structured data rather than text — the model sees a schema, and downstream code gets an object. This is the pattern that makes an agent composable with the rest of your service.

```python
from pydantic import BaseModel

class Passage(BaseModel):
    id: str
    text: str
    score: float

@agent.tool_plain
def retrieve(query: str) -> list[Passage]:
    """Retrieve passages for a query. Returns ids, text and relevance scores."""
    rows = _index.search(query, k=3)
    return [Passage(id=r.id, text=r.text[:240], score=round(r.score, 3)) for r in rows]
```

The TypeScript equivalent lives in the zod world, and the same principle applies: parse at the boundary, and treat a parse failure as a retry with the error text rather than an exception.

```ts
import { z } from "zod";

export const ReportSchema = z.object({
  title: z.string().max(120),
  summary: z.string().min(50).max(600),
  claims: z.array(z.object({ claim: z.string(), sourceId: z.string().regex(/^doc-\d+$/) })).min(1),
  verdict: z.enum(["supported", "unsupported", "mixed"]),
});

export type Report = z.infer<typeof ReportSchema>;

export async function parseWithRetry(
  call: (feedback?: string) => Promise<string>,
  maxAttempts = 2,
): Promise<Report> {
  let feedback: string | undefined;
  for (let i = 0; i < maxAttempts; i++) {
    const parsed = ReportSchema.safeParse(JSON.parse(await call(feedback)));
    if (parsed.success) return parsed.data;
    feedback = parsed.error.issues.map((e) => `${e.path.join(".")}: ${e.message}`).join("; ");
  }
  throw new Error(`model output did not validate: ${feedback}`);
}
```

## Real use case

A credit team extracted covenants from loan agreements: for each clause, the obligation, the threshold and the measurement period, feeding a monitoring system that raised alerts. Their first version asked for JSON and parsed it by hand; malformed output crashed the batch roughly every few hundred documents, and a missing threshold silently became `null` in the monitoring system.

With a typed output model, `threshold` became required with a numeric bound, and a clause that did not validate triggered one retry with the validator's message. The batch stopped crashing, and — the part they cared about — the monitoring system could no longer receive a null threshold, because the schema refused to produce one. Failures moved from "silent null" to "loud validation error with the clause attached".

## Common mistakes

- **Mistake:** an output model with every field optional — **Instead:** required fields with real constraints, so invalid output is impossible to accept.
- **Mistake:** tools reading module globals — **Instead:** declare dependencies and read them from the run context.
- **Mistake:** no retry bound on validation failure — **Instead:** cap retries and surface the last validation error.
- **Mistake:** using it for branching flows — **Instead:** pair it with a graph for control flow.
- **Mistake:** testing against a live model — **Instead:** use `TestModel` for shape tests and keep a separate eval set for quality.

## Checkpoint

1. Add a `confidence: float = Field(ge=0, le=1)` to `Report` and confirm an out-of-range value triggers a retry rather than an exception.
2. Replace `Deps.db` with a fake in a test and assert `search` returns the "no results" string for an unknown query.
3. Move `usage_limits` into a config object and log the remaining request budget at the end of a run.

**Self-check:** Exercise 1 — the validator rejects the value, the framework re-prompts with the error text, and after the retry cap it raises a typed failure you can catch and log.

## Go deeper

- [Type Hints and Pydantic](../01-programming-basics/02-type-hints-pydantic.md) — the modelling fundamentals.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — provider-level mechanics.
- [DSPy](08-dspy.md) — the next step: optimising the prompt, not just typing it.
- [Unit Tests](../15-evals/01-unit-tests.md) — testing agents with fakes and `TestModel`.

---

[← Previous: Semantic Kernel](06-semantic-kernel.md) · [Stage index](README.md) · [Next: DSPy](08-dspy.md)
