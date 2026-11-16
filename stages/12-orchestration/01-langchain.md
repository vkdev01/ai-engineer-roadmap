# 1. LangChain

> **Stage 12 · Orchestration** — what LangChain is genuinely good for, where the abstraction earns its place, and the places where it gets between you and the loop.
>
> **You'll be able to:** use LangChain for the parts it does well — model adapters, prompt templates, retrievers, output parsers — and drop to plain code for the loop.
>
> **Prerequisites:** [Ship: A Harness Agent](../11-agents/12-ship-harness-agent.md) · [Prompt Templates](../08-prompt-and-context/03-templates.md)

## Why this exists

Before LangChain, every project rewrote the same glue: a wrapper that normalises provider SDKs, a prompt template with variables, a retriever interface over three vector stores, a parser that turns text into JSON. LangChain's contribution was to standardise those interfaces so a component from one library plugs into another. That is real value and it is still the main reason to install it.

The trouble starts when you ask it to own control flow. Chains and legacy agents hide the loop behind builders, so a bug means reading framework source, and the abstraction you saved three days on costs you a week at the second debugging session. The working compromise most teams settle on: **use LangChain for components, write the loop yourself or use LangGraph.**

The mental model: **LangChain is a components library with an orchestration layer attached; use the first, be choosy about the second.**

## The mental model

```text
        what LangChain gives you            what you still own
        ─────────────────────────           ──────────────────────
        model adapters (one interface)      the loop and its stop rules
        prompt templates + partials         the budget and the trace
        retrievers over many stores         the policy and approvals
        output parsers / structured output  the evals
        callbacks for logging
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ your agent loop (Stage 11)         │
        │  registry · guard · trace · gate   │
        └────────────────────────────────────┘
```

## Core concepts

### The component interfaces are the durable part

`BaseChatModel`, `BaseRetriever`, `BaseOutputParser` and the prompt template classes are the parts that survive version churn, because they are contracts. Learning them well pays off even if you never use a chain: they are a vocabulary for "any model", "any retriever", "any parser", which makes your own code provider-agnostic.

### LCEL: pipes for data flow

The expression language composes components with `|`, which reads like a pipeline: prompt, model, parser. For linear flows it is clear and compact, and streaming works through it. Where it struggles is the moment you need a branch that depends on a tool result, or a retry that changes strategy — the pipe is a straight line by design, and forcing a loop into it produces clever code that nobody wants to maintain.

### Where it gets in the way

Three recurring complaints, all structural. **Depth:** a stack trace runs through five framework layers before reaching your bug. **Implicit state:** callbacks and config objects carry behaviour you cannot see at the call site. **Version drift:** package splits and renames mean tutorials from last year do not run. None of these are fatal; all of them argue for keeping your control flow in code you wrote.

### Adapters are worth it when you have more than one provider

If you genuinely route between providers — a cheap model for classification, a strong one for synthesis — a normalising adapter layer saves real work, and LangChain's is well tested. If you use one provider, a thin wrapper of your own is often clearer than a dependency.

### Callbacks for observability, then move on

The callback system gives you token counts and timings with little code. Use it to get started, but for anything you operate, export to a tracing tool rather than growing a callback class — see [Langfuse and LangSmith](../18-mlops-llmops/02-langfuse-langsmith.md).

## Code

A prompt template plus a retriever plus a parser, then the same pieces inside your own loop. The second snippet is the point: the components are useful, the loop stays yours. Check the current package names and model ids in the provider's docs before running.

```python
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Templates: variables are declared once and type-checked by usage.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the passages provided. Cite ids like [doc-3]."),
    ("user", "Passages:\n{passages}\n\nQuestion: {question}"),
])

# 2. A parser that fails loudly instead of returning prose.
class Answer(JsonOutputParser):
    """Expects {"answer": str, "citations": [str]}."""

async def answer_question(llm: Any, retriever: Any, question: str) -> dict[str, Any]:
    docs = await retriever.ainvoke(question)
    passages = "\n".join(f"[{d.metadata['id']}] {d.page_content[:400]}" for d in docs)
    messages = ANSWER_PROMPT.format_messages(passages=passages, question=question)
    reply = await llm.ainvoke(messages)
    return Answer().parse(reply.content)
```

Now the loop around it. The registry, the budget and the trace come from Stage 11; LangChain supplies the model adapter and the retriever, nothing more.

```python
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

@dataclass
class Agent:
    llm: Any                                   # any BaseChatModel
    tools: dict[str, Callable[[dict[str, Any]], Awaitable[str]]]
    max_steps: int = 6
    trace_path: str = "runs/langchain.jsonl"
    steps: int = 0
    ledger: set[str] = field(default_factory=set)

    async def run(self, task: str) -> str:
        history: list[dict[str, str]] = [{"role": "user", "content": task}]
        while self.steps < self.max_steps:
            self.steps += 1
            # bind_tools accepts the JSON-schema list your registry already produces
            bound = self.llm.bind_tools([
                {"name": n, "description": f"Run {n}", "parameters": {"type": "object"}}
                for n in self.tools
            ])
            reply = await bound.ainvoke(history)
            with open(self.trace_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "step": self.steps,
                                     "tool_calls": len(reply.tool_calls)}) + "\n")

            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:      # one action per turn: take the first
                key = f"{call['name']}|{json.dumps(call['args'], sort_keys=True)}"
                if key in self.ledger:
                    return "Stopped: repeated identical tool call."
                self.ledger.add(key)
                result = await self.tools[call["name"]](call["args"])
                history.append({"role": "tool", "content": result})
        return "Stopped: step budget exhausted."
```

On the TypeScript side, LangChain.js gives you the same adapters. Most teams use it for the model interface and keep the flow in their own code, which is what this snippet does.

```ts
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

export async function summarise(llm: BaseChatModel, passages: string[], question: string): Promise<string> {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Answer only from the passages. Cite ids."],
    ["user", "Passages:\n{passages}\n\nQuestion: {question}"],
  ]);
  const messages = await prompt.formatMessages({
    passages: passages.join("\n"),
    question,
  });
  const reply = await llm.invoke(messages);
  return typeof reply.content === "string" ? reply.content : JSON.stringify(reply.content);
}
```

## Real use case

A B2B SaaS team had four retrieval sources: Postgres full-text, a vector store, a Notion export and a legacy search API. Writing four clients and normalising their result shapes was a week of work they kept redoing in every service. They adopted LangChain's retriever interface for exactly that layer, with a small adapter class per source, and kept their own agent loop.

The result: adding the fifth source took an afternoon, because it only needed to satisfy one interface. Debugging stayed fast, because a bad answer traced to their own loop code rather than framework internals. They never adopted chains, and said so in code review when a contributor tried.

## Common mistakes

- **Mistake:** putting the agent loop inside a chain builder — **Instead:** keep control flow in your code or in a graph.
- **Mistake:** adopting LangChain to use one provider — **Instead:** a thin wrapper of your own is often clearer.
- **Mistake:** pinning nothing and upgrading weekly — **Instead:** pin versions and read the migration notes before upgrading.
- **Mistake:** hiding retries inside callback/config magic — **Instead:** write retries where the trace can see them.
- **Mistake:** treating the framework's memory classes as your state — **Instead:** own the checkpoint; see [Memory and State](../11-agents/05-memory-and-state.md).

## Checkpoint

1. Wrap a plain HTTP call to your provider as a `BaseChatModel` subclass and run one prompt through it.
2. Build two retriever adapters (one vector store, one SQL) behind the same interface and route between them by query type.
3. Replace a LangChain chain in an existing script with your own loop and compare lines of code, trace quality and latency.

**Self-check:** Exercise 1 — implement `_generate` returning a `ChatResult` with one `ChatGeneration` whose message carries the text; delegate everything else to the base class.

## Go deeper

- [LangGraph](02-langgraph.md) — the orchestration layer from the same family, done as a graph.
- [RAG Architecture](../10-rag/01-rag-architecture.md) — the retrieval shapes these components wrap.
- [Trade-offs](10-tradeoffs.md) — where LangChain sits against the alternatives.
- [Prompt Templates](../08-prompt-and-context/03-templates.md) — template design independent of any framework.

---

[← Previous: Stage 11 — AI Agents](../11-agents/README.md) · [Stage index](README.md) · [Next: LangGraph](02-langgraph.md)
