# Stage 12 — Orchestration

Once an agent loop has more than two branches, the loop stops fitting in your head. This stage is about the frameworks that give that structure a name — nodes, edges, state, checkpoints — and, just as importantly, about knowing when a framework is costing you more than it saves.

Read it as a comparison, not a shopping list. Every framework here solves the same underlying problem: moving typed state through a graph of steps, with retries, persistence and human approval. The differences are in how much they hide from you.

**Time:** 2 weeks · **You need first:** [Stage 11 — AI Agents](../11-agents/README.md) · **You leave with:** a LangGraph flow with persisted state, conditional routing and an approval interrupt

## What you will be able to do

- Read a framework's abstractions and map them onto the plain loop you already wrote
- Build a state graph with nodes, conditional edges, checkpoints and interrupts
- Choose between LangGraph, LlamaIndex, Haystack, Pydantic AI, DSPy and a role-based framework for a given job
- Program prompts as modules and optimise them with DSPy instead of hand-tuning strings
- Argue the trade-offs of a framework versus 200 lines of your own code — with numbers
- Draw your system on a whiteboard in 30 seconds, with a reason for every edge

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [LangChain](01-langchain.md) | What it is good for, and where it gets in the way |
| 2 | [LangGraph](02-langgraph.md) | State, nodes, edges, checkpoints, interrupts |
| 3 | [Graph Engineering](03-graph-engineering.md) | Designing the graph before you write it |
| 4 | [LlamaIndex](04-llamaindex.md) | The data-centric framework: indexes and query engines |
| 5 | [Haystack](05-haystack.md) | Pipelines, components and typed connections |
| 6 | [Semantic Kernel](06-semantic-kernel.md) | The .NET-adjacent option, and its plugin model |
| 7 | [Pydantic AI](07-pydantic-ai.md) | Agents whose outputs are typed and validated |
| 8 | [DSPy](08-dspy.md) | Prompts as programs, compiled against a metric |
| 9 | [CrewAI and AutoGen](09-crewai-autogen.md) | Role-based and conversational multi-agent |
| 10 | [Trade-offs](10-tradeoffs.md) | An honest comparison, and when to write your own loop |
| 11 | [Ship: A LangGraph Flow](11-ship-langgraph-flow.md) | State, routing, persistence and approval in one graph |

## The shipping project

[Ship: A LangGraph Flow](11-ship-langgraph-flow.md) — you rebuild the harness agent from Stage 11 as a state graph: a router node that sends cheap requests down a short path, a retrieval node, a tool node, a critic node, and an approval interrupt before anything with a side effect. State is persisted so a crashed run resumes, and the graph is drawn in the README.

## How to know you are done

- You can rebuild any framework example in this stage as a plain Python loop, and say what the framework added
- Your graph has a conditional edge you actually needed, not one added for symmetry
- A run can be killed mid-execution and resumed from the last checkpoint
- You have a written answer to "why not just write a while-loop?" for your own project
- You have tried one optimisation-based tool (DSPy) and can say whether it beat your hand-written prompt on your data

---
[← Back to roadmap](../../README.md) · [Next stage: Fine-Tuning](../13-fine-tuning/README.md)
