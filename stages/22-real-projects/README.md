# Stage 22 — Real Projects

Stages 1 to 21 gave you parts: an API client, a retriever, an agent loop, an eval suite, a deployment. This stage gives you the thing employers actually ask about — systems that a stranger can run, that fail in ways you predicted, and that carry a number saying how well they work. Each file is a build brief, not a tutorial: a user problem, a scope, an architecture, a milestone plan, the one hard snippet, and an eval plan.

**Time:** 10-16 weeks (one project at a time, evenings and weekends) · **You need first:** Stage 21 — System Design · **You leave with:** three finished projects with evals, cost budgets and a two-minute demo each

## What you will be able to do

- Take a vague product idea and turn it into a scoped brief with explicit non-goals
- Design a data model and an eval plan before writing the first prompt
- Estimate cost and latency per request from token counts, not vibes
- Name the failure modes of your own system and point at the code that handles each
- Demo a working system in two minutes and defend its trade-offs for twenty
- Ship a repo whose README lets a reviewer understand the project without you

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Research Assistant](01-research-assistant.md) | Search, read, synthesise — with verifiable citations |
| 2 | [Document Intelligence](02-document-intelligence.md) | Scanned PDFs into validated structured JSON |
| 3 | [Production RAG](03-production-rag.md) | Multi-tenant retrieval that stays fresh and evaluated |
| 4 | [Multi-Agent Research](04-multi-agent-research.md) | A supervisor with specialist workers and shared memory |
| 5 | [Support Agent](05-support-agent.md) | Tool side effects, refunds, and clean human handoff |
| 6 | [Coding Assistant](06-coding-assistant.md) | Repo context, patch generation, sandboxed tests |
| 7 | [Multimodal Assistant](07-multimodal-assistant.md) | Images, audio and documents in one interface |
| 8 | [Meeting Intelligence](08-meeting-intelligence.md) | Transcription, diarisation, decisions, action items |
| 9 | [Knowledge Assistant](09-knowledge-assistant.md) | Answers over a company wiki with permissions |
| 10 | [Workflow Automation](10-workflow-automation.md) | Events to agents to actions, with approvals |
| 11 | [Fine-Tuning Pipeline](11-fine-tuning-pipeline.md) | Data, training, eval, deploy — one repeatable path |
| 12 | [Eval Platform](12-eval-platform.md) | Datasets, graders, dashboards, regression gates |
| 13 | [AI SaaS](13-ai-saas.md) | Auth, billing, quotas, tenants and cost control |

## The shipping project

There is no separate ship file here — the projects *are* the artifact. Build three, not thirteen. A good trio that covers most interview ground: [Production RAG](03-production-rag.md) (retrieval, tenancy, evals), [Support Agent](05-support-agent.md) (tools, side effects, handoff) and [Eval Platform](12-eval-platform.md) (measurement, dashboards). Each ships with a working demo, a README with a results table, and an eval you can rerun.

## How to know you are done

- Three repos exist, each with a README that states the problem, the architecture, and a measured result
- Each project has an eval script that runs offline and prints numbers you can compare across changes
- You can state the cost per 1,000 requests and the p95 latency of your own system without looking
- Each README lists the failure modes you found and what the code does about them
- A reviewer who has never spoken to you can run the demo from the README alone

---

[← Back to roadmap](../../README.md) · [Next stage: Career](../23-career/README.md)
