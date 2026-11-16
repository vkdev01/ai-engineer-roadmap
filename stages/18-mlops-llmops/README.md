# Stage 18 — MLOps and LLMOps

Stage 17 gave you a backend that serves model calls. This stage gives you the machinery that keeps that backend honest once real users touch it: what you logged, what you can replay, what you can roll back, and what you are spending. It sits here because observability only makes sense after you have a service to observe.

**Time:** 3-4 weeks of evenings · **You need first:** Stage 17 — AI Backend · **You leave with:** a trace dashboard that reports latency, cost, errors and eval scores per prompt version

## What you will be able to do

- Log params, prompts, datasets, scores and cost for a run, and compare runs side by side in MLflow or W&B.
- Instrument an LLM service so every request becomes one trace with a span per model and tool call, and read those traces in Langfuse or LangSmith.
- Treat a prompt as a versioned artefact with a registry entry, a diff and a rollback path.
- Version datasets and model files with DVC so a run can be reproduced from a git commit alone.
- Put an eval gate in a pull request so a prompt change cannot merge if quality drops.
- Attribute spend per feature and per tenant, and alert before a budget is breached.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Experiment Tracking with MLflow and W&B](01-mlflow-wandb.md) | Runs, params, metrics, artefacts and what LLM work adds |
| 2 | [LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) | Traces, spans, sessions and user feedback in production |
| 3 | [Prompt Versioning and Registries](03-prompt-versioning.md) | Prompts as diffable, deployable, revertible artefacts |
| 4 | [DVC for Data and Models](04-dvc.md) | Datasets and weights tied to commits, not to laptops |
| 5 | [Tracing Design](05-tracing.md) | One trace per request, span per call, sampling that still tells the truth |
| 6 | [Cost Monitoring and Budgets](06-cost-monitoring.md) | Per-feature and per-tenant spend, alerts, unit economics |
| 7 | [CI/CD for Evals](07-ci-cd-evals.md) | Eval gates in PRs, thresholds, and flaky test handling |

## The shipping project

[08-ship-trace-dashboard.md](08-ship-trace-dashboard.md) — you build a service that ingests trace records from your own AI app and a dashboard that breaks latency, cost, error rate and eval score down by prompt version. You prove it works by replaying a recorded week of traces, then shipping a deliberately worse prompt version and watching the dashboard catch it.

## How to know you are done

- You can take a trace id from a user complaint and land on the exact prompt version, model, token counts and cost that produced the bad answer.
- `git log` plus DVC pointers is enough to rebuild the dataset a reported run used, on a machine that has never seen your laptop.
- A pull request that lowers eval scores is blocked by CI with a readable diff, and a pull request that raises them merges.
- Your dashboard answers "what did feature X cost per successful request last week" without you opening a billing console.
- You can name the rollback step for a bad prompt release and it takes one command.

---

[← Back to roadmap](../../README.md) · [Next stage: Cloud and DevOps](../19-cloud-devops/README.md)
