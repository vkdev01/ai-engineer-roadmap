# 1. Experiment Tracking with MLflow and W&B

> **Stage 18 · MLOps and LLMOps** — the run log that turns "it worked yesterday" into a queryable record of every attempt you made.
>
> **You'll be able to:** log params, prompts, datasets, scores and cost for an LLM pipeline, then compare runs and pick the winner on evidence rather than memory.
>
> **Prerequisites:** [Stage 17 — AI Backend](../17-ai-backend/README.md) · [Eval Suites](../15-evals/10-ship-eval-suite.md)

## Why this exists

You change a prompt, run your eval set, and see the score go from 0.71 to 0.78. You change it again, get 0.74, and now you cannot remember which of the two edits caused the difference, or whether the judge model was the same, or whether the dataset had the two broken rows you deleted last Tuesday. Nothing is wrong with your code. You simply have no record.

A spreadsheet is the first thing people try. It fails for one reason: the interesting facts are not scalars. A prompt is multi-line text, a dataset is a file with a hash, an eval run is a list of per-example scores. You need a store that takes arbitrary artefacts, ties them to a run, and lets you diff runs.

The mental model: **a run is an immutable row, and everything about that attempt hangs off it.** Params go on the row, metrics go on the row, files and text go into the artefact store keyed by the row.

## The mental model

```text
experiment: "support-rag"
|-- run a1f3  (git 9c2e, 2026-03-04)
|     params : model=gpt-4.1-mini, temperature=0.2, top_k=5
|     prompt : hash 8ac1  -> artefact  prompts/answer.v3.txt
|     data   : dvc hash 4b77 (golden set, 240 rows)
|     metrics: faithfulness 0.71, p95_latency_ms 1180, cost_usd 0.0031
|     files  : per_example.jsonl, confusion.png
`-- run b902  (git 41ad, 2026-03-05)
      params : ... top_k=8, prompt hash c40f (answer.v4.txt)
      metrics: faithfulness 0.78, p95_latency_ms 1540, cost_usd 0.0042
```

Comparing `a1f3` and `b902` is now a query, not a memory test.

## Core concepts

### Run, param, metric, artefact

A run is one execution of your pipeline you want to remember. Params are inputs you chose (model id, temperature, top_k, prompt hash, dataset hash). Metrics are numbers you measured, and MLflow treats them as a time series, so you can log `loss` every 10 steps and still plot it. Artefacts are files: the rendered prompt, the per-example eval JSONL, a chart.

The rule that keeps this useful: log the **inputs that determine the output** as params, including hashes of things too big to inline. If two runs differ in behaviour and their params are identical, you have an unlogged input, and that is a bug in your instrumentation.

### What LLM work must log that classic ML did not

Classic training runs have one output: the model. An LLM pipeline's "model" is prompt plus model id plus retrieval config plus tools, and it changes every week without any training at all. So add three things: the **rendered prompt text** (not just the template, because variable values change behaviour), the **model id with version date**, and **token counts and cost**. Without the last two you cannot answer "was the better score worth 40% more spend", which is the question your manager will actually ask.

### W&B vs MLflow in one paragraph each

MLflow is open source, runs locally as a server or on a SQLite file, and is the default when you want experiment tracking to live next to the rest of your infra with no vendor. W&B is hosted, its UI is stronger for comparing many runs and for sharing dashboards, and it is priced per seat and tracked hours. Both have Python clients with the same shape: start a run, log params, log metrics, log artefacts. Pick one per repo and do not mix, or your comparisons split across two stores.

### Comparing runs is the whole point

A tracker you only write to is a very expensive log file. The value appears when you filter: "runs in the last 14 days where faithfulness > 0.75 and cost_usd < 0.005, sorted by p95 latency". Learn the filter syntax of your tool early; it is the difference between having data and having answers.

## Code

This logs one eval run to MLflow. Note that the prompt and dataset are stored as hashes plus artefacts, and cost is computed from tokens, not guessed. `MLFLOW_TRACKING_URI` points at your server; the default writes to `./mlruns` which is fine for one machine.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlflow
from pydantic import BaseModel


class EvalRow(BaseModel):
    question: str
    answer: str
    score: float


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def log_eval_run(
    *,
    name: str,
    prompt_template: str,
    dataset_path: Path,
    model_id: str,          # check provider docs for current ids
    temperature: float,
    top_k: int,
    rows: list[EvalRow],
    input_tokens: int,
    output_tokens: int,
    price_in_per_1k: float,  # pass real numbers from the provider's price page
    price_out_per_1k: float,
) -> str:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("support-rag")

    dataset_text = dataset_path.read_text()
    cost = (input_tokens / 1000) * price_in_per_1k + (output_tokens / 1000) * price_out_per_1k
    mean_score = sum(r.score for r in rows) / len(rows)

    with mlflow.start_run(run_name=name) as run:
        # Inputs you chose, including hashes of the big ones.
        mlflow.log_params(
            {
                "model_id": model_id,
                "temperature": temperature,
                "top_k": top_k,
                "prompt_hash": sha(prompt_template),
                "dataset_hash": sha(dataset_text),
                "dataset_rows": len(rows),
            }
        )
        # Outputs you measured.
        mlflow.log_metrics(
            {
                "mean_score": mean_score,
                "min_score": min(r.score for r in rows),
                "cost_usd": round(cost, 6),
                "cost_per_example": round(cost / len(rows), 6),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        # The exact text and the per-example detail, so a run is replayable.
        mlflow.log_text(prompt_template, "prompt.txt")
        mlflow.log_dict({"rows": [r.model_dump() for r in rows]}, "per_example.json")
        return run.info.run_id
```

Second snippet: comparing two runs by reading them back. This is the query you will run a hundred times, so put it in a script.

```python
from __future__ import annotations

import mlflow


def compare(experiment: str, run_ids: list[str]) -> list[dict[str, object]]:
    client = mlflow.MlflowClient()
    out: list[dict[str, object]] = []
    for rid in run_ids:
        run = client.get_run(rid)
        row: dict[str, object] = {"run_id": rid[:8], **run.data.params, **run.data.metrics}
        out.append(row)
    return out


if __name__ == "__main__":
    for r in compare("support-rag", ["a1f3...", "b902..."]):
        print(r["run_id"], r["mean_score"], r["cost_usd"], r["top_k"])
```

## Real use case

A support team ships a RAG assistant over 4,000 help-centre articles. Two engineers take turns improving the answer prompt. Without tracking, "the new prompt is better" is an argument between two people with different memories of last week's run. With tracking, the disagreement becomes a table: v3 has faithfulness 0.71 at roughly 0.003 per answer with `top_k=5`; v4 has 0.78 at roughly 0.004 with `top_k=8`. The extra retrieval is what bought the gain, not the prompt wording, and the table shows it because both runs logged `top_k`.

The failure mode before was concrete: a prompt that looked better in a demo was worse on the long-tail questions in the golden set, and nobody could prove it, so the worse version shipped for three weeks. Logging the per-example JSONL meant the first time it happened after tracking, the engineer opened the artefact and found the regression in twelve rows.

## Common mistakes

**Mistake:** logging only the prompt template name, not the rendered prompt — **Instead:** log the hash of the template and the fully rendered text of at least one example, because variable values are where behaviour changes.

**Mistake:** logging a score with no record of the judge model or judge prompt — **Instead:** log judge model id and judge prompt hash as params, or a score from last month is not comparable to today's.

**Mistake:** tracking metrics but never opening the UI to compare — **Instead:** write the `compare()` script in week one and use it in every review; unused tracking is pure cost.

**Mistake:** logging cost as a hardcoded constant per call — **Instead:** compute from logged token counts and a price table stored in config, so a provider price change is one edit.

**Mistake:** putting datasets and model weights in the run's artefact store as the primary copy — **Instead:** keep them in DVC or object storage and log only the hash; see [DVC for Data and Models](04-dvc.md).

## Checkpoint

1. Take your Stage 15 eval script and make it log one MLflow run with params, three metrics, and the per-example JSONL as an artefact. Confirm you can see it in the UI.
2. Add cost: count tokens with your provider's tokenizer or the SDK's usage field, multiply by a price table loaded from YAML, and log `cost_usd` and `cost_per_example`.
3. Run the same prompt twice with only `temperature` changed and write a one-paragraph note on which metrics moved, using `compare()` output rather than the UI.

**Self-check:** exercise 1 is done when `mlflow ui` lists a run whose params include a prompt hash and a dataset hash, and whose artefact list contains `per_example.json`.

## Go deeper

- [LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) — the production-side counterpart to offline experiment tracking.
- [Prompt Versioning and Registries](03-prompt-versioning.md) — how the prompt hash in your params becomes a deployable artefact.
- [CI/CD for Evals](07-ci-cd-evals.md) — turning the comparison into an automated gate.
- [Eval Suites](../15-evals/10-ship-eval-suite.md) — the scores you are logging come from here.

---

[← Previous: Stage 17 — AI Backend](../17-ai-backend/README.md) · [Stage index](README.md) · [Next: LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md)
