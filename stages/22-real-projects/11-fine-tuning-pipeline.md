# 11. Fine-Tuning Pipeline

> **Stage 22 · Real Projects** — one repeatable path from raw examples to a deployed adapter, with a gate that blocks any model that is not measurably better than the base.
>
> **You'll be able to:** curate and decontaminate a training set, train a LoRA adapter, compare it against the base model on a held-out eval, and deploy or roll it back without a manual checklist.
>
> **Prerequisites:** [LoRA, QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) · [Fine-Tune Evals](../13-fine-tuning/06-finetune-evals.md)

## Why this exists

Anyone can run a training script. What companies cannot do reliably is answer "is this model better than what we have, and can we prove it, and can we undo it". Teams fine-tune once, get a model that feels better, ship it, and then discover months later that it got worse at the one thing nobody tested — usually refusals, JSON validity, or a rare label.

Fine-tuning also attracts the wrong projects. It fixes *format and style and narrow behaviour*. It does not add knowledge that changes weekly (that is retrieval) and it does not fix reasoning that the base model cannot do. Part of the pipeline's job is to say no early, which is why step one is a baseline eval, not a training run.

Buyers are teams with a repetitive, well-specified task: classification into a fixed taxonomy, extraction into a schema, or rewriting in a house style. The economics are tokens per request and consistency, not novelty.

Mental model: the pipeline is a *ratchet*. Data in, adapter out, eval gate in the middle, and nothing reaches production without beating the previous version on the same frozen test set.

## The mental model

```text
raw logs / human labels / synthetic pairs
        |
   [curate: dedupe, filter, normalise, split]
        |
   [decontaminate: remove eval-set overlap]  <-- blocks silent leakage
        |
   [train: LoRA on base model, seed, config pinned]
        |
   [eval: task metrics + judge + regression suite]
        |
   [gate: beats base on primary AND no regression?]
        |                              |
      no|                              |yes
        v                              v
   [reject, log why]            [deploy adapter vN] --> [canary] --> [rollback if drift]
```

## Core concepts

### Scope and non-goals

In scope: a dataset builder, decontamination, LoRA training with pinned config, an eval harness comparing base and adapter, a gate, adapter versioning, and canary deployment with rollback.

Non-goals: pretraining, full fine-tuning of large models, multi-task model merging, and any automatic data generation loop without human review. Also non-goal: chasing a benchmark — your eval set is the only thing that matters.

### Curate before you train

Most fine-tuning failures are data failures. The minimum bar: exact and near-duplicate removal (hash plus embedding similarity), length filtering, label distribution check, and a deterministic split that never changes. Keep a `source` field per example so you can answer "why is this in the set" later, and cap the share of any single source so one prolific annotator cannot define the model.

Synthetic data is fine as a *supplement*, with two rules: every synthetic example is either checked by a model you trust more or spot-checked by a human, and synthetic share is recorded per example so an eval can be sliced by it. See [Synthetic Data](../13-fine-tuning/03-synthetic-data.md).

### Decontamination is not optional

If three eval questions appear in training, your gate is measuring memorisation. Normalise text (lowercase, collapse whitespace), then compare each training example against the eval set with both an exact hash and a fuzzy match (character n-gram overlap, or cosine on embeddings above a threshold you tune on known-duplicate pairs). Report the number removed. A pipeline that does not do this cannot be trusted with its own numbers.

### The gate is the deliverable

Define the primary metric and the guardrail metrics *before* training. Primary: task accuracy, or a rubric win rate against the base model on paired outputs. Guardrails: JSON validity rate, refusal correctness on a safety set, average output length within a band, and latency. The gate passes only if primary improves by more than your noise band *and* no guardrail regresses. Estimate the noise band by running the same eval twice on the base model — the difference you see is noise, not signal.

### Deploy and roll back like any other artifact

An adapter is a versioned artifact: base model id, adapter weights, config, dataset hash, eval report. Serve it alongside the base, route a small share of traffic to the new one, and compare online metrics (task success, human thumbs, refusal rate) with the eval numbers. Rollback is a config change, not a redeploy. If you cannot roll back in one minute, you are not ready to ship.

### Data model, stack, milestones, budget

| table | key fields |
|-------|-----------|
| `example` | id, task, input, output, source, synthetic, split, hash |
| `dataset_version` | id, name, n_train, n_eval, decontam_removed, git_sha |
| `model_version` | id, base_model, adapter_key, config_json, dataset_version |
| `eval_report` | id, model_version, metrics_json, passed, judge_model |

Stack, loose: `datasets` and `transformers` with `peft` and `trl` for training, one GPU for a small adapter, your eval harness from [Fine-Tune Evals](../13-fine-tuning/06-finetune-evals.md), and an inference server that supports adapters ([vLLM](../16-inference/08-vllm-sglang.md) does). Pin nothing in the README; state the base model family and note that ids change.

Four weeks: week 1 dataset builder and split; week 2 decontamination and baseline eval; week 3 training and the eval harness comparison; week 4 gate, canary, rollback, write-up.

Budget, illustrative: for a few thousand examples on a 7B-class model with LoRA, training takes on the order of a GPU-hour or a few, depending on sequence length and epochs — measure it once and record it. Eval cost is model calls: base plus adapter on the same set. The ongoing cost is serving: adapters are small, so the marginal memory is modest, but every distinct adapter loaded on a server costs memory — cap how many are hot.

## Code

The dataset builder with dedupe and decontamination is the part that determines whether your numbers mean anything.

```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rapidfuzz.fuzz import token_set_ratio


@dataclass
class Example:
    id: str
    task: str
    prompt: str
    completion: str
    source: str
    synthetic: bool = False
    split: str = "train"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_hash(ex: Example) -> str:
    return hashlib.sha256((norm(ex.prompt) + "\x00" + norm(ex.completion)).encode()).hexdigest()


def dedupe(examples: list[Example]) -> tuple[list[Example], int]:
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        h = content_hash(ex)
        if h in seen:
            continue
        seen.add(h)
        out.append(ex)
    return out, len(examples) - len(out)


def decontaminate(train: list[Example], evalset: list[Example], threshold: float = 88.0) -> tuple[list[Example], list[Example]]:
    """Drop training examples that look like an eval question. Report what went."""
    eval_prompts = [norm(e.prompt) for e in evalset]
    kept, removed = [], []
    for ex in train:
        p = norm(ex.prompt)
        if any(token_set_ratio(p, ep) >= threshold for ep in eval_prompts):
            removed.append(ex)
        else:
            kept.append(ex)
    return kept, removed
```

Training stays boring on purpose — the value is in the pinned config and the recorded hashes:

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def train(dataset, dataset_version_id: str):
    cfg = SFTConfig(
        output_dir=f"out/{dataset_version_id}",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=2048,
        seed=1337,                       # fix the seed: reruns must be comparable
        logging_steps=10,
        save_strategy="epoch",
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],   # check names for your architecture
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(model=BASE_MODEL_ID, args=cfg, peft_config=peft_cfg, train_dataset=dataset)
    trainer.train()
    return trainer.save_model(cfg.output_dir)
```

The gate, which is where most teams are missing code rather than ideas:

```python
def gate(report_base: dict, report_new: dict, noise_band: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if report_new["primary"] <= report_base["primary"] + noise_band:
        reasons.append(f"primary {report_new['primary']:.3f} not above base + noise")
    for metric in ("json_validity", "refusal_correct", "length_in_band"):
        if report_new[metric] < report_base[metric]:
            reasons.append(f"guardrail regression: {metric}")
    return (not reasons), reasons
```

## Real use case

A support organisation needs incoming tickets classified into a fixed set of forty categories with a house-specific label vocabulary. Prompting works but drifts: category names get paraphrased, and the long tail is inconsistent.

After: a pipeline trains a small adapter on a few thousand labelled tickets, decontaminates against a held-out set of tickets from the last quarter, and gates on exact-label accuracy plus JSON validity. The first two candidate datasets were rejected by the gate — one because decontamination removed a large share of examples that were near-copies of eval tickets, which is exactly the silent leakage the gate exists to catch.

The design choice that made it work is the guardrail set: the adapter that won on accuracy also had to keep JSON validity and refusal behaviour intact.

Demo script, two minutes: show the dataset report (dedupe and decontamination counts), show base vs adapter on ten examples side by side, show the gate output for a rejected run, then show the canary and the one-command rollback. Talking points: why decontamination matters, how you set the noise band, what fine-tuning does not fix, and adapter versioning.

## Common mistakes

**Mistake:** training before you have a baseline eval — **Instead:** measure the base model first, or you cannot claim an improvement.
**Mistake:** skipping decontamination — **Instead:** dedupe and decontaminate, and report how much was removed.
**Mistake:** judging by eyeballing ten outputs — **Instead:** paired comparison on a frozen set with a noise band.
**Mistake:** using fine-tuning for changing knowledge — **Instead:** retrieval; fine-tuning is for format and narrow behaviour.
**Mistake:** shipping without a rollback path — **Instead:** versioned adapter, canary share, one-line revert.
**Mistake:** letting one annotator or one synthetic generator dominate — **Instead:** cap per-source share and record provenance per example.

## Checkpoint

1. Run your eval twice on the base model and compute the difference. Use it as the noise band in the gate.
2. Add a slice to the eval report for synthetic-share buckets. Does the adapter do worse on real examples?
3. Add a canary: route a small share of traffic to the adapter and compare online task success with the base. What would make you roll back?

**Self-check:** Exercise 1 usually shows a non-zero difference, which is why a gate without a noise band rejects or accepts runs at random.

## Go deeper

- [Fine-Tune vs RAG](../13-fine-tuning/07-finetune-vs-rag.md) — the decision you should make before writing any training code
- [Eval Platform](12-eval-platform.md) — the harness this pipeline gates on
- [vLLM and SGLang](../16-inference/08-vllm-sglang.md) — serving several adapters without duplicating the base
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — automating the gate on every data change

---

[← Previous: Workflow Automation](10-workflow-automation.md) · [Stage index](README.md) · [Next: Eval Platform](12-eval-platform.md)
