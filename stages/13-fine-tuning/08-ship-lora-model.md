# 8. Ship: A LoRA Fine-Tune of a Small Open Model

> **Stage 13 · Fine-Tuning** — the stage project: everything a fine-tune needs to be shippable, not just trainable.
>
> **You'll be able to:** take a task from raw examples to a versioned LoRA adapter with a dataset card, a reproducible training config, a before/after eval and a model card.
>
> **Prerequisites:** [2. LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) · [6. Evaluating a Fine-Tune Honestly](06-finetune-evals.md)

## Why this exists

A trained adapter on your laptop is not a deliverable. Anyone who inherits it needs to know what data it saw, what the numbers were against what baseline, what it must not be used for, and how to rebuild it. Without those four things the adapter is an unexplainable binary that nobody will dare deploy, and it will be retrained from scratch by the next person.

This project builds the whole artifact around a small task of your choosing. The task matters less than the packaging: pick something with verifiable outputs so your eval is honest — classification into a fixed label set, extraction into a fixed schema, or rewriting to a house style with a rule-checkable constraint.

The mental model: **four files ship together — data, config, eval report, model card. The adapter alone is not the deliverable.**

## The mental model

```text
data/            train.jsonl  valid.jsonl  test.jsonl (locked)  DATASET_CARD.md
config/          lora_sft.yaml            (every number, justified)
eval/            baseline.json  tuned.json  report.md   (base vs tuned, same rows)
adapters/run1/   adapter_config.json  adapter_model.safetensors
MODEL_CARD.md    intended use, data, metrics, limits, how to reproduce
```

| Artifact | Must answer |
|---|---|
| Dataset card | where the data came from, how filtered, what is in the held-out set |
| Training config | rank, alpha, LR, epochs, batch, template, seed |
| Eval report | task metric + general battery + pairwise, base vs tuned |
| Model card | intended use, known failures, out-of-scope use |

## Core concepts

### The dataset card is a bias disclosure, not a changelog

Write it before training. It states: source of each slice, collection method, licence and consent, the placeholder scheme for PII, the filters and how many rows each removed, the split policy (by source, not by row), the label distribution, and known gaps. "Known gaps" is the section reviewers actually read: if 90% of your examples are English and your users are not, that belongs on the card, not in your head.

### The config must be a file, and it must carry a seed

Every number that affects the result goes in `lora_sft.yaml`: model id and revision, template, `r`, `alpha`, dropout, target modules, learning rate, epochs, batch and accumulation, max length, warmup, seed, and the exact data file hashes. Then anyone can rerun and get a comparable adapter. Seed does not make training bit-identical on GPUs, but it removes one source of unexplained variance.

### The eval report is a table with three rows, always

Base model, prompt-only baseline, your adapter — same held-out rows, same metric code. Plus the general battery delta and the pairwise result with ties. Put the CI next to every difference and add the sentence "differences whose interval crosses zero are reported as no measurable change." That one convention stops most overclaiming.

### The model card closes the loop with limits

The limits section is where you earn trust: what the model does badly, the input lengths and languages it was not trained on, and the explicit out-of-scope list ("do not use for medical advice", "not evaluated on children's data"). Name at least two observed failure modes from your own eval, with an example. A card that claims no weaknesses is a card nobody believes.

## Code

The training entry point: one YAML in, one adapter out, hashes recorded. Run it with `python train.py config/lora_sft.yaml`.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

cfg_path = Path("config/lora_sft.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

data_files = {split: Path(p) for split, p in cfg["data"].items()}
ds = load_dataset("json", data_files={k: str(v) for k, v in data_files.items()})

lora = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    target_modules=cfg["lora"]["target_modules"],
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["train"]["epochs"],
    per_device_train_batch_size=cfg["train"]["batch_size"],
    gradient_accumulation_steps=cfg["train"]["grad_accum"],
    learning_rate=cfg["train"]["lr"],
    max_length=cfg["train"]["max_length"],
    seed=cfg["train"]["seed"],
    train_on_inputs=False,          # loss on answers only
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to=[],
)

trainer = SFTTrainer(
    model=cfg["model"]["id"],
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    peft_config=lora,
)
trainer.train()
trainer.save_model(cfg["output_dir"])

# Provenance: the adapter is only reproducible with the data it saw.
provenance = {
    "config_sha": sha256(cfg_path),
    "data_sha": {k: sha256(v) for k, v in data_files.items()},
    "model": cfg["model"],
}
Path(cfg["output_dir"], "provenance.json").write_text(json.dumps(provenance, indent=2))
print(json.dumps(provenance, indent=2))
```

The eval report generator. It runs three systems over the same locked rows and writes a markdown table with a paired bootstrap interval, so the report cannot drift from the numbers.

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def bootstrap_ci(base: list[int], tuned: list[int], n: int = 10_000) -> tuple[float, float, float]:
    d = np.asarray(tuned, float) - np.asarray(base, float)
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)

test = [json.loads(line) for line in Path("data/test.jsonl").read_text().splitlines()]
scores = {name: [int(pred == ex["label"]) for pred, ex in zip(preds, test)]
          for name, preds in json.loads(Path("eval/preds.json").read_text()).items()}

base, tuned = scores["base"], scores["adapter"]
delta, lo, hi = bootstrap_ci(base, tuned)
verdict = "measurable difference" if (lo > 0 or hi < 0) else "no measurable change"

Path("eval/report.md").write_text(
    "| System | Correct | Accuracy |\n|---|---|---|\n"
    + "".join(f"| {k} | {sum(v)}/{len(v)} | {sum(v)/len(v):.3f} |\n" for k, v in scores.items())
    + f"\nAdapter minus base: {delta:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}) — {verdict}.\n"
)
```

Finally the serving call, from the app side. The base model id and adapter path come from config; keep the chat template in sync with training or the adapter will look broken.

```ts
interface GenerateBody {
  prompt: string;
  maxTokens?: number;
}

export async function generate(body: GenerateBody): Promise<string> {
  const res = await fetch(`${process.env.INFER_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.INFER_KEY}`,
    },
    body: JSON.stringify({
      model: process.env.ADAPTER_ID, // adapter name served on the base model
      messages: [{ role: "user", content: body.prompt }],
      max_tokens: body.maxTokens ?? 512,
      temperature: 0,
    }),
  });
  if (!res.ok) throw new Error(`inference failed: ${res.status}`);
  const data = (await res.json()) as { choices: { message: { content: string } }[] };
  return data.choices[0].message.content;
}
```

## Real use case

A regional e-commerce team ships a "product description normaliser": sellers submit messy descriptions, and the output must be three fields — material, care instructions, size guidance — or an explicit `unknown`. They collect 1,400 seller listings, strip phone numbers and addresses, hold out two entire categories, and fine-tune a small open model with rank 16 on attention projections.

The first eval reported only the adapter's field accuracy and looked excellent. Adding the prompted base on the same rows showed the prompt-only baseline was close on material and care, and better on size guidance — so they shipped the adapter for the two fields it won and kept the prompt for the third, in one pipeline. The general battery flagged a drop in Hindi listings; they added 120 Hindi examples and re-ran. The artifact they handed over was four files plus a one-page model card listing "hallucinates care symbols on leather goods" as a known failure with an example. Six months later a new engineer retrained it from the config and hashes without asking anyone a question — which is the actual success criterion for this project.

## Common mistakes

- **Mistake:** shipping the adapter with no eval report — **Instead:** the report is part of the deliverable; no report, no merge.
- **Mistake:** a config that lives in a notebook cell — **Instead:** one YAML with every number, plus data hashes in `provenance.json`.
- **Mistake:** reporting only your adapter's score — **Instead:** three rows: base, prompt-only, adapter, same rows.
- **Mistake:** a model card that only lists strengths — **Instead:** at least two observed failure modes with examples and an out-of-scope list.
- **Mistake:** training with one chat template and serving with another — **Instead:** pin the tokenizer revision and template string in config; assert at serve time.

## Checkpoint

1. Write the dataset card for your data before training, including a "known gaps" section with at least three real gaps.
2. Produce the three-row eval report and state which rows differ measurably and which do not.
3. Write the model card, then hand it to someone who did not build the model and ask them to list what it must not be used for. If they cannot, rewrite the limits section.

**Self-check:** four artifacts ship together — the dataset card, the training config with data hashes, the three-row eval report, and the model card — and the adapter alone is not a deliverable.

## Go deeper

- Next stage: [Stage 14 — Multimodal](../14-multimodal/README.md).
- Serve the adapter properly: [8. vLLM and SGLang](../16-inference/08-vllm-sglang.md).
- A bigger version of this project: [11. Fine-Tuning Pipeline](../22-real-projects/11-fine-tuning-pipeline.md).
- The eval harness around it: [10. Ship: An Eval Suite](../15-evals/10-ship-eval-suite.md).

---
[← Previous: 7. Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md) · [Stage index](README.md) · [Next: Stage 14 — Multimodal](../14-multimodal/README.md)
