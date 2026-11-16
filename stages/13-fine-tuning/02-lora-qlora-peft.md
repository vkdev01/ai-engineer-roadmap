# 2. LoRA, QLoRA and PEFT

> **Stage 13 · Fine-Tuning** — makes fine-tuning fit on one consumer or rented GPU instead of a cluster.
>
> **You'll be able to:** explain what a low-rank update is, pick rank and alpha deliberately, choose which modules to adapt, and size the GPU memory you need before you start.
>
> **Prerequisites:** [1. Supervised Fine-Tuning (SFT)](01-sft.md) · [9. Transfer Learning](../04-deep-learning/09-transfer-learning.md)

## Why this exists

Full fine-tuning updates every weight. For a model with several billion parameters that means storing the weights, the gradients and the optimiser state — optimiser state alone is often several times the weight size for Adam. Your laptop cannot do it, and a single rented GPU usually cannot either.

LoRA starts from an observation about *what a fine-tune changes*: the update to a weight matrix tends to be low-rank — a small number of directions carry almost all of the change. So instead of learning a full `d×d` update, learn two thin matrices whose product is that update. You freeze the original weights, train the thin ones, and at inference add the product back. Same behaviour, a tiny fraction of the trainable parameters, and the original model stays on disk untouched.

The mental model: **LoRA bolts a small trainable detour onto a frozen road.** The road is unchanged; you only learn where the detour goes.

## The mental model

```text
frozen W (d x d)                 trainable detour
   x ──► W ──► +  ──► y
                ▲
   x ──► A (d x r) ──► B (r x d) ──┘      r << d
        W' = W + (alpha / r) * B @ A
```

| Variant | Base weights | Trainable | Typical GPU need |
|---|---|---|---|
| Full fine-tune | fp16/bf16, all updated | 100% | many |
| LoRA | frozen, fp16/bf16 | adapter only | one mid-range |
| QLoRA | frozen, 4-bit quantised | adapter only | one small |

## Core concepts

### The low-rank update, in one formula

For a weight matrix `W` of shape `d×d`, LoRA learns `A` of shape `d×r` and `B` of shape `r×d`, with `r` much smaller than `d`, and uses `W' = W + (α/r)·BA`. Trainable parameters drop from `d²` to `2dr`. Worked example, illustrative: `d = 4096`, `r = 16` → full matrix is about 16.8M parameters, the adapter is about 131k, roughly 0.8% of that matrix. `B` is initialised to zero so the adapter starts as an exact no-op — training begins from the base model's behaviour, never from noise.

### Rank and alpha are one knob wearing two hats

`r` is the adapter's capacity; `alpha` scales the update. What the model actually feels is the ratio `α/r`. A common habit is to set `alpha = 2r` (so `r=16, α=32`) and tune `r` only; if you double `r`, double `alpha` to keep the effective scale constant. Small `r` (8-16) is usually enough for style and format changes; larger `r` (32-64) helps when the task is genuinely new. Raising `r` past the point where eval loss stops improving just adds parameters and overfitting risk.

### Which modules to target

Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are the default and the safest choice. Adding the MLP projections (`gate_proj`, `up_proj`, `down_proj`) roughly triples the adapter size and often helps on tasks needing new knowledge or heavier reasoning. Never adapt embeddings or the output head unless you have a vocabulary reason (new tokens) — they are huge and rarely the bottleneck. Check the module names with `model.named_modules()`; they differ by architecture.

### Memory maths you can do on paper

Rough budget, illustrative for a 7B model: frozen bf16 weights ≈ 14 GB; LoRA adapter parameters are negligible; gradients only for adapter; Adam state for adapter only. So LoRA fits in roughly 16-20 GB with activations. QLoRA loads the frozen base in 4-bit: `7B × 0.5 bytes ≈ 3.5 GB`, plus small dequantisation overhead, plus adapter and activations — often under 8 GB. Activation memory scales with batch size × sequence length, so when you run out of memory, cut sequence length or use gradient checkpointing (slower, less memory) before blaming the adapter. These are estimates for planning; measure with `torch.cuda.max_memory_allocated()`.

### Merging, and why you might not

`merge_and_unload()` folds `BA` into `W`, producing a normal model with no adapter overhead and slightly faster inference. You lose the ability to swap adapters. Keep the adapter unmerged if you serve many tenants or tasks from one base model — load the base once, attach the adapter per request. Serving details are in [Stage 16 — Inference](../16-inference/README.md).

## Code

Define the adapter config explicitly so every number is visible, then check how many parameters you are actually training.

```python
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL_ID = "some-small-open-model"  # verify the id on the model card

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

lora = LoraConfig(
    r=16,                          # capacity of the detour
    lora_alpha=32,                 # effective scale is alpha/r = 2.0
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()
# Expect something like: trainable 0.x% || trainable 1-4M || all 1-8B
```

QLoRA is the same config plus a quantised load. `prepare_model_for_kbit_training` casts norms to fp32 and enables gradient checkpointing so 4-bit training is stable.

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # normal-float 4bit, the QLoRA default
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # quantise the quantisation constants too
)

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
base = prepare_model_for_kbit_training(base)
model = get_peft_model(base, lora)      # same LoraConfig as above

print(torch.cuda.max_memory_allocated() / 1e9, "GB peak so far")
```

## Real use case

A hospital-network intake team needs to turn free-text referral notes into a fixed set of fields — specialty, urgency, language, insurance scheme. A large hosted model does it well but the notes are patient data, so they must stay in their own cloud. Full fine-tuning of a 7B model needs a GPU class their budget does not cover.

They run QLoRA on one 24 GB GPU: base in 4-bit, rank 16 on attention projections, about 2,000 synthetic-plus-real examples. The adapter file is a few tens of megabytes, so they version it in the same registry as their code and can roll back. Before, they were using a prompted 3B model that leaked free text into the urgency field and misread Hindi-English mix; after, field-level exact match improves and the model stops inventing specialties that the hospital does not have — because the adapter was trained on their actual specialty list. The design choice that made it work was keeping the base frozen and unmerged: one base model on one GPU serves four hospital tenants with four small adapters.

## Common mistakes

- **Mistake:** raising `r` to fix bad outputs — **Instead:** fix the data first; most LoRA failures are label noise or unmasked prompts.
- **Mistake:** changing `r` without changing `alpha` — **Instead:** keep `alpha = 2r` while you tune `r`, so one variable moves.
- **Mistake:** targeting every linear layer "for quality" — **Instead:** start with attention projections, measure, then add MLP if eval loss plateaus.
- **Mistake:** QLoRA with a high learning rate and no gradient checkpointing — **Instead:** use `prepare_model_for_kbit_training`, bf16 compute, and expect to need a lower LR than LoRA on bf16.
- **Mistake:** assuming low memory means fast — **Instead:** 4-bit adds dequantisation cost per step; QLoRA trades speed for fitting at all.

## Checkpoint

1. For `d = 2048` and `r = 8`, compute adapter parameters per adapted matrix and the percentage of the original matrix. Show your arithmetic.
2. Train two adapters, `r=8` and `r=64`, on the same 300 examples. Compare eval loss and adapter file size; write one sentence on which you would ship.
3. Merge an adapter into its base, then compare peak memory and tokens per second against serving the adapter unmerged.

**Self-check:** each adapted `d×d` matrix goes from `d²` to `2dr` trainable parameters, so for `d=2048, r=8` that is 4,194,304 down to 32,768 — 0.78% of the matrix.

## Go deeper

- Next: [3. Synthetic Data for Fine-Tuning](03-synthetic-data.md) — where the examples come from.
- Quantisation in depth: [6. Quantization](../16-inference/06-quantization.md).
- Serving many adapters: [8. vLLM and SGLang](../16-inference/08-vllm-sglang.md).

---
[← Previous: 1. Supervised Fine-Tuning (SFT)](01-sft.md) · [Stage index](README.md) · [Next: 3. Synthetic Data for Fine-Tuning](03-synthetic-data.md)
