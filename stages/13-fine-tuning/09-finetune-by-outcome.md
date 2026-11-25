# 9. Fine-Tuning by Outcome: The Notebook Map

> **Stage 13 · Fine-Tuning** — six things you might want a model to do, the training technique each one needs, the data shape it eats, and the notebook that already runs on a free GPU.
>
> **You'll be able to:** pick the right technique from the outcome you want, shape the dataset for it, size the GPU before you rent one, and read an open notebook critically instead of running it blind.
>
> **Prerequisites:** [Supervised Fine-Tuning](01-sft.md) · [LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) · [DPO](05-dpo.md)

## Why this exists

The hard part of fine-tuning is no longer the training loop. A library handles that. The hard part is knowing which of six techniques answers your problem, and every one of them fails differently: supervised fine-tuning on the wrong data teaches a model to sound right and be wrong, reinforcement learning on a lazy reward teaches it to game you, and a preference pass with pairs that differ in the wrong way teaches nothing at all.

The trigger for this lesson is a practical resource: Unsloth's public notebook repository now holds **250+ fine-tuning and RL notebooks** for text, vision, audio, embedding and TTS models — 608 notebook files when this lesson was written — each one runnable in Colab on a free GPU, covering data prep, training and inference. Ved Vekhande's write-up of it made the useful observation that the notebooks are best read by **outcome** rather than by model: "answer in my format", "reason through maths", "call my functions", "prefer the better answer", "find my documents", "read scans or talk".

That is the organising idea here. Each outcome maps to exactly one technique, one data shape and one class of failure. Learn the map and the notebook becomes a worked example of something you already understand, instead of a wall of cells you copy and hope about.

A second reason this matters: these notebooks are how most people meet fine-tuning, and they are **demos**, not recipes. They run 60 steps on a 1,000-row slice. That teaches a format in ten minutes; it does not build a capability. Knowing which part of a notebook is load-bearing and which part is there to keep the free tier alive is the difference between a portfolio project and a wasted weekend.

## Core concepts

### The map

| What you want | Technique | Data shape | Notebook family | Smallest example in the repo |
|---|---|---|---|---|
| Answer in your format and tone | SFT on chat turns, loss on responses only | `messages` (user/assistant) | Conversational | Llama 3.2 1B/3B, Qwen3 14B |
| Reason through maths and puzzles | GRPO, scored by a reward you write | `prompt`, `answer` | GRPO and RL | Gemma 3 1B, Qwen3 4B, gpt-oss 20B |
| Call your functions | SFT on transcripts with tool schemas | `messages`, `tools` | Tool calling | FunctionGemma 270M, Qwen2.5-Coder 1.5B |
| Prefer the better answer | DPO on pairs, or ORPO to drop the reference model | `prompt`, `chosen`, `rejected` | Preference | Zephyr 7B DPO, Llama 3 8B ORPO |
| Find the right documents | Contrastive fine-tune of an embedding model | `anchor`, `positive` | Embedding | Qwen3-Embedding 0.6B, EmbeddingGemma 300M |
| Read scans, transcribe or speak | Vision or audio fine-tune on paired data | `image`/`audio`, `text` | Vision, ASR, TTS | DeepSeek-OCR 3B, Whisper large-v3, Orpheus 3B |

Six rows, six losses. SFT and tool calling share a loss and differ only in the data, which is why they sit next to each other. GRPO optimises a reward. DPO optimises a preference. Embedding training optimises a distance. OCR, ASR and TTS optimise token likelihood on a different input modality, which makes them SFT with a processor in front.

### 1. Format and tone — SFT

**What changes:** the model learns a distribution over your answers — the shape, the register, the length, the refusal style. What it does not reliably learn is new facts; that is what retrieval is for (see [Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md)).

**Data:** chat turns, templated with the model's own chat template.

```json
{"messages": [
  {"role": "system", "content": "You are a support agent for a payments app. Short answers."},
  {"role": "user", "content": "Refund not received, UPI id xyz@okhdfc, 4 days."},
  {"role": "assistant", "content": "UPI refunds land in 3-5 working days. Yours is on day 4, so it should arrive tomorrow. If not, reply here and we will raise a trace with the bank."}
]}
```

**What to set:** LoRA `r=16`, `lora_alpha=16`, learning rate `2e-4`, one to three epochs, sequence length 2048. The one hyperparameter people get wrong is the mask: without `train_on_responses_only` the loss lands on the user's question too, and you spend GPU time teaching the model to write questions.

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
                                         use_gradient_checkpointing="unsloth")
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(dataset_text_field="text", max_steps=60,
                   per_device_train_batch_size=2, learning_rate=2e-4))
trainer = train_on_responses_only(trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")
trainer.train()
```

**Failure mode:** the model reproduces your data's flaws at higher volume. If 30% of your answers say "kindly revert", you will ship a model that says it every time. Curate, then train; see [Synthetic Data](03-synthetic-data.md) for the filters.

### 2. Reasoning — GRPO

**What changes:** nothing about the weights' knowledge. GRPO shapes *how* the model spends its tokens: it samples several completions per prompt, scores each with a reward function you write, and pushes probability toward the ones that scored. Group-relative means the comparison is against the other samples of the same prompt, so you never need a value model.

**Data:** a prompt and a checkable answer. Maths, puzzles, unit tests, SQL that runs, a grid that must be valid.

```json
{"prompt": "A shop sells pens at 12 rupees. Ravi buys 7 and pays with 100. How much change?", "answer": "16"}
```

**What to set:** `num_generations=4` to start, `per_device_train_batch_size=1`, a low learning rate around `5e-6`, and 50-100 steps to look at behaviour. Memory scales with generations, not just batch: four rollouts means four sequences in flight per prompt.

```python
import re
from trl import GRPOConfig, GRPOTrainer

def format_reward(completions, **kwargs):
    return [0.2 if re.search(r"<answer>.*?</answer>", c, re.S) else 0.0 for c in completions]

def answer_reward(completions, answer, **kwargs):
    out = []
    for completion, truth in zip(completions, answer):
        found = re.search(r"<answer>(.*?)</answer>", completion, re.S)
        out.append(1.0 if found and found.group(1).strip() == truth.strip() else 0.0)
    return out

trainer = GRPOTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
    reward_funcs=[format_reward, answer_reward],
    args=GRPOConfig(num_generations=4, per_device_train_batch_size=1,
                    max_steps=100, learning_rate=5e-6))
```

The repository's notebooks follow the same shape with several small rewards — exact format match, approximate format match, answer match, number extraction — which is a good default: one reward for the shape you need, one for being right, and nothing for length.

**Failure mode:** reward hacking, and it is not subtle. Give 0.2 for a format tag and the model learns to emit the tag and nothing else. Read the completions, not just the reward curve. The repository's Sudoku and 2048 notebooks are worth opening precisely because their rewards are hard to game: the board either solves or it does not.

### 3. Tool calling — SFT on transcripts

**What changes:** the model learns to emit a structured call, with the right name and the right arguments, from a schema it is shown at inference time.

**Data:** the schema, the user turn, the call, and the result. Include the failure turn — tool error, then a corrected call — or the model never learns what to do when an argument is wrong.

```json
{"tools": [{"name": "get_order_status",
            "parameters": {"order_id": "string, format ORD-####"}}],
 "messages": [
   {"role": "user", "content": "Where is ORD-1042?"},
   {"role": "assistant", "content": "<tool_call>get_order_status(order_id='ORD-1042')</tool_call>"},
   {"role": "tool", "content": "in transit, Pune hub, 2 days late"},
   {"role": "assistant", "content": "It is at the Pune hub and running two days late."}]}
```

**What to set:** LoRA `r=128` with `lora_alpha=256` is what the FunctionGemma 270M notebook uses, because tool syntax is a narrow skill that needs capacity in the adapter rather than the base weights. A few hundred steps at batch 4 is enough for a demo.

**Failure mode:** the model learns your tool names but not your argument formats, because your data always showed well-formed ids. Hold out entire tool schemas in the eval, not just examples: a model that only works on the schemas it trained on has memorised, not learned. Then it is a lookup table with extra steps.

### 4. Preference — DPO and ORPO

**What changes:** relative taste. You give a prompt with a chosen and a rejected answer; the model is pushed to raise the likelihood of the chosen one relative to a frozen reference copy of itself.

**Data:** the pair, and the pair must differ in the way you care about — accuracy, or length, or hedging, but one axis at a time.

```json
{"prompt": "Explain a KV cache to a new backend engineer.",
 "chosen": "A KV cache stores the keys and values of past tokens, so each new token attends to memory instead of re-reading the prompt. It trades GPU memory for latency; that is why long contexts are expensive.",
 "rejected": "It caches things to make it faster."}
```

**What to set:** `beta=0.1` is the usual start, batch 2, learning rate `5e-6`. Budget roughly double the weights, because the reference model sits frozen next to the one you train. ORPO merges the SFT and preference signals into one pass and drops the reference model entirely, which is the cheaper choice when you are also teaching a format.

```python
from unsloth import PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOConfig, DPOTrainer

trainer = DPOTrainer(model=model, ref_model=None, tokenizer=tokenizer, train_dataset=pairs,
    args=DPOConfig(beta=0.1, per_device_train_batch_size=2, max_steps=60, learning_rate=5e-6))
```

With a PEFT model, `ref_model=None` is correct: the adapter is switched off to produce the reference logits, so you do not load a second copy.

**Failure mode:** pairs where the rejected answer is obviously bad. The model learns "long and specific beats short and vague" and nothing about correctness. Rejected answers should be the plausible wrong ones your current model actually produces — which means generating them from your model, not writing them by hand.

### 5. Retrieval — fine-tuning an embedding model

**What changes:** where your documents sit in vector space. A general embedding model puts "how long does a UPI refund take" near generic finance text; a fine-tuned one puts it near your refund policy.

**Data:** query and positive passage pairs, ideally with hard negatives — passages that look right and are not. In the repository, this is a `datasets` split with `anchor` and `positive` columns.

```json
{"anchor": "how long does a upi refund take",
 "positive": "UPI refunds are credited within 3-5 working days of the original transaction."}
```

**What to set:** LoRA `r=32`, `lora_alpha=32`, batch 256 with `BatchSamplers.NO_DUPLICATES` so no batch is all one topic, and a few hundred steps. Then re-embed the corpus — a fine-tuned encoder with an old index is worse than no fine-tune at all, because the query and document spaces no longer agree.

```python
from unsloth import FastSentenceTransformer
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

model = FastSentenceTransformer.from_pretrained("unsloth/Qwen3-Embedding-0.6B", max_seq_length=512)
model = FastSentenceTransformer.get_peft_model(model, r=32, lora_alpha=32)
trainer = SentenceTransformerTrainer(model=model, train_dataset=pairs,
    args=SentenceTransformerTrainingArguments(output_dir="out", per_device_train_batch_size=256,
        batch_sampler=BatchSamplers.NO_DUPLICATES, max_steps=100))
```

**Failure mode:** measuring with cosine similarity on a handful of pairs you picked yourself. Use recall@10 and nDCG over a held-out query set, and compare against the base model on the same set; see [Evaluating a Fine-Tune](06-finetune-evals.md).

### 6. Documents and voice — OCR, ASR, TTS

**What changes:** the input modality. All three are the same supervised loss on paired data, with a processor that turns pixels or waveforms into tokens.

- **OCR:** `(image, transcription)`. The repository trains DeepSeek-OCR 3B with `FastVisionModel`, `r=16`, on a synthetic OCR dataset, and keeps a separate evaluation notebook — which is the right instinct, because OCR quality is per document type and a single average hides everything.
- **ASR:** Whisper large-v3 through `FastModel`, trained with `Seq2SeqTrainer`. One detail is load-bearing: Whisper needs `task_type=None` in `get_peft_model`, or the adapter attaches to the wrong layers and training quietly does nothing useful.
- **TTS:** a language model over codec tokens. The Orpheus 3B notebook turns waveforms into discrete audio codes, trains on those, and redistributes codes back to audio at inference. There is no metric that replaces listening to ten samples.

```python
from unsloth import FastModel
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

model, tokenizer = FastModel.from_pretrained("unsloth/whisper-large-v3",
    auto_model=WhisperForConditionalGeneration, load_in_4bit=True)
model = FastModel.get_peft_model(model, r=32, task_type=None)  # None is required for Whisper
trainer = Seq2SeqTrainer(model=model, train_dataset=dataset, args=Seq2SeqTrainingArguments(
    output_dir="out", per_device_train_batch_size=2, max_steps=60, predict_with_generate=True))
```

**Failure mode:** training on clean, centred, high-resolution scans and deploying on phone photos. The distribution shift lives in the camera, not the language, so the training set has to contain your worst inputs.

### The rung before fine-tuning

Fine-tuning is the third rung, and the first two are cheaper:

| Symptom | Fix first | Why not fine-tuning |
|---|---|---|
| The model does not know our policy | Retrieval | Facts change; weights do not |
| The output format is wrong | Prompt, schema, constrained decoding | A JSON schema beats a training run |
| The model is too slow or too expensive | A smaller model, distillation, caching | Fine-tune only after the smaller model fails on quality |
| The task is a fixed style across thousands of calls | Fine-tuning | Style is exactly what SFT is good at |
| One narrow behaviour, many attempts | Fine-tuning, or a hook in the harness | Cheap, measurable, and it generalises |

### Reading the notebooks honestly

Four things to keep in mind when you open one of these notebooks:

- **60 steps is a demo.** It teaches a format in minutes. A capability takes thousands of steps and a curated set; the repository's own defaults are chosen to fit a free GPU, not to ship.
- **The slices are tiny.** `split="train[:1000]"` is a smoke test. Your data pipeline is the actual work, and it is the part a notebook does not give you.
- **Model ids move.** The repository lists Gemma 4, Qwen3.5, gpt-oss 20B, FunctionGemma 270M, Qwen3-Embedding 0.6B and others; ids, sizes and licences change, so check the current list and the model card before you pin one.
- **Check the licence.** The notebooks are LGPL-3.0 and the models carry their own terms. Copying a notebook into a product is a licensing decision, not just a technical one.

## Code

A runnable decision layer: it holds the map, checks that a sample of your data has the right shape for the outcome you chose, and sizes the GPU before you rent one. No torch, no GPU, no downloads.

```python
"""Pick a fine-tuning recipe by outcome, check your data has the right shape, size the GPU.

    python finetune_recipes.py                 # the whole map
    python finetune_recipes.py preference      # one recipe, with the data checks

No torch, no GPU, no downloads: this file is the decision layer that sits in front of the
Unsloth notebooks, so you know which notebook you need before you open one.
"""
from __future__ import annotations

import sys
from typing import Any

RECIPES: dict[str, dict[str, Any]] = {
    "chat": {
        "goal": "Answer in your format and tone",
        "technique": "Supervised fine-tuning on chat turns, loss on responses only",
        "trainer": "trl.SFTTrainer + unsloth.chat_templates.train_on_responses_only",
        "data_keys": {"messages"},
        "example_model": "unsloth/Llama-3.2-3B-Instruct",
        "params_b": 3.0, "extra_weight_copies": 0,
        "vram_note": "1B-3B fits a free T4 at 4-bit; 14B needs a 24GB card",
        "watch": ["one epoch is often enough", "learning rate 2e-4 for LoRA, not 2e-5",
                  "mask the prompt tokens or you teach the model to write questions"],
    },
    "reasoning": {
        "goal": "Reason through maths and puzzles",
        "technique": "GRPO: reinforcement learning scored by a reward function you write",
        "trainer": "trl.GRPOTrainer + GRPOConfig(num_generations=4)",
        "data_keys": {"prompt", "answer"},
        "example_model": "unsloth/Gemma-3-1B-it",
        "params_b": 1.0, "extra_weight_copies": 0,
        "vram_note": "rollouts multiply memory: batch x num_generations sequences live at once",
        "watch": ["the reward must be cheap and deterministic",
                  "reward hacking: check the passing answers by hand, not just the score",
                  "start at 50-100 steps and look at the completions"],
    },
    "tools": {
        "goal": "Call your functions with the right arguments",
        "technique": "SFT on transcripts that contain tool schemas, calls and results",
        "trainer": "trl.SFTTrainer + tokenizer.apply_chat_template(tools=...)",
        "data_keys": {"messages", "tools"},
        "example_model": "unsloth/FunctionGemma-270M-it",
        "params_b": 0.27, "extra_weight_copies": 0,
        "vram_note": "270M-4B: trains on a free T4, and 270M can run on a phone",
        "watch": ["include the failure turns (tool error, then a corrected call)",
                  "hold out tool schemas, not just examples, to test generalisation",
                  "score exact tool name and argument match separately"],
    },
    "preference": {
        "goal": "Prefer the better of two answers",
        "technique": "DPO on (prompt, chosen, rejected) triples, or ORPO to skip the reference model",
        "trainer": "trl.DPOTrainer + DPOConfig(beta=0.1); unsloth.PatchDPOTrainer()",
        "data_keys": {"prompt", "chosen", "rejected"},
        "example_model": "unsloth/zephyr-sft-bnb-4bit",
        "params_b": 7.0, "extra_weight_copies": 1,  # the frozen DPO reference model
        "vram_note": "DPO holds a frozen reference model too: budget roughly double the weights",
        "watch": ["chosen and rejected must differ in the way you care about",
                  "beta 0.1 is a start; too low and the model ignores the preference",
                  "rejected answers should be plausible, not garbage"],
    },
    "embedding": {
        "goal": "Find the right documents in your own domain",
        "technique": "Contrastive fine-tune of an embedding model on (query, positive) pairs",
        "trainer": "SentenceTransformerTrainer + BatchSamplers.NO_DUPLICATES",
        "data_keys": {"anchor", "positive"},
        "example_model": "unsloth/Qwen3-Embedding-0.6B",
        "params_b": 0.6, "extra_weight_copies": 0,
        "vram_note": "0.3B-0.6B trains on a free T4 at batch 256 with no duplicates per batch",
        "watch": ["hard negatives matter more than more pairs",
                  "re-embed the whole corpus after training, or the index is stale",
                  "measure recall@10 and nDCG, never cosine similarity on one pair"],
    },
    "ocr": {
        "goal": "Read scans, forms and handwriting",
        "technique": "Vision-language fine-tune on (image, transcription) pairs",
        "trainer": "trl.SFTTrainer + FastVisionModel",
        "data_keys": {"image", "text"},
        "example_model": "unsloth/DeepSeek-OCR",
        "params_b": 3.0, "extra_weight_copies": 0,
        "vram_note": "3B vision at 4-bit fits a 24GB card; crop-heavy pages cost more tokens",
        "watch": ["character error rate by document type, not one global number",
                  "keep a sample of clean digital text in the training set",
                  "check orientation and resolution handling before you train"],
    },
    "speech": {
        "goal": "Transcribe audio, or speak in a chosen voice",
        "technique": "ASR: Seq2Seq fine-tune of Whisper. TTS: language-model fine-tune on codec tokens",
        "trainer": "Seq2SeqTrainer (Whisper, task_type=None) / Trainer (TTS codec tokens)",
        "data_keys": {"audio", "text"},
        "example_model": "unsloth/whisper-large-v3",
        "params_b": 1.55, "extra_weight_copies": 0,
        "vram_note": "large-v3 at 4-bit needs a 24GB card; small models fit a T4",
        "watch": ["Whisper LoRA needs task_type=None, or the adapter is attached to the wrong layer",
                  "word error rate by accent and channel, not one average",
                  "for TTS, listen to samples; a metric will not catch a bad voice"],
    },
}


def check_data(outcome: str, rows: list[dict[str, Any]]) -> list[str]:
    """Return the problems in a sample of rows, empty when the shape is right."""
    recipe = RECIPES[outcome]
    problems: list[str] = []
    if not rows:
        return ["no rows"]
    for i, row in enumerate(rows):
        missing = recipe["data_keys"] - row.keys()
        if missing:
            problems.append(f"row {i}: missing {sorted(missing)}")
            continue
        if outcome == "preference":
            if row["chosen"] == row["rejected"]:
                problems.append(f"row {i}: chosen == rejected, no signal")
            if len(row["chosen"]) < 10:
                problems.append(f"row {i}: chosen answer too short to learn from")
        if outcome == "chat":
            roles = {m.get("role") for m in row["messages"]}
            if not {"user", "assistant"} <= roles:
                problems.append(f"row {i}: needs a user turn and an assistant turn, got {sorted(roles)}")
        if outcome == "tools":
            if not row["tools"]:
                problems.append(f"row {i}: tools is empty, the model cannot learn a schema it never sees")
            if not any(m.get("role") == "tool" for m in row["messages"]):
                problems.append(f"row {i}: no tool result turn, so the model never sees the observation")
        if outcome == "reasoning":
            if not row["prompt"].strip().endswith(("?", ".")):
                problems.append(f"row {i}: prompt looks truncated")
    return problems


def vram_gb(params_b: float, bits: int = 4, adapters_pct: float = 0.01,
            rollouts: int = 1, extra_weight_copies: int = 0) -> dict[str, float]:
    """A rough floor, not a guarantee: weights, LoRA optimiser states, a rollout allowance."""
    weights = params_b * bits / 8 * (1 + extra_weight_copies)
    adapter_params = params_b * adapters_pct
    adapter_states = adapter_params * 12 / 1  # bf16 weight + bf16 grad + 8 bytes AdamW
    activations = 0.20 * params_b * rollouts
    return {"weights": round(weights, 2), "adapter_states": round(adapter_states, 2),
            "activations": round(activations, 2),
            "floor": round(weights + adapter_states + activations, 2)}


SAMPLE: dict[str, list[dict[str, Any]]] = {
    "chat": [{"messages": [{"role": "user", "content": "Where is order ORD-1042?"},
                           {"role": "assistant", "content": "It left the Pune hub this morning."}]}],
    "reasoning": [{"prompt": "What is 18% of 250?", "answer": "45"}],
    "tools": [{"tools": [{"name": "get_order_status", "parameters": {"order_id": "string"}}],
               "messages": [{"role": "user", "content": "Status of ORD-1042?"},
                            {"role": "assistant", "content": "calling get_order_status(order_id='ORD-1042')"},
                            {"role": "tool", "content": "in transit, Pune hub"}]}],
    "preference": [{"prompt": "Explain a KV cache to a new backend engineer.",
                    "chosen": "A KV cache stores the keys and values of past tokens so each new token "
                              "does not re-read the whole prompt. It trades GPU memory for latency.",
                    "rejected": "It is a cache."}],
    "embedding": [{"anchor": "how long does a UPI refund take", "positive": "Refunds land in 3-5 working days"}],
    "ocr": [{"image": "prescription_0412.png", "text": "Tab Metformin 500mg BD x 30 days"}],
    "speech": [{"audio": "call_0091.wav", "text": "The payment failed twice, please check."}],
}


def main() -> None:
    wanted = sys.argv[1:] or list(RECIPES)
    for name in wanted:
        recipe = RECIPES[name]
        print(f"\n{name.upper()} — {recipe['goal']}")
        print(f"  technique : {recipe['technique']}")
        print(f"  trainer   : {recipe['trainer']}")
        print(f"  data keys : {sorted(recipe['data_keys'])}")
        print(f"  model     : {recipe['example_model']}")
        problems = check_data(name, SAMPLE[name])
        print(f"  data check: {'ok' if not problems else problems}")
        est = vram_gb(recipe["params_b"],
                      rollouts=4 if name == "reasoning" else 1,
                      extra_weight_copies=recipe["extra_weight_copies"])
        print(f"  vram floor at 4-bit: {est}")
        for note in recipe["watch"]:
            print(f"  watch     : {note}")


if __name__ == "__main__":
    main()
```

Output, trimmed to two recipes:

```text
CHAT — Answer in your format and tone
  technique : Supervised fine-tuning on chat turns, loss on responses only
  trainer   : trl.SFTTrainer + unsloth.chat_templates.train_on_responses_only
  data keys : ['messages']
  model     : unsloth/Llama-3.2-3B-Instruct
  data check: ok
  vram floor at 4-bit: {'weights': 1.5, 'adapter_states': 0.36, 'activations': 0.6, 'floor': 2.46}

PREFERENCE — Prefer the better of two answers
  technique : DPO on (prompt, chosen, rejected) triples, or ORPO to skip the reference model
  trainer   : trl.DPOTrainer + DPOConfig(beta=0.1); unsloth.PatchDPOTrainer()
  data keys : ['chosen', 'prompt', 'rejected']
  model     : unsloth/zephyr-sft-bnb-4bit
  data check: ok
  vram floor at 4-bit: {'weights': 7.0, 'adapter_states': 0.84, 'activations': 1.4, 'floor': 9.24}
```

The estimator is deliberately blunt: 4-bit weights, LoRA optimiser states at 1% of parameters, and a fixed allowance for activations. It is a floor, not a promise — a real run adds attention buffers, fragmentation and the tokenizer's own working set. The point is the arithmetic you can do on a napkin: a 7B DPO run carries two copies of the weights, so it needs roughly twice what a 7B SFT run needs, which is why the 20B GRPO notebooks ask for an A100 and the 270M tool-calling notebook runs on a phone.

## Real use case

A diagnostics lab in Hyderabad digitises handwritten prescriptions. Two-thirds arrive as phone photos: mixed English and Devanagari, uneven light, a doctor's handwriting that no OCR service reads well. The off-the-shelf transcription missed drug names and doses, which is the only part that matters.

They fine-tuned DeepSeek-OCR 3B on 2,000 labelled pages, weighted toward the worst inputs — the dim photos and the scribbled doses — and held out 300 pages by clinic, not at random, so the eval measured generalisation to a clinic the model had never seen. Character error rate on the held-out set went from 18% to 4.2%, and drug-name accuracy from 71% to 96%. They kept a rule in the harness: any prescription with a confidence below a threshold or an unrecognised drug name goes to a human, because a 4% error rate on a dose is not a number you accept silently.

The same map fixed a second problem in the same company. Their document search ran on a general embedding model and missed Hindi-English mixed queries. A contrastive fine-tune of a 0.6B embedding model on 40,000 query-passage pairs from their own search logs — plus hard negatives taken from the top results that users skipped — moved recall@10 from 61% to 84%. The corpus was re-embedded overnight; the index from the day before was thrown away, which is the step people forget.

Two different outcomes, two different techniques, one map, and in both cases the eval came before the training run.

## Common mistakes

- **Mistake:** fine-tuning to add facts the model should look up — **Instead:** retrieval for facts, fine-tuning for behaviour; see [Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md).
- **Mistake:** training on the whole conversation, loss included on the user turns — **Instead:** mask with `train_on_responses_only` and confirm by printing one decoded batch.
- **Mistake:** running a 20B notebook on a free T4 and calling fine-tuning broken — **Instead:** do the napkin arithmetic first; GRPO multiplies memory by `num_generations`.
- **Mistake:** a format-only reward in GRPO — **Instead:** one reward for shape, one for correctness, and read twenty completions before trusting the curve.
- **Mistake:** preference pairs where the rejected answer is garbage — **Instead:** generate rejected answers from your current model; the model should learn taste, not the difference between an answer and a shrug.
- **Mistake:** fine-tuning an embedding model and leaving the old index in place — **Instead:** re-embed the corpus in the same job that ships the adapter, and keep the two versions apart.
- **Mistake:** evaluating on prompts that look like the training data — **Instead:** hold out by source — a clinic, a customer, a document type — and report per slice, not one average.
- **Mistake:** copying a notebook's hyperparameters into production — **Instead:** keep the recipe, replace the numbers; `max_steps=60` exists to fit a free tier.

## Checkpoint

1. Run `finetune_recipes.py`, then change the `chat` sample to a row with only an assistant turn. Confirm the data check catches it, and add a check for an assistant turn that is shorter than ten characters.
2. Take the outcome you actually care about and write ten real data rows for it in that shape. Run `check_data` on them. Fix whatever it flags before you open a notebook.
3. Size your run: use `vram_gb` to compare a 7B SFT run and a 7B DPO run, then add `rollouts=4` for a GRPO run on the same model. Write down which of the three fits a 24GB card.
4. Open one notebook from the repository in your outcome's family and answer three questions in writing: what does the loss land on, what does the reward (if any) measure, and what would you change to train on your own data for a thousand steps.

**Self-check:** Exercise 3 — 7B SFT is about 4.3GB of floor, 7B DPO about 9.2GB because the reference weights are a second copy, and 7B GRPO with four rollouts lands between them on weights but grows with every generation. Only the SFT and GRPO runs fit comfortably on 24GB; the DPO run fits only with a smaller batch or a shorter sequence.

## Go deeper

- [Supervised Fine-Tuning](01-sft.md) — the loss, the chat template and prompt masking.
- [LoRA, QLoRA and PEFT](02-lora-qlora-peft.md) — rank, alpha and target modules, and the memory maths.
- [Synthetic Data for Fine-Tuning](03-synthetic-data.md) — generating and filtering training rows.
- [DPO: Preference Tuning Without RLHF](05-dpo.md) — the loss in plain words, and pair construction.
- [Evaluating a Fine-Tune Honestly](06-finetune-evals.md) — held-out sets, regression checks, judge comparisons.
- [Fine-Tune vs RAG vs Prompting](07-finetune-vs-rag.md) — the decision table this lesson's third rung comes from.
- [Ship: A LoRA Fine-Tune of a Small Open Model](08-ship-lora-model.md) — the artifact end to end.
- [Multimodal AI](../14-multimodal/README.md) — OCR, speech and vision in depth.
- The notebooks: [unslothai/notebooks](https://github.com/unslothai/notebooks) (LGPL-3.0) and the [Unsloth docs](https://unsloth.ai/docs). The outcome-first framing comes from Ved Vekhande's [post](https://lnkd.in/p/d7EtjGrR), November 2026.

---

[← Previous: Ship: A LoRA Fine-Tune of a Small Open Model](08-ship-lora-model.md) · [Stage index](README.md) · [Next stage: Multimodal](../14-multimodal/README.md)
