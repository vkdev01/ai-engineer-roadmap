# 4. Instruction Tuning vs Continued Pretraining

> **Stage 13 · Fine-Tuning** — the fork in the road: teach the model a new behaviour, or teach it a new body of text.
>
> **You'll be able to:** tell which of the two you need from the symptom you are seeing, and run a continued-pretraining pass without destroying the model's ability to follow instructions.
>
> **Prerequisites:** [1. Supervised Fine-Tuning (SFT)](01-sft.md) · [5. Pretraining](../06-llm-fundamentals/05-pretraining.md)

## Why this exists

"I want the model to know our product" and "I want the model to answer in our format" sound like the same request and are not. One is a knowledge problem, the other a behaviour problem, and the fix is different in data, cost and risk.

Continued pretraining (CPT) keeps the same next-token objective but on a large corpus from your domain — standards, manuals, code, filings. Instruction tuning uses prompt-answer pairs. CPT changes what the model has read; instruction tuning changes what it does when asked. Pick wrong and you either waste a GPU-month learning nothing useful, or you watch a model confidently invent facts in a beautiful new format.

The mental model: **CPT adds reading material; instruction tuning adds a job description.**

## The mental model

| Symptom | Likely missing | Fix |
|---|---|---|
| Right answer, wrong shape or tone | behaviour | instruction tuning |
| Refuses or rambles instead of using a tool | behaviour | instruction tuning |
| Does not know domain vocabulary at all | knowledge | CPT, or retrieval |
| Knows words, cannot reason over them | both, often | CPT on documents, then instruction tuning |
| Facts change monthly | freshness | retrieval, not either |

```text
base model ──► CPT (raw domain text, next token) ──► instruction tuning (pairs) ──► your model
                    thousands of docs                        hundreds of examples
```

## Core concepts

### The objective is the tell

Both loops are cross-entropy on the next token, but the data shape differs. CPT takes long documents and supervises *every* token — the model learns to predict the domain's own prose. Instruction tuning takes short turns and supervises only the answer. So CPT needs volume (documents, millions of tokens) and instruction tuning needs precision (hundreds to thousands of verified pairs). If your corpus is 300 pages, CPT will barely move the model and may overfit it; that corpus is better used as a retrieval index.

### CPT has a forgetting problem, and it is real

Training on one narrow corpus pushes the model's output distribution toward that corpus. Do it carelessly and the model starts answering in the style of a legal standard, loses general instruction-following, or degrades on other languages. Mitigations that actually help: mix a share of general instruction data or general web text into every batch (a rehearsal ratio), use a low learning rate, and evaluate on general tasks every N steps — not just at the end. If general scores fall more than you can accept, stop and lower the rate or the domain share.

### CPT + instruction tuning is the usual pairing

The pipeline for a genuinely new domain is: CPT on unlabelled domain text to absorb vocabulary and structure, then instruction tuning on a small set of pairs to restore the job description. Order matters — instruction tuning last, so the behaviour is what the model ends with. Skipping the second step gives you a model that talks about the domain but ignores your request format; skipping the first gives you a model that formats beautifully and knows nothing.

### Token budget, honestly

There is no standard "enough" figure. Plan in tokens rather than documents: an illustrative CPT pass might see tens of millions of domain tokens over one or two epochs. Below roughly a million tokens, expect marginal change for the cost and prefer retrieval or instruction tuning. Before committing, run a cheap probe: fine-tune a tiny model on the same corpus and see whether held-out domain perplexity drops at all.

## Code

A CPT loop that mixes domain text with rehearsal data and logs domain perplexity alongside a small general-ability check. The key line is the mixture: never let a batch be 100% domain.

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "some-small-open-model"   # check the model card for the current id
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.train()

def pack(text: str, max_len: int = 2048) -> dict[str, torch.Tensor]:
    """Pack a document into a fixed window; every token is supervised (CPT)."""
    ids = tokenizer(text, truncation=True, max_length=max_len,
                    return_tensors="pt")["input_ids"][0]
    return {"input_ids": ids, "labels": ids.clone()}   # NO masking in CPT

def mixed_batch(domain: list[str], rehearsal: list[str], domain_share: float = 0.8):
    n_dom = int(round(8 * domain_share))
    batch = [pack(t) for t in domain[:n_dom]] + [pack(t) for t in rehearsal[: 8 - n_dom]]
    return tokenizer.pad(batch, padding=True, return_tensors="pt")

opt = torch.optim.AdamW(model.parameters(), lr=2e-5)   # low LR: protect general ability
dl = DataLoader(list(range(200)), batch_size=1)        # one step = one mixed batch

for step, _ in enumerate(dl):
    batch = mixed_batch(domain_corpus, general_corpus)
    out = model(**batch)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        print(step, "domain loss", float(out.loss))    # log general eval separately
```

The probe before you spend real money: compare held-out domain perplexity of the base model against a lightly trained one. If it barely moves, CPT is not your answer.

```python
@torch.no_grad()
def perplexity(model, texts: list[str]) -> float:
    total, count = 0.0, 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
        loss = model(ids, labels=ids).loss          # mean NLL over the window
        total += float(loss) * ids.shape[1]; count += ids.shape[1]
    return float(torch.exp(torch.tensor(total / count)))
```

## Real use case

A diagnostics lab wants a model that reads its own assay reports. They have 4,000 PDFs — enough text, but no labelled question-answer pairs. First attempt: they instruction-tune on 300 examples that quote report excerpts, and the model learns the *format* of an answer while inventing reference ranges it never saw. It sounds right and is dangerous.

They switch to the two-step pipeline. CPT on the extracted report text with a 20% rehearsal mix of general data, low learning rate, checking general instruction-following every 100 steps. Then instruction tuning on the same 300 pairs. After: the model recognises assay codes and their units without being told, and the instruction step restores the "answer in two sentences, then a table" format. The design choice that made it work was evaluating general ability during CPT — at their first learning rate, general scores dropped sharply by step 200 and they halved it. The failure before was not a data problem; it was choosing the wrong objective for the gap they had.

## Common mistakes

- **Mistake:** CPT on a small corpus and calling it domain knowledge — **Instead:** probe with perplexity first; under about a million tokens, use retrieval.
- **Mistake:** CPT with 100% domain batches — **Instead:** mix rehearsal data every batch and evaluate general ability every N steps.
- **Mistake:** instruction tuning to fix missing facts — **Instead:** name the gap as knowledge or behaviour before choosing the objective.
- **Mistake:** instruction tuning before CPT in a new domain — **Instead:** CPT first, instruction tuning last, so behaviour is the final state.
- **Mistake:** using full fine-tuning learning rates for CPT — **Instead:** a low rate with gradient clipping, and stop on general-ability regression.

## Checkpoint

1. Classify five real requests from your own context as knowledge or behaviour gaps, and name the fix for each.
2. Run the perplexity probe on a 200-page corpus with a tiny model, before and after 50 steps. Report the delta.
3. Design a rehearsal mixture for a corpus that is 90% Hindi legal text, and say what you would put in the other 10% and why.

**Self-check:** "the model refuses to output our JSON schema" is a behaviour gap, so instruction tuning — the model already understands the request, it just has not been trained on the response format.

## Go deeper

- Next: [5. DPO: Preference Tuning Without RLHF](05-dpo.md) — the step after instruction tuning.
- Background: [5. Pretraining](../06-llm-fundamentals/05-pretraining.md).
- When retrieval is the better answer: [1. RAG Architecture](../10-rag/01-rag-architecture.md).

---
[← Previous: 3. Synthetic Data for Fine-Tuning](03-synthetic-data.md) · [Stage index](README.md) · [Next: 5. DPO: Preference Tuning Without RLHF](05-dpo.md)
