# 7. Fine-Tune vs RAG vs Prompting

> **Stage 13 · Fine-Tuning** — the decision that saves you the most money in this entire roadmap.
>
> **You'll be able to:** choose between prompting, retrieval and weight updates using cost, latency and data-freshness, and defend the choice in a design review.
>
> **Prerequisites:** [1. RAG Architecture](../10-rag/01-rag-architecture.md) · [1. Supervised Fine-Tuning (SFT)](01-sft.md)

## Why this exists

Every "the model does not do X" request arrives with fine-tuning as the assumed answer, because fine-tuning sounds like the serious option. Usually it is the most expensive way to solve the problem. Fine-tuning cannot update a fact that changes on Tuesday, cannot cite a source, and cannot be rolled back per user. Retrieval can do all three and ships in days.

But the opposite mistake is just as common: teams bolt a retrieval pipeline onto a problem that is really about behaviour — output format, tone, refusal policy, tool discipline — where the right document does not exist because the answer is not a fact. Then retrieval returns plausible chunks and the model still writes the wrong shape.

The mental model: **retrieval changes what the model can see right now; fine-tuning changes what the model does by default; prompting changes both weakly and cheaply.**

## The mental model

| Axis | Prompting | RAG | Fine-tuning |
|---|---|---|---|
| Time to first version | hours | days | weeks |
| Marginal cost per request | higher (long prompt) | prompt + retrieval + index | lower (short prompt, small model) |
| Latency added | none | retrieval hop | none at inference |
| Data freshness | instant (edit prompt) | as fresh as the index | as fresh as the last training run |
| Can cite sources | no | yes | no |
| Teaches new facts | poorly | yes | poorly, unreliably |
| Fixes format / tone / tool use | partially | no | yes, strongly |
| Rollback granularity | per prompt version | per index | per adapter |
| Ongoing maintenance | prompt edits | ingest pipeline | data, training, evals |
| Failure mode | context dilution, injection | bad chunks, stale index | confidently wrong, regression |

```text
Is the problem "the model does not know / cannot cite"?
        └─ yes ──► RAG. Stop here.
        └─ no  ──► Is it "the model knows but behaves wrong"?
                     └─ yes ──► prompt first. Still wrong after real effort?
                                 └─ yes ──► fine-tune (LoRA).
```

## Core concepts

### Separate knowledge from behaviour, then from economics

Ask three questions in order. Does the model need information it does not have? That is knowledge — retrieval, or continued pretraining if the corpus is huge and static. Does it have the information but produce the wrong shape, tone, or policy? That is behaviour — prompting first, fine-tuning when prompting has plateaued. Does the current solution work but cost too much or run too slow? That is economics — a fine-tuned smaller model with a shorter prompt can be cheaper per request, and that is a legitimate reason to fine-tune even when quality is already acceptable.

### Cost is a per-request number, not a project number

Compare three quantities per 1,000 requests: input tokens, output tokens, and any retrieval calls. A long system prompt is paid on every request forever; a fine-tune moves that instruction into weights, so the prompt shrinks. Illustratively: a 1,500-token system prompt versus a 150-token one is a 1,350-token saving per request, which matters at volume and is irrelevant at 200 requests a day. RAG adds retrieval latency and index cost but keeps prompts short-ish and lets you use a smaller model. Write the arithmetic down with your own token counts and current prices from the provider's docs — never from memory.

### Freshness is the axis people forget

Ask how often the ground truth changes. Product prices change hourly; a company's tone of voice changes yearly. Anything changing faster than you can retrain belongs in retrieval, full stop. A useful hybrid: retrieval for facts, fine-tune for behaviour *about* those facts — a fine-tune that reliably says "I could not find that in the provided documents" instead of guessing, fed by a retrieval pipeline. That combination is often the real answer, and it is the one you can defend.

### The combination is usually the answer

Production systems tend to be prompt + retrieval + small fine-tune, each doing what it is best at: the fine-tune handles format, refusals and tool-call discipline; retrieval supplies current facts with citations; the prompt carries the few things that change weekly. Building all three at once is wrong, but planning for all three is right — design your prompt so the fine-tune can later absorb the stable parts of it.

## Code

A cost model you can run on your own numbers. Put your real token counts in; the point is the shape of the comparison, not these values.

```python
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Plan:
    name: str
    input_tokens: int          # average per request, including system prompt
    output_tokens: int
    retrieval_calls: int = 0
    index_cost_per_1k: float = 0.0     # from your vector DB bill
    train_cost: float = 0.0            # one-off, amortised below

def cost_per_1k(plan: Plan, in_price: float, out_price: float,
                retr_price: float, requests: int = 100_000) -> float:
    """in_price/out_price are currency per 1M tokens; retr_price per 1k calls.
    Check current prices in the provider's docs before trusting any output."""
    per_req = (plan.input_tokens / 1e6) * in_price + (plan.output_tokens / 1e6) * out_price
    per_req += (plan.retrieval_calls / 1e3) * retr_price + plan.index_cost_per_1k / 1e3
    return per_req * 1000 + plan.train_cost / (requests / 1000)

prompt_only = Plan("prompt only", input_tokens=1600, output_tokens=350)
with_rag    = Plan("prompt + rag", input_tokens=900, output_tokens=350,
                   retrieval_calls=1, index_cost_per_1k=0.02)
tuned       = Plan("lora + short prompt", input_tokens=180, output_tokens=350,
                   train_cost=40.0)

for p in (prompt_only, with_rag, tuned):
    print(f"{p.name:22s} {cost_per_1k(p, 3.0, 15.0, 0.4):8.2f} per 1k requests")
# Replace 3.0 / 15.0 / 0.4 with today's prices; the ranking is what you report.
```

And the routing decision as code, so the rule lives next to the service rather than in a slide. This is TypeScript because it typically sits in the API gateway.

```ts
type Strategy = "prompt" | "rag" | "finetune";

interface Signals {
  needsFactsNotInModel: boolean;
  needsCitations: boolean;
  dataChangesFasterThanRetrain: boolean;
  behaviourWrongAfterPrompting: boolean;
  perRequestCostTooHigh: boolean;
}

export function chooseStrategy(s: Signals): Strategy {
  // Fresh, citable facts must come from retrieval — weights cannot keep up.
  if (s.needsCitations || s.dataChangesFasterThanRetrain) return "rag";
  if (s.needsFactsNotInModel) return "rag";
  // Behaviour problems only fall to fine-tuning after prompting plateaued.
  if (s.behaviourWrongAfterPrompting || s.perRequestCostTooHigh) return "finetune";
  return "prompt";
}

const decision = chooseStrategy({
  needsFactsNotInModel: false,
  needsCitations: false,
  dataChangesFasterThanRetrain: false,
  behaviourWrongAfterPrompting: true,
  perRequestCostTooHigh: false,
});
console.log(decision); // "finetune"
```

## Real use case

A fintech support bot must answer questions about transaction disputes. The first proposal is a fine-tune on 10,000 past tickets. Two facts kill it: the dispute policy changed four times last year, and compliance requires every answer to cite the policy clause. A model with that in its weights cannot cite and cannot be updated weekly.

They ship RAG: policy documents chunked by clause, answers must quote a clause id, and the system refuses when retrieval returns nothing above threshold. Two months later the remaining complaint is behavioural — the model quotes three clauses when one is enough, and sometimes answers before retrieving. That is a fine-tune-sized problem: 400 examples of "one clause, then the answer" and "retrieve first, then answer" fix it, on top of the same retrieval pipeline. The decision table earned its keep twice: it stopped an expensive wrong first project, then pointed at the right small second one. Cost per request also fell, because the fine-tune let them drop a 2,000-token instruction block.

## Common mistakes

- **Mistake:** fine-tuning to add facts that change — **Instead:** retrieval; weights cannot cite or update.
- **Mistake:** reaching for RAG to fix output format — **Instead:** prompt, then fine-tune; retrieval will not change shape.
- **Mistake:** comparing project costs instead of per-request costs — **Instead:** compute cost per 1,000 requests with your own token counts.
- **Mistake:** assuming a fine-tune removes the need for retrieval — **Instead:** plan for prompt + retrieval + small adapter as the common end state.
- **Mistake:** quoting prices from memory — **Instead:** read current prices from the provider's docs the day you write the comparison.

## Checkpoint

1. Fill the cost model with your real token counts and current prices for three plans. Report the ranking and the break-even volume.
2. Take three product problems you know and route each through the decision function. Write one sentence of justification per case.
3. Design a hybrid: name which parts of your system prompt the fine-tune should absorb and which must stay dynamic.

**Self-check:** a request to "know our new refund policy, with clause citations" is retrieval — the facts are current and must be cited, and neither property can live in weights.

## Go deeper

- Next: [8. Ship: A LoRA Fine-Tune of a Small Open Model](08-ship-lora-model.md).
- Retrieval in depth: [1. RAG Architecture](../10-rag/01-rag-architecture.md) and [12. RAG Evals](../10-rag/12-rag-evals.md).
- The economics at scale: [8. Cost and Latency](../21-system-design/08-cost-latency.md).

---
[← Previous: 6. Evaluating a Fine-Tune Honestly](06-finetune-evals.md) · [Stage index](README.md) · [Next: 8. Ship: A LoRA Fine-Tune of a Small Open Model](08-ship-lora-model.md)
