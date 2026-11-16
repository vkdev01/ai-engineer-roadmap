# 8. Open-Weight vs Closed Models

> **Stage 06 · LLM Fundamentals** — What "open" actually grants you, what a licence restricts, and how to choose without wishful thinking.
>
> **You'll be able to:** read a model licence for the three things that matter, and justify an open or closed choice in terms your team can act on.
>
> **Prerequisites:** [7. RLHF and DPO](07-rlhf-dpo.md) · [1. OpenAI and Anthropic APIs](../07-llm-apis/01-openai-anthropic.md)

## Why this exists

"Open source model" is used loosely. Three different things get called open, and only one of them is like open-source software:

1. **Open weights** — you can download the trained parameters. The training data, the code and the data pipeline may not be available.
2. **Open weights plus a permissive licence** — Apache 2.0 or MIT, so you can use it commercially, modify it, and redistribute it with few conditions.
3. **Genuinely open** — weights, training code, and often the data mixture, with an OSI-style licence.

Most models called "open" are the first or second. The licence, not the download link, decides what you may build.

The mental model: *open weights buy you control and cost you operations; closed APIs buy you operations and cost you control.*

## The mental model

| Question | Open weights | Closed API |
|---|---|---|
| Who runs the GPU? | you | the provider |
| Data leaves your network? | no | yes, to their endpoint |
| Can you fine-tune? | yes, weights in hand | only if the provider offers it |
| Cost shape | fixed infrastructure, variable utilisation | per token, scales with use |
| Latency | yours to tune | theirs to tune |
| Model can change under you | no — you pinned a version | yes, deprecations happen |
| Ops burden | serving, scaling, monitoring, upgrades | near zero |
| Best quality available? | usually a step behind the frontier | usually the frontier |

The right answer is often **both**: a closed frontier model for the hard 10% of requests, an open small model for the routine 90%. That routing pattern is covered in [4. Model Routing](../21-system-design/04-model-routing.md).

## Core concepts

### Licences differ in three dimensions

Read any model licence for: (a) **commercial use** — allowed, restricted, or capped by a user-count threshold; (b) **redistribution** — may you ship the weights inside your product, or only use them behind an API; (c) **acceptable use** — an attached policy that can prohibit specific applications, which a permissive software licence would not.

Apache 2.0 and MIT are permissive and well understood. Some popular open-weight models ship a **custom community licence** with an acceptable-use policy and, in at least one family, a clause that adds conditions once a product exceeds a large monthly-user threshold. Others ship **terms of use** that restrict redistribution or require you to pass terms downstream. Always read the licence file in the repository; it is short, and the summary blog post is not the licence.

### Weights are not the whole model

You get parameters and an architecture. You usually do not get the data mixture, and you cannot reproduce the training run. That matters for audits: if you must explain *why* a model behaves a certain way, open weights give you the ability to inspect and test, not to reconstruct.

### Serving an open model is a real engineering job

A downloaded checkpoint is not a product. You need a serving engine, batching, quantisation, a GPU or CPU budget, health checks, an autoscaling policy, and an upgrade path when the next version lands. [Stage 16 — Inference](../16-inference/README.md) and [8. vLLM and SGLang](../16-inference/08-vllm-sglang.md) cover that work. Budget for it honestly: the model is free, the serving is not.

### Data residency and privacy usually decide it

If prompts contain regulated data, "the data never leaves our VPC" can be a hard requirement, and that alone selects open weights. Conversely, if you handle no sensitive data and traffic is spiky, paying per token to avoid idle GPUs is straightforwardly correct.

### Pin versions, on both sides

Open weights are pinnable by hash — you control upgrades. Closed models are pinned by model id, and providers deprecate ids. Keep the model id in config, log it with every request, and re-run your evaluation suite before switching. See [5. Prompt Versioning](../08-prompt-and-context/05-prompt-versioning.md).

## Code

A small registry that carries the facts you need to decide, plus a gate that refuses an illegal configuration. This is the file to write before any code that calls a model.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Hosting = Literal["api", "self_hosted", "both"]


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    hosting: Hosting
    licence: str
    commercial_ok: bool
    redistributable: bool         # may you ship the weights in your product?
    finetunable: bool
    notes: str


# Fill licence facts from the model card in the repo, not from memory.
REGISTRY: dict[str, ModelEntry] = {
    "frontier-closed": ModelEntry(
        model_id="check-provider-docs-for-current-id",
        hosting="api", licence="provider terms",
        commercial_ok=True, redistributable=False, finetunable=False,
        notes="no weights; usage governed by provider terms and data policy",
    ),
    "small-open-permissive": ModelEntry(
        model_id="check-model-card-for-current-id",
        hosting="both", licence="Apache-2.0",
        commercial_ok=True, redistributable=True, finetunable=True,
        notes="weights downloadable; you run and scale the serving",
    ),
    "small-open-community": ModelEntry(
        model_id="check-model-card-for-current-id",
        hosting="both", licence="custom community licence",
        commercial_ok=True, redistributable=False, finetunable=True,
        notes="acceptable-use policy applies; read the licence file",
    ),
}


def check(entry: ModelEntry, *, ships_weights: bool, handles_pii: bool) -> list[str]:
    """Return the reasons a configuration is not allowed. Empty means fine."""
    problems: list[str] = []
    if not entry.commercial_ok:
        problems.append(f"{entry.model_id}: licence forbids commercial use")
    if ships_weights and not entry.redistributable:
        problems.append(f"{entry.model_id}: weights may not be redistributed")
    if handles_pii and entry.hosting == "api":
        problems.append(f"{entry.model_id}: prompts leave your network")
    return problems


for name, entry in REGISTRY.items():
    issues = check(entry, ships_weights=True, handles_pii=True)
    print(name, "->", issues or ["ok"])
```

## Real use case

A health-tech company builds a patient-message summariser. They prototype on a closed API, and it works well. Then their compliance review asks two questions: where does the prompt go, and can we show the model's provenance?

The first answer rules out the API for real patient text. They move to an open-weight model that runs inside their own cloud account, accepting a quality drop on the hardest summaries. To keep the quality, they route: routine summaries to the self-hosted model, and de-identified, low-sensitivity documents to the API where it is genuinely better.

The decision cost them two engineering weeks — serving, batching, monitoring — and it was still the right call, because the constraint was regulatory rather than technical. Their mistake in the first place was choosing on quality alone, without asking which of the three licence dimensions applied to them. Writing the registry above, with `handles_pii` as an explicit argument, made that question impossible to skip.

## Common mistakes

- **Mistake:** treating "open weights" as "open source". — **Instead:** check the licence file; weights alone do not grant redistribution or unrestricted use.
- **Mistake:** choosing open weights to save money without costing the GPUs. — **Instead:** estimate serving cost at your real utilisation; idle GPUs are expensive.
- **Mistake:** hardcoding a closed model id with no fallback. — **Instead:** keep it in config with a tested second option.
- **Mistake:** shipping weights inside a product under a licence that forbids it. — **Instead:** run them behind your own API if redistribution is restricted.
- **Mistake:** assuming a provider's data policy matches your compliance needs. — **Instead:** read it, and check whether zero-retention or regional endpoints exist.

## Checkpoint

1. Pick two models you might use, find their licence files, and fill in the three licence dimensions for each. Note which one you had to read the licence to answer.
2. Extend `check` with a `monthly_active_users` argument and a per-entry threshold field, and make it fail when the threshold is exceeded.
3. Sketch the routing rule you would use for a product with 90% routine requests and 10% hard ones. What fraction of traffic would you need to send to the expensive model to keep quality?

**Self-check:** Commercial use, redistribution, and acceptable-use policy. Redistribution is the one most often answered wrong from memory, because a model can be free to use and still not free to ship inside your product.

## Go deeper

- [9. Ship: Tokenizer and Token-Budget Analyser](09-ship-tokenizer.md) — measure cost before you commit to either.
- [8. vLLM and SGLang](../16-inference/08-vllm-sglang.md) — serving an open model properly.
- [4. Model Routing](../21-system-design/04-model-routing.md) — the mixed strategy, designed.
- [9. Ollama and TensorRT](../16-inference/09-ollama-tensorrt.md) — running open weights locally.

---
[← Previous: RLHF and DPO](07-rlhf-dpo.md) · [Stage index](README.md) · [Next: Ship: Tokenizer and Token-Budget Analyser](09-ship-tokenizer.md)
