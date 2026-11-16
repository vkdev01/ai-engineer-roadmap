# 2. Few-Shot Prompting and Chain of Thought

> **Stage 08 · Prompt and Context** — examples and reasoning steps, and the cost of each.
>
> **You'll be able to:** choose between zero-shot, few-shot and chain-of-thought with a reason, and build examples that teach the right thing.
>
> **Prerequisites:** [System Instructions That Actually Constrain](01-system-instructions.md) · [Self-Attention](../06-llm-fundamentals/04-self-attention.md)

## Why this exists

Two techniques dominate practical prompting, and both are widely misapplied. Few-shot prompting adds worked examples so the model can copy a pattern instead of inferring it. Chain-of-thought asks the model to write intermediate reasoning before answering, which helps on problems that need several steps.

Neither is a free upgrade. Examples consume context on every call and can teach the wrong lesson if they are unrepresentative. CoT spends output tokens, adds latency, and can actively hurt on simple tasks where it invents problems to solve. Choosing deliberately is the skill.

The mental model: **examples teach format and edge cases; reasoning steps buy accuracy on multi-step problems; both are paid for in tokens.**

## The mental model

```text
zero-shot      instruction only              cheap, fast, fine for easy tasks
few-shot       instruction + k examples      teaches format, edge cases, tone
CoT            instruction + "think first"   buys accuracy on multi-step tasks
few-shot CoT   examples that show reasoning  best for hard, patterned tasks
```

| Task type | Start with | Why |
|---|---|---|
| Classification with clear labels | zero-shot | labels are self-describing |
| Output format the model keeps missing | few-shot | one example beats ten words of description |
| Arithmetic, multi-hop, policy application | CoT | errors compound without intermediate steps |
| Latency-critical, simple extraction | zero-shot, no CoT | reasoning tokens are pure cost |
| Rare edge cases | few-shot with those cases | the model cannot invent them |

## Core concepts

### Examples are a specification, not decoration

Each example is a `(input, output)` pair that defines the mapping. The model infers the rule from the pairs, so the pairs must be consistent: if one example includes a reasoning paragraph and another does not, you have taught inconsistency. Keep the format byte-identical across examples, and include at least one case near a boundary you care about.

### Balance and order matter

If four of five examples are one label, the model leans that way. Balance the labels you care about, and put the example closest to the real query last, because recent context has more influence. Do not shuffle examples randomly per call unless you are deliberately testing robustness.

### Negative examples need care

An example showing a wrong answer can teach the wrong answer. If you include a rejection case, show the correct response to it ("declined because X") rather than the bad output you want avoided.

### CoT helps when the task has hidden intermediate steps

Ask for reasoning on tasks where the answer depends on several facts combined, unit conversions, or applying a written rule to a case. Ask for the reasoning in a defined slot, so it does not leak into the user-facing answer: a `reasoning` field, or a section you strip before display.

### CoT hurts when the task is simple or the reasoning is unverifiable

On a straightforward classification, a reasoning trace gives the model room to talk itself out of the right answer, and it triples latency. Also be honest about what CoT is: it produces plausible reasoning text, not a faithful record of computation. Treat the reasoning as scratch space, never as evidence you show to a user as proof.

## Code

A few-shot builder with strict formatting, plus a CoT variant whose reasoning is kept separate from the answer.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Label = Literal["bug", "feature_request", "question", "outage"]


@dataclass(frozen=True)
class Example:
    text: str
    label: Label
    reasoning: str = ""     # only used in the CoT variant


# Chosen to cover one boundary case (a question that is also an outage).
EXAMPLES: tuple[Example, ...] = (
    Example("The dashboard shows a spinner forever.", "bug"),
    Example("Please add dark mode to the settings page.", "feature_request"),
    Example("How do I rotate an API key?", "question"),
    Example("Nothing loads for any customer since 09:10.", "outage",
            reasoning="Total failure across customers is an outage, not a bug."),
)


def render_few_shot(examples: tuple[Example, ...], query: str) -> str:
    """Byte-identical blocks per example; the query goes last."""
    blocks: list[str] = []
    for ex in examples:
        blocks.append(f"Input: {ex.text}\nLabel: {ex.label}")
    return (
        "Classify each report. Reply with the label only.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nInput: {query}\nLabel:"
    )


def render_cot(examples: tuple[Example, ...], query: str) -> str:
    """Reasoning first, then the label on its own final line."""
    blocks: list[str] = []
    for ex in examples:
        if not ex.reasoning:
            continue
        blocks.append(
            f"Input: {ex.text}\nReasoning: {ex.reasoning}\nLabel: {ex.label}"
        )
    return (
        "Classify each report. Write one sentence of reasoning, then the label.\n"
        "Put the label on the final line as 'Label: <value>'.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nInput: {query}\nReasoning:"
    )
```

The consumer must parse the last line, not the whole text — otherwise the reasoning leaks into your UI.

```python
import re


def parse_label(answer: str) -> Label:
    # take the LAST "Label:" occurrence so reasoning cannot hijack it
    matches = re.findall(r"Label:\s*(bug|feature_request|question|outage)", answer)
    if not matches:
        raise ValueError(f"no label in answer: {answer[:120]}")
    return matches[-1]  # type: ignore[return-value]
```

In TypeScript, keep the same separation: the model writes reasoning into a field, and the UI reads only the other field.

```ts
export type Example = { text: string; label: string; reasoning?: string };

export function renderCoTExamples(examples: Example[], query: string): string {
  const blocks = examples
    .filter((e) => e.reasoning)
    .map((e) => `Input: ${e.text}\nReasoning: ${e.reasoning}\nLabel: ${e.label}`);
  return [
    "Classify each report. One sentence of reasoning, then the label.",
    "Put the label on the final line as 'Label: <value>'.",
    ...blocks,
    `Input: ${query}\nReasoning:`,
  ].join("\n\n");
}

export function extractReasoningAndLabel(
  answer: string,
): { reasoning: string; label: string | null } {
  const labelMatch = [...answer.matchAll(/Label:\s*([a-z_]+)/g)].at(-1);
  const reasoning = answer.split("Label:")[0]?.trim() ?? "";
  return { reasoning, label: labelMatch?.[1] ?? null };
}
```

## Real use case

A logistics company classified driver messages into `delay`, `breakdown`, `address_issue` and `other`. Zero-shot classification kept confusing `address_issue` with `delay`, because both mention time.

They added four few-shot examples — one per label, with the `address_issue` example deliberately containing the word "late" to break the surface pattern — and put the closest boundary example last. Confusions dropped in their own labelled sample. When they tried adding chain-of-thought on top, latency roughly doubled and accuracy moved within noise on that set, so they removed it and kept few-shot only. The failure mode before was an unrepresentative prompt: the model had inferred "mentions time means delay" from the label names alone, and the examples were what corrected it.

## Common mistakes

- **Mistake:** inconsistent example formats. — **Instead:** render every example from one template so the shape is identical.
- **Mistake:** adding CoT to every prompt. — **Instead:** use it for multi-step tasks and measure latency and accuracy before keeping it.
- **Mistake:** letting reasoning text reach the user as proof. — **Instead:** parse a final labelled line or a separate field and discard the trace.
- **Mistake:** unbalanced examples that skew a class. — **Instead:** cover each label you care about, including the boundary cases.
- **Mistake:** shuffling examples per call and calling it robustness. — **Instead:** fix the order and put the nearest analogue last; test order sensitivity separately.

## Checkpoint

1. Take one classification prompt and run it zero-shot, few-shot and few-shot with CoT over the same 20 examples; record accuracy and latency for each.
2. Add a boundary example that contradicts a surface pattern and check whether the confusion disappears.
3. Write a parser that fails loudly when the label is missing, and confirm it rejects a response that only contains reasoning.

**Self-check:** you compare the three variants on your own labelled set and keep the cheapest one that meets your accuracy bar; if CoT does not measurably help, you drop it because it costs latency and output tokens.

## Go deeper

- [Prompt Templates with Variables and Validation](03-templates.md) — turning these renders into reusable artefacts.
- [Prompt Evals](07-prompt-evals.md) — the measurement that decides between these techniques.
- [Reasoning and Planning](../11-agents/03-reasoning-planning.md) — CoT as the inner loop of an agent.
- [Loss Functions](../02-math-and-ml-basics/05-loss-functions.md) — why token-level training makes format examples so effective.

---
[← Previous: System Instructions That Actually Constrain](01-system-instructions.md) · [Stage index](README.md) · [Next: Prompt Templates with Variables and Validation](03-templates.md)
