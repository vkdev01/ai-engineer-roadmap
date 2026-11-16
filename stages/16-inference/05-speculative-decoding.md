# 5. Speculative Decoding

> **Stage 16 · Inference** — getting several tokens out of one sequential model step by guessing and checking.
>
> **You'll be able to:** compute the expected speedup from an acceptance rate and decide when speculation is worth the complexity.
>
> **Prerequisites:** [Prefill and Decode](01-prefill-decode.md) · [Quantization](06-quantization.md)

## Why this exists

Decode is sequential and bandwidth-bound: one step reads all the weights and produces one
token. The card is doing a large amount of memory traffic per useful output, and much of its
compute sits idle. Speculative decoding exploits that idle compute.

A small, fast **draft** model proposes the next few tokens. The large **target** model then
scores all of those positions in a single forward pass — one pass, several tokens checked —
and accepts the longest prefix that matches what it would have produced itself. The output is
identical in distribution to the target model's own output, because the check is a
verification, not an approximation.

The mental model: **you cannot make the big model faster per step, but you can make each step
produce more than one token.**

## The mental model

```text
draft model (cheap, fast)   ->  "the  contract  is  governed  by"     (4 guesses)
target model (one pass)     ->  accept "the contract is", reject "governed"
emit 3 tokens from 1 target forward pass, plus the corrected 4th
```

| Component | Role | Cost |
|---|---|---|
| Draft model | proposes k tokens autoregressively | small, sequential |
| Target model | verifies k+1 positions in one pass | one decode step |
| Accept rule | keep the longest matching prefix | free |

## Core concepts

### The expected accepted tokens formula

If the draft's tokens are accepted independently with probability `alpha`, and you propose `k`
tokens per round, the expected number of tokens emitted per target pass is:

```text
E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)
```

The extra `+1` is the bonus token the target model emits after the accepted prefix. With
`alpha = 0.7` and `k = 4`:

```text
alpha^5 = 0.16807
E = (1 - 0.16807) / 0.3 = 2.77 tokens per target pass
```

Compare with `k = 0`, where you get exactly 1 token per pass. So the *upper bound* on speedup
from the target model's perspective is 2.77 times.

### The speedup is discounted by the draft's cost

Drafting is not free. If the draft model costs a fraction `c` of a target step (its own
sequential decode), then:

```text
speedup  ~=  E[tokens] / (1 + c)
```

With `c = 0.15` — plausible when the draft is roughly a tenth of the target's size — the
speedup is `2.77 / 1.15 ≈ 2.4 times`. That is why draft models are chosen small: a draft that
costs half a target step eats most of the gain.

### Acceptance rate depends on the task, not the hardware

`alpha` is high when the text is predictable — code with obvious boilerplate, structured JSON,
summaries of the source text — and low when the model is genuinely uncertain or creative. This
is the honest caveat: speculation helps most where you already knew the answer was easy. It
can even hurt on high-entropy creative writing, because rejected drafts still cost draft time.

| Task shape | Typical alpha (illustrative) | Worth it? |
|---|---|---|
| JSON extraction with a fixed schema | high | yes |
| Code completion in a repetitive file | high | yes |
| Open-ended creative writing | low | measure first |
| Long chain-of-thought reasoning | mixed | measure first |

### Variants you will meet

- **Draft-model speculation** — a separate small model, as above. Simplest to reason about.
- **Self-speculation** — the target model skips layers to draft, then verifies with all layers.
- **Medusa-style heads** — extra prediction heads on the target model propose several futures.
- **Prompt lookup** — draft by copying n-grams from the prompt or context. Free, no extra model,
  and surprisingly effective for editing and RAG answers where output echoes input.

Prompt lookup is worth knowing because it requires no second model and no training: if the
user asks you to fix a function, the answer contains the function, so n-grams from the prompt
are good guesses.

## Code

This simulates the acceptance process to give you an intuition for how the expected value and
its variance behave, and computes the break-even acceptance rate below which speculation is
slower than plain decoding.

```python
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecResult:
    expected_tokens: float
    speedup: float
    break_even_alpha: float


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens emitted per target forward pass, assuming independent acceptance."""
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def evaluate(alpha: float, k: int, draft_cost_fraction: float) -> SpecResult:
    e = expected_tokens(alpha, k)
    speedup = e / (1.0 + draft_cost_fraction)
    # Speedup > 1 requires E[tokens] > 1 + c.
    # Solve (1 - a^(k+1))/(1 - a) = 1 + c by bisection on alpha.
    lo, hi = 0.0, 1.0
    target = 1.0 + draft_cost_fraction
    for _ in range(60):
        mid = (lo + hi) / 2
        if expected_tokens(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return SpecResult(round(e, 3), round(speedup, 3), round(hi, 3))


def simulate(alpha: float, k: int, rounds: int = 20_000, seed: int = 7) -> float:
    """Empirical mean tokens per round, to check the closed form."""
    rng = random.Random(seed)
    total = 0
    for _ in range(rounds):
        accepted = 0
        for _ in range(k):
            if rng.random() < alpha:
                accepted += 1
            else:
                break
        total += accepted + 1      # +1 for the bonus token
    return total / rounds


if __name__ == "__main__":
    print(f"{'alpha':>6} {'k':>3} {'E[tok]':>7} {'speedup':>8} {'sim':>7}")
    for alpha in (0.4, 0.5, 0.6, 0.7, 0.8):
        r = evaluate(alpha, k=4, draft_cost_fraction=0.15)
        print(f"{alpha:6.2f} {4:3d} {r.expected_tokens:7.2f} "
              f"{r.speedup:8.2f} {simulate(alpha, 4):7.2f}")
    print("break-even alpha at c=0.15, k=4:", evaluate(0.7, 4, 0.15).break_even_alpha)
```

You should see the simulation track the closed form closely, and the break-even acceptance
rate come out near 0.13 — below that, the draft cost outweighs the tokens saved. Raise
`draft_cost_fraction` to 0.5 and the break-even acceptance rate climbs sharply, which is the
quantitative case for a small draft model.

## Real use case

A team built a schema-extraction service: every request turned an invoice into a fixed JSON
object with a known set of fields. Their 70B target model produced accurate output but at a
per-token cost that made the unit economics uncomfortable at their volume.

They added a draft model roughly a tenth of the target's size, with `k = 4`. Acceptance was
high — around 0.8 on their test set — because the field names and formatting are highly
predictable once the model has committed to a schema. Plugging in: `E = (1 - 0.8^5) / 0.2 =
3.36` tokens per pass, and with a draft cost of about 0.15 of a target step, an expected
speedup of `3.36 / 1.15 ≈ 2.9`. Their measured gain was lower, in the region of 2 times,
because real acceptance varies across documents and the draft occasionally diverges early.

The lesson they carried forward: they measured acceptance per endpoint rather than globally.
On their free-text "summarise this invoice" endpoint acceptance fell far enough that they
disabled speculation there, because rejected drafts were pure added latency.

## Common mistakes

- **Mistake:** enabling speculation everywhere and assuming a uniform gain — **Instead:** measure acceptance per endpoint and per task type; disable it where it is low.
- **Mistake:** picking a draft model that is nearly as large as the target — **Instead:** keep the draft small enough that `c` stays well under 0.2.
- **Mistake:** using a draft from a different tokenizer family — **Instead:** use a draft with the same tokenizer, or verification cannot align positions.
- **Mistake:** expecting speculative decoding to change output quality — **Instead:** it is designed to preserve the target distribution; if quality moved, your acceptance check is wrong.
- **Mistake:** forgetting the draft's memory footprint — **Instead:** the draft's weights and cache also occupy the GPU; budget both in [GPU Memory Budgeting](10-gpu-memory.md).

## Checkpoint

1. With `alpha = 0.6` and `k = 4`, how many tokens do you expect per target pass, and what speedup at `c = 0.2`?
2. Show algebraically why `E[tokens]` approaches `1/(1 - alpha)` as `k` grows, and explain what stops you from setting `k = 100`.
3. Implement prompt-lookup drafting: take the last 3 tokens of context, find their last occurrence earlier in the prompt, and propose the following tokens. Measure acceptance on 50 completions of a code-editing task.

**Self-check:** `alpha^5 = 0.07776`, so `E = (1 - 0.07776) / 0.4 = 2.31` tokens per pass, and the speedup is `2.31 / 1.2 ≈ 1.92 times`.

## Go deeper

- [Prefill and Decode](01-prefill-decode.md) — why a decode step is bandwidth-bound and leaves compute spare.
- [Quantization](06-quantization.md) — the other way to buy speed, by moving fewer bytes per step.
- [Distillation](07-distillation.md) — how you would obtain a good draft model in the first place.
- [Cost and Latency](../21-system-design/08-cost-latency.md) — deciding how much of this optimisation your product needs.

---
[← Previous: PagedAttention](04-pagedattention.md) · [Stage index](README.md) · [Next: Quantization](06-quantization.md)
