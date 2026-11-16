# 7. Distillation

> **Stage 16 · Inference** — training a small model to behave like a large one, so inference gets cheaper at the source.
>
> **You'll be able to:** explain soft-label distillation, compute its cost and payoff, and run a small distillation loop.
>
> **Prerequisites:** [Quantization](06-quantization.md) · [SFT](../13-fine-tuning/01-sft.md)

## Why this exists

Quantization shrinks the bytes per weight but keeps the architecture. Distillation shrinks the
model itself: you train a smaller **student** to reproduce the outputs of a larger **teacher**.
The result is fewer parameters, which cuts both memory and per-token work permanently.

The interesting part is *what* you copy. Hard labels say "the answer is class 7". A teacher's
full probability distribution says "mostly 7, a bit of 3, almost nothing else" — and that
shape carries information about similarity between classes that the hard label throws away.
Learning from the soft distribution is why distillation beats simply training the small model
on the teacher's outputs as if they were ground truth, especially when data is limited.

The mental model: **distillation is compression of a function, not of a file.**

## The mental model

```text
prompts --> teacher (large, slow, expensive) --> soft distributions (logits)
                                                      |
                                                      v
prompts --> student (small, fast) --> logits --> KL divergence loss
                                                      ^
                                                      |
                                       hard labels (optional, weighted)
```

| | Teacher | Student |
|---|---|---|
| Parameters | large | 5-30x smaller |
| Runs at inference | no | yes |
| Runs at training | yes, to label data | yes, to learn |
| Cost driver | one-off labelling pass | your serving bill |

## Core concepts

### Soft labels need a temperature

Logits from a confident model are extremely peaked, so the small differences between
near-zero probabilities vanish into numerical noise. Dividing logits by a temperature `T > 1`
before the softmax flattens the distribution and makes those differences visible:

```text
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```

Typical values are `T` between 2 and 10 for classification-style distillation, with `T = 1`
for the final student inference. The student trains against the softened teacher distribution
and, if you have real labels, a weighted cross-entropy against those too. The standard loss:

```text
L = (1 - w) * KL( softmax(teacher/T) || softmax(student/T) ) * T^2  +  w * CE(student, labels)
```

The `T^2` factor keeps gradient magnitudes comparable when you change temperature.

### For LLMs, distillation usually means data, not logits

With language models you often cannot get the teacher's full logit vector — many APIs expose
only sampled text. So the common pattern is **sequence-level distillation**: generate a large
corpus of teacher completions on your own prompts, then fine-tune the student on them with
standard SFT. See [Synthetic Data](../13-fine-tuning/03-synthetic-data.md) for how to build
that set without teaching the student the teacher's mistakes.

If you do have logits — an open-weight teacher you run yourself — token-level distillation
with a KL loss over the vocabulary is stronger per example. It requires the student and
teacher to share a tokenizer, or you must map vocabularies.

### The arithmetic: what you pay and what you save

Costs, all computable before you start:

```text
labelling cost  ~= prompts x teacher_cost_per_call
training cost   ~= tokens x epochs x student_training_flops
serving saving  ~= (teacher_bytes - student_bytes) / teacher_bytes   in decode time
```

Take a teacher at 70B bf16 (130 GiB of weights) and a student at 7B bf16 (13 GiB). On a
2 TB/s card, decode at batch one is bandwidth-bound:

```text
teacher: 130 GiB / 2000 GB/s  ~= 70 ms per token
student:  13 GiB / 2000 GB/s  ~=  7 ms per token
```

That is roughly a 10x reduction in per-token decode time, before any batching, and the student
fits on a card the teacher never could. Label 50,000 prompts once at an average of 800 output
tokens each: that is 40 million teacher tokens, a one-off bill. If you serve millions of
requests, the payback period is short. If you serve thousands, it is not — this is the honest
gate on distillation.

### Distillation does not create capability

A student cannot reliably learn what the teacher cannot demonstrate in the data you generate.
Narrow the student's job and distillation works well: a classifier, an extractor, a router, a
domain summariser. Ask it to be a general assistant at one-tenth the size and you will find
the ceiling quickly. Distillation also inherits teacher failure modes — including any bias or
hallucination pattern present in the generated corpus.

## Code

A minimal logit-distillation loop. It uses a toy teacher and student of different widths on a
synthetic classification task, so it runs on a CPU in seconds while showing the real loss
shape.

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    hard_weight: float = 0.3,
) -> torch.Tensor:
    """KL against softened teacher + weighted cross-entropy against real labels."""
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    # batchmean divides by batch size, matching the scale of CE below
    kd = F.kl_div(soft_student, soft_teacher, log_target=True, reduction="batchmean")
    ce = F.cross_entropy(student_logits, labels)
    return (1.0 - hard_weight) * kd * temperature**2 + hard_weight * ce


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2048, 32)
    labels = (x[:, 0] + x[:, 1] > 0).long()

    teacher = TinyNet(32, 256, 2)
    student = TinyNet(32, 16, 2)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)

    with torch.no_grad():                       # teacher is frozen, logits precomputed once
        teacher_logits = teacher(x)

    for step in range(400):
        student_logits = student(x)
        loss = distill_loss(student_logits, teacher_logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            acc = (student_logits.argmax(-1) == labels).float().mean().item()
            print(f"step {step:4d} loss {loss.item():.4f} acc {acc:.3f}")


if __name__ == "__main__":
    main()
```

The student has one-sixteenth the hidden width of the teacher and still learns the boundary,
because the soft targets tell it which inputs the teacher finds ambiguous. Remove the KL term
and train on hard labels alone to see how much slower convergence becomes on this task.

## Real use case

A routing layer at an AI product decided, for every incoming message, which of six downstream
pipelines should handle it: refund, technical bug, billing question, sales, abuse report, or
general chat. They started with a large general model and a careful prompt. Accuracy was good,
but routing ran on every single message, so it dominated both cost and the latency floor of the
whole system — every conversation paid a full large-model call before anything useful started.

They distilled the router. First they collected 40,000 real messages already routed in
production and labelled them by the large model with a fixed prompt, keeping the model's
per-class probabilities as soft targets. Then they trained a small encoder classifier against
the KL loss above plus the human-corrected labels they had for 4,000 of those messages. The
student was three orders of magnitude smaller than the teacher.

Two design choices made it work. The soft targets transferred the teacher's uncertainty, so
"billing question" and "refund" stayed separable — hard labels alone had blurred them because
annotators disagreed at the boundary. And they kept a confidence threshold: when the student's
top probability was below a cut-off, the request escalated to the teacher. That escalation
path meant a distillation regression degraded cost, not correctness.

## Common mistakes

- **Mistake:** distilling without checking the teacher is actually good on your task — **Instead:** evaluate the teacher first; the student inherits its errors faithfully.
- **Mistake:** using `T = 1` and wondering why the KL term is nearly zero — **Instead:** raise the temperature so the soft distribution carries signal.
- **Mistake:** training on hard labels only and calling it distillation — **Instead:** use soft targets, or be honest that this is plain SFT on synthetic data.
- **Mistake:** distilling for a general assistant role with a 10x size gap — **Instead:** narrow the task; distillation compresses a function, not general capability.
- **Mistake:** skipping the serving-side comparison — **Instead:** measure student versus teacher on latency, cost and quality before deleting the teacher path.

## Checkpoint

1. A teacher takes 70 ms per token and a student 7 ms. If a request generates 300 tokens, what is the decode-time difference, ignoring prefill?
2. Train the toy student with `hard_weight = 1.0` and again with `hard_weight = 0.0`. Plot loss curves and explain the difference.
3. Take a real task you have prompts for, generate 2,000 teacher completions, fine-tune a small open model on them, and compare quality and tokens per second against the teacher. Report where the student fails.

**Self-check:** `(70 - 7) ms x 300 = 18.9 seconds` saved per request — which is why distillation matters for high-volume, short-output tasks like routing and extraction.

## Go deeper

- [SFT](../13-fine-tuning/01-sft.md) — the mechanics of the fine-tuning step you will reuse here.
- [Synthetic Data](../13-fine-tuning/03-synthetic-data.md) — building the teacher-labelled corpus properly.
- [Quantization](06-quantization.md) — stack it on top of the student for a further memory cut.
- [Model Routing](../21-system-design/04-model-routing.md) — the product pattern this file's use case comes from.

---
[← Previous: Quantization](06-quantization.md) · [Stage index](README.md) · [Next: vLLM and SGLang](08-vllm-sglang.md)
