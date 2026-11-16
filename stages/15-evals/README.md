# Stage 15 — Evals and Testing

This is the stage that separates people who build demos from people who build systems. Everything before it makes something work once; evals are how you know it still works, and how you find out what you broke when you changed one sentence in a prompt.

It is also the most under-practised skill in the field, which makes it the highest-leverage thing you can learn.

**Time:** 2-3 weeks · **You need first:** [Stage 08 — Prompt and Context](../08-prompt-and-context/README.md) · **You leave with:** an eval suite with datasets, graders, thresholds and a report you can put in a slide

## What you will be able to do

- Write deterministic tests for the parts of an LLM system that can be deterministic
- Use a model as a judge with a rubric, and calibrate it against human labels
- Build and version a golden dataset from real usage
- Measure faithfulness and groundedness, not just "looks right"
- Detect and classify hallucinations
- Evaluate an agent on trajectory as well as outcome
- Design a human eval that produces an agreement number
- Gate a pull request on a pass rate
- Run online evals on live traffic and close the loop into offline sets

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Unit Tests for LLM Code](01-unit-tests.md) | Deterministic seams, fakes and golden files |
| 2 | [LLM as Judge](02-llm-as-judge.md) | Rubrics, bias, calibration, cost |
| 3 | [Golden Datasets](03-golden-datasets.md) | Building the reference set |
| 4 | [Faithfulness](04-faithfulness.md) | Is the answer supported by the sources |
| 5 | [Hallucinations](05-hallucinations.md) | Detection and classification |
| 6 | [Agent Evals](06-agent-evals.md) | Trajectory versus outcome |
| 7 | [Human Evals](07-human-evals.md) | Annotation design and agreement |
| 8 | [Regression Tests](08-regression-tests.md) | CI gates and flaky-test discipline |
| 9 | [Online and Offline Evals](09-online-offline-evals.md) | Production feedback loops |
| 10 | [Ship: An Eval Suite](10-ship-eval-suite.md) | Datasets, graders, thresholds, report |

## The shipping project

[Ship: An Eval Suite](10-ship-eval-suite.md) — a suite that runs in one command, prints a scorecard, compares against a stored baseline, and exits non-zero on regression. It contains deterministic tests, judge-graded cases and at least five adversarial inputs.

## How to know you are done

- Your suite runs in one command and prints a scorecard with a baseline comparison
- You know your judge's agreement rate against your own labels, and you have written it down
- You can name the number that would tell you the system got worse
- A deliberately broken prompt fails your CI gate
- Your dataset is versioned, and you have never edited it to make a score look better

---
[← Back to roadmap](../../README.md) · [Next stage: Inference](../16-inference/README.md)
