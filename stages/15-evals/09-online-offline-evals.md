# 9. Online and Offline Evals

> **Stage 15 · Evals and Testing** — closing the loop between what you measure before shipping and what actually happens after.
>
> **You'll be able to:** define online signals that are honest, sample production traffic into your offline set, and run a feedback loop that improves the dataset instead of just producing dashboards.
>
> **Prerequisites:** [Regression Tests](08-regression-tests.md) · [Golden Datasets](03-golden-datasets.md) · [Tracing](../18-mlops-llmops/05-tracing.md)

## Why this exists

Offline evals answer "did this change break anything I already knew about". They cannot answer "is the system helping people" — because your dataset is a snapshot of the past, written by the team that built the system.

Online evals answer the second question, but they are seductive and easy to fake. Thumbs-up rates, "helpful" clicks and session length are all measurable and all misleading. The discipline is choosing signals that correlate with real outcomes and treating them as hypotheses, not verdicts.

The mental model: **offline protects quality; online discovers what quality means.**

## The mental model

```text
   production traffic
        │
        ├──► online signals (implicit + explicit) ──► dashboard + alerts
        │            │
        │            └──► anomaly → sample cases → label → offline set
        │                                                  │
        └──► traces ───────────────────────────────────────┤
                                                           ▼
                                              offline suite (frozen)
                                                           │
                                                     gate on PRs
```

The arrow that matters most is the one from anomaly to dataset. Without it, you have monitoring. With it, you have a system that learns.

## Core concepts

### Pick signals that survive scrutiny

| Signal | What it really tells you | Use with care |
|---|---|---|
| Explicit thumbs up/down | user's judgement of this answer | low volume, biased to extremes |
| Citation clicked | answer made the user check a source | may mean distrust, not trust |
| Answer copied | answer was useful enough to reuse | strong positive in B2B |
| Immediate rephrase | first answer failed | strong negative; the best signal you have |
| Escalation to human | the system could not resolve it | gold, if your product has a human path |
| Follow-up question on same topic | partial answer | ambiguous; may be healthy exploration |
| Session ended without action | unknown — silence is not satisfaction | never treat as positive |

The best signals are behavioural and cost the user something: a rephrase, an escalation, a copy. Ratings are cheap to give and easy to game.

### Online metrics need a denominator you can defend

"94% satisfaction" is meaningless without saying over what: which requests, which users, which intents, and how many abstained. Publish the denominator, the abstention rate, and the sample size next to every online number. When someone asks "is it working", the honest answer is usually "for questions like these, at this volume, measured this way".

### Sampling into the offline set

Do not sample randomly. Sample by *failure signature*: high latency, refusals, negative feedback, tool errors, loops, and any intent with a falling signal. Then label a batch and add the confirmed failures to the golden dataset, with the category `regression`.

```python
def select_for_review(traces: list[dict], limit: int = 50) -> list[dict]:
    """Priority sample: the cases most likely to teach you something."""
    def score(t: dict) -> float:
        s = 0.0
        if t.get("feedback") == "down":
            s += 5
        if t.get("rephrased"):
            s += 4
        if t.get("escalated"):
            s += 4
        if t.get("refused"):
            s += 2
        if t.get("tool_errors"):
            s += 3
        if t.get("latency_ms", 0) > 15_000:
            s += 2
        if t.get("loop_detected"):
            s += 5
        return s

    ranked = sorted(traces, key=score, reverse=True)
    picked, seen_intents = [], {}
    for t in ranked:
        intent = t.get("intent", "unknown")
        if seen_intents.get(intent, 0) >= 5:      # avoid one intent dominating the batch
            continue
        seen_intents[intent] = seen_intents.get(intent, 0) + 1
        picked.append(t)
        if len(picked) >= limit:
            break
    return picked
```

### Offline-to-online consistency

Your offline suite should predict online behaviour. Check that occasionally: take a metric that improved offline and see whether the corresponding online signal moved. When they disagree, one of them is measuring the wrong thing — usually the online signal, but sometimes the dataset, and finding out which is the real work.

### Alert on distributions, not single requests

One slow request is noise. p95 latency crossing a threshold for fifteen minutes is a signal. Same for refusal rate, tool-error rate and step count. Alert on the distribution shift, with a minimum sample size so a quiet hour does not page anyone.

## Code

A nightly job that turns production traces into review candidates and a drift report.

```python
from __future__ import annotations

import json
from datetime import date

def drift_report(today: list[dict], last_week: list[dict]) -> dict:
    def rate(rows: list[dict], key: str) -> float:
        return round(sum(1 for r in rows if r.get(key)) / max(len(rows), 1), 3)

    signals = ["refused", "escalated", "rephrased", "loop_detected"]
    return {
        "n_today": len(today),
        "n_last_week": len(last_week),
        **{f"{s}_today": rate(today, s) for s in signals},
        **{f"{s}_last_week": rate(last_week, s) for s in signals},
        "latency_p95_today": percentile([r.get("latency_ms", 0) for r in today], 95),
        "latency_p95_last_week": percentile([r.get("latency_ms", 0) for r in last_week], 95),
    }


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100) * len(ordered))) - 1))
    return ordered[idx]


def nightly(traces_path: str, out_path: str) -> None:
    traces = [json.loads(l) for l in open(traces_path) if l.strip()]
    week_ago = [t for t in traces if t.get("day") != str(date.today())]
    today = [t for t in traces if t.get("day") == str(date.today())]

    report = drift_report(today, week_ago)
    review = select_for_review(today, limit=50)

    with open(out_path, "w") as f:
        json.dump({"report": report, "review_batch": [r["trace_id"] for r in review]}, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    nightly("traces/requests.jsonl", "evals/nightly.json")
```

TypeScript, for the client that reports the strongest signal — the immediate rephrase:

```ts
type Turn = { sessionId: string; question: string; answerId: string; at: number };

export function detectRephrase(previous: Turn | null, current: Turn): boolean {
  if (!previous) return false;
  if (previous.sessionId !== current.sessionId) return false;
  const gap = current.at - previous.at;
  if (gap > 90_000) return false;                        // not a reaction, a new question
  const a = previous.question.toLowerCase().trim();
  const b = current.question.toLowerCase().trim();
  const overlap = [...new Set(a.split(/\s+/))].filter((w) => b.includes(w)).length;
  return overlap >= 2 || gap < 20_000;                   // same topic, immediately re-asked
}
```

## Real use case

A knowledge assistant has a healthy offline suite and a support team that keeps saying "it does not answer our questions". The team adds two online signals: escalation rate and immediate rephrase, both computed per intent.

Two intents account for most escalations: pricing and integration questions. Neither is represented in the offline dataset, because the engineers wrote cases about features they had built. A two-week labelling sprint adds forty cases from those intents; the offline pass rate on them starts around 0.55, and two iterations of retrieval tuning later it is above 0.8 — with the escalation rate falling in the same weeks.

The general lesson: **offline evals keep you honest about what you know; online evals tell you what you do not know you are missing.** You need both, and the loop between them is the actual system.

## Common mistakes

- **Mistake:** treating thumbs-up rate as quality — **Instead:** prefer behavioural signals (rephrase, escalate, copy) and publish sample sizes.
- **Mistake:** random sampling for review — **Instead:** sample by failure signature and stratify by intent.
- **Mistake:** alerting on single requests — **Instead:** alert on distribution shifts with a minimum sample size.
- **Mistake:** a dashboard nobody acts on — **Instead:** the job must emit a review batch; if nobody labels it, the loop is broken and you should say so.
- **Mistake:** never checking that offline improvements show up online — **Instead:** pick one metric per quarter and verify the correlation.
- **Mistake:** collecting user feedback without a privacy review — **Instead:** know what you store, for how long, and who can read it (Stage 20).

## Checkpoint

1. Instrument two behavioural signals in your app — immediate rephrase and one explicit signal — and log them with intent and latency.
2. Write the nightly drift report and run it over a week of traces. Identify the intent with the worst signal.
3. Sample twenty cases from that intent, label them, and add the failures to your golden dataset. Re-run your offline suite and record the drop.

**Self-check:** if your online and offline numbers never disagree, you are probably measuring the same thing twice — find a signal that comes from user behaviour alone.

## Go deeper

- [Tracing](../18-mlops-llmops/05-tracing.md) — the data these evals are computed from
- [Cost Monitoring](../18-mlops-llmops/06-cost-monitoring.md) — the other dashboard that matters
- [Ship: An Eval Suite](10-ship-eval-suite.md) — packaging offline and online together

---
[← Previous: Regression Tests](08-regression-tests.md) · [Stage index](README.md) · [Next: Ship: An Eval Suite](10-ship-eval-suite.md)
