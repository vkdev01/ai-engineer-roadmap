# 8. Production Case Studies

> **Stage 23 · Career** — turning your projects into stories with a shape, so an interviewer remembers the decision rather than the technology.
>
> **You'll be able to:** structure a project story in five beats, answer follow-ups without losing the thread, and tell the truth about what went wrong in a way that increases trust.
>
> **Prerequisites:** [Production RAG](../22-real-projects/03-production-rag.md) · [Ship: A Design Document](../21-system-design/10-ship-design-doc.md) · [Resume](02-resume.md)

## Why this exists

Two candidates built comparable systems. One says "I built a RAG pipeline with LangChain and Pinecone"; the other says "support agents were answering from superseded policy versions; I found the ingestion was indexing both, fixed it, and recall on our frozen set went from 0.58 to 0.83".

The second candidate is not exaggerating — they are just narrating the work in the order that makes it meaningful. That structure is learnable, and it is most of the difference in how interviews go.

The mental model: **context → constraint → decision → measurement → lesson.**

## The mental model

```text
   1 context      what was happening, who was affected
   2 constraint   the limit that made this hard
   3 decision     what you chose, and what you rejected
   4 measurement  the number before and after
   5 lesson       what you would do differently

   ────────────────────────────────────────────────────► 90 seconds
   then: one follow-up per beat, expanded on request
```

Five beats, ninety seconds. The interviewer will pick a beat and go deeper; that is the interview working correctly.

## Core concepts

### Beat 1-2: context and constraint

Name the user, the failure and the limit. "Support agents were answering from superseded policies" is context. "The corpus was 40,000 documents, re-embedding took six hours, and we could not take the service down" is constraint. Without a constraint, your decisions look arbitrary.

### Beat 3: the decision, with the rejected option

This is the beat that separates senior from mid. Always name what you did not do and why: "I considered fine-tuning, but the failure was freshness, not format — fine-tuning would have made stale answers more fluent."

### Beat 4: the measurement

One number, before and after, with the method: "recall on a frozen 40-case set, 0.58 → 0.83 after parent-child expansion". If you have no number, say what you observed instead and be explicit that it is qualitative — an honest "we saw complaints drop but did not instrument it" is better than an invented percentage, and interviewers can tell.

### Beat 5: the lesson

The lesson must be about judgement, not technology: "I now put the eval set before the pipeline, because I spent two weeks tuning prompts for what turned out to be an ingestion bug."

### Prepare three, not ten

Three stories, each covering a different axis: one retrieval/quality, one systems/cost-latency, one agent/reliability. Most behavioural and technical questions can be answered from one of the three. Rehearse each at 30 seconds, 90 seconds and 5 minutes.

## Code

Write the stories down, because writing them is what exposes the gaps in your numbers.

```markdown
# Story: policy-version answers

## Context
Internal support assistant over 40,000 policy documents, ~600 questions/day.
Agents reported answers citing superseded policies. Trust in the tool was falling.

## Constraint
- Cannot take the service down; agents work 24/7
- Re-embedding the full corpus took ~6 hours on our hardware
- Three engineers, no dedicated ML platform team

## Decision
Chose: version-aware ingestion (index only the current version, keep history keyed by
effective date) + parent-child retrieval for long clauses.
Rejected: fine-tuning (failure was freshness, not format); a bigger model (cost tripled,
and faithfulness was already 0.93); agentic retrieval (latency budget was 4s p95).

## Measurement
Frozen 40-case set: context recall 0.58 -> 0.83; faithfulness 0.90 -> 0.93;
invalid citations 3/40 -> 0/40. Complaint volume roughly halved over the next month
(self-reported, not instrumented).

## Lesson
I spent two weeks tuning prompts before checking ingestion. The eval set now exists
before the pipeline, and I read the parse failures first.

## Follow-ups I expect
- Why not version at query time instead of ingest time?
- How did you re-embed 40k docs without downtime?
- What would you do at 10x the corpus?
- How did you build the 40 cases, and how do you know they are representative?
```

```python
# story_check.py — a brutal but useful self-check on your own stories.
import re

REQUIRED = ("context", "constraint", "decision", "measurement", "lesson")

def check(story: str) -> list[str]:
    text = story.lower()
    problems = [f"missing beat: {b}" for b in REQUIRED if b not in text]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", story)
    if len(numbers) < 3:
        problems.append("fewer than 3 numbers — the measurement beat is thin")
    if "considered" not in text and "rejected" not in text:
        problems.append("no rejected option — add what you decided against")
    if re.search(r"\b(helped|improved|better)\b", text) and not numbers:
        problems.append("qualitative claim with no number")
    return problems


if __name__ == "__main__":
    print(check(open("stories/policy-version.md").read()) or "story has all five beats")
```

TypeScript has no role here beyond one practical tip: keep your stories in a repo, not in your head.

```ts
type Story = { slug: string; beats: Record<string, string>; followUps: string[] };

export function coverage(stories: Story[]): string {
  const axes = new Set(stories.map((s) => s.slug));
  const missing = ["retrieval", "systems", "agents"].filter((a) => !axes.has(a));
  return missing.length ? `missing axis: ${missing.join(", ")}` : "all three axes covered";
}
```

## Real use case

A candidate with strong work gets rejected after a final round. Feedback: "technically deep, hard to follow". Their project description had been a technology list.

They rewrite three stories in the five-beat shape, rehearse them at three lengths, and prepare four likely follow-ups each. Two weeks later, in a different company, the same work lands: the interviewer spends the second half of the session asking about the decision to reject fine-tuning, which is exactly where the candidate is strongest.

Nothing about their engineering changed. What changed is that the interviewer could now see it.

## Common mistakes

- **Mistake:** starting with the stack — **Instead:** start with the user and the failure.
- **Mistake:** no constraint, so decisions look arbitrary — **Instead:** name the limit: time, money, headcount, uptime, data.
- **Mistake:** hiding what went wrong — **Instead:** include the mistake in the lesson beat; it is the fastest trust-builder in an interview.
- **Mistake:** invented metrics — **Instead:** say what you observed and how; a stated limitation beats a suspiciously round number.
- **Mistake:** ten stories, all half-rehearsed — **Instead:** three, at three lengths, each with four prepared follow-ups.
- **Mistake:** describing only the final architecture — **Instead:** narrate the sequence of decisions; that is what is being assessed.

## Checkpoint

1. Write three stories in the five-beat format: retrieval/quality, systems/cost, agent/reliability. Run `story_check.py` over each and fix the gaps it finds.
2. Record yourself telling each in 90 seconds. Note where you hesitate — that is the beat you do not actually understand.
3. Write four follow-up questions per story and answer them out loud without notes.

**Self-check:** if a story has no number in the measurement beat, either find one by instrumenting retroactively or mark it explicitly as qualitative — do not round up.

## Go deeper

- [Resume](02-resume.md) — the one-line version of each story
- [System Design Practice](05-system-design-practice.md) — the design-interview version
- [Ship: A Design Document](../21-system-design/10-ship-design-doc.md) — the written artefact behind the story

---
[← Previous: Inference Optimisation](07-inference-optimization.md) · [Stage index](README.md) · [Next: Open Source](09-open-source.md)
