# 12. Applying for Roles

> **Stage 23 · Career** — running a search like a project: targeting, referrals, take-homes, negotiation, and the arithmetic of not spraying applications.
>
> **You'll be able to:** build a target list with a route into each company, run referrals instead of cold applications, handle a take-home without losing a week, and negotiate without bluffing.
>
> **Prerequisites:** [Resume](02-resume.md) · [Production Case Studies](08-production-case-studies.md) · [Building in Public](11-build-in-public.md)

## Why this exists

Most job searches fail on process, not ability: two hundred cold applications, three replies, no referrals, and a take-home that eats a weekend per company. The engineers who get hired quickly are usually not better — they are targeting a smaller list, entering through people, and refusing work that does not test the job.

The mental model: **a search is a funnel with a fixed budget of your hours; spend them where the conversion is.**

## The mental model

```text
   20 targets ──► 12 with a route in ──► 8 conversations ──► 4 processes ──► 1-2 offers
        │                │                     │                  │
   small companies   referral or          30-min call        paid, bounded
   actually hiring   public work           with the team      take-home
   for this role     they have seen        (not a recruiter)
```

Track it. The numbers tell you where the funnel is broken — if 12 routes produce 1 conversation, your first message is the problem, not your skills.

## Core concepts

### Target, do not spray

Twenty companies is a full search. Choose on four filters: they are hiring for this specific role now; the work is the work you want (LLM applications, not GPU kernels); someone you can reach works there; and the company will still exist in two years. Rank by route quality, not by brand.

### Referrals beat applications by a wide margin

A referral is not a favour, it is a shortcut past the resume screen. The ask that works is short, specific and easy to say yes to:

> "Hi [name] — I saw [company] is hiring a [role]. I built [one specific thing, with a link] which is close to what your team does. Would you be open to referring me, or telling me if I am off-target? Either answer is useful."

Note what it does not do: it does not ask for a job, it does not attach a resume, and it gives the person an easy no. The specific artefact is what makes it credible.

### The take-home is a specification test

Before accepting: ask how long it should take, whether it is paid, and what they evaluate. If it is more than a few hours, ask whether an existing project of yours can substitute — many teams accept that, and the ones that do not are telling you something about how they work.

Then bound it: timebox, ship something working rather than something complete, write a README that states your assumptions and what you would do with more time, and include the tests. Reviewers spend ten minutes on your submission; the README is where those ten minutes go.

### Negotiation without bluffing

Know your number before the call: a range you would accept, based on what you can find out about the band for the role and level in your market. Never invent a competing offer; it is checkable and it ends processes. Ask for the band first, then respond with a specific figure and a reason tied to scope. Everything is negotiable to some degree — base, level, start date, review timing — and asking once is normal.

### The Indian market and remote roles, honestly

Hiring bars, titles and compensation vary widely between Indian product companies, services companies, startups and global remote employers, and they change year to year. Do not plan around numbers you read in a post. Do plan around three durable facts: referrals convert far better than portals; a public artefact (repo, article, contribution) substitutes for brand-name experience; and timezone overlap is a real filter for global remote roles, so state your working hours early. Treat anything you read about specific salary bands, including in this repo, as something to verify.

## Code

Run the search as a small system, because the failure mode is forgetting who you spoke to and when.

```python
# search.py — the funnel tracker. Keep it in a private repo, not public.
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date

STAGES = ["target", "routed", "conversation", "process", "offer", "closed"]

@dataclass
class Target:
    company: str
    role: str
    route: str                      # referral | public work | cold | recruiter
    stage: str = "target"
    last_touch: date | None = None
    next_action: str = ""
    notes: str = ""

    def advance(self, stage: str, on: date | None = None) -> None:
        if STAGES.index(stage) < STAGES.index(self.stage):
            raise ValueError(f"cannot move backwards: {self.stage} -> {stage}")
        self.stage = stage
        self.last_touch = on or date.today()


def funnel(targets: list[Target]) -> dict:
    counts = {s: sum(1 for t in targets if t.stage == s) for s in STAGES}
    routed = counts["routed"] + counts["conversation"] + counts["process"] + counts["offer"]
    conversations = counts["conversation"] + counts["process"] + counts["offer"]
    return {
        "targets": len(targets),
        "with_route": routed,
        "route_to_conversation": round(conversations / routed, 2) if routed else 0.0,
        "processes": counts["process"] + counts["offer"],
        "offers": counts["offer"],
        "stale": [t.company for t in targets
                  if t.last_touch and (date.today() - t.last_touch).days > 14
                  and t.stage in {"routed", "conversation"}],
    }


def next_actions(targets: list[Target]) -> list[str]:
    return [f"{t.company}: {t.next_action}" for t in targets if t.next_action]
```

TypeScript, for a dashboard you will actually look at:

```ts
type Stage = "target" | "routed" | "conversation" | "process" | "offer" | "closed";

export function weakestStep(counts: Record<Stage, number>): string {
  const pairs: [Stage, Stage][] = [
    ["target", "routed"],
    ["routed", "conversation"],
    ["conversation", "process"],
    ["process", "offer"],
  ];
  let worst = { label: "", rate: Infinity };
  for (const [from, to] of pairs) {
    if (!counts[from]) continue;
    const rate = counts[to] / counts[from];
    if (rate < worst.rate) worst = { label: `${from} -> ${to}`, rate };
  }
  return `weakest step: ${worst.label} at ${(worst.rate * 100).toFixed(0)}%`;
}
```

### The weekly review that keeps a search sane

Fifteen minutes, once a week: update stages, list stale threads, write next actions, and check the weakest funnel step. If `routed → conversation` is under about a fifth, rewrite the first message. If `conversation → process` is low, the problem is usually the project story (see [Production Case Studies](08-production-case-studies.md)).

## Real use case

An engineer spends two months on portals: 180 applications, four replies, one interview, no offer. They stop, build a list of twenty companies, and spend the next three weeks finding a route into twelve — through alumni, open-source maintainers they had contributed to, and two people who had replied to their posts.

Eight of the twelve turn into calls. Three become processes. Two offers arrive in the same week, which is when negotiation stops being a bluff and becomes a choice. The total number of applications in that second phase was four.

The difference was not a better resume. It was twenty targeted routes instead of a hundred and eighty anonymous submissions.

## Common mistakes

- **Mistake:** hundreds of cold applications — **Instead:** twenty targets, twelve routes, and a tracker.
- **Mistake:** asking for a job in the first message — **Instead:** a specific artefact, one question, and an easy no.
- **Mistake:** accepting an unbounded take-home — **Instead:** ask the expected hours and whether existing work can substitute; timebox and ship.
- **Mistake:** inventing a competing offer — **Instead:** state a number and a reason; it is checkable and honesty compounds.
- **Mistake:** planning around salary numbers from posts — **Instead:** verify bands per company and level; treat every figure as dated.
- **Mistake:** no weekly review, so threads go stale — **Instead:** fifteen minutes a week; the stale list is the most useful output.

## Checkpoint

1. Build a list of twenty targets, each with a named route or "no route yet". If more than half have no route, narrow the list.
2. Send five referral messages using the template, each with one specific artefact. Record replies in the tracker.
3. Run one take-home with a timebox and a README stating assumptions. Afterwards, write three lines on what the exercise revealed about how that team works.

**Self-check:** if your funnel shows plenty of conversations and no processes, the problem is your project story, not your search volume.

## Go deeper

- [Production Case Studies](08-production-case-studies.md) — what to say in the conversation
- [Resume](02-resume.md) — the artefact behind the route
- [GitHub Portfolio](01-github-portfolio.md) — what a referrer will look at

---
[← Previous: Building in Public](11-build-in-public.md) · [Stage index](README.md) · [Back to the roadmap](../../README.md)
