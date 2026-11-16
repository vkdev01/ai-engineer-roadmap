# 11. Building in Public

> **Stage 23 · Career** — a posting cadence you can sustain for a year, aimed at the people who hire rather than the people who cheer.
>
> **You'll be able to:** choose a cadence that survives a bad week, pick what to share and what to withhold, and convert visibility into conversations with hiring managers.
>
> **Prerequisites:** [Technical Articles](10-technical-articles.md) · [GitHub Portfolio](01-github-portfolio.md) · [Applying for Roles](12-apply-for-roles.md)

## Why this exists

Building in public works, and it is also the most common way engineers burn three months of evenings on content nobody who hires them ever sees. The difference is not effort — it is audience and cadence.

Audience: build for the twenty people who could hire or refer you, not for the thousand who might like a thread. Cadence: choose the smallest thing you can do on a bad week, because the bad weeks are what kill the habit.

The mental model: **small, regular, specific, and aimed at practitioners.**

## The mental model

```text
   what you build ──► what you learned ──► who needs that ──► where they read
        │                   │                   │                 │
   a project           one specific          a job title       one channel
   or a bug            lesson + number       + a company       you can keep

   cadence options, pick one and hold it:
   weekly  : one short post (300-600 words) + a screenshot
   biweekly: one longer write-up with code
   monthly : one substantial artefact (article, tool, talk)
```

## Core concepts

### Pick the smallest sustainable unit

The failure mode is starting with a plan of daily threads and stopping in week two. Choose a unit you can produce in under ninety minutes on a bad day: one paragraph and one screenshot; one diff and what it fixed; one number that moved. Consistency beats volume by a wide margin, and an audience that expects weekly posts forgives a quiet week far more than a silent month.

### Share the decision, not the diary

"I'm learning RAG" is a diary entry. "We had a two-chunk failure, so we added parent-child retrieval, and recall went 0.58 → 0.83" is a decision someone else can reuse. Post decisions, measurements, failures and corrections. Do not post daily progress logs, motivational content, or opinions about tools you have not used.

### What to withhold

Never post employer code, customer data, internal metrics, unreleased product details, or screenshots with identifiers in them. If you are unsure, ask; a written "can I share this" message costs a day and protects your job. Personal projects, public libraries and your own time are fair game. Also withhold certainty you do not have — a correction post later is fine, but a confident wrong claim travels further than the fix.

### The conversion nobody talks about

Visibility is not the goal; conversations are. The mechanism that works is: post something specific → a practitioner replies or messages → you ask one informed question about their problem → that becomes a call. Roughly one in twenty posts produces a real conversation. That is a good ratio, and it only works if the posts are specific enough to be findable.

### One channel, syndicated

Pick one primary channel and cross-post with canonical links. Being half-present on five platforms produces five weak presences. For AI engineering roles, the channels that actually convert are: a personal site or blog (durable, searchable), GitHub (where reviewers look), and one discussion venue where your target practitioners actually are. Choose that third one by looking at where the engineers at your target companies post, not where the engagement is highest.

## Code

Treat the habit like a system: a backlog, a template, a log. That is what makes it survive.

```python
# publishing/log.py — one row per post, so you can see what works.
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date

@dataclass
class Post:
    slug: str
    published: date
    channel: str
    topic: str                # bug | benchmark | explainer | failure | tool
    minutes_written: int
    conversations: int = 0    # replies that became a real conversation
    referrals: int = 0

def load(path: str = "publishing/log.csv") -> list[Post]:
    with open(path) as f:
        return [Post(**{**r, "published": date.fromisoformat(r["published"])}) for r in csv.DictReader(f)]


def summary(posts: list[Post]) -> dict:
    if not posts:
        return {}
    by_topic: dict[str, list[Post]] = {}
    for p in posts:
        by_topic.setdefault(p.topic, []).append(p)
    return {
        "posts": len(posts),
        "weeks_covered": round((posts[-1].published - posts[0].published).days / 7, 1),
        "avg_minutes": round(sum(p.minutes_written for p in posts) / len(posts)),
        "conversations": sum(p.conversations for p in posts),
        "best_topic": max(by_topic, key=lambda t: sum(p.conversations for p in by_topic[t])),
        "conversion_per_post": round(sum(p.conversations for p in posts) / len(posts), 2),
    }
```

```markdown
<!-- publishing/template.md — the 90-minute post -->
# <symptom as a search query>

**Setup:** <stack, versions, one line>
**What went wrong:** <two sentences, with the number>
**What I tried first:** <the wrong explanation, one sentence>
**What it actually was:** <the mechanism>
**The fix:** <code block, under 20 lines>
**Result:** <before -> after, with n>
**Lesson:** <two sentences, no moralising>

Repo: <link>
```

TypeScript, if you post a small tool with each write-up:

```ts
export type Post = { slug: string; topic: "bug" | "benchmark" | "explainer" | "failure" | "tool" };

export function weeklySlot(posts: Post[], today = new Date()): string {
  const weekday = today.getDay();
  if (weekday === 3) return "draft today, publish tomorrow";      // Wednesday: draft
  if (weekday === 4) return "publish today";                      // Thursday: publish
  return `next slot: Thursday · ${posts.length} posts so far`;
}
```

## Real use case

An engineer job-hunting for AI roles commits to one post a week for twelve weeks, each under 600 words with a screenshot or a number. Topics come from the Stage 22 project they are building: a chunking bug, a reranker comparison on their own data, a cost model, an agent that looped.

Four posts get almost no engagement. Three get a handful of replies. One — the agent looping story — is shared by a practitioner at a target company, which produces a direct message, which produces a call, which produces a referral into a role that was never publicly posted.

Twelve posts, roughly fifteen hours total. The return was not the audience; it was four real conversations with people who already knew how the candidate thinks.

## Common mistakes

- **Mistake:** daily cadence that collapses in week two — **Instead:** the smallest unit you can do on a bad day; weekly is plenty.
- **Mistake:** posting progress, not decisions — **Instead:** one decision, one number, one lesson per post.
- **Mistake:** sharing employer material without permission — **Instead:** ask in writing; keep private work out of public posts.
- **Mistake:** optimising for engagement — **Instead:** optimise for findability by the twenty people who could hire you.
- **Mistake:** five channels, none maintained — **Instead:** one primary, syndicated with canonical links.
- **Mistake:** never logging what worked — **Instead:** keep the log; it tells you which topic converts and which is wasted effort.

## Checkpoint

1. Write your smallest sustainable unit and the weekday it happens. If it takes more than ninety minutes, shrink it.
2. Post three items in three weeks, using the template. Log minutes, topic and conversations.
3. Review the log and pick the topic with the most conversations. Write two more posts in that vein.

**Self-check:** if after eight posts you have zero conversations, the posts are too generic — narrow the topic until it reads like a search query someone would actually type.

## Go deeper

- [Technical Articles](10-technical-articles.md) — the longer-form version of the same material
- [GitHub Portfolio](01-github-portfolio.md) — where the work lives
- [Applying for Roles](12-apply-for-roles.md) — converting attention into interviews

---
[← Previous: Technical Articles](10-technical-articles.md) · [Stage index](README.md) · [Next: Applying for Roles](12-apply-for-roles.md)
