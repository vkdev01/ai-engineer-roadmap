# 2. Resume

> **Stage 23 · Career** — one page that makes a hiring manager believe you have shipped something, in the ten seconds before they decide whether to keep reading.
>
> **You'll be able to:** write bullets that state an outcome and a measurement, structure a one-page resume for AI-engineer roles, and cut the parts that make you look junior.
>
> **Prerequisites:** [GitHub Portfolio](01-github-portfolio.md) · [Stage 22 — Real Projects](../22-real-projects/README.md)

## Why this exists

The resume is not a biography. It is a filter, read by someone who has a role in mind and a stack of forty others. They are asking two questions: has this person built the kind of thing we build, and can they tell me what happened when they did.

Most AI-engineer resumes answer neither. They list frameworks — LangChain, LlamaIndex, Pinecone, Hugging Face — as if the tool were the skill, and describe work as "built a chatbot using RAG", which is true of thousands of people and tells a reviewer nothing about difficulty or result.

Mental model: every bullet is a *claim plus evidence*, in one line, about work you personally did.

## The mental model

```text
[role, company, dates]
   bullet = verb + what you built + for whom + measured outcome + how
   |
   "Cut support first-response time from 6h to 20m by shipping a
    retrieval agent with a human-handoff packet; 62% deflection
    on 1.2k tickets/month"

page order:
  header (name, role, links)
  summary (2 lines, optional)
  experience OR projects (the bulk)
  skills (short, honest)
  education (last, one line)
```

One page. Everything on it exists to support a bullet.

## Core concepts

### The bullet formula

`Verb + object + scale + measurable outcome + mechanism`. "Reduced retrieval latency p95 from 4.1s to 1.3s by adding a rerank cache and streaming the first token" is a bullet. "Worked on RAG performance" is a heading with no content.

The measurement is what separates the two. If you do not have a business number, use an engineering one you genuinely measured: eval score on a frozen set, cost per 1,000 requests, latency, error rate, test coverage of the eval suite, number of documents or tickets processed. Never invent a percentage. If a project had no measurement, say what you built and what constraint you met — "shipped within a two-week deadline under a fixed per-request budget" is honest and still specific.

For a portfolio project with no employer, the measurement is your own eval: "citation precision 0.94 on 40 frozen questions; cost per query tracked in the ledger". This is stronger than most internship bullets because it is checkable.

### Structure for a career switcher

If you have no AI job yet, put **Projects** above **Experience**, and make the projects look like work: a one-line context, then two or three bullets each with a measurement. Keep prior non-AI experience but compress it to one or two lines per role, emphasising transferable facts — ownership, scale, on-call, cross-team delivery — not a list of duties.

If you do have relevant experience, lead with it and keep projects to a single "Selected projects" section with links.

### What to avoid

- Skill bars, star ratings, or "expert" labels — unverifiable and read as junior
- An objective statement ("seeking a challenging role where…") — replaced by a two-line summary naming what you build
- Every framework you have touched — list what you can defend under questioning
- Buzzword density ("cutting-edge, innovative, passionate") — no information
- Photos, dates of birth, marital status, full address — omit; in India these are common on CVs and are not needed for a tech role
- Tables and multi-column layouts that break in an applicant tracking system
- More than one page before you have several years of experience

### Make it machine-readable, then human-readable

Export as a PDF from a plain single-column document. Use standard headings ("Experience", "Projects", "Education", "Skills"), real text rather than images, and no text in headers or footers, where parsers often drop it. Keep the file name as your name. The same file should survive being copied into a plain text box.

### Tailor, but not by rewriting

For each role, reorder bullets to put the most relevant first and adjust the summary line to name the domain. Do not rewrite the whole document per application; you will stop doing it. Keep a master file with more bullets than fit, and cut per role — that is the only tailoring that scales.

## Code

Bullets are the unit of quality, so make the check mechanical. This linter flags the bullets that will be skimmed past: no number, first person, buzzwords, or a duty rather than an outcome.

```python
from __future__ import annotations

import re

WEAK = {
    "passionate", "innovative", "cutting-edge", "dynamic", "hardworking",
    "responsible for", "worked on", "helped with", "familiar with",
    "various", "several", "team player",
}
HAS_NUMBER = re.compile(r"\d")
STRONG_VERB = re.compile(
    r"^(built|shipped|reduced|cut|increased|designed|migrated|automated|"
    r"launched|scaled|measured|replaced|eliminated|led|wrote)\b",
    re.I,
)


def review(bullet: str) -> list[str]:
    text = bullet.strip().lstrip("-• ").strip()
    problems: list[str] = []
    low = text.lower()
    if any(w in low for w in WEAK):
        problems.append("weak phrase: " + ", ".join(w for w in WEAK if w in low))
    if not HAS_NUMBER.search(text):
        problems.append("no measurement")
    if not STRONG_VERB.match(text):
        problems.append("does not start with an outcome verb")
    if re.match(r"^(i|my)\b", low):
        problems.append("first person; drop the pronoun")
    if len(text.split()) > 30:
        problems.append("too long; aim for under 30 words")
    return problems


def main(text: str) -> None:
    for line in text.splitlines():
        if line.strip().startswith(("-", "•")):
            issues = review(line)
            status = "ok" if not issues else "; ".join(issues)
            print(f"{status:<60} {line.strip()[:60]}")


if __name__ == "__main__":
    import sys
    main(open(sys.argv[1], encoding="utf-8").read())
```

A before-and-after pair to copy the shape from:

```text
before:  Worked on a RAG chatbot using LangChain and Pinecone for document Q&A.
after:   Shipped a multi-tenant document assistant (FastAPI, pgvector) serving
         3 internal teams; raised answer citation precision to 0.94 on a frozen
         40-question eval and held p95 latency under 2s.
```

The second version is not longer because it is padded — it is longer because it carries a mechanism and a measurement.

## Real use case

A candidate switching from backend engineering applied to eleven AI-engineer roles with a resume that listed every model and framework they had tried and described each project as "built X using Y". No callbacks. They rewrote three bullets per project using the formula, moved Projects above Experience, cut the skills list to tools they could defend, and added one line to each project pointing at the repo and its eval command.

The design choice that made the difference is measurement: each bullet now answers "so what happened", which is the question the reviewer was silently asking.

## Common mistakes

**Mistake:** listing tools as skills — **Instead:** one line of tools you can defend, and let bullets show them in use.
**Mistake:** numbers with no basis — **Instead:** only measurements you can reproduce from a script or a dashboard.
**Mistake:** two pages of duties — **Instead:** one page of outcomes; duties belong in the interview.
**Mistake:** a photo, address and personal details — **Instead:** name, role, email, GitHub, one link to a project.
**Mistake:** a layout that a parser cannot read — **Instead:** single column, standard headings, text not images, PDF export.
**Mistake:** rewriting the resume for every application — **Instead:** a master file, cut and reordered per role.

## Checkpoint

1. Run the linter on your current resume and fix every bullet it flags as having no measurement or a weak phrase.
2. Take one Stage 22 project and write three bullets for it: the problem, the hard mechanism, the measured result. Cut whichever is weakest.
3. Give the page to someone who does not know your work and ask them to state what you are good at. If they cannot, the summary or the first bullet is wrong.

**Self-check:** Exercise 1 usually flags most bullets on a first draft, and the fix is to add the number you already have in your eval output rather than to write new claims.

## Go deeper

- [GitHub Portfolio](01-github-portfolio.md) — the repos your bullets point at
- [Production Case Studies](08-production-case-studies.md) — the long form of the same claims
- [Apply for Roles](12-apply-for-roles.md) — targeting and referral paths that make the resume matter
- [Python and SQL Interviews](03-python-sql-interviews.md) — what you must be able to defend from the skills line

---

[← Previous: GitHub Portfolio](01-github-portfolio.md) · [Stage index](README.md) · [Next: Python and SQL Interviews](03-python-sql-interviews.md)
