# 1. GitHub Portfolio

> **Stage 23 · Career** — making your repositories readable in the sixty seconds a reviewer actually gives them.
>
> **You'll be able to:** choose what to pin, structure a README so the value is visible before any scrolling, record a demo that survives a muted autoplay, and delete the repos that are costing you credibility.
>
> **Prerequisites:** [Stage 22 — Real Projects](../22-real-projects/README.md) · [Git and GitHub](../01-programming-basics/05-git-github.md)

## Why this exists

A reviewer opens your profile with a task in mind: decide in under a minute whether there is any evidence you can build. They are not reading your code at that point. They are scanning repo names, descriptions, and the top of each README. Most profiles fail this not because the work is bad but because the evidence is buried under setup instructions, or the profile's first impression is twelve forked tutorial repos.

The other failure is subtler: a repo that only works on your machine. If the README says "clone and run" and it does not run, a reviewer concludes you have not shipped, which is worse than having no repo at all.

Mental model: your profile is a *shelf, not an archive*. Three to six repos, each with a visible result, beat twenty with none.

## The mental model

```text
profile page (60 seconds)
   |
   +-- pinned repos (3-6)     <- names + one-line descriptions do the work
   |      |
   |      +-- README top fold: problem -> result -> screenshot/GIF
   |      +-- README body: architecture, evals, limits, run it yourself
   |
   +-- profile README (optional): what you work on, links to the best three
   |
   +-- everything else: unpinned, archived, or deleted
```

## Core concepts

### Pin three to six, and pick them for coverage

Choose repos that together show breadth without looking scattered: one retrieval or RAG system, one agent or tool-use system with a real side effect, one measurement artifact (an eval harness or a benchmark you wrote). The trio recommended in [Stage 22 — Real Projects](../22-real-projects/README.md) is designed for exactly this. Each pinned repo gets a description that states what it does and what makes it hard — not "my project" and not a technology list.

If a repo is a tutorial clone, a course exercise, or a fork with no changes, unpin it. You do not have to delete it; just do not put it in the window.

### The top fold decides everything

The first screen of a README — before any scrolling — must contain: one sentence naming the user problem, one sentence naming the result with a number you measured, and an image or GIF. A reviewer who reads only that should be able to describe your project to a colleague. Everything else (install, architecture, design decisions) goes below.

The number must be yours and honest: "citation precision 0.94 on 40 frozen questions, eval script in `evals/`" is credible because it is checkable. Never invent a benchmark comparison.

### A demo that survives a muted autoplay

Record a 20-40 second GIF or short video: type the input, show the output, show one thing going wrong gracefully. No narration needed, no cursor hunting, no terminal scrolling at speed. Put it at the top of the README and keep the file under a few megabytes so GitHub renders it inline. A still screenshot is better than nothing and much better than a link to a video host that may not load in a corporate network.

### Show the engineering, not just the result

The sections that make a reviewer trust you: an ASCII or exported architecture diagram, the eval table with the command that regenerates it, a short "limits and failure modes" section, and a "what I would do next" list. The limits section is the one most people omit and the one that signals seniority — anyone can claim it works; naming where it breaks means you tested it.

### Make it runnable

`make setup && make demo`, or a single `docker compose up`, or a `uv run` command. Provide a `.env.example` with placeholder keys and a small sample dataset so the demo does not require credentials. If a paid API key is required, say so at the top and provide a recorded output file so the reviewer can see results without signing up for anything.

### What to leave out

Leave out: notebooks with no narrative, datasets you do not have rights to, scraped data, anything with a key in the history, "assignment" folders, and long personal statements. Also leave out the commit history you are ashamed of — nobody reads it, and rewriting it is not worth the risk of a broken repo. If a key ever leaked, rotate it; do not just delete the file.

## Code

A README skeleton that encodes the order above. Copy it, fill it, delete sections you genuinely do not need — but keep the result line and the limits section.

````markdown
# project-name

One sentence: what user problem this solves.

**Result:** <metric> = <value> on <n> frozen cases (`make eval`). Cost <per 1k requests>, p95 <latency>.

![demo](docs/demo.gif)

## The problem
Two or three sentences. Who has this problem and what they do today.

## How it works
```
[ascii diagram: 6-10 lines]
```
Short paragraph per component.

## Results
| metric | baseline | this system |
|---|---|---|
| ... | ... | ... |

## Run it
```bash
cp .env.example .env      # add your key
make setup && make demo
```

## Limits and failure modes
- Fails on X because Y; handled by Z.
- Not tested at more than N concurrent users.

## What I would do next
- ...
````

A small hygiene check you can run before you pin anything — it catches the mistakes that are invisible to you because you wrote the file:

```python
from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED = ["## the problem", "## run it", "## limits"]
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def check(repo: Path) -> list[str]:
    problems: list[str] = []
    readme = repo / "README.md"
    if not readme.exists():
        return ["no README.md"]
    text = readme.read_text(encoding="utf-8")
    low = text.lower()
    for heading in REQUIRED:
        if heading not in low:
            problems.append(f"missing section: {heading}")
    if not re.search(r"\d", text.split("\n\n")[1] if "\n\n" in text else text):
        problems.append("no number in the opening block")
    for path in repo.rglob("*"):
        if path.is_file() and path.stat().st_size < 2_000_000 and path.suffix in {".py", ".md", ".env", ".yaml", ".yml", ".json"}:
            if SECRET.search(path.read_text(encoding="utf-8", errors="ignore")):
                problems.append(f"possible secret in {path.relative_to(repo)}")
    return problems


if __name__ == "__main__":
    for repo in map(Path, sys.argv[1:]):
        found = check(repo)
        print(f"{repo.name}: " + ("ok" if not found else "; ".join(found)))
```

## Real use case

A candidate with strong Stage 22 work was getting no replies. Their profile showed nine repos: four tutorial clones, two forks, three real projects whose READMEs opened with `pip install -r requirements.txt`. The fix was not more work — it was ordering. They pinned three projects, moved a measured result and a GIF into the first screen of each, added a limits section, and made `make demo` work from a clean clone.

The design choice that made the difference is the top fold: a reviewer who reads only the first screen now knows what was built and whether it works, which is the entire decision they were making.

## Common mistakes

**Mistake:** pinning everything you have — **Instead:** three to six repos chosen for coverage; the rest stay unpinned.
**Mistake:** opening the README with installation steps — **Instead:** problem, result, demo first; setup belongs below.
**Mistake:** describing a repo by its stack ("FastAPI + LangChain") — **Instead:** describe the user problem and the measured outcome.
**Mistake:** claiming results with no script to reproduce them — **Instead:** commit the eval script and the frozen dataset version.
**Mistake:** a demo video that requires narration to make sense — **Instead:** show the input and the output on screen; assume sound is off.
**Mistake:** hiding known failures — **Instead:** a limits section, which is the strongest seniority signal on the page.

## Checkpoint

1. Run the hygiene script against your three best repos and fix every finding. Rotate any key it flags, then rewrite history or start a fresh repo — deletion alone does not remove a leaked key from forks.
2. Rewrite the first screen of one README so it contains a problem sentence, a measured result and an image. Ask someone to read only that screen and describe your project back to you.
3. Record a 30-second demo GIF and measure the file size. Get it under a few megabytes without losing legibility.

**Self-check:** Exercise 1's secret check matters because a committed key stays in history and in any fork — rotation is the only real fix.

## Go deeper

- [Resume](02-resume.md) — the same evidence, compressed to one page
- [Production Case Studies](08-production-case-studies.md) — turning these repos into interview stories
- [Technical Articles](10-technical-articles.md) — one repo, one post, more reach
- [Stage 22 — Real Projects](../22-real-projects/README.md) — the projects worth pinning

---

[← Previous: Stage 22 — Real Projects](../22-real-projects/README.md) · [Stage index](README.md) · [Next: Resume](02-resume.md)
