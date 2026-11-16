# 6. Coding Assistant

> **Stage 22 · Real Projects** — an assistant that reads the right part of a repository, writes a patch, and proves it with tests in a sandbox before you see it.
>
> **You'll be able to:** build a coding assistant whose context is selected by symbols rather than by similarity, whose output is a patch rather than prose, and whose claims are backed by a test run.
>
> **Prerequisites:** [Sandboxing](../20-ai-security/05-sandboxing.md) · [Tool Calling](../07-llm-apis/05-tool-calling.md)

## Why this exists

Pasting a file into a chat window works for a snippet and fails for a change. Real changes touch several files, depend on a helper three imports away, and break a test that encodes a rule nobody documented. A model given only the file you pasted will confidently write code that calls a function with the wrong signature.

Two things make this hard, and neither is the prompt. First, context selection: a mid-size repo does not fit in a context window, so you must choose the few files and definitions that matter, using the code's own structure. Second, verification: a patch that looks right and does not pass tests is worse than no patch, because reviewing it costs as much as writing it.

Buyers are engineering teams measuring cycle time, and open-source maintainers drowning in small issues. Both want a diff they can review in a minute, with evidence.

Mental model: the repository is a *graph of symbols*, the change is a *diff*, and correctness is a *test exit code*, not an opinion.

## The mental model

```text
issue text
    |
    v
[symbol index: definitions, imports, tests]  <-- built by parsing, not by embedding alone
    |
    v
[context builder: entry points + callers + tests, token-budgeted]
    |
    v
[planner: files to touch, plan]
    |
    v
[patch generator: unified diff, one hunk set per file]
    |
    v
[sandbox: apply patch, run targeted tests, capture failures]
    |
    +--> tests fail --> feed failures back (max 2 rounds)
    |
    +--> tests pass --> present diff + test output
```

## Core concepts

### Scope and non-goals

In scope: a Python (or TypeScript) repository, an issue in text, symbol indexing, context assembly, patch generation, sandboxed test execution, and up to two repair rounds.

Non-goals: multi-repo changes, dependency upgrades, refactors across hundreds of files, and running on untrusted code without isolation. Also non-goal: replacing the reviewer — the output is a pull request draft, not a merge.

### Context by structure, not similarity

Embeddings find files that *talk about* the same words. Code needs files that *call* the same symbols. Index definitions (functions, classes, methods) with file and line, then build a graph: what this symbol imports, what imports it, and which tests reference it. For an issue, retrieve a seed set by text similarity, expand one hop along the graph, and include the tests that cover the seeds. This is the single highest-leverage decision in the project.

Then budget it. Rank candidates by (a) direct mention in the issue, (b) test file for a touched symbol, (c) caller/callee, and cut at a token budget. Show the model a file header comment plus the relevant definitions, not whole files, and tell it explicitly which lines it may change.

### Output a patch, not a file

Ask for a unified diff. It is reviewable, it is small, and it fails loudly if the context drifted. Apply it with `git apply --3way` inside the sandbox; a patch that does not apply is a signal that the model's view of the file was stale, and the retry should re-read the file rather than re-emit the same diff.

### The sandbox is the product

Run tests in a container with no network, a read-only mount of the repo, a writable temp copy, a CPU and memory cap, and a hard timeout. Targeted tests only — running the full suite makes every iteration slow and flaky. Feed failures back as structured text (test name, assertion, file, line), not as a wall of logs; the model repairs faster with the signal.

### Milestones, data model, eval plan

Four weeks: week 1 symbol index and context builder; week 2 patch generation and application; week 3 sandbox and test loop; week 4 eval harness and demo.

| table | key fields |
|-------|-----------|
| `repo` | id, url, sha, language |
| `symbol` | id, repo_id, name, kind, file, line_start, line_end |
| `task` | id, repo_id, issue_text, status, rounds |
| `attempt` | id, task_id, round, diff, tests_passed, failure_json, cost_usd |

Eval: collect a set of real merged pull requests and revert them to create tasks with a known-good patch. Then measure **apply rate** (patch applies cleanly), **test pass rate** (targeted tests green), **regression rate** (previously passing tests still pass), **review edit distance** (how much a human changed your diff — the honest quality number), and **cost per task**. Report apply rate and test pass rate separately; conflating them hides which stage is broken.

Budget, illustrative: context of roughly 25k input tokens, a 1.5k-token diff output, and two repair rounds at 8k each — so around 50k tokens per task. Cost per task is dominated by repair rounds, so the lever is better context on round one.

## Code

The context builder is where the quality lives. This builds a symbol index with `ast`, then expands a seed set along the call/import graph within a token budget.

```python
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbol:
    name: str
    kind: str                 # function | class | method
    path: str
    start: int
    end: int
    calls: set[str] = field(default_factory=set)


def index_repo(root: Path) -> dict[str, Symbol]:
    """Map symbol name -> Symbol, parsing only; no execution."""
    index: dict[str, Symbol] = {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                calls = {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                index[node.name] = Symbol(
                    name=node.name,
                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                    path=str(path.relative_to(root)),
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    calls=calls,
                )
    return index


def expand(seeds: list[str], index: dict[str, Symbol], budget_chars: int) -> list[Symbol]:
    """Seeds, then their callees, then anything that calls them — until budget."""
    chosen: dict[str, Symbol] = {}
    used = 0
    def add(name: str) -> bool:
        nonlocal used
        sym = index.get(name)
        if not sym or name in chosen:
            return True
        cost = len(sym.path) + (sym.end - sym.start) * 60   # rough chars per line
        if used + cost > budget_chars:
            return False
        chosen[name] = sym
        used += cost
        return True

    for name in seeds:
        add(name)
    for name in list(chosen):                       # one hop down
        for callee in index[name].calls:
            add(callee)
    for name, sym in index.items():                 # one hop up, cheapest last
        if any(c in chosen for c in sym.calls):
            add(name)
    return list(chosen.values())
```

Then the sandbox step, which turns a diff into evidence:

```python
import asyncio

async def run_tests(workdir: str, tests: list[str], timeout_s: int = 120) -> tuple[bool, str]:
    """No network, fixed timeout, targeted tests only."""
    proc = await asyncio.create_subprocess_exec(
        "python", "-m", "pytest", "-x", "-q", *tests,
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={"PATH": "/usr/local/bin:/usr/bin", "NO_NETWORK": "1"},
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        return False, f"timeout after {timeout_s}s"
    text = out.decode(errors="replace")
    return proc.returncode == 0, text[-4000:]     # tail only: failures live at the end
```

## Real use case

A team maintaining a Python library uses the assistant on "good first issue" style tickets: add a validation branch, fix an error message, handle an edge case. Before, contributors waited for a maintainer to explain where the code lives; the assistant's symbol index answers that in the first message.

The design choice that made it work is the graph expansion: the model receives the target function, its callee, and the test that covers it — so patches apply and tests pass on the first round far more often than with similarity-only retrieval. The failure mode before was a plausible diff against a function signature that had changed.

Demo script, two minutes: paste an issue, show the selected context (three files, not thirty), show the diff, show the test run output, then show a repair round on a deliberately broken patch. Talking points: structure-based retrieval, why diffs, what the sandbox blocks, and how you measured review edit distance.

## Common mistakes

**Mistake:** stuffing whole files into context — **Instead:** send definitions and line ranges; whole files crowd out the test that matters.
**Mistake:** embedding-only retrieval for code — **Instead:** build the symbol graph; callers and callees are the real context.
**Mistake:** asking for a full rewritten file — **Instead:** ask for a unified diff; it is reviewable and its failure is obvious.
**Mistake:** running the full test suite each round — **Instead:** targeted tests plus one regression pass at the end.
**Mistake:** feeding back full logs — **Instead:** structured failures (test, assertion, file, line) so the model repairs the right thing.
**Mistake:** running untrusted code on your host — **Instead:** container, no network, timeout, memory cap; see the sandboxing file.

## Checkpoint

1. Add the tests that reference a seed symbol to the context automatically. Measure apply rate before and after.
2. Add a regression pass that runs the full suite once after the targeted tests pass, and report how often it catches a break.
3. Compare two context strategies (similarity-only vs graph-expanded) on ten tasks and write the table. Which one costs more tokens per successful patch?

**Self-check:** Exercise 1 typically lifts apply rate because the test file names the expected behaviour and the exact call signature.

## Go deeper

- [Harness Engineering](../11-agents/01-harness-engineering.md) — the agent loop your patch generator sits inside
- [Agent Design](../21-system-design/03-agent-design.md) — describe this system under interview pressure
- [Sandboxing](../20-ai-security/05-sandboxing.md) — the isolation rules that keep this safe
- [Unit Tests](../15-evals/01-unit-tests.md) — turning test output into a gradeable signal

---

[← Previous: Support Agent](05-support-agent.md) · [Stage index](README.md) · [Next: Multimodal Assistant](07-multimodal-assistant.md)
