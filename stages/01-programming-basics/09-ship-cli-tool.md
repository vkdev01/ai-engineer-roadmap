# 9. Ship a CLI Tool

> **Stage 01 · Programming Basics** — turns nine separate lessons into one command another person can install and run.
>
> **You'll be able to:** package a Python project, expose a `summarise` command with arguments and a `--json` flag, and prove it works from a clean checkout.
>
> **Prerequisites:** [JSON and YAML](08-json-yaml.md) · [Git and GitHub](05-git-github.md)

## Why this exists

Reading about async, pydantic and generators changes nothing until you assemble them into something with an entry point. A CLI is the smallest honest version of "shipped": no server, no container, no cloud account — just a command that takes input, does work, and writes a result you can inspect.

The tool you build here, `summarise-cli`, walks a folder of text files, streams them through a batched pipeline, counts tokens per file and per batch, and writes a JSON report plus a Markdown summary. Every idea from this stage appears once, in a place where you can point at it.

The mental model: **a CLI is a function with a contract — arguments in, exit code and files out.**

## The mental model

```text
summarise run ./docs --batch-size 8 --out reports/
        |
        +-- discover  find *.md / *.txt under ./docs        (linux-cli)
        +-- read      one file at a time, never all          (generators)
        +-- batch     chunks of 8 documents                  (generators)
        +-- count     tokens per doc and per batch           (algorithms: one pass)
        +-- validate  every row is a Document model          (pydantic)
        +-- write     reports/run.json + reports/run.md      (json)
                                                             exit 0 / 1 / 2
```

Exit codes are part of the contract: `0` success, `1` bad input (missing folder), `2` partial success (some files failed).

## Core concepts

### Layout matters more than cleverness

A `src/` layout with a `pyproject.toml` means `pip install -e .` works, imports resolve the same way everywhere, and a test can import the package without path hacks.

```text
summarise-cli/
├── pyproject.toml
├── README.md
├── src/
│   └── summarise_cli/
│       ├── __init__.py
│       ├── models.py        # pydantic models: Document, FileResult, Report
│       ├── discover.py      # walk a directory, yield paths
│       ├── counter.py       # token counting + batching
│       └── cli.py           # argparse wiring, exit codes
└── tests/
    └── test_counter.py
```

### `pyproject.toml` declares the command

The `[project.scripts]` table is what makes `summarise` a real command on the user's PATH. No shell wrapper, no `python script.py`.

### `argparse` subcommands beat flag soup

`summarise run ./docs` and `summarise inspect ./docs/file.md` read better than twelve flags. Subparsers also give you free `--help` per command, which is most of your documentation.

### Errors go to stderr, data goes to stdout

This is what makes your tool composable: `summarise run docs --json | jq '.total_tokens'` works only if progress chatter never pollutes stdout. Print human messages to `sys.stderr`.

### Token counting: be honest about what you are counting

Real tokenisation depends on the model's tokenizer, so use the provider's tokenizer library or a documented approximation, and label the number as an estimate in the output. Never present a rough count as exact.

## Code

The package metadata and the counting core. `counter.py` holds the logic you can unit-test without touching the filesystem.

```toml
# pyproject.toml
[project]
name = "summarise-cli"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2", "pyyaml>=6"]

[project.scripts]
summarise = "summarise_cli.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

```python
# src/summarise_cli/counter.py
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from .models import Document, FileResult

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: words plus punctuation marks.

    Swap this for the provider's tokenizer when exact counts matter, and keep
    the field name honest (see `estimated` in models.py).
    """
    return len(_WORD.findall(text))


def batch(docs: Iterable[Document], size: int) -> Iterator[list[Document]]:
    current: list[Document] = []
    for doc in docs:
        current.append(doc)
        if len(current) == size:
            yield current
            current = []
    if current:              # do not drop a short final batch
        yield current


def count_documents(docs: Iterable[Document]) -> list[FileResult]:
    return [
        FileResult(path=d.path, chars=len(d.text), estimated_tokens=estimate_tokens(d.text))
        for d in docs
    ]
```

The CLI entry point: argparse, streaming pipeline, exit codes, two output files.

```python
# src/summarise_cli/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .counter import batch, count_documents
from .discover import iter_documents
from .models import Report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="summarise", description="Count tokens across a folder.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="process a directory")
    run.add_argument("root", type=Path)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--out", type=Path, default=Path("reports"))
    run.add_argument("--json", action="store_true", help="print the report to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 1

    docs = list(iter_documents(args.root))          # discovery is small; texts stream below
    results: list = []
    batches = 0
    for group in batch(iter_documents(args.root), args.batch_size):
        batches += 1
        results.extend(count_documents(group))
        print(f"batch {batches}: {len(group)} files", file=sys.stderr)   # stderr, not stdout

    report = Report(
        root=str(args.root),
        files=results,
        batches=batches,
        total_estimated_tokens=sum(r.estimated_tokens for r in results),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    report.write(args.out / "run.json", args.out / "run.md")

    if args.json:
        sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    failed = len(docs) - len(results)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## How to prove it works

Three checks, in increasing strength. Run them in order on a clean clone.

```bash
# 1. It installs and the command exists
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
summarise --help                      # exit 0, usage printed

# 2. Unit test the logic without the filesystem
pytest -q                             # tests/test_counter.py passes

# 3. End-to-end on real files, with exit codes checked
mkdir -p /tmp/demo && printf 'alpha beta gamma.\n' > /tmp/demo/a.md
summarise run /tmp/demo --out /tmp/out --json | jq '.total_estimated_tokens'
echo $?                               # 0
summarise run /tmp/nope; echo $?      # error on stderr, exit 1
```

What good output looks like: `reports/run.json` validates against `Report`, `reports/run.md` has a table of files with token estimates, and the exit code is `0` for a clean run, `1` for a missing directory, `2` when some files were skipped.

The test that makes the tool trustworthy:

```python
# tests/test_counter.py
from summarise_cli.counter import batch, estimate_tokens
from summarise_cli.models import Document


def test_estimate_tokens_counts_words_and_punctuation() -> None:
    assert estimate_tokens("hello, world") == 3      # 2 words + 1 comma


def test_batch_keeps_short_final_batch() -> None:
    docs = [Document(path=f"f{i}.md", text="x") for i in range(5)]
    sizes = [len(b) for b in batch(docs, 2)]
    assert sizes == [2, 2, 1]
```

## Real use case

A content team wanted to know which of their 3,000 Markdown pages would fit in a model's context window before an indexing project started. Doing it by hand was out of the question, and an earlier ad-hoc script had printed counts that turned out to include front-matter and code blocks, so every estimate was wrong in the same direction.

`summarise-cli` became the answer: the team ran `summarise run content/ --batch-size 32 --out reports/`, got a JSON report they loaded into a spreadsheet, and a Markdown summary they pasted into the planning doc. Two design choices made it usable rather than a one-off: progress went to stderr so the `--json` output could be piped into `jq`, and the field is called `estimated_tokens` rather than `tokens`, so nobody mistook it for a billing figure.

Six months later the same command runs in CI on every content PR, failing the build when a page exceeds the context budget.

## Common mistakes

- **Mistake:** flat layout with `from counter import ...` that only works from one directory. — **Instead:** `src/` layout plus `pip install -e .`.
- **Mistake:** printing progress to stdout. — **Instead:** stderr for humans, stdout for data.
- **Mistake:** `sys.exit(1)` for every failure. — **Instead:** distinct codes for bad input and partial success, documented in `--help`.
- **Mistake:** calling a token count exact when it is an approximation. — **Instead:** name the field `estimated_tokens` and say which tokenizer would be exact.
- **Mistake:** no tests, so the first refactor breaks the batch boundary. — **Instead:** unit-test the pure functions and one end-to-end run.

## Checkpoint

1. Add an `inspect` subcommand that prints one file's token estimate and first 200 characters.
2. Make the report include the five largest files by token count, using a heap rather than a full sort.
3. Add a `--fail-over-tokens N` flag that exits non-zero when any file exceeds `N`, and wire it into a CI step.

**Self-check:** a subparser added with `sub.add_parser("inspect")` plus an `args.command == "inspect"` branch in `main` is enough; keep the exit codes consistent with `run`.

## Go deeper

- [Stage 02 — Math and ML Basics](../02-math-and-ml-basics/README.md) — next stage; the maths behind the models your tool will call.
- [Tokenization](../06-llm-fundamentals/02-tokenization.md) — why the estimate in this tool is an estimate.
- [GitHub Actions](../19-cloud-devops/06-github-actions.md) — running `summarise` on every pull request.

---
[← Previous: JSON and YAML](08-json-yaml.md) · [Stage index](README.md) · [Next: Stage 02 — Math and ML Basics](../02-math-and-ml-basics/README.md)
