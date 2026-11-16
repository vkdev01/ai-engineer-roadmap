# Stage 01 — Programming Basics

This stage turns "I can read a little Python" into "I can ship a small tool that other people can run." Everything later in the roadmap — calling an LLM API, loading a dataset, deploying a service — is built from the nine ideas in this stage. Nothing here is AI-specific, which is exactly why it comes first: you cannot debug a retry loop in an agent if you have never written a retry loop.

**Time:** roughly 3-4 weeks part-time · **You need first:** nothing — this is the entry point · **You leave with:** a working CLI tool that counts tokens and summarises batches of files, published on GitHub

## What you will be able to do

- Write `async` Python that fetches many things at once with timeouts and per-item error handling.
- Model your data with type hints and pydantic v2 so bad input fails at the edge, not three layers deep.
- Stream large files through generators instead of loading them into memory.
- Reason about the handful of algorithms that show up in interviews and in real pipelines.
- Use git and the Linux command line without copying commands you do not understand.
- Talk to an HTTP API, read its JSON, and validate it — from both Python and TypeScript.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Python Async](01-python-async.md) | Concurrency for I/O-bound work without threads |
| 2 | [Type Hints and Pydantic v2](02-type-hints-pydantic.md) | Models that validate, parse and document themselves |
| 3 | [Generators](03-generators.md) | Lazy iteration and memory-safe pipelines over big data |
| 4 | [Core Algorithms](04-algorithms.md) | Complexity, hashing, sorting, searching, two pointers |
| 5 | [Git and GitHub](05-git-github.md) | Commits, branches, pull requests, review, recovery |
| 6 | [The Linux Command Line](06-linux-cli.md) | Pipes, text tools, processes, permissions, SSH |
| 7 | [HTTP and REST APIs](07-http-rest-apis.md) | Requests, status codes, headers, pagination, retries |
| 8 | [JSON and YAML](08-json-yaml.md) | The two data formats every config and API speaks |
| 9 | [Ship a CLI Tool](09-ship-cli-tool.md) | Package the above into an installable command |

## The shipping project

[09-ship-cli-tool.md](09-ship-cli-tool.md) — you build `summarise-cli`, a pip-installable command that walks a folder of text files, streams them through a batched pipeline, counts tokens, and writes a JSON report plus a Markdown summary. You prove it works with a smoke test, a typed report file, and a clean `git` history someone else can clone and run.

## How to know you are done

- Your CLI installs with `pip install -e .` and runs as `summarise --help` with no traceback.
- Every data structure crossing a boundary in your code is a pydantic model, and invalid input raises a readable validation error.
- Your repo has at least five meaningful commits on a branch that was merged through a pull request.
- You can explain out loud why `asyncio.gather` is not parallelism, and where it would not help.
- The CLI's output report validates against a TypeScript contract you also wrote.

---
[← Back to roadmap](../../README.md) · [Next stage: Math and ML Basics](../02-math-and-ml-basics/README.md)
