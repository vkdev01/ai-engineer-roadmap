# 6. The Linux Command Line

> **Stage 01 · Programming Basics** — the interface every server, container and CI job actually gives you.
>
> **You'll be able to:** find files, inspect a huge log, pipe tools together, and keep a long-running process alive on a remote machine.
>
> **Prerequisites:** [Git and GitHub](05-git-github.md) · [Generators](03-generators.md)

## Why this exists

Your laptop has a friendly interface. The machine your model runs on does not. When a container exits with code 137, when a log file is 12 GB, when you need to know which process is holding port 8000, the answer is four shell commands, not a settings dialog. Every cloud instance, Docker image and CI runner you touch for the rest of this roadmap speaks this language.

The good news: you need about twenty commands and one idea. The idea is that small programs doing one thing compose through pipes.

The mental model: **a pipe turns a big problem into a chain of small ones.**

## The mental model

```text
cat app.log | grep ERROR | cut -d' ' -f3 | sort | uniq -c | sort -rn | head -5
    read        filter       reshape     order   count      rank      show top
```

Each stage reads from stdin, writes to stdout, and knows nothing about the others. `>` sends output to a file, `>>` appends, `|` hands it to the next program.

| Need | Command |
|---|---|
| Where is that file | `find . -name "*.jsonl"` |
| Which text is in it | `rg "timeout"` |
| Slice columns | `cut -d, -f1,3`, `awk '{print $2}'` |
| Sort and count | `sort`, `uniq -c` |
| Replace text | `sed 's/old/new/g'` |
| Watch a growing file | `tail -f app.log` |
| Disk and memory | `df -h`, `du -sh *`, `free -h` |
| Processes | `ps aux`, `top`, `kill -TERM <pid>` |

## Core concepts

### Everything is a file, and stdin/stdout are your seams

Programs read from file descriptor 0 and write to 1; errors go to 2. That is why `2>/dev/null` hides errors while `1>out.txt` captures normal output, and why a command can be a filter in the middle of a pipeline.

### `grep` and `rg` are for asking questions of text

`grep -n -i -C 3 "error"` gives line numbers, case-insensitivity and context. `rg` (ripgrep) is faster and skips `.git` by default. Learn `-l` (files only), `-c` (count) and `-v` (invert) — most log questions are answered by combining two of those.

### `find` selects files; `xargs` applies a command to them

`find . -name "*.py" -newer README.md` finds files; `-exec` or `xargs` runs something on each. Always test with `-print` first, because a destructive `-exec rm` does exactly what you asked.

### Processes, signals and exit codes

`ps aux | grep uvicorn` finds a server; `kill <pid>` sends SIGTERM (a polite request to stop) and `kill -9` sends SIGKILL (no cleanup, no flush). Exit code 0 means success; anything else means failure, and the number matters — 137 means killed, often out of memory.

### `nohup`, `tmux` and `&` for work that outlives your SSH session

A process started in a terminal dies with it. `nohup cmd > out.log 2>&1 &` survives; `tmux` gives you a session you can reattach to and read scrollback. Long training or ingestion jobs belong in tmux or a proper job runner, never in a bare SSH shell.

### Permissions in one line

`rwx` for owner, group, others. `chmod +x script.sh` makes it runnable; `chmod 600 key.pem` makes a key readable only by you — which is what `ssh` demands.

## Code

A realistic triage of an application log. This answers "which errors are most common, and when did they start" without opening the file.

```bash
# Top 5 error messages by frequency
grep -i "error" app.log \
  | cut -d']' -f3- \
  | sed 's/[0-9]\+/N/g' \
  | sort | uniq -c | sort -rn | head -5

# First and last occurrence of a specific error, with timestamps
grep -n "TimeoutError" app.log | head -1
grep -n "TimeoutError" app.log | tail -1

# How big is this directory, by child, largest first
du -sh ./* 2>/dev/null | sort -h | tail -10

# What is listening on port 8000
ss -ltnp | grep 8000
```

Running a long ingestion job on a remote box and keeping it readable afterwards:

```bash
# Start a detached session, then run work inside it
tmux new -s ingest
python -m token_cli batch data/raw --out reports/run1.json

# Detach with Ctrl-b then d. Later, from any shell:
tmux attach -t ingest
```

A small script that shows pipes, variables and exit codes together — the shape of every shell script you will write.

```bash
#!/usr/bin/env bash
set -euo pipefail   # exit on error, on unset var, and on failed pipe stage

input_dir="${1:?usage: count.sh <dir>}"
total=0

while IFS= read -r file; do
  lines=$(wc -l < "$file")
  total=$((total + lines))
  printf '%8d  %s\n' "$lines" "$file"
done < <(find "$input_dir" -name '*.md' -type f)

echo "total lines: $total"
```

## Real use case

A nightly job that embeds new support articles started failing with no useful output. The cron entry sent stdout to `/dev/null` and stderr to nowhere, so all the team knew was "no new articles in search this morning". The container restarted around 2 a.m. with exit code 137.

The investigation was pure command line: `du -sh` showed the raw export directory had grown far beyond expectation because a new upstream feed never deleted old files; `tail -n 200` of the container log showed the process being killed mid-batch; `free -h` on the host showed memory exhausted by other work. Redirecting to a real log file (`>> /var/log/ingest.log 2>&1`), adding a `find -mtime +7 -delete` cleanup step, and running the job inside `tmux` instead of a bare SSH session turned a mystery into a five-minute check each morning.

The design choice that mattered was writing output and errors to a file that outlives the process — the failure mode before was not a crash, it was an absence of evidence.

## Common mistakes

- **Mistake:** `rm -rf $DIR/` with `DIR` unset. — **Instead:** `set -u` in every script, and `"${DIR:?unset}"` when it matters.
- **Mistake:** piping through `grep` without `-i` and concluding there are no errors. — **Instead:** `grep -i -c` first, then narrow.
- **Mistake:** `kill -9` as the first resort. — **Instead:** SIGTERM lets the process flush and clean up; use `-9` only after it refuses.
- **Mistake:** running a long job in a plain SSH shell. — **Instead:** `tmux` or `nohup ... > log 2>&1 &`.
- **Mistake:** chmod 777 to fix a permission error. — **Instead:** grant the minimum; `600` for keys, `755` for scripts and dirs.

## Checkpoint

1. Write a pipeline that prints the five most frequent words in a Markdown file, ignoring case.
2. Find every file over 50 MB under a directory and print its size in human units.
3. Write a script that takes a directory, counts files and total bytes, exits non-zero if the directory is missing, and is safe under `set -euo pipefail`.

**Self-check:** `tr ' ' '\n' < file.md | tr -d '[:punct:]' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn | head -5`.

## Go deeper

- [Git and GitHub](05-git-github.md) — version control, run from the same shell.
- [Docker](../19-cloud-devops/04-docker.md) — where knowing `ps`, `kill` and exit codes pays off daily.
- [GPU Cloud](../19-cloud-devops/08-gpu-cloud.md) — the SSH-and-tmux workflow you will use on rented machines.

---
[← Previous: Git and GitHub](05-git-github.md) · [Stage index](README.md) · [Next: HTTP and REST APIs](07-http-rest-apis.md)
