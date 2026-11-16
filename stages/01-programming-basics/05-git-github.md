# 5. Git and GitHub

> **Stage 01 · Programming Basics** — gives your work a history you can branch, review, share and undo.
>
> **You'll be able to:** work on a feature in its own branch, open a pull request, review your own diff before anyone else sees it, and recover a mistake without panic.
>
> **Prerequisites:** [Core Algorithms](04-algorithms.md) · [The Linux Command Line](06-linux-cli.md)

## Why this exists

Without version control, collaboration looks like `final_v2_notes_updated(1).docx` and code that only exists on one laptop. You cannot tell what changed, why, or who broke it. With git, every change is a labelled snapshot with an author and a reason, and every experiment can happen on a branch that leaves `main` untouched.

GitHub adds the social layer: pull requests turn a change into a conversation with a diff attached, which is how essentially every engineering team works.

The mental model: **commits are snapshots; branches are movable labels pointing at a snapshot; `main` is the branch that must always work.**

## The mental model

```text
main      A --- B --- C ---------------- F   (merge commit)
                   \                    /
feature/login       D --- E ----------
                          ^
                          your branch label, moving as you commit
```

| Area | What it holds | Command that moves work into it |
|---|---|---|
| Working tree | your edited files | (editing) |
| Staging area | what the next commit will contain | `git add` |
| Repository | committed snapshots | `git commit` |
| Remote | a copy on GitHub | `git push` |

## Core concepts

### A commit is a snapshot, not a diff

Git stores the state of the tree, then computes diffs for you to read. That is why reverting to an old commit is cheap and safe. A good commit is one idea: message in the imperative — "add retry to embedding client" — and small enough to review.

### The staging area is a staging area on purpose

`git add -p` lets you stage part of a file. This is how you keep an unrelated debugging print out of a commit that is about to be reviewed. Staging also means you can commit before your code is finished.

### Branches are cheap, so use them

`git switch -c feat/token-cli` costs nothing and isolates the work. Never develop on `main`: `main` is what you deploy, and every merge into it should be reviewed and green.

### `git diff` before you push, always

The single habit that separates tidy history from noise: run `git diff --staged` and read it. You will find the leftover `print("here")`, the commented-out block, the accidentally committed `.env`.

### Pull requests are a review surface, not a formality

A PR shows the diff, the commits, and the checks. Small PRs get real reviews; a 900-line PR gets a thumbs-up. Write the description as "what changed, why, how I verified" — your future self reads it in six months.

### Undo is routine, not an emergency

| Situation | Command | What it does |
|---|---|---|
| Unstage a file | `git restore --staged f.py` | keeps your edits |
| Discard local edits | `git restore f.py` | destroys uncommitted changes |
| Fix the last message | `git commit --amend` | rewrites the last commit (unpushed only) |
| Undo a pushed commit | `git revert <sha>` | new commit that reverses it |
| Recover after a bad reset | `git reflog` | shows where HEAD has been |

## Code

The whole loop, as a script you can read top to bottom. It assumes a repo already exists on GitHub and you have cloned it.

```bash
# 1. Start from an up-to-date main
git switch main && git pull --ff-only

# 2. Branch for one piece of work
git switch -c feat/token-cli

# 3. Commit in small, reviewable steps
git add src/token_cli/counter.py
git commit -m "add token counting for a single file"

git add src/token_cli/cli.py tests/test_counter.py
git commit -m "wire counter into CLI with a --json flag"

# 4. Read your own diff before anyone else does
git diff main...HEAD

# 5. Push and open a pull request
git push -u origin feat/token-cli
gh pr create --title "Add token counting CLI" --body "Counts tokens per file and per batch. Verified with tests/test_counter.py and a manual run on 3 sample files."
```

A `.gitignore` that stops the usual accidents, plus a quick way to see what you are about to commit:

```bash
# .gitignore
.venv/
__pycache__/
*.pyc
.env
data/raw/
*.parquet

# what is staged, in one line each
git diff --staged --stat
```

When two branches touch the same lines, resolve deliberately:

```bash
git switch feat/token-cli
git fetch origin
git rebase origin/main      # replay your commits on top of latest main
# resolve conflicts in the named files, then:
git add <resolved-files>
git rebase --continue
git push --force-with-lease # safe force: refuses if someone else pushed
```

## Real use case

A two-person team building an internal RAG service kept everything on `main` and pushed directly. One Friday a change to the chunking function shipped with a stray `break` that made every document produce exactly one chunk. Retrieval quality fell through the floor, nobody knew which commit did it, and the fix took a day of bisecting by hand because the history read `update`, `fix`, `fix2`, `final`.

They moved to a light process: one branch per change, PR with a one-paragraph description, and a CI job that ran the eval script. The next chunking change was caught in review because the diff was six lines and the description said "changes overlap from 50 to 100 tokens". When a similar bug did slip through months later, `git log -p -- src/chunking.py` pointed at the exact commit in under a minute.

The design choice that mattered was small PRs on branches: it made the diff readable, which is what turned review from a formality into a real check.

## Common mistakes

- **Mistake:** committing `.env` or API keys. — **Instead:** add them to `.gitignore` before the first commit; if pushed, rotate the key — history is public.
- **Mistake:** one giant commit called "changes". — **Instead:** one idea per commit, imperative message.
- **Mistake:** `git push --force` on a shared branch. — **Instead:** `--force-with-lease`, and only on your own branch.
- **Mistake:** merging without pulling first, then resolving conflicts in a panic. — **Instead:** `git fetch` and rebase early and often.
- **Mistake:** deleting a branch after a bad merge and losing the work. — **Instead:** `git reflog` finds it; nothing committed is truly lost for weeks.

## Checkpoint

1. Create a branch, make two commits, then squash the message of the second with `--amend` and verify with `git log --oneline`.
2. Deliberately edit the same line on two branches, merge, and resolve the conflict by hand.
3. Use `git reflog` to recover a commit after a `git reset --hard HEAD~1`.

**Self-check:** `git commit --amend` rewrites the most recent commit; it is safe before pushing and needs `--force-with-lease` after.

## Go deeper

- [The Linux Command Line](06-linux-cli.md) — the shell skills that make git comfortable.
- [GitHub Portfolio](../23-career/01-github-portfolio.md) — turning your commit history into something a hiring manager reads.
- [GitHub Actions](../19-cloud-devops/06-github-actions.md) — running tests and evals on every pull request.

---
[← Previous: Core Algorithms](04-algorithms.md) · [Stage index](README.md) · [Next: The Linux Command Line](06-linux-cli.md)
