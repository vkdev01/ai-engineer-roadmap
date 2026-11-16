# Contributing

This repo is meant to be forked, corrected and extended. If a lesson is wrong, out of date, or explains something badly, that is a bug — fix it.

## The kinds of contributions that help most

1. **Corrections.** A wrong shape, a broken snippet, an SDK call that has moved on. Include the error you hit and the version you hit it with.
2. **Freshness notes.** Model ids, prices and SDK signatures change. If a lesson names something that no longer exists, add a note in the lesson and open a PR.
3. **New worked examples.** A real use case beats another abstract explanation. Keep it specific: what the system did, what broke, what the measurement was.
4. **Checkpoint answers.** Exercises without a self-check are half-finished.
5. **Translations.** Keep the file structure and headings identical, and add a language suffix to the folder (for example `stages-hi/`).

## Rules for a lesson file

Every topic file follows the structure in the repo's authoring spec, in this order:

```text
# N. Title
> stage · you'll be able to · prerequisites
## Why this exists
## The mental model
## Core concepts
## Code
## Real use case
## Common mistakes
## Checkpoint
## Go deeper
---
nav line
```

- 550-850 words of prose plus 40-90 lines of code. Dense beats long.
- Explain **why** before **how**. One-line intuition, then the mechanism, then code.
- Python first, TypeScript where the topic has a Node, browser or API surface.
- Every relative link must resolve. Use `README.md` for the stage index, `../<stage>/<file>.md` across stages.
- No emoji. No hype. No filler introductions.
- **No invented numbers.** No fabricated benchmarks, prices, paper titles, dates or quotes. If a figure is illustrative, label it and show the arithmetic. If a provider price matters, write a placeholder and say to substitute the current one.
- Code fences always closed and always language-tagged.
- Never write inline citation markers into a file. Files have no renderer for them, so they appear to the reader as broken text.

## Rules for a stage README

- Two or three sentences on what the stage is for.
- A `Topics` table listing every file in the folder, in reading order.
- The shipping project, linked.
- Three to five checkable completion criteria.

## Adding a new stage or topic

1. Keep the numbering scheme: `stages/NN-kebab-name/MM-kebab-topic.md`.
2. Add the topic to the stage `README.md` table **and** to the stage table in the root `README.md`.
3. Add it to `PROGRESS.md` so it appears in the tracker.
4. If it introduces a term, add it to `GLOSSARY.md`.
5. Check that the previous file's "Next" link and the following file's "Previous" link point at it.

## Style notes

- Second person, present tense, plain words. "You send the prompt and get tokens back", not "the prompt is transmitted".
- British or American spelling is fine; be consistent within a file.
- Prefer a short table over three paragraphs of comparison.
- Code comments explain the non-obvious line, not the syntax.
- When you are unsure about a fast-moving fact, say so in the text. Honest uncertainty is more useful than confident error.

## Review checklist for a PR

- [ ] Every file follows the required heading order.
- [ ] All relative links resolve.
- [ ] All code fences closed and language-tagged.
- [ ] Snippets that claim to run, run — or are clearly marked as sketches.
- [ ] No invented statistics, prices or quotes.
- [ ] Stage README table and `PROGRESS.md` updated.
- [ ] New terms added to `GLOSSARY.md`.

## Licence

Contributions are accepted under the MIT licence, same as the repo.

[← Back to the roadmap](README.md)
