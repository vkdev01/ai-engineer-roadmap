# 5. Prompt Versioning as Code

> **Stage 08 · Prompt and Context** — prompt changes as reviewable, reversible artefacts.
>
> **You'll be able to:** give every prompt an id and version, diff changes, tie answers to the version that produced them, and roll back safely.
>
> **Prerequisites:** [Prompt Templates with Variables and Validation](03-templates.md) · [Git and GitHub](../01-programming-basics/05-git-github.md)

## Why this exists

Prompts change more often than code and, unlike code, they often change without a test failing. Someone tweaks a sentence in a dashboard, quality shifts, and nobody can reconstruct what the prompt looked like when the bad answers started. The result is a class of bug that is invisible in your repo history because the prompt never lived there.

Versioning fixes it by making the prompt a file with an id, a version and a diff — and by recording which version produced each answer. That single log field turns "the assistant got worse last Tuesday" into a query.

The mental model: **a prompt is code: it has a name, a version, a diff and a rollback.**

## The mental model

```text
prompts/
  ticket-answer/
    v3.yaml     <- current
    v2.yaml     <- kept for comparison
    CHANGELOG.md

request --> template_id + version + content_hash
              |
              v
          answer log row: {request_id, template_id, version, hash, score}
```

| Practice | Why it matters |
|---|---|
| Prompt in the repo, not a dashboard | it gets reviewed and diffed |
| Monotonic versions, old ones kept | you can A/B and roll back |
| Version in every log row | you can attribute a regression |
| Content hash alongside version | catches edits that skipped the version bump |

## Core concepts

### The version is an identity, not a label

`v3` should mean exactly one text. If two different texts share a version, attribution is broken. Enforce that with a content hash stored next to the version and a CI check that recomputes it.

### Keep the old versions

Do not edit `v3` in place. Add `v4`. Old files cost nothing and let you run a comparison: same fixtures, both versions, side-by-side scores. Deleting the previous version makes "was this change good?" unanswerable.

### Prompts are configuration, so treat config carefully

A prompt in a YAML file next to its schema and fixtures is reviewable. A prompt in a vendor dashboard is not, and it also cannot be tested in CI. If you must edit in a UI, mirror it back into the repo on every save and treat the repo as the source of truth.

### Version the model id with the prompt

A prompt tuned for one model is not automatically right for another. Record the model id alongside the version in your answer logs, and re-run your eval set when either changes. A "prompt regression" is often a model migration in disguise.

### Rollback must be a one-line change

If rolling back requires editing prompt text from memory, you will not do it under pressure. Keep the previous version file present and make the active version a config value, so rollback is a config edit and a deploy.

## Code

A prompt file with its metadata, a loader that verifies the hash, and a registry that makes the active version explicit.

```yaml
# prompts/ticket-answer/v3.yaml
id: ticket-answer
version: v3
model: set-model-id-in-config   # check provider docs for the current id
temperature: 0.2
variables:
  question: {type: string, max: 2000, required: true}
  context: {type: string, max: 20000, default: ""}
  product_area: {type: enum, values: [general, billing, auth, api], default: general}
body: |
  Answer the question for the {product_area} area.

  <context>
  {context}
  </context>

  <question>
  {question}
  </question>
changelog: "v3: added product_area routing and a strict context delimiter."
```

The loader recomputes the hash on every load, so an edit that skipped the version bump fails loudly instead of silently changing behaviour.

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptFile:
    id: str
    version: str
    body: str
    model: str
    temperature: float
    content_hash: str


def load_prompt(path: Path) -> PromptFile:
    data = yaml.safe_load(path.read_text())
    body = data["body"]
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]

    # The version must identify exactly one body. Store the expected hash in
    # the file and compare, so an edit without a bump cannot slip through.
    expected = data.get("body_sha256")
    if expected and expected != digest:
        raise ValueError(
            f"{path}: body changed without a version bump "
            f"(expected {expected}, got {digest})"
        )
    return PromptFile(
        id=data["id"],
        version=data["version"],
        body=body,
        model=data["model"],
        temperature=float(data.get("temperature", 0.2)),
        content_hash=digest,
    )


ACTIVE = {"ticket-answer": "v3"}   # rollback = change this string


def load_active(prompt_id: str, root: Path = Path("prompts")) -> PromptFile:
    version = ACTIVE[prompt_id]
    return load_prompt(root / prompt_id / f"{version}.yaml")
```

Every answer log row then carries the attribution fields, which is what makes the version worth having.

```python
import time, uuid


def log_answer(prompt: PromptFile, request_id: str, score: float | None) -> dict:
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "template_id": prompt.id,
        "template_version": prompt.version,
        "content_hash": prompt.content_hash,
        "model": prompt.model,
        "score": score,          # filled in by evals or by human review
        "ts": time.time(),
    }
```

In TypeScript, keep prompts as JSON modules so the version travels with the import and the type system can check the shape.

```ts
import { z } from "zod";

export const PromptFileSchema = z.object({
  id: z.string(),
  version: z.string(),
  model: z.string(),
  temperature: z.number().default(0.2),
  body: z.string(),
  changelog: z.string().optional(),
});
export type PromptFile = z.infer<typeof PromptFileSchema>;

// Static imports: the version is visible at the call site and in code review.
import ticketV3 from "../prompts/ticket-answer/v3.json" with { type: "json" };

export const ACTIVE: Record<string, PromptFile> = {
  "ticket-answer": PromptFileSchema.parse(ticketV3),
};

export function promptFor(id: string): PromptFile {
  const p = ACTIVE[id];
  if (!p) throw new Error(`no active prompt for ${id}`);
  return p; // caller logs p.id + p.version with every request
}
```

## Real use case

An ed-tech company's tutor assistant used a prompt edited directly in a vendor dashboard by a content lead. Quality was fine for weeks, then complaints about wrong syllabus references appeared. Nobody could say what had changed, because there was no history and no version in the logs.

The team moved prompts into the repo as YAML with ids and versions, added a `body_sha256` check, and put `template_version` in every answer log. The next complaint was resolved in minutes: filtering logs by version showed the regression began the day `v4` shipped, and comparing `v3` and `v4` on the fixture set showed the syllabus table had been dropped from the context block. The failure mode before was untracked state — the prompt was production code living outside version control.

## Common mistakes

- **Mistake:** editing an existing prompt version in place. — **Instead:** add a new version file and keep the old one for comparison.
- **Mistake:** leaving prompts in a vendor dashboard as the source of truth. — **Instead:** mirror them into the repo and deploy from there.
- **Mistake:** logging only the template id. — **Instead:** log id, version, content hash and model id on every request.
- **Mistake:** changing the model without re-running the eval set. — **Instead:** treat a model migration as a prompt change and compare versions on fixtures.
- **Mistake:** making rollback a manual text edit. — **Instead:** keep the active version in config so rollback is one string.

## Checkpoint

1. Add a `body_sha256` to an existing prompt file, then edit the body without changing the version and confirm the loader raises.
2. Run the same ten fixtures against `v2` and `v3` and produce a side-by-side score table.
3. Implement `ACTIVE` as an environment override so you can pin a version per environment without a code change.

**Self-check:** you add a new version file and keep the old one, because a version must identify exactly one body and the previous text is needed for comparison and rollback.

## Go deeper

- [Prompt Evals](07-prompt-evals.md) — the measurement that makes a version bump meaningful.
- [Prompt Templates with Variables and Validation](03-templates.md) — the template engine these files feed.
- [Prompt Versioning in MLOps](../18-mlops-llmops/03-prompt-versioning.md) — the operational side, with registries.
- [Git and GitHub](../01-programming-basics/05-git-github.md) — the review workflow prompts should go through.

---
[← Previous: Context Compression](04-context-compression.md) · [Stage index](README.md) · [Next: Prompt-Injection Defence](06-injection-defense.md)
