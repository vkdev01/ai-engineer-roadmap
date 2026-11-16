# 6. Supply Chain Security

> **Stage 20 · AI Security** — your app is mostly other people's artefacts: packages, models, datasets and prompts; this file is how you know what you are actually running.
>
> **You'll be able to:** pin and verify what goes into a build, check model and dataset licences before you ship, and refuse to load an untrusted model file.
>
> **Prerequisites:** [Open vs Closed Models](../06-llm-fundamentals/08-open-vs-closed.md) · [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md)

## Why this exists

An LLM app has a dependency tree most teams never draw: the Python or Node packages you import, the base images your containers build on, the model weights you load, the tokenizer and its config, the datasets your evals run on, and the prompts you copy from a blog post. Each one is code or data you did not write, running with your privileges.

The classic failure is a malicious package release — a typosquatted name, or a compromised maintainer account publishing a new version of a real package. In the AI stack there is a second failure the classic tools do not cover: a model file. Modern model formats can carry executable Python (custom layers, remote code options) and pickle-based formats can execute arbitrary code on load. "Download the model from a Hugging Face repo and load it" is, strictly, running untrusted code with your GPU and your filesystem.

The mental model: **the build is a claim about provenance. Every artefact needs a name, a version, a digest, a licence and a reviewer — or it does not go in.**

## The mental model

```text
   source ──────────► build ──────────► runtime
     │                  │                 │
  package lock      image digest      model weights
  (hash-pinned)     (pinned tag+sha)   (sha256 + format)
  model repo id      SBOM generated     tokenizer config
  dataset revision                      prompt version
     │                  │                 │
     └──────── every artefact: pinned, hashed, licensed, recorded ────────┘
                              │
                        SBOM + licence file + model card
```

| Artefact | Pin it how | Verify it how |
|---|---|---|
| Python packages | lock file with hashes | `pip install --require-hashes`, CI re-resolves |
| Node packages | `package-lock.json` / `pnpm-lock.yaml` | `npm ci`, not `npm install` |
| Base image | digest, not `:latest` | pull by `@sha256:...` |
| Model weights | repo id + revision + file sha256 | verify digest before load |
| Dataset | dataset revision / commit | record it in the eval config |
| Prompt | versioned file, reviewed | same PR review as code |

## Core concepts

### Pin everything, and make CI prove the pin

A lock file that CI does not enforce is decoration. The pattern that works: lock with hashes, install with hash verification, and fail the build if resolution would change anything not committed. For images, pin by digest so a mutable tag cannot swap the base under you. For anything downloaded at runtime, fetch by revision *and* verify a recorded digest — a tag like `main` on a model repo is mutable.

This is the same discipline as [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md): the build either reproduces or it fails. Do not let a "convenience" path download the newest version of anything in production.

### Model provenance: treat weights as untrusted input

Practical rules for model files:

1. **Prefer safe formats.** Use `safetensors` over pickle-based checkpoints. `safetensors` is a pure data format — it cannot execute code on load. If you must load a pickle-based checkpoint, only do so from a source you would trust with shell access, and do it in a sandbox.
2. **Disable remote code.** Loading libraries often offer an option to execute custom code shipped in the repo. Keep it off unless you have read that code; it is a full remote-code-execution path by design.
3. **Pin the revision and the digest.** Record the repo id, the commit/revision, and the sha256 of each weight file in your config, and verify before load. A repo can be updated, deleted or transferred.
4. **Record what it is trained on, if known.** Provenance of the data matters for licence and for bias review (see [Bias and Fairness](07-bias-fairness.md)); "not stated" is a legitimate and important entry.
5. **Mirror what you depend on.** A model you cannot re-download is a model you cannot redeploy. Keep a copy in storage you control, with the digest recorded.

### Licences: check before you build the product on it

Two separate licence questions, and teams routinely answer only the first.

- **The code licence** — the library's licence governs your use of the library. Permissive (MIT, Apache-2.0, BSD) is usually fine; copyleft (GPL, AGPL) can impose obligations on your distributed service. Have someone who can read the licence confirm it for anything copyleft in a shipped path.
- **The weights licence** — a model's weights carry their own licence, separate from the code that runs them. Some permit commercial use, some restrict it, some add acceptable-use terms that flow down to your users. Read the model card's licence section, record it, and keep the terms with the model.
- **The data licence** — datasets used for fine-tuning or evaluation carry licences too, and "scraped from the web" is not a licence.

Record all three in one place: a small `MODEL_LICENCES.md` or a field in your model card. When a licence is unclear, the answer is "not stated — escalate", not "probably fine". This is engineering diligence, not legal advice; get a lawyer for a shipping decision.

### The supply-chain artefacts you actually maintain

- **An SBOM** (software bill of materials) generated at build: every package and version. It is what lets you answer "are we affected" in ten minutes instead of two days when a vulnerability is published.
- **A model inventory**: model id, revision, digest, licence, hosting location, and which features use it.
- **A prompt and eval manifest**: which prompt version and which eval dataset revision produced the numbers you shipped.

Three files, updated by CI where possible. They are also what a security questionnaire or an enterprise customer will ask for, so they pay for themselves twice.

### Prompts and configs copied from the internet

A prompt from a blog post or a shared snippet is untrusted text that will run with your system privileges — it can instruct the model to call tools, exfiltrate context, or ignore your policy. Treat copied prompts as code: read them, review them, put them in version control, and never interpolate an unreviewed remote prompt into a production path.

## Code

A verifier that checks a model file's digest and licence before anything loads it. Run this in the build, and fail the build rather than the request.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelPin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str
    revision: str            # a commit sha, never "main"
    filename: str
    sha256: str
    licence: str             # "apache-2.0", "llama-3-community", "not stated"
    commercial_ok: bool
    format: str              # "safetensors" preferred


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, pin: ModelPin) -> list[str]:
    problems: list[str] = []
    if path.name != pin.filename:
        problems.append(f"unexpected filename {path.name}")
    if sha256_of(path) != pin.sha256:
        problems.append("digest mismatch")
    if pin.format != "safetensors":
        problems.append("pickle-based format: load only in a sandbox")
    if pin.licence == "not stated":
        problems.append("licence not stated: escalate before shipping")
    if not pin.commercial_ok:
        problems.append("licence does not permit commercial use")
    return problems


def main() -> None:
    pin = ModelPin.model_validate_json(Path("models/pins/qwen-pin.json").read_text())
    failures = verify(Path("models") / pin.filename, pin)
    print(json.dumps({"model": pin.repo_id, "problems": failures}, indent=2))
    if failures:
        raise SystemExit("model provenance check failed")


if __name__ == "__main__":
    main()
```

On the Node side, the same idea at install time: enforce the lock and refuse to run with a dirty tree.

```ts
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

/** Fails the build if the lockfile is out of sync with package.json. */
export function assertLockfileClean(): void {
  const before = readFileSync("package-lock.json", "utf8");
  // --dry-run resolves without writing; a diff means package.json drifted.
  execFileSync("npm", ["install", "--package-lock-only", "--dry-run"], {
    stdio: "pipe",
  });
  const after = readFileSync("package-lock.json", "utf8");
  if (before !== after) {
    throw new Error("lockfile out of date: run npm install and commit the result");
  }
}
```

## Real use case

**A document-intelligence SaaS that fine-tuned an open-weight model.** The team had downloaded weights from a popular repo at `main`, loaded them with a library's default settings (which enabled the repo's custom code), and never recorded a licence. An enterprise customer's security review asked three questions: which model version is in production, what is its licence, and can you show the weights you deployed. The team could answer none of them, and discovered the licence forbade the commercial use they were making of it.

The remediation was a week of unglamorous work: pin the revision and record the sha256 of each weight file; switch to `safetensors`; disable remote code; generate an SBOM at build; add a `models/pins/` directory with one JSON per model including licence and commercial-use flag; make the build fail on a digest mismatch or an unstated licence. They replaced the model with one whose licence permitted commercial use, which cost a re-run of the eval suite but was cheaper than the alternative.

## Common mistakes

- **Mistake:** `pip install` and `npm install` in CI without hash verification — **Instead:** `pip install --require-hashes` and `npm ci`, with a lockfile check.
- **Mistake:** loading weights from a mutable branch or tag — **Instead:** pin a commit/revision and verify a recorded sha256 before load.
- **Mistake:** leaving remote-code execution on because the model "needed it" — **Instead:** keep it off; if a model genuinely requires custom code, vendor and review that code.
- **Mistake:** assuming the library licence covers the weights — **Instead:** record code, weights and data licences separately; treat "not stated" as a blocker to escalate.
- **Mistake:** copying a prompt from a blog straight into production — **Instead:** review and version it like code, with the same PR and test path.

## Checkpoint

1. Write a `models/pins/` entry for any model you have used, including the licence field. Mark anything you cannot confirm as "not stated".
2. Add `pip install --require-hashes -r requirements.txt` (or `npm ci`) to a CI job and confirm it fails when you add an unpinned dependency.
3. Generate an SBOM for one of your projects and answer, from it alone, whether a named package is present and at what version.

**Self-check:** The pin records repo id, revision, filename, sha256, licence, commercial-use flag and format, and the verifier fails the build on any mismatch or on an unstated licence — a pin without a digest verifies nothing.

## Go deeper

- [Sandboxing Tool Execution](05-sandboxing.md) — where to load a format you do not fully trust.
- [Bias and Fairness](07-bias-fairness.md) — why data provenance matters beyond licensing.
- [Governance](08-governance.md) — the model inventory and audit trail as compliance artefacts.
- [CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — the CI discipline these checks plug into.

---
[← Previous: Sandboxing Tool Execution](05-sandboxing.md) · [Stage index](README.md) · [Next: Bias and Fairness](07-bias-fairness.md)
