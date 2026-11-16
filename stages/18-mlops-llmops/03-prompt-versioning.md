# 3. Prompt Versioning and Registries

> **Stage 18 · MLOps and LLMOps** — turning a prompt from a string in a Python file into an artefact you can diff, deploy and revert.
>
> **You'll be able to:** keep prompts in versioned files with metadata, resolve a named version at runtime, and roll back a bad prompt release in one command.
>
> **Prerequisites:** [Prompt Templates](../08-prompt-and-context/03-templates.md) · [Experiment Tracking with MLflow and W&B](01-mlflow-wandb.md)

## Why this exists

A prompt is the most-edited piece of an LLM application and the least controlled. It starts as a triple-quoted string inside `chain.py`. Then someone edits it directly in production to fix an urgent bug, at 11 p.m., without a commit. Three weeks later nobody knows why a paragraph is worded the way it is, the eval scores in the tracker do not correspond to any text you can find, and the only rollback path is hoping git history contains the version that was live.

Every serious incident review of an LLM app contains the same line: "we could not determine which prompt was deployed". Prompt versioning exists to delete that line.

The mental model: **a prompt is an artefact with a name, a version, a content hash and metadata — and the application asks for it by name at runtime, never by inlining it.**

## The mental model

```text
prompts/
|-- answer/
|   |-- v1.md      (deprecated)
|   |-- v2.md      (deprecated)
|   `-- v3.md      <- current prod
`-- summarize/
    `-- v1.md

registry entry:  answer
  label prod      -> v3   hash c40f   owner @priya   eval faithfulness 0.78
  label staging   -> v4   hash 9de2   eval faithfulness 0.79  (unmerged)
  rollback        -> point prod at v2, one command, no code deploy
```

The application logs `prompt=answer.v3 hash=c40f` on every trace. Now any answer can be traced to text.

## Core concepts

### Store prompts as files, not strings

A prompt file is markdown or a `.txt` with `{{variable}}` placeholders. Files give you `git diff` for free, code review for free, and blame for free. The alternative — a database row edited through an admin UI — gives you faster edits and worse history unless you build the history yourself.

The file's front matter is where the metadata lives: version, owner, model it targets, and the eval run that justified it. Keep it machine-readable so CI can read it.

### The content hash is the identity

A version number is a human label and humans get it wrong (two people both ship `v4`). A SHA-256 of the rendered template body is the identity: if two deploys have the same hash, they are the same prompt, guaranteed. Log the hash on every trace and every eval run. Then "which prompt produced this?" is answerable by lookup, not by guesswork.

### Labels, not branches

A registry maps a mutable **label** (`prod`, `staging`) to an immutable **version**. Deploying is moving a label. Rolling back is moving it back. That is why registries beat a `PROMPT_VERSION` environment variable: reverting an env var means a redeploy, reverting a label is a single API call that takes effect on the next request.

Both MLflow (prompt registry in recent versions) and Langfuse (prompt management with labels) support this shape, as do vendor tools like PromptLayer. If you use none of them, a git tag plus a config map gets you most of the way.

### Rendered prompts differ from templates

The template is versioned; the rendered prompt includes the variables. Two runs of the same version with different retrieved context are not the same input, which is why you log both: the version hash for comparison across time, and a sample rendered prompt for debugging. This is also why prompt changes must be evaluated with realistic variable values, not `{{question}} = "hello"`.

## Code

First, a small resolver that loads a prompt by name and label, verifies the hash, and caches it. The cache matters: reading and hashing a file on every request is cheap but pointless, and a registry call per request is a latency and availability risk. Refresh on a timer, and fall back to the last good copy if the registry is unreachable.

```python
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
REFRESH_SECONDS = 60


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    hash: str
    template: str

    def render(self, **variables: str) -> str:
        out = self.template
        for key, value in variables.items():
            out = out.replace("{{" + key + "}}", value)
        if "{{" in out:
            missing = out[out.index("{{") : out.index("}}") + 2]
            raise ValueError(f"unrendered placeholder {missing} in {self.name}.{self.version}")
        return out


class PromptStore:
    """Resolves name+version from disk. Swap the loader for a registry API later."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Prompt] = {}
        self._loaded_at = 0.0

    def get(self, name: str, version: str) -> Prompt:
        if time.monotonic() - self._loaded_at > REFRESH_SECONDS:
            self._cache.clear()
            self._loaded_at = time.monotonic()
        key = (name, version)
        if key not in self._cache:
            body = (PROMPT_DIR / name / f"{version}.md").read_text()
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
            self._cache[key] = Prompt(name=name, version=version, hash=digest, template=body)
        return self._cache[key]


store = PromptStore()
p = store.get("answer", "v3")
prompt_text = p.render(question="how do I cancel an order?", context="...")
print(p.hash, len(prompt_text))
```

Second, the registry contract as a tiny HTTP service so deploys and rollbacks are one call. A YAML file is the source of truth; the service is a thin reader, which keeps git as the audit trail.

```yaml
# prompts/registry.yaml — committed to git; the service only reads it.
labels:
  answer:
    prod: v3
    staging: v4
  summarize:
    prod: v1
versions:
  answer:
    v3:
      hash: c40f
      owner: priya
      eval_run: a1f3
      faithfulness: 0.78
    v4:
      hash: 9de2
      owner: priya
      eval_run: b902
      faithfulness: 0.79
```

```python
from __future__ import annotations

import subprocess

import yaml
from fastapi import FastAPI, HTTPException

app = FastAPI()


def load_registry() -> dict:
    return yaml.safe_load(open("prompts/registry.yaml"))


@app.get("/prompts/{name}/{label}")
def resolve(name: str, label: str) -> dict[str, str]:
    reg = load_registry()
    version = reg["labels"].get(name, {}).get(label)
    if version is None:
        raise HTTPException(404, f"no label {label} for {name}")
    return {"name": name, "label": label, "version": version,
            "hash": reg["versions"][name][version]["hash"]}


@app.post("/prompts/{name}/{label}/{version}")
def move_label(name: str, label: str, version: str) -> dict[str, str]:
    # Rollback is just a label move to an older version.
    reg = load_registry()
    if version not in reg["versions"].get(name, {}):
        raise HTTPException(404, "unknown version")
    reg["labels"][name][label] = version
    open("prompts/registry.yaml", "w").write(yaml.safe_dump(reg, sort_keys=True))
    subprocess.run(["git", "commit", "-am", f"prompt: move {name} {label} -> {version}"], check=True)
    return {"status": "moved", "name": name, "label": label, "version": version}
```

## Real use case

A fintech app has a `extract_transaction` prompt that turns a bank SMS into structured JSON. During a festive sale week, an engineer tightens the instructions to handle a new merchant format and deploys directly. Extraction accuracy for three other banks collapses, and the app starts mislabelling spends.

With prompts as artefacts, the sequence is: the dashboard breaks error rate down by `prompt_version` and shows the jump landing exactly on `extract_transaction.v7`; the on-call moves the `prod` label back to `v6`; the API resolves the new label within the cache refresh window, no redeploy, no container restart. Total time under five minutes, and the incident review has a real diff of v6 against v7 to argue about.

The failure mode before was a 40-minute rollback: revert the commit, rebuild the image, wait for the deploy, then discover a second prompt had been edited in the same commit.

## Common mistakes

**Mistake:** versioning prompts in a database edited through a UI with no git history — **Instead:** keep the text in files under version control and let the UI read from them, so review and blame still work.

**Mistake:** a single `prod` prompt with no staging label — **Instead:** at least `staging` and `prod`, so you can compare live behaviour against a candidate on real traffic.

**Mistake:** reading the registry on every request — **Instead:** cache with a short TTL and a last-known-good fallback, or the registry becomes a hard dependency in your latency path.

**Mistake:** changing a prompt without an eval run attached — **Instead:** make `eval_run` a required metadata field; a version with no eval is a version nobody has measured.

**Mistake:** inlining prompts again "just for this small feature" — **Instead:** one resolver, one store; the exception is how the mess starts.

## Checkpoint

1. Move your main prompt into `prompts/<name>/v1.md`, load it through a resolver, and log `prompt_version` and `prompt_hash` on every trace.
2. Add a `registry.yaml` with `staging` and `prod` labels and a `/resolve` endpoint. Deploy a `v2` to `staging`, point one internal user at it via a header, and confirm both versions appear in traces.
3. Implement label move as the rollback action, run it against a deliberately broken `v2`, and measure how long until traffic is back on `v1` — including cache TTL.

**Self-check:** exercise 1 is done when two traces from the same endpoint report the same `prompt_hash`, and the hash matches `sha256sum` of the prompt file's first 12 characters.

## Go deeper

- [LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) — where the prompt version tag is consumed for filtering.
- [CI/CD for Evals](07-ci-cd-evals.md) — blocking a prompt version from reaching `prod` until its evals pass.
- [Prompt Evaluations](../08-prompt-and-context/07-prompt-evals.md) — what to measure before promoting a version.
- [Ship: A Trace Dashboard](08-ship-trace-dashboard.md) — grouping cost and scores by prompt version.

---

[<- Previous: LLM Observability with Langfuse and LangSmith](02-langfuse-langsmith.md) · [Stage index](README.md) · [Next: DVC for Data and Models](04-dvc.md)
