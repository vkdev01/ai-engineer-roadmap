# 4. DVC for Data and Models

> **Stage 18 · MLOps and LLMOps** — git for the things git refuses to hold: datasets, embeddings and model weights.
>
> **You'll be able to:** put a dataset or checkpoint under DVC, commit the pointer, and reproduce an exact training or eval run on another machine from the commit alone.
>
> **Prerequisites:** [Git and GitHub](../01-programming-basics/05-git-github.md) · [Dataset Preparation](../03-data-engineering/08-dataset-prep.md)

## Why this exists

Git stores text diffs. A 2 GB JSONL of training data or a 14 GB checkpoint is not text, and putting it in git makes clones slow forever — the history keeps every version, so a repo with ten dataset revisions is unusable. So people do the obvious wrong thing: they keep data in a shared drive and tell each other the filename in Slack. `golden_set_final_v3.csv` becomes `golden_set_final_v3_actually.csv`, and the eval number in your tracker no longer points at any file you can find.

DVC fixes exactly one problem: **it separates the content from the pointer.** The data lives in remote storage; git holds a small `.dvc` file containing a hash. Checkout the old commit, run `dvc pull`, and you get the bytes that were used. That is the whole idea.

The mental model: **a `.dvc` file is a symlink by hash.** Git versions the symlink; DVC resolves it.

## The mental model

```text
git repo                          remote storage (S3 / GCS / local NAS)
|-- data/golden.dvc   --------+
|     md5: 4b77c1...          |   dvc pull  ->  data/golden.jsonl (240 rows)
|     size: 412000            |
|-- models/lora.dvc   --------+
|     md5: 9de2aa...
`-- src/…  (normal code, normal git)
```

```text
git log                       dvc.lock
a1f3  prompt v3 + golden 4b77  <- the run in your tracker
b902  prompt v4 + golden 4b77
c10d  prompt v4 + golden 5a01  <- golden set changed; scores not comparable
```

## Core concepts

### Init, track, push, pull

`dvc init` creates a `.dvc` directory. `dvc add data/golden.jsonl` writes `data/golden.dvc` and adds the data file to `.gitignore`. `dvc push` copies the bytes to your configured remote. `dvc pull` fetches whatever the current `.dvc` files point at.

The habit to build: **commit the `.dvc` file, never the data.** Your CI clones the repo, runs `dvc pull`, and has the exact dataset. If the remote is S3, the CI runner needs credentials scoped to that bucket prefix and nothing else.

### The remote is just object storage

A DVC remote is a bucket with a prefix. For a solo project it can be a directory on a NAS or another disk; for a team it is S3 or GCS. Versioning and lifecycle rules belong on the bucket: keep all dataset versions, expire intermediate checkpoints after 30 days.

Be honest about the trade-off: DVC is not a database and does not deduplicate at row level. If you change one row of a 500 MB file, you store 500 MB again. That is fine for golden sets of thousands of rows; for a constantly growing corpus, keep the corpus in a warehouse or document store and DVC only the frozen eval slice.

### Pipelines make reproduction real

`dvc.yaml` describes stages: command, dependencies, outputs. `dvc repro` reruns only stages whose inputs changed, and `dvc.lock` records the hashes actually used. This is what lets you say "this eval run used this dataset and this prompt file" and have it be true — the lock file is committed alongside the code.

For LLM work the useful stages are usually `prepare_dataset`, `build_index`, `run_evals`. Training is often skipped entirely.

### What to track and what not to

Track: frozen eval sets, the documents you indexed, embedding indexes, LoRA adapters, any checkpoint you intend to serve. Do not track: the raw scrape you can re-download, caches, anything with secrets inside, and per-run logs (those go to your tracker). A useful test: if losing it would make a past score unreproducible, track it.

## Code

A `dvc.yaml` for an eval pipeline. Three stages, and only the changed one reruns. Note `prompts/answer/v3.md` as a dependency — that is how a prompt edit invalidates the eval stage without you remembering to force it.

```yaml
# dvc.yaml
stages:
  prepare_dataset:
    cmd: python -m src.prepare --raw data/raw_faq.jsonl --out data/golden.jsonl
    deps:
      - data/raw_faq.jsonl
      - src/prepare.py
    outs:
      - data/golden.jsonl

  build_index:
    cmd: python -m src.index --dataset data/golden.jsonl --out indexes/faq
    deps:
      - data/golden.jsonl
      - src/index.py
    outs:
      - indexes/faq

  run_evals:
    cmd: python -m src.evals --dataset data/golden.jsonl --index indexes/faq --out reports/evals.json
    deps:
      - data/golden.jsonl
      - indexes/faq
      - prompts/answer/v3.md
      - src/evals.py
    metrics:
      - reports/evals.json:
          cache: false
```

Commands to run it, and what each one is for. `dvc repro` after a prompt edit reruns only `run_evals`; `dvc push` publishes the new artefacts so a teammate can `dvc pull` and see the same result.

```bash
dvc init
dvc remote add -d storage s3://my-org-dvc/ai-roadmap   # check bucket policy first
dvc add data/golden.jsonl
git add data/golden.jsonl.dvc .gitignore dvc.yaml
git commit -m "data: freeze golden set v1"
dvc push
dvc repro run_evals      # only reruns what changed
git add dvc.lock reports/evals.json && git commit -m "evals: rerun on golden v1"
```

Second snippet: the CI step that makes reproduction actually happen. It fails loudly when a dataset is missing rather than silently evaluating on nothing — a real failure mode when `dvc pull` is skipped.

```yaml
# .github/workflows/evals.yml (excerpt)
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: dvc pull --no-run-cache
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.DVC_AWS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.DVC_AWS_SECRET }}
      - run: python -m src.evals --dataset data/golden.jsonl --index indexes/faq --out reports/evals.json
      - run: python -m src.gate --report reports/evals.json --min-faithfulness 0.75
```

## Real use case

A three-person team builds a policy-document assistant. Their golden set grows from 80 to 240 questions over two months as they discover edge cases. Every time it grows, the mean faithfulness score drops — not because the system got worse, but because the new questions are harder. Without dataset versioning, this reads as a regression and triggers a pointless week of prompt tuning.

With DVC, each eval run records the dataset hash. The dashboard plots score against `dataset_hash` and the two series are visibly separate: 0.81 on the 80-question set, 0.74 on the 240-question set. The team stops chasing a phantom regression and instead tracks improvement within a stable dataset hash, bumping the version deliberately when they want to move the baseline. The failure mode before was worse than wasted time — they had tuned the prompt to the easy 80 questions and made it brittle on the hard ones.

## Common mistakes

**Mistake:** committing the data file and the `.dvc` file together — **Instead:** `.gitignore` the data path (DVC adds it for you) and commit only the pointer; check `git status` before every commit.

**Mistake:** no remote configured, so `dvc push` does nothing and the pointer is dangling — **Instead:** set a remote in `dvc init` day one and test `dvc pull` on a fresh clone.

**Mistake:** editing a tracked file in place and expecting DVC to notice — **Instead:** rerun `dvc add` after changing the file, or you keep publishing the old hash with new bytes locally.

**Mistake:** tracking a dataset that is regenerated on every run — **Instead:** track the frozen eval slice and the raw source separately, and let the derived file be an output.

**Mistake:** giving CI broad storage credentials to make `dvc pull` work — **Instead:** scope the key to the one bucket prefix the repo needs.

## Checkpoint

1. `dvc init`, track your golden set, push to any remote you have (even a local directory), then clone the repo to a temp directory and confirm `dvc pull` restores the file with an identical hash.
2. Write a `dvc.yaml` with `prepare_dataset` and `run_evals` stages. Touch the prompt file and confirm `dvc repro` reruns only `run_evals`.
3. Break the remote path deliberately and confirm CI fails with a readable error rather than evaluating on an empty dataset.

**Self-check:** exercise 1 is done when `dvc status` reports "Data and pipelines are up to date" in the fresh clone and the restored file's md5 matches the `md5` line in the `.dvc` file.

## Go deeper

- [Experiment Tracking with MLflow and W&B](01-mlflow-wandb.md) — log the dataset hash from `dvc.lock` as a run param.
- [CI/CD for Evals](07-ci-cd-evals.md) — the full workflow this file's excerpt belongs to.
- [Golden Datasets](../15-evals/03-golden-datasets.md) — how to build the frozen slice you are versioning.
- [Dataset Preparation](../03-data-engineering/08-dataset-prep.md) — the pipeline upstream of `prepare_dataset`.

---

[<- Previous: Prompt Versioning and Registries](03-prompt-versioning.md) · [Stage index](README.md) · [Next: Tracing Design](05-tracing.md)
