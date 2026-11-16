# 9. Ship: A Clean Dataset with a Data Card

> **Stage 03 · Data Engineering** — the stage project: raw mess in, documented dataset out
>
> **You'll be able to:** build, validate and document a dataset end to end, and prove to another engineer that it is reproducible.
>
> **Prerequisites:** [Dataset Preparation](08-dataset-prep.md) · [Data Pipelines](06-data-pipelines.md)

## Why this exists

You have learnt the pieces separately: clean, explore, validate, dedup, split. In a
real job they arrive as one job — "here is a raw export, give me something we can
fine-tune and evaluate on, and tell me what is in it". Nobody will accept a notebook.

This project is deliberately small in scope and complete in shape. You take a messy
export, run it through one script that validates and transforms, and emit three splits
plus a `data_card.md` and a `stats.json`. The artefact is not the dataset — it is the
proof that the dataset is what you say it is.

Mental model: **a shipped dataset is data plus the argument for its trustworthiness,
produced by one command.**

## The mental model

```text
data/raw/tickets_export.csv  (messy, 1 file)
        |
        |  python -m pipeline.build --config config.yaml
        v
  extract -> validate (pydantic) -> normalise -> dedup (shingle sketch)
        -> group split -> contamination check -> write
        |
        v
out/
  train.jsonl  val.jsonl  test.jsonl      one record per line, validated
  stats.json                              counts, lengths, contamination
  data_card.md                            source, splits, gaps, licence
  run_manifest.json                       input hash, git sha, row counts
```

## Core concepts

### One command, no manual steps

Everything a human would otherwise do by hand — renaming a column, dropping a row —
belongs in the script. The acceptance test is that a colleague with the repo and the
raw file can reproduce your output exactly, without asking you anything. That is why
the input path, the thresholds and the split ratios live in `config.yaml` and not in
your head.

### Validate before transform, again after

Run the pydantic model on each raw record (catch drift early) and on each output record
(because your transform can introduce nulls). Both passes count failures and both
write to `stats.json`. A high raw-failure rate stops the run; a small one quarantines
the rows.

### Reproducibility has three parts

Fixed seed for any sampling, deterministic sort before writing, and a manifest that
records the input file hash. With those, a second run produces byte-identical output —
which you can check with `sha256sum`, and which is the strongest claim you can make
about a dataset.

### The data card is not optional

Source and collection method, what each field means, split sizes, what was removed and
why, known gaps, licence and permitted use, and the date it was built. Six short
sections. Without them the dataset is unusable by anyone but you — and unusable by you
in three months.

## Code

The project layout, then the core of the build script. Run
`python -m pipeline.build --config config.yaml`; you should see one line per stage and
three files in `out/`.

```text
03-clean-dataset/
├── config.yaml
├── pipeline/
│   ├── __init__.py
│   ├── schemas.py       # pydantic models
│   ├── build.py         # the one command
│   └── dedup.py         # shingle sketch + clustering
├── data/raw/tickets_export.csv
└── out/                 # generated: splits, stats, data card, manifest
```

`schemas.py` is the contract — the same model validates input and output.

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Ticket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=10)
    answer: str = Field(min_length=1)
    product: Literal["billing", "shipping", "account", "other"]
    group: str = Field(min_length=1, description="customer id; never split across splits")

    @model_validator(mode="after")
    def answer_differs_from_question(self) -> "Ticket":
        if self.answer.strip().lower() == self.text.strip().lower():
            raise ValueError("answer duplicates the question")
        return self


class TrainingRecord(BaseModel):
    """The shape written to JSONL — the chat format fine-tuning APIs expect."""
    model_config = ConfigDict(extra="forbid")
    id: str
    messages: list[dict[str, str]]
    product: str
```

`build.py` — the whole pipeline, short enough to read in one sitting.

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from .dedup import cluster_docs, group_split
from .schemas import Ticket, TrainingRecord


def load(cfg: dict) -> tuple[pd.DataFrame, str]:
    raw = Path(cfg["input"])
    df = pd.read_csv(raw, dtype=str).drop_duplicates()
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()[:16]
    return df, digest


def to_tickets(df: pd.DataFrame) -> tuple[list[Ticket], int]:
    good: list[Ticket] = []
    failed = 0
    for row in df.to_dict(orient="records"):
        try:
            good.append(Ticket.model_validate(row))
        except ValidationError:
            failed += 1
    return good, failed


def write_split(path: Path, tickets: list[Ticket]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for t in sorted(tickets, key=lambda x: x.id):     # deterministic order
            rec = TrainingRecord(
                id=t.id,
                messages=[{"role": "user", "content": t.text},
                          {"role": "assistant", "content": t.answer}],
                product=t.product,
            )
            f.write(rec.model_dump_json() + "\n")


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=True)

    df, digest = load(cfg)
    tickets, failed = to_tickets(df)
    print(f"loaded {len(df)} rows, {len(tickets)} valid, {failed} rejected")
    if failed > len(df) * cfg["max_failure_rate"]:
        raise SystemExit("too many invalid rows: upstream schema drift")

    clusters = cluster_docs(tickets, threshold=cfg["dedup_threshold"])
    splits = group_split(clusters, ratios=tuple(cfg["split_ratios"]))
    for name, rows in splits.items():
        write_split(out / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} records")

    stats = {"input_hash": digest, "rows_in": len(df), "rejected": failed,
             "clusters": len(clusters),
             "sizes": {k: len(v) for k, v in splits.items()}}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote stats.json — fill data_card.md from it")


if __name__ == "__main__":
    main()
```

### How to prove it works

Run these four checks in order. Each one is a sentence you can put in a pull request.

1. **Reproducibility.** `sha256sum out/*.jsonl`, delete `out/`, run the build again,
   `sha256sum` again. The hashes must match. If they do not, you have a seed, a sort
   order or a timestamp leaking into the output.
2. **Validation is real.** Copy `train.jsonl` to `bad.jsonl`, change one record's
   `messages` to a single message, and run your validator over it. It must report one
   failure on one line. A validator that never fails is not a validator.
3. **No leakage.** Write a test that recomputes cluster ids and asserts no cluster
   appears in two splits. This is the assertion that protects your future eval scores.
4. **The card matches the data.** Every number in `data_card.md` is copied from
   `stats.json`. Diff them; if they disagree, the card is stale.

## Real use case

An internal tools team needed a support-reply model for four product areas. Their
first attempt used a raw export split with `train_test_split`, and the eval reported a
score high enough that the lead asked for a second opinion. The project above was built
to answer that question, and it answered it quickly: 18% of the raw rows failed
validation (blank answers from a legacy form), and a third of the eval set had a
near-duplicate in training.

The shipped dataset is smaller than the original — roughly two thirds of the rows — and
the eval score fell. It was also the first version whose number anyone trusted, because
`data_card.md` stated what had been removed and `stats.json` recorded the contamination
rate. When the model later shipped and behaved as the eval predicted, the card is what
the team pointed at.

## Common mistakes

- **Mistake:** shipping the dataset without the card — **Instead:** generate `data_card.md` from `stats.json` so the two can never disagree.
- **Mistake:** thresholds and paths in the script — **Instead:** put them in `config.yaml`; the run is then reproducible and reviewable.
- **Mistake:** no `run_manifest.json` — **Instead:** record the input hash and row counts so "which version of the data is this?" has an answer.
- **Mistake:** random sampling without a seed — **Instead:** seed it and sort output deterministically; otherwise check 1 above fails.
- **Mistake:** treating a validation failure rate of zero as success — **Instead:** print the rejection count; zero rejections on messy data usually means your validator is too weak.
- **Mistake:** a licence field left as "TBD" — **Instead:** state the licence and permitted use, or state explicitly that it is unresolved and who must decide.

## Checkpoint

1. Build the project on a real export of your own (any CSV with a text column will do) and get all four proof checks passing.
2. Add a `--sample N` flag that draws a stratified sample by product for a fast local run, with a fixed seed, and confirm the output is still reproducible.
3. Extend `stats.json` with per-split median and p95 token counts, then update `data_card.md` to report them and note any split that looks unbalanced.

**Self-check:** reproducibility means two runs on the same input produce byte-identical files, which requires a fixed seed, a deterministic sort, and no timestamps written into the output.

## Go deeper

- [Stage 04 — Deep Learning](../04-deep-learning/README.md) — what you train on these splits.
- [Instruction Tuning](../13-fine-tuning/04-instruction-tuning.md) — consuming the JSONL you just produced.
- [Golden Datasets](../15-evals/03-golden-datasets.md) — maintaining the test split as a living artefact.
- [DVC](../18-mlops-llmops/04-dvc.md) — versioning the dataset alongside the code that built it.

---
[← Previous: Dataset Preparation](08-dataset-prep.md) · [Stage index](README.md) · [Next: Stage 04 — Deep Learning](../04-deep-learning/README.md)
