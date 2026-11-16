# 8. Ship: A Reusable Training Loop

> **Stage 05 · DL Frameworks** — Turn the loop you now understand into a package: config-driven, resumable, and logging metrics you can plot.
>
> **You'll be able to:** ship a `trainer/` package that any project in this roadmap can reuse, and prove a run survives being killed.
>
> **Prerequisites:** [3. Anatomy of a Training Loop](03-training-loops.md) · [4. GPU Memory and Checkpoints](04-gpu-checkpoints.md)

## Why this exists

You have written the same loop four times now, each time slightly differently, and each time with a different place where the checkpoint lived. That is fine while learning and expensive afterwards: every experiment you run later — fine-tuning, distillation, a toy model for a demo — starts by copy-pasting a loop and re-fixing the same four bugs.

This project fixes the shape once. One `Trainer` class, one `RunConfig` object, a registry so the model and data are chosen by name, checkpoints that resume, and metrics as JSONL so any plotting tool can read them. From here on, a new experiment is a YAML file, not a new script.

The mental model: *a run is fully described by its config plus its checkpoint; anything not in those two is a bug.*

## The mental model

```text
        config.yaml  ──► RunConfig (pydantic, validated)
                             │
        registry.get("mlp") ─┴─► model + optimizer + scheduler
                             │
   run_dir/run-001/
     ├── config.json          what was run (frozen copy)
     ├── metrics.jsonl        one JSON object per step
     ├── ckpt-last.pt         model+opt+sched+step (atomic write)
     └── ckpt-best.pt         best val metric so far
```

## Core concepts

### The config is data, validated at the boundary

A pydantic model gives you three things a dict does not: defaults, type coercion from YAML strings, and an error at startup instead of at step 4000. If `lr` is a string `"3e-4"`, you want to know before the GPU is warm.

### A registry makes the model a choice, not a code edit

`{"mlp": build_mlp, "linear": build_linear}` maps a name in the config to a factory that returns a model. Adding an experiment means adding a function and a YAML file; the trainer never changes. This is the single highest-value pattern in the whole package.

### Resume means four things, not one

Weights, optimizer state, scheduler state, and the step counter. Plus the RNG state if you care about exact replay. `Trainer.fit` should look at `resume_from` and continue the loop from `start_step`, not restart it. The test for correctness is that the metrics file has no gap and no repeated step number.

### Metrics as JSONL, not print statements

One JSON object per line, appended, flushed. `{"step": 120, "split": "train", "loss": 0.431, "lr": 3e-4}`. JSONL survives a crash mid-run (the last line may be truncated; skip it when parsing), appends in O(1), and loads into pandas in one call. Printing is for humans watching; JSONL is for the analysis afterwards.

### Separate the concerns into files

Four small files beat one large one: `config.py` (schema), `registry.py` (factories), `trainer.py` (the loop), `run.py` (CLI entry). When something is wrong you know which file to open.

## Code

Layout first, so the imports make sense:

```text
trainer/
  config.py      RunConfig, load_config
  registry.py    MODEL_REGISTRY, register, build
  trainer.py     Trainer.fit / evaluate / _save / _load
  run.py         python -m trainer.run --config config.yaml
configs/
  mlp.yaml
tests/
  test_resume.py
```

The config and registry. Note the `Literal` types — they make invalid values a startup error, not a runtime surprise.

```python
# trainer/config.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    run_name: str = "run-001"
    out_dir: Path = Path("runs")
    model: str = "mlp"                                  # key in the registry
    model_kwargs: dict = Field(default_factory=lambda: {"in_dim": 20, "n_classes": 3})
    epochs: int = Field(5, ge=1)
    batch_size: int = Field(32, ge=1)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = 0.01
    scheduler: Literal["cosine", "none"] = "cosine"
    precision: Literal["fp32", "bf16"] = "bf16"
    grad_clip: float = 1.0
    log_every: int = 10
    resume_from: Path | None = None
    seed: int = 0


def load_config(path: Path) -> RunConfig:
    # validate at startup: a bad config must fail before the GPU is warm
    return RunConfig(**yaml.safe_load(path.read_text()))
```

```python
# trainer/registry.py
from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str) -> Callable:
    def wrap(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = fn
        return fn
    return wrap


@register("mlp")
def build_mlp(in_dim: int, n_classes: int, hidden: int = 64) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, n_classes),
    )


@register("linear")
def build_linear(in_dim: int, n_classes: int) -> nn.Module:
    return nn.Linear(in_dim, n_classes)          # (B, in_dim) -> (B, n_classes)


def build(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
```

The trainer itself. Read `fit` as the loop from [3. Anatomy of a Training Loop](03-training-loops.md) with bookkeeping around it.

```python
# trainer/trainer.py
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import RunConfig
from .registry import build


class Trainer:
    def __init__(self, cfg: RunConfig, train_dl: DataLoader, val_dl: DataLoader) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: nn.Module = build(cfg.model, **cfg.model_kwargs).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs) if cfg.scheduler == "cosine" else None)
        self.loss_fn = nn.CrossEntropyLoss()
        self.run_dir = cfg.out_dir / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.global_step = 0
        self.best_val = float("-inf")      # best validation ACCURACY (we maximise it)

    def _log(self, split: str, **kv: float | int) -> None:
        row = {"step": self.global_step, "split": split, **kv}
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")      # one object per line, append-only

    def _save(self, path: Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "cfg": self.cfg.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)                     # atomic: no half-written checkpoints

    def _load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])  # without this, loss jumps on resume
        if self.sched and ck["scheduler"]:
            self.sched.load_state_dict(ck["scheduler"])
        self.global_step = ck["global_step"]
        self.best_val = ck["best_val"]

    def evaluate(self) -> float:
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for xb, yb in self.val_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred: Tensor = self.model(xb).argmax(dim=-1)
                correct += int((pred == yb).sum())
                total += yb.numel()
        return correct / max(total, 1)

    def fit(self) -> None:
        start_epoch = 0
        if self.cfg.resume_from:
            self._load(self.cfg.resume_from)
            start_epoch = self.global_step // max(len(self.train_dl), 1)
        use_amp = self.cfg.precision == "bf16" and self.device == "cuda"
        for epoch in range(start_epoch, self.cfg.epochs):
            self.model.train()
            for xb, yb in self.train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss: Tensor = self.loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                self.global_step += 1
                if self.global_step % self.cfg.log_every == 0:
                    self._log("train", loss=round(float(loss.detach()), 5),
                              lr=self.opt.param_groups[0]["lr"])
            if self.sched:
                self.sched.step()
            val_acc = self.evaluate()
            self._log("val", acc=round(val_acc, 4), epoch=epoch)
            self._save(self.run_dir / "ckpt-last.pt")
            # state the direction once, here: higher accuracy is better
            if val_acc > self.best_val:
                self.best_val = val_acc
                self._save(self.run_dir / "ckpt-best.pt")
```

Entry point, so a run is one command:

```python
# trainer/run.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)

    ds = TensorDataset(torch.randn(1024, 20), torch.randint(0, 3, (1024,)))
    train_dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(ds, batch_size=64)
    Trainer(cfg, train_dl, val_dl).fit()


if __name__ == "__main__":
    main()
```

## Real use case

A three-person ML team runs six experiments a week across two projects: a ticket classifier and a small language model. Each project had its own training script, and each script had quietly diverged — one saved the optimizer state, one did not; one logged to CSV, one to a print statement that nobody could plot.

After a machine restart killed a nine-hour run, the post-mortem found that the surviving script *could* resume but the checkpoint it wrote lacked the scheduler state, so the learning rate restarted at its initial value. Nobody noticed for two epochs because the loss curve looked plausible.

The rebuild is the package above, used by both projects. A new experiment is a YAML file and a registry entry. Resume is one flag, and the metrics file makes the gap obvious. The measurable outcome is not speed — it is that the team can now answer "what exactly did that run do" from `config.json` and `metrics.jsonl` alone, without asking whoever launched it.

## How to prove it works

Four checks, each under two minutes. Run them in this order and keep the output; this is the evidence a reviewer will ask for.

| Check | Command | What proves it |
|-------|---------|----------------|
| Config validation | edit `lr: abc` in the YAML and run | Fails at startup with a field error, not mid-run |
| Registry | set `model: nonexistent` | `KeyError` naming the available keys |
| Resume | run 10 steps, `kill -9`, restart with `resume_from` | `metrics.jsonl` has no duplicate or missing step numbers, and loss continues near where it stopped |
| Best checkpoint | compare `ckpt-last.pt` and `ckpt-best.pt` val acc in metrics | Best file corresponds to the highest `val.acc` row |

A test for the resume claim, which is the one that matters:

```python
# tests/test_resume.py
import json
from pathlib import Path


def test_metrics_have_no_gaps(metrics_path: Path) -> None:
    steps = [json.loads(line)["step"] for line in metrics_path.read_text().splitlines() if line]
    assert steps == sorted(steps)
    assert len(steps) == len(set(steps)), "duplicate step after resume"
    assert steps[-1] > steps[0], "run produced no progress"
```

## Common mistakes

- **Mistake:** putting the run name in code instead of config. — **Instead:** config only; a run must be reproducible from its YAML.
- **Mistake:** resuming weights but not optimizer state. — **Instead:** save and load all four pieces, and test with a `kill -9`.
- **Mistake:** logging with `print` and losing the numbers. — **Instead:** JSONL, one line per logged event, appended and flushed.
- **Mistake:** tracking "best" without deciding the direction. — **Instead:** store the metric and its sign explicitly; mixing up loss and accuracy silently saves the worst checkpoint.
- **Mistake:** overwriting the run directory on every start. — **Instead:** timestamp the run dir or refuse to start if `ckpt-last.pt` exists without `resume_from`.

## Checkpoint

1. Add an `EarlyStopping` field to `RunConfig` and stop `fit` when validation accuracy has not improved for `patience` epochs.
2. Add a `"steps"` scheduler option that steps per batch, and prove via `metrics.jsonl` that the learning rate decays faster than the cosine-per-epoch option.
3. Make the trainer refuse to start when `run_dir` already contains checkpoints and `resume_from` is `None`. Then add a `--force` flag.

**Self-check:** Early stopping needs the best value, a patience counter and a stop flag on the trainer; reset the counter whenever `val_acc` beats `best_val`, increment otherwise, and break out of the epoch loop when it exceeds patience.

## Go deeper

- [6. Hugging Face Transformers and Datasets](06-hugging-face.md) — the same ideas, in `Trainer` form.
- [1. MLflow and W&B](../18-mlops-llmops/01-mlflow-wandb.md) — where the JSONL metrics go next.
- [5. CI/CD Evals](../18-mlops-llmops/07-ci-cd-evals.md) — turning the resume test into a gate.
- [1. SFT](../13-fine-tuning/01-sft.md) — point this trainer at a language model.

---
[← Previous: CUDA, Devices and Mixed Precision](07-cuda.md) · [Stage index](README.md) · [Next: Stage 06 — LLM Fundamentals](../06-llm-fundamentals/README.md)
