# 8. GPU Cloud Practicalities

> **Stage 19 · Cloud and DevOps** — the parts of renting a GPU that no tutorial warns you about: quota, capacity, interruption and where your weights actually live.
>
> **You'll be able to:** plan a GPU workload around quota limits and regional availability, use spot capacity safely with checkpointing, and keep model weights on persistent storage instead of inside an image.
>
> **Prerequisites:** [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) · [GPU Memory](../16-inference/10-gpu-memory.md)

## Why this exists

Renting a GPU is the one part of cloud work where the failure is not a config error but a wall: you ask for a machine and it does not exist. Either your account has no quota for that family in that region, or the region has no capacity today, or the spot instance you were relying on disappeared mid-training and your four hours of progress went with it.

None of these are visible from the console until you hit them. The fix is planning: know your quota before you need it, know which regions have the family you want, design for interruption, and keep state off the machine.

The mental model: **a GPU instance is a scarce, expensive, interruptible machine. Assume it can vanish at any minute, and design so that costs you minutes rather than a day.**

## The mental model

```text
request GPU instance
        |
        |-- quota check (per family, per region) -- fails -> increase quota, days of lead time
        |-- capacity check (is it in stock today?) - fails -> another region or another family
        `-- got it
              |
              |-- spot?  -> 2-minute interruption notice, sometimes less
              |              -> must checkpoint to durable storage
              `-- on-demand -> runs until you stop it (and bills until you do)

storage tiers:
  instance-local NVMe  fast, vanishes with the instance
  attached volume      survives stop/start, same AZ only
  object storage       survives everything, slower to read, cheap to keep
```

## Core concepts

### Quota is the first blocker, and it is per family per region

Cloud accounts start with low or zero GPU quota. Requesting it is a support process measured in hours to days, and it applies per instance family and per region. This is the single most common reason a planned fine-tune does not happen on schedule: someone discovers on the day that they cannot launch the instance type they designed around.

Practical routine: check the quota page for the exact family and region, request an increase well ahead, and keep a second region in mind with the quota already granted. Treat the request as a week-one task, not a launch-day task.

### Instance names change, prices change, availability changes

GPU families are renamed and refreshed constantly, and the same family may be offered in one region and not another. Any specific name or hourly figure written in a tutorial — including this one — is a snapshot that will go stale. Before you commit to a plan, check the provider's current instance-type listing for your region and its current pricing page. Write the date you checked into your notes, so a future reader knows how old the number is.

What is stable is the shape of the decision: **memory per device, devices per node, interconnect between nodes, and whether the family is offered in your region.** Those four properties drive everything else, and they are worth reasoning about explicitly rather than by memorising SKUs.

### Spot: cheap and interruptible, and the notice is short

Spot capacity is spare capacity at a discount, reclaimed when demand rises. You get a short interruption notice — on AWS historically two minutes, and you should design for that order of magnitude and verify the current behaviour — then the instance is taken.

Three habits make spot usable:

1. **Checkpoint to durable storage on a timer**, not only at the end. Every N steps, write the optimiser state and step count to object storage. Resuming from a checkpoint loses N steps, not the run.
2. **Handle the notice.** Most clouds deliver an interruption signal or a metadata endpoint you can poll; catch it, save, and exit cleanly. A process that ignores it is killed mid-write and can leave a corrupt checkpoint.
3. **Never use spot for a serving endpoint** unless you have replicas across capacity pools and accept occasional churn. Spot is for jobs with a start and an end.

### Persistent storage: where the weights live

Model weights are the biggest object in the system and the one people handle worst. Three rules:

- **Do not bake weights into the container image.** A 14 GB layer makes every pull slow, every node's disk fill, and every registry bill large.
- **Do not download them on every start** from a public host over the internet. It is slow, it costs egress, and it is a single point of failure at cold start.
- **Do** keep them in object storage in the same region, and mount or fetch once per node into a persistent volume or a fast local cache. Then a restart is seconds, not minutes.

Also watch the disk: a training run that writes checkpoints to instance-local disk will fill it, and a full disk manifests as a confusing failure deep in a library. Set a checkpoint retention policy — keep the last few plus the best, delete the rest.

### Right-sizing: memory is the constraint, not FLOPs

A GPU has a fixed memory budget and everything competes for it: weights, activations, KV cache for serving, optimiser state for training. Sizing by "which card is fastest" is the wrong instinct; size by "does my workload fit with headroom".

The practical sequence: compute the weight footprint from parameter count and precision, add the activation or KV-cache requirement, add a margin, then pick the smallest device that fits. For serving, that number determines concurrency; see [GPU Memory](../16-inference/10-gpu-memory.md). Getting this wrong shows up as an out-of-memory error partway through a run, which is why the checkpoint timer matters even on on-demand.

## Code

A checkpointing training loop that survives interruption: it writes on a step interval and on the spot notice, and resumes from the latest checkpoint. The storage path is object storage, so a new instance in a new AZ can pick it up.

```python
from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path

import torch

CKPT_DIR = Path(os.environ.get("CKPT_DIR", "/mnt/ckpt"))  # mounted persistent volume
INTERRUPTED = False


def _handle_sigterm(signum: int, frame: object) -> None:
    # Spot reclaim and normal shutdown both arrive here; save and exit cleanly.
    global INTERRUPTED
    INTERRUPTED = True


signal.signal(signal.SIGTERM, _handle_sigterm)


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 0

    def save(self) -> None:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CKPT_DIR / f"step-{self.step}.pt.tmp"
        final = CKPT_DIR / f"step-{self.step}.pt"
        torch.save(
            {"model": self.model.state_dict(), "optim": self.optimizer.state_dict(), "step": self.step},
            tmp,
        )
        tmp.replace(final)   # atomic: never leave a half-written checkpoint
        # Keep only the three most recent to avoid filling the volume.
        for old in sorted(CKPT_DIR.glob("step-*.pt"))[:-3]:
            old.unlink()

    def resume(self) -> None:
        ckpts = sorted(CKPT_DIR.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
        if not ckpts:
            return
        state = torch.load(ckpts[-1], map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optim"])
        self.step = state["step"]
        print(f"resumed at step {self.step}")


def train(trainer: Trainer, batches: list[torch.Tensor], save_every: int = 200) -> None:
    trainer.resume()
    for i, batch in enumerate(batches[trainer.step :], start=trainer.step):
        loss = trainer.model(batch).sum()
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.step = i + 1
        if trainer.step % save_every == 0 or INTERRUPTED:
            trainer.save()
            print(f"checkpoint at step {trainer.step} (interrupted={INTERRUPTED})")
        if INTERRUPTED:
            return   # let the orchestrator replace us; the next run resumes here
```

A preflight script that fails fast on the two things that waste the most time: quota and capacity. Run it the day before a run, not on the day.

```python
from __future__ import annotations

import boto3
from botocore.exceptions import ClientError


def check_quota(instance_family: str, region: str) -> None:
    """Print the GPU vCPU quota for a family; raise if it looks like zero."""
    ec2 = boto3.client("ec2", region_name=region)
    try:
        quotas = ec2.describe_service_quotas(ServiceCode="ec2")["Quotas"]
    except ClientError as exc:
        print(f"quota lookup failed in {region}: {exc}")
        return
    for q in quotas:
        if instance_family.lower() in q["QuotaName"].lower():
            print(f"{region}: {q['QuotaName']} = {q['Value']}")


def can_launch(instance_type: str, region: str) -> bool:
    """Dry-run launch: checks capacity and permission without starting anything."""
    ec2 = boto3.client("ec2", region_name=region)
    try:
        ec2.run_instances(
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            DryRun=True,          # raises DryRunOperation when you WOULD be allowed
            ImageId="ami-00000000000000000",
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "DryRunOperation":
            return True
        print(f"{region} {instance_type}: {code}")
        return False
    return False


if __name__ == "__main__":
    for r in ["ap-south-1", "us-east-1"]:
        print(f"--- {r}")
        check_quota("g5", r)
        can_launch("g5.xlarge", r)
```

## Real use case

A team plans a weekend LoRA fine-tune on a specific GPU family in their home region. On Friday evening the launch fails: quota for that family is zero in that region. The increase request is filed, but the window is gone.

The retry a week later is planned differently. Quota is requested in two regions a fortnight ahead. The training loop checkpoints to object storage every 200 steps and handles `SIGTERM`. The run starts on spot capacity in the second region and is interrupted twice in six hours; each interruption costs about ten minutes of progress instead of the whole run. Weights are fetched once from a bucket in the same region into a volume, so a replacement instance is training within two minutes instead of re-downloading for fifteen.

The total was not cheaper than the on-demand plan — spot savings partly offset by restarts — but it actually completed inside the available window, which the on-demand plan did not. The lesson they wrote into their runbook: **quota and interruption are scheduling problems, not cost problems, and you solve them before the run starts.**

## Common mistakes

**Mistake:** assuming quota exists because you can see the instance type in the console — **Instead:** check the quota for that family and region and request an increase early.

**Mistake:** quoting an hourly price from a blog or a tutorial — **Instead:** check the current pricing page for your region and record the date checked.

**Mistake:** checkpointing only at the end of training — **Instead:** periodic atomic checkpoints to durable storage, plus a `SIGTERM` handler.

**Mistake:** baking weights into the image or downloading them from a public host at every start — **Instead:** object storage in-region plus a node-level cache or persistent volume.

**Mistake:** writing checkpoints to instance-local disk with no retention policy — **Instead:** durable storage and keep only the last few.

**Mistake:** running a serving endpoint on spot with one replica — **Instead:** on-demand for serving, or multiple replicas across capacity pools with health-checked failover.

## Checkpoint

1. Run the preflight script for the family and region you plan to use, and record the quota values plus the date. File any increase request you need today.
2. Add atomic checkpointing and a `SIGTERM` handler to a small training loop, then kill the process mid-run with `SIGTERM` and confirm a resume continues from the last checkpoint rather than step zero.
3. Write a one-page GPU plan: instance family, memory justification from parameter count, storage tier for weights, spot or on-demand with a reason, and the interruption cost you accept.

**Self-check:** exercise 2 is done when a resumed run prints a starting step greater than zero and the loss curve continues from roughly where it stopped rather than restarting from the initial value.

## Go deeper

- [GPU Memory](../16-inference/10-gpu-memory.md) — the sizing maths behind the family choice.
- [LoRA, QLoRA and PEFT](../13-fine-tuning/02-lora-qlora-peft.md) — what you are actually training on that GPU.
- [Kubernetes Essentials for AI Services](05-kubernetes.md) — GPU node groups, taints and device plugins.
- [Terraform Basics](07-terraform.md) — declaring quota-adjacent infra and node groups reproducibly.

---

[← Previous: Terraform Basics](07-terraform.md) · [Stage index](README.md) · [Next: Ship: Dockerised Deploy with Rollback](09-ship-docker-deploy.md)
