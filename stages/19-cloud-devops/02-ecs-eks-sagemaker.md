# 2. Containers at Scale: ECS, EKS, SageMaker

> **Stage 19 · Cloud and DevOps** — deciding where your container runs once "one machine" stops being enough.
>
> **You'll be able to:** choose between ECS, EKS and SageMaker for a given AI workload, and explain the trade-off in operational cost rather than marketing terms.
>
> **Prerequisites:** [AWS for AI Apps: S3, EC2, Lambda](01-aws-s3-ec2-lambda.md) · [Docker Done Properly](04-docker.md)

## Why this exists

You have an image that works on your laptop. Now it needs to run in three places with different needs: the API that answers users (bursty, needs to scale, must never go down), a batch job that embeds 200,000 documents (long, resource-heavy, tolerant of interruption), and a fine-tune (GPU, hours, produces an artefact).

Putting all three on one big EC2 instance works until it does not: deploys take the whole app down, one batch job starves the API of memory, and nobody can tell which process is using the GPU. Container orchestration exists to schedule those workloads onto machines with resource guarantees, health checks and restarts.

The mental model: **ECS is a scheduler you configure with AWS-specific JSON; EKS is Kubernetes with a steeper curve and more portability; SageMaker is managed ML plumbing that takes over training and endpoint hosting entirely.**

## The mental model

```text
                    who schedules?        how much do you operate?
AWS-native, simple  [ ECS Fargate ]  ----  very little: task defs + services
Industry standard   [ EKS ]          ----  a lot: cluster, add-ons, upgrades, RBAC
Managed ML          [ SageMaker ]    ----  least ML work, most AWS lock-in

typical AI app:
  API + workers        -> ECS Fargate (or EKS if you already run k8s)
  fine-tune / batch    -> SageMaker training job  OR  a one-off GPU container
  self-hosted model    -> EKS with GPU node group  OR  SageMaker endpoint
```

## Core concepts

### ECS: the smallest step from one machine

An ECS **task definition** is a JSON document describing containers, CPU and memory, environment, ports and logging. A **service** keeps N copies running behind a load balancer, restarts unhealthy ones, and rolls out new task definitions. With Fargate you do not manage instances at all; you pay per task-hour for the CPU and memory you reserved.

This is the right default for a small team. You write two JSON documents, get rolling deploys, health-check restarts and autoscaling on CPU or request count, and you never think about the cluster's operating system. Its ceiling: less flexible scheduling, no rich ecosystem, and GPU support is narrower than EKS.

The one ECS habit that saves pain: **reserve CPU and memory honestly.** A task that asks for 512 CPU units but uses 2 vCPU gets throttled and behaves in ways that look like model latency.

### EKS: Kubernetes when you need it

EKS runs a managed Kubernetes control plane; you still own node groups, add-ons, upgrades and access control. You get the whole ecosystem — HPA, pod disruption budgets, operators, and portability across clouds — at the price of a platform skill you must actually acquire. If your company already runs Kubernetes, use it; if you are a team of three, do not adopt Kubernetes for one API.

GPU scheduling is where EKS earns its keep: you create a node group of GPU instances, install the NVIDIA device plugin, and pods request `nvidia.com/gpu: 1`. Kubernetes then guarantees that no two pods share a GPU, which is exactly the isolation you want for a model server.

### SageMaker: managed training and endpoints

SageMaker covers three things worth knowing separately:

- **Training jobs** — you point it at an image, an S3 input and an instance type; it provisions, runs, writes output to S3, and shuts down. This is the cleanest way to run a fine-tune without leaving a GPU instance running by accident.
- **Endpoints** — managed hosting with autoscaling and rolling updates. Convenient, and it prices per instance-hour, so an idle endpoint costs the same as a busy one.
- **Processing and pipelines** — batch transform jobs for offline scoring over large S3 datasets.

The trade-off is lock-in: your training script becomes a SageMaker-shaped script, and leaving means re-platforming. For a fine-tune every few months, that is often a fair trade. For the core of your product, think twice.

### Choosing, honestly

| Signal | Choose |
|---|---|
| Small team, one API, no existing k8s | ECS Fargate |
| Already run Kubernetes, or need GPU pods with scheduling control | EKS |
| Occasional fine-tune, want no GPU babysitting | SageMaker training job |
| Batch scoring of millions of documents offline | SageMaker batch transform or a one-off ECS task |
| Must keep everything in your own VPC with full control | EKS on your own node groups |

Note the pattern: the deciding factor is almost always **who operates what**, not which service is technically capable. All three can run your container.

### The thing all three share: statelessness

Whatever you pick, the container must hold no state. Uploads go to S3, cache goes to Redis, sessions go to a database. The reason is not purity — it is that every one of these systems will kill and replace your container at a moment it chooses, and if a request was mid-write to local disk, that data is gone.

## Code

An ECS task definition for the API, trimmed to the fields that matter. `healthCheck` here is the container-level check ECS uses before it routes traffic; the `secrets` block pulls from Secrets Manager rather than baking values into the definition.

```json
{
  "family": "support-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/supportApiTaskRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "123456789012.dkr.ecr.ap-south-1.amazonaws.com/support-api:9c2e1f0",
      "essential": true,
      "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
      "environment": [
        { "name": "APP_ENV", "value": "prod" },
        { "name": "LOG_LEVEL", "value": "info" }
      ],
      "secrets": [
        { "name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:ap-south-1:123456789012:secret:openai-key" },
        { "name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:ap-south-1:123456789012:secret:db-url" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -fsS http://localhost:8000/healthz || exit 1"],
        "interval": 15,
        "timeout": 3,
        "retries": 3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/support-api",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

The SageMaker training job that fine-tunes a LoRA adapter. Two lines carry the cost control: the instance type and `StoppingCondition`, which caps a runaway job.

```python
from __future__ import annotations

import sagemaker
from sagemaker.huggingface import HuggingFace

# Verify current instance type availability and price in the console before running.
estimator = HuggingFace(
    entry_point="train.py",
    source_dir="src/train",
    instance_type="ml.g5.2xlarge",   # GPU family names change; check availability in your region
    instance_count=1,
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    volume_size=100,                 # GB of attached storage; too small fails mid-run
    max_run=4 * 60 * 60,             # hard cap: 4 hours, then it is killed
    hyperparameters={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",  # check the model card for current id
        "epochs": 2,
        "lora_r": 16,
        "learning_rate": 2e-4,
    },
    use_spot_instances=True,
    max_wait=6 * 60 * 60,            # spot: wait up to 6h for capacity
    checkpoint_s3_uri="s3://my-ai-artefacts/checkpoints/lora-run-42",
)

estimator.fit({"train": "s3://my-ai-datasets/instructions/v3/"})
print(estimator.model_data)  # s3://.../output/model.tar.gz — the artefact to register
```

## Real use case

An ed-tech startup runs a lesson-plan generator: a FastAPI API calling hosted models, plus a nightly job that re-embeds the whole content library when editors change it. Their first setup was both on one ECS service. The nightly job used 8 GB of memory, so the task definition reserved 8 GB, which meant the API's tasks were also 8 GB, and the bill reflected a capacity they needed for two hours a day.

The fix was two task definitions with different shapes. The API: 1 vCPU, 2 GB, three tasks behind a load balancer, autoscaling on request count, deployed on every merge. The embedder: a scheduled ECS task with 4 vCPU and 16 GB, run once nightly, exiting when done. Cost dropped because each workload paid for its own shape, and deploys stopped being scary because a bad API deploy could no longer take the embedder with it.

They later moved the fine-tune of their classifier to a SageMaker training job purely to stop leaving a GPU instance running: with a training job, the instance exists only for the duration of the job and the console shows nothing left behind. The lesson generalised into a rule they wrote down: **anything with a start and an end should be a job, not a service.**

## Common mistakes

**Mistake:** reserving resources for the largest workload and running everything in it — **Instead:** one task definition per workload shape, sized to that workload.

**Mistake:** adopting Kubernetes for a single API with a three-person team — **Instead:** ECS Fargate unless you already have k8s skills or need GPU pod scheduling.

**Mistake:** baking secrets into the task definition or image — **Instead:** `secrets` from Secrets Manager or SSM, injected as env vars at runtime.

**Mistake:** a health check that only returns 200 without touching dependencies — **Instead:** check the things that break: database reachability, model endpoint, vector store.

**Mistake:** running a fine-tune on a persistent GPU instance — **Instead:** a training job with `max_run` and S3 checkpoints, so a crash or a forgotten stop costs minutes, not days.

**Mistake:** trusting an instance-type name or price from a tutorial — **Instead:** verify availability per region and current pricing in the console.

## Checkpoint

1. Write an ECS task definition for your Stage 17 API with environment and secrets separated, a healthcheck, and log configuration to a named log group. Validate it with `aws ecs register-task-definition --cli-input-json`.
2. Move one batch workload from a long-running service into a scheduled task that exits on completion, and confirm it stops billing when idle.
3. Compare ECS Fargate and EKS for your workload in a short written note: list the three operational tasks you would own on EKS that you do not own on Fargate, and what you would gain.

**Self-check:** exercise 1 is done when the definition registers without error and its `secrets` entries reference ARNs while `environment` contains no credential values.

## Go deeper

- [Kubernetes Essentials for AI Services](05-kubernetes.md) — the EKS side in detail.
- [Docker Done Properly](04-docker.md) — the image these task definitions run.
- [GPU Cloud Practicalities](08-gpu-cloud.md) — quota and spot behaviour for the training side.
- [Ship: Dockerised Deploy with Rollback](09-ship-docker-deploy.md) — the deploy path for the API service.

---

[← Previous: AWS for AI Apps: S3, EC2, Lambda](01-aws-s3-ec2-lambda.md) · [Stage index](README.md) · [Next: GCP and Azure AI Equivalents](03-gcp-azure-ai.md)
