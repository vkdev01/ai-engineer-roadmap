# Stage 19 — Cloud and DevOps

Stage 17 and 18 gave you a service and the means to watch it. This stage gives you somewhere to run it. It covers the small set of cloud primitives an AI app actually needs, then the container, orchestration, CI and infrastructure-as-code layers that make deployment repeatable instead of a sequence of console clicks someone half-remembers.

**Time:** 3-4 weeks of evenings · **You need first:** Stage 18 — MLOps and LLMOps · **You leave with:** a Dockerised AI service deployed by CI, with a healthcheck, environment-based config and a tested rollback

## What you will be able to do

- Pick the right AWS primitive for documents, GPU work and webhook glue, and explain the IAM and cost traps of each.
- Write a multi-stage Dockerfile for a Python AI service that is small, non-root, healthchecked and reproducible.
- Deploy that image to Kubernetes with resource limits, secrets from the environment, and an HPA that makes sense for your workload.
- Build a GitHub Actions pipeline that tests, builds, pushes and deploys, with secrets handled properly and layers cached.
- Manage infrastructure with Terraform using a remote backend, so `plan` is reviewable and state is not on a laptop.
- Rent GPU capacity without being surprised by quota, region availability or lost spot instances.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [AWS for AI Apps: S3, EC2, Lambda](01-aws-s3-ec2-lambda.md) | Storage, GPU compute, glue code, IAM basics and cost traps |
| 2 | [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) | Choosing between managed containers and managed ML |
| 3 | [GCP and Azure AI Equivalents](03-gcp-azure-ai.md) | Vertex AI and Azure OpenAI mapped onto what you know |
| 4 | [Docker Done Properly](04-docker.md) | Multi-stage builds, slim images, GPU bases, healthchecks |
| 5 | [Kubernetes Essentials for AI Services](05-kubernetes.md) | Deployments, services, config, secrets, HPA, GPU scheduling |
| 6 | [GitHub Actions CI/CD](06-github-actions.md) | Test, build, push, deploy, with secrets and caching |
| 7 | [Terraform Basics](07-terraform.md) | State, plan and apply, modules, remote backend |
| 8 | [GPU Cloud Practicalities](08-gpu-cloud.md) | Spot instances, quota, regions, persistent storage |

## The shipping project

[09-ship-docker-deploy.md](09-ship-docker-deploy.md) — you containerise your Stage 17 AI service, deploy it through a CI pipeline with a real healthcheck, move all config to environment variables, and rehearse a rollback to the previous image tag. You prove it works by breaking the new version on purpose and recovering in under five minutes.

## How to know you are done

- A fresh clone of your repo can be built and deployed by CI with no local Docker daemon involved.
- Your image runs as a non-root user, is a fraction of the naive build's size, and answers a healthcheck that actually checks the model dependency.
- Killing a pod or rolling out a bad tag does not take the service down permanently — you can name the exact rollback command.
- `terraform plan` on a clean checkout against your remote backend shows no drift.
- You can state, without checking, what one GPU-hour costs you today and where you wrote down the date you last verified it.

---

[← Back to roadmap](../../README.md) · [Next stage: AI Security](../20-ai-security/README.md)
