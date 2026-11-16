# 6. GitHub Actions CI/CD

> **Stage 19 · Cloud and DevOps** — one pipeline that tests, builds, pushes and deploys, with secrets that are not in git and caches that make it fast enough to keep.
>
> **You'll be able to:** write a workflow that gates merges on tests, builds and pushes an image tagged by commit, and deploys it with an environment approval and a rollback path.
>
> **Prerequisites:** [Git and GitHub](../01-programming-basics/05-git-github.md) · [Docker Done Properly](04-docker.md)

## Why this exists

Deploying by hand has three failure modes that all look different and are all the same: the steps live in someone's memory, so they drift; the credentials are pasted into a terminal, so nobody can audit them; and the "same" deploy differs between runs because a step was skipped.

CI/CD replaces the ritual with a file in the repo. That file is reviewable, versioned and identical on every run. When it fails, it fails the same way for everyone, which is the only property that makes a pipeline trustworthy.

The mental model: **a workflow is a YAML file triggered by an event, made of jobs, each job a fresh machine running steps in order.** Nothing survives between jobs except what you upload as an artefact.

## The mental model

```text
push / PR / tag
      |
      v
+-----------+   +-----------+   +-----------+   +------------+
|  test     | -> |  build    | -> |   push    | -> |  deploy    |
| ruff,pytest|   | docker    |   | registry  |   | kubectl/ECS|
| (no creds) |   | buildx    |   | sha tag   |   | needs      |
+-----------+   +-----------+   +-----------+   | approval   |
                                                 +------------+
secrets: OIDC role assumption preferred > repo secrets > nothing
caches : pip/npm deps, docker layers (buildx gha cache)
```

## Core concepts

### Jobs, steps, and the fresh-machine rule

Each job gets a clean runner. `actions/checkout` clones the repo; anything else you need, you install. This is why caching matters so much and why people are surprised when a file written in job A is missing in job B — pass it with `actions/upload-artifact` and `download-artifact`, or don't split the jobs.

`needs:` creates the dependency chain. A failed test job stops the chain, which is the gate.

### Triggers that match your branching

`on: pull_request` runs against the merge result, which is what you want for tests. `on: push: branches: [main]` runs on merge, which is where build-and-deploy belongs. `on: workflow_dispatch` gives you a manual button for hotfixes. Do not deploy from `pull_request` — that would let a fork's code reach production.

### Secrets: what belongs where

- **Repo or environment secrets** for values CI needs. They are masked in logs, but they are readable by any workflow on a branch you trust, so restrict who can push.
- **OIDC federation** is better for cloud access: the workflow proves its identity to AWS/GCP/Azure and receives short-lived credentials. No long-lived key exists to leak. This is the single biggest security improvement available to most pipelines.
- **Environment protection rules** for production: require a reviewer, restrict to `main`, and keep production secrets on that environment so a feature branch cannot read them.

Never echo a secret, never pass one on a command line where it lands in a log, and never build it into an image — an image layer is readable by anyone who can pull it.

### Caching that actually helps

Two caches matter. **Dependency cache** via `actions/setup-python` with `cache: pip`, keyed on the lockfile. **Docker layer cache** via `docker/build-push-action` with `cache-from: type=gha` and `cache-to: type=gha,mode=max`. Without the second, every build recompiles your dependency wheels and a two-minute pipeline becomes eight.

`mode=max` caches intermediate stages too, which is exactly what makes a multi-stage build fast.

### Tagging: one immutable tag, plus a moving alias

Tag the image with the commit SHA. That is the deployable identity and the rollback target. Optionally also tag `main` for convenience, but deploy by SHA so "which code is running" is never a guess. Avoid `latest` for deploys; it makes rollback ambiguous by construction.

### Rollback belongs in the pipeline, not in a runbook

A pipeline that can only go forward is half a pipeline. Keep two levers: `kubectl rollout undo` (fast, works if revisions are retained) and redeploying a previous SHA tag (explicit, works everywhere). Test the second one before you need it, because the first depends on revision history that a cleanup step may have pruned.

## Code

A complete workflow: test on PR, build and push on main, deploy with an environment approval. Note the `permissions` block (least privilege for `GITHUB_TOKEN`), the SHA tag, and the buildx cache.

```yaml
name: ci-cd

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  packages: write        # push to GHCR
  id-token: write        # required for OIDC cloud auth

env:
  IMAGE: ghcr.io/${{ github.repository }}/support-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements*.txt
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: pytest -q --maxfail=1
        env:
          APP_ENV: test

  build:
    needs: test
    if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.meta.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        run: echo "tag=${GITHUB_SHA::7}" >> "$GITHUB_OUTPUT"
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ${{ env.IMAGE }}:${{ steps.meta.outputs.tag }}
            ${{ env.IMAGE }}:main
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true          # signed build metadata, useful for audits

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production       # requires reviewer approval + environment secrets
    steps:
      - uses: actions/checkout@v4
      - name: Configure kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config
          chmod 600 ~/.kube/config
      - name: Deploy image tag
        run: |
          kubectl -n ai set image deploy/support-api \
            api=${{ env.IMAGE }}:${{ needs.build.outputs.tag }}
          kubectl -n ai rollout status deploy/support-api --timeout=180s
      - name: Rollback on failure
        if: failure()
        run: kubectl -n ai rollout undo deploy/support-api
```

The OIDC variant for AWS, which removes the long-lived key entirely. The role's trust policy must allow this repository and branch; the `role-to-assume` is an ARN, not a secret.

```yaml
      - name: Assume deploy role via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-deploy-support-api
          aws-region: ap-south-1
      - name: Update ECS service
        run: |
          aws ecs update-service --cluster prod --service support-api \
            --task-definition support-api:${{ needs.build.outputs.tag }} \
            --force-new-deployment
          aws ecs wait services-stable --cluster prod --services support-api
```

## Real use case

A four-person team deploys a RAG service to EKS. Their first pipeline ran everything in one job: test, build, push, deploy, all in sequence, all with a single long-lived AWS key in repo secrets. Two things went wrong within a month. A test failure still built and pushed an image, leaving untested tags in the registry that someone later deployed by mistake. And the AWS key, shared across three repos, could not be rotated without breaking all of them.

The rewrite split the jobs with `needs:`, so a red test stops the chain before anything is built. It moved cloud access to OIDC with a role scoped to one repository and one environment, which removed the key entirely. And it put the deploy behind a production environment with a required reviewer, which turned "who deployed at 6 p.m.?" from a mystery into an audit log entry.

The last change was the one that saved an incident: `rollout status` with a timeout and an automatic `rollout undo` on failure. A bad image that passed health checks but failed readiness rolled itself back in under three minutes while the on-call was still reading the alert.

## Common mistakes

**Mistake:** one giant job doing test, build and deploy — **Instead:** separate jobs with `needs:` so a failure stops the chain.

**Mistake:** long-lived cloud keys in repo secrets — **Instead:** OIDC federation with a role scoped to the repo and environment.

**Mistake:** deploying on `pull_request`, or deploying a `:latest` tag — **Instead:** test on PR, deploy on push to main tagged by commit SHA, so rollback names an exact artefact.

**Mistake:** no Docker layer cache — **Instead:** `cache-from`/`cache-to` with `type=gha,mode=max`.

**Mistake:** a deploy step with no failure handling — **Instead:** `rollout status` with a timeout and an automatic rollback step.

**Mistake:** echoing secrets for debugging — **Instead:** never print them; if you must inspect, print a length or a hash.

## Checkpoint

1. Split your pipeline into `test`, `build` and `deploy` jobs with `needs:`, and prove a failing test prevents an image from being pushed.
2. Replace a static cloud credential with OIDC role assumption, and confirm the workflow has no long-lived secret for that provider.
3. Add the deploy job with `rollout status --timeout` and an `if: failure()` rollback, then deploy a deliberately broken tag and time the recovery.

**Self-check:** exercise 1 is done when a deliberately failing `pytest` run shows `test` red, `build` skipped and `deploy` skipped, with no new tag in the registry.

## Go deeper

- [Kubernetes Essentials for AI Services](05-kubernetes.md) — what the deploy step actually applies.
- [CI/CD for Evals](../18-mlops-llmops/07-ci-cd-evals.md) — adding a quality gate alongside the test gate.
- [Terraform Basics](07-terraform.md) — managing the infra this pipeline deploys onto.
- [Ship: Dockerised Deploy with Rollback](09-ship-docker-deploy.md) — the project that puts all of this together.

---

[← Previous: Kubernetes Essentials for AI Services](05-kubernetes.md) · [Stage index](README.md) · [Next: Terraform Basics](07-terraform.md)
