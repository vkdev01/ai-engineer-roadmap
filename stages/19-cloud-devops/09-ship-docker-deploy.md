# 9. Ship: Dockerised Deploy with Rollback

> **Stage 19 · Cloud and DevOps** — the project that turns your Stage 17 service into a deployable artefact with a pipeline, a healthcheck, env-based config and a tested way back.
>
> **You'll be able to:** containerise an AI service, deploy it from CI, configure it entirely through environment variables, and recover from a bad release in minutes.
>
> **Prerequisites:** [GitHub Actions CI/CD](06-github-actions.md) · [Docker Done Properly](04-docker.md)

## Why this exists

Every earlier file in this stage adds one capability. This project is where they have to coexist: an image that builds the same everywhere, a pipeline that refuses to deploy untested code, a healthcheck that reflects real readiness, config that comes from the environment rather than the image, and a rollback that you have actually performed rather than imagined.

The last point is the one that separates a portfolio project from a tutorial. Anyone can write a workflow. Proving you can break production and recover — with timings — is the thing an interviewer remembers.

The mental model: **deployment is a reversible operation on an immutable artefact. If it is not reversible, it is not a deployment, it is a gamble.**

## The mental model

```text
git push main
   |
   v
[ test ] -> [ build image :<sha> ] -> [ push registry ] -> [ deploy :<sha> ]
                                                                  |
                                                          healthcheck passes?
                                                          /                  \
                                                        yes                  no
                                                         |                    |
                                                  traffic on new      rollback to previous :<sha>
                                                  revision (recorded)  (one command, timed)
```

The two facts you must be able to state at the end: the exact image SHA currently serving traffic, and how many seconds a rollback takes.

## Core concepts

### What you build

Four artefacts in one repo. Keep it boring; the discipline is the deliverable.

1. **`Dockerfile`** — multi-stage, non-root, `HEALTHCHECK`, exec-form `CMD`, no secrets baked in. Reuse the one from [Docker Done Properly](04-docker.md).
2. **`.github/workflows/deploy.yml`** — `test` → `build` → `push` → `deploy`, with the SHA as the tag, `needs:` between jobs, and an automatic rollback step on failure.
3. **Config contract** — a single `settings.py` that reads every value from the environment, plus a documented `.env.example` listing every key. No defaults for secrets; fail loudly at startup if a required one is missing.
4. **`deploy/`** — the Kubernetes manifests (or an ECS task definition) with the image tag as the only thing CI changes, plus a `ROLLBACK.md` with the exact commands.

## Code

The config contract. This is small and it is the difference between "works on my machine" and a deployable service: one place that validates the environment at boot, so a missing variable fails immediately and visibly instead of at the first request.

```python
from __future__ import annotations

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="dev", pattern="^(dev|staging|prod)$")
    release_sha: str = Field(default="dev")          # injected by CI, logged on every trace

    openai_api_key: str                              # no default: fail at boot if absent
    model_base_url: HttpUrl
    database_url: str
    redis_url: str

    request_timeout_s: float = 30.0
    log_level: str = "info"


# Importing this raises a clear validation error listing missing keys, which is
# exactly what you want: the container fails its startup probe instead of 500-ing.
settings = Settings()
```

The deploy workflow, complete. Note that the only thing that changes between releases is `RELEASE_SHA`, and that the rollback step names an explicit previous tag rather than relying on cluster history alone.

```yaml
name: deploy

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      image_tag:
        description: "Existing image tag to redeploy (rollback path)"
        required: true

permissions:
  contents: read
  packages: write

env:
  IMAGE: ghcr.io/${{ github.repository }}/support-api
  NAMESPACE: ai

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12", cache: pip }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check . && pytest -q --maxfail=1

  build:
    needs: test
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.tag.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: tag
        run: echo "tag=${GITHUB_SHA::7}" >> "$GITHUB_OUTPUT"
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ env.IMAGE }}:${{ steps.tag.outputs.tag }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: [test, build]
    if: always() && (needs.test.result == 'success')
    runs-on: ubuntu-latest
    environment: production
    env:
      RELEASE_SHA: ${{ github.event.inputs.image_tag || needs.build.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
      - name: Write kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config
          chmod 600 ~/.kube/config
      - name: Record current image (for rollback)
        id: prev
        run: |
          prev=$(kubectl -n $NAMESPACE get deploy/support-api \
            -o jsonpath='{.spec.template.spec.containers[0].image}')
          echo "image=$prev" >> "$GITHUB_OUTPUT"
          echo "Previous image: $prev"
      - name: Deploy
        run: |
          kubectl -n $NAMESPACE set image deploy/support-api api=$IMAGE:$RELEASE_SHA
          kubectl -n $NAMESPACE set env deploy/support-api RELEASE_SHA=$RELEASE_SHA
          kubectl -n $NAMESPACE rollout status deploy/support-api --timeout=180s
      - name: Verify health through the service
        run: |
          kubectl -n $NAMESPACE run smoke --rm -i --restart=Never --image=curlimages/curl -- \
            curl -fsS "http://support-api/readyz"
      - name: Rollback on failure
        if: failure()
        run: |
          echo "Deploy failed; rolling back to ${{ steps.prev.outputs.image }}"
          kubectl -n $NAMESPACE set image deploy/support-api api=${{ steps.prev.outputs.image }}
          kubectl -n $NAMESPACE rollout status deploy/support-api --timeout=180s
```

The manifests CI mutates, kept minimal so the image tag really is the only moving part.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: support-api
  namespace: ai
spec:
  replicas: 2
  revisionHistoryLimit: 10
  selector:
    matchLabels: { app: support-api }
  template:
    metadata:
      labels: { app: support-api }
    spec:
      containers:
        - name: api
          image: ghcr.io/example/support-api:0000000   # CI replaces this tag only
          ports: [{ containerPort: 8000 }]
          env:
            - { name: APP_ENV, value: "prod" }
            - { name: RELEASE_SHA, value: "unknown" }
          envFrom:
            - secretRef: { name: support-api-secrets }
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits: { memory: "1Gi" }
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            periodSeconds: 15
```

### How to prove it works

Three tests, and the third is the one you show people.

1. **Clean-machine deploy.** Clone the repo into a fresh directory, push a commit, and let CI do everything. No local Docker daemon, no manual `kubectl` command. Record the total pipeline duration.
2. **Healthcheck that fails honestly.** Stop the vector store or set a bad `MODEL_BASE_URL` in the deployment. Confirm `/readyz` returns 503, the pod is removed from endpoints but not restarted, and the rollout halts instead of claiming success.
3. **Rehearsed rollback.** Ship a tag that passes liveness but fails readiness (a wrong model URL is perfect). Watch the automatic rollback run, then do it manually via `workflow_dispatch` with the previous tag. Time both.

Write the timings down. A table like this, with your own numbers, is the evidence:

| Step | Command | Observed |
|---|---|---|
| Bad deploy detected | `rollout status` timeout | (your seconds) |
| Automatic rollback complete | `rollout status` after undo | (your seconds) |
| Manual redeploy of old tag | `workflow_dispatch` + deploy job | (your seconds) |

## Real use case

A two-person team runs a document Q&A product on a single Kubernetes namespace. Their first deploys were manual: build locally, push, `kubectl set image`, watch logs. One Friday a prompt-plus-code change shipped that passed local tests, and `readyz` failed in production because a new required environment variable was missing. The pod crash-looped, the old pods were already terminated, and the service was down for 25 minutes while someone found the previous tag in a terminal history.

The rebuild changed three things. Config became a validated `Settings` class, so a missing variable fails at boot with a message naming the key — and CI now runs a container smoke test with the production key list before deploying. The deploy job records the previous image and rolls back automatically if `rollout status` or the smoke check fails. And `maxUnavailable: 0` on the rollout means old pods keep serving until new ones are ready, so a bad image never takes capacity away.

The next bad release behaved differently: the rollout stalled with old pods still serving, `rollout status` timed out at 180 seconds, the rollback step ran, and the only user-visible symptom was one slow request. The incident review was two paragraphs instead of a post-mortem.

## Common mistakes

**Mistake:** config defaults for secrets so the app "starts anyway" — **Instead:** required fields with no default, validated at boot, so failures happen at deploy time not request time.

**Mistake:** deploying `:latest` — **Instead:** the commit SHA as the tag, recorded before and after every deploy.

**Mistake:** a rollback plan that exists only in someone's memory — **Instead:** an automatic rollback step plus a `workflow_dispatch` redeploy path, both rehearsed and timed.

**Mistake:** a healthcheck that returns 200 without checking dependencies — **Instead:** `/healthz` for liveness and `/readyz` for readiness, and a smoke check through the Service after rollout.

**Mistake:** `maxUnavailable: 1` with two replicas — **Instead:** `maxUnavailable: 0` with `maxSurge: 1`, so capacity never drops during a bad rollout.

**Mistake:** mutating manifests by hand in the cluster — **Instead:** the only change CI makes is the image tag, so the running state matches the repo.

## Checkpoint

1. Containerise your Stage 17 service, deploy it from CI with a SHA tag, and confirm a clean clone plus one push is enough to reach a running, healthy deployment.
2. Break readiness deliberately (bad `MODEL_BASE_URL`), confirm the rollout halts with old pods serving, and confirm the automatic rollback restores service. Record the timings.
3. Add the `workflow_dispatch` rollback path, redeploy a previous tag manually, and write `ROLLBACK.md` with the two commands and the numbers you measured.

**Self-check:** exercise 1 is done when the pipeline's `deploy` job completes, `kubectl get deploy support-api -o jsonpath='{..image}'` reports the SHA tag from that commit, and a request to the Service returns 200 from `/readyz`.

## Go deeper

- [Stage 20 — AI Security](../20-ai-security/README.md) — hardening what you just deployed.
- [GitHub Actions CI/CD](06-github-actions.md) — the pipeline mechanics in detail.
- [Terraform Basics](07-terraform.md) — declaring the cluster and namespace reproducibly.
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — designing for failure rather than only recovering from it.

---

[<- Previous: GPU Cloud Practicalities](08-gpu-cloud.md) · [Stage index](README.md) · [Next: Stage 20 — AI Security](../20-ai-security/README.md)
