# 4. Docker Done Properly

> **Stage 19 · Cloud and DevOps** — packaging an AI service so it builds the same everywhere, starts fast, and does not ship a compiler to production.
>
> **You'll be able to:** write a multi-stage Dockerfile for a Python AI service, cut image size dramatically, run as non-root with a real healthcheck, and pick the right base image for GPU work.
>
> **Prerequisites:** [Linux and the CLI](../01-programming-basics/06-linux-cli.md) · [AWS for AI Apps: S3, EC2, Lambda](01-aws-s3-ec2-lambda.md)

## Why this exists

The naive Python Dockerfile is four lines and produces a 1.4 GB image that takes minutes to pull, rebuilds from scratch on every code change, runs as root, and contains a C compiler your production container will never use. It works on your laptop and behaves differently in CI because the base image tag moved.

Every one of those problems is a design decision, and each has a standard fix. The result of doing them properly: an image measured in hundreds of megabytes, rebuilds in seconds when only code changed, a non-root user, a healthcheck the orchestrator can trust, and a deterministic dependency set.

The mental model: **a build stage installs and compiles; a runtime stage receives only the finished artefacts.** Everything you leave out of the final stage is size, attack surface and startup time you do not pay.

## The mental model

```text
stage 1 "builder"                      stage 2 "runtime"
  python:3.12-slim (or -bookworm)         python:3.12-slim
  apt-get build-essential                 only libpq / libgomp etc.
  pip wheel -> /wheels                    COPY --from=builder /wheels
                                          pip install --no-index /wheels
  ~1.2 GB with compilers                  ~250-400 MB, no compiler
        |                                        ^
        +----------- COPY --from=builder --------+

also: non-root USER · ENV PYTHONUNBUFFERED · HEALTHCHECK · pinned base digest
```

## Core concepts

### Multi-stage builds: build once, ship little

Stage one installs compilers and builds wheels. Stage two copies only the wheels and installs them with `--no-index`, so nothing is fetched and nothing compiles at runtime. For a service with a few numeric dependencies this typically cuts the image by more than half.

Two details that make it actually work: pin the base image by tag at minimum (and by digest if you want true reproducibility), and keep `apt-get` in a single `RUN` with `rm -rf /var/lib/apt/lists/*` at the end, or you ship the package index.

### Layer order is a caching strategy

Docker caches a layer until its inputs change, and every later layer is invalidated too. So: copy dependency manifests first, install, then copy source. A one-line code change then rebuilds only the last two layers. Copying `COPY . .` before `pip install` is the most common reason a 10-second rebuild takes four minutes.

Use a `.dockerignore` with `.git`, `__pycache__`, `.venv`, `node_modules`, `data/`, `*.ipynb`, and any `.env`. Without it you copy your secrets and your datasets into the build context.

### Slim versus alpine versus distroless

- **`python:3.12-slim`** (Debian-based) is the pragmatic default. Most wheels work; glibc is present.
- **Alpine** is smaller but uses musl, so many scientific wheels have no prebuilt binary and fall back to compiling from source — often slower and occasionally broken. For numpy/scipy/torch stacks, avoid it.
- **Distroless** images have no shell, which is excellent for security and terrible for debugging; `docker exec` gives you nothing. Good for a stable production service, painful during development.

For GPU work the base must match the CUDA runtime your libraries expect. NVIDIA publishes CUDA base images and PyTorch publishes versioned `pytorch/pytorch` images; using one of those is far easier than installing CUDA yourself. The rule: **the CUDA driver comes from the host, the CUDA runtime comes from the image** — a mismatch shows up as an error about a driver version being insufficient.

### Healthchecks that mean something

A healthcheck is a command the orchestrator runs inside the container. `curl -f localhost:8000/healthz` is only useful if `/healthz` actually checks something. Split the two:

- **Liveness** — is the process alive and not deadlocked? Keep it cheap and local. If it fails, the container is restarted.
- **Readiness** — can it serve traffic right now? Check the database, the model endpoint, the vector store. If it fails, traffic is removed but the container is not killed.

Getting this wrong is a classic outage: a readiness check that fails on a transient database blip causes every replica to be restarted at once, turning a blip into an outage.

### Signals and shutdown

Your service will be told to stop. If it ignores `SIGTERM`, the orchestrator waits and then kills it, dropping in-flight requests. Run the app as PID 1 or use an init that forwards signals, handle `SIGTERM` by finishing current requests, and give the platform a grace period longer than your slowest request.

## Code

A complete multi-stage Dockerfile for a FastAPI AI service. Read the comments: the `PYTHONPATH`/`PATH` line, the non-root user, and the healthcheck are the parts people omit.

```dockerfile
# syntax=docker/dockerfile:1

# ---------- stage 1: build wheels ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Compilers and headers live only in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# Copy manifests first so dependency layers cache across code changes.
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Only runtime shared libraries, no compilers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user; never run a web service as root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=appuser:appuser app ./app
USER appuser

EXPOSE 8000

# Liveness only: cheap, local, no dependencies.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# Exec form so SIGTERM reaches uvicorn directly.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "25"]
```

The two health endpoints, separated by purpose. Readiness returns 503 when a dependency is down, which removes the pod from the load balancer without killing it.

```python
from __future__ import annotations

import httpx
from fastapi import FastAPI, Response, status
from sqlalchemy import text

from app.db import engine
from app.config import settings

app = FastAPI()


@app.get("/healthz")
def liveness() -> dict[str, str]:
    """Cheap and local: is the event loop responsive?"""
    return {"status": "ok"}


@app.get("/readyz")
async def readiness(response: Response) -> dict[str, object]:
    """Checks dependencies. Failure means 'do not send traffic', not 'restart me'."""
    checks: dict[str, bool] = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = True
    except Exception:
        checks["db"] = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.model_base_url}/health")
        checks["model"] = r.status_code == 200
    except Exception:
        checks["model"] = False

    ok = all(checks.values())
    response.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ok, "checks": checks}
```

A GPU variant, and the two lines that matter. `--gpus all` (or the Kubernetes device plugin) provides the driver; the CUDA runtime comes from the base image, so the image tag must be compatible with the host driver.

```dockerfile
# NVIDIA publishes CUDA runtime bases; PyTorch publishes versioned images too.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app ./app
EXPOSE 8000
# Model weights are mounted or downloaded at start; do not bake 14 GB into the image.
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Real use case

A team ships a document-classification API. Their first image was a single-stage build from `python:3.12` with `COPY . .` before `pip install`: 1.9 GB, four-minute rebuilds, and the build context included a `data/` folder with 600 MB of sample PDFs. CI ran out of disk on the runner, so they added a cache-clear step and called it fixed.

The rebuild used three changes. Multi-stage so compilers stayed out; manifests copied before source so a code edit rebuilt in seconds; and a `.dockerignore` that removed data, notebooks and `.venv` from the context. Final image: roughly 380 MB, builds in under a minute on a warm cache, pulls fast enough that autoscaling a new replica no longer takes longer than the traffic spike that triggered it.

The healthcheck split mattered later. Their original single `/health` checked the vector store, and when the store had a 40-second blip, every replica failed its check and was restarted simultaneously — a full outage caused by a transient dependency. After splitting liveness from readiness, the same blip removed replicas from the load balancer, traffic queued, and service resumed without a single restart.

## Common mistakes

**Mistake:** `COPY . .` before installing dependencies, with no `.dockerignore` — **Instead:** copy `requirements.txt` first, install, then copy source, and exclude `.git`, `data/` and `.env` in `.dockerignore`.

**Mistake:** running as root because it is the default — **Instead:** create a user with a fixed uid and `USER` it before `CMD`.

**Mistake:** one health endpoint that checks everything — **Instead:** `/healthz` local and cheap, `/readyz` checks dependencies and returns 503.

**Mistake:** `alpine` for a numpy/torch stack — **Instead:** `python:3.12-slim`, or a distroless image for a stable service.

**Mistake:** baking model weights into the image — **Instead:** mount or download at start from object storage, and cache on a volume.

**Mistake:** shell-form `CMD` so `SIGTERM` goes to the shell — **Instead:** exec form, and a graceful shutdown timeout shorter than the platform's grace period.

## Checkpoint

1. Convert your service to a multi-stage Dockerfile and record image size before and after, plus rebuild time for a one-line code change with a warm cache.
2. Split health into `/healthz` and `/readyz`, point the Dockerfile `HEALTHCHECK` at liveness, and prove that stopping a dependency makes readiness return 503 while the container keeps running.
3. Add `.dockerignore`, run as a non-root user, and verify with `docker run --rm <image> id` that the uid is not 0. Then send `SIGTERM` and confirm in-flight requests complete.

**Self-check:** exercise 1 is done when the final image contains no `gcc` (`docker run --rm <image> which gcc` returns nothing) and a one-line code change rebuilds without reinstalling dependencies.

## Go deeper

- [Kubernetes Essentials for AI Services](05-kubernetes.md) — where these probes and limits get wired up.
- [GitHub Actions CI/CD](06-github-actions.md) — building and pushing this image automatically.
- [Ship: Dockerised Deploy with Rollback](09-ship-docker-deploy.md) — the full pipeline this file feeds.
- [GPU Memory](../16-inference/10-gpu-memory.md) — why the GPU image needs headroom.

---

[<- Previous: GCP and Azure AI Equivalents](03-gcp-azure-ai.md) · [Stage index](README.md) · [Next: Kubernetes Essentials for AI Services](05-kubernetes.md)
