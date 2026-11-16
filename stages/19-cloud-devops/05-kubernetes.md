# 5. Kubernetes Essentials for AI Services

> **Stage 19 · Cloud and DevOps** — the six objects that run an AI service, and why GPUs break the assumptions the others rely on.
>
> **You'll be able to:** deploy your container with a Deployment, expose it with a Service, configure it with ConfigMaps and Secrets, autoscale it with an HPA, and request GPU capacity correctly.
>
> **Prerequisites:** [Docker Done Properly](04-docker.md) · [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md)

## Why this exists

Kubernetes is a large system with a small useful core for an AI service. You can run a production API with six object types and never touch the rest. The trouble is that tutorials start with the whole vocabulary, so people either avoid it or copy manifests they do not understand and then cannot debug.

This file is the six objects, the fields inside them that actually matter, and the two places where AI workloads differ from web workloads: resource limits and GPUs.

The mental model: **a Deployment says what should run; a Service says how to reach it; a ConfigMap and Secret say what to configure it with; an HPA says how many; and requests/limits say how much of a machine each copy may take.**

## The mental model

```text
Deployment (replicas: 3, image: api:9c2e)
   `-- ReplicaSet --> Pod --> container: api
                        |        |-- envFrom: ConfigMap(app-config)
                        |        |-- envFrom: Secret(app-secrets)
                        |        |-- resources: requests/limits
                        |        `-- probes: /healthz, /readyz
                        v
Service (ClusterIP :80 -> pod :8000) --> Ingress (TLS, hostname)
                        ^
HPA (min 2, max 10, target CPU 70%)
```

## Core concepts

### Deployment: desired state, rolling updates

A Deployment declares an image, a replica count and a pod template. The controller reconciles reality to it. Change the image tag and it rolls out gradually, keeping old pods until new ones are ready — provided your readiness probe is honest.

Three fields do most of the work: `strategy.rollingUpdate.maxUnavailable` and `maxSurge` control how aggressive the rollout is; `revisionHistoryLimit` keeps old ReplicaSets around so `kubectl rollout undo` has something to go back to. Keep at least a few revisions; a rollback with no history is just a redeploy of a tag you may not remember.

### Requests, limits, and why they decide whether you stay up

`requests` is what the scheduler reserves — the pod will not be placed on a node without that much free. `limits` is the ceiling; exceeding a CPU limit throttles you, exceeding a memory limit kills the container with OOMKilled.

For AI services this is the highest-value part of the whole file. A pod with no requests is scheduled anywhere, gets starved next to a batch job, and your p95 latency triples with no code change. A pod with a memory limit below the model's footprint restarts forever in a crash loop that looks like a code bug.

Practical rule: set memory `request == limit` for predictable services (avoids eviction surprises), set CPU request to steady-state usage and leave CPU limit off or generous, because CPU throttling on a Python service shows up as mysterious latency spikes.

### ConfigMaps and Secrets: configuration out of the image

Same image, different environments. `envFrom` a ConfigMap for non-sensitive config and a Secret for credentials. Secrets in Kubernetes are base64-encoded, not encrypted at rest by default — enable encryption at rest in the cluster, restrict RBAC on secret reads, and prefer an external secrets operator pulling from a managed store so the values never live in git.

Mounting a Secret as a file is often better than an env var for certificates, because file mounts update on change and env vars do not. For API keys, env vars are fine as long as you rotate by restarting pods, which a rolling update does anyway.

### Services, Ingress and probes

A Service gives a stable DNS name and load-balances to pods matching its selector. `ClusterIP` for internal, and an Ingress (or a cloud load balancer) for public traffic with TLS. The selector must match your pod labels exactly; a typo produces a Service with no endpoints, which presents as a 503 with healthy pods.

Probes: `livenessProbe` restarts, `readinessProbe` gates traffic, `startupProbe` protects slow starts. An AI container that downloads weights at boot can take 60 seconds to be ready — without a `startupProbe`, the liveness probe kills it before it finishes and you get an infinite restart loop that looks like a broken image.

### HPA: scale on the signal that matters

The HorizontalPodAutoscaler changes replica count based on metrics. CPU is the default and often the wrong signal: an LLM API waiting on a remote model uses little CPU while its queue grows. Better signals are in-flight requests or queue depth, exposed as a custom metric.

Two constraints people forget. Scaling up needs a node with room, so the cluster autoscaler must be able to add one — a GPU node group with a fixed size will not. And a new pod must be ready before it helps, so scale-up time includes image pull plus model load; for a heavy service, keep a warm minimum replica count rather than scaling to zero.

### Why GPU scheduling is different

GPUs are not divisible the way CPU is. A pod requests `nvidia.com/gpu: 1` and gets a whole device, exclusively. Consequences:

- **Requests equal limits** for GPUs; there is no partial allocation, so fractional requests are invalid.
- **The node must have the device plugin** installed, or the resource never appears and pods stay Pending forever with "insufficient nvidia.com/gpu".
- **A GPU node is expensive and bin-packs badly.** Two pods each wanting a whole GPU need two nodes even if each uses 30% of the device. That is why people serve several models from one process with a scheduler inside it, or use MIG partitioning where supported.
- **Taints and tolerations** keep ordinary workloads off GPU nodes so a web pod does not consume a machine you are paying a premium for.

## Code

A complete Deployment and Service for the API, with the fields that matter annotated. Note `startupProbe`, memory request equal to limit, and the readiness probe pointing at `/readyz` rather than `/healthz`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: support-api
  labels: { app: support-api }
spec:
  replicas: 3
  revisionHistoryLimit: 5          # keep old ReplicaSets so rollout undo works
  strategy:
    rollingUpdate: { maxUnavailable: 0, maxSurge: 1 }
  selector:
    matchLabels: { app: support-api }
  template:
    metadata:
      labels: { app: support-api }
    spec:
      terminationGracePeriodSeconds: 40
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
      containers:
        - name: api
          image: registry.example.com/support-api:9c2e1f0   # pin a tag or digest, never :latest
          ports: [{ containerPort: 8000 }]
          envFrom:
            - configMapRef: { name: app-config }
            - secretRef: { name: app-secrets }
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits:   { memory: "1Gi" }        # memory limit == request; no CPU limit
          startupProbe:                        # protects a slow model load
            httpGet: { path: /healthz, port: 8000 }
            failureThreshold: 30
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            periodSeconds: 15
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            periodSeconds: 10
            failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: support-api
spec:
  selector: { app: support-api }   # must match pod labels exactly
  ports:
    - port: 80
      targetPort: 8000
```

The config objects and an HPA. The HPA scales on CPU here because it is universally available; swap in a custom metric such as in-flight requests when you have a metrics adapter.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_ENV: "prod"
  LOG_LEVEL: "info"
  MODEL_BASE_URL: "http://model-server.ai.svc.cluster.local:8080"
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:                       # stringData lets you write plain text; it is stored encoded
  OPENAI_API_KEY: "replace-with-real-value-or-use-external-secrets"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: support-api
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: support-api }
  minReplicas: 2                  # keep warm: scale-up includes image pull + startup
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300   # avoid flapping on short lulls
```

The GPU pod and the node group it needs. `nvidia.com/gpu` requests must be integers and must equal the limit; the toleration is what lets this pod land on a tainted GPU node.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 1
  selector:
    matchLabels: { app: model-server }
  template:
    metadata:
      labels: { app: model-server }
    spec:
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: server
          image: registry.example.com/model-server:0.3.1
          resources:
            requests: { cpu: "4", memory: "24Gi", nvidia.com/gpu: 1 }
            limits:   { memory: "24Gi", nvidia.com/gpu: 1 }   # GPU: request == limit
          volumeMounts:
            - { name: weights, mountPath: /models }
      volumes:
        - name: weights
          persistentVolumeClaim: { claimName: model-weights }   # do not bake 14 GB into the image
```

## Real use case

A media company runs a transcription-and-summary service. On first deploy it crash-looped in staging and behaved erratically in production: the container downloaded a 3 GB model at boot, the liveness probe fired at 20 seconds, and Kubernetes restarted it before it was ready. Adding a `startupProbe` with a five-minute budget fixed the loop without touching the image.

Then latency became unpredictable at peak. The pods had no CPU requests, so the scheduler packed them onto nodes already running a batch job, and p95 doubled in the afternoon. Setting CPU requests to measured steady-state usage, and removing the CPU limit that was causing throttling, made latency flat across the day.

Finally, the summariser moved to a GPU node group with a taint and a single replica. That was a deliberate cost decision: one GPU pod serving a queue, rather than two pods each holding a whole GPU at 30% utilisation. They wrote the trade-off in a comment in the manifest so the next person would not "fix" the replica count to two and double the bill.

## Common mistakes

**Mistake:** no `resources.requests`, or a memory limit below the model footprint — **Instead:** measure RSS after load, set requests from that, and set memory request equal to limit with headroom.

**Mistake:** a liveness probe that hits dependencies — **Instead:** liveness local and cheap, readiness checks dependencies, `startupProbe` for slow boot.

**Mistake:** `image: myapp:latest` — **Instead:** an immutable tag or digest, so a rollback is unambiguous.

**Mistake:** fractional or over-subscribed GPU requests — **Instead:** integer `nvidia.com/gpu` equal in requests and limits, plus a device plugin on the node.

**Mistake:** an HPA with no capacity to grow into — **Instead:** a cluster autoscaler, and a warm `minReplicas` for slow-starting services.

**Mistake:** committing real secret values to a manifest — **Instead:** an external secrets operator or sealed secrets; never plain values in git.

## Checkpoint

1. Deploy your service with a Deployment, Service, ConfigMap and Secret, using probes and resource requests. Confirm `kubectl rollout undo` restores the previous revision after a bad tag.
2. Break the readiness dependency on purpose and confirm the pod stays running while `kubectl get endpoints` drops it, then restore and watch traffic return.
3. Add an HPA and load-test it. Record how long a new replica takes from scale-up trigger to ready, and use that number to justify your `minReplicas`.

**Self-check:** exercise 1 is done when `kubectl rollout history deploy/<name>` lists at least two revisions and `kubectl rollout undo` returns the Deployment to the prior image tag without downtime.

## Go deeper

- [Docker Done Properly](04-docker.md) — the image and probes these manifests assume.
- [GitHub Actions CI/CD](06-github-actions.md) — applying these manifests from a pipeline.
- [Terraform Basics](07-terraform.md) — provisioning the cluster and node groups.
- [GPU Cloud Practicalities](08-gpu-cloud.md) — where the GPU nodes come from and what they cost.

---

[<- Previous: Docker Done Properly](04-docker.md) · [Stage index](README.md) · [Next: GitHub Actions CI/CD](06-github-actions.md)
