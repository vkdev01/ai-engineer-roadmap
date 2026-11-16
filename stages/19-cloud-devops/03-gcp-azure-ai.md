# 3. GCP and Azure AI Equivalents

> **Stage 19 · Cloud and DevOps** — mapping what you learned on AWS onto Vertex AI and Azure OpenAI, including the parts that are genuinely different.
>
> **You'll be able to:** name the GCP and Azure service for each AWS primitive you use, and write code that talks to Azure OpenAI and Vertex AI without re-learning the architecture.
>
> **Prerequisites:** [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) · [OpenAI, Anthropic and Friends](../07-llm-apis/01-openai-anthropic.md)

## Why this exists

Cloud choice is usually made by someone else: an enterprise agreement, a data-residency rule, or the fact that your company already runs on one. If you only know AWS, an Azure shop's architecture diagrams look unfamiliar even though they are doing the same thing.

The good news is that the primitives converge. Every cloud has object storage, container hosting, managed Kubernetes, a managed ML platform and at least one hosted frontier-model API. What differs is naming, IAM shape, and a few real architectural differences — notably that Azure OpenAI gives you OpenAI models inside your tenant with Azure's identity and networking, and that Vertex AI bundles model garden access with training and pipelines.

The mental model: **learn the layer, then look up the product name.** The layer is: store bytes, run containers, run models, manage identity, watch spend.

## The mental model

| Layer | AWS | GCP | Azure |
|---|---|---|---|
| Object storage | S3 | Cloud Storage (GCS) | Blob Storage |
| Container hosting | ECS Fargate | Cloud Run | Container Apps |
| Managed Kubernetes | EKS | GKE | AKS |
| Managed ML platform | SageMaker | Vertex AI | Azure Machine Learning |
| Hosted frontier models | Bedrock | Vertex AI Model Garden | Azure OpenAI Service |
| Functions | Lambda | Cloud Functions | Azure Functions |
| Secrets | Secrets Manager / SSM | Secret Manager | Key Vault |
| Identity for workloads | IAM roles | Service accounts | Managed identities |
| Relational DB | RDS | Cloud SQL | Azure SQL / Flexible Server |
| Cache | ElastiCache | Memorystore | Azure Cache for Redis |

Read the table as "same job, different name". The two rows where the shape actually differs are hosted models and identity, so those get their own sections.

## Core concepts

### Azure OpenAI is the OpenAI API inside your subscription

The request and response shapes are close to the OpenAI API — the SDK can often be pointed at an Azure endpoint with a different auth mechanism. What changes:

- **Deployment names, not model names, in the URL.** You create a deployment (say `chat-prod`) of a model version; calls reference the deployment. That indirection is useful: you can swap the model behind a deployment name, which is a rollback lever.
- **`api-version` is a required query parameter** and it pins the API surface. Pin it in config, and treat an upgrade as a change to test.
- **Auth is either an API key or Microsoft Entra ID (formerly Azure AD) tokens.** Prefer the token path so no long-lived key exists; it also gives you per-identity audit trails.
- **Quota is measured in tokens-per-minute per deployment**, not in concurrent requests. A throttled call returns 429 with a retry hint; your retry logic needs to honour it.
- **Data stays in the deployment's region** and the model is not used to train anything, which is often the reason an enterprise picks it.

Availability of specific model versions differs by region and changes over time. Check the current model availability table in the Azure docs rather than assuming a model you use elsewhere is offered.

### Vertex AI is the model garden plus the ML platform

On GCP, hosted models, custom training and endpoints all live under Vertex AI. Practically:

- **Model Garden** lists partner and first-party models; you enable one and call it through a Vertex endpoint.
- **Auth is a service account or workload identity**, with the `aiplatform` scopes. A JSON key file in a container is the anti-pattern; use workload identity on GKE or an attached service account.
- **Batch prediction** jobs read from GCS and write to GCS, which is the analogue of SageMaker batch transform.
- **Vertex Pipelines** is Kubeflow-based, which is more portable than a proprietary pipeline format and correspondingly more verbose.

Cloud Run deserves a mention because it is the simplest "run my container" product on any cloud: it scales to zero, it takes a container image, and it charges per request-time. For an AI API with spiky traffic it is often the cheapest sane option — with the caveat that scale-to-zero means cold starts, and a container that loads a 4 GB model does not cold start well.

### Identity: the one concept worth re-learning per cloud

AWS: a role is assumed by a principal and grants temporary credentials. GCP: a service account is both an identity and a credential holder, and impersonation is the safe pattern. Azure: a managed identity is attached to a resource and gets tokens from the platform, so no secret exists at all.

The invariant across all three: **the workload gets short-lived credentials from the platform, and no human copies a key into a config file.** If you internalise that, you can find the right mechanism in any cloud in ten minutes.

### Portability, and how much to pay for it

Containers and Kubernetes are the portable part. Managed model APIs are not: each has its own SDK shape, auth and quota model. Two practical hedges:

1. **Keep an internal interface** for model calls (`generate(messages, model, temperature) -> Response`) with one adapter per provider. This is the same idea as [Multi-Model Client](../07-llm-apis/10-ship-multi-model-client.md), and it is what lets you move traffic when one provider throttles or re-prices.
2. **Keep the cloud-specific parts thin.** Terraform modules and CI steps differ per cloud; your application code should not know which cloud it is on beyond reading environment variables.

## Code

Calling Azure OpenAI through the OpenAI SDK. The differences from the plain OpenAI call are the endpoint, the `api_version`, the deployment name in place of a model id, and — if you use Entra ID — a bearer token provider instead of an API key.

```python
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# DefaultAzureCredential picks up managed identity in Azure, and your CLI login locally.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],   # https://<resource>.openai.azure.com
    azure_ad_token_provider=token_provider,
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),  # pin it; check docs
)

resp = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],  # a deployment name, not a model id
    messages=[
        {"role": "system", "content": "Answer in two sentences."},
        {"role": "user", "content": "What is a vector index?"},
    ],
    temperature=0.2,
)
print(resp.choices[0].message.content, resp.usage.total_tokens)
```

Calling a model through Vertex AI, with retries that respect throttling. The `location` must match where the model is enabled, and the model id format includes a publisher prefix.

```python
from __future__ import annotations

import os
import time

import vertexai
from google.api_core.exceptions import ResourceExhausted
from vertexai.generative_models import GenerativeModel

vertexai.init(project=os.environ["GCP_PROJECT"], location=os.environ.get("GCP_LOCATION", "asia-south1"))


def generate(prompt: str, model_name: str = "gemini-2.0-flash") -> str:
    """Model ids and available regions change; check the Vertex model garden for current ones."""
    model = GenerativeModel(model_name)
    for attempt in range(5):
        try:
            resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
            return resp.text
        except ResourceExhausted:
            # Honour backoff; do not hammer a throttled quota.
            time.sleep(2**attempt)
    raise RuntimeError("vertex quota exhausted after retries")
```

## Real use case

An insurance company must keep customer data in-country and inside its own tenant. Its AI assistant was prototyped on the OpenAI API directly, which legal blocked. The rebuild kept the application architecture identical and swapped the provider layer: the internal `generate()` interface gained an Azure OpenAI adapter using Entra ID tokens and a deployment name, and the deployment name became the rollback lever — when a model version misbehaved, they pointed `chat-prod` at the previous deployment instead of changing code.

Two things needed real work. Token-per-minute quota was per deployment, so a burst of claims processing starved the customer-facing assistant until they split into two deployments with separate quota. And the pinned `api-version` meant that upgrading the API surface was a deliberate, tested change rather than something that happened overnight.

The portability hedge paid off a year later when a second region required GCP: because the app only knew the `generate()` interface, that region got a Vertex adapter and the rest of the codebase was untouched.

## Common mistakes

**Mistake:** assuming an Azure deployment name is a model name — **Instead:** treat deployments as stable aliases you control, and record which model version sits behind each.

**Mistake:** leaving `api-version` unpinned — **Instead:** pin it in config, and add it to your release notes when it changes.

**Mistake:** shipping a GCP service-account JSON key inside a container — **Instead:** workload identity on GKE or an attached service account.

**Mistake:** modelling Azure OpenAI quota as concurrent requests — **Instead:** think tokens-per-minute per deployment, and honour 429 backoff.

**Mistake:** letting cloud SDK calls spread through the codebase — **Instead:** one internal model interface with per-provider adapters.

**Mistake:** copying a model availability or price table between clouds — **Instead:** verify per region in the provider's current docs.

## Checkpoint

1. Fill the mapping table from memory, then check the three rows you were least sure about against each provider's current docs. Note any name that has changed recently.
2. Add a second adapter to your `generate()` interface for either Azure OpenAI or Vertex AI, driven entirely by environment variables, and switch providers without changing calling code.
3. Write a short migration note for your app: what would move unchanged (containers, app code), what would be rewritten (auth, SDK adapters), and what would need a new decision (model availability in region).

**Self-check:** exercise 2 is done when the same test call succeeds against two providers by changing only env vars, and the returned text is logged with the provider and deployment name.

## Go deeper

- [Multi-Model Client](../07-llm-apis/10-ship-multi-model-client.md) — the interface pattern this file leans on.
- [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) — the AWS equivalents in detail.
- [Kubernetes Essentials for AI Services](05-kubernetes.md) — GKE and AKS are the same Kubernetes.
- [Data Leakage](../20-ai-security/03-data-leakage.md) — why residency rules shape provider choice.

---

[← Previous: Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) · [Stage index](README.md) · [Next: Docker Done Properly](04-docker.md)
