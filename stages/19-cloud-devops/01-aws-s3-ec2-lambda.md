# 1. AWS for AI Apps: S3, EC2, Lambda

> **Stage 19 · Cloud and DevOps** — the three primitives that carry most AI workloads, and the traps that make the bill surprising.
>
> **You'll be able to:** store documents and artefacts in S3, reason about when a GPU instance beats an API call, and use Lambda for webhooks and glue without hitting its limits.
>
> **Prerequisites:** [Stage 18 — MLOps and LLMOps](../18-mlops-llmops/README.md) · [HTTP and REST APIs](../01-programming-basics/07-http-rest-apis.md)

## Why this exists

You can build an AI app on one machine. The moment it has users, three problems appear: files that must survive the machine (uploaded PDFs, generated artefacts, model weights), compute that must be bigger than your laptop (a GPU for a fine-tune or a self-hosted model), and small event handlers that must run when something happens (a payment webhook, a file landing in a bucket).

AWS solves these with S3, EC2 and Lambda respectively. They are not the only options, and often they are not the cheapest, but they are the vocabulary every cloud job assumes you know. This file is about the primitives plus the two things tutorials skip: IAM and cost.

The mental model: **S3 is a durable key-value store for bytes; EC2 is a rented computer you must care for; Lambda is a function that runs for a few minutes and forgets everything.**

## The mental model

```text
user uploads PDF
      |
      v
[ S3 bucket: documents/ ]  --event-->  [ Lambda: parse + chunk ]  --writes-->  [ S3: chunks/ ]
      |                                        |
      |                                        +--> queue / index
      v
[ EC2 GPU instance ]  (self-hosted embedding or vLLM endpoint)
      |
      v
[ your API on ECS/EKS ]  --reads-->  S3 artefacts, calls GPU endpoint
```

| Need | Primitive | Watch out for |
|---|---|---|
| Store documents, artefacts, weights | S3 | Egress charges; no per-object locking by default |
| Run a GPU model 24x7 | EC2 (GPU family) | Idle cost; driver and CUDA versions |
| Run a GPU job occasionally | EC2 spot or a managed job | Interruption; quota limits |
| React to a webhook or an S3 event | Lambda | 15-minute ceiling; cold starts |
| Long batch job | Not Lambda | Use a container on ECS or a batch service |

## Core concepts

### S3 is the default home for bytes

Buckets are global names, objects are keys, and durability is the reason people trust it. For AI work the important design choices are:

- **Prefix layout by lifecycle, not by date.** `documents/{tenant}/{doc_id}/original.pdf` next to `documents/{tenant}/{doc_id}/chunks.jsonl` makes deletion and per-tenant export trivial. Date-based prefixes (`2026/03/04/...`) are for logs, not documents.
- **Versioning on, lifecycle rules on.** Versioning saves you from an accidental overwrite of a golden dataset. A lifecycle rule that moves raw uploads to infrequent-access storage after 30 days is free money.
- **Egress is the cost that bites.** Bytes leaving AWS cost money; bytes moving from S3 to EC2 or Lambda in the same region generally do not. Serving model outputs straight from S3 to browsers is a way to build a surprisingly large bill.
- **Presigned URLs instead of public buckets.** The browser uploads directly to S3 with a short-lived signed URL; your API never proxies gigabytes. Never make a bucket of customer documents public.

### EC2: when to rent a GPU and when not to

Rent a GPU instance when you need a model running continuously at high utilisation, when you need to fine-tune, or when data cannot leave your environment. Do not rent one to serve a few thousand requests a day — an API call to a hosted model is usually cheaper and definitely less work, because a GPU instance bills for the hours it exists, not the tokens it serves.

Instance family names and prices change constantly and vary by region. Treat any specific name or figure as something to verify in the console's current pricing page rather than a fact to memorise. Two operational rules that survive all changes: put the instance in a private subnet and reach it via a load balancer or a bastion, and stop it when idle — a stopped instance stops compute billing but still bills for attached storage.

### Lambda: glue, not a model server

Lambda is right for event handlers: an S3 upload triggers parsing, a Stripe-style webhook triggers an LLM call and a database write, a cron triggers a nightly cleanup. It is wrong for anything long, stateful, or GPU-bound.

Its practical limits shape the design: a maximum runtime on the order of 15 minutes, a payload ceiling, cold starts measured in hundreds of milliseconds to seconds depending on runtime and package size, and no GPU. Keep the deployment package small (a slim container image if dependencies are heavy), keep the handler idempotent because retries are normal, and move anything slower than a minute into a queue with a worker.

### IAM in one page

Every AWS call is authorised by a policy attached to a principal. Three habits prevent 90% of incidents:

1. **One role per workload**, not one shared "app" user with broad rights. The Lambda that reads `documents/*` gets exactly that prefix.
2. **No long-lived access keys in code or CI.** Use instance profiles for EC2, execution roles for Lambda and ECS, and OIDC federation for GitHub Actions.
3. **Deny by default, grant by prefix and action.** `s3:GetObject` on `arn:aws:s3:::bucket/documents/${tenant}/*` is the shape; `s3:*` on `*` is how a breach becomes a headline.

### The cost traps worth knowing before the invoice

- **Idle GPU instances.** The single largest surprise. A stopped instance still pays for volumes; a running one pays for everything.
- **NAT Gateway data processing.** If your private-subnet instances pull model weights from the internet through a NAT gateway, the per-gigabyte charge adds up fast. Use a VPC endpoint for S3.
- **Cross-region and internet egress.** Serving artefacts from a bucket in another region is both slower and billable.
- **Log retention.** CloudWatch log groups default to never expiring. Set a retention period on day one.
- **Forgotten dev resources.** A load balancer or a managed database left in a dev account costs more than the dev instance.

## Code

Uploading a document and handing the browser a presigned URL, so the API never touches the bytes. The content-type constraint and the short expiry are the security-relevant lines.

```python
from __future__ import annotations

import boto3
from botocore.config import Config

s3 = boto3.client("s3", config=Config(signature_version="s3v4", retries={"max_attempts": 5}))
BUCKET = "my-ai-app-documents"


def presign_upload(tenant_id: str, doc_id: str, content_type: str = "application/pdf") -> dict[str, str]:
    key = f"documents/{tenant_id}/{doc_id}/original.pdf"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=600,  # ten minutes; long enough for a slow phone upload
    )
    return {"upload_url": url, "key": key}
```

Reading the object back and writing a derived artefact. Note `Body.read()` is fine for small JSONL; stream large files to disk instead.

```python
from __future__ import annotations

import json

import boto3

s3 = boto3.client("s3")


def read_chunks(tenant_id: str, doc_id: str) -> list[dict]:
    key = f"documents/{tenant_id}/{doc_id}/chunks.jsonl"
    body = s3.get_object(Bucket="my-ai-app-documents", Key=key)["Body"].read().decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def write_summary(tenant_id: str, doc_id: str, summary: str) -> None:
    s3.put_object(
        Bucket="my-ai-app-documents",
        Key=f"documents/{tenant_id}/{doc_id}/summary.txt",
        Body=summary.encode("utf-8"),
        ContentType="text/plain",
        ServerSideEncryption="AES256",
    )
```

The IAM policy the parsing Lambda actually needs. Compare it to `s3:*` — this is what "deny by default" looks like in practice.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadUploadsWriteDerived",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::my-ai-app-documents/documents/*/original.pdf"
    },
    {
      "Sid": "WriteDerivedOnly",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::my-ai-app-documents/documents/*/chunks.jsonl"
    }
  ]
}
```

## Real use case

A legal-tech product lets firms upload contracts and get clause summaries. The first version ran everything in one Flask app on a single EC2 instance: uploads hit the app, parsing ran inline, summaries were stored on the instance's disk. Two things broke. When the instance was replaced during a deploy, every stored summary vanished — the disk was ephemeral. And a firm uploading a 300-page contract held a worker for four minutes, so other users timed out.

The rebuild used three primitives for three jobs: presigned S3 uploads so the API never handled the file, an S3 event to a Lambda that parsed and chunked, and the summariser on ECS behind the API. The instance disk stopped mattering because artefacts lived in S3 under `documents/{tenant}/{doc_id}/`, which also made per-tenant export and deletion a prefix operation.

The cost trap they hit next is worth repeating: the parsing Lambda pulled a Python dependency bundle through a NAT gateway on every cold start until they moved it into a container image stored in the same region. That single change removed most of the NAT data-processing line on their bill.

## Common mistakes

**Mistake:** proxying uploads through your API — **Instead:** presigned URLs so the browser talks to S3 directly, with a content-type and size constraint.

**Mistake:** a public bucket "just for the demo" — **Instead:** private bucket plus presigned GET with short expiry, and a bucket policy that blocks public access.

**Mistake:** one IAM user with broad keys pasted into CI secrets — **Instead:** roles for AWS workloads and OIDC federation for CI, so no long-lived keys exist.

**Mistake:** running a GPU instance for a workload that uses it 5% of the time — **Instead:** a hosted model API, or a spot instance with checkpointing.

**Mistake:** putting a multi-minute job in Lambda — **Instead:** queue plus worker, and keep Lambda for handlers under a minute.

**Mistake:** quoting an instance price from a blog post — **Instead:** check the current pricing page for your region and record the date you checked it.

## Checkpoint

1. Create a private bucket with versioning on, write a presigned upload URL from Python, upload a file with `curl`, then read it back with a presigned GET. Confirm the URL stops working after expiry.
2. Attach the minimal policy above to a Lambda role and prove that writing outside the allowed prefix is denied. Capture the error message.
3. Write a one-page cost note for your app: which resources bill by hour, which by request, which by gigabyte, and which three you would check first if the bill doubled.

**Self-check:** exercise 1 is done when the PUT succeeds with the presigned URL, the object appears under `documents/{tenant}/{doc_id}/original.pdf`, and a second PUT after the expiry window returns an access-denied error.

## Go deeper

- [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) — where the API and the GPU endpoint actually run.
- [Docker Done Properly](04-docker.md) — packaging the parser so it cold-starts faster.
- [GPU Cloud Practicalities](08-gpu-cloud.md) — quota, spot and storage for the GPU side.
- [PII Redaction](../20-ai-security/04-pii-redaction.md) — what to strip before documents land in a bucket.

---

[<- Previous: Stage 18 — MLOps and LLMOps](../18-mlops-llmops/README.md) · [Stage index](README.md) · [Next: Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md)
