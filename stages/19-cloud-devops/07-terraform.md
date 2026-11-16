# 7. Terraform Basics

> **Stage 19 · Cloud and DevOps** — describing your infrastructure as a file, so it can be reviewed, diffed and rebuilt instead of remembered.
>
> **You'll be able to:** write Terraform for a bucket and a container service, read a `plan` before applying, use a module, and move state to a remote backend so a team can share it.
>
> **Prerequisites:** [GitHub Actions CI/CD](06-github-actions.md) · [AWS for AI Apps: S3, EC2, Lambda](01-aws-s3-ec2-lambda.md)

## Why this exists

Infrastructure created by clicking is infrastructure nobody can reproduce. A bucket with the right policy, a role with the right permissions and a queue with the right retry settings exist as facts in one account, documented by a wiki page that was last accurate eight months ago. The staging environment drifts from production because someone fixed something in a hurry. Nobody knows whether a resource is safe to delete.

Terraform turns that into a text file. It reads the current state of the cloud, compares it to your file, and shows you a plan: what it will create, change, or — the part that deserves your full attention — destroy.

The mental model: **your `.tf` files say what should exist; the state file says what Terraform believes exists; `plan` is the diff; `apply` closes it.**

## The mental model

```text
  main.tf  (desired)          terraform.tfstate  (believed)         cloud (actual)
       \                            |                                 /
        \---- plan: read all three, show the diff -------------------/
                              |
                        apply: make actual match desired,
                               then update state
```

```text
resource "aws_s3_bucket" "docs" { ... }        # declared: created or updated
data   "aws_caller_identity" "me" { ... }      # read-only lookup, not managed
module "api" { source = "./modules/service" }  # reusable group of resources
```

## Core concepts

### Providers, resources, and the state file

A **provider** is the plugin that talks to an API (AWS, GCP, Azure, Kubernetes). A **resource** is one thing Terraform manages. The **state file** is Terraform's record of the mapping between your resource addresses and real cloud ids — it is the most important and most fragile part of the setup.

State is not optional and not disposable. If you lose it, Terraform thinks nothing exists and will try to create duplicates; if two people apply with different copies, one silently overwrites the other's view. That is why remote state with locking is not an advanced topic, it is the first thing you set up for any shared project.

### Plan before apply, always

`terraform plan` is the review artefact. Read it every time, and read the destroy lines first. A small edit — renaming a resource, changing an identifier — can be interpreted as "destroy and recreate", and on a database or a bucket with data that is a disaster.

The two-step habit that prevents it: run plan in CI and post it on the pull request, and make `apply` a separate, approved action. `-out=tfplan` then `apply tfplan` guarantees you applied exactly what you reviewed.

### Variables, outputs, and not committing secrets

Variables make a configuration reusable across environments; `terraform.tfvars` supplies values per environment. Outputs export ids and endpoints for other modules or for your CI to consume.

Secrets are the trap. Never put credentials in a `.tf` file or a committed `tfvars`. Two acceptable patterns: reference a secret manager data source so Terraform reads the value at apply time, or let the CI runner's short-lived credentials do the work and keep only resource definitions here. State can contain sensitive values, which is another reason the backend must be access-controlled and encrypted.

### Modules: the unit of reuse

A module is a directory of resources with inputs and outputs. Use them for things you create more than once: a "service" module that makes an ECS service, its task definition, log group and autoscaling policy. The official registry modules for common patterns (VPC, EKS) are usually better than hand-rolling, and they are versioned so upgrades are deliberate.

Do not over-modularise a two-environment project. One flat directory plus a small module for the repeated service is enough until you feel real pain.

### Remote backend and locking

Put state in object storage with a lock: on AWS, an S3 bucket with a DynamoDB lock table, or the newer lockfile-based mechanism depending on your Terraform version. Then state is encrypted, versioned, and one apply at a time. Add `prevent_destroy` on anything holding data — a bucket of customer documents, a database — so a refactor cannot delete it by accident.

## Code

A root configuration: provider, remote backend, and two resources with the safety settings that matter. The backend block must be present before the first apply, or you will migrate state later under pressure.

```hcl
terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }

  # Remote state: encrypted, versioned, and locked. Create this bucket out-of-band once.
  backend "s3" {
    bucket         = "my-org-tfstate"
    key            = "ai-app/prod/terraform.tfstate"
    region         = "ap-south-1"
    encrypt        = true
    dynamodb_table = "tfstate-locks"
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "ai-app"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

variable "region" {
  type    = string
  default = "ap-south-1"
}

variable "environment" {
  type    = string
  default = "prod"
}

resource "aws_s3_bucket" "documents" {
  bucket = "ai-app-documents-${var.environment}"

  lifecycle {
    prevent_destroy = true   # a refactor must not delete customer documents
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "documents_bucket" {
  value = aws_s3_bucket.documents.bucket
}
```

A module call for the API service, showing the input/output contract. The module itself holds the ECS service and task definition; the root only decides the shape per environment.

```hcl
module "support_api" {
  source = "./modules/service"

  name           = "support-api"
  environment    = var.environment
  image          = "123456789012.dkr.ecr.ap-south-1.amazonaws.com/support-api:${var.image_tag}"
  cpu            = 1024
  memory         = 2048
  desired_count  = var.environment == "prod" ? 3 : 1
  container_port = 8000
  secret_arns = {
    OPENAI_API_KEY = aws_secretsmanager_secret.openai.arn
    DATABASE_URL   = aws_secretsmanager_secret.db_url.arn
  }
}

variable "image_tag" {
  type        = string
  description = "Commit SHA of the image to run; set by CI, never 'latest'"
}

output "service_url" {
  value = module.support_api.url
}
```

The commands, in the order you run them. `-out` then `apply tfplan` is what guarantees you applied the reviewed plan.

```bash
terraform init                      # downloads providers, configures the backend
terraform fmt -check -recursive
terraform validate
terraform plan -out=tfplan          # review this output before continuing
terraform apply tfplan
terraform state list                # see what Terraform believes it manages
```

## Real use case

A startup's AI platform grew from one developer clicking in the console to four engineers and two environments. Nobody could stand up a third environment for a new region, because the knowledge lived in three people's heads and a stale wiki. Worse, a security review found a bucket that had been public for months — someone had opened it for a demo and never closed it.

They rebuilt the core in Terraform: buckets with public access blocked and versioning on, the ECS services as a module, roles with scoped policies, and remote state in S3 with locking. The immediate wins were the security ones: `prevent_destroy` and `block_public_*` became defaults that nobody could forget, and every change to a policy showed up in a plan for review.

The lasting win was the third environment. Standing one up became a new `tfvars` file plus `apply`, in about an hour, with identical networking and IAM. Before, it was an estimate of "two weeks, and we should really write it down". The team also learned the destroy lesson cheaply: renaming a resource in a plan showed "must be replaced" on a bucket, they caught it in review, and switched to `moved` blocks instead.

## Common mistakes

**Mistake:** local state on a laptop, shared by copying the file — **Instead:** remote backend with locking from the first commit.

**Mistake:** `apply` without reading the plan — **Instead:** `plan -out=tfplan`, review, `apply tfplan`, and post plans on PRs.

**Mistake:** credentials in `.tf` or a committed `tfvars` — **Instead:** short-lived CI credentials, and secret manager data sources for values.

**Mistake:** no `prevent_destroy` on data-holding resources — **Instead:** `prevent_destroy` on buckets and databases, plus versioning as a second net.

**Mistake:** renaming a resource and accepting the destroy/recreate — **Instead:** a `moved` block so Terraform relocates the address without touching the resource.

**Mistake:** hand-editing state, or deleting it to "start clean" — **Instead:** `terraform state mv` / `import`, and never delete state that describes real resources.

## Checkpoint

1. Write a root configuration for one bucket with versioning, public access blocked and `prevent_destroy`. Apply it, then prove `terraform plan` shows no changes on a second run.
2. Extract the API service into a module with inputs and one output, and show that changing only a variable changes only the expected resource in the plan.
3. Move state to an S3 backend with locking, run two applies concurrently, and confirm the second is blocked rather than corrupting state.

**Self-check:** exercise 1 is done when `terraform plan` immediately after `apply` reports "No changes", and a plan that tries to delete the bucket is refused because of `prevent_destroy`.

## Go deeper

- [GitHub Actions CI/CD](06-github-actions.md) — running `plan` on PRs and `apply` behind an approval.
- [Kubernetes Essentials for AI Services](05-kubernetes.md) — provisioning the cluster these manifests run on.
- [Containers at Scale: ECS, EKS, SageMaker](02-ecs-eks-sagemaker.md) — the services a module usually wraps.
- [GPU Cloud Practicalities](08-gpu-cloud.md) — quota and node groups, both of which need to be declared.

---

[<- Previous: GitHub Actions CI/CD](06-github-actions.md) · [Stage index](README.md) · [Next: GPU Cloud Practicalities](08-gpu-cloud.md)
