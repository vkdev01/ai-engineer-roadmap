# 8. Governance

> **Stage 20 · AI Security** — the paperwork that makes an AI system operable: where data lives, how long it lives, who can prove what happened, and what you told users.
>
> **You'll be able to:** write a DPIA-style assessment, an audit trail and a model card for a real feature, and explain your duties under India's DPDP Act and similar regimes at an engineering level.
>
> **Prerequisites:** [PII Redaction and Detection](04-pii-redaction.md) · [Prompt Versioning](../18-mlops-llmops/03-prompt-versioning.md)

## Why this exists

Six months after launch, someone asks: does this feature store customer text, in which country, for how long, and who read it last Tuesday? If the answer takes three engineers a week to reconstruct, you do not have governance — you have archaeology.

Governance is not a document you write once for a compliance team. It is a set of small, durable decisions that make the system answerable: a data map, a retention config, an audit log, a model card, and a review that happens *before* you build. Teams that do this ship faster, because the security questionnaire, the enterprise deal and the incident review all become lookups instead of projects.

**This file is engineering guidance, not legal advice.** India's Digital Personal Data Protection Act and the rules around it, plus sectoral regulators, set obligations that a lawyer must interpret for your product. What you can do is build so that the answers exist.

The mental model: **governance is observability for obligations — you cannot comply with what you cannot see.**

## The mental model

```text
        before build                during build              after ship
   ┌──────────────────┐      ┌────────────────────┐    ┌──────────────────┐
   │ DPIA-style note  │      │ data map           │    │ audit trail      │
   │ purpose, lawful  │ ───► │ residency config   │───►│ retention jobs   │
   │ basis, risks     │      │ retention config   │    │ model card       │
   │ mitigations      │      │ PII controls       │    │ access reviews   │
   └──────────────────┘      └────────────────────┘    └──────────────────┘
             │                        │                        │
             └──────── every claim traces to a config or a log ┘
```

| Question a reviewer asks | Artefact that answers it |
|---|---|
| Why do you hold this data? | DPIA-style note: purpose and minimisation |
| Where is it stored? | data map: system, region, vendor, field list |
| How long? | retention config, with the job that enforces it |
| Who saw it? | audit log: actor, action, record, timestamp |
| What does the model do and not do? | model card: purpose, limits, eval results |
| What did users agree to? | privacy notice and AUP, versioned |

## Core concepts

### Data residency: decide per field, not per product

"We are hosted in India" is usually false at field level. Your database may be in one region while your model provider processes prompts elsewhere, your tracing vendor stores them in a third, and your backups land wherever the managed service puts them.

What to do: build a **data map** with one row per data flow — source, fields, destination system, region, vendor, retention, and whether it is personal data. Then make residency a config value per tenant or per deployment, so a customer who requires in-region processing can be served by a deployment that keeps prompts, embeddings and logs in that region. Where a sub-processor cannot meet the requirement, that is a product limitation you must state, not a detail to discover later.

### Retention: a number, a job, and a test

Retention that exists only in a policy document is not retention. For each store, write the number and implement the deletion:

- **Prompt and response logs:** often 7–30 days is enough for debugging; permanent retention is rarely justifiable.
- **Vector embeddings:** derived data, but linkable to the source; delete with the source document, and verify the delete propagates.
- **Evaluation sets:** keep, but de-identify; a golden set built from real traffic is a personal-data store.
- **Audit records:** usually kept longer than content — keep the *event* (who, what, when) and not the content, so you get accountability without keeping the payload.

Then test it: a job that deletes rows older than the window, and an assertion in CI or a scheduled check that no row is older than the window. A retention policy without a failing test is a wish.

### DPIA-style thinking, as an engineer

A data protection impact assessment asks: what is the purpose, what is the minimum data needed, who is affected, what could go wrong, and what reduces the risk. You do not need a template to start; you need the five answers written down before you build, because two of them change the design.

1. **Purpose** — one sentence. "Answer customer support questions using the ticket history" is a purpose. "Improve the product" is not.
2. **Minimisation** — which fields, and can the task be done without the sensitive ones? This is where the design decision lives (see [Data Leakage](03-data-leakage.md)).
3. **Affected people** — whose data, and are any of them children or in a vulnerable category? That changes your obligations materially.
4. **Risks** — leakage, wrong answer acted upon, bias, unavailable service. Rank them plainly.
5. **Mitigations** — the controls from this stage, named with an owner.

Keep it to two pages. Attach it to the design doc for the feature (see [Ship: Design Document](../21-system-design/10-ship-design-doc.md)) so it is reviewed with the code, not after it.

### Audit trails that survive an incident

An audit record answers "who did what, to which record, when, and under what authority". For an AI system, that includes model and prompt version, tool calls made, and human approvals given. Design rules:

- **Append-only** and separate from the application database, so a compromised app cannot rewrite history.
- **Hash arguments, not raw content**, when the content is personal data; keep the decision, drop the payload.
- **Correlate with a request id** that appears in traces, so one id ties together the prompt version, the model, the retrieved doc ids, the tool calls and the response.
- **Version everything** — prompt, model, retrieval config — so "which version produced this answer" is a field, not a guess.

### Model cards: what you tell the next engineer

A model card is a short, honest description of a system's purpose and limits. For an applied LLM feature, useful sections are: purpose and intended users; what it must not be used for; the models and prompt versions used; evaluation results including per-slice numbers; known limitations (languages, document types, freshness); data handling summary; and the owner. Two paragraphs of honesty about limits saves months of wrong assumptions downstream.

### DPDP Act and similar regimes, at a practical level

India's Digital Personal Data Protection Act establishes duties around notice and consent, purpose limitation, minimisation, accuracy, storage limitation, security safeguards, breach notification, and the rights of data principals, with obligations on significant data fiduciaries that include impact assessments and audits. Other regimes you will meet — the EU GDPR, sectoral Indian rules, and customer-imposed contractual terms — rhyme with these.

What that means for your build, in engineering terms: you need a lawful basis recorded for each purpose; notice that names the processing; the ability to delete and correct on request; security controls you can evidence; and a way to detect and report a breach. Whether a given feature is compliant is a legal question — ask a lawyer, and give them the data map and the DPIA note to work from. What you must not do is assume an LLM feature is outside these duties because "we do not store the prompt" — you may be storing it in four other places.

## Code

A retention enforcement job plus the config that defines it. The config is the artefact a reviewer reads; the job is what makes it true.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class Store:
    name: str
    table: str
    ts_column: str
    retention_days: int
    holds_personal_data: bool


STORES: list[Store] = [
    Store("prompt_logs", "llm_events", "created_at", 14, True),
    Store("traces", "spans", "started_at", 14, True),
    Store("embeddings", "doc_chunks", "deleted_at", 0, True),  # delete with source
    Store("audit", "audit_events", "created_at", 400, False),  # events, no payload
]


def cutoff(store: Store, now: datetime | None = None) -> datetime | None:
    if store.retention_days == 0:
        return None  # handled by the source-deletion path, not by age
    now = now or datetime.now(timezone.utc)
    return now - timedelta(days=store.retention_days)


async def purge(conn, store: Store) -> int:
    limit = cutoff(store)
    if limit is None:
        return 0
    # Parameterised; never interpolate identifiers from config into SQL blindly.
    sql = f"DELETE FROM {store.table} WHERE {store.ts_column} < $1"
    result = await conn.execute(sql, limit)
    return int(result.split()[-1]) if isinstance(result, str) else 0


async def check(conn, store: Store) -> None:
    """Fail loudly if any row violates the window. Run this on a schedule."""
    limit = cutoff(store)
    if limit is None:
        return
    sql = f"SELECT count(*) FROM {store.table} WHERE {store.ts_column} < $1"
    n = await conn.fetchval(sql, limit)
    if n:
        raise RuntimeError(f"{store.name}: {n} rows older than retention window")
```

The audit record shape below is what makes the rest answerable: versioned, correlated, and free of payload.

```python
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    actor_id: str          # hashed user or service id
    tenant_id: str
    action: str            # "answer", "tool_call", "human_approval", "delete"
    prompt_version: str
    model_id: str
    retrieved_doc_ids: list[str] = []
    tool_names: list[str] = []
    approved_by: str | None = None
    created_at: datetime = datetime.now(timezone.utc)


def to_row(e: AuditEvent) -> dict[str, object]:
    """Append-only write. No prompt text, no response text, no PII."""
    return e.model_dump(mode="json")
```

## Real use case

**A resume-screening feature inside an HR platform used by Indian enterprises.** The feature summarised applications and ranked them. Before launch, the team wrote a two-page DPIA-style note: purpose (shortlist assistance, human decides), minimisation (no date of birth, no address, no photo passed to the model), affected people (candidates), risks (bias against regional institutions, wrong rejection, retention creep), mitigations (probe-pair tests, slice reporting, 30-day prompt retention, append-only audit).

The note changed the design twice. It removed the photo and address fields from the prompt, and it added a hard rule that the model could rank but never reject — the human decision was logged with `approved_by`. Six months later, when a candidate asked what data the company held, the answer came from the data map and the audit trail in under an hour. When a customer's security team asked about residency, the data map named each sub-processor and its region. None of it was heroic; all of it was written down before launch.

## Common mistakes

- **Mistake:** a retention policy in a wiki with no job enforcing it — **Instead:** a config value, a purge job, and a scheduled check that fails on a violation.
- **Mistake:** assuming "we do not store prompts" because the database table is small — **Instead:** build the data map across provider, tracing, backups and evals; that is where the copies are.
- **Mistake:** keeping raw prompt text in the audit log — **Instead:** log the event and hashes; keep content in the short-retention store.
- **Mistake:** writing the model card from the marketing page — **Instead:** include per-slice eval results and known limitations; the honest version is the useful one.
- **Mistake:** treating compliance as a legal team task with no engineering input — **Instead:** write the DPIA note as engineering, hand it to legal, and keep it attached to the design doc.

## Checkpoint

1. Write the data map for one feature: five rows, one per data flow, with region and retention. Name one flow you cannot currently answer.
2. Implement the retention check for one store and run it. Then plant an old row and confirm it fails.
3. Draft a one-page model card for an app you built, including per-slice results and three honest limitations. Note which claims are engineering input rather than legal conclusions.

**Self-check:** The data map should have one row per data flow — source, fields, destination, region, vendor, retention — and the flow you cannot answer is usually a sub-processor (tracing, evals, or backups) rather than the primary database.

## Go deeper

- [Ship: Design Document](../21-system-design/10-ship-design-doc.md) — where the DPIA-style note attaches in a real design review.
- [PII Redaction and Detection](04-pii-redaction.md) — the minimisation controls this file records.
- [Bias and Fairness](07-bias-fairness.md) — the numbers a model card must include.
- [Prompt Versioning](../18-mlops-llmops/03-prompt-versioning.md) — making "which version" a field, not a guess.

---
[← Previous: Bias and Fairness](07-bias-fairness.md) · [Stage index](README.md) · [Next: Guardrails](09-guardrails.md)
