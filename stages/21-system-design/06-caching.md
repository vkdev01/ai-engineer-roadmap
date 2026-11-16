# 6. Caching Layers

> **Stage 21 · System Design** — four caches that matter in an AI app, what each one saves, and the invalidation rule that keeps it from serving a stale or wrong answer.
>
> **You'll be able to:** place exact, semantic, prefix and retrieval caches correctly, size the saving with arithmetic, and invalidate without leaking across tenants.
>
> **Prerequisites:** [Redis Caching](../17-ai-backend/04-redis-caching.md) · [Context Compression](../08-prompt-and-context/04-context-compression.md)

## Why this exists

An AI app repeats itself constantly. Every request re-sends the same system prompt. Users ask the same twenty questions in slightly different words. A single answer needs the same policy document retrieved again and again. Each repetition is latency you pay and tokens you buy twice.

Caching is the cheapest performance work available, and the most dangerous, because a cache turns a correctness bug into a persistent one: a wrong answer computed once can be served a thousand times, and a cache that ignores tenant identity is a data breach with a hit rate. The rule that makes caching safe is simple and non-negotiable — **every cache key includes everything the answer depends on.**

The mental model: **cache the expensive, deterministic-ish parts; key them by identity plus policy version; and know exactly what invalidates them.**

## The mental model

```text
 request
   │
   ├─[1] exact cache ──────── identical request → stored answer   (hit: ~0 tokens)
   ├─[2] semantic cache ───── near-identical question → answer    (hit: ~0 tokens)
   ├─[3] prefix/prompt cache  same system prefix → cheaper input  (hit: partial)
   └─[4] retrieval cache ──── same query → same chunks            (hit: saves search)
```

| Cache | Key | Saves | Invalidation |
|---|---|---|---|
| Exact | hash(tenant, normalized request, policy version) | the whole model call | TTL + policy version |
| Semantic | embedding of question, similarity ≥ threshold, tenant | the whole model call | TTL + policy version + threshold review |
| Prefix / prompt | static system prefix, model, version | most input tokens | prefix version |
| Retrieval | hash(tenant, query, filters, index version) | the search hop | index version + TTL |

## Core concepts

### Exact cache: boring, safe, and worth it

Hash the normalised request (lowercased, whitespace-collapsed), include the tenant id and the policy version, and store the validated answer with a TTL. Safe because two identical requests should get the same answer — provided the answer does not depend on anything else (time, user history, fresh data). If it does, include that in the key or do not cache it.

Two rules. Cache **only answers that passed output validation** (see [Guardrails](../20-ai-security/09-guardrails.md)) — a cache is an amplifier, so never store something you would not send. And include the **policy version** so a prompt or filter change invalidates everything at once, which is the single most common cache bug in AI apps.

### Semantic cache: powerful, and only as good as its threshold

Embed the incoming question, search a cache of previously answered questions, and if cosine similarity exceeds a threshold, return the stored answer. This catches "how do I reset my password" versus "password reset help", which exact caching never will.

The threshold is a quality dial you must tune with data, not intuition. Too low and you serve an answer to a different question — the worst kind of bug, because it looks confident and is subtly wrong. Too high and you get no hits. Practical rules: measure precision on a labelled sample of near-duplicate pairs before enabling; keep the threshold in config; namespace by tenant (a semantic cache shared across tenants is a leak); exclude high-stakes intents (billing disputes, legal, medical) from semantic caching entirely; and always keep the TTL short enough that a policy change surfaces.

### Prefix / prompt caching: the one with no quality risk

Providers can cache a long, repeated prompt prefix so that the cached portion is billed and processed more cheaply on later calls. This is the best-value cache in the stack because it cannot produce a wrong answer — the tokens are still there, just cheaper to process.

To use it: keep the static part (system instructions, tool schemas, few-shot examples) at the **start** of the prompt and byte-identical between requests; put the variable part (the user's question, retrieved chunks) at the end. Any change early in the prefix invalidates the cached portion, so avoid interpolating timestamps or request ids into the system prompt — a mistake that silently disables the cache. Prefix caching is a provider feature; check the current docs for how it is enabled and what counts as a stable prefix.

### Retrieval cache, and the index-version key

Cache the result of a retrieval query — the chunk ids and scores — keyed by tenant, normalised query, filter set and **index version**. Bump the index version on every re-index or embedding-model change and the whole retrieval cache is invalidated in one step, which is exactly what you want when the corpus changes.

Cache ids, not text, where you can: the text may be large, and fetching by id keeps a single source of truth. It also means a deleted document disappears from the cache's effect as soon as the fetch misses.

### The arithmetic: does caching pay?

Labelled example — **all values illustrative; substitute your own measurements and prices**:

| Assumption | Value |
|---|---|
| Requests per day | 100,000 |
| Exact-cache hit rate (measured) | 12% |
| Semantic-cache additional hit rate | 18% |
| Combined hit rate | 30% |
| Input tokens per request | 2,500 |
| Output tokens per request | 350 |
| Input price (placeholder) | `$P_in` per 1M tokens |
| Output price (placeholder) | `$P_out` per 1M tokens |
| Cache lookup cost | ~2 ms, negligible vs a model call |

Saved per day: `100,000 × 0.30 = 30,000` model calls avoided, i.e. `30,000 × 2,500 = 75,000,000` input tokens and `30,000 × 350 = 10,500,000` output tokens. Money saved: `75 × $P_in + 10.5 × $P_out` per day. If `$P_in` were 1.0 and `$P_out` were 4.0 per million (a placeholder ratio, not a real price), that is roughly `75 + 42 = $117` a day — and the same 30% of requests also skip the model latency entirely, which is the bigger product win.

The honest counterweight: a 30% hit rate means 70% of lookups are pure overhead (small), and the semantic cache needs tuning and monitoring. Set the threshold from measured precision, and alert on the hit rate falling after a prompt change — that usually means the key changed.

## Code

A cache layer that makes the key explicit and refuses to serve an answer that was not validated. The tenant id and policy version are part of the key by construction, not by convention.

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

POLICY_VERSION = "2026-11-01"  # bump on any prompt, filter or guardrail change
INDEX_VERSION = "idx-42"       # bump on re-index or embedding-model change


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def exact_key(tenant_id: str, question: str) -> str:
    digest = hashlib.sha256(normalise(question).encode()).hexdigest()
    return f"ans:{tenant_id}:{POLICY_VERSION}:{digest}"


def retrieval_key(tenant_id: str, question: str, filters: dict[str, str]) -> str:
    blob = json.dumps({"q": normalise(question), "f": filters}, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return f"ret:{tenant_id}:{INDEX_VERSION}:{digest}"


@dataclass(frozen=True)
class CachedAnswer:
    text: str
    validated: bool  # only True answers are ever stored


async def get_or_answer(redis, session, question: str, produce) -> tuple[str, bool]:
    key = exact_key(session.tenant_id, question)
    hit = await redis.get(key)
    if hit:
        payload = json.loads(hit)
        if payload.get("validated"):
            return payload["text"], True  # cache hit, no model call
        await redis.delete(key)           # never serve an unvalidated answer

    text, ok = await produce(question)     # the real pipeline
    if ok:                                # store only what passed validation
        await redis.set(key, json.dumps({"text": text, "validated": True}), ex=3600)
    return text, False
```

The semantic cache, with the threshold in config and a guard that keeps high-stakes intents out. The similarity check happens on the tenant's own namespace, so a cross-tenant hit is impossible.

```python
from __future__ import annotations

from dataclasses import dataclass

SEMANTIC_THRESHOLD = 0.93  # tune from measured precision; do not guess
NO_SEMANTIC = {"billing_dispute", "legal", "medical", "account_closure"}


@dataclass(frozen=True)
class SemanticHit:
    answer: str
    score: float
    question: str


async def semantic_lookup(store, session, embedding: list[float],
                          intent: str) -> SemanticHit | None:
    if intent in NO_SEMANTIC:
        return None  # never reuse an answer for a high-stakes intent
    rows = await store.search(
        vector=embedding,
        k=1,
        filter={
            "tenant_id": {"$eq": session.tenant_id},  # namespace by tenant
            "policy_version": {"$eq": POLICY_VERSION},  # and by policy version
        },
    )
    if not rows:
        return None
    best = rows[0]
    if best["score"] < SEMANTIC_THRESHOLD:
        return None
    return SemanticHit(answer=best["answer"], score=best["score"], question=best["question"])
```

## Real use case

**An internal HR policy assistant for a company with about 8,000 employees.** Traffic was highly repetitive: leave policy, reimbursement limits, notice periods, insurance enrolment. Every request re-sent an 1,800-token system prompt plus the tool schemas, and every answer re-retrieved the same policy chunks.

Three changes, in order of risk. First, the static prefix moved to the front, byte-identical, with the timestamp removed from the system prompt — that alone made prefix caching effective and was pure gain. Second, retrieval results were cached keyed by query and index version, cutting the search hop for repeated questions. Third, an exact cache with a one-hour TTL went in for questions with no user-specific component, and a semantic cache with a high threshold was enabled only for a small set of intents after the team labelled 200 near-duplicate pairs and measured precision at the threshold they chose.

They deliberately excluded anything involving a specific employee's balance, because those answers depend on the person. The measured effect was a large drop in model calls for the repetitive majority and a visible drop in p50 latency; the semantic cache was the smallest contributor by volume, and the one that needed the most tuning. Bumping the policy version after a leave-policy change invalidated all three layers in one deploy.

## Common mistakes

- **Mistake:** caching an answer before output validation — **Instead:** store only validated answers; treat the cache as an amplifier of things you would send.
- **Mistake:** a semantic cache shared across tenants — **Instead:** namespace by tenant and policy version; exclude high-stakes intents.
- **Mistake:** interpolating a timestamp or request id into the system prompt — **Instead:** keep the static prefix byte-identical so prefix caching can work.
- **Mistake:** no policy version in the key, so a prompt change serves old answers — **Instead:** version the key and bump it on every prompt, filter or guardrail change.
- **Mistake:** choosing the similarity threshold by intuition — **Instead:** label a sample of near-duplicate pairs and measure precision at each candidate threshold.

## Checkpoint

1. Add an exact cache to an app you have built with the tenant and policy version in the key. Measure the hit rate over a day of sampled traffic.
2. Move your system prompt and tool schemas into a byte-identical static prefix and confirm from the provider's usage response that the cached portion is being billed differently. (Check the docs for how that is reported.)
3. Label 50 pairs of questions as "same answer" or "different answer", then compute precision at two candidate similarity thresholds. Choose one and write the sentence defending it.

**Self-check:** The key must include tenant id, normalised request and policy version — anything the answer depends on and is not in the key becomes a bug that the cache will serve repeatedly.

## Go deeper

- [Retrieval Cache and Reranking](../10-rag/06-reranking.md) — where retrieval caching interacts with reranking.
- [Cost and Latency Engineering](08-cost-latency.md) — the unit economics these caches move.
- [Fault Tolerance](07-fault-tolerance.md) — caches as a degraded mode when the model is down.
- [Multi-Tenant Design](05-multi-tenant.md) — the namespacing rule in full.

---
[← Previous: Multi-Tenant Design](05-multi-tenant.md) · [Stage index](README.md) · [Next: Fault Tolerance](07-fault-tolerance.md)
