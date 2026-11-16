# 4. Redis Caching

> **Stage 17 · AI Backend** — answering again without paying the provider again.
>
> **You'll be able to:** build an exact-match cache and a semantic cache, and measure whether either one is worth keeping.
>
> **Prerequisites:** [Redis and caching basics](../21-system-design/06-caching.md) · [Embeddings](../09-vector-databases/01-embeddings.md)

## Why this exists

Model calls are the expensive, slow part of an AI backend. Many of them are repeats. Support
users ask the same five questions in slightly different words. A retry sends the identical
prompt. An evaluation run re-sends yesterday's inputs. Every one of those is a full-price call
for an answer you already have.

A cache turns a 2-second, paid call into a 5-millisecond, free one. Redis is the usual store
because it is fast, supports TTLs natively, and is already running in most backends.

Two flavours matter. **Exact-match** caching keys on the request itself — cheap, safe, and it
only catches byte-identical repeats. **Semantic** caching keys on the meaning of the request —
catches paraphrases, but can return a slightly wrong answer, so it needs a threshold and a
safety story.

The mental model: **exact cache asks "have I seen this exact request"; semantic cache asks "have
I seen something close enough".**

## The mental model

```text
request ---> normalise ---> sha256(key) ---> Redis GET  --hit--> return (5 ms)
                                        |
                                      miss
                                        v
                        embed ---> cosine vs cached vectors
                                        |
                              score >= 0.95 ? --yes--> return cached answer
                                        |
                                       no
                                        v
                              call provider, store both keys
```

| | Exact-match | Semantic |
|---|---|---|
| Key | hash of model + messages + params | embedding of the query |
| Catches | identical repeats | paraphrases |
| Hit rate | low to moderate | higher |
| Risk | none | wrong-but-similar answer |
| Extra cost | negligible | one embedding call per miss |
| Tuning | TTL | similarity threshold, TTL, namespace |

## Core concepts

### The cache key must include every parameter that changes the answer

A key of `hash(prompt)` is a bug waiting to happen. The model id, temperature, top_p,
max_tokens, system prompt and any tool definitions all change the output. Serialise them
deterministically and hash the result:

```text
key = sha256(json({"model": m, "messages": msgs, "temperature": t, "max_tokens": n,
                   "prompt_version": v}, sort_keys=True))
```

Include a prompt version so that shipping a new system prompt invalidates the cache instead of
serving yesterday's behaviour forever. Never cache requests with `temperature > 0` unless you
have decided that approximate repeatability is acceptable — the cached answer will not match what
the model would produce now, which is fine for a FAQ and not fine for a creative tool.

### Semantic caching needs a threshold you chose deliberately

Embed the incoming query, compare against cached queries in the same namespace, and return the
cached answer if cosine similarity is above a threshold. The threshold is the whole risk
decision:

- Too low (0.85): "how do I reset my password" and "how do I change my password" collapse into
  one answer, and one of them is wrong.
- Too high (0.99): only near-identical phrasings hit, and you barely beat exact matching.

Most teams land somewhere around 0.93 to 0.97 and tune on a sample of real queries. Whatever you
pick, log every hit with its score so you can review the marginal ones — that log is how you
discover the threshold is wrong before a user does.

### Namespace by tenant, model and prompt version

A semantic cache shared across tenants can leak one customer's answer to another. Prefix every
key with tenant id, model id and prompt version: `cache:{tenant}:{model}:{prompt_version}:{hash}`.
This also gives you clean invalidation — bump the version and the old entries expire on their own.

### Do the arithmetic before you build it

Cache value is `hit_rate x calls x cost_per_call`, minus the cost of embedding misses and
Redis. For a workload of 100,000 calls a month at an illustrative 0.002 currency units per call:

```text
30% hit rate  -> 30,000 saved calls -> 60 units saved
70% misses    -> 70,000 embedding calls -> a fraction of that, if embeddings are cheap
```

The saving is real but modest at low volume. The latency saving is often the stronger
justification: a cache hit returns in milliseconds instead of seconds, which shows up directly in
your p95.

## Code

An async cache client with both modes. It uses `redis.asyncio`, stores vectors as raw float32
bytes, and keeps a bounded candidate list per namespace so a comparison stays cheap.

```python
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis


@dataclass(frozen=True)
class CacheHit:
    answer: str
    score: float
    kind: str          # "exact" or "semantic"


def exact_key(model: str, messages: list[dict], params: dict, prompt_version: str) -> str:
    """Deterministic key over everything that changes the output."""
    blob = json.dumps(
        {"model": model, "messages": messages, "params": params, "pv": prompt_version},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)


class LlmCache:
    def __init__(self, redis: Redis, namespace: str, threshold: float = 0.95,
                 ttl_seconds: int = 24 * 3600, max_candidates: int = 500) -> None:
        self.redis = redis
        self.ns = namespace
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_candidates = max_candidates

    async def get_exact(self, key: str) -> str | None:
        return await self.redis.get(f"llm:exact:{self.ns}:{key}")

    async def put_exact(self, key: str, answer: str) -> None:
        await self.redis.set(f"llm:exact:{self.ns}:{key}", answer, ex=self.ttl)

    async def get_semantic(self, query_vec: np.ndarray) -> CacheHit | None:
        raw = await self.redis.lrange(f"llm:sem:{self.ns}", 0, self.max_candidates - 1)
        best: tuple[float, str] | None = None
        for item in raw:
            payload = json.loads(item)
            stored = np.frombuffer(bytes.fromhex(payload["v"]), dtype=np.float32)
            score = cosine(query_vec, stored)
            if best is None or score > best[0]:
                best = (score, payload["a"])
        if best and best[0] >= self.threshold:
            return CacheHit(answer=best[1], score=round(best[0], 4), kind="semantic")
        return None

    async def put_semantic(self, query_vec: np.ndarray, answer: str) -> None:
        item = json.dumps({
            "v": np.asarray(query_vec, dtype=np.float32).tobytes().hex(),
            "a": answer,
        })
        pipe = self.redis.pipeline()
        pipe.lpush(f"llm:sem:{self.ns}", item)
        pipe.ltrim(f"llm:sem:{self.ns}", 0, self.max_candidates - 1)
        pipe.expire(f"llm:sem:{self.ns}", self.ttl)
        await pipe.execute()

    async def lookup(self, model: str, messages: list[dict], params: dict,
                     prompt_version: str, embed) -> CacheHit | None:
        """Exact first (cheap), then semantic (one embedding call on exact miss)."""
        key = exact_key(model, messages, params, prompt_version)
        if (answer := await self.get_exact(key)) is not None:
            return CacheHit(answer=answer, score=1.0, kind="exact")

        query_vec = np.asarray(await embed(messages[-1]["content"]), dtype=np.float32)
        if (hit := await self.get_semantic(query_vec)) is not None:
            return hit
        return None
```

Called in a handler, a hit returns in a few milliseconds and a miss costs one embedding plus the
provider call. Note the shape: exact lookup first, so the common repeat case never pays for an
embedding at all.

## Real use case

A fintech's internal policy assistant served 400 employees. Analysts asked about expense limits,
travel approval and vendor onboarding, and the same twenty questions accounted for most traffic.
Each call cost money and took about three seconds; the model was also, in the compliance team's
words, "confidently inconsistent" about the exact expense threshold, which was a separate
problem they solved with retrieval.

They added a two-level cache. The exact layer caught retries and the copy-pasted questions
people send in a shared channel. The semantic layer, with a threshold of 0.95, caught the
paraphrases: "what is the meal limit", "how much can I claim for meals", "meal expense cap".
Because the answers were policy statements, a slightly-off match was not dangerous in the way it
would be for a medical or legal answer — that judgement is what made semantic caching acceptable
here, and the team wrote it down.

Two guardrails mattered. They namespaced by tenant and by prompt version, so a policy update
shipped with a version bump and the cache emptied itself. And they logged every semantic hit with
its score into a weekly review queue; that review found four mismatched pairs in the first month,
which they fixed by tightening the threshold to 0.96 for the two most sensitive topics. Their
provider spend on that assistant fell substantially, and p50 latency dropped from seconds to
milliseconds for the majority of questions.

## Common mistakes

- **Mistake:** hashing only the prompt text — **Instead:** include model, all sampling params, system prompt and a prompt version.
- **Mistake:** sharing one semantic namespace across tenants — **Instead:** prefix keys with tenant, model and prompt version.
- **Mistake:** caching high-temperature creative output and expecting it to feel consistent — **Instead:** cache deterministic or FAQ-shaped requests; skip caching where variety is the point.
- **Mistake:** setting the threshold by intuition and never reviewing hits — **Instead:** log every hit with its score and review the marginal band.
- **Mistake:** caching errors or empty answers — **Instead:** only store successful, non-empty completions; otherwise a transient failure sticks for the TTL.

## Checkpoint

1. A workload makes 80,000 calls a month, the hit rate is 35 percent, and each call costs an illustrative 0.004 units. What is the monthly saving, and what does the embedding cost on misses have to stay under to keep it worthwhile?
2. Implement a `stats` method that tracks hits by kind and misses in Redis counters, then run 200 real queries and report the hit rate for each layer.
3. Take 50 real queries with known answers, run them through the semantic cache at thresholds 0.90, 0.95 and 0.98, and plot hit rate against wrong-answer rate. Choose and justify a threshold.

**Self-check:** `80,000 x 0.35 = 28,000` saved calls at 0.004 is 112 units a month; the 52,000 embedding calls on misses must cost well under that in total, which is usually true because embeddings are far cheaper than completions.

## Go deeper

- [Job Queues](05-job-queues.md) — the other place repeats show up, as duplicate job submissions.
- [Semantic Search](../09-vector-databases/03-semantic-search.md) — the similarity machinery underneath the semantic layer.
- [Caching](../21-system-design/06-caching.md) — the system-design view, including invalidation strategy.
- [Cost Tracking](../07-llm-apis/08-cost-tracking.md) — measure the saving rather than assuming it.

---
[← Previous: Async Processing](03-async-processing.md) · [Stage index](README.md) · [Next: Job Queues](05-job-queues.md)
