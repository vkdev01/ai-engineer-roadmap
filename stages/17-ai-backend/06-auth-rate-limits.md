# 6. Auth and Rate Limits

> **Stage 17 · AI Backend** — knowing who is calling, and capping what they can spend.
>
> **You'll be able to:** authenticate API keys and JWTs, and enforce per-tenant quotas with a token bucket.
>
> **Prerequisites:** [FastAPI and Flask](01-fastapi-flask.md) · [Redis Caching](04-redis-caching.md)

## Why this exists

An AI endpoint with no auth is a free GPU for anyone who finds the URL. An AI endpoint with auth
but no limits is a way for one customer's runaway script to consume your entire provider quota and
take the service down for everyone else. Both failures are common, and both are cheap to prevent.

Two separate questions need separate answers. **Authentication** is "who is this". **Authorisation
and quota** is "how much may they do". A key identifies a tenant; a limiter and a quota decide how
much of your capacity that tenant may occupy.

The mental model: **auth gives every request an owner; limits make each owner's consumption
finite and visible.**

## The mental model

```text
request -> [ API key or JWT ] -> tenant_id + scopes
              |                        |
              v                        v
        reject 401              token bucket (per second)
                                       |
                                       v
                               monthly token quota (per tenant)
                                       |
                                       v
                                handler / model call
```

| Mechanism | Answers | Revocable | Good for |
|---|---|---|---|
| API key | which tenant, which environment | instantly | server-to-server, long-lived integrations |
| JWT | which user, which scopes, until when | at expiry | browser sessions, short-lived tokens |
| Token bucket | requests per second | n/a | burst protection |
| Token quota | spend per month | n/a | billing and fairness |

## Core concepts

### Store a hash of the API key, never the key

A key is a password. Store `sha256(key)` with a short visible prefix for lookup, so a database
leak does not hand out working credentials. Show the full key once, at creation, and never again.

```text
key format:  sk_live_<random 32 bytes base62>
stored:      prefix "sk_live_ab12" (indexed) + sha256(full key) + tenant_id + scopes
```

On each request, hash the presented key, look up by prefix, and compare in constant time. A
plain `==` on strings is a timing side channel, though a minor one next to the bigger risks.

### JWTs are signed claims, not encrypted secrets

A JWT is `header.payload.signature`, base64url encoded, with the payload readable by anyone who
holds it. Sign with RS256 or ES256 so verifiers do not need the signing key. Put the minimum in
the payload: `sub` (user), `tenant`, `scopes`, `exp`, `iat`, `jti`. Never put a secret or
personal data in it.

Verify three things: the signature, the expiry, and the issuer/audience. A library that decodes
without verifying is not authentication.

### Token buckets, with arithmetic

A bucket holds up to `capacity` tokens and refills at `rate` per second. Each request costs one
token; an empty bucket means 429 with a `Retry-After`.

With `capacity = 100` and `rate = 20` per second: a tenant can burst 100 requests instantly, then
is held to 20 per second. A tenant that sends 25 per second steadily empties the bucket in 20
seconds (`100 / (25 - 20)`) and then runs at 20 per second. Choosing `capacity` is choosing how
much burst you tolerate; choosing `rate` is choosing steady-state fairness.

### Quotas are about money, not requests

Requests are a poor unit for AI: one request might cost 100 tokens and the next 100,000. Cap on
tokens, or on currency, per tenant per period. A tenant with a 1,000,000-token monthly allowance
and an average of 800 tokens per request gets about 1,250 requests — and if their average jumps to
8,000 tokens, they get 125. That is the point: the cap tracks cost.

Track usage where it is counted, which is the provider's `usage` object or your own tokenizer,
and record it atomically so concurrent requests cannot overshoot by more than one in-flight
request's worth.

### Limits belong at the gateway, not in every handler

Implement the limiter once, as middleware or a FastAPI dependency, and make the tenant id a
required argument to every handler. A limiter that must be remembered per endpoint will be
forgotten on the endpoint that matters.

## Code

An auth dependency that accepts either an API key or a JWT and returns a typed principal, plus a
Redis token-bucket limiter. Both run before the handler, so no endpoint can forget them.

```python
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, status
from redis.asyncio import Redis


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    subject: str
    scopes: frozenset[str]
    monthly_token_quota: int


async def principal_from_api_key(key: str, redis: Redis) -> Principal | None:
    """Look up by visible prefix, then compare hashes in constant time."""
    prefix = key[:12]
    record = await redis.hgetall(f"apikey:{prefix}")
    if not record:
        return None
    digest = hashlib.sha256(key.encode()).hexdigest()
    if not hmac.compare_digest(digest, record["sha256"]):
        return None
    return Principal(
        tenant_id=record["tenant"],
        subject=f"key:{prefix}",
        scopes=frozenset(record["scopes"].split(",")),
        monthly_token_quota=int(record["quota"]),
    )


def principal_from_jwt(token: str, public_key: str) -> Principal:
    try:
        claims = jwt.decode(token, public_key, algorithms=["RS256"],
                            options={"require": ["exp", "iat", "sub"]})
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    return Principal(
        tenant_id=claims["tenant"],
        subject=claims["sub"],
        scopes=frozenset(claims.get("scope", "").split()),
        monthly_token_quota=int(claims.get("quota", 0)),
    )


async def require_principal(
    authorization: str = Header(...),
    x_api_key: str | None = Header(default=None),
) -> Principal:
    redis: Redis = get_redis()          # module-level pool created at startup
    if x_api_key:
        if (p := await principal_from_api_key(x_api_key, redis)) is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad api key")
        return p
    if authorization.startswith("Bearer "):
        return principal_from_jwt(authorization[7:], PUBLIC_KEY)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing credentials")


async def check_token_bucket(redis: Redis, key: str, capacity: int, rate: float) -> None:
    """Fixed-window-free bucket: store tokens and last refill timestamp."""
    now = time.time()
    state = await redis.hgetall(f"bucket:{key}")
    tokens = float(state.get("tokens", capacity))
    last = float(state.get("ts", now))
    tokens = min(capacity, tokens + (now - last) * rate)
    if tokens < 1.0:
        retry_after = max(1, int((1.0 - tokens) / rate))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    pipe = redis.pipeline()
    pipe.hset(f"bucket:{key}", mapping={"tokens": tokens - 1.0, "ts": now})
    pipe.expire(f"bucket:{key}", 3600)
    await pipe.execute()
```

The quota check goes after the model call, when you know the token count. Doing it before the call
is a reservation; doing it after is an accounting record. Most services do both: check the
remaining allowance before, then record actual usage after.

```python
from __future__ import annotations

from datetime import UTC, datetime

from redis.asyncio import Redis


def period_key() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


async def check_quota(redis: Redis, tenant: str, quota: int) -> int:
    used = int(await redis.get(f"quota:{tenant}:{period_key()}") or 0)
    if used >= quota:
        raise PermissionError(f"monthly quota exhausted: {used}/{quota} tokens")
    return quota - used


async def record_usage(redis: Redis, tenant: str, tokens: int) -> int:
    key = f"quota:{tenant}:{period_key()}"
    total = await redis.incrby(key, tokens)   # atomic, so concurrent calls cannot lose usage
    await redis.expire(key, 60 * 60 * 24 * 40)
    return total
```

## Real use case

A startup launched a public API for document summarisation with API keys but no limits. Two weeks
later a customer pointed a script at it that retried aggressively on any error, including their own
429-less rate errors, and generated tens of thousands of calls overnight. Because there was no
per-tenant cap, the account consumed the shared provider quota, and every other customer got
provider errors for a morning.

The fix was three layers, each cheap. Per-key token buckets at 20 requests per second with a burst
of 100 stopped the hammering immediately and returned proper 429s with `Retry-After`, which also
taught the client's retry logic to back off. Monthly token quotas per tenant made spend predictable
and let them price plans honestly, since the quota was denominated in the thing that costs money.
And a usage dashboard per tenant meant the next runaway script was visible in hours, not after a
support escalation.

The mistake worth recording is the original design, not the incident: the key existed to identify
the tenant, and the team assumed identification implied control. It does not. Identification is
what makes control possible; the limiter is the control.

## Common mistakes

- **Mistake:** storing API keys in plaintext — **Instead:** store a prefix for lookup plus a SHA-256 hash, and show the key once.
- **Mistake:** decoding a JWT without verifying the signature — **Instead:** verify signature, `exp`, issuer and audience, and require the claims you depend on.
- **Mistake:** limiting per IP address only — **Instead:** limit per tenant; many tenants share IPs behind NAT, and one tenant can use many IPs.
- **Mistake:** counting requests instead of tokens for AI quotas — **Instead:** meter tokens or currency, since request cost varies by orders of magnitude.
- **Mistake:** checking the quota non-atomically with read-then-write — **Instead:** use `INCRBY` or a Lua script so concurrent requests cannot overshoot.

## Checkpoint

1. A tenant sends 30 requests per second against a bucket with capacity 50 and rate 10 per second. How long until the bucket empties, and what rate are they held to afterwards?
2. Add a `scopes` check to the chat endpoint that requires `chat:write`, and write a test proving a read-only key gets 403.
3. Implement the token bucket as a Redis Lua script that refills, checks and decrements in one atomic operation. Compare its behaviour under 100 concurrent requests with the read-modify-write version above.

**Self-check:** the bucket drains at `30 - 10 = 20` tokens per second from a capacity of 50, so it empties in 2.5 seconds; after that the tenant is held to 10 requests per second with 429s and `Retry-After`.

## Go deeper

- [API Security](07-api-security.md) — the broader hardening checklist this file is one part of.
- [Multi-Tenant Design](../21-system-design/05-multi-tenant.md) — quotas as an architectural constraint, not just a middleware.
- [Cost Monitoring](../18-mlops-llmops/06-cost-monitoring.md) — turning usage records into a dashboard.
- [Prompt Injection](../20-ai-security/01-prompt-injection.md) — the threat that survives correct authentication.

---
[← Previous: Job Queues](05-job-queues.md) · [Stage index](README.md) · [Next: API Security](07-api-security.md)
