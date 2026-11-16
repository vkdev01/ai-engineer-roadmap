# 7. HTTP and REST APIs

> **Stage 01 · Programming Basics** — the transport layer under every model call you will ever make.
>
> **You'll be able to:** read an API's documentation, send a correct request with headers and a JSON body, and handle pagination, retries and error statuses properly.
>
> **Prerequisites:** [The Linux Command Line](06-linux-cli.md) · [JSON and YAML](08-json-yaml.md)

## Why this exists

An LLM API call is an HTTP POST with a JSON body and a bearer token. So is an embedding request, a vector database query, and a webhook your service sends when a document finishes processing. When a call fails at 3 a.m., the debugging question is always the same: what status code came back, and what did the body say?

Understanding HTTP means you stop treating the SDK as magic. You can reproduce any SDK call with `curl`, you know why 429 is different from 500, and you know what `Retry-After` is telling you.

The mental model: **a request is method + URL + headers + body; a response is status + headers + body.**

## The mental model

```text
POST /v1/messages HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>      <- who you are
Content-Type: application/json     <- what the body is
Idempotency-Key: 8f2c...           <- safe to retry without double-charging

{"model": "...", "messages": [...]}
        |
        v
HTTP/1.1 429 Too Many Requests
Retry-After: 20                    <- when to come back
{"error": {"type": "rate_limit_error", "message": "..."}}
```

| Status | Meaning | What you do |
|---|---|---|
| 2xx | worked | parse the body |
| 400 | your request is malformed | fix the request; do not retry |
| 401 / 403 | bad or missing key, no permission | fix credentials; do not retry |
| 404 | wrong URL or id | check the path |
| 409 | conflict | resolve state, maybe retry |
| 429 | rate limited | wait `Retry-After`, then retry with backoff |
| 5xx | server-side | retry with exponential backoff and jitter |

## Core concepts

### Methods are verbs about resources

`GET` reads and must not change state. `POST` creates or triggers. `PUT` replaces. `PATCH` partially updates. `DELETE` removes. `GET` and `PUT` and `DELETE` should be idempotent — repeating them gives the same result — which is what makes retrying them safe.

### Status codes are the API's error vocabulary

Read the status before the body, always. Retrying a 400 wastes time and quota; not retrying a 503 loses work. Log status, request id, and a truncated body — three fields that answer most incidents.

### Headers carry everything the URL cannot

`Authorization` authenticates. `Content-Type` declares the body format. `Accept` asks for one. `Idempotency-Key` lets a POST be retried without duplicating a charge or a job. `X-Request-Id` (or whatever the provider calls it) is what support asks for first.

### Pagination is a loop you must write correctly

Three common shapes: `?page=2&per_page=100`, a `next_cursor` in the response body, and a `Link` header. Always loop until the cursor is absent or a page returns fewer than `per_page` items, and always cap the loop so a buggy server cannot spin forever.

### Retries need backoff, jitter and a ceiling

Retry only 429 and 5xx. Wait `Retry-After` if present, otherwise exponential: 1s, 2s, 4s, 8s, with random jitter so a thousand clients do not retry in lockstep. Stop after a handful of attempts and surface the last error.

### Timeouts are mandatory

Every client call needs a timeout; a hung socket with no timeout is an outage waiting for a quiet weekend. Set a connect timeout and a read timeout, and make the read timeout generous enough for a slow generation.

## Code

A typed client with retries, timeouts and status-aware error handling. It uses `httpx` directly so nothing is hidden.

```python
from __future__ import annotations

import asyncio
import random

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, request_id: str | None) -> None:
        super().__init__(f"HTTP {status}: {body[:300]} (request_id={request_id})")
        self.status = status
        self.body = body
        self.request_id = request_id


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json_body: dict[str, object] | None = None,
    attempts: int = 4,
) -> dict[str, object]:
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            resp = await client.request(method, url, json=json_body)
        except httpx.TimeoutException:
            if attempt == attempts:
                raise
            await asyncio.sleep(delay + random.random())
            delay *= 2
            continue

        if resp.status_code < 400:
            return resp.json()

        # 4xx (except 429) will not get better by asking again.
        if resp.status_code != 429 and resp.status_code < 500:
            raise ApiError(resp.status_code, resp.text, resp.headers.get("x-request-id"))

        if attempt == attempts:
            raise ApiError(resp.status_code, resp.text, resp.headers.get("x-request-id"))

        # Honour Retry-After when the server tells us how long to wait.
        retry_after = resp.headers.get("retry-after")
        wait = float(retry_after) if retry_after else delay
        await asyncio.sleep(wait + random.random())   # jitter avoids a thundering herd
        delay *= 2
    raise AssertionError("unreachable")
```

Paging through every result, with a hard cap:

```python
from __future__ import annotations


async def all_pages(client, base_url: str, max_pages: int = 100) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        page = await request_json(client, "GET", base_url + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
        items = page.get("data", [])
        out.extend(items)                 # type: ignore[arg-type]
        cursor = page.get("next_cursor")  # type: ignore[assignment]
        if not cursor or len(items) == 0:  # type: ignore[arg-type]
            break
    return out
```

The same call from TypeScript, with `fetch` — useful when your frontend or a Node script needs to hit your own endpoint:

```ts
export async function postJson<T>(url: string, body: unknown, token: string): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status} ${res.headers.get("x-request-id") ?? ""}: ${text.slice(0, 300)}`);
  }
  return (await res.json()) as T;
}
```

## Real use case

A batch summarisation service called a provider's API for 5,000 documents each night. It worked for weeks, then started dying partway through with an unhelpful `KeyError` on `response["choices"]` — because the code parsed the body without checking the status, and on a 429 the body has no `choices`.

The rewrite checked the status first, treated 429 as "wait `Retry-After` and retry", treated 400 as "this document is malformed, quarantine it", and logged the provider's request id on every failure. Nights went from "crashed at some point" to "4,912 summarised, 88 rate-limited then succeeded on retry, 2 quarantined with a reason". When a genuine provider incident happened a month later, the request ids made the support ticket a two-line email.

The design choice that mattered was separating retryable from non-retryable statuses — before, every error looked the same.

## Common mistakes

- **Mistake:** parsing `response.json()` before checking `status_code`. — **Instead:** check the status, then parse, then read fields defensively.
- **Mistake:** retrying a 400 or a 401. — **Instead:** fix the request or the key; retries just burn quota.
- **Mistake:** retrying immediately with no wait. — **Instead:** exponential backoff with jitter, and honour `Retry-After`.
- **Mistake:** no timeout on the client. — **Instead:** always set connect and read timeouts.
- **Mistake:** paging with `while True` and no cursor check. — **Instead:** stop on missing cursor or empty page, and cap the page count.

## Checkpoint

1. Reproduce an SDK call with `curl -i` and read the headers; identify the request id and rate-limit headers.
2. Add `Idempotency-Key` to the POST in `request_json` and explain what changes on a retry.
3. Write `all_pages` for a cursor API, then make it handle a `Link: <...>; rel="next"` header instead.

**Self-check:** an idempotency key lets the server recognise a repeated POST, so a retry after a timeout returns the original result instead of creating a second job.

## Go deeper

- [JSON and YAML](08-json-yaml.md) — the payload formats you just sent and parsed.
- [Retries and Rate Limits](../07-llm-apis/07-retries-rate-limits.md) — the same patterns tuned for LLM providers.
- [FastAPI and Flask](../17-ai-backend/01-fastapi-flask.md) — the other side: serving an API of your own.

---
[← Previous: The Linux Command Line](06-linux-cli.md) · [Stage index](README.md) · [Next: JSON and YAML](08-json-yaml.md)
