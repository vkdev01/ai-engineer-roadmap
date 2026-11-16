# 7. API Security

> **Stage 17 · AI Backend** — the checklist that stops a working service from becoming an incident.
>
> **You'll be able to:** harden an AI endpoint against oversized input, secret leaks, hung requests and unsafe outbound fetches.
>
> **Prerequisites:** [Auth and Rate Limits](06-auth-rate-limits.md) · [Prompt Injection Defense](../08-prompt-and-context/06-injection-defense.md)

## Why this exists

AI backends have a security surface that ordinary CRUD APIs do not. Inputs are free-form text of
unbounded length, which means unbounded cost. Outputs are model-generated, which means
untrusted. And many AI features fetch URLs, call tools and reach out to third-party APIs, which
means your server becomes a proxy that an attacker can aim.

None of this needs exotic attacks. The realistic failures are a 40 MB request that costs real
money, an API key committed to a repo, a request that hangs forever holding a worker, and a
"summarise this URL" feature that will happily fetch your cloud metadata endpoint.

The mental model: **treat every input as hostile, every outbound call as privileged, and every
secret as already half-leaked.**

## The mental model

```text
   input                     your service                    outside
+-----------+   validate    +-------------+   allowlist    +----------+
| body size |---cap-------->|  handler    |---egress------>| provider |
| schema    |   parse       |  timeouts   |                 | tools    |
| prompt    |   sanitise    |  redaction  |                 | URLs     |
+-----------+               +-------------+                 +----------+
                                  |
                            secrets from env, never code
```

| Risk | Control | Where |
|---|---|---|
| Huge payload, huge bill | request size and token caps | middleware + schema |
| Secret in a repo | env/secret manager, scanning | deploy |
| Hung request, stuck worker | timeouts at every hop | client + server |
| SSRF via URL fetch | allowlist, no redirects, no private ranges | fetcher |
| Prompt injection | separation, allowlisted tools | orchestrator |
| PII in logs | redaction before logging | logger |

## Core concepts

### Cap the input, because cost is a function of size

Set an explicit maximum body size at the proxy (for example a few hundred KB), a maximum message
length in your Pydantic model, and a maximum `max_tokens` the client may request. Without these,
one request can cost a thousand times another, and your rate limiter — which counts requests —
will not notice.

Do the arithmetic once: if your per-request ceiling is 8,000 input tokens and 4,000 output tokens,
then a tenant at 20 requests per second can consume 240,000 tokens per second of your capacity. If
that number is alarming, either the token cap or the rate limit is wrong.

### Timeouts at every hop, and a deadline that propagates

A request path typically has three timeouts: client to your API, your API to the provider, and the
provider's own generation. Set all three, and make them decrease as you go inward so the outer one
does not fire first and leave work running.

```text
client timeout 30s  >  your handler deadline 25s  >  provider request timeout 20s
```

Without this, a provider stall holds a connection until the client gives up, and on an async
service you accumulate tasks that are doing nothing but waiting.

### Egress control is the one people skip

If your service can fetch a user-supplied URL, an attacker can ask it to fetch
`http://169.254.169.254/` (cloud metadata) or an internal admin host. That is server-side request
forgery. The controls are: an allowlist of hostnames, no automatic redirect following, resolution
to a public IP only, and a short timeout with a size cap on the response.

Allowlisting is stronger than blocklisting. "Only these twelve documentation domains" is auditable;
"not localhost and not 10.x" is a list you will get wrong.

### Secrets come from the environment, never from code

Provider keys, database URLs and JWT signing keys belong in a secret manager or environment
variables injected at deploy time. Add a pre-commit scanner so a key pasted into a test file is
caught before it reaches a remote. Rotate on any suspicion — a key that appeared in a chat message
or a log line is compromised.

### Output is untrusted input to the next system

Model output can contain markdown with links, HTML, SQL-ish strings or tool arguments. Treat it as
untrusted: escape before rendering in a browser, never `eval` it, validate before passing it to a
tool, and never let it select a database query directly. The same discipline as
[Agent Security](../11-agents/11-agent-security.md).

### Redact before you log

Logs are the most commonly over-shared artefact in an AI system, because prompts are the most
useful debugging information and also the most sensitive. Redact emails, phone numbers, tokens and
keys at the logger boundary, and keep full prompt logs only in a restricted store with a retention
limit.

## Code

A hardened outbound fetcher. It enforces an allowlist, resolves the host and rejects private
addresses, does not follow redirects, and caps both time and size — the four things that turn a
URL-fetch feature from a hole into a tool.

```python
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

ALLOWED_HOSTS = {"docs.python.org", "fastapi.tiangolo.com", "www.python.org"}
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0)


def is_public_host(host: str) -> bool:
    """Resolve and reject anything that is not a public IP address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


async def safe_fetch(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"https"}:
        raise ValueError("only https is allowed")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"host not allowed: {parsed.hostname}")
    if not is_public_host(parsed.hostname):
        raise ValueError("host resolves to a non-public address")

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise ValueError(f"upstream status {resp.status_code}")
            total = 0
            chunks: list[bytes] = []
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("response too large")
                chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")
```

A redacting logger and a settings object, so secrets never appear in code or in logs. The
`SecretStr` type keeps the value out of tracebacks and `repr` output.

```python
from __future__ import annotations

import logging
import re

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    provider_api_key: SecretStr = Field(..., alias="PROVIDER_API_KEY")
    jwt_public_key: str = Field(..., alias="JWT_PUBLIC_KEY")
    max_input_chars: int = 8_000
    handler_deadline_s: float = 25.0
    model_config = {"env_file": ".env", "extra": "ignore"}


REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"), "[phone]"),
    (re.compile(r"\b(sk|pk)_[A-Za-z0-9]{16,}\b"), "[key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "[jwt]"),
]


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = tuple(redact(str(a)) for a in record.args)
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
```

## Real use case

A B2B product shipped a "summarise a link" feature so sales users could paste a customer's
public page and get a brief. The implementation fetched whatever URL it was given with a plain
HTTP client and passed the text to the model. During a security review, an engineer demonstrated
that pointing it at an internal wiki URL returned the wiki page summarised in the response — the
server had network access the browser did not, and nothing checked where it was going.

The fix was the fetcher above: an allowlist of documentation and news domains the product actually
needed, HTTPS only, no redirects, private-address rejection after DNS resolution, and a 2 MB cap
with a 10-second read timeout. Legitimate use barely changed, because the real traffic was
concentrated on a handful of domains.

The same review found two smaller issues worth recording. Their logs contained full prompt text
including customer emails, so they added the redaction filter and cut retention to thirty days.
And their provider key was in a `.env` file that had been copied into a Docker image in a
debugging session, so they rotated the key, moved to a secret manager, and added a pre-commit
scanner. None of these were exotic; all three are on the checklist precisely because they are
ordinary.

## Common mistakes

- **Mistake:** trusting a client-supplied `max_tokens` — **Instead:** clamp it server-side to your own ceiling.
- **Mistake:** fetching user-supplied URLs with a general HTTP client — **Instead:** allowlist, no redirects, reject private IPs, cap size and time.
- **Mistake:** one timeout, at the client — **Instead:** decreasing timeouts at each hop, with the innermost shortest.
- **Mistake:** logging raw prompts because they help debugging — **Instead:** redact at the logger and keep full text only in a restricted store.
- **Mistake:** rendering model output as HTML — **Instead:** escape it, and validate any structured output before acting on it.

## Checkpoint

1. Your per-request ceiling is 8,000 input plus 4,000 output tokens and the limit is 10 requests per second per tenant. What is the maximum token rate one tenant can consume, and what cap would you set to keep it under a target you choose?
2. Add a middleware that rejects request bodies over 256 KB with a 413, and test it with a 1 MB payload.
3. Extend `safe_fetch` to log every rejected host with the reason, then attempt a fetch to an internal address and confirm it is refused and recorded.

**Self-check:** `12,000 tokens x 10 requests/s = 120,000 tokens per second`; to hold a tenant to, say, 10,000 tokens per second, cap output at 1,000 tokens or reduce the request rate to about 0.8 per second.

## Go deeper

- [Auth and Rate Limits](06-auth-rate-limits.md) — the identity layer this file assumes.
- [Prompt Injection Defense](../08-prompt-and-context/06-injection-defense.md) — what to do when hostile text arrives inside legitimate input.
- [PII Redaction](../20-ai-security/04-pii-redaction.md) — redaction as a product requirement, not just a logging habit.
- [Sandboxing](../20-ai-security/05-sandboxing.md) — the next step when tools need to run code.

---
[← Previous: Auth and Rate Limits](06-auth-rate-limits.md) · [Stage index](README.md) · [Next: AI App Architecture](08-ai-app-architecture.md)
