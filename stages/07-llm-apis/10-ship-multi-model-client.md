# 10. Ship: A Multi-Model Client

> **Stage 07 · LLM APIs** — one interface over several providers, with typed responses, retries, a ledger and fallback.
>
> **You'll be able to:** ship a client library your whole codebase calls, and prove it survives a provider failure.
>
> **Prerequisites:** [Routing and Fallbacks](09-routing-fallbacks.md) · [Token Accounting and Cost Tracking](08-cost-tracking.md)

## Why this exists

By now you have written the same call eight times: build messages, call, parse, retry, log tokens, fall back. Left as copies, those eight versions drift — one retries, one does not; one logs cost, one forgets. The library is not bureaucracy, it is the place where the hard-won details are written once and enforced for every caller.

The design constraint that makes it work: **the interface belongs to you, not to the provider.** Callers ask for a completion with a task name and requirements. They never see a provider SDK type.

The mental model: **one `complete()` function, a registry of adapters, and three cross-cutting layers — retry, ledger, routing — wrapped around every call.**

## The mental model

```text
caller: client.complete(task="summarise", prompt=..., tenant=...)
   |
   v
[ router: pick candidates from policy ]
   |
   v
[ retry: per candidate, backoff + jitter ]
   |
   v
[ adapter: openai | anthropic | hosted ]  --> SDK
   |
   v
[ ledger: usage -> cost row ]
   |
   v
Completion(text, model, provider, cost, attempts)
```

| Layer | Owns | Test |
|---|---|---|
| Adapter | provider shapes | same prompt, same text |
| Retry | transient failure | forced 429 recovers |
| Ledger | cost attribution | one row per attempt |
| Router | preference order | forced failure falls back |

## Core concepts

### Define the interface before the adapters

Write the dataclasses first: `CompletionRequest`, `Completion`, `Usage`. Every adapter converts *into* these and never the reverse. If an adapter needs a field the interface lacks, that is a design conversation, not a special case in the caller.

### Adapters are pure translators

An adapter takes `(model, request)` and returns a `Completion` or raises. No retries, no logging, no routing. This keeps them short and makes them trivially testable against recorded fixtures.

### Cross-cutting layers compose, in a fixed order

Retry wraps the adapter; the router wraps retry; the ledger observes both. Getting the order wrong is subtle: if you retry *outside* the router, a failure on candidate B retries candidate A. Retry inside, routing outside.

### Make failure observable without leaking it

The returned `Completion` carries `provider`, `model` and `attempts`. The caller can log them, the UI can ignore them. When all candidates fail, raise a typed error carrying the failure list — that object is what your alert fires on.

### Ship it with tests that do not need the network

Inject the adapter registry. Tests supply a fake adapter that returns canned text, sleeps, raises, or times out. This is how you prove retry and fallback work without spending money.

## Code

The core of the library: interface types, the registry, and the composed call path.

```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Protocol

Task = Literal["classify", "summarise", "reason"]


@dataclass(frozen=True)
class CompletionRequest:
    task: Task
    prompt: str
    system: str = ""
    max_output_tokens: int = 800
    tenant_id: str = "anonymous"


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    provider: str
    usage: Usage
    attempts: list[str] = field(default_factory=list)  # failed candidates


class Adapter(Protocol):
    name: str

    async def complete(self, model: str, req: CompletionRequest) -> Completion: ...


class AllProvidersFailed(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures) or "no candidates")
        self.failures = failures


# --- retry layer: inside the router, one candidate at a time ---------------
async def _retry(coro_factory, *, attempts: int = 3, base: float = 0.4,
                 deadline_s: float = 45.0):
    import random

    started = time.monotonic()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:                 # adapters raise; policy decides
            last = exc
            wait = random.uniform(0, base * 2**attempt)   # full jitter
            if time.monotonic() - started + wait > deadline_s:
                break
            await asyncio.sleep(wait)
    raise last if last else RuntimeError("retry failed")


# --- the client ------------------------------------------------------------
class LLMClient:
    def __init__(
        self,
        adapters: dict[str, Adapter],
        policies: dict[Task, list[tuple[str, str]]],  # (provider, model) order
        ledger,
        min_chars: dict[Task, int] | None = None,
    ) -> None:
        self._adapters = adapters
        self._policies = policies
        self._ledger = ledger
        self._min_chars = min_chars or {}

    async def complete(self, req: CompletionRequest) -> Completion:
        failures: list[str] = []
        floor = self._min_chars.get(req.task, 1)

        for provider, model in self._policies[req.task]:
            adapter = self._adapters.get(provider)
            if adapter is None:
                failures.append(f"{provider}: no adapter registered")
                continue
            try:
                comp = await _retry(
                    lambda: adapter.complete(model, req), attempts=3
                )
            except Exception as exc:
                failures.append(f"{provider}/{model}: {exc}")
                continue

            # quality gate: a bad answer is a reason to try the next candidate
            if len(comp.text.strip()) < floor:
                failures.append(f"{provider}/{model}: below quality floor")
                continue

            await self._ledger.record(comp, req, failures)
            return Completion(
                text=comp.text, model=comp.model, provider=comp.provider,
                usage=comp.usage, attempts=failures,
            )

        raise AllProvidersFailed(failures)
```

A test that proves the fallback without touching the network — this is the part that makes the library trustworthy.

```python
import pytest
from decimal import Decimal


class FakeAdapter:
    def __init__(self, name: str, *, fail: bool = False, text: str = "ok"):
        self.name, self._fail, self._text = name, fail, text
        self.calls = 0

    async def complete(self, model: str, req) -> Completion:
        self.calls += 1
        if self._fail:
            raise RuntimeError("provider down")
        return Completion(
            text=self._text, model=model, provider=self.name,
            usage=Usage(10, 5, Decimal("0.00001")),
        )


class MemoryLedger:
    def __init__(self) -> None:
        self.rows: list[tuple[str, list[str]]] = []

    async def record(self, comp, req, failures) -> None:
        self.rows.append((req.tenant_id, failures))


@pytest.mark.asyncio
async def test_falls_back_when_primary_is_down() -> None:
    ledger = MemoryLedger()
    client = LLMClient(
        adapters={
            "primary": FakeAdapter("primary", fail=True),
            "backup": FakeAdapter("backup", text="from backup"),
        },
        policies={"summarise": [("primary", "m1"), ("backup", "m2")]},
        ledger=ledger,
    )
    out = await client.complete(
        CompletionRequest(task="summarise", prompt="hello", tenant_id="t1")
    )
    assert out.provider == "backup"
    assert out.text == "from backup"
    assert ledger.rows[0][0] == "t1"        # ledger saw the tenant
```

The TypeScript surface is a thin wrapper over the same idea, for Node services that need the identical behaviour.

```ts
export type CompletionRequest = {
  task: "classify" | "summarise" | "reason";
  prompt: string;
  system?: string;
  tenantId: string;
};

export type Completion = {
  text: string;
  model: string;
  provider: string;
  inputTokens: number;
  outputTokens: number;
  attempts: string[];
};

export interface Adapter {
  readonly name: string;
  complete(model: string, req: CompletionRequest): Promise<Completion>;
}

export async function complete(
  req: CompletionRequest,
  adapters: Record<string, Adapter>,
  policy: { provider: string; model: string }[],
): Promise<Completion> {
  const failures: string[] = [];
  for (const { provider, model } of policy) {
    const adapter = adapters[provider];
    if (!adapter) {
      failures.push(`${provider}: no adapter`);
      continue;
    }
    try {
      const c = await adapter.complete(model, req);
      if (!c.text.trim()) throw new Error("empty answer");
      return { ...c, attempts: failures };
    } catch (err) {
      failures.push(`${provider}/${model}: ${String(err)}`);
    }
  }
  throw new Error(`all providers failed: ${failures.join("; ")}`);
}
```

## Real use case

An engineering team shipped an internal "ask the docs" assistant built directly on one SDK, in eleven handlers. When the provider had a regional incident, every handler failed at once and the on-call engineer spent the first twenty minutes grepping for `client.` to find them all.

They extracted this library in a day: one `LLMClient`, three adapters, policies in a YAML config, a ledger writing to Postgres. The next incident produced degraded answers and a fallback-rate spike on a dashboard, and the on-call's action was to check one config file. The failure mode before was not the outage — outages are normal — it was that the blast radius of a provider problem was the entire codebase.

## Common mistakes

- **Mistake:** letting adapters return SDK objects. — **Instead:** convert into your own `Completion` at the adapter boundary.
- **Mistake:** putting retries outside the router. — **Instead:** retry each candidate, then route; otherwise a retry can jump providers unpredictably.
- **Mistake:** testing only against live APIs. — **Instead:** inject adapters and use fakes so retry and fallback are testable offline.
- **Mistake:** swallowing the failure list. — **Instead:** return it on success and attach it to the typed error on total failure.
- **Mistake:** hardcoding provider names in business logic. — **Instead:** keep policies in config so a model change is a deploy, not a refactor.

## Checkpoint

1. Add a third adapter for an OpenAI-compatible hosted model and route `classify` to it first.
2. Write tests for timeout, empty answer and total failure — three tests, no network.
3. Add a `max_cost_usd` per request and reject a candidate whose estimated cost would exceed it, recording the rejection in `attempts`.

**Self-check:** the client walks the task's policy list in order, retries each candidate internally, applies a quality floor before accepting, writes a ledger row, and raises `AllProvidersFailed` with every failure recorded.

## Go deeper

- [Prompt Templates with Variables and Validation](../08-prompt-and-context/03-templates.md) — the next layer up: what you send through this client.
- [Ship: A Prompt Test Suite](../08-prompt-and-context/09-ship-prompt-test-suite.md) — testing the prompts this client carries.
- [Tracing](../18-mlops-llmops/05-tracing.md) — instrumenting `attempts` and latency in production.
- [Scalable AI Apps](../21-system-design/01-scalable-ai-apps.md) — where this client sits in a larger architecture.

---
[← Previous: Routing and Fallbacks](09-routing-fallbacks.md) · [Stage index](README.md) · [Next: Stage 08 — Prompt and Context](../08-prompt-and-context/README.md)
