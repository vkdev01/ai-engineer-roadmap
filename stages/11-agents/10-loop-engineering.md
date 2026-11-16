# 10. Loop Engineering

> **Stage 11 · AI Agents** — making the loop terminate: step budgets, no-progress detection, retry semantics and the small proof that your agent cannot run forever.
>
> **You'll be able to:** instrument a loop so every run ends for a stated reason, and prove that no path through it can spin.
>
> **Prerequisites:** [Agent Architecture](02-agent-architecture.md) · [Memory and State](05-memory-and-state.md)

## Why this exists

A loop that cannot terminate is a production incident waiting for load. The usual version is quiet: an agent that retries a failing search twelve times, spends a large bill, and then returns "I could not complete the task" — having learned nothing. The louder version is a `while True` with no counter that a customer triggers at 2am.

Loop engineering is the part of agent work that looks like ordinary systems engineering: counters, deadlines, retry budgets, idempotency and a reason for every exit. The model is the only unbounded part of the system, so the loop around it must be bounded by construction.

The mental model: **every iteration must either add information or be refused.**

## The mental model

```text
        ┌──────────────────────────────────────────────┐
        │ iteration guard                              │
        │  steps  < max_steps        (count)           │
        │  tools  < max_tools        (count)           │
        │  now    < deadline         (clock)           │
        │  cost   < max_cost         (tokens × price)  │
        │  progress_signal == NEW    (information)     │
        └──────────────────────────────────────────────┘
                    │ all pass                │ any fails
                    ▼                         ▼
              run one step            exit with a named reason
```

Five independent bounds. Any one alone is not enough: a step cap still lets one step hang, a clock does not stop twenty fast useless calls.

## Core concepts

### Four counters, one deadline

Track model calls, tool calls, total tokens and wall-clock. Set the deadline when the run starts and compare against it at the top of every iteration — never rely on a single outer request timeout, because the agent may be resuming across several requests. Each bound exists to catch a different pathology: steps catch thrash, tool calls catch retry storms, tokens catch huge context growth, the clock catches slow dependencies.

### Progress is a value, not a feeling

Define a progress signal you can compute: the set of new tool results this step, the reduction in open sub-goals, the change in a hash of the working document. If two consecutive iterations produce the same signal, you are not making progress. This is how you catch the polite failure — an agent that keeps doing something plausible and never gets anywhere.

### Retry with a changing strategy

A retry that repeats the same call with the same arguments is not a retry, it is a loop. Attach a strategy per attempt: broaden a query, drop a filter, switch tool, then give up. Cap retries per tool (two is a sane default) and record attempts in the ledger so the counter survives a resume.

### Make the exit reason part of the answer

Never return an empty string on exhaustion. Return a partial result plus the reason: which bound fired, what was completed, what was not. The caller can then decide to raise a budget, ask a human, or fall back. An unexplained failure costs more support time than a slow success.

### Termination argument

You do not need a formal proof, but you do need an argument you can state in one sentence: "each iteration increments a counter that is compared against a fixed cap, and every code path either returns or continues to the top of the loop." Write that sentence in a comment. If you cannot write it, the loop has a hole.

## Code

A guarded loop with a progress signal, per-tool retry caps and named exits. The important lines are the guard block and the `signature` comparison.

```python
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

Exit = Literal["answered", "step_budget", "tool_budget", "deadline", "cost_budget",
               "no_progress", "tool_retry_exhausted"]

@dataclass
class Guard:
    max_steps: int = 10
    max_tool_calls: int = 20
    max_tokens: int = 60_000
    max_seconds: float = 90.0
    max_retries_per_tool: int = 2
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    retries: dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)

    def check(self) -> Exit | None:
        # Evaluated at the TOP of each iteration: no path skips it.
        if self.steps >= self.max_steps: return "step_budget"
        if self.tool_calls >= self.max_tool_calls: return "tool_budget"
        if self.tokens >= self.max_tokens: return "cost_budget"
        if time.monotonic() - self.started >= self.max_seconds: return "deadline"
        return None

def signature(tool: str, args: dict[str, Any], result: str) -> str:
    # Identity of "what happened", not "what was asked": catches rephrased repeats.
    payload = f"{tool}|{sorted(args.items())}|{result[:200]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

async def guarded_run(task: str, model: Any, tools: dict[str, Any],
                      guard: Guard) -> tuple[Exit, str]:
    last_sig: str | None = None
    log: list[str] = []

    while True:
        stop = guard.check()
        if stop:
            return stop, f"{stop}: completed {guard.steps} steps. Trace: {'; '.join(log[-3:])}"

        guard.steps += 1
        reply = await model.complete([{"role": "user", "content": task}], list(tools))
        guard.tokens += reply.get("usage", {}).get("total_tokens", 0)

        if "tool" not in reply:
            return "answered", reply["text"]

        name, args = reply["tool"], reply.get("args", {})
        tool = tools.get(name)
        if tool is None:
            log.append(f"unknown tool {name}")
            continue

        guard.tool_calls += 1
        try:
            result = await tool["run"](args)
        except Exception as exc:
            guard.retries[name] = guard.retries.get(name, 0) + 1
            if guard.retries[name] > guard.max_retries_per_tool:
                return "tool_retry_exhausted", f"{name} failed {guard.retries[name]} times: {exc}"
            log.append(f"{name} error: {exc}")
            continue

        sig = signature(name, args, result)
        if sig == last_sig:                        # same call, same output: no progress
            return "no_progress", f"Repeated identical outcome at step {guard.steps}."
        last_sig = sig
        log.append(f"{name} ok")
```

The same guards belong on the client side of an agent server: a client that reconnects must not be able to restart the loop with a fresh budget. Pass a run id and let the server own the counters.

```ts
export type RunLimits = { maxSteps: number; maxSeconds: number };

export function clientGuard(limits: RunLimits) {
  const started = Date.now();
  let steps = 0;
  return {
    // Server-side counters are authoritative; this only stops runaway UI retries.
    allow(): boolean {
      steps += 1;
      return steps <= limits.maxSteps && Date.now() - started < limits.maxSeconds * 1000;
    },
    used: (): number => steps,
  };
}
```

## Real use case

A monitoring agent investigated alert spikes: pull metrics, check recent deploys, read the runbook, summarise. In the first version, a metrics query that timed out caused a retry, which timed out, which caused another — eleven attempts in under a minute, then a silent empty answer.

The rebuild added three things: a retry cap of two per tool with a strategy change (narrow the time window on attempt two), a deadline set at the start of the run and checked each iteration, and a progress signature. When the metrics API was down the agent now exits with `tool_retry_exhausted` and reports which query failed, so on-call gets a diagnosis instead of an empty summary.

## Common mistakes

- **Mistake:** one `while True` with a `break` buried in the happy path — **Instead:** a guard checked at the top of every iteration.
- **Mistake:** a single timeout on the HTTP request — **Instead:** counters and a deadline owned by the loop, which survive resumes.
- **Mistake:** retrying the identical call — **Instead:** change the strategy per attempt and cap retries per tool.
- **Mistake:** detecting no-progress by comparing raw arguments only — **Instead:** compare tool, arguments and a result signature.
- **Mistake:** returning an empty string on exhaustion — **Instead:** return a partial result with the exit reason and what completed.
- **Mistake:** budgets reset when a run resumes — **Instead:** store counters in the checkpoint.

## Checkpoint

1. Add a `max_cost` guard that converts token usage to rupees using your provider's current price sheet, and stop with `cost_budget`.
2. Make the no-progress rule tolerate one repeat (two identical signatures) before stopping, and note what new failure it allows.
3. Write the termination sentence for your loop as a comment above it, then try to break your own argument by finding a path that skips the guard.

**Self-check:** Exercise 1 — track cumulative tokens in the guard, multiply by the input and output price per token, and include the result in the `check` method before the clock.

## Go deeper

- [Harness Engineering](01-harness-engineering.md) — where the budgets live in the scaffold.
- [Reflection and Self-Critique](09-reflection.md) — a critique loop that also needs bounds.
- [Fault Tolerance](../21-system-design/07-fault-tolerance.md) — retries and deadlines at system level.
- [Cost Tracking](../07-llm-apis/08-cost-tracking.md) — measuring the number your budget guards.

---

[← Previous: Reflection and Self-Critique](09-reflection.md) · [Stage index](README.md) · [Next: Agent Security](11-agent-security.md)
