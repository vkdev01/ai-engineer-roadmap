# 11. The Nine-Layer Agentic Stack

> **Stage 21 · System Design** — the nine layers a production agentic application actually has, what each one owns, the incident it prevents, and the metric that tells you it is working.
>
> **You'll be able to:** walk a design review layer by layer, say what happens when each one is down, price a workflow instead of a model call, and build a miniature of the whole stack on one laptop.
>
> **Prerequisites:** [Scalable AI Apps](01-scalable-ai-apps.md) · [Agent Design](03-agent-design.md) · [Fault Tolerance](07-fault-tolerance.md)

## Why this exists

Most architecture diagrams for an AI product are three boxes: user, LLM, answer. Every one of them is a diagram of a demo. The system that survives a Monday has a gateway that knows who is calling and what they may spend, an orchestrator that can pause and resume, a model tier it can fall back from, tools with approval gates, memory that is not just the chat window, a queue so a nine-minute job does not hold an HTTP connection, a trust layer wrapped around all of it, and a deploy pipeline that refuses to ship a regression.

That is nine layers, and the model is one of them. The framing comes from Gaurav Pandey's sketch of a large agentic app, which is worth reading for the ordering alone: channels, gateway, orchestration, models, tools, knowledge and memory, async backbone, trust layer, CI/CD. The comment thread on it added the four sharpest details, and they are in this lesson because they are the ones that bite in production:

- **Per-workflow token and latency caps**, so a stuck multi-agent loop trips a breaker before the invoice does.
- **Cost per workflow, not per model.** A cheap model in a loop costs more than an expensive model that finishes in one pass, so the escalation rate is the number to watch.
- **Side-effect safety.** Event delivery is at-least-once, so a resumed checkpoint replays a step and issues the refund twice unless every write carries an idempotency key and multi-step actions have a compensation path.
- **Version pinning for in-flight work.** A task that started on config version N can resume after a rollout to N+1; without pinning, the checkpoint schema is the thing that breaks.

The reason to learn the nine layers as a checklist is that design reviews run in that order. Someone asks "what happens when the model provider rate-limits you", then "what happens when the supervisor picks the wrong specialist", then "who can stop this at 2am". If your design has an answer for each box and each seam, the review is a conversation. If it has a diagram of three boxes, the review is an incident report written early.

## Core concepts

### The nine layers at a glance

| # | Layer | Owns | Prevents | Metric |
|---|---|---|---|---|
| 1 | Channels | Web, mobile, Teams, Slack, API | Five clients each with their own half-implemented auth | Requests per channel, error rate per channel |
| 2 | AI gateway | Auth, tenant, rate limits, quotas, caching, cost, model routing | One tenant spending everyone's budget | Cost per workflow, p95 gateway latency |
| 3 | Orchestration | Supervisor, specialists, state, checkpoints | A 30-step task that cannot resume | Steps per workflow, resume rate |
| 4 | Models | Tiers, fallback, escalation | Paying frontier prices for classification | Escalation rate, cost per 1k tokens |
| 5 | Tools | MCP servers, schemas, approvals, idempotency | A duplicate refund | Writes per workflow, approval latency |
| 6 | Knowledge and memory | RAG, GraphRAG, session state, long-term memory | An agent that forgets the ticket it is working on | Retrieval recall, memory hit rate |
| 7 | Async backbone | Queues, workers, retries, DLQ | A timeout that loses nine minutes of work | Queue depth, job age, retry rate |
| 8 | Trust layer | Guardrails, fallbacks, kill switch, evals, traces, audit, identity | Nobody can explain or stop what happened | Blocked inputs, trace coverage, kill-switch latency |
| 9 | CI/CD | Eval gate, canary, rollback, versioned prompts | Shipping a prompt change that halves quality | Gate pass rate, rollback count |

Two structural facts make this list more than a table.

**The trust layer is not a hop.** Layers 1-7 are in the request path; layer 8 wraps every one of them. Guardrails run on input and output, the kill switch is read on every step, the audit log records every action with the identity that took it, and traces span the whole path. If you draw the trust layer as a box between tools and the queue, you will forget to put a guardrail on the channel that a colleague added last week.

**Versioning cuts across all nine.** Prompts, agent configs, tool schemas, checkpoint schemas and eval sets are all versioned artefacts deployed like code. The comment that named this is the one that saves you at 3am: pin the version on a running task, or keep checkpoint schemas backward compatible, because rollouts and long tasks overlap.

### 1. Channels

**Owns:** the entry points — web, mobile, Teams, Slack, public API, webhooks, scheduled jobs.

**What to get right:** every channel goes through the same gateway. The common failure is a Slack bot that talks to the orchestrator directly because it was built by a different team, and therefore has no tenant, no quota and no guardrail. Channel adapters should be thin: authenticate the surface, normalise the payload, attach a request id, call the gateway.

**Design question:** which channel is allowed to do which actions? A read-only query from Slack is not the same risk as an approval click from the same user in the web app.

### 2. AI gateway

**Owns:** identity, tenant, rate limits, quotas, caching, cost attribution, model routing.

**Example:** a token bucket per tenant, a quota check against a monthly token budget, an exact and semantic cache keyed by (tenant, policy version, normalised prompt), and a routing decision that returns a tier name rather than a model id.

```python
def cache_key(tenant: str, prompt: str, policy_version: str) -> str:
    # Policy-versioned: change the prompt and the cache stops serving yesterday's answer.
    blob = f"{tenant}|{policy_version}|{re.sub(r'\\s+', ' ', prompt.lower().strip())}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
```

**The incident it prevents:** one customer's batch job consuming the shared rate limit at 09:00 and every interactive request failing with 429 for twenty minutes.

**Metric:** cost per workflow, not per model call. The gateway is the only place that can see a workflow as a unit.

### 3. Orchestration

**Owns:** the plan, the specialist agents, the state machine, the checkpoints.

**What to get right:** a supervisor that hands work to named specialists, a state machine with explicit states, and a checkpoint after every step so a pause or a crash resumes rather than restarts. Record **why** a specialist was chosen, not just which one: the most common production question is "why did it route this billing question to the research agent", and without the reason in the trace, the answer is a guess.

**The incident it prevents:** a nine-minute task that dies at minute eight and starts over, paying twice and possibly writing twice.

**Metric:** steps per workflow and resume rate. A rising step count with flat quality is a supervisor that has started thrashing.

### 4. Models

**Owns:** the tiers, the prices, the fallback chain, the escalation rule.

**What to get right:** small model first, escalate on a signal you can name — task class, prompt length, retrieval confidence, or a failed validation — not on vibes. Keep one client in front of every provider so retries, timeouts, fallbacks and cost accounting exist once.

```python
def route(prompt: str, task: str) -> Tier:
    if task in {"code_review", "legal"} or len(prompt) > 1500:
        return TIERS["frontier"]
    if task in {"summarise", "classify"}:
        return TIERS["small"]
    return TIERS["mid"]
```

**The incident it prevents:** the frontier model's provider rate-limiting you at peak and the whole product returning errors, because nobody wrote the fallback.

**Metric:** escalation rate per workflow, and the cost of a workflow that escalates. This is where the arithmetic below comes from.

### 5. Tools

**Owns:** the tool registry (MCP servers where you can), schemas, approvals, idempotency.

**What to get right:** reads retry freely; writes carry an idempotency key derived from the validated arguments, recorded before the call; risky writes wait for a human. Multi-step writes need a compensation path — if step 3 of 3 fails after steps 1 and 2 committed, something has to undo or flag them.

```python
def call(self, name, args, approval, done):
    if name in self.writes:
        key = idem_key(name, args)
        if key in done:                       # at-least-once delivery, exactly-once effect
            return f"skipped duplicate write {key}; already {done[key]}"
        if not approval:
            return "needs approval: issue_refund moves money"
        result = self.registry[name](**args)
        done[key] = result
        return result
    return self.registry[name](**args)
```

**The incident it prevents:** the double refund. It is the most expensive bug in this lesson and the cheapest to prevent.

**Metric:** writes per workflow (should be exactly one where you expect one) and approval latency.

### 6. Knowledge and memory

**Owns:** retrieval (vector, hybrid, graph), session state, long-term memory about the user or the account.

**What to get right:** three different things live here and they have different lifetimes — the retrieved context for this turn, the session state for this conversation, and durable memory about the customer. Conflating them is how you get an agent that re-reads the same document on every turn or, worse, remembers a preference from a different customer.

**The incident it prevents:** an agent that answers confidently from a stale index after a policy change, because nobody versioned the index with the prompt.

**Metric:** retrieval recall on a held-out query set, and memory hit rate — how often the session state saves a tool call.

### 7. Async backbone

**Owns:** queues, workers, retries with backoff, dead-letter queues, the job cursor.

**What to get right:** anything that might exceed a couple of seconds becomes a job with an id, and the cursor is the resume point. A worker that dies at step 3 of 10 continues at step 3. The delivery guarantee is at-least-once, so this layer is where the idempotency from layer 5 earns its keep.

```python
def work(self, jid, crash_at=None):
    job = self.jobs[jid]
    job["state"] = "running"
    while job["cursor"] < job["steps"]:
        job["cursor"] += 1
        if crash_at is not None and job["cursor"] == crash_at:
            job["state"] = "crashed"
            return f"crashed at step {job['cursor']}"
    job["state"] = "done"
    return f"done in {job['steps']} steps"
```

**The incident it prevents:** the load balancer cutting a request at 60 seconds while the agent keeps working, so the work is paid for and the user never sees it.

**Metric:** queue depth, age of the oldest job, retry rate, dead-letter count.

### 8. Trust layer

**Owns:** guardrails on input and output, model fallbacks, the kill switch, evals, traces, the audit log, and an identity for every agent.

**What to get right:** the kill switch is a flag read on every step, not a redeploy. A stop is not an undo — work already in flight can still complete, so the switch needs a report of what slipped through. Every action carries the identity that took it, so "which agent issued this refund" has an answer that is not "an agent".

```python
class KillSwitch:
    def trip(self, reason: str) -> None:
        self.on = True
        audit(event="kill_switch", reason=reason)

    def check(self, agent: str) -> None:
        if self.on:
            raise PermissionError(f"kill switch is on; refusing {agent}")
```

**The incident it prevents:** the 40-minute window in which an agent with a bad prompt is posting incorrect answers, because stopping it required a deploy.

**Metric:** blocked inputs per thousand, trace coverage (what share of actions appear in a trace), and kill-switch latency from decision to first refusal.

### 9. CI/CD

**Owns:** the build, the eval gate, the canary, the rollback, versioned prompts and configs.

**What to get right:** a change ships only if it passes the eval set, then to 5% of traffic, then to 100%, and rolls back automatically when quality or cost moves the wrong way.

```python
def canary_gate(new: dict[str, float], baseline: dict[str, float]) -> str:
    if new["quality"] < baseline["quality"] - 0.02:
        return "rollback: quality dropped"
    if new["cost"] > baseline["cost"] * 1.15:
        return "rollback: cost up more than 15%"
    return "promote to 100%"
```

**The incident it prevents:** the prompt tweak that reads better on your three examples and is worse on 30% of real traffic, discovered by customers.

**Metric:** gate pass rate and rollback count. A gate that never fails is not measuring anything.

### The arithmetic: price the workflow, not the call

Take a six-step workflow with 2,000-token calls. The cheap tier costs $0.0002 per 1k tokens, so a call is $0.0004 and six cheap steps are $0.0024. Escalate one step to the frontier tier at $0.0100 per 1k and the same workflow costs $0.022 — 9.2 times more.

So the number that decides your bill is not the price list. It is the escalation rate:

| Escalation rate | Cost per workflow | Per 10,000 workflows |
|---|---|---|
| 10% | $0.00436 | $43.60 |
| 25% | $0.00730 | $73.00 |
| 40% | $0.01024 | $102.40 |
| 60% | $0.01416 | $141.60 |

Read the bottom row as a design finding, not a pricing finding: 60% escalation means your cheap tier cannot do the job, and the fix is better retrieval or a narrower task, not a discount.

## Code

One file, all nine layers, no network and no API key: the model is a stub with prices, latency and a failure switch. Run it and read the summary table — each row names the layer that decided the outcome.

```python
"""Nine layers of a production agentic app, in one runnable file.

    python agent_stack.py

No network, no GPU, no API key: the model is a stub with prices, latency and a failure
switch, so every layer can be seen doing its job. Run it and read the summary table at
the end: each row names the layer that decided the outcome.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

AUDIT = Path("audit.jsonl")
CHECKPOINT = Path("stack_checkpoint.json")
for p in (AUDIT, CHECKPOINT):
    p.unlink(missing_ok=True)

# ------------------------------------------------------------------ 8. trust layer
class KillSwitch:
    """Stops every agent at once, without a redeploy. A flag, read on every step."""

    def __init__(self) -> None:
        self.on = False

    def trip(self, reason: str) -> None:
        self.on = True
        audit(event="kill_switch", reason=reason)

    def check(self, agent: str) -> None:
        if self.on:
            raise PermissionError(f"kill switch is on; refusing {agent}")


def audit(**event: Any) -> None:
    with AUDIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": round(time.time(), 3), **event}) + "\n")


GUARDRAILS = [re.compile(r"\bdrop\s+table\b", re.I), re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")]


def guardrail(text: str) -> str | None:
    """Input and output checks. Cheap, deterministic, and they run before money is spent."""
    for pattern in GUARDRAILS:
        if pattern.search(text):
            return f"blocked by guardrail {pattern.pattern!r}"
    return None


class Breaker:
    """Per-workflow caps: a stuck multi-agent loop trips this before the invoice does."""

    def __init__(self, max_tokens: int = 4000, max_seconds: float = 5.0) -> None:
        self.max_tokens, self.max_seconds = max_tokens, max_seconds
        self.tokens, self.started = 0, time.monotonic()

    def charge(self, tokens: int) -> None:
        self.tokens += tokens
        if self.tokens > self.max_tokens or time.monotonic() - self.started > self.max_seconds:
            raise RuntimeError("circuit breaker: workflow cap exceeded")


# ---------------------------------------------------------------------- 4. models
@dataclass(frozen=True)
class Tier:
    name: str
    price_per_1k: float
    latency_ms: int
    quality: float


TIERS = {
    "small": Tier("small", 0.0002, 120, 0.72),
    "mid": Tier("mid", 0.0015, 350, 0.86),
    "frontier": Tier("frontier", 0.0100, 900, 0.94),
}


class ModelGateway:
    """One client in front of every provider: retries, fallback, cost, and a failure switch."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.spend = 0.0

    def complete(self, tier: Tier, prompt: str, tokens: int = 400) -> dict[str, Any]:
        if tier.name in self.failing:
            raise TimeoutError(f"{tier.name} timed out")
        cost = tokens / 1000 * tier.price_per_1k
        self.spend += cost
        return {"text": f"[{tier.name}] answer to: {prompt[:40]}", "tokens": tokens,
                "cost": cost, "quality": tier.quality, "latency_ms": tier.latency_ms}


def route(prompt: str, task: str) -> Tier:
    """Small model first; escalate on task class or length, not on vibes."""
    if task in {"code_review", "legal"} or len(prompt) > 1500:
        return TIERS["frontier"]
    if task in {"summarise", "classify"}:
        return TIERS["small"]
    return TIERS["mid"]


# ------------------------------------------------------------- 5. tools (with side effects)
class Tools:
    """MCP-style registry. Writes are gated and idempotent; reads retry freely."""

    def __init__(self) -> None:
        self.ledger: dict[str, str] = {}
        self.registry: dict[str, Callable[..., str]] = {
            "search_docs": lambda q: f"3 docs about {q}",
            "issue_refund": lambda order_id, amount: self._refund(order_id, amount),
        }
        self.writes = {"issue_refund"}

    def _refund(self, order_id: str, amount: int) -> str:
        self.ledger[order_id] = f"refunded {amount}"
        return f"refund {amount} issued for {order_id}"

    def call(self, name: str, args: dict[str, Any], approval: bool, done: dict[str, str]) -> str:
        if name not in self.registry:
            return f"unknown tool {name}; available: {list(self.registry)}"
        if name in self.writes:
            key = idem_key(name, args)
            if key in done:                       # at-least-once delivery, exactly-once effect
                return f"skipped duplicate write {key}; already {done[key]}"
            if not approval:
                return "needs approval: issue_refund moves money"
            result = self.registry[name](**args)
            done[key] = result
            return result
        return self.registry[name](**args)


def idem_key(name: str, args: dict[str, Any]) -> str:
    return hashlib.sha256((name + json.dumps(args, sort_keys=True)).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ 3. orchestration
@dataclass
class Checkpoint:
    """Resume from the last good step; a replay must not repeat a write."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        self.data.setdefault("done", {})
        self.data.setdefault("steps", [])
        return self.data

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


SPECIALISTS = {"billing": "billing_agent", "refund": "billing_agent", "docs": "research_agent",
               "how": "research_agent", "policy": "research_agent"}


def supervisor(task: str) -> str:
    """Plans and hands off. The trace must record which specialist was chosen, and why."""
    for keyword, agent in SPECIALISTS.items():
        if keyword in task.lower():
            audit(event="route", agent=agent, because=f"matched {keyword!r}")
            return agent
    audit(event="route", agent="general_agent", because="no keyword matched")
    return "general_agent"


# ---------------------------------------------------------------- 2. gateway + cache
CACHE: dict[str, dict[str, Any]] = {}


def cache_key(tenant: str, prompt: str, policy_version: str) -> str:
    """Policy-versioned key: change the prompt and the cache stops lying to you."""
    blob = f"{tenant}|{policy_version}|{re.sub(r'\\s+', ' ', prompt.lower().strip())}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Request:
    token: str
    tenant: str
    prompt: str
    task: str
    approval: bool = False
    order_id: str = "ORD-1042"


def gateway(req: Request, kill: KillSwitch) -> tuple[Request, str | None]:
    """Auth, tenant from the token, guardrail, cache lookup. Everything before the spend."""
    if req.token != "valid-token":
        return req, "401 unauthenticated"
    kill.check("gateway")
    problem = guardrail(req.prompt)
    if problem:
        audit(event="guardrail", tenant=req.tenant, detail=problem)
        return req, f"400 {problem}"
    key = cache_key(req.tenant, req.prompt, "v3")
    if key in CACHE:
        audit(event="cache_hit", tenant=req.tenant, key=key)
        return req, None
    return req, None


# ------------------------------------------------------------------- 1. the request path
def handle(req: Request, models: ModelGateway, tools: Tools, ckpt: Checkpoint,
           kill: KillSwitch, breaker: Breaker) -> dict[str, Any]:
    try:
        _, error = gateway(req, kill)
    except PermissionError as exc:
        return {"status": "refused", "reason": str(exc), "layer": "8 trust"}
    if error:
        layer = "2 gateway" if error.startswith("401") else "8 trust"
        return {"status": "rejected", "reason": error, "layer": layer}

    key = cache_key(req.tenant, req.prompt, "v3")
    if key in CACHE:
        return {"status": "cached", "answer": CACHE[key]["text"], "cost": 0.0, "layer": "2 gateway/cache"}

    agent = supervisor(req.prompt)
    tier = route(req.prompt, req.task)
    try:
        result = models.complete(tier, req.prompt)
    except TimeoutError as exc:
        audit(event="fallback", from_tier=tier.name, reason=str(exc))
        result = models.complete(TIERS["small"], req.prompt)     # degrade, do not fail

    try:
        breaker.charge(result["tokens"])
    except RuntimeError as exc:
        audit(event="escalated", tenant=req.tenant, reason=str(exc))
        return {"status": "escalated", "reason": str(exc), "layer": "8 trust"}
    CACHE[key] = result
    ckpt.load()
    ckpt.data["steps"].append({"agent": agent, "tier": tier.name})
    ckpt.save()

    out = {"status": "ok", "agent": agent, "tier": tier.name, "cost": round(result["cost"], 6),
           "layer": "4 models"}
    if req.task == "refund":
        msg = tools.call("issue_refund", {"order_id": req.order_id, "amount": 750},
                         req.approval, ckpt.data["done"])
        out["tool"] = msg
        out["layer"] = "5 tools"
        ckpt.save()
    audit(event="answered", tenant=req.tenant, agent=agent, tier=tier.name, cost=out["cost"])
    return out



# ------------------------------------------------------------------- 7. async backbone
class JobQueue:
    """Long tasks run as events with an id, never as an HTTP call that times out.

    The cursor is the resume point: a worker that dies at step 3 of 10 continues at step 3,
    and no completed step is paid for twice.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def submit(self, task: str, steps: int) -> str:
        jid = f"job-{len(self.jobs) + 1:03d}"
        self.jobs[jid] = {"task": task, "state": "queued", "cursor": 0, "steps": steps}
        audit(event="enqueued", job=jid, task=task, steps=steps)
        return jid

    def work(self, jid: str, crash_at: int | None = None) -> str:
        job = self.jobs[jid]
        job["state"] = "running"
        while job["cursor"] < job["steps"]:
            job["cursor"] += 1
            if crash_at is not None and job["cursor"] == crash_at:
                job["state"] = "crashed"
                audit(event="crashed", job=jid, at_step=job["cursor"])
                return f"crashed at step {job['cursor']}"
            audit(event="step_done", job=jid, step=job["cursor"])
        job["state"] = "done"
        return f"done in {job['steps']} steps"


# ------------------------------------------------------------------------ 9. CI/CD gate
def canary_gate(new: dict[str, float], baseline: dict[str, float]) -> str:
    """Quality must not drop and cost must not rise: the gate that decides a rollout."""
    if new["quality"] < baseline["quality"] - 0.02:
        return "rollback: quality dropped"
    if new["cost"] > baseline["cost"] * 1.15:
        return "rollback: cost up more than 15%"
    return "promote to 100%"


if __name__ == "__main__":
    kill, models, tools = KillSwitch(), ModelGateway(failing={"frontier"}), Tools()
    ckpt = Checkpoint(CHECKPOINT)
    rows: list[dict[str, Any]] = []

    def run(label: str, req: Request, breaker: Breaker | None = None) -> None:
        out = handle(req, models, tools, ckpt, kill, breaker or Breaker())
        rows.append({"scenario": label, **{k: out.get(k) for k in ("status", "layer", "reason", "tool")}})

    run("first request", Request("valid-token", "acme", "How do I request a refund?", "refund", True))
    run("same request again", Request("valid-token", "acme", "How do I request a refund?", "refund", True))
    run("bad token", Request("nope", "acme", "hi", "chat"))
    run("guardrail", Request("valid-token", "acme", "drop table users", "chat"))
    run("unapproved write", Request("valid-token", "acme", "refund for ORD-2001", "refund", False,
                                    order_id="ORD-2001"))
    run("replayed write", Request("valid-token", "acme", "refund again for ORD-1042", "refund", True))
    run("frontier down -> fallback", Request("valid-token", "beta", "review this 2000 token diff " + "x" * 1600, "code_review"))
    run("loop trips breaker", Request("valid-token", "gamma", "policy question", "policy"), Breaker(max_tokens=100))
    kill.trip("operator: bad outputs reported")
    run("kill switch on", Request("valid-token", "acme", "another question", "chat"))

    print(f"{'scenario':28} {'status':10} {'layer':18} detail")
    for row in rows:
        detail = row.get("reason") or row.get("tool") or ""
        print(f"{row['scenario']:28} {row['status']:10} {str(row['layer']):18} {str(detail)[:44]}")

    queue = JobQueue()
    jid = queue.submit("ingest 5,000 support documents", steps=5)
    print(f"\n{jid} first attempt: {queue.work(jid, crash_at=3)}")
    print(f"{jid} resumed      : {queue.work(jid)}")
    print(f"{jid} state        : {queue.jobs[jid]['state']}, cursor {queue.jobs[jid]['cursor']}")

    print(f"\nspend this run: {models.spend:.6f}   cache entries: {len(CACHE)}")
    print(f"writes recorded: {ckpt.load()['done']}")
    print("canary:", canary_gate({"quality": 0.83, "cost": 0.004}, {"quality": 0.86, "cost": 0.004}))
    print(f"audit lines: {len(AUDIT.read_text().splitlines())} -> {AUDIT}")
```

Output:

```text
scenario                     status     layer              detail
first request                ok         5 tools            refund 750 issued for ORD-1042
same request again           cached     2 gateway/cache
bad token                    rejected   2 gateway          401 unauthenticated
guardrail                    rejected   8 trust            400 blocked by guardrail '\bdrop\s+table\...
unapproved write             ok         5 tools            needs approval: issue_refund moves money
replayed write               ok         5 tools            skipped duplicate write df143b0e7d; already
frontier down -> fallback    ok         4 models
loop trips breaker           escalated  8 trust            circuit breaker: workflow cap exceeded
kill switch on               refused   8 trust            kill switch is on; refusing gateway

job-001 first attempt: crashed at step 3
job-001 resumed      : done in 5 steps
job-001 state        : done, cursor 5

spend this run: 0.002480   cache entries: 4
writes recorded: {'df143b0e7d': 'refund 750 issued for ORD-1042'}
canary: rollback: quality dropped
audit lines: 20 -> audit.jsonl
```

Nine rows, nine answers to "what happens when". The duplicate write is skipped because the idempotency key was in the checkpoint, not because the model was careful. The breaker trips on a per-workflow token cap, which is the comment that turned into a line of code. The job resumes at step 3 and does not repeat steps 1 and 2. The canary refuses a 3-point quality drop, which is the layer that would have caught the prompt regression.

None of this is what the model does. All of it is what the harness, the platform and the pipeline do.

## Real use case

A 4,000-person services company built an internal IT-helpdesk assistant: Teams and Slack first, web later, roughly 9,000 conversations a month. The pilot was three boxes and it worked, because thirty people were using it.

The first month of production taught the nine layers in order.

**Cost.** A single workflow — "check my laptop ticket, find the policy, draft a reply" — looped eleven times on one prompt and cost $1.90. At the pilot's volume that was invisible. At 9,000 conversations it was the whole month's budget in a week. The fix was a per-workflow token cap with a breaker that escalated to a human, plus cost recorded per workflow so the number was visible at all.

**Duplicate writes.** A worker restart replayed a step and created two tickets for the same issue in the ITSM tool, then sent two emails. The fix was an idempotency key on the ticket-create tool and a compensation path for the two-step "create ticket, then notify requester" action.

**Routing.** The supervisor sent payroll questions to the technical specialist because the word "account" appeared in both. The fix was not a better prompt; it was recording the routing reason in the trace, which showed the keyword match, and then replacing keyword matching with an explicit classifier plus a fallback to the general agent.

**The outage.** The frontier provider rate-limited them at 10:40 on a Monday. Everything failed for 22 minutes because there was no fallback. The fix was a tiered gateway with a documented degradation: frontier down, mid takes over, and legal questions queue for a human instead of answering badly.

**The 2am question.** During a bad week of answers, nobody could stop the agent without a deploy. The kill switch was added as a flag read on every step, and the first drill found the gap the comment predicted: three jobs were mid-flight when the switch tripped and completed anyway, so the switch now writes a report of what slipped through.

**The rollout.** The prompt regression that would have shipped — a new system prompt that dropped "cite the policy id" — was caught by the eval gate at 5% canary, with a 6-point drop in citation rate, and rolled back automatically. That is the layer people skip and the one that pays for itself once.

Nine layers, one product, five incidents. Every incident was in a layer, not in the model.

## Project briefs

Three projects that exercise the whole stack without a cloud bill. Each is small enough for a weekend and defensible in an interview.

**A. The nine-layer miniature.** Build the stack from this lesson for one real task — a document Q&A bot with one write tool. Compose LiteLLM (gateway), Qdrant (retrieval), Redis (cache), Postgres (state and audit), a FastAPI service and one worker, all in `docker compose`. *Acceptance:* you can demo a cache hit, a fallback when the frontier tier is disabled, an idempotent write replay, a breaker trip and a kill switch, each in under a minute. *Layers exercised:* 2-8.

**B. Cost attribution service.** Add a ledger that records cost per workflow id, per tenant and per agent, and a small dashboard showing escalation rate and cost per workflow. *Acceptance:* a deliberately looping workflow trips its cap and appears in the ledger as one workflow with a breach flag, not as forty anonymous model calls. *What to measure:* escalation rate, cost per workflow, p95 steps per workflow.

**C. Canary and eval gate.** A GitHub Actions pipeline that runs 50 eval cases on every prompt change, deploys to 5% of traffic, compares quality and cost against the baseline for an hour, and rolls back automatically on a regression. *Acceptance:* a deliberately worse prompt is rolled back without a human touching it, and the rollback is visible in the audit log. *Layers exercised:* 8-9.

## Common mistakes

- **Mistake:** drawing the trust layer as one box in the middle — **Instead:** wrap it around every layer and name where each guardrail runs.
- **Mistake:** tracking cost per model call — **Instead:** track cost per workflow, with the escalation rate beside it; that is the number that explains the bill.
- **Mistake:** a kill switch that requires a deploy — **Instead:** a flag read on every step, plus a report of in-flight work that completed after the trip.
- **Mistake:** idempotency added after the first duplicate write — **Instead:** key every write from day one, and record the key before the call, not after.
- **Mistake:** a supervisor with no recorded reason for its routing — **Instead:** log the matched rule or classifier score; "why this specialist" is the first question in every incident review.
- **Mistake:** long tasks run inside the request — **Instead:** a job id, a cursor and a worker; the HTTP call returns in a second.
- **Mistake:** resuming an in-flight task after a rollout with no version pinning — **Instead:** pin config and prompt versions per task, or keep checkpoint schemas backward compatible.
- **Mistake:** an eval gate that has never failed — **Instead:** test the gate with a deliberately worse prompt; if it promotes, the gate is decoration.

## Checkpoint

1. Run `agent_stack.py`, then delete `stack_checkpoint.json` and re-run. Explain which line of output changes and why the refund is issued twice.
2. Add a seventh scenario: two tenants, one of them over its monthly quota. Decide where the check belongs (gateway or orchestrator) and write the one-line reason in the audit record.
3. Give the supervisor a second failure mode: when the chosen specialist raises, hand off to the general agent and record the handoff in the trace. Confirm the run still ends in a final answer.
4. Write the nine-layer table for a product you know, and fill in the metric column with a number you would actually put on a dashboard. Any row you cannot fill is the layer your design is missing.

**Self-check:** Exercise 1 — with the checkpoint deleted, the `writes recorded` line and the `replayed write` row change: the idempotency key is no longer in the ledger, so the second refund is issued instead of skipped. The lesson is that the checkpoint is the memory of side effects, not a cache.

## Go deeper

- [Scalable AI Apps](01-scalable-ai-apps.md) — the pipeline reference architecture and its seams.
- [Agent Design](03-agent-design.md) — state machines, budgets, idempotency in depth.
- [Model Routing](04-model-routing.md) — tiers, cascades and measured routes.
- [Multi-Tenant](05-multi-tenant.md) — isolation, quotas and cost attribution.
- [Caching](06-caching.md) — the four cache layers and their invalidation.
- [Fault Tolerance](07-fault-tolerance.md) — timeouts, breakers and degraded modes.
- [Cost and Latency](08-cost-latency.md) — the unit economics behind the arithmetic above.
- [Ship: A Design Document](10-ship-design-doc.md) — turn this stack into ten pages a reviewer can check.
- [Async Processing](../17-ai-backend/03-async-processing.md) — queues, workers and retries.
- [Tracing](../18-mlops-llmops/05-tracing.md) — spans across all nine layers.
- [AI Security](../20-ai-security/README.md) — guardrails, identity and least privilege.
- [Ship: A Harness Agent](../11-agents/12-ship-harness-agent.md) — layers 3 to 5 as working code.
- The nine-layer split follows Gaurav Pandey's [architecture sketch](https://lnkd.in/p/dp3nBPq2), November 2026. The per-workflow caps, cost-per-workflow framing, side-effect safety and version-pinning points come from the discussion under it.

---

[← Previous: Ship: A Design Document](10-ship-design-doc.md) · [Stage index](README.md) · [Next stage: Real Projects](../22-real-projects/README.md)
