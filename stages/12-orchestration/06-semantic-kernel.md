# 6. Semantic Kernel

> **Stage 12 · Orchestration** — typed plugins, kernel functions and planners: Microsoft's framework, and the one that matters when the agent has to live inside a .NET estate.
>
> **You'll be able to:** explain the plugin model, write a plugin as a function with a schema, and judge when Semantic Kernel is the right call over a Python-first stack.
>
> **Prerequisites:** [Haystack](05-haystack.md) · [Tool Calling](../11-agents/04-tool-calling.md)

## Why this exists

Most of this roadmap is Python-first, and most LLM work is too. But plenty of companies run their core systems on C# and .NET — banking back ends, ERP integrations, Windows-heavy infrastructure — and the AI feature has to call into those systems, with the same authentication, the same logging and the same deployment pipeline. Adding a Python sidecar with its own secrets and its own deploy is real cost.

Semantic Kernel exists for that situation. It brings the plugin and function vocabulary, a model abstraction that spans providers, and first-class support in C# alongside a Python SDK. If your team ships .NET, it lets the agent live in the same process as the code it calls.

The mental model: **a kernel is a registry of typed functions, and the model chooses which function to run.**

## The mental model

```text
                 ┌──────────────── kernel ────────────────┐
 request ──────► │  services:  chat model, embeddings      │
                 │  plugins:   inventory.get_stock         │
                 │             inventory.reserve           │
                 │             crm.lookup_customer         │
                 │  filters:   auth, logging, telemetry    │
                 └────────────────────────────────────────┘
                          │ model picks a function
                          ▼
                 typed call ──► your C# / Python code ──► result
```

| Concept | Meaning | Python | C# |
|---|---|---|---|
| Kernel | Container plus registry | `Kernel()` | `Kernel.CreateBuilder()` |
| Plugin | Group of related functions | class of methods | class with `[KernelFunction]` |
| Function | One callable capability | `@kernel_function` | method + attribute |
| Planner | Chooses functions for a goal | available | available |
| Filter | Wraps every call | yes | yes |

## Core concepts

### Plugins are classes, functions are methods

A plugin groups functions that share a dependency — an inventory plugin holds the repository client. Each function declares its parameters and description, and that declaration is what the model sees. Grouping by dependency rather than by task keeps constructors honest and makes the plugin testable without a model at all.

### Descriptions are still prompts

The same rule as [Tool Calling](../11-agents/04-tool-calling.md): a function's description and parameter descriptions are the model's documentation. Semantic Kernel makes them attributes, which tempts people to write `"Gets stock"`. Write the when, the when-not and the return shape instead.

### Filters are the reason to use it in an enterprise

Every function invocation and model call passes through filters, so authentication, audit logging, PII redaction and rate limiting go in one place instead of in every plugin. This is the framework's strongest argument in a regulated environment: policy is a cross-cutting concern, exactly as it would be in an ASP.NET pipeline.

### Planners: useful, then superseded

Older versions shipped planners that generated a plan of function calls from a goal. Modern practice moved toward native function calling, where the model picks functions directly, because it is more reliable and easier to trace. Treat planners as a legacy surface: prefer function calling with your own loop and a step budget.

### Interop is the point, not a bonus

The realistic reason to choose Semantic Kernel is that the agent calls existing .NET services in-process. If you are building greenfield in Python, LangGraph or Pydantic AI will feel more natural; if the feature must ship inside an existing .NET service, Semantic Kernel avoids an entire second deployment.

## Code

Python first: a plugin, a kernel with the plugin registered, and automatic function calling with a step budget. The model id must be checked against the provider's current docs.

```python
from __future__ import annotations

from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

class InventoryPlugin:
    """Functions for checking and reserving warehouse stock."""

    def __init__(self, repo: object) -> None:
        self._repo = repo

    @kernel_function(
        name="get_stock",
        description=("Returns available units for a SKU. Use before promising a delivery date. "
                     "Returns {sku, available, warehouse}."),
    )
    def get_stock(
        self,
        sku: Annotated[str, "SKU like 'ACME-4410'; not a product name"],
    ) -> dict[str, object]:
        return self._repo.stock_for(sku)          # plain Python; no model involved

    @kernel_function(
        name="reserve",
        description=("Reserves units for an order. Irreversible: needs a confirmed order id. "
                     "Idempotent on (order_id, sku)."),
    )
    def reserve(
        self,
        order_id: Annotated[str, "Confirmed order id, format ORD-######"],
        sku: Annotated[str, "SKU to reserve"],
        units: Annotated[int, "Whole units, 1-500"],
    ) -> dict[str, object]:
        return self._repo.reserve(order_id, sku, units)

async def ask(question: str, repo: object) -> str:
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id="<current-chat-model-id>"))
    kernel.add_plugin(InventoryPlugin(repo), plugin_name="inventory")

    settings = kernel.get_prompt_execution_settings_from_service_id()
    # Let the model call functions, but keep a cap on automatic invocations.
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=5)

    result = await kernel.invoke_prompt(question, execution_settings=settings)
    return str(result)
```

A filter, which is where enterprise policy belongs: it runs on every function call, so no plugin can skip it.

```python
from semantic_kernel.filters import FunctionInvocationContext

async def audit_filter(context: FunctionInvocationContext, next) -> None:
    # Runs before every kernel function; the single place to enforce policy.
    if context.function.plugin_name == "inventory" and context.function.name == "reserve":
        if not context.arguments.get("order_id"):
            raise ValueError("reserve requires a confirmed order_id")
    await next(context)
    # context.result now holds the function's return value; log it, never the secrets.
```

Now C#, the reason this framework exists. The same plugin model with attributes and dependency injection, so the agent runs in the same process as the services it calls.

```csharp
using Microsoft.SemanticKernel;
using System.ComponentModel;

public sealed class InventoryPlugin
{
    private readonly IInventoryRepository _repo;

    public InventoryPlugin(IInventoryRepository repo) => _repo = repo;

    [KernelFunction("get_stock")]
    [Description("Returns available units for a SKU. Use before promising a delivery date.")]
    public Task<StockLevel> GetStockAsync(
        [Description("SKU like 'ACME-4410'; not a product name")] string sku) =>
        _repo.StockForAsync(sku);

    [KernelFunction("reserve")]
    [Description("Reserves units for an order. Irreversible; idempotent on (orderId, sku).")]
    public Task<Reservation> ReserveAsync(
        [Description("Confirmed order id, format ORD-######")] string orderId,
        [Description("SKU to reserve")] string sku,
        [Description("Whole units, 1-500")] int units) =>
        _repo.ReserveAsync(orderId, sku, units);
}

// Registration keeps DI intact: the agent is just another consumer of your services.
var builder = Kernel.CreateBuilder();
builder.Services.AddSingleton<IInventoryRepository, InventoryRepository>();
builder.Plugins.AddFromType<InventoryPlugin>();
```

## Real use case

A manufacturer's order desk ran on .NET services with strict audit requirements. The team prototyped the assistant in Python, and the prototype worked — but shipping it meant a second deployment, a second secret store and a duplicate audit path, which their security review refused.

Rebuilt with Semantic Kernel inside the existing .NET service, the agent called the same repository classes as the web app, and an audit filter logged every `reserve` call to the same sink the rest of the system used. The model quality was identical to the Python prototype; what changed was that the feature could actually be deployed, reviewed and monitored by the team that owned the service.

## Common mistakes

- **Mistake:** one-word function descriptions — **Instead:** state when to use it, when not, and what it returns.
- **Mistake:** letting automatic invocation run unbounded — **Instead:** set a maximum auto-invoke count and keep a step budget.
- **Mistake:** relying on planners — **Instead:** prefer native function calling with your own loop.
- **Mistake:** putting auth and logging in each plugin — **Instead:** use a filter, once.
- **Mistake:** choosing it for a Python greenfield project — **Instead:** pick it when in-process .NET interop is the actual requirement.
- **Mistake:** side-effecting functions without idempotency — **Instead:** key on `(order_id, sku)` and let the repository collapse repeats.

## Checkpoint

1. Add a `cancel_reservation` function with an idempotency note, and mark it as needing approval in your loop.
2. Write a filter that measures function latency and writes one line per call to a log.
3. Port the `get_stock` plugin to C# and call it from a console app with a local model endpoint.

**Self-check:** Exercise 1 — accept `(order_id, sku)`, make the repository treat a repeat cancel as a no-op returning the existing state, and return that state so the model sees confirmation.

## Go deeper

- [Tool Calling](../11-agents/04-tool-calling.md) — the schema discipline that applies to plugins.
- [Pydantic AI](07-pydantic-ai.md) — the typed-agent idea in Python.
- [Trade-offs](10-tradeoffs.md) — where this sits against the Python-first stack.
- [API Security](../17-ai-backend/07-api-security.md) — the policy concerns filters implement.

---

[← Previous: Haystack](05-haystack.md) · [Stage index](README.md) · [Next: Pydantic AI](07-pydantic-ai.md)
