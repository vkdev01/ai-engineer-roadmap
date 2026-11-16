# 8. DSPy

> **Stage 12 · Orchestration** — programming prompts as modules and letting an optimiser find the instructions you would have hand-tuned for a week.
>
> **You'll be able to:** express a multi-step LLM program as DSPy signatures and modules, compile it against a metric, and judge whether the compiled version beats your hand-written prompt.
>
> **Prerequisites:** [LangGraph](02-langgraph.md) · [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md) · [Prompt Templates](../08-prompt-and-context/03-templates.md)

## Why this exists

Every prompt you have written so far is a string you tuned by eye. You changed a sentence, re-ran five examples, decided it felt better, and shipped it. That works until you have twelve prompts in a pipeline and one metric that matters, because now you are hand-tuning twelve strings against a single number, and you have no idea which sentence moved it.

DSPy starts from a different premise: the *prompt text is a parameter*. You declare what each step takes and returns (a signature), wire the steps into a module, give it a metric and a small training set, and let an optimiser search for the instructions and few-shot examples that maximise the metric. You stop writing prompts and start writing programs with prompts inside them.

The mental model: **you wrote a function; DSPy compiles the prompt for it.**

## The mental model

```text
   hand-written prompting              DSPy
   ----------------------              ----
   string  -> model -> parse           signature  (inputs -> outputs)
     ^                                   |
     | edit by eye                      module     (chain of signatures)
     |                                   |
   "feels better"                      metric + trainset
                                         |
                                       compiler  (searches instructions,
                                                  demos, and step order)
                                         |
                                       optimised program  (prompt text is
                                                           now an artefact)
```

The trade: you give up fine control of wording, and you gain a loop that can improve wording without you. That trade is only worth it when you have a metric and a dataset — which is exactly what Stage 15 makes you build.

## Core concepts

### Signatures: the contract, not the wording

A signature declares inputs and outputs with a docstring that states the task. You never write "You are a helpful assistant..." — the optimiser supplies instruction text.

```python
import dspy

class ExtractAction(dspy.Signature):
    """Read a support email and extract the customer's requested action."""

    email: str = dspy.InputField(desc="the raw email body")
    action: str = dspy.OutputField(desc="one of: refund, replace, cancel, question")
    urgency: int = dspy.OutputField(desc="1 (low) to 5 (high)")
```

Two things matter here. The `desc=` fields are load-bearing: they are the only place you influence behaviour directly, and vague descriptions produce vague output. And the output fields are typed by intent, not by JSON schema — DSPy handles the parsing, and a `Predict` module will retry when the model returns something unparseable.

### Modules: signatures composed into a program

A module is a class with a `forward` method. Inside it you call predictors. This is where the pipeline shape lives — and it is ordinary Python, which means you can branch, loop and validate however you like.

```python
class TriageAndRoute(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractAction)
        self.draft = dspy.Predict("email, action -> reply")

    def forward(self, email: str) -> dspy.Prediction:
        parsed = self.extract(email=email)
        # cheap guard: only draft a reply for actionable requests
        if parsed.action == "question":
            return dspy.Prediction(reply="", route="human")
        reply = self.draft(email=email, action=parsed.action)
        return dspy.Prediction(reply=reply.reply, route="auto")
```

`ChainOfThought` inserts a reasoning field before the answer; `Predict` does not. That single swap is the kind of decision you would otherwise bury in prompt prose.

### Metrics: the thing that makes this work

A metric is a function that takes an example and a prediction and returns a score. It is where the real thinking goes — a bad metric produces a beautifully optimised wrong program.

```python
def triage_metric(example, pred, trace=None) -> float:
    score = 0.0
    if pred.action == example.action:
        score += 0.7
    if abs(int(pred.urgency) - int(example.urgency)) <= 1:
        score += 0.3
    return score
```

Note what is *not* in the metric: length, tone, fluency. Optimisers exploit whatever you measure, so measure only what you would defend in a review.

### Compiling: the optimiser does the prompt engineering

You give it the module, a training set of a few dozen examples, and a metric. Different optimisers do different things: some generate and select instructions, some pick the best few-shot demonstrations per step, some bootstrap traces from successful runs and reuse them as examples.

```python
from dspy.teleprompt import BootstrapFewShot

trainset = load_labeled_emails()          # list[dspy.Example]
optimizer = BootstrapFewShot(metric=triage_metric, max_bootstrapped_demos=4)
compiled = optimizer.compile(TriageAndRoute(), trainset=trainset)
compiled.save("triage_compiled.json")
```

The output is a program whose prompts are now data. That is the actual product of this file: **prompts you can regenerate and version rather than retype.**

## Code

A minimal end-to-end run, with the parts that bite annotated.

```python
from __future__ import annotations

import dspy

# 1. Configure once. The LM is a DSPy object, not a client you pass around.
lm = dspy.LM("openai/gpt-4o-mini", temperature=0)   # check current model ids in provider docs
dspy.configure(lm=lm)


class ClassifyIntent(dspy.Signature):
    """Classify a customer message into exactly one support intent."""

    message: str = dspy.InputField(desc="raw customer text, may contain typos")
    intent: str = dspy.OutputField(desc="one of: billing, delivery, returns, account, other")


class SupportPipeline(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict(ClassifyIntent)

    def forward(self, message: str) -> dspy.Prediction:
        result = self.classify(message=message)
        # normalise: optimisers and models both drift towards casing variants
        return dspy.Prediction(intent=result.intent.strip().lower())


def metric(example, pred, trace=None) -> bool:
    return pred.intent == example.intent


# 2. A tiny trainset is enough to bootstrap demos; 20-60 examples is a normal start.
trainset = [
    dspy.Example(message="charged twice for one order", intent="billing").with_inputs("message"),
    dspy.Example(message="parcel says delivered but nothing arrived", intent="delivery").with_inputs("message"),
    dspy.Example(message="how do I change my email", intent="account").with_inputs("message"),
]

from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=2, max_labeled_demos=2)
compiled = optimizer.compile(SupportPipeline(), trainset=trainset)

# 3. Evaluate the compiled program on held-out data, never on the trainset.
devset = [
    dspy.Example(message="refund has not appeared after 9 days", intent="billing").with_inputs("message"),
    dspy.Example(message="courier left it with a neighbour", intent="delivery").with_inputs("message"),
]
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=False)
print(evaluate(compiled))
```

Three things to expect on your first run. The compiled program's prompts will be longer than yours and will contain demonstrations you did not write. Scores on the trainset will look flattering — only the devset number counts. And with fewer than ~20 examples, the optimiser will overfit so fast that you will conclude DSPy does not work, when the real problem is the dataset size.

TypeScript has no first-class equivalent; the practical pattern is to compile in Python, export the resulting instructions and demos as JSON, and consume them from your Node service:

```ts
import { z } from "zod";

const CompiledProgram = z.object({
  signature: z.string(),
  instructions: z.string(),
  demos: z.array(z.object({ input: z.string(), output: z.string() })),
});

type CompiledProgram = z.infer<typeof CompiledProgram>;

// The Node side just renders the compiled artefact into messages — no optimiser needed.
export function renderMessages(p: CompiledProgram, message: string) {
  const demos = p.demos
    .map((d) => `input: ${d.input}\noutput: ${d.output}`)
    .join("\n\n");
  return [
    { role: "system" as const, content: `${p.instructions}\n\n${demos}` },
    { role: "user" as const, content: message },
  ];
}
```

## Real use case

A logistics company has an email triage pipeline: classify, extract the order id, decide whether to auto-reply. The hand-written version works at roughly four in five correct on their internal sample, and two engineers spend a week per quarter adjusting sentences as new phrasings appear.

They move the pipeline to DSPy with 120 labelled emails collected from six months of the shared inbox and a metric that weights intent (0.6), order-id exactness (0.3) and refusal correctness (0.1). They bootstrap with `BootstrapFewShot`, evaluate on a 40-email held-out set, and get a small but real improvement over the hand-tuned baseline — the useful part is not the delta, it is that the delta was *measurable* and the prompts are now regenerated from data whenever a new batch of labels arrives.

The failure they hit first: their metric rewarded order-id exactness so heavily that the optimiser learned to answer "no order id found" for anything ambiguous, which scored 0.3 with certainty instead of risking a wrong id. Reweighting to punish false negatives fixed it. This is the general lesson of optimiser-driven work — **you get exactly the behaviour you measure, including the behaviour you forgot to mention.**

## Common mistakes

- ****Mistake:** compiling without a held-out set, then reporting the trainset score — **Instead:** split the data before you start; report devset only, and say how large it was.
- **Mistake:** a metric that measures fluency or length because it is easy to compute — **Instead:** measure the business outcome (correct intent, valid id, answer supported by context) even if it needs a second model or a human.
- **Mistake:** starting with a 10-example trainset and concluding the framework does not work — **Instead:** 20-60 examples is the practical floor for bootstrapping; below that, hand-written prompts are honestly better.
- **Mistake:** treating the compiled program as opaque and shipping it unread — **Instead:** print the generated instructions, read them, and delete anything that contradicts your policy.
- **Mistake:** reaching for DSPy on a single-prompt task — **Instead:** use it when you have multiple steps, a metric, and enough labels; otherwise a template and an eval suite is cheaper.
- **Mistake:** assuming compiled prompts transfer across model upgrades — **Instead:** recompile and re-evaluate after any model change; the optimiser's output is tuned to a model, not to your task in the abstract.

## Checkpoint

1. Take one prompt you already have and rewrite it as a DSPy signature plus a single `Predict` module. Compare output on ten examples against your original string.
2. Build a 30-example trainset and a 15-example devset for that task, write a metric, and compile with `BootstrapFewShot`. Report both scores and the gap.
3. Read the compiled instructions. Find one sentence the optimiser added that you would not have written, decide whether you agree with it, and write two lines on why.

**Self-check:** if your devset score is far below your trainset score, you have too little data or too many demonstrations — cut `max_bootstrapped_demos` first, then collect more labels.

## Go deeper

- [Trade-offs](10-tradeoffs.md) — where DSPy fits against hand-written prompts and plain graphs
- [Prompt Evals](../08-prompt-and-context/07-prompt-evals.md) — the metric discipline this file depends on
- [Golden Datasets](../15-evals/03-golden-datasets.md) — how to build the labelled set DSPy needs

---
[← Previous: Pydantic AI](07-pydantic-ai.md) · [Stage index](README.md) · [Next: CrewAI and AutoGen](09-crewai-autogen.md)
