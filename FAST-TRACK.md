# The fast track

**Seven stages, in this order: 7 → 8 → 10 → 11 → 12 → 15 → 22.**

This is the shortest honest path from "I can write Python" to "I have shipped an LLM system and can prove it works." It is the same path the roadmap calls *Start Here*.

It is not a shortcut past understanding. It is a bet: that you will learn more from building one measured system than from reading six months of theory, and that the theory you skipped will be much easier to absorb once you have felt the problem it solves.

---

## What you skip, and what that costs you

| Skipped | What you lose by skipping | When you will come back |
|---------|---------------------------|--------------------------|
| 1-2 Programming, maths | Fine, if you already write async Python and understand a dot product | When a similarity score surprises you |
| 3-4 Data, deep learning | How embeddings are made, why shapes break, what a loss curve is telling you | When fine-tuning or debugging a model |
| 5-6 DL frameworks, LLM fundamentals | Tokenisation, context limits, why the model behaves the way it does | When a prompt "should work" but does not |
| 9 Vector databases | Index trade-offs, recall measurement | When search gets slow or misses |
| 13-14 Fine-tuning, multimodal | Alternative to RAG; image and audio inputs | When retrieval is not enough, or a PDF has figures |
| 16-21 Inference, backend, ops, security, design | The production half | The day your demo becomes a product |

Rule of thumb: **the fast track gets you to a working system; stages 16-21 are what let you keep it working.**

---

## Week by week

### Week 1 — You can call a model properly

Read: [7. LLM APIs](stages/07-llm-apis/README.md) in full.

Do, in this order:

1. `01-openai-anthropic` → one script that sends a prompt and prints the response and the token counts.
2. `03-streaming` → make the same call stream to the terminal, then to a browser with Server-Sent Events.
3. `04-structured-outputs` → get a validated pydantic object back, with a repair loop on parse failure.
4. `05-tool-calling` → one round of tool call, result, follow-up answer. Do not use a framework.
5. `07-retries-rate-limits` → make it survive a 429 and a timeout with exponential backoff and jitter.
6. `08-cost-tracking` → log tokens and cost per request to a JSONL file.
7. `10-ship-multi-model-client` → collapse all of it into one client with two providers and a fallback.

**Checkpoint:** you can explain to a colleague why time-to-first-token and total latency are different numbers, and what you would change to improve each.

### Week 2 — You can control the context

Read: [8. Prompt and Context](stages/08-prompt-and-context/README.md).

Do:

1. Write a system instruction with explicit scope, refusal behaviour and output shape. Test it with ten adversarial inputs.
2. Build a prompt template with typed variables; fail loudly on a missing variable.
3. Take a 40-page document and compress it into a 1,500-token context without losing the facts that matter for a specific question.
4. Version a prompt as a file with an id and a changelog; make the app load the version, not a string literal.
5. Defend an injection: wrap untrusted content in delimiters, tell the model it is data, and check the output.
6. `09-ship-prompt-test-suite` → fixtures, assertions, pass rate, exit code.

**Checkpoint:** you have a prompt that survives a red-team pass, and a test suite that fails when you break it on purpose.

### Week 3 — Retrieval that actually retrieves

Read: [10. RAG](stages/10-rag/README.md). Skim [9](stages/09-vector-databases/README.md) for embeddings, similarity and indexes as you go.

Do:

1. Ingest a real corpus you care about — your notes, a product's docs, a set of PDFs.
2. Build the naive version: fixed chunks, one embedding model, top-k, answer with citations. Measure it.
3. Break it deliberately: ask a question whose answer spans two chunks. Watch it fail.
4. Add hybrid retrieval (BM25 + vectors, fused) and measure again.
5. Add a reranker and measure again.
6. Add parent-child retrieval so the answer sees the full section.
7. `12-rag-evals` → build 30 question/answer pairs and score context recall, faithfulness and answer relevance.
8. `13-ship-rag-with-evals` → ship it with the numbers in the README.

**Checkpoint:** you can state your retrieval score before and after each change, and explain which change bought the most.

### Week 4 — The agent loop

Read: [11. AI Agents](stages/11-agents/README.md).

Do:

1. `02-agent-architecture` → write the loop yourself in ~80 lines: messages, tool registry, execute, observe, stop conditions.
2. `04-tool-calling` → three tools with real schemas, one of which can fail, and error messages the model can act on.
3. `10-loop-engineering` → add a step budget, a no-progress detector and a hard timeout. Watch an agent get stuck and recover.
4. `08-human-in-the-loop` → pause before a destructive tool call, resume after approval.
5. `11-agent-security` → treat tool output as untrusted; make one tool return an injection and prove your app does not obey it.
6. `12-ship-harness-agent` → the full harness: typed tools, budget, trace, approval, and a five-case eval.

**Checkpoint:** you can explain why "give the model more tools" is usually the wrong fix for a failing agent.

### Week 5 — Graphs, not vibes

Read: [12. Orchestration](stages/12-orchestration/README.md) — specifically LangGraph and the trade-offs file.

Do:

1. Rebuild your agent loop as a state graph: nodes, edges, one conditional edge.
2. Add a persistence layer so a run can resume after a crash.
3. Add an approval node as an interrupt.
4. Add a router node that sends simple queries to a cheap path and hard ones to the full path.
5. Read `10-tradeoffs` and write down, in your own words, when you would *not* use a framework.

**Checkpoint:** your graph can be drawn on a whiteboard in 30 seconds, and every edge has a reason.

### Week 6 — Prove it works

Read: [15. Evals and Testing](stages/15-evals/README.md).

Do:

1. Build a golden dataset of 50 cases from real usage, including the ugly ones.
2. Write deterministic tests for everything that can be deterministic (parsing, chunking, routing, schema).
3. Add an LLM-as-judge grader with a rubric, then calibrate it against 20 human labels and report the agreement.
4. Add a regression gate: fail the build when pass rate drops below your baseline.
5. `10-ship-eval-suite` → one command that prints a scorecard.

**Checkpoint:** you would be comfortable putting your numbers in a slide, because you know how they were measured.

### Weeks 7-8 — One real project

Read: [22. Real Projects](stages/22-real-projects/README.md) and pick exactly one brief. If unsure, pick **production RAG** or **support agent** — they exercise the widest surface.

Deliverables:

- A repo with a README that opens with the problem, the design decision, the measurement and the lesson.
- A running deployment (a container, a URL, or a documented local run).
- An eval report with numbers and a baseline.
- A cost model: cost per request, per thousand requests, per month at your assumed volume.
- A demo: a 2-minute screen recording or a live link.

**Then stop and read [Stage 21](stages/21-system-design/README.md).** It is the file that turns a working project into a design you can defend in an interview.

---

## Daily rhythm that works

| Time | Activity |
|------|----------|
| 20 min | Read the lesson file once, without a terminal |
| 60 min | Retype the code, then change one thing and see what happens |
| 20 min | Do the checkpoint exercises |
| 10 min | Write three lines in a lab notebook: what worked, what broke, what you still do not understand |

The lab notebook is not optional. It is your interview material.

## What "done" looks like at the end of the fast track

- You have one deployed system with citations, evals and a cost model.
- You can explain the failure modes of your system before anyone asks.
- You can name the specific number that would tell you the system got worse.
- You know which three stages you skipped, and you have a plan to come back to them.

At that point, go back and take stages 3, 6, 9, 16, 17 and 21 properly. That is the difference between someone who can build a demo and someone who can be trusted with a product.

---

[← Back to the roadmap](README.md) · [Stage 7: LLM APIs →](stages/07-llm-apis/README.md)
