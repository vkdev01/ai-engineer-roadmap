# Glossary

Every term the roadmap uses, defined the way an engineer needs it — short, and tied to the thing you actually do with it. Cross-references point at the lesson where the idea is taught properly.

**Adam** — An optimiser that adapts a per-parameter learning rate using running averages of gradients and squared gradients. The default choice for most training. → [Optimizers](stages/04-deep-learning/03-optimizers.md)

**Agent** — A model plus a loop: it chooses a tool, sees the result, and decides what to do next until a stop condition is met. The loop, not the model, is the engineering. → [Agent architecture](stages/11-agents/02-agent-architecture.md)

**Agentic RAG** — Retrieval where the model decides what to search for and when, instead of a fixed retrieve-then-answer pipeline. Better on multi-hop questions, more expensive and harder to bound. → [Agentic RAG](stages/10-rag/11-agentic-rag.md)

**ANN (approximate nearest neighbour)** — Search that trades a little recall for a lot of speed by not comparing against every vector. Measure the recall you gave up. → [ANN search](stages/09-vector-databases/05-ann-search.md)

**Attention** — A weighted lookup: each token builds a query, compares it against every token's key, and takes a weighted sum of their values. Attention weights are the softmax of those comparisons. → [Attention](stages/04-deep-learning/06-attention.md)

**Autograd** — Automatic differentiation over a computation graph. PyTorch records the operations you perform on tensors, then walks backwards to get gradients. → [Autograd](stages/05-dl-frameworks/02-autograd.md)

**Backpropagation** — The chain rule applied backwards through a network to compute each parameter's gradient. → [Backprop](stages/02-math-and-ml-basics/06-backprop-gradients.md)

**BM25** — A classic lexical ranking function: scores documents by term frequency, inverse document frequency and length normalisation. Still beats embeddings on exact identifiers, names and codes. → [Hybrid search](stages/09-vector-databases/09-hybrid-search.md)

**BPE (byte pair encoding)** — A tokeniser that starts from bytes and repeatedly merges the most frequent adjacent pair. Explains why "tokenisation" is not "words" and why pricing is per token. → [Tokenization](stages/06-llm-fundamentals/02-tokenization.md)

**Chunking** — Splitting documents into retrievable units. The single highest-leverage decision in most RAG systems. → [Chunking](stages/10-rag/03-chunking.md)

**Circuit breaker** — A wrapper that stops calling a failing dependency after repeated errors, failing fast instead of queueing up timeouts. → [Fault tolerance](stages/21-system-design/07-fault-tolerance.md)

**Context window** — The maximum number of tokens the model can attend to in one call: prompt plus output. A hard budget you must engineer, not a soft suggestion. → [Context windows](stages/06-llm-fundamentals/03-context-windows.md)

**Context engineering** — Deciding which tokens occupy the window: what to include, what to compress, what to drop, and in what order. The successor to "prompt engineering". → [Context engineering](stages/08-prompt-and-context/08-context-engineering.md)

**Continuous batching** — A serving scheduler that adds new requests to a running batch as slots free up, instead of waiting for the slowest sequence to finish. The main reason modern servers get high throughput. → [Continuous batching](stages/16-inference/03-continuous-batching.md)

**Cosine similarity** — The cosine of the angle between two vectors: dot product divided by the product of their lengths. Ignores magnitude, which is usually what you want for text embeddings. → [Similarity metrics](stages/09-vector-databases/02-similarity-metrics.md)

**Cross-encoder** — A reranker that reads the query and document together and outputs a relevance score. Accurate and slow, so it runs on a shortlist, not the whole corpus. → [Reranking](stages/10-rag/06-reranking.md)

**Dense retrieval** — Retrieval by embedding similarity, as opposed to lexical matching. Good at meaning, bad at rare exact strings. → [Semantic search](stages/09-vector-databases/03-semantic-search.md)

**Distillation** — Training a small model to imitate a large one's outputs or distributions, trading a little quality for a lot of cost and latency. → [Distillation](stages/16-inference/07-distillation.md)

**DPO (direct preference optimisation)** — Preference tuning that optimises directly on chosen/rejected pairs, without a separate reward model. Cheaper than RLHF and often good enough. → [RLHF and DPO](stages/06-llm-fundamentals/07-rlhf-dpo.md)

**Embedding** — A fixed-length vector representing meaning, positioned so that similar inputs are close together. Everything in retrieval is built on this one idea. → [Embeddings](stages/09-vector-databases/01-embeddings.md)

**Eval** — A measurement of system quality on a fixed set of cases, with a grader and a threshold. If you cannot run it in one command, you do not have one. → [Evals](stages/15-evals/README.md)

**Faithfulness** — Whether the answer is supported by the retrieved context. Distinct from correctness: an answer can be true and still unfaithful to the sources you gave it. → [Faithfulness](stages/15-evals/04-faithfulness.md)

**Few-shot prompting** — Putting worked examples in the prompt to demonstrate the task. Powerful, and it costs tokens on every call. → [Few-shot and CoT](stages/08-prompt-and-context/02-few-shot-cot.md)

**Fine-tuning** — Continuing training on your own data to change behaviour or format. Poor at adding fresh facts; good at teaching a shape, a tone or a narrow skill. → [Fine-tuning](stages/13-fine-tuning/README.md)

**Function calling** — See *tool calling*.

**Golden dataset** — A curated, versioned set of inputs with expected outputs or rubrics, used as the reference for every eval run. → [Golden datasets](stages/15-evals/03-golden-datasets.md)

**Gradient descent** — Repeatedly stepping parameters in the direction that reduces the loss. Every training run is this loop with embellishments. → [Ship: gradient descent](stages/02-math-and-ml-basics/07-ship-gradient-descent.md)

**Guardrail** — A check that constrains what goes in or comes out: schema validation, allowlisted tools, filters, approval gates. Defence in layers, never a single wall. → [Guardrails](stages/20-ai-security/09-guardrails.md)

**Hallucination** — Fluent output that is not supported by the input or the world. Reduced by grounding, constrained output, and refusal being an allowed answer. → [Hallucinations](stages/15-evals/05-hallucinations.md)

**Harness** — Everything around the model that makes it a system: tool registry, context assembly, loop control, budgets, retries, tracing, approval. The thing you actually engineer. → [Harness engineering](stages/11-agents/01-harness-engineering.md)

**HNSW** — A graph-based ANN index: a hierarchy of navigable small-world graphs. Fast and accurate, memory-hungry, needs tuning of `M` and `ef`. → [Vector indexing](stages/09-vector-databases/04-vector-indexing.md)

**Hybrid search** — Combining lexical (BM25) and dense results, usually fused with reciprocal rank fusion. The most reliable single upgrade to a mediocre retriever. → [Hybrid retrieval](stages/10-rag/07-hybrid-retrieval.md)

**Idempotency** — Doing the same operation twice has the same effect as doing it once. Essential for tool calls, job queues and retries. → [Job queues](stages/17-ai-backend/05-job-queues.md)

**Instruction tuning** — Fine-tuning on instruction/response pairs so a pretrained model follows directions instead of continuing text. → [Instruction tuning](stages/06-llm-fundamentals/06-instruction-tuning.md)

**JSON mode / structured outputs** — Constraining the model to emit output that matches a schema, then validating it anyway. → [Structured outputs](stages/07-llm-apis/04-structured-outputs.md)

**KV cache** — Stored keys and values from previous tokens so each new token does not recompute them. The reason generation is memory-bound, and the reason long contexts cost what they cost. → [KV cache](stages/16-inference/02-kv-cache.md)

**LangGraph** — A library for expressing agent workflows as a state graph with nodes, edges, checkpoints and interrupts. → [LangGraph](stages/12-orchestration/02-langgraph.md)

**LLM-as-judge** — Using a model to grade outputs against a rubric. Useful at scale, biased in specific ways, and must be calibrated against human labels. → [LLM as judge](stages/15-evals/02-llm-as-judge.md)

**LoRA** — Low-rank adaptation: freeze the base weights and train small low-rank matrices that are added to selected layers. Trains a tiny fraction of parameters. → [LoRA and QLoRA](stages/13-fine-tuning/02-lora-qlora-peft.md)

**Lost in the middle** — The empirical tendency for models to use information at the start and end of a long context better than information in the middle. Order your context deliberately. → [Context engineering](stages/08-prompt-and-context/08-context-engineering.md)

**Multi-agent** — Several model-driven workers with separate roles and context. Buys specialisation and parallelism, costs coordination and tokens; often a graph is the better answer. → [Multi-agent](stages/11-agents/07-multi-agent.md)

**Multi-query** — Rewriting one question into several retrieval queries and merging the results, to cover different phrasings. → [Multi-query](stages/10-rag/09-multi-query.md)

**PagedAttention** — Managing the KV cache in fixed-size blocks, like virtual memory pages, to cut fragmentation and allow more concurrent sequences. → [PagedAttention](stages/16-inference/04-pagedattention.md)

**Parent-child retrieval** — Index small chunks for precision, but hand the model the larger parent section for context. Also called small-to-big. → [Parent-child](stages/10-rag/08-parent-child.md)

**PII** — Personally identifiable information. Detect and redact before it reaches a model or a log. → [PII redaction](stages/20-ai-security/04-pii-redaction.md)

**Prefill and decode** — The two phases of generation: prefill processes the prompt in parallel (compute-bound), decode produces one token at a time (memory-bound). They explain TTFT vs tokens/sec. → [Prefill and decode](stages/16-inference/01-prefill-decode.md)

**Prompt injection** — Untrusted content in the context steering the model to do something the developer did not intend. The core security problem of LLM applications. → [Prompt injection](stages/20-ai-security/01-prompt-injection.md)

**Prompt registry** — Prompts stored as versioned artefacts with ids, changelogs and rollback, instead of strings buried in code. → [Prompt versioning](stages/18-mlops-llmops/03-prompt-versioning.md)

**Quantization** — Storing and computing with lower-precision numbers (int8, int4) to cut memory and raise throughput, at some cost in quality. → [Quantization](stages/16-inference/06-quantization.md)

**RAG (retrieval-augmented generation)** — Retrieve relevant context, put it in the prompt, generate an answer grounded in it. Freshness and citations without retraining. → [RAG architecture](stages/10-rag/01-rag-architecture.md)

**Reranking** — Reordering a retrieved shortlist with a stronger, slower scorer before it reaches the model. → [Reranking](stages/10-rag/06-reranking.md)

**Reciprocal rank fusion (RRF)** — Merging ranked lists by summing 1/(k + rank) across lists. Simple, robust, no score calibration needed. → [Hybrid retrieval](stages/10-rag/07-hybrid-retrieval.md)

**Reflection** — Having the model critique and revise its own output. Helps with format and obvious errors; does not fix missing knowledge. → [Reflection](stages/11-agents/09-reflection.md)

**RLHF** — Reinforcement learning from human feedback: train a reward model on preferences, then optimise the policy against it. The heavyweight path that DPO often replaces. → [RLHF and DPO](stages/06-llm-fundamentals/07-rlhf-dpo.md)

**Routing** — Sending each request to the cheapest model or path that can handle it, escalating on failure or low confidence. The main lever on cost. → [Model routing](stages/21-system-design/04-model-routing.md)

**Sandboxing** — Running tool side effects in a restricted environment: no shell by default, allowlisted paths and hosts, timeouts, resource caps. → [Sandboxing](stages/20-ai-security/05-sandboxing.md)

**Semantic cache** — A cache keyed by embedding similarity rather than exact string match, so paraphrases hit. Needs a similarity threshold you actually tuned. → [Caching](stages/21-system-design/06-caching.md)

**SFT (supervised fine-tuning)** — Training on input/output pairs to teach format, tone or a narrow task. → [SFT](stages/13-fine-tuning/01-sft.md)

**Speculative decoding** — A small draft model proposes tokens and the large model verifies them in one pass, raising throughput when acceptance is high. → [Speculative decoding](stages/16-inference/05-speculative-decoding.md)

**SSE (server-sent events)** — A one-way HTTP stream of text events. The simplest way to stream tokens to a browser. → [Streaming](stages/07-llm-apis/03-streaming.md)

**System instruction** — The persistent prompt that sets role, scope, constraints and output shape. The first place behaviour is defined, and the first place to look when behaviour is wrong. → [System instructions](stages/08-prompt-and-context/01-system-instructions.md)

**Temperature** — A sampling parameter controlling randomness. Lower for extraction and classification, higher for brainstorming; never a substitute for a better prompt. → [LLM APIs](stages/07-llm-apis/01-openai-anthropic.md)

**Token** — The unit models read and write, roughly a word-piece. Pricing, limits and latency are all counted in tokens. → [Tokenization](stages/06-llm-fundamentals/02-tokenization.md)

**Tool calling** — Giving the model a typed list of functions; it returns a structured request to call one, your code executes it, and the result goes back into the conversation. → [Tool calling](stages/11-agents/04-tool-calling.md)

**Trace** — A recorded tree of one request: every model call, tool call, retrieval, latency and cost. The only way to debug a multi-step system. → [Tracing](stages/18-mlops-llmops/05-tracing.md)

**TTFT (time to first token)** — Latency until the first streamed token. Dominated by prefill and queueing; the number users feel most. → [Prefill and decode](stages/16-inference/01-prefill-decode.md)

**Vector database** — A store built for embedding search: it holds vectors plus metadata, supports filtered similarity search, and maintains an ANN index. → [Vector databases](stages/09-vector-databases/README.md)

**vLLM** — A high-throughput open-model serving engine built on PagedAttention and continuous batching, with an OpenAI-compatible API. → [vLLM and SGLang](stages/16-inference/08-vllm-sglang.md)

**Zero-shot** — Asking for the task with instructions only, no examples. Cheapest, and often sufficient with a strong model and a clear output contract. → [Few-shot and CoT](stages/08-prompt-and-context/02-few-shot-cot.md)

---

Missing a term you keep meeting? Add it — see [CONTRIBUTING.md](CONTRIBUTING.md).

[← Back to the roadmap](README.md)
