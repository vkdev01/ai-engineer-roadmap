# 5. Haystack

> **Stage 12 · Orchestration** — pipelines and components: the explicit-wiring framework for search and RAG services, where the graph is a first-class object you can inspect.
>
> **You'll be able to:** build a typed pipeline of components, run it, and explain why an explicit wiring model helps in a search service.
>
> **Prerequisites:** [LlamaIndex](04-llamaindex.md) · [Hybrid Retrieval](../10-rag/07-hybrid-retrieval.md)

## Why this exists

Some teams do not want a framework that decides things. They want a library of well-tested components and a way to wire them that they can read in a code review. Haystack is built around exactly that: you declare components, connect named inputs and outputs, and the pipeline object validates the wiring before it runs.

That explicitness is the framework's whole personality. It is less magical than LlamaIndex and less general than LangGraph, and for a retrieval service — index, retrieve, rerank, prompt, answer, with an evaluation component on the side — it is unusually easy to reason about, because the data flow is written down.

The mental model: **a pipeline is a wiring diagram where every arrow is a named socket.**

## The mental model

```text
 query ──► [text_embedder] ──► [retriever] ──► [joiner] ──► [ranker] ──► [prompt] ──► [llm] ──► answers
             (query →          (embedding →    (merge         (reorder)   (build     (generate)
              embedding)        documents)      branches)                  prompt)

 every connection is declared:  prompt.messages ◄── ranker.documents
 pipeline.connect("ranker.documents", "prompt.documents")
```

| Idea | What it buys you |
|---|---|
| Component | One job, typed inputs and outputs |
| Pipeline | Explicit connections, validated before run |
| Socket names | A wiring diagram you can diff in a PR |
| Serialisation | A pipeline definition you can version and store |

## Core concepts

### Components are small and honest

Each component does one thing — embed, retrieve, join, rank, prompt, generate — and declares the names of its inputs and outputs. Because the names are explicit, a mismatched connection fails at build time with a message naming the socket, not at runtime with a `KeyError` three layers down. That is the practical difference from frameworks that pass a dictionary around.

### The pipeline is a graph you can inspect

`pipeline.connect("a.out", "b.in")` is the whole language. The result can be drawn, serialised to YAML, and reviewed. When someone asks "what does this service do", the answer is the graph, not a reading of five functions. For regulated environments this is a real advantage: the artefact you review is the artefact that runs.

### Branches need a joiner, and a joiner needs a policy

Running a keyword retriever and a vector retriever in parallel is two connections into a joiner, which merges the document lists. The policy matters: deduplicate by document id, decide how scores from different retrievers are combined (or keep them separate and let the ranker sort), and cap the number of documents before the prompt so the context does not explode. See [Hybrid Search](../09-vector-databases/09-hybrid-search.md).

### Reranking is a component, not a magic step

A cross-encoder ranker takes the query and the candidate documents and reorders them. Placing it after the joiner and before the prompt is the standard shape, and it is visible in the wiring — which is the point. Teams using opaque frameworks often cannot say where reranking happens.

### Evaluation lives beside the pipeline

Haystack ships evaluators as components, which makes an offline evaluation pipeline a normal pipeline: feed a labelled dataset, run the same retriever, score faithfulness or context relevance, and write a report. Building this as a second pipeline rather than a separate script keeps the two in sync — the same components run in both.

## Code

An explicit pipeline: embed the query, retrieve, join a second retriever, rerank, prompt, generate. Component class names have changed across major versions; check the installed version's docs if an import fails.

```python
from __future__ import annotations

from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

PROMPT = """Answer using only the documents below. Cite ids in square brackets.
If the documents do not answer the question, say you do not know.

Documents:
{% for d in documents %}[{{ d.meta.id }}] {{ d.content }}
{% endfor %}
Question: {{ question }}"""

def build_pipeline(store: Any, generator: Any) -> Pipeline:
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryEmbeddingRetriever(store, top_k=10))
    pipe.add_component("joiner", DocumentJoiner(sort_by_score=False))
    pipe.add_component("ranker", TransformersSimilarityRanker(top_k=5))
    pipe.add_component("prompt", PromptBuilder(template=PROMPT))
    pipe.add_component("llm", generator)

    # Explicit wiring: every arrow is a named socket, validated at build time.
    pipe.connect("retriever.documents", "joiner.documents")
    pipe.connect("joiner.documents", "ranker.documents")
    pipe.connect("ranker.documents", "prompt.documents")
    pipe.connect("prompt.prompt", "llm.prompt")
    return pipe

def answer(pipe: Pipeline, store: Any, question: str) -> dict[str, Any]:
    result = pipe.run({
        "retriever": {"query_embedding": _embed(question)},
        "ranker": {"query": question},
        "prompt": {"question": question},
    })
    return {
        "answer": result["llm"]["replies"][0],
        "documents": [d.meta.get("id") for d in result["ranker"]["documents"]],
    }
```

Second, a small custom component: the socket pattern means your own logic plugs in exactly like a library component. Here it filters out documents whose effective date is in the future.

```python
from datetime import date

@component
class ValidAsOf:
    """Drops documents whose effective_date is after the query date."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], as_of: str) -> dict[str, list[Document]]:
        cutoff = date.fromisoformat(as_of)
        kept: list[Document] = []
        for d in documents:
            raw = d.meta.get("effective_date")
            if raw is None:
                kept.append(d)                       # unknown date: keep, ranker decides
                continue
            if date.fromisoformat(raw) <= cutoff:
                kept.append(d)
        return {"documents": kept}
```

The serialised form is the reviewable artefact. A pipeline definition in YAML can be diffed, stored next to the code, and loaded at startup.

```yaml
# pipeline.yaml — generated from the code, reviewed in the PR
components:
  retriever:
    type: InMemoryEmbeddingRetriever
    init_parameters: { top_k: 10 }
  ranker:
    type: TransformersSimilarityRanker
    init_parameters: { top_k: 5 }
connections:
  - sender: retriever.documents
    receiver: joiner.documents
  - sender: joiner.documents
    receiver: ranker.documents
```

## Real use case

A bank's internal policy search had a compliance requirement: reviewers had to approve the exact data flow, not just the code. Their earlier framework-based service passed a dictionary between functions, so the flow lived in the reader's head.

Rebuilt in Haystack, the flow became a declared pipeline with named sockets, plus a `ValidAsOf` component of their own that dropped superseded circulars before ranking. The compliance review read the serialised pipeline and signed off; the superseded-circular problem — which had produced a wrong answer in an audit — became a component with a name, a test and a place in the diagram.

## Common mistakes

- **Mistake:** connecting by position or guessing socket names — **Instead:** read the component's declared outputs; the build fails loudly if you guess.
- **Mistake:** a joiner with no dedupe policy — **Instead:** deduplicate by document id and cap documents before the prompt.
- **Mistake:** reranking after the prompt is built — **Instead:** ranker before prompt builder, visible in the wiring.
- **Mistake:** evaluation as a separate script that drifts — **Instead:** build it as a second pipeline over the same components.
- **Mistake:** version upgrades without reading migration notes — **Instead:** pin versions; component names have changed across majors.

## Checkpoint

1. Add a keyword retriever as a second branch into the joiner and set a dedupe policy on document id.
2. Write a `MaxLength` component that truncates documents to 1200 characters and place it before the prompt builder.
3. Build an evaluation pipeline that runs ten labelled questions and reports how often the correct document id appears in the top five.

**Self-check:** Exercise 1 — connect both retrievers into `joiner.documents` and pass a `DocumentJoiner` configured to deduplicate by id; assert the output length is less than the sum of the inputs.

## Go deeper

- [Hybrid Retrieval](../10-rag/07-hybrid-retrieval.md) — the retrieval shapes you are wiring.
- [Reranking](../10-rag/06-reranking.md) — why the ranker sits where it sits.
- [Semantic Kernel](06-semantic-kernel.md) — a different wiring model, aimed at .NET shops.
- [RAG Evals](../10-rag/12-rag-evals.md) — what the evaluation pipeline should measure.

---

[← Previous: LlamaIndex](04-llamaindex.md) · [Stage index](README.md) · [Next: Semantic Kernel](06-semantic-kernel.md)
