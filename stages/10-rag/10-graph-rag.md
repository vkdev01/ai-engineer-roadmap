# 10. Graph RAG

> **Stage 10 · RAG** — turning documents into entities and relations so questions that need several hops can be answered by walking a graph instead of hoping one chunk contains everything.
>
> **You'll be able to:** extract a knowledge graph from a corpus, answer multi-hop questions by traversal, and say when graph RAG is worth its cost.
>
> **Prerequisites:** [Chunking](03-chunking.md) · [Reranking](06-reranking.md) · [Embeddings](../09-vector-databases/01-embeddings.md)

## Why this exists

Vector RAG answers "find me the passage that is similar to this question". It fails on a different shape of question: *"Which suppliers of ours are also customers of the company that acquired our logistics provider?"* No single chunk contains that. The facts live in three documents, and similarity search will return whichever one happens to share vocabulary with the question.

Graph RAG changes the unit of retrieval from a passage to a *relationship*. You extract entities and edges once, at ingestion, then answer by traversing: find the company, walk its edges, collect the connected facts, and hand those to the model as context.

The mental model: **vector RAG finds a page; graph RAG follows a thread.**

## The mental model

```text
   corpus                            graph                     answer
   ------                            -----                     ------
   doc A: "Acme acquired ..."   ┌── (Acme) ──acquired──► (FastFreight)
   doc B: "FastFreight uses      │        │                       │
           Northwind for ..."    │        └──headquartered──► (Pune)
   doc C: "Northwind supplies    │                                │
           Acme with ..."        └── (Northwind) ──supplies──► (Acme)
                                          │
                                    traverse 2 hops from "Acme"
                                          ▼
                            facts: acquired FastFreight; FastFreight
                            uses Northwind; Northwind supplies Acme
```

Two hops is usually enough. Beyond three, traversal explodes and you are better off with a summarisation step or agentic retrieval.

## Core concepts

### Extraction is the whole ballgame

You get a graph by asking a model to read text and emit typed triples. Everything downstream inherits the extractor's quality: a missed edge is an unanswerable question. Two rules keep this honest — fix a small ontology of entity and relation types rather than letting the model invent them, and keep the source span for every triple so you can show evidence.

```python
from pydantic import BaseModel, Field

class Triple(BaseModel):
    subject: str
    relation: str = Field(description="one of: acquired, supplies, uses, employs, located_in")
    object: str
    evidence: str = Field(description="verbatim span from the source text")
```

### Storage: a graph plus a vector index

In practice you do not need a graph database to start. A table of edges (subject, relation, object, doc_id, evidence) in Postgres answers most traversal needs with recursive CTEs, and you keep a vector index over *entities* so you can resolve a question's phrasing to a node.

```sql
-- two-hop neighbourhood of an entity, deduplicated
WITH RECURSIVE walk AS (
  SELECT subject, relation, object, 1 AS depth
  FROM edges WHERE lower(subject) = lower($1)
  UNION
  SELECT e.subject, e.relation, e.object, w.depth + 1
  FROM edges e JOIN walk w ON lower(e.subject) = lower(w.object)
  WHERE w.depth < 2
)
SELECT DISTINCT subject, relation, object FROM walk;
```

### Community summaries, if you need global questions

Questions like "what are the main themes across these 400 reports" are not traversal questions at all. The standard answer is to cluster the graph into communities, summarise each community once at ingestion, and retrieve summaries rather than edges. That is more ingestion cost, paid once, and it converts a hopeless query into a normal retrieval query.

### Hybrid, always

Graph RAG does not replace vector RAG — it sits beside it. The reliable production shape is: retrieve chunks by hybrid search *and* expand the graph around the entities named in the question, then merge both into the context with provenance.

```python
def retrieve(question: str) -> list[dict]:
    chunks = hybrid_search(question, k=8)              # from 07-hybrid-retrieval
    entities = resolve_entities(question)              # embedding match on entity names
    facts = []
    for name in entities[:3]:
        facts += graph_walk(name, hops=2)
    return chunks + [{"text": render(f), "source": f["evidence"]} for f in facts]
```

## Code

A minimal pipeline: extract triples, store edges, answer a two-hop question.

```python
from __future__ import annotations

import json
import sqlite3

from openai import OpenAI

client = OpenAI()
DB = sqlite3.connect("graph.db")
DB.execute("CREATE TABLE IF NOT EXISTS edges (subject TEXT, relation TEXT, object TEXT, doc_id TEXT, evidence TEXT)")
DB.execute("CREATE INDEX IF NOT EXISTS i_sub ON edges(lower(subject))")

SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string", "enum": ["acquired", "supplies", "uses", "employs", "located_in"]},
                    "object": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["subject", "relation", "object", "evidence"],
            },
        }
    },
    "required": ["triples"],
}


def extract(text: str, doc_id: str) -> list[dict]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",                     # check the provider's current model list
        messages=[
            {"role": "system", "content": "Extract company relationships. Only use the fixed relation set. "
                                          "Quote the exact span as evidence. Return JSON."},
            {"role": "user", "content": text[:6000]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    triples = json.loads(resp.choices[0].message.content or "{}").get("triples", [])
    DB.executemany(
        "INSERT INTO edges VALUES (?,?,?,?,?)",
        [(t["subject"], t["relation"], t["object"], doc_id, t["evidence"]) for t in triples],
    )
    DB.commit()
    return triples


def walk(name: str, hops: int = 2) -> list[tuple]:
    sql = """
    WITH RECURSIVE w(s, r, o, d) AS (
      SELECT subject, relation, object, 1 FROM edges WHERE lower(subject) = lower(:n)
      UNION
      SELECT e.subject, e.relation, e.object, w.d + 1
      FROM edges e JOIN w ON lower(e.subject) = lower(w.o) WHERE w.d < :h
    ) SELECT DISTINCT s, r, o FROM w
    """
    return DB.execute(sql, {"n": name, "h": hops}).fetchall()
```

Then answer from facts, not from chunks, and keep the evidence so the answer can cite:

```python
def answer(question: str, entity: str) -> str:
    facts = walk(entity, hops=2)
    if not facts:
        return "I could not find that entity in the corpus."
    context = "\n".join(f"- {s} {r} {o}" for s, r, o in facts)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only these facts. If they do not answer the question, say so."},
            {"role": "user", "content": f"Facts:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""
```

TypeScript, for the retrieval endpoint:

```ts
export async function graphAnswer(question: string, entity: string): Promise<{ answer: string; facts: string[] }> {
  const res = await fetch(`${process.env.RAG_URL}/graph`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, entity, hops: 2 }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`graph rag failed: ${res.status}`);
  return (await res.json()) as { answer: string; facts: string[] };
}
```

## Real use case

A procurement team asks questions across contracts, supplier records and news clippings: "which of our tier-1 suppliers are exposed to the same logistics provider?" Vector RAG returns three unrelated passages containing the word "logistics". The team builds a graph over the same corpus — suppliers, providers, acquisitions, locations — and the question becomes a two-hop traversal that returns a short, checkable list.

The cost is honest: extraction ran once over roughly 12,000 documents, and the resulting graph had to be reviewed for entity-resolution errors (the same supplier appearing as "Northwind Ltd", "Northwind", and "NW"). A resolver step — normalise names, merge on exact alias match, flag the rest for human review — was necessary before any answer could be trusted. Their rule afterwards: **graph RAG is worth it when your questions are about relationships and your entity names are clean.**

## Common mistakes

- **Mistake:** letting the model invent relation types — **Instead:** fix an enum, and add a type only when you have found a real question that needs it.
- **Mistake:** skipping entity resolution — **Instead:** normalise and alias-merge before storing edges; duplicated entities silently halve your recall.
- **Mistake:** traversing four hops because two were not enough — **Instead:** cap at two or three, and add a summarisation step for global questions.
- **Mistake:** replacing vector retrieval with graph traversal — **Instead:** run both and merge; graph answers relationships, vectors answer "what does the document say".
- **Mistake:** storing triples without evidence — **Instead:** keep the source span and document id on every edge, or you cannot cite and cannot debug.
- **Mistake:** extracting with a cheap model on a regulated corpus without review — **Instead:** sample-audit the extractor output and report the error rate before trusting it.

## Checkpoint

1. Run the extractor over twenty documents and count triples per document. Look for the three most common entity-name variants and write the resolver rule that merges them.
2. Answer three questions that require two hops, using only traversal output. Then answer the same three with vector RAG and compare.
3. Add community summaries over a cluster of your graph and test one "what are the main themes" question that traversal cannot answer.

**Self-check:** if your two-hop answers are wrong, check entity resolution first — duplicate entities break traversal more often than missing edges do.

## Go deeper

- [Agentic RAG](11-agentic-rag.md) — when the model should choose the traversal instead
- [RAG Evaluation](12-rag-evals.md) — measuring a graph pipeline honestly
- [Hybrid Retrieval](07-hybrid-retrieval.md) — the retrieval half that graph RAG does not replace

---
[← Previous: Multi-Query Retrieval](09-multi-query.md) · [Stage index](README.md) · [Next: Agentic RAG](11-agentic-rag.md)
