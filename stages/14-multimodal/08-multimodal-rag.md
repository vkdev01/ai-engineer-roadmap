# 8. Multimodal RAG

> **Stage 14 · Multimodal** — retrieval over documents where the answer might be in a chart, a table, or a figure caption.
>
> **You'll be able to:** choose a representation for pages with figures and tables, retrieve the right page for a question, and return an answer with a page-level citation.
>
> **Prerequisites:** [1. RAG Architecture](../10-rag/01-rag-architecture.md) · [6. Multimodal Embeddings](06-multimodal-embeddings.md)

## Why this exists

Text RAG assumes the answer is a paragraph. Real documents — annual reports, standards, equipment manuals, research papers — put the important numbers in tables and the important relationships in figures. Extract text only and you get "as shown in Figure 4" with no Figure 4, or a table flattened into a stream of numbers with no headers, which is worse than nothing because it looks like data.

Multimodal RAG is the same retrieve-then-generate loop with three additions: pages are represented in more than one form, tables are serialised into something a model can read, and every chunk keeps a page (and ideally a region) so the answer can be checked. The generation model may be text-only or a VLM; the difference is what you put in its context.

The mental model: **a page has three usable forms — its text, its structure, and its picture. Choose per document type, and always keep the page number.**

## The mental model

```text
PDF page ──┬──► text layer (if any)  ──► chunks
           ├──► layout parse         ──► tables as markdown + figures as crops
           └──► page image           ──► multimodal embedding

query ──► retrieve (text + image + metadata filter) ──► assemble context
                                                        ├─ text chunks
                                                        ├─ table markdown
                                                        └─ page image (if VLM)
                                                        └──► answer + [page N]
```

| Approach | Retrieval unit | Best for | Cost |
|---|---|---|---|
| Text-only | paragraph chunks | prose-heavy documents | lowest |
| Text + table serialisation | paragraph and table chunks | reports, standards, manuals | low |
| Multimodal embeddings | page or region vectors | figure-heavy, scanned, slide decks | medium |
| Page image + VLM at answer time | page ids | dense layout, handwriting, charts | highest |

## Core concepts

### Tables must be serialised, not extracted

A table parsed into a grid of cells is not useful to a language model until it is rendered back into a linear form. Markdown tables work well: headers repeat, rows align, and the model can read across a row. Always include the caption and the surrounding sentence, because "Table 3: Revenue by segment" is what makes the numbers meaningful. Store the table's page and bounding box alongside, so a citation points at the table, not the whole document. If a table spans pages, keep it as one chunk with a page range — splitting it produces rows with no headers.

### Choose the retrieval unit by where meaning lives

If meaning lives in sentences, chunk paragraphs. If it lives in the layout — a two-column scan, a form, a slide — chunk pages or regions. For annual reports and manuals, a good default is: page-level image embedding for recall, plus text and table chunks for precision, retrieved together and merged by page. That way a figure-only question still finds the page, and a numeric question gets the exact table.

### Citations are the whole product

An answer without a page number is a claim. With a page number it is a lead a reviewer can verify in five seconds. Enforce this in the output schema: every factual sentence must carry a `page` from the retrieved set, and the generator must be allowed to say "not found in the provided pages". Then validate: reject any citation whose page is not in the retrieved set — that single check catches most fabricated references, because a model inventing a citation usually invents a page number too.

### Figures need a caption even when you have the image

If you send page images to a VLM, ask it for what it needs: "describe the chart on page 12 including axis labels and the trend". If you send text only, pre-generate captions for figures at ingestion time and store them as retrievable text. Pre-generated captions are cheap, reviewable, and make figures searchable by a text-only pipeline — often the pragmatic choice.

## Code

Ingestion: parse a page into text chunks, serialised tables and a page embedding, all sharing a page id. Uses a layout-aware parser; swap the library if yours differs.

```python
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Chunk:
    doc_id: str
    page: int
    kind: str                 # "text" | "table" | "caption"
    text: str
    bbox: tuple[float, float, float, float] | None = None

def table_to_markdown(rows: list[list[str]], caption: str = "") -> str:
    """Serialise a parsed table so a language model can read across a row."""
    if not rows:
        return caption
    header, *body = rows
    lines = ["| " + " | ".join(c.strip() for c in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(c.strip() for c in r) + " |" for r in body]
    return (caption + "\n\n" if caption else "") + "\n".join(lines)

def ingest_page(doc_id: str, page_no: int, blocks: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for b in blocks:
        if b["type"] == "table":
            out.append(Chunk(doc_id, page_no, "table",
                             table_to_markdown(b["rows"], b.get("caption", "")), b.get("bbox")))
        elif b["type"] == "figure":
            # Pre-generated caption makes figures searchable by a text-only index.
            out.append(Chunk(doc_id, page_no, "caption", b["caption"], b.get("bbox")))
        else:
            out.append(Chunk(doc_id, page_no, "text", b["text"], b.get("bbox")))
    return out
```

Retrieval and answer assembly with citation validation. The generator sees numbered context blocks and must cite a block id; anything else is rejected.

```python
import asyncio
import json

from anthropic import AsyncAnthropic

client = AsyncAnthropic()
MODEL = "check-current-model-id"

async def answer(question: str, chunks: list[Chunk]) -> dict:
    ctx = "\n\n".join(f"[{i}] (page {c.page}, {c.kind})\n{c.text}" for i, c in enumerate(chunks))
    prompt = (
        "Answer using ONLY the context. Every factual sentence must end with the block ids "
        "you used, like [2]. If the answer is not in the context, reply exactly: NOT_FOUND.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\n"
        'Return JSON: {"answer": str, "used": [int], "pages": [int]}'
    )
    r = await client.messages.create(model=MODEL, max_tokens=800, temperature=0,
                                     messages=[{"role": "user", "content": prompt}])
    rec = json.loads("".join(b.text for b in r.content if b.type == "text"))

    # Reject citations the model could not have read: invented pages fail here.
    allowed = {c.page for c in chunks}
    rec["pages"] = [p for p in rec["pages"] if p in allowed]
    rec["grounded"] = bool(rec["pages"]) and rec["answer"] != "NOT_FOUND"
    return rec
```

## Real use case

An asset-management analyst tool answers questions over 300 listed companies' annual reports — 40 to 200 pages each, dense with segment tables and charts. Their first RAG used plain text chunks. Questions about a specific segment's margin returned the narrative section that *mentions* the table, so the model either refused or guessed, and analysts stopped using it after a week.

They re-ingested with layout awareness. Tables are serialised to markdown with their captions and kept whole, including the page range for tables that spill. Figures get pre-generated captions at ingestion, so "what does the debt maturity profile look like" retrieves the chart's caption and its page. Pages are also embedded as images for recall. Answers must cite a block id, and the page must be in the retrieved set or the citation is stripped. The change that made it usable was the table serialisation: with headers present, the model could read across a row and report a segment margin, and the analyst could click through to page 47 and see the same table.

## Common mistakes

- **Mistake:** extracting text only from table-heavy documents — **Instead:** parse layout and serialise tables to markdown with captions.
- **Mistake:** splitting a table across chunks — **Instead:** keep it whole with a page range; a headerless row is noise.
- **Mistake:** dropping the page number from chunks — **Instead:** carry `page` and `bbox` on every chunk from ingestion onward.
- **Mistake:** trusting citations without validation — **Instead:** strip any cited page that is not in the retrieved set, and allow `NOT_FOUND`.
- **Mistake:** sending every page image to a VLM — **Instead:** use page images for recall and let text and table chunks do the answering.

## Checkpoint

1. Take one page with a table, parse it, and serialise it to markdown. Ask a model a question answerable only from the table and see whether it works.
2. Add citation validation and measure how many fabricated page numbers it catches over 30 questions.
3. Compare two retrieval units — paragraph chunks versus page-level image embeddings — on ten figure-only questions, and report hit rate for each.

**Self-check:** a table parsed to cells loses the mapping between a value and its column header, so it must be rendered back into rows and columns (markdown) with its caption before a model can read it correctly.

## Go deeper

- Next: [9. Ship: A Document Reader with Citations](09-ship-doc-reader.md).
- Foundations: [2. Ingestion and Parsing](../10-rag/02-ingestion-parsing.md) and [12. RAG Evals](../10-rag/12-rag-evals.md).
- Faithfulness measurement: [4. Faithfulness and Groundedness](../15-evals/04-faithfulness.md).

---
[← Previous: 7. Diffusion Models for Engineers](07-diffusion.md) · [Stage index](README.md) · [Next: 9. Ship: A Document Reader with Citations](09-ship-doc-reader.md)
