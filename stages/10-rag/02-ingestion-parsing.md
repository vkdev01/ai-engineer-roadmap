# 2. Ingestion and Parsing

> **Stage 10 · RAG** — getting text out of real documents without silently losing the parts that matter.
>
> **You'll be able to:** parse PDFs with layout awareness, keep tables intact, detect scanned pages, and verify the output instead of assuming it.
>
> **Prerequisites:** [RAG Architecture](01-rag-architecture.md) · [Data Cleaning](../03-data-engineering/02-data-cleaning.md)

## Why this exists

Nobody's documents are clean text files. They are PDFs with two-column layouts, tables whose meaning lives in the row headers, scanned invoices, and slides where the text is in an image. A parser that extracts "all the text" in reading order will interleave two columns, flatten a table into a word salad, and return an empty string for a scanned page — all without an error.

The failure is silent, and it happens *before* everything else in RAG. A lost table cannot be chunked well, embedded well, or retrieved. This is why parsing is the first place to look when answers are confidently wrong.

The mental model: **parsing is a lossy compression of a document into text, and your job is to know exactly what you lost.**

## The mental model

```text
PDF ─┬─ has text layer? ──yes──► extract per page, in reading order
     │                            ├─ two columns? → use layout blocks, not raw stream
     │                            └─ tables?      → detect, render as markdown
     └─ no text layer (scanned) ──► OCR the rendered page
                                     └─ if OCR confidence is low → flag, do not guess

  every parsed page gets: {source, page, text, has_table, ocr_used, char_count}
  pages with char_count ≈ 0 are a BUG, not an empty page
```

## Core concepts

### Always classify each page before extracting

The first decision is per page, not per document: does this page have a text layer? Extract text with a PDF library and check the character count. A page with under roughly 50 characters is either genuinely blank or an image — render it and look. Mixing a text PDF and a scan in one pipeline is normal; assuming uniformity is how half a corpus disappears.

### Reading order is a real problem, not a detail

PDFs store text as positioned glyphs, not sentences. Two-column academic papers and three-column brochures come out interleaved if you read the raw stream. Layout-aware extraction groups text into blocks by position and then orders them, which preserves sentences. If your chunks read like "the model was trained on a large corpus of the results show that accuracy" you have a reading-order bug, not a chunking bug.

### Tables must be converted, not flattened

A table's meaning is in the relationship between cells. Flattening it to "Region Q1 Q2 North 120 145" loses which number belongs to which quarter. Convert to markdown or CSV so headers stay attached to values:

```text
| Region | Q1  | Q2  |
|--------|-----|-----|
| North  | 120 | 145 |
```

A markdown table embeds far better than the flat string, because the header words sit next to each value. If a table is too large for one chunk, split by row group and repeat the header in every piece.

### OCR is a fallback with a confidence problem

For scanned pages, render at a reasonable DPI (150 is a common starting point) and run OCR. Two cautions. OCR errors are systematic — digits and similar-looking characters — which is exactly what matters in invoices and part numbers. And OCR gives you confidence scores, so keep them: store a per-page confidence and route low-confidence pages to a human review queue instead of pretending they parsed.

### Verify with numbers, not vibes

Three cheap checks catch most ingestion bugs: total characters per document versus page count (a 40-page PDF yielding 300 characters is broken), the share of pages with zero extracted text, and a random sample of twenty chunks read by a human. Do these before you embed anything — re-embedding a fixed corpus costs real money and time.

## Code

Page-by-page parsing with PyMuPDF: classify, extract with layout awareness, detect tables, and record a per-page record you can audit.

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

@dataclass
class PageRecord:
    source: str
    page: int
    text: str
    char_count: int
    needs_ocr: bool
    has_table: bool

def parse_pdf(path: Path, min_chars: int = 50) -> list[PageRecord]:
    out: list[PageRecord] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")          # plain extraction first
            needs_ocr = len(text.strip()) < min_chars
            has_table = bool(page.find_tables().tables) if hasattr(page, "find_tables") else False
            out.append(PageRecord(source=path.name, page=i, text=text,
                                  char_count=len(text), needs_ocr=needs_ocr,
                                  has_table=has_table))
    return out

def audit(records: list[PageRecord]) -> dict[str, float]:
    total = len(records) or 1
    return {
        "pages": float(len(records)),
        "chars_per_page": sum(r.char_count for r in records) / total,
        "empty_page_share": sum(r.needs_ocr for r in records) / total,
        "table_page_share": sum(r.has_table for r in records) / total,
    }
```

Run `audit` across the corpus and look at `empty_page_share` first. A corpus at 30% is telling you it is scanned, and you need the OCR path before anything else.

Tables as markdown, with the header repeated when a table is split — this is the conversion that keeps numbers attached to their labels.

```python
from __future__ import annotations

def table_to_markdown(rows: list[list[str | None]], max_rows: int = 20) -> list[str]:
    """Convert a detected table into one or more markdown blocks, header repeated."""
    clean = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows if any(row)]
    if len(clean) < 2:
        return []
    header, body = clean[0], clean[1:]
    head = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|"
    blocks: list[str] = []
    for start in range(0, len(body), max_rows):
        chunk = body[start : start + max_rows]
        lines = [head] + ["| " + " | ".join(r) + " |" for r in chunk]
        blocks.append("\n".join(lines))
    return blocks
```

## Real use case

A procurement team built a supplier-terms assistant over 4,000 contracts. Their first parser used a plain text extraction and everything looked fine in spot checks of clean, single-column contracts. Then a user asked about payment terms for a supplier and got an answer citing a clause from a *different* section.

The audit found the cause: about a fifth of the contracts were scanned image PDFs, and their extraction had returned nothing for those pages. The pipeline had not errored — it had simply indexed the text-layer contracts and silently skipped the rest, so retrieval happily returned the nearest available chunk from a document that happened to be in the index.

They added the page classifier, an OCR fallback for image pages, and a per-page `ocr_used` flag with a confidence threshold. Low-confidence pages were queued for human review instead of being indexed. The audit numbers went into the ingest job's log, and the job now fails loudly if `empty_page_share` crosses a threshold for a new document batch.

The design choice that mattered: **per-page records with flags**, not one big text blob per document. It made a silent gap visible in a single aggregate number.

## Common mistakes

- **Mistake:** extracting a whole PDF as one string — **Instead:** keep per-page records with flags, so gaps are countable.
- **Mistake:** trusting text order from the raw stream on two-column documents — **Instead:** use layout-aware blocks, then check twenty chunks by eye.
- **Mistake:** flattening tables into space-separated words — **Instead:** convert to markdown or CSV with the header repeated on splits.
- **Mistake:** treating a scanned page as empty — **Instead:** render and OCR, and keep the confidence score.
- **Mistake:** indexing low-confidence OCR as if it were text — **Instead:** flag and review; a wrong digit is worse than a missing page.
- **Mistake:** skipping the audit because "the text looks fine" — **Instead:** measure chars per page and empty-page share for every batch.

## Checkpoint

1. Parse a 20-page PDF from your own domain and print the audit dict. Then render three pages with the lowest character counts and confirm they really are blank or scanned.
2. Find a document with a table and convert it with `table_to_markdown`. Read the output and confirm every number still sits under its column header.
3. Add an OCR fallback for pages where `needs_ocr` is true, record the OCR confidence, and compare the OCR text against a human transcription of one page. Count the errors.

**Self-check:** an empty-page share near zero with a healthy chars-per-page figure means the text layer is intact; a high share means the corpus is largely scanned and needs the OCR path before indexing.

## Go deeper

- [Chunking](03-chunking.md) — what to do with the parsed output next.
- [OCR](../14-multimodal/02-ocr.md) — the full treatment of the OCR path.
- [Data Cleaning](../03-data-engineering/02-data-cleaning.md) — normalising the text once you have it.

---
[← Previous: 1. RAG Architecture](01-rag-architecture.md) · [Stage index](README.md) · [Next: 3. Chunking](03-chunking.md)
