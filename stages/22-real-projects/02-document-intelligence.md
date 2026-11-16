# 2. Document Intelligence

> **Stage 22 · Real Projects** — scanned PDFs and photographed forms turned into validated structured JSON, with a review queue for the fields the machine is unsure about.
>
> **You'll be able to:** build an extraction pipeline whose output schema is enforced, whose confidence is per-field, and whose errors land in a human queue instead of a database.
>
> **Prerequisites:** [OCR](../14-multimodal/02-ocr.md) · [Validation](../03-data-engineering/07-validation.md)

## Why this exists

Every operations team has a folder of documents only a human can read: scanned invoices, KYC forms, lab reports, purchase orders, claims. A model that reads them at high accuracy sounds good and is useless on its own, because nobody knows *which* fields are wrong, and a wrong invoice number in accounts payable costs more than the manual entry it replaced.

The buyers are finance and operations teams with volume and a reconciliation deadline. They do not want a chatbot over their documents. They want rows.

The mental model: extraction is a *classification of fields*, not a summarisation of pages. Each field gets a value, a source location and a confidence; the pipeline routes on confidence, and the schema is the contract with the downstream system.

## The mental model

```text
PDF/scan --> [page render] --> [OCR words + boxes] --> [layout: lines, columns, cells]
                                                             |
                                        +--------------------+--------------------+
                                        v                                         v
                              [rule extractor: regex,                             [vision-LLM
                               anchors, table geometry]                            extractor]
                                        +--------------------+--------------------+
                                                             v
                                              [merge + per-field confidence]
                                                             |
                                    +------------------------+------------------------+
                                    v                                                 v
                        [auto-accept: JSON to API]                    [review queue: crop + value]
```

Two extractors, one merge. Rules are cheap and exact; the vision model handles layout nobody wrote a rule for. The merge decides, per field, who to believe.

## Core concepts

### Scope and non-goals

In scope: PDF and image input, OCR with word boxes, table reconstruction, schema-constrained extraction, per-field confidence, a review queue with crops, an audit trail of what changed and who approved it.

Non-goals: cursive handwriting, OCR beyond the languages your engine ships, form *discovery* (you get a fixed schema per document type), and corrections learning back into a model — a rules file edited by a human is enough for v1. Write these down; the first customer request will be handwriting.

### Tables are geometry, not text

OCR gives you words with bounding boxes in no useful order. A table is a set of columns whose x-ranges stay consistent down the page, and rows you recover by clustering y-centres. Rebuilding rows from boxes is what actually breaks in practice: a value that wraps onto a second line becomes a phantom row, and a merged cell duplicates.

Practical rule: cluster words into lines by y-overlap, assign each line to a column by x-overlap with column anchors detected from the header, then merge a line into the row above when it has no cell to its left. Keep cell crop coordinates — the review UI needs them.

### Confidence is per field, and it is a routing decision

A document-level score cannot route anything. Give every field three things: the value, the extractor that produced it (`rule`, `model`, or `agreement`), and a confidence in [0,1]. Confidence rises when both extractors agree after normalisation and falls when a value fails a checksum or a cross-field rule (line items must sum to the total; an invoice date cannot follow its due date). Route on confidence: auto-accept above your floor, review in the middle, reject and re-scan below it. Calibrate the floor on a labelled sample instead of guessing.

### The schema is the contract

Model output with `pydantic` v2, not a dict. The schema constrains the model when your provider supports structured outputs, validates and normalises dates and currency, and is the thing you version. A new required field breaks the downstream system — bump the schema version and keep both shapes in the store for a while.

### Data model and stack

| table | key fields |
|-------|-----------|
| `document` | id, sha256, doc_type, pages, received_at, status |
| `page` | id, document_id, page_no, image_key, ocr_json |
| `field` | id, document_id, name, value, confidence, extractor, page_no, bbox |
| `review` | id, document_id, field_id, reviewer, before, after, approved_at |

Stack, loose: `pymupdf` to render pages, an OCR engine with box output (Tesseract locally, or a hosted OCR API — check current docs for limits), `pdfplumber` as a cross-check for born-digital PDFs, `pydantic` v2, FastAPI for the review API, and a vision-capable LLM SDK for the fallback extractor. Keep images in object storage, never in the database.

### Milestones: four weeks

- Week 1 — ingest and OCR. Given a PDF, store pages and word boxes. Success: five document types, every page has boxes.
- Week 2 — rules. Line and table reconstruction plus regex anchors for stable fields. Success: rules alone clear your floor on the easy document type.
- Week 3 — model fallback and merge. Success: fields rules miss get filled and agreement between extractors is recorded.
- Week 4 — review queue and audit. Success: a reviewer fixes a field from a crop in under ten seconds and the audit row exists.

### Eval plan and budget

Label a sample per document type — a few dozen documents is enough to start — at *field* level, not document level. Then measure:

- **Field accuracy** per field name, exact match after normalisation. Report the worst three fields, not the average; averages hide the one field that breaks the integration.
- **Auto-accept precision** — of the fields you did not send to review, how many were right. This is the error rate the business inherits, so it is the number they care about.
- **Review load** — fields per document sent to a human. Accurate but reviewing everything has saved nobody time.
- **Cost per 1,000 pages**, from your own traces.

Budget arithmetic, illustrative: a 3-page scan costs one OCR pass plus, say, one vision call per page at roughly 1,500 image tokens and 500 output tokens. At 1,000 pages that is about 1,000 vision calls; multiply by your provider's current image and text rates, and measure image tokenisation rather than assuming it, since it differs by provider. The lever that matters: run rules first and call the model only for the fields rules missed. In a mature pipeline that is a minority of pages, and it is where the cost curve flattens.

## Code

The hard part is turning unordered OCR words into table rows. This snippet clusters words into lines, detects column anchors from the header line, and assigns cells.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass
class Line:
    words: list[Word]
    y: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.x0))


def cluster_lines(words: list[Word], tol: float = 4.0) -> list[Line]:
    """Group words whose vertical centres are within tol of each other."""
    ordered = sorted(words, key=lambda w: (w.y0 + w.y1) / 2)
    lines: list[Line] = []
    for w in ordered:
        centre = (w.y0 + w.y1) / 2
        if lines and abs(centre - lines[-1].y) <= tol:
            lines[-1].words.append(w)
            lines[-1].y = sum((x.y0 + x.y1) / 2 for x in lines[-1].words) / len(lines[-1].words)
        else:
            lines.append(Line(words=[w], y=centre))
    return lines


def column_anchors(header: Line) -> list[tuple[float, float]]:
    """Each header word defines a column x-range, widened to the midpoint gaps."""
    words = sorted(header.words, key=lambda w: w.x0)
    anchors: list[tuple[float, float]] = []
    for i, w in enumerate(words):
        left = -1e9 if i == 0 else (words[i - 1].x1 + w.x0) / 2
        right = 1e9 if i == len(words) - 1 else (w.x1 + words[i + 1].x0) / 2
        anchors.append((left, right))
    return anchors


def build_rows(lines: list[Line], anchors: list[tuple[float, float]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines[1:]:                       # skip the header line
        cells = [""] * len(anchors)
        for w in line.words:
            centre = (w.x0 + w.x1) / 2
            col = next((i for i, (lo, hi) in enumerate(anchors) if lo <= centre < hi), None)
            if col is not None:
                cells[col] = f"{cells[col]} {w.text}".strip()
        # a continuation line has text only in a column that already has text
        if rows and all(c == "" or c for c in cells) and sum(bool(c) for c in cells) == 1:
            col = next(i for i, c in enumerate(cells) if c)
            if rows[-1][col]:
                rows[-1][col] = f"{rows[-1][col]} {cells[col]}"
                continue
        rows.append(cells)
    return rows
```

Then route on confidence, which is the part reviewers notice:

```python
from pydantic import BaseModel, Field

class ExtractedField(BaseModel):
    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    extractor: Literal["rule", "model", "agreement"]

AUTO_ACCEPT = 0.95      # calibrate on your labelled sample; do not copy this blindly
REVIEW_FLOOR = 0.60

def route(fields: list[ExtractedField]) -> tuple[list[ExtractedField], list[ExtractedField]]:
    accepted = [f for f in fields if f.confidence >= AUTO_ACCEPT]
    review = [f for f in fields if REVIEW_FLOOR <= f.confidence < AUTO_ACCEPT]
    return accepted, review
```

## Real use case

A logistics firm's accounts-payable team receives supplier invoices as scans and photographs. Before: two people key them in, and reconciliation breaks monthly because a mistyped amount surfaces only at payment. After: the pipeline auto-accepts header fields and line items it agrees on, and pushes only the mismatched ones — typically the total that fails the line-item sum check — into a queue that shows the crop beside the value.

The design choice that made it work is per-field routing: the team's error rate became visible and bounded, and they could widen auto-accept field by field as they gained trust. The failure mode before was a single confidence score that was either ignored or trusted wholesale.

Demo script, two minutes: drop in a scan, show the extracted JSON with confidence badges, open the review queue on a deliberately bad scan, fix one field, show the audit row and the versioned schema. Interview talking points: why tables are geometry, how you calibrated the auto-accept floor, what the merge does when rules and model disagree, and which document type you refuse to support yet.

## Common mistakes

**Mistake:** scoring the whole document instead of each field — **Instead:** route per field; the business decision is per field.
**Mistake:** trusting the vision model for tables that rules already get right — **Instead:** rules first, model as fallback; you cut cost and keep determinism where you have it.
**Mistake:** writing extracted values straight into the system of record — **Instead:** write to a staging table with provenance, and let an approval step promote rows.
**Mistake:** normalising dates and currency only downstream — **Instead:** normalise at the schema boundary, or your accuracy metric measures formatting, not extraction.
**Mistake:** storing page images in the database — **Instead:** object storage plus a key, so the review queue loads crops without bloating backups.
**Mistake:** evaluating on documents you tuned on — **Instead:** hold out a per-type sample collected after your last rule change.

## Checkpoint

1. Add a cross-field rule that rejects any invoice whose line items do not sum to the stated total, and route those to review. Check it fires on a real mismatch.
2. Compute per-field accuracy on your held-out sample and find the field with the widest gap between auto-accept precision and overall accuracy. What does that tell you to fix?
3. Add a `doc_type` classifier so one pipeline handles two document types with different schemas. Where does the schema version live?

**Self-check:** Exercise 1 routes on a *rule*, not a model score, so the confidence you assign can be 0 on failure — deterministic checks should dominate the merge.

## Go deeper

- [Multimodal Assistant](07-multimodal-assistant.md) — when documents arrive as photos and voice notes in one conversation
- [Knowledge Assistant](09-knowledge-assistant.md) — search over the extracted rows once you have them
- [Data Cleaning](../03-data-engineering/02-data-cleaning.md) — normalisation rules that survive real supplier data
- [Sandboxing](../20-ai-security/05-sandboxing.md) — running OCR and parsers on untrusted uploads safely

---

[← Previous: Research Assistant](01-research-assistant.md) · [Stage index](README.md) · [Next: Production RAG](03-production-rag.md)
