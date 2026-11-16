# 9. Ship: A Document Reader with Citations

> **Stage 14 · Multimodal** — the stage project: a scanned PDF in, typed JSON out, every field traceable to a page and a region.
>
> **You'll be able to:** build and evaluate a document reader whose output a human reviewer can verify in seconds, because every value carries its evidence.
>
> **Prerequisites:** [2. OCR: Traditional and Model-Based](02-ocr.md) · [8. Multimodal RAG](08-multimodal-rag.md)

## Why this exists

Extraction demos look finished long before they are shippable. A model reads a scanned invoice, prints JSON, and everyone claps. Then a reviewer asks "where did you get 4,250?" and there is no answer, and a wrong number reaches an accounting system with no way to find it.

This project adds the two things that make extraction usable: a **schema** so the output is typed and validated, and **provenance** so every field points at a page and a rectangle. Then it adds the third thing: an **evaluation** on a labelled set, scored per field, so you know which fields you can trust and which still need a human.

The mental model: **never return a value without its evidence — a field is (value, page, region, confidence), not a scalar.**

## The mental model

```text
scanned PDF ──► render pages (PNG, fixed DPI) ──► [classic OCR] ──► words + boxes
                                                    │
                                                    ├──► [model reader] ──► field candidates
                                                    ▼
                                         locate evidence per field
                                                    │
                    {value, page, bbox, method} ────┴──► validate (schema + arithmetic)
                                                    │
                                        accept │ review (low confidence) │ null
```

| Field | Evidence needed | Validation |
|---|---|---|
| `invoice_no` | page, region | format / checksum |
| `date` | page, region | parses as a date, plausible range |
| `total` | page, region | equals sum of line amounts |
| `lines[]` | page, regions | non-empty, amounts numeric |

## Core concepts

### A pydantic schema is the contract

Define the record once, in `pydantic`, with optional fields defaulting to `None` rather than a guessed value. Validation then does two jobs: it rejects malformed model output at the boundary, and it documents for every downstream consumer exactly what may be missing. Add a field-level `confidence` and `method` (`ocr`, `model`, `reconciled`) so a reviewer sees how each value was produced — this is what lets you route only the risky fields to humans.

### Evidence means page plus rectangle, and it must be checkable

For every field, store the page index and a bounding box in page pixel coordinates at your render DPI. Then a reviewer UI can draw the rectangle over the page image and a human confirms in one glance. Where the value came from classic OCR, you have word boxes and can compute a union — exact. Where it came from a model reader, ask for the text span it read and search for it in the OCR word list to recover a box; if the span is not found, downgrade the field to `needs_review`. That fallback is what keeps provenance honest for model-extracted values.

### Validation gates, not prompt hopes

The reader should refuse rather than guess. Three gates cover most documents: schema validity (types, enums, required keys), arithmetic reconciliation (`sum(lines) == total` within a tolerance), and cross-field consistency (the date is not in the future, the currency matches the symbol). A field that fails a gate becomes `None` with `needs_review: true` — never a best guess. This is the difference between a tool an accountant will use and one they will not.

### Score per field, on a labelled set, and report the routing rate

Overall accuracy hides the field that matters. Report exact match per field, plus the share of documents that pass all gates untouched — that last number is the one operations cares about, because it is the manual workload. A reader that gets 95% of fields right but routes 60% of documents to review saves less time than one at 90% that routes 20%. Keep a labelled set of 50-100 pages per document type and re-run on every change.

## Code

The typed record with evidence, and a gate function. Note that failures produce `None` plus a review flag, never a guess.

```python
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

class Evidence(BaseModel):
    page: int                                   # 0-based page index of the render
    bbox: tuple[int, int, int, int]             # left, top, right, bottom in page pixels
    method: str = Field(pattern="^(ocr|model|reconciled)$")

class Field(BaseModel):
    value: str | float | None = None
    evidence: Evidence | None = None
    needs_review: bool = False

class LineItem(BaseModel):
    desc: str
    amount: float
    evidence: Evidence | None = None

class Invoice(BaseModel):
    invoice_no: Field = Field(default_factory=Field)
    date: Field = Field(default_factory=Field)
    total: Field = Field(default_factory=Field)
    lines: list[LineItem] = Field(default_factory=list)

def apply_gates(inv: Invoice, tol: float = 0.01) -> Invoice:
    """Reject rather than guess: failing fields become None + needs_review."""
    if inv.lines and isinstance(inv.total.value, (int, float)):
        if abs(sum(l.amount for l in inv.lines) - float(inv.total.value)) > tol:
            inv.total.needs_review = True          # arithmetic did not reconcile
            inv.total.value = None
    if isinstance(inv.date.value, str):
        try:
            parsed = date.fromisoformat(inv.date.value)
            if parsed > date.today():
                inv.date.needs_review, inv.date.value = True, None
        except ValueError:
            inv.date.needs_review, inv.date.value = True, None
    for name in ("invoice_no", "date", "total"):
        f: Field = getattr(inv, name)
        if f.value is not None and f.evidence is None:
            f.needs_review = True                  # no evidence -> not shippable
    return inv
```

Recovering a bounding box for a model-extracted value by finding its text among OCR words. This is the step that makes model output citable.

```python
def locate(value: str, words: list[Word], page: int) -> Evidence | None:
    """Find the value's text in OCR words and return the union box, else None."""
    target = value.replace(" ", "").lower()
    joined, spans, cursor = "", [], 0
    for w in words:
        joined += w.text.lower()
        spans.append((cursor, cursor + len(w.text), w))
        cursor += len(w.text)
    idx = joined.replace(" ", "").find(target)
    if idx < 0:
        return None                                # not found: downgrade to needs_review
    hits = [w for s, e, w in spans if s < idx + len(target) and e > idx]
    if not hits:
        return None
    left = min(w.box[0] for w in hits); top = min(w.box[1] for w in hits)
    right = max(w.box[0] + w.box[2] for w in hits); bottom = max(w.box[1] + w.box[3] for w in hits)
    return Evidence(page=page, bbox=(left, top, right, bottom), method="model")
```

Scoring per field and the untouched-document rate, which is the number operations will ask about.

```python
def score(gold: list[Invoice], pred: list[Invoice]) -> dict[str, float]:
    fields = ("invoice_no", "date", "total")
    out: dict[str, float] = {}
    for name in fields:
        hits = sum(1 for g, p in zip(gold, pred)
                   if getattr(g, name).value == getattr(p, name).value
                   and getattr(p, name).value is not None)
        out[name] = hits / len(gold)
    out["untouched"] = sum(1 for p in pred
                           if not any(getattr(p, n).needs_review for n in fields)) / len(pred)
    return out
```

## Real use case

A freight-forwarding company receives supplier invoices as scans and PDFs, and two clerks key them into an ERP. Their first automation called a vision model for the whole page and wrote the JSON straight to the ERP. Within a week it posted a total that appeared nowhere on the page, and the team turned it off.

The rebuild kept the model but changed the contract. The output is an `Invoice` with evidence on every field; totals must reconcile against line items; any field without a locatable bounding box is set to `None` with `needs_review`. The reviewer UI shows the page with the rectangle drawn over the value, so confirming a field is a glance rather than a search. On their labelled 80-invoice set, most documents now pass every gate untouched and the rest route to a queue that takes a fraction of the old keying time. The design decision that made it deployable was not a better model — it was refusing to return a value without evidence, so the failure mode became "clerk checks it" instead of "wrong number in the ledger".

## Common mistakes

- **Mistake:** returning a scalar with no evidence — **Instead:** `(value, page, bbox, method)` on every field, enforced by the schema.
- **Mistake:** letting the model fill missing fields — **Instead:** `None` plus `needs_review`; a gap is recoverable, a guess is not.
- **Mistake:** skipping arithmetic reconciliation — **Instead:** gate `sum(lines) == total` and route failures; it catches most invention.
- **Mistake:** reporting one overall accuracy — **Instead:** per-field exact match plus the untouched-document rate.
- **Mistake:** rendering pages at a low DPI — **Instead:** pick a DPI where the smallest field text is comfortably readable, and keep it in config so bboxes stay comparable.

## Checkpoint

1. Build the schema for one document type with evidence on every field, and make the validator reject a field that has a value but no box.
2. Run your reader on 20 labelled pages and report per-field exact match plus the untouched-document rate.
3. Deliberately break the arithmetic gate by corrupting one total, and confirm the field comes back `None` with `needs_review` instead of the wrong number.

**Self-check:** a field is only shippable when it has a value, a page, a bounding box and a passing gate — otherwise it is set to `None` and flagged for review, because a wrong number in a ledger costs more than a blank.

## Go deeper

- Next stage: [Stage 15 — Evals and Testing](../15-evals/README.md).
- A larger version: [2. Document Intelligence](../22-real-projects/02-document-intelligence.md).
- Serving this behind an API: [1. FastAPI and Flask](../17-ai-backend/01-fastapi-flask.md).

---
[← Previous: 8. Multimodal RAG](08-multimodal-rag.md) · [Stage index](README.md) · [Next: Stage 15 — Evals and Testing](../15-evals/README.md)
