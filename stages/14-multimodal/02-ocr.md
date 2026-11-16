# 2. OCR: Traditional and Model-Based

> **Stage 14 · Multimodal** — getting text out of paper, reliably and cheaply, before a language model touches it.
>
> **You'll be able to:** run a classic OCR engine and a model-based reader on the same page, and choose between them per document type using measurable criteria.
>
> **Prerequisites:** [1. Vision-Language Models](01-vision-language.md) · [2. Ingestion and Parsing](../10-rag/02-ingestion-parsing.md)

## Why this exists

Optical character recognition is decades old and mostly solved for clean, printed, upright text. It is not solved for a phone photo of a crumpled receipt, a two-column scan with a stamp over the totals, or a form where the meaning lives in the layout rather than the words.

Two families of tools exist and they fail differently. Classic OCR engines (Tesseract and its descendants) detect text regions and classify characters; they are fast, cheap, deterministic, run offline, and give you word-level bounding boxes. Model-based readers (a VLM prompted to transcribe, or a document-understanding model) use layout and language context; they handle rotation, handwriting, tables and messy backgrounds, but they can *hallucinate plausible text*, are slower and cost per page.

The mental model: **classic OCR tells you what pixels look like; a model tells you what the page means — and sometimes invents what it means.**

## The mental model

| Property | Classic OCR | Model-based reader |
|---|---|---|
| Cost per page | negligible, local CPU | model call per page |
| Speed | fast | seconds |
| Bounding boxes | yes, word level | approximate or none |
| Rotated / skewed | needs preprocessing | usually fine |
| Handwriting | weak | good |
| Tables | structure lost | structure recoverable |
| Hallucination risk | none (garbage, not invention) | real |
| Determinism | yes | no |

```text
photo ──► deskew + denoise + binarise ──► classic OCR ──► words + boxes
                    │                                        │
                    └──► if low confidence or no boxes ──► model reader ──► structured text
```

## Core concepts

### Preprocessing buys more accuracy than model choice

Classic OCR fails on geometry before it fails on language. Grayscale, adaptive threshold, deskew (estimate the rotation from text-line angles and rotate back), and crop the page border. A skewed scan can lose a large share of its words; the same page deskewed reads cleanly. Do this in a fixed, testable function and you will get consistent results across scanners. Model readers tolerate skew better, but they still read better on a straightened image — preprocessing helps both.

### Bounding boxes are the reason to prefer classic OCR

Every word comes with a rectangle. That lets you do layout reasoning without a model: group words into lines by vertical overlap, group lines into columns by x-ranges, and detect tables by aligning x-coordinates across rows. It also gives you citations — the exact region a value came from — which is what turns an extraction into something auditable. A VLM will happily tell you the total is 4,250 but often cannot tell you where on the page it read it.

### Model readers win when the page's meaning is structural

Two-column academic scans, tables with merged cells, forms with labels beside values, and handwriting are model territory. Use the model to produce *structure*, not just text: ask for JSON with rows and columns, or for a labelled field set. Then validate: a total must equal the sum of its lines, a date must parse, an account number must match a checksum. Validation is what makes model-based extraction safe, because it catches invention.

### Measure with the right metric, on your own documents

Character error rate (CER) is the standard for OCR: the edit distance between predicted and true characters, divided by the true length. Worked example: true string of 100 characters, 4 substitutions and 1 deletion → CER = 5/100 = 0.05. For extraction tasks, field-level exact match is more useful than CER, because a single wrong digit in an amount is a total failure even at 0.99 CER-accuracy. Build a 50-page labelled set with the fields you care about and report both.

## Code

A classic OCR pass with preprocessing and word boxes, printing the confidence distribution so you know which pages need a model.

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

@dataclass
class Word:
    text: str
    conf: float
    box: tuple[int, int, int, int]   # left, top, width, height

def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)                       # kill scanner speckle
    # Deskew: estimate the dominant text angle and rotate back.
    coords = np.column_stack(np.where(img < 200))
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = angle if angle < 45 else angle - 90
    if abs(angle) > 0.3:
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

def read_words(path: str) -> list[Word]:
    data = pytesseract.image_to_data(preprocess(path), output_type=Output.DICT)
    out: list[Word] = []
    for i, txt in enumerate(data["text"]):
        if txt.strip() and float(data["conf"][i]) > 0:
            out.append(Word(txt, float(data["conf"][i]) / 100.0,
                            (data["left"][i], data["top"][i], data["width"][i], data["height"][i])))
    return out

words = read_words("scan_0042.png")
mean_conf = sum(w.conf for w in words) / max(len(words), 1)
print(f"{len(words)} words, mean confidence {mean_conf:.2f}")
# Route low-confidence pages to a model reader; keep the rest local and free.
needs_model = mean_conf < 0.80 or len(words) < 20
```

The model reader for the pages that failed, asked for structure rather than raw text, with a validation gate that rejects invented totals.

```python
import json
from anthropic import Anthropic

client = Anthropic()
MODEL = "check-current-model-id"

def read_structured(image_b64: str) -> dict:
    prompt = (
        "Transcribe this invoice as JSON: {\"invoice_no\": str|null, "
        "\"date\": str|null, \"lines\": [{\"desc\": str, \"amount\": number}], "
        "\"total\": number|null}. Use null when not visible. No prose."
    )
    r = client.messages.create(
        model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return json.loads("".join(b.text for b in r.content if b.type == "text"))

def total_reconciles(rec: dict, tol: float = 0.01) -> bool:
    """Arithmetic gate: invented totals fail here even when they read plausibly."""
    if rec.get("total") is None or not rec.get("lines"):
        return True                       # nothing to reconcile, not a failure
    return abs(sum(l["amount"] for l in rec["lines"]) - rec["total"]) <= tol
```

## Real use case

An accounts-payable team processes supplier invoices from hundreds of vendors, mostly printed but often scanned crooked with a stamp across the footer. Their first system sent every page to a vision model and asked for the total. It read most of them, but on roughly one page in ten the total was a number that appeared nowhere on the page — a plausible sum of nearby figures — and each one cost a manual correction plus a trust problem.

They split the work by document type. Printed, upright, single-column invoices go through classic OCR with deskew and thresholding; the word boxes let them locate the totals block by x-alignment and read it directly. Only pages where mean confidence is low, or where no totals block is found, go to the model reader — and every model-extracted total must pass the arithmetic gate or it is routed to a human. Cost per page falls because most pages never touch a model, and invented totals drop sharply because invention now fails a check instead of reaching the ledger.

## Common mistakes

- **Mistake:** running OCR on a raw skewed photo — **Instead:** deskew and threshold first; geometry loses more words than language does.
- **Mistake:** trusting a model reader's numbers with no validation — **Instead:** add arithmetic, checksum or format gates and route failures to a human.
- **Mistake:** measuring only CER when the task is field extraction — **Instead:** report field-level exact match; one wrong digit in an amount is a failure.
- **Mistake:** sending clean printed pages to an expensive model — **Instead:** route by confidence; classic OCR is free and deterministic.
- **Mistake:** discarding bounding boxes — **Instead:** keep them; they power layout grouping and citations.

## Checkpoint

1. Compute CER by hand for one page: true length, edits, ratio. Then compute it with `jiwer` or your own edit distance and confirm you match.
2. Build the confidence-based router and run it over 30 pages. Report what fraction goes to the model.
3. Add one validation gate (arithmetic or checksum) to a model-based extraction and count how many pages it catches.

**Self-check:** with 5 character edits over a 100-character true string, CER is 5/100 = 0.05, so the page is 95% correct at the character level — which still allows an amount to be wrong, which is why field-level match matters.

## Go deeper

- Next: [3. Speech-to-Text](03-speech-to-text.md).
- Where the extracted text goes: [2. Ingestion and Parsing](../10-rag/02-ingestion-parsing.md).
- The full pipeline: [9. Ship: A Document Reader with Citations](09-ship-doc-reader.md).

---
[← Previous: 1. Vision-Language Models](01-vision-language.md) · [Stage index](README.md) · [Next: 3. Speech-to-Text](03-speech-to-text.md)
