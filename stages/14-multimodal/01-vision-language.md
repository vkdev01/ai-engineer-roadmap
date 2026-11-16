# 1. Vision-Language Models

> **Stage 14 · Multimodal** — the model that reads an image and answers about it, and the cost model behind it.
>
> **You'll be able to:** explain what a vision-language model receives, predict how many tokens an image costs, and choose resolution deliberately instead of by accident.
>
> **Prerequisites:** [6. Multimodal APIs](../07-llm-apis/06-multimodal-apis.md) · [2. Tokenization](../06-llm-fundamentals/02-tokenization.md)

## Why this exists

Text models read tokens. Images are pixels. A vision-language model (VLM) bridges the gap with a small encoder that turns the image into a sequence of vectors, which are then projected into the same space as text tokens and fed to the language model as if they were tokens. The language model never "sees" the picture; it reads a summary of patches, exactly like it reads words.

That has three consequences you will meet on day one. Cost scales with image resolution, because more patches means more tokens. Small text in a downscaled image is simply gone — no prompt recovers it. And the model can only describe what the encoder preserved, so cropping beats zooming-by-instruction.

The mental model: **an image becomes a fixed-length-ish block of pseudo-tokens; everything else follows from how many.**

## The mental model

```text
image (H x W)
   │  resize / tile to model's expected grid
   ▼
patch encoder  ──►  N vectors  ──►  projector  ──►  N pseudo-tokens
                                                    │
text prompt ──► tokens ─────────────────────────────┴──► language model ──► answer
```

| You control | Effect on tokens | Effect on accuracy |
|---|---|---|
| Image dimensions | roughly quadratic | small text becomes readable |
| Tiling / high-res mode | multiplies token count | big win for dense documents |
| Number of images | linear | more context, more cost |
| Downscaling | large saving | silent loss of fine detail |

## Core concepts

### Patches are the unit of vision

The encoder splits the image into a grid of patches — commonly 14 or 16 pixels per side — and each patch becomes one vector. A 1024×1024 image at 16-pixel patches gives a 64×64 grid, so 4,096 patches. Because both dimensions scale, doubling resolution quadruples the patch count. Providers then compress, pool or cap that sequence, which is why their token counts differ from a naive patch count. Check the provider's docs for the current rule; treat any number you compute yourself as an estimate.

### Resolution is a budget decision, and downscaling is lossy in a specific way

Default pipelines resize to a square-ish max dimension, which is right for photographs of scenes and wrong for a photographed invoice. When text is the payload, you want either a higher resolution mode, tiling (the image split into overlapping tiles, each encoded, plus a downscaled overview), or your own crop of the region that matters. The failure is silent: the model answers confidently about a number it read from three blurry pixels. When accuracy on documents matters, always render at a size where the smallest text you care about is at least a dozen pixels tall.

### Cost arithmetic you can do before sending

Illustrative: if a provider charges image tokens proportional to patches at roughly `(H/16) × (W/16)` with a cap, then a 2,000×3,000 phone photo of a form is about 23,000 patches before capping — far more than most caps allow, so it gets downscaled and the fine print dies. Two ways out: crop the region of interest and send that (a 600×400 crop is about 940 patches, cheap and sharp), or send the whole page in a high-resolution/tiled mode and accept the token bill. Compute both, then decide. Never send a full-resolution photo and hope.

### Order and instruction placement matter more than you expect

Put the image before the question in the message, and refer to it explicitly ("in the table in image 1"). With multiple images, number them in the text and in the order you attach them; models mix up unlabelled images surprisingly often. Ask for the answer in a schema when you need to parse it — a VLM asked for free text will write a paragraph around the number you wanted.

## Code

One request, with the image size computed first so you know what you are paying for. The token estimate is a planning figure, not the provider's exact count.

```python
from __future__ import annotations

import base64
import io
import math

from PIL import Image
from anthropic import Anthropic

client = Anthropic()
MODEL = "check-current-model-id"   # verify the vision model id in provider docs

def estimate_image_tokens(img: Image.Image, patch: int = 16) -> int:
    """Planning estimate only: patches before the provider's own cap/compression."""
    return math.ceil(img.width / patch) * math.ceil(img.height / patch)

def prepare(path: str, max_side: int = 1568) -> tuple[str, int, tuple[int, int]]:
    """Downscale to a sane max side, re-encode as PNG, and report the estimate."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))              # keeps aspect ratio
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), estimate_image_tokens(img), img.size

b64, est_tokens, size = prepare("invoice_page1.png")
print(size, "estimated image tokens:", est_tokens)

resp = client.messages.create(
    model=MODEL,
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "Return JSON: {\"invoice_no\": str, \"total\": str, \"currency\": str}. "
                                     "If a field is not visible, use null. No prose."},
        ],
    }],
)
print("".join(b.text for b in resp.content if b.type == "text"))
```

A crop-first helper for document images. Cropping beats downscaling when you know roughly where the field lives — and it costs a fraction of a full-page high-resolution request.

```python
def crop_region(path: str, box: tuple[int, int, int, int]) -> str:
    """box = (left, top, right, bottom) in pixels. Returns base64 PNG of the crop."""
    img = Image.open(path).convert("RGB").crop(box)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# A 700x260 strip around the totals block: sharp text, small token bill.
totals_b64 = crop_region("invoice_page1.png", (100, 1150, 800, 1410))
```

## Real use case

A warranty-claims team receives photos of damaged appliances and a photo of the purchase receipt. Their first implementation sent all images at full resolution and asked for the model to "read the receipt and describe the damage". It worked on clean photos and failed on anything shot at an angle: receipt dates came back as plausible wrong dates, and the damage description drifted into generic language.

The fix was structural, not a better prompt. Receipts are text payloads, so they are cropped to the receipt region and sent in a high-resolution mode; the appliance photos are downscaled because only coarse visual features matter. The prompt asks for a JSON record with `null` for anything not visible. Before, roughly a third of receipts needed manual re-entry because of misread dates; after the crop-plus-resolution change, most come through without review, and — the important part — the ones that fail return `null` rather than a confident guess, which is what made the team willing to trust the automation.

## Common mistakes

- **Mistake:** sending full-resolution phone photos — **Instead:** crop the region that carries the text, or use a documented high-resolution mode.
- **Mistake:** assuming the model reads small text in a downscaled image — **Instead:** check the rendered pixel height of the smallest text you need; if under about 12 px, crop or render again.
- **Mistake:** asking for prose when you need a field — **Instead:** request JSON with explicit `null` for missing values, then validate the schema.
- **Mistake:** attaching several images without labels — **Instead:** number them in the text and attach in the same order.
- **Mistake:** quoting an exact token count from a formula — **Instead:** treat computed tokens as an estimate and check the provider's current image token rule.

## Checkpoint

1. Take a document photo, compute the patch estimate at full size and at a 1,568 px max side, and write down the ratio.
2. Ask the same question of a full-page image and of a tight crop of the answer region. Compare accuracy and estimated tokens.
3. Build a schema-validated extraction for three fields, and confirm missing fields come back as `null` rather than invented.

**Self-check:** token cost grows roughly with width times height divided by patch area squared, so halving each side cuts the estimate to about a quarter — and also removes most small text.

## Go deeper

- Next: [2. OCR: Traditional and Model-Based](02-ocr.md) — for when text is the whole payload.
- API shapes: [6. Multimodal APIs](../07-llm-apis/06-multimodal-apis.md).
- Embedding images for search: [6. Multimodal Embeddings](06-multimodal-embeddings.md).

---
[← Previous: Stage 13 — Fine-Tuning](../13-fine-tuning/README.md) · [Stage index](README.md) · [Next: 2. OCR: Traditional and Model-Based](02-ocr.md)
