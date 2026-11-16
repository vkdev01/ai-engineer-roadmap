# 7. Multimodal Assistant

> **Stage 22 · Real Projects** — one assistant that accepts a photo, a voice note and a PDF in the same conversation, and answers with grounded references back into what it was shown.
>
> **You'll be able to:** build a multimodal assistant with one message model across images, audio and documents, a router that picks a model per input type, and answers that point at regions, pages and timestamps.
>
> **Prerequisites:** [Vision-Language Models](../14-multimodal/01-vision-language.md) · [Speech to Text](../14-multimodal/03-speech-to-text.md)

## Why this exists

Users do not think in modalities. Someone photographing a leaking pipe sends the photo, then a voice note explaining it, then the warranty PDF. If your product accepts only text, they retype. If it accepts images but not audio, you have built half a feature and the conversation breaks in the middle.

The naive implementation sends every attachment to one vision-capable model, which is expensive and bad at audio, or converts everything to text first, which throws away layout and tone — a photo of a wiring diagram is not its OCR text.

Buyers are field-service apps, insurance intake, medical intake, and consumer apps where the camera is the input. They pay for "the user gets an answer without typing".

Mental model: a message is a *list of typed content blocks*, and each block is routed to the cheapest model that can handle that type — with one model at the end composing the answer.

## The mental model

```text
user turn
  |-- image block --> [vision model: describe + ground boxes]  --\
  |-- audio block --> [STT: transcript + word timings] ---------> [text-only
  |-- doc block   --> [OCR/layout: text + page refs] -----------/   reasoner]
  |-- text block -----------------------------------------------/
                                                                     |
                                                          answer + citations
                                                     (page, bbox, timestamp)
```

Attachments are *normalised to text plus references* before reasoning. The reasoner never needs to be multimodal, which makes it cheap, testable and swappable.

## Core concepts

### Scope and non-goals

In scope: image, audio and PDF input; per-block processing; a grounded answer that references regions, pages and timestamps; a conversation that mixes them; cost accounting per block type.

Non-goals: video (start with audio extracted from video), real-time streaming audio, editing images, and handwriting. Also non-goal: perfect transcription of overlapping speakers — that is [Meeting Intelligence](08-meeting-intelligence.md).

### Normalise, then reason

Define one message shape: `role` plus a list of blocks, where a block is `text`, `image`, `audio` or `document`. Each block type has a *processor* that returns `{text, references, cost}`. Images become a description plus optional bounding boxes. Audio becomes a transcript with word timings. Documents become text per page with layout hints. After normalisation, everything downstream is a text problem — retrieval, prompting, evals.

This is the decision that keeps the project small. You get one eval harness, one prompt suite, one logging path.

### Grounding means coordinates, not adjectives

"The wire is connected incorrectly" is unhelpful. "The red wire at the second terminal from the left (region 3) is loose" is actionable. Ask the vision model for bounding boxes normalised to the image, validate that each box is inside the frame and has non-zero area, and render them as overlays. For audio, reference timestamps from word timings so the UI can play the exact two seconds. Reject references that do not validate — an invalid box is worse than no box.

### Route by block, and budget per block

Pick the processor per block type and per size: a small image can go to a cheap model, a dense scanned page to a stronger one, and audio to a dedicated STT endpoint that is priced per minute rather than per token. Log cost per block so you can answer "what does one conversation cost". Latency is dominated by the slowest block, so run independent blocks concurrently and stream the answer as soon as reasoning starts.

### Data model, stack, milestones

| table | key fields |
|-------|-----------|
| `conversation` | id, user_id, channel |
| `message` | id, conversation_id, role, created_at |
| `block` | id, message_id, type, storage_key, derived_json, cost_usd |
| `reference` | id, block_id, kind, page, bbox, t_start, t_end |

Stack, loose: FastAPI, object storage for originals, `pillow` for resizing and overlays, `ffmpeg` for audio normalisation, an STT endpoint, a vision model, and a text model for reasoning. Model ids and per-minute audio prices change often — read them from the provider's docs and keep the rates in config, not in code.

Four weeks: week 1 block model and text path; week 2 image processing with validated boxes; week 3 audio with timings; week 4 documents, evals, demo.

### Eval plan and budget

- **Answer accuracy** on a task set with known answers, split by input modality so you can see which type drags the score.
- **Grounding validity** — share of references that pass geometric validation. Deterministic; must be near-total.
- **Grounding usefulness** — on a hand-read sample, does the reference point at the thing the answer is about? This needs human judgement.
- **Transcription word error rate** on your own audio sample, computed against hand transcripts.
- **p95 latency per block type** and **cost per conversation**.

Budget, illustrative: one image at roughly 1,500 tokens, one minute of audio as an STT call priced per minute plus about 200 transcript tokens, one 5-page document as about 3,000 tokens of extracted text. Reason over all of it with 4k input and 500 output tokens. Multiply by current rates from the docs; the ratio between modalities is what matters — audio is usually the biggest line item per minute, images per page, so cap attachment counts per conversation and say so in the UI.

## Code

The block model plus the per-block processor registry is the spine. This is the part that stops the codebase turning into modality-specific spaghetti.

```python
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class Reference(BaseModel):
    kind: Literal["bbox", "page", "time"]
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None   # normalised 0..1
    t_start: float | None = None
    t_end: float | None = None


class Derived(BaseModel):
    text: str
    references: list[Reference] = Field(default_factory=list)
    cost_usd: float = 0.0


class Processor(Protocol):
    async def __call__(self, storage_key: str) -> Derived: ...


REGISTRY: dict[str, Processor] = {}       # "image" | "audio" | "document"


def valid_ref(ref: Reference) -> bool:
    if ref.kind == "bbox":
        if ref.bbox is None:
            return False
        x0, y0, x1, y1 = ref.bbox
        return 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    if ref.kind == "time":
        return ref.t_start is not None and ref.t_end is not None and ref.t_end > ref.t_start
    return ref.page is not None and ref.page >= 1


async def normalise(blocks: list[dict]) -> Derived:
    """Run every non-text block concurrently; keep only valid references."""
    import asyncio

    async def one(block: dict) -> Derived:
        proc = REGISTRY[block["type"]]
        out = await proc(block["storage_key"])
        out.references = [r for r in out.references if valid_ref(r)]
        return out

    parts = await asyncio.gather(*[one(b) for b in blocks if b["type"] != "text"])
    text_blocks = [b["text"] for b in blocks if b["type"] == "text"]
    return Derived(
        text="\n\n".join(text_blocks + [p.text for p in parts]),
        references=[r for p in parts for r in p.references],
        cost_usd=sum(p.cost_usd for p in parts),
    )
```

The vision processor asks for boxes in normalised coordinates and validates them, which is what turns a description into a usable answer:

```python
async def process_image(storage_key: str) -> Derived:
    image = load_and_resize(storage_key, max_side=1024)     # cap tokens
    resp = await vision_client.messages.create(
        model=VISION_MODEL,                                  # check current model id in docs
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": encode(image)},
                {"type": "text", "text": (
                    "Describe what matters in this image. For each notable object, "
                    "return a JSON list of {label, bbox:[x0,y0,x1,y1]} with coordinates "
                    "normalised to 0..1."
                )},
            ],
        }],
    )
    return Derived(
        text=resp.text,
        references=parse_boxes(resp.text),     # invalid boxes are dropped in normalise()
        cost_usd=estimate_cost(resp.usage),
    )
```

## Real use case

A field-service app lets a technician photograph an installed unit, add a voice note ("the seal is weeping on the left"), and attach the warranty PDF. Before: the technician typed a description into a form and the office called back for photos. After: one conversation, and the reply names the part, cites the warranty page that covers it, and shows the photo with the region highlighted.

The design choice that made it work is normalisation to text plus references: the reasoner is a plain text model over transcript, description and page text, so the same eval harness scores all three input types and a model swap touches one processor.

Demo script, two minutes: send an image and a voice note in one turn, show the concurrent processing, show the overlay and the page citation, then show the cost breakdown per block. Talking points: why the reasoner stays text-only, how you validate boxes, what breaks on bad audio, and how you cap cost per conversation.

## Common mistakes

**Mistake:** sending every attachment to the most capable model — **Instead:** route per block; cheap processors for easy inputs.
**Mistake:** describing images in prose and calling it grounding — **Instead:** return validated coordinates and render them.
**Mistake:** letting the reasoner be multimodal "for flexibility" — **Instead:** normalise first; a text-only reasoner is cheaper and far easier to evaluate.
**Mistake:** ignoring image size before upload — **Instead:** resize to a max side; tokens scale with pixels and the extra detail rarely changes the answer.
**Mistake:** processing blocks serially — **Instead:** gather them; latency is the slowest block, not the sum.
**Mistake:** storing originals in the database — **Instead:** object storage plus keys, with retention rules for personal data.

## Checkpoint

1. Add a per-block cost cap and a rule that rejects an attachment above it with a helpful message. Confirm the rejection is logged.
2. Add timestamp references to the audio processor and make the answer link to the exact moment. Measure grounding usefulness on ten clips.
3. Add a fallback: if the vision model returns no valid boxes, answer from text alone and mark the answer as ungrounded. Why is that better than inventing a box?

**Self-check:** Exercise 3 is better because a wrong box teaches the user to distrust all boxes, while a labelled ungrounded answer keeps trust intact.

## Go deeper

- [Document Intelligence](02-document-intelligence.md) — the document block done properly
- [Meeting Intelligence](08-meeting-intelligence.md) — when audio has several speakers
- [Multimodal RAG](../14-multimodal/08-multimodal-rag.md) — retrieving over images and pages
- [Cost Tracking](../07-llm-apis/08-cost-tracking.md) — per-block accounting in practice

---

[← Previous: Coding Assistant](06-coding-assistant.md) · [Stage index](README.md) · [Next: Meeting Intelligence](08-meeting-intelligence.md)
