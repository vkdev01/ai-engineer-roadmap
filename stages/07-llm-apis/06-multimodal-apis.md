# 6. Multimodal APIs

> **Stage 07 · LLM APIs** — sending images, PDFs and audio as request content, not as attachments.
>
> **You'll be able to:** build a multimodal request correctly, control image cost, and handle the file-size and format limits that break most first attempts.
>
> **Prerequisites:** [Gemini and Open-Weight Models via Hosted Endpoints](02-gemini-open-weight.md) · [Context Windows](../06-llm-fundamentals/03-context-windows.md)

## Why this exists

Text-only APIs forced a preprocessing pipeline: OCR the PDF, transcribe the audio, describe the image with a separate model, then paste the result into a prompt. Each step lost information, and layout — the part of a document that carries meaning — was the first thing to go.

Multimodal APIs collapse that. You send the bytes and the model sees them. But "send the bytes" hides three real problems: images are expensive in tokens, every provider has its own encoding rules, and the model still cannot read a 200-page PDF for free.

The mental model: **media is content blocks, and every block costs tokens proportional to how much detail you ask for.**

## The mental model

```text
request messages
  user: [
    {type: text,  text: "What is wrong with this invoice?"},
    {type: image, source: bytes|url},   --> image tokens
    {type: file,  source: pdf bytes},   --> page/image tokens
    {type: audio, source: bytes}        --> audio tokens
  ]

cost = text tokens + media tokens
       media tokens ~ pixels x detail setting
```

| Media | Typical transport | Cost driver | Gotcha |
|---|---|---|---|
| Image | base64 bytes or a URL | resolution, tiling | huge images get resized, sometimes badly |
| PDF | file upload or inline bytes | pages x per-page tokens | page limits per request |
| Audio | inline bytes or upload | duration | format and sample-rate limits |
| Video | frames or a video file | frames x image tokens | usually needs sampling yourself |

## Core concepts

### Base64 versus URL versus upload

Three transports, three trade-offs. Base64 inline is simplest and works everywhere, but inflates the request by roughly a third and has a size ceiling. A URL is cheap to send but the provider must be able to fetch it, so private objects need a signed link. A dedicated upload endpoint returns a file id you can reference repeatedly — best for the same document used in many calls.

### Detail settings are a cost dial

Providers expose a resolution or detail control. High detail tiles the image and multiplies tokens; low detail downsamples to a small grid. For a receipt photo, high detail may be needed to read a total. For "is this a photo of a dog", low detail is plenty. Choosing deliberately is the single biggest lever on multimodal spend.

### Preprocess before you send

Downscale to the largest size that preserves the information you need, convert to a sane format, and crop away margins. A 4000px phone photo of a form usually gives the same answer as a 1600px version at a fraction of the tokens. Do this in code, not by hoping the provider resizes well.

### PDFs are images plus text, and providers differ

Some providers accept PDFs natively and internally render pages as images, sometimes also extracting the text layer. Others want you to render pages yourself. Native support is convenient but you lose control over DPI and page selection, which is where cost lives. For long documents, render selected pages and send them as images.

### Audio is a separate model family

Speech-capable models accept audio blocks and can transcribe, translate or answer about the clip. Limits are tighter than for images — file size, duration, supported containers. If you need timestamps or speaker labels, a dedicated speech-to-text API is usually the better tool.

## Code

An image request to both providers, plus the downscale step that most tutorials skip.

```python
from __future__ import annotations

import base64
import io
import os

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from PIL import Image

openai_client = AsyncOpenAI()
anthropic_client = AsyncAnthropic()
# Placeholders: verify current multimodal model ids in provider docs.
OPENAI_VISION_MODEL = os.environ["OPENAI_VISION_MODEL"]
ANTHROPIC_VISION_MODEL = os.environ["ANTHROPIC_VISION_MODEL"]


def shrink(path: str, max_side: int = 1568) -> tuple[bytes, str]:
    """Downscale to a sane size. Returns PNG bytes and the media type."""
    img = Image.open(path)
    img.thumbnail((max_side, max_side))     # keeps aspect ratio
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


async def openai_vision(path: str, question: str) -> str:
    data, media_type = shrink(path)
    b64 = base64.b64encode(data).decode()
    resp = await openai_client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            # data URL form; a public https URL also works
                            "url": f"data:{media_type};base64,{b64}",
                            "detail": "high",   # cost dial: low | high | auto
                        },
                    },
                ],
            }
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content or ""


async def anthropic_vision(path: str, question: str) -> str:
    data, media_type = shrink(path)
    resp = await anthropic_client.messages.create(
        model=ANTHROPIC_VISION_MODEL,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(data).decode(),
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
```

Note the ordering difference: OpenAI puts text first in the example, Anthropic puts the image first. Both work, but be consistent — position affects how the model weights the instruction.

On the Node side, a browser upload arrives as a buffer and goes out as base64. The `sharp` library does the downscale.

```ts
import sharp from "sharp";

export function toDataUrl(buf: Buffer, mediaType: string): string {
  return `data:${mediaType};base64,${buf.toString("base64")}`;
}

export async function prepareImage(buf: Buffer): Promise<string> {
  // resize before encoding: fewer pixels, fewer tokens, same answer
  const resized = await sharp(buf)
    .resize({ width: 1568, height: 1568, fit: "inside", withoutEnlargement: true })
    .png()
    .toBuffer();
  return toDataUrl(resized, "image/png");
}

export async function askAboutImage(
  dataUrl: string,
  question: string,
  model: string, // model id from config; vision support varies by model
): Promise<string> {
  const res = await fetch(`${process.env.OPENAI_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: 600,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: question },
            { type: "image_url", image_url: { url: dataUrl, detail: "high" } },
          ],
        },
      ],
    }),
  });
  const json = (await res.json()) as {
    choices: { message: { content: string | null } }[];
  };
  return json.choices[0]?.message.content ?? "";
}
```

## Real use case

A medical-billing back office receives photographed claim forms. The team's first attempt sent the full-resolution phone photos inline as base64 to a vision model. Requests regularly exceeded the size limit, and the ones that succeeded were expensive because each photo was tiled into many patches.

They added a preprocessing step: resize to a maximum side, crop the form region using a detected bounding box, and send `detail: "high"` only for the fields needing careful reading. They also switched long multi-page PDFs from native upload to per-page images, because they only needed the two pages containing line items. The failure mode before was treating a phone photo as a document — the model was capable, the request was just the wrong shape and the wrong size.

## Common mistakes

- **Mistake:** sending full-resolution photos inline. — **Instead:** downscale to the largest useful size; tokens scale with pixels.
- **Mistake:** using `detail: "high"` by default. — **Instead:** match detail to the task — classification rarely needs it.
- **Mistake:** assuming every model in a family accepts images. — **Instead:** check capability per model id in the docs before wiring the UI.
- **Mistake:** base64-encoding without checking the request size ceiling. — **Instead:** use the provider's upload endpoint for large or reused files.
- **Mistake:** sending a 60-page PDF and expecting page-level citations. — **Instead:** render the pages you need and ask about those.

## Checkpoint

1. Add a `detail` parameter to the vision function and compare the answers and token counts for one receipt at low versus high detail.
2. Build a PDF helper that renders pages 1-3 at a chosen DPI and sends them as images; report the usage numbers.
3. Send a short audio clip to a speech-capable model and ask for a summary, noting which limits you hit.

**Self-check:** you downscale the image first, base64-encode it into a data URL, and send it as an `image_url` content block alongside a text block — the same request in Anthropic's shape uses an `image` block with a base64 source.

## Go deeper

- [Vision and Language](../14-multimodal/01-vision-language.md) — how these models are trained and where they fail.
- [OCR](../14-multimodal/02-ocr.md) — when a dedicated OCR pipeline beats a vision model.
- [Speech to Text](../14-multimodal/03-speech-to-text.md) — the audio path in depth.
- [Ship: A Doc Reader](../14-multimodal/09-ship-doc-reader.md) — a full document-understanding project.

---
[← Previous: Tool Calling](05-tool-calling.md) · [Stage index](README.md) · [Next: Retries and Rate Limits](07-retries-rate-limits.md)
