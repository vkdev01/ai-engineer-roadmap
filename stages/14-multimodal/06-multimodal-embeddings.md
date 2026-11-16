# 6. Multimodal Embeddings

> **Stage 14 · Multimodal** — one vector space where a photo and a sentence can be compared.
>
> **You'll be able to:** embed images and text into a shared space, retrieve across modalities, and set similarity thresholds that survive real queries.
>
> **Prerequisites:** [1. Embeddings](../09-vector-databases/01-embeddings.md) · [2. Similarity Metrics](../09-vector-databases/02-similarity-metrics.md)

## Why this exists

Text search cannot find a photo of a cracked screen. Keyword metadata only works if someone labelled it, and nobody labels 50,000 images. A dual-encoder model — the family usually called CLIP-style — trains an image encoder and a text encoder so that a photo of a red bicycle and the sentence "a red bicycle" land close together in the same vector space. That single property gives you search by description, clustering without labels, and deduplication.

The trade-off you must internalise: these models are trained to match *gist*, not detail. They are excellent at "is this a kitchen or a warehouse" and unreliable at "which invoice number is printed on this page". Detail work belongs to OCR or a VLM; gist retrieval belongs to embeddings. Mixing the two is the most common design error in image search.

The mental model: **a shared space where distance means "these are about the same thing", not "these look identical".**

## The mental model

```text
image ──► image encoder ──► vec_i ─┐
                                   ├─ same d dimensions, cosine-comparable
text  ──► text encoder  ──► vec_t ─┘

query "cracked phone screen" ──► vec_q ──► top-k nearest vec_i ──► image ids
```

| Query type | Good at | Bad at |
|---|---|---|
| "a dog on a beach" | scene gist | specific breeds |
| "invoice with a red stamp" | coarse appearance | reading the amount |
| "chart showing a decline" | chart-like images | the axis labels |
| "photo taken in Mumbai" | maybe | almost certainly not |

## Core concepts

### Two encoders, one space, and the training objective that makes it work

The model is trained contrastively: for a batch of image-text pairs, push each image's vector toward its own caption and away from every other caption in the batch. That is why the space is comparable across modalities and why it captures whatever the training captions described — usually objects, scenes and actions, rarely text content or fine attributes. A consequence worth knowing: a model trained on short captions responds badly to long queries. Query with a phrase, not a paragraph.

### Normalise, then use cosine — and know what a score means

Most of these models are trained with a normalised dot product, so normalise both vectors and use cosine similarity. The absolute value is not calibrated: 0.30 may be a strong match for one model and noise for another. Do not hardcode a threshold from a blog post. Instead, build a labelled set of 50 queries with known correct images, plot the score distribution for correct and incorrect matches, and pick the threshold where precision is acceptable for your product. Also note that image-image and text-image scores sit on different scales; never reuse one threshold for both.

### Chunk images the way you chunk text

A full-page scan embedded as one vector loses the paragraph on page two. If your images are documents or screenshots, crop into regions — a page into its blocks, a screenshot into panels — embed each region, and store the parent id. Then a query returns a *region*, which is a much better citation than "somewhere in this 40-page PDF". For photographs, one vector per image is usually right; for screenshots of a UI, embed the visible text via OCR *and* the image, and search both indexes.

### Hybrid beats pure vector for real products

Pair the vector index with whatever structured metadata you have: capture date, camera, document type, language, customer id. Filter first, then rank — a query restricted to "invoices from supplier X" is far more accurate than the same query over everything. This is the same hybrid pattern as text RAG, and it costs one extra field in your schema.

## Code

Embed a folder of images and run a text query against them. Note the explicit normalisation and the score printed alongside each hit, so you can calibrate instead of guessing.

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_ID = "clip-model-id-from-your-registry"   # verify the id; CLIP-family models differ
model = SentenceTransformer(MODEL_ID)

def embed_images(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    vecs = model.encode(imgs, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
    return [str(p) for p in paths], vecs.astype(np.float32)

paths, mat = embed_images(sorted(Path("photos").glob("*.jpg")))
print(mat.shape, "vectors, each L2-normalised -> dot product == cosine")

def search(query: str, k: int = 5, min_score: float = 0.24) -> list[tuple[str, float]]:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = mat @ q                                   # cosine, because both are unit norm
    order = np.argsort(-scores)[:k]
    return [(paths[i], float(scores[i])) for i in order if scores[i] >= min_score]

for p, s in search("cracked phone screen close up"):
    print(f"{s:.3f}  {p}")
```

Calibrating the threshold from labelled pairs instead of trusting a default. This is the step that decides whether your search feels good or random.

```python
def calibrate(pairs: list[tuple[str, str, bool]], min_precision: float = 0.9) -> float:
    """pairs = (query, image_path, is_relevant). Returns the lowest threshold meeting precision."""
    qs = model.encode([q for q, _, _ in pairs], convert_to_numpy=True, normalize_embeddings=True)
    imgs = model.encode([Image.open(p).convert("RGB") for _, p, _ in pairs],
                        convert_to_numpy=True, normalize_embeddings=True)
    scores = np.array([float(qs[i] @ imgs[i]) for i in range(len(pairs))])
    labels = np.array([rel for _, _, rel in pairs])

    for thr in np.arange(0.10, 0.50, 0.01):
        predicted = scores >= thr
        tp = int((predicted & labels).sum())
        precision = tp / max(int(predicted.sum()), 1)
        if precision >= min_precision and tp >= 10:
            return float(thr)
    return 0.50      # nothing met the bar: your set is too small or the model is wrong
```

The client side, where a search UI asks the API for matches and renders the score. Keep the threshold server-side so you can change it without a redeploy.

```ts
interface ImageHit {
  imageId: string;
  page?: number;
  score: number;
  thumbUrl: string;
}

export async function searchImages(query: string, k = 12): Promise<ImageHit[]> {
  const res = await fetch(`/api/image-search?q=${encodeURIComponent(query)}&k=${k}`);
  if (!res.ok) throw new Error(`image search failed: ${res.status}`);
  const data = (await res.json()) as { hits: ImageHit[] };
  // Server already applied the calibrated threshold; show score only in debug builds.
  return data.hits;
}
```

## Real use case

An insurance claims team receives 30,000 damage photos a month, and adjusters need to find comparable past claims. Keyword search on the free-text notes worked only when the notes were good, which was most of the time not.

They embed each photo and each note, plus a structured filter on vehicle make, claim type and region. A query like "rear bumper dent on a white hatchback" filters to the right make and claim type first, then ranks by cosine similarity. The design choice that mattered was calibration: their first launch used a threshold copied from a tutorial and returned hundreds of loose matches, so adjusters stopped trusting it. They labelled 120 query-photo pairs, found the threshold that held precision where they needed it, and accepted lower recall — a search that returns five right answers beats one that returns fifty mixed ones. They also learned the boundary the hard way: adjusters tried searching for policy numbers in the photos, which embeddings cannot read; that request now routes to OCR.

## Common mistakes

- **Mistake:** using a long paragraph as a query — **Instead:** phrase-length queries; these models are trained on short captions.
- **Mistake:** copying a similarity threshold from a tutorial — **Instead:** calibrate on 50-120 labelled pairs and re-calibrate when you change the model.
- **Mistake:** one vector per multi-page document — **Instead:** embed regions or pages and store the parent id for citations.
- **Mistake:** expecting embeddings to read text in images — **Instead:** OCR the text and index it separately; embeddings give you gist.
- **Mistake:** no metadata filter — **Instead:** filter by type, date or owner before ranking; it is cheap accuracy.

## Checkpoint

1. Embed 200 images and 20 queries you wrote yourself, and hand-label which pairs are relevant.
2. Run the calibration function and report the threshold you get at your chosen precision. State the recall you gave up.
3. Add a metadata filter to one query and measure precision before and after on the same labelled set.

**Self-check:** the score is a calibrated-by-you ranking signal, not a probability, so the threshold must come from your labelled pairs and be re-derived whenever you change the embedding model.

## Go deeper

- Next: [7. Diffusion Models for Engineers](07-diffusion.md).
- Index mechanics: [5. ANN Search](../09-vector-databases/05-ann-search.md) and [9. Hybrid Search](../09-vector-databases/09-hybrid-search.md).
- Applied to documents: [8. Multimodal RAG](08-multimodal-rag.md).

---
[← Previous: 5. Video and Audio Processing](05-video-audio.md) · [Stage index](README.md) · [Next: 7. Diffusion Models for Engineers](07-diffusion.md)
