# 7. Diffusion Models for Engineers

> **Stage 14 · Multimodal** — enough of the image-generation stack to run it, cost it, and make it reproducible.
>
> **You'll be able to:** explain what a sampler does, control a generation with seeds and guidance, and pick settings that give you the result you asked for twice.
>
> **Prerequisites:** [6. Multimodal Embeddings](06-multimodal-embeddings.md) · [6. Backprop and Gradients](../02-math-and-ml-basics/06-backprop-gradients.md)

## Why this exists

You will be asked to add image generation to a product: product mockups, marketing variants, placeholder art, or background removal. You do not need to derive the mathematics to ship it, but you do need to understand four controls — sampler, steps, guidance, seed — because without them the feature behaves like a slot machine and every bug report says "it came out wrong" with no way to reproduce it.

The core idea is simple. Training adds noise to real images in small steps and teaches a network to predict that noise. Generation runs it backwards: start from pure noise, and at each step ask the network what noise it thinks is there, then remove a little of it. Repeat, and structure emerges. Everything else is bookkeeping around that loop.

The mental model: **generation is iterative denoising; a sampler is the schedule for the loop, and the seed is which noise you started from.**

## The mental model

```text
training:   real image ──► add noise at step t ──► network predicts the noise
generation: pure noise z ──► for t = T..1: predict noise, subtract a bit, add a little back
                                            │
                          text prompt ──────┘ (via cross-attention, scaled by guidance)

same seed + same settings + same model version  ->  same image
```

| Control | What it changes | Typical effect |
|---|---|---|
| seed | starting noise | completely different composition |
| steps | loop length | detail and coherence, with diminishing returns |
| guidance (CFG) | prompt adherence | higher = more literal, less natural |
| sampler | update rule | speed and character, not subject |
| model version | everything | do not mix seeds across versions |

## Core concepts

### The seed is the identity of the image

The starting latent is drawn from a seeded generator. Same seed, same prompt, same model, same sampler and steps → the same image, bit for bit on the same hardware. Change any of those and it changes. So store all five with every generated asset. This is not optional hygiene: it is the only way to answer "make that one again but slightly different" or to debug a bad batch. If you use a hosted API, check whether it exposes a seed at all; if it does not, log the full request instead.

### Guidance scale trades obedience for naturalness

Classifier-free guidance runs the model twice — once with your prompt, once without — and extrapolates away from the unconditional prediction. A scale of 1 means no guidance; higher values push harder toward the text. Low guidance gives softer, more natural, sometimes off-prompt images; high guidance gives literal adherence with over-saturated colour and artefacts. Moderate values are the usual working range. If your images look plasticky and over-contrasted, guidance is the first knob to lower.

### Samplers are schedules, and steps are diminishing returns

A sampler decides how big a step to take at each noise level. Some need many steps to converge, some produce usable images in far fewer, and some are better at fine detail. Practical rule: pick one sampler, keep it fixed, and vary steps until quality stops improving for your prompt set — then halve the steps and check whether anyone notices. More steps costs proportionally more compute, so measure rather than maximising.

### Beyond text-to-image: the four modes you will actually use

Text-to-image generates from scratch. Image-to-image starts from your image plus noise at a chosen strength, which controls how much it changes — the workhorse for restyling. Inpainting regenerates only inside a mask, which is how you fix a hand or replace an object. Control adapters condition on structure (edges, depth, pose) so you keep the composition while changing the surface. All four take the same controls plus their own, and all four need the seed logged.

## Code

A local generation with every reproducibility control explicit, and a manifest written next to the output. Uses the `diffusers` library; the pipeline class name depends on the model family, so check the model card.

```python
from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stable-diffusion-family-model-id"   # verify on the model card
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

def generate(prompt: str, *, seed: int, steps: int = 28, guidance: float = 6.0,
             out: str = "out/img.png") -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)   # the identity of the image
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,          # loop length; more is slower, not always better
        guidance_scale=guidance,            # higher = more literal, more artefacts
        generator=gen,
        height=768, width=768,
    ).images[0]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    manifest = {"model": MODEL_ID, "prompt": prompt, "seed": seed,
                "steps": steps, "guidance": guidance, "sampler": pipe.scheduler.__class__.__name__,
                "size": [768, 768], "file": out}
    Path(out).with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest

print(generate("a flat illustration of a blue ceramic mug on a white desk", seed=1234))
```

Reproducing and then varying: same seed gives the same image, and image-to-image with a strength below 1 keeps the composition while changing the surface.

```python
from diffusers import AutoPipelineForImage2Image
from PIL import Image

img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)   # reuse loaded weights

def restyle(src: str, prompt: str, *, seed: int, strength: float = 0.45) -> Image.Image:
    """strength = how much noise to add: 0 keeps the original, 1 ignores it."""
    base = Image.open(src).convert("RGB")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return img_pipe(prompt=prompt, image=base, strength=strength,
                    guidance_scale=6.0, num_inference_steps=30, generator=gen).images[0]

# Same seed as before -> the same composition family, new surface treatment.
restyle("out/img.png", "the same mug rendered as a watercolour sketch", seed=1234, strength=0.5).save("out/img_sketch.png")
```

## Real use case

A D2C furniture brand generates room mockups for 400 SKUs, and their marketing team reviews batches weekly. The first version called a hosted generator with a prompt per SKU and no stored parameters. Reviewers would approve one image out of six and ask for "the second one again, warmer" — and nobody could reproduce it, because the parameters were never captured and the seed was not exposed.

They rebuilt around a manifest. Every generation writes model id, prompt, seed, steps, guidance, sampler and size next to the file; approved images become a reference set. For "same composition, warmer light", they run image-to-image from the approved image at a moderate strength with a fixed seed, so the sofa stays put and only the light changes. Guidance came down from an over-literal setting that was producing plasticky fabric texture. The measurable change was in review cycles per SKU, which fell because reviewers could ask for a variation of a specific image instead of a fresh roll of the dice.

## Common mistakes

- **Mistake:** not storing the seed and settings — **Instead:** write a manifest beside every asset; without it the image is unreproducible.
- **Mistake:** changing the model version and reusing old seeds — **Instead:** treat model version as part of the identity; regenerate reference assets after upgrades.
- **Mistake:** cranking guidance to force prompt adherence — **Instead:** lower it and improve the prompt; high guidance adds artefacts.
- **Mistake:** maximising steps by default — **Instead:** find where quality plateaus for your prompt set, then check whether half the steps is visibly worse.
- **Mistake:** using image-to-image at strength 1 to "refine" — **Instead:** strength below 1 preserves composition; above that you have text-to-image with extra steps.

## Checkpoint

1. Generate the same prompt twice with the same seed and confirm the files match. Then change only the seed and describe what changed.
2. Sweep guidance across three values on one prompt and write down what you observe about adherence and artefacts.
3. Take one approved image and produce two variations with image-to-image at different strengths, keeping the seed fixed.

**Self-check:** identical seed, prompt, sampler, step count and model version produce the same image; change any one of them and it does not.

## Go deeper

- Next: [8. Multimodal RAG](08-multimodal-rag.md).
- Where the assets live: [1. AWS S3, EC2 and Lambda](../19-cloud-devops/01-aws-s3-ec2-lambda.md).
- The product version: [7. Multimodal Assistant](../22-real-projects/07-multimodal-assistant.md).

---
[← Previous: 6. Multimodal Embeddings](06-multimodal-embeddings.md) · [Stage index](README.md) · [Next: 8. Multimodal RAG](08-multimodal-rag.md)
