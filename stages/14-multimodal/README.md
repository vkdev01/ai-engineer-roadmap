# Stage 14 — Multimodal

Most real business data is not text. It is scanned invoices, photographs, phone calls, screen recordings and PDFs full of tables. This stage teaches you to move that material into and out of models: what a vision-language model actually sees, when classic OCR beats a model, how speech and audio pipelines work, how to search across images and text in one index, and what an engineer needs to know about diffusion. It follows Stage 13 because the same data discipline applies — and it feeds Stage 15, where you learn to prove any of it works.

**Time:** roughly 14-20 hours · **You need first:** Stage 10 — RAG, and Stage 7 — LLM APIs · **You leave with:** a document reader that turns a scanned PDF into cited structured JSON

## What you will be able to do

- Explain image tokens, resolution and cost, and predict the token bill for a vision request.
- Choose between classic OCR and a model-based reader for a given document type.
- Build a speech-to-text pipeline with timestamps, and a text-to-speech pipeline with a stable voice.
- Chunk and process video and audio without transcribing it all twice.
- Index images and text in one embedding space and retrieve across modalities.
- Explain samplers, seeds and guidance well enough to run a diffusion model reproducibly.
- Build multimodal RAG over documents with figures and tables, with page-level citations.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Vision-Language Models](01-vision-language.md) | What the model sees, image tokens, resolution and cost |
| 2 | [OCR: Traditional and Model-Based](02-ocr.md) | Classic engines versus model readers, and when each wins |
| 3 | [Speech-to-Text](03-speech-to-text.md) | Transcription, timestamps, diarisation, word error rate |
| 4 | [Text-to-Speech](04-text-to-speech.md) | Voices, streaming audio, pronunciation control |
| 5 | [Video and Audio Processing](05-video-audio.md) | Frame sampling, audio windows, cheap pipelines |
| 6 | [Multimodal Embeddings](06-multimodal-embeddings.md) | One vector space for images and text, cross-modal search |
| 7 | [Diffusion Models for Engineers](07-diffusion.md) | Samplers, seeds, guidance, and what you control |
| 8 | [Multimodal RAG](08-multimodal-rag.md) | Documents with figures and tables, cited answers |
| 9 | [Ship: A Document Reader with Citations](09-ship-doc-reader.md) | Scanned PDF to structured JSON with page citations |

## The shipping project

[09-ship-doc-reader.md](09-ship-doc-reader.md) — you build a reader that takes a scanned PDF, extracts text and tables, returns a typed JSON record, and cites the page and bounding region for every field, then prove the citations are right on a labelled set.

## How to know you are done

- You can estimate the token cost of a vision request from image dimensions before you send it.
- Your pipeline states, per document type, whether it uses classic OCR, a model, or both — and why.
- Every extracted field in your reader carries a page number and a region, not just a value.
- You have measured transcription quality with a word error rate on your own audio, not a vendor claim.
- Your multimodal index returns the right page for a text query and the right page for an image query.

---
[← Back to roadmap](../../README.md) · [Next stage: Evals and Testing](../15-evals/README.md)
