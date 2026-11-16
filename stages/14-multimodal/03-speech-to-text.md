# 3. Speech-to-Text

> **Stage 14 · Multimodal** — turning calls, meetings and voice notes into text you can index, quote and evaluate.
>
> **You'll be able to:** run a transcription pipeline with timestamps and speaker labels, measure its word error rate on your own audio, and decide what to preprocess.
>
> **Prerequisites:** [1. Vision-Language Models](01-vision-language.md) · [4. Streaming](../07-llm-apis/03-streaming.md)

## Why this exists

A support call contains the answer to a customer's problem, and it is locked in audio. Speech-to-text (ASR) unlocks it — but the transcript is not the point. What you actually need is *retrievable, quotable* text: who said what, at which second, so a summary can cite it and a reviewer can check it.

Modern ASR is good enough that the naive version works in a demo and disappoints in production. The failures are consistent: domain words (product names, drug names, Indian place names) get mangled; overlapping speakers become one stream; numbers and units get written in a form your parser cannot read; and long silences produce hallucinated text in some models. All four are engineering problems you can measure and fix.

The mental model: **ASR is a lossy compression of audio into text — your job is to know exactly what it loses and add the metadata back.**

## The mental model

```text
audio ──► resample 16 kHz mono ──► VAD (drop silence) ──► ASR ──► segments
                                                                   │
                            {start, end, text, speaker?, words[]} ─┘
                                        │
                              ┌─────────┴─────────┐
                        glossary fix          index + citations
```

| Field | Why you need it |
|---|---|
| `start` / `end` | citations, playback, alignment with video |
| `text` per segment | chunking unit that respects speech |
| `speaker` | who committed to what |
| `confidence` | route low-confidence audio to a human |

## Core concepts

### Word error rate, and why it misleads

WER = (substitutions + deletions + insertions) / number of reference words. Worked example: 100 reference words, 6 substitutions, 2 deletions, 1 insertion → (6+2+1)/100 = 0.09. Two cautions. First, WER can exceed 1.0 when the system inserts a lot. Second, it weights every word equally, so mangling a customer's account number costs the same as mangling "um" — for business use, compute a domain-term error rate on your own vocabulary list as well, and report both.

### Preprocessing decisions that change the number

Resample to the rate the model expects (16 kHz mono is common), and run voice activity detection to drop silence. VAD matters twice: it reduces cost and it removes the long silences that make some models hallucinate filler text. Do not aggressively denoise unless you have measured that it helps — over-filtering removes the consonants that carry meaning. Chunk long audio on VAD boundaries rather than fixed windows, so you never split a word, and keep segment timestamps relative to the original file so citations still line up.

### Diarisation is a separate model, not a prompt

"Who spoke when" is a clustering problem: embed short windows, cluster them into speakers, then assign segments. Some APIs return it; if not, run a diarisation model alongside and merge by timestamp overlap. Always label speakers as `speaker_1`, `speaker_2` and let a human or a later step map them to roles — guessing names from content is a privacy and accuracy hazard.

### Domain vocabulary is fixed with a glossary, not with a bigger model

Collect the terms that get mangled — product names, drug names, place names, internal codes — from your own error analysis. Then apply a post-processing pass: fuzzy-match transcript tokens against the glossary and replace above a similarity threshold, or pass the glossary to the ASR model if it supports prompting/biasing. Keep the pass conservative: a wrong replacement in a medical note is worse than a miss. Log every replacement so you can audit the pass later.

## Code

A transcription pipeline with segment timestamps and a glossary correction pass. Local model for cost control; swap in a hosted API for accuracy by changing one function.

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from faster_whisper import WhisperModel

MODEL_SIZE = "small"          # pick by measuring WER on your own audio, not by reputation
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

@dataclass
class Segment:
    start: float
    end: float
    text: str

def transcribe(path: str) -> list[Segment]:
    # vad_filter drops silence, which also removes hallucinated filler segments.
    segments, info = model.transcribe(path, language=None, vad_filter=True,
                                      word_timestamps=False)
    return [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]

GLOSSARY = ["Nimbus Pay", "Kaveri Nagar", "settlement cycle", "GSTIN"]

def fix_terms(text: str, threshold: float = 0.86) -> tuple[str, list[tuple[str, str]]]:
    """Replace near-matches of glossary terms. Conservative: high threshold only."""
    changes: list[tuple[str, str]] = []
    out = text
    for term in GLOSSARY:
        for cand in re.findall(r"[A-Za-z][A-Za-z\- ]{3,25}", out):
            if cand != term and SequenceMatcher(None, cand.lower(), term.lower()).ratio() >= threshold:
                out = out.replace(cand, term, 1)
                changes.append((cand, term))
    return out, changes

segs = transcribe("call_2024_11_03.mp3")
for s in segs[:3]:
    fixed, changes = fix_terms(s.text)
    print(f"[{s.start:6.1f}-{s.end:6.1f}] {fixed}", changes or "")
```

Measuring WER on your own reference set, including a domain-term error rate. This is the only number worth quoting about your pipeline.

```python
from jiwer import wer

def domain_term_error(ref: str, hyp: str, terms: list[str]) -> float:
    """Share of glossary terms that are missing or mangled in the hypothesis."""
    missing = sum(1 for t in terms if t.lower() in ref.lower() and t.lower() not in hyp.lower())
    present = sum(1 for t in terms if t.lower() in ref.lower())
    return missing / present if present else 0.0

pairs = [("reference transcript text ...", "hypothesis text ...")]   # load your labelled set
refs = [r for r, _ in pairs]
hyps = [h for _, h in pairs]
print("WER:", round(wer(refs, hyps), 3))
print("domain term error:", round(sum(domain_term_error(r, h, GLOSSARY) for r, h in pairs) / len(pairs), 3))
```

## Real use case

A health-insurance call centre wants to search 40,000 calls for complaints about claim rejections. Their first attempt transcribed everything with a general-purpose model and indexed raw text. Search mostly worked, but agents could not trust results: the transcript said "cashless" where the caller said "Kaveri Nagar", and summaries cited timestamps that pointed at the wrong minute after their fixed-window chunking shifted offsets.

Three changes fixed it. VAD-based segmentation replaced fixed windows, so segment timestamps were accurate and citations landed on the right second. A glossary of plan names, hospital names and process terms was applied as a conservative post-pass, cutting domain-term errors sharply. And speaker labels were added so a summary could say "the agent promised X" rather than "someone said X". The measurable win was in review time, not in WER: reviewers stopped hunting for the right minute, because every quote carried a segment that actually contained it.

## Common mistakes

- **Mistake:** chunking long audio into fixed 30-second windows — **Instead:** split on VAD boundaries and keep original-file timestamps.
- **Mistake:** quoting a vendor's WER — **Instead:** measure WER and domain-term error on 50-100 clips of your own audio.
- **Mistake:** aggressive noise reduction by default — **Instead:** test it; over-filtering removes consonants and hurts accuracy.
- **Mistake:** guessing speaker names from content — **Instead:** label `speaker_N` and map roles in a separate, reviewable step.
- **Mistake:** indexing raw transcripts — **Instead:** apply a conservative glossary pass, log replacements, and index with timestamps.

## Checkpoint

1. Take five one-minute clips, write reference transcripts by hand, and compute WER with `jiwer`. Report your pipeline's number.
2. Build a 15-term glossary for a domain you know and measure domain-term error before and after your fix pass.
3. Add VAD and compare total audio duration processed and segment count against fixed windows.

**Self-check:** 6 substitutions, 2 deletions and 1 insertion against 100 reference words gives WER = 9/100 = 0.09; the same call can still fail a domain-term check if the one word wrong is an account number.

## Go deeper

- Next: [4. Text-to-Speech](04-text-to-speech.md).
- Meeting transcripts end to end: [8. Meeting Intelligence](../22-real-projects/08-meeting-intelligence.md).
- Chunking transcribed text: [3. Chunking](../10-rag/03-chunking.md).

---
[← Previous: 2. OCR: Traditional and Model-Based](02-ocr.md) · [Stage index](README.md) · [Next: 4. Text-to-Speech](04-text-to-speech.md)
