# 5. Video and Audio Processing

> **Stage 14 · Multimodal** — handling media that is minutes long, not a single image or clip.
>
> **You'll be able to:** sample frames and audio windows deliberately, estimate cost before processing, and build a pipeline that produces timestamped, searchable output from a long recording.
>
> **Prerequisites:** [3. Speech-to-Text](03-speech-to-text.md) · [1. Vision-Language Models](01-vision-language.md)

## Why this exists

A one-hour meeting is 3,600 seconds of audio and 108,000 frames of video. Sending all of it to a model is both impossible and pointless: most seconds contain no information, and the information that matters is usually in the transcript plus a handful of slides.

So video work is mostly a *sampling* problem. You decide what to look at and what to listen to, and you keep timestamps so anything you extract can be traced back to a moment. Get the sampling wrong and you either pay for silence or miss the one frame where the error message appears.

The mental model: **compress time before you spend tokens — sample by change, not by clock.**

## The mental model

```text
media file
  ├─ audio track ──► 16 kHz mono ──► VAD segments ──► ASR ──► transcript + timestamps
  └─ video track ──► scene-change frames ──► thumbnails ──► VLM captions (only key frames)

merge by timestamp ──► timeline records ──► index / summary with citations
```

| Sampling strategy | Cost | Catches |
|---|---|---|
| Fixed every N seconds | predictable | everything, wastefully |
| Scene-change detection | low | slides, cuts, screen changes |
| Motion / difference threshold | low | anything that moves |
| Audio-gated (frames near speech) | low | talking-head content |

## Core concepts

### Never decode more than you need

Video decoding is the expensive step. Use `ffmpeg` to extract the audio track in one pass and to pull frames by timestamp or by scene score rather than decoding every frame. For a one-hour recording, an audio-only pass is small; a frame pass at one frame every five seconds yields 720 images, which is still far too many to caption all of them. Choose one strategy, measure the count, and cap it.

### Scene-change detection is the cheapest useful frame sampler

`ffmpeg`'s scene score compares consecutive frames and reports how much changed. Selecting frames above a threshold gives you slide transitions and screen changes and skips static footage. Pair it with a maximum-gap rule so a static talking-head video still gets a frame every minute. This is the difference between 40 useful frames and 720 useless ones for a typical presentation.

### Speech segments are the natural chunk for long recordings

Chunk a transcript by speaker turn or by topic boundary, not by fixed token count — a meeting summary that cites a chunk spanning two unrelated agenda items is worse than no citation. Keep `(start, end)` on every chunk, and store the media id so a UI can deep-link to the moment. This one design choice is what makes meeting search feel magical instead of approximate.

### Cost control is arithmetic, and you do it first

Before processing, compute the two budgets: transcription minutes, and number of images × tokens per image. Illustratively, 60 minutes of audio is 60 transcription minutes; 40 key frames at roughly 1,000 image tokens each is 40,000 image tokens, which dwarfs a text-only summary of the same meeting. If the frame budget is the problem, drop to 10 frames around slide transitions and skip captioning static video entirely.

## Code

Extract audio and scene-change frames with `ffmpeg`, then keep only frames that are far enough apart to be distinct.

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

def extract_audio(src: str, dst: str = "work/audio.wav") -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        check=True, capture_output=True,
    )
    return dst

def scene_frames(src: str, out_dir: str = "work/frames",
                 threshold: float = 0.30, max_gap_s: float = 60.0) -> list[dict]:
    """Select frames above a scene-change score; force one every max_gap_s seconds."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # scene score is printed on stderr as lavfi.scene_score in the metadata filter
    cmd = ["ffmpeg", "-i", src, "-vf",
           f"select='gt(scene,{threshold})',metadata=print:file=-",
           "-vsync", "vfr", f"{out_dir}/f_%05d.jpg"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    times = [float(line.split("pts_time:")[1].split()[0])
             for line in proc.stderr.splitlines() if "pts_time:" in line]

    # Enforce a maximum gap so static video still yields frames.
    filled: list[float] = []
    last = 0.0
    for t in times + [max(times[-1] if times else 0.0, last)]:
        while t - last > max_gap_s:
            last += max_gap_s
            filled.append(round(last, 2))
        filled.append(round(t, 2))
        last = t
    return [{"time_s": t, "path": f"{out_dir}/f_{i + 1:05d}.jpg"} for i, t in enumerate(filled)]

print(json.dumps(scene_frames("demo.mp4")[:5], indent=2))
```

Build the timeline: transcript segments plus frame captions, merged by time so one record answers "what was said and what was on screen at 00:14:20".

```python
from dataclasses import dataclass, asdict

@dataclass
class TimelineItem:
    start: float
    end: float
    kind: str            # "speech" | "slide"
    text: str
    media_id: str

def build_timeline(segments: list[Segment], captions: list[dict], media_id: str) -> list[TimelineItem]:
    items = [TimelineItem(s.start, s.end, "speech", s.text, media_id) for s in segments]
    items += [TimelineItem(c["time_s"], c["time_s"], "slide", c["caption"], media_id) for c in captions]
    return sorted(items, key=lambda i: i.start)

timeline = build_timeline(segs, [{"time_s": 12.0, "caption": "Title slide: Q3 churn review"}], "mtg-001")
print(json.dumps([asdict(t) for t in timeline[:3]], indent=2))
```

## Real use case

A product team records 200 customer-discovery calls a year, some as audio only, some as screen-share video with a demo. Their first pipeline sent whole video files to a vision model and asked for a summary; it was slow, expensive, and summarised the presenter's face rather than the prototype on screen.

They rebuilt it as two independent passes. Audio goes to ASR with VAD, producing timestamped speaker-labelled segments. Video gets scene-change sampling, so the frames captured are the moment the demo switches screens, and only those frames are captioned — with a maximum gap so long stretches still get coverage. Both land in one timeline table keyed by call id. The result is that a product manager can search "where did the customer say the onboarding was confusing" and land on a transcript chunk, with the screen state from that minute attached. Cost fell because most calls are audio-only and never touch a vision model at all.

## Common mistakes

- **Mistake:** decoding every frame — **Instead:** sample by scene change with a maximum gap; measure the frame count first.
- **Mistake:** transcribing silence — **Instead:** VAD-gate the audio; it cuts cost and hallucinated filler.
- **Mistake:** chunking transcripts by fixed token count — **Instead:** chunk on speaker turns or topic boundaries and keep `(start, end)`.
- **Mistake:** losing the media id — **Instead:** store it with every chunk so a citation can deep-link to the moment.
- **Mistake:** captioning all frames with a VLM — **Instead:** caption only frames near speech or scene changes; arithmetic first.

## Checkpoint

1. Run scene-change extraction on a 10-minute video and compare frame count against one frame every 5 seconds. Write down the ratio.
2. Build the merged timeline for one recording and verify three citations point at the right second.
3. Compute the token budget for your pipeline for a 60-minute recording, both passes, and state which pass dominates.

**Self-check:** scene-change sampling with a maximum gap gives you the slide transitions plus coverage of static stretches, typically a small fraction of the frames a fixed 5-second sampler would produce.

## Go deeper

- Next: [6. Multimodal Embeddings](06-multimodal-embeddings.md).
- The product version: [8. Meeting Intelligence](../22-real-projects/08-meeting-intelligence.md).
- Async processing for long media: [3. Async Processing](../17-ai-backend/03-async-processing.md).

---
[← Previous: 4. Text-to-Speech](04-text-to-speech.md) · [Stage index](README.md) · [Next: 6. Multimodal Embeddings](06-multimodal-embeddings.md)
