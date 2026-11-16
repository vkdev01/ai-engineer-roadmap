# 4. Text-to-Speech

> **Stage 14 · Multimodal** — making the model talk, in a voice you can ship.
>
> **You'll be able to:** generate speech that streams, control pronunciation of domain words, keep a voice consistent across requests, and measure latency the way users feel it.
>
> **Prerequisites:** [3. Speech-to-Text](03-speech-to-text.md) · [3. Streaming](../07-llm-apis/03-streaming.md)

## Why this exists

Text output is not always the right interface. A delivery driver needs hands-free instructions; a caller waiting on a support line needs a voice, not a wall of text; a language-learning app needs correct pronunciation. Modern TTS is good enough that the hard part is no longer naturalness — it is *control*: saying "GSTIN" correctly, not reading "Rs. 4,250/-" as gibberish, keeping the same voice across a session, and starting to speak before the whole paragraph is generated.

The engineering shape is a pipeline, not a single call: normalise the text, chunk it into sentences, synthesise, and stream the audio so playback starts early. Each stage has a failure mode you can hear.

The mental model: **TTS is text normalisation plus a voice model plus a streaming contract.**

## The mental model

```text
raw text ──► normalise (numbers, dates, units, acronyms) ──► sentence split
                                                                 │
                            ┌────────────────────────────────────┘
                            ▼
                    synthesise per sentence ──► audio chunks ──► play / save
                            │
                     voice id + speed + format pinned in config
```

| Problem you hear | Where it comes from | Fix |
|---|---|---|
| "four two five zero" instead of "four thousand two hundred fifty" | normalisation | rewrite numbers before synthesis |
| "gee-ess-tin" for a known acronym | pronunciation | custom lexicon or phonetic spelling |
| Voice changes between turns | not pinning voice id / seed | pin the voice, reuse the session |
| Long pause before speech starts | generating whole text first | sentence-level streaming |
| Wrong language stress in a mixed sentence | mixed-script text | split by language and stitch |

## Core concepts

### Normalise before you synthesise

TTS models are trained on written text, and written text in your product is full of symbols. Expand what must be spoken: currency amounts, dates, ranges, units, abbreviations, URLs, and acronyms that are said letter by letter. Do it with a deterministic function you can unit test, not with a prompt. Keep the mapping table in version control; when a customer complains that "approx." was read as "approx-dot", the fix is one table row and a test.

### A custom lexicon beats prompt engineering for names

Most engines accept a pronunciation override — a mapping from a word to a phonetic form or a recorded alias. Use it for product names, people, and any word where a wrong stress makes you sound foreign or careless. Where no lexicon exists, spell the word phonetically in a scratch copy for synthesis only, never in the text you show the user. Test with a listening check on your ten most important words; it takes minutes and catches most embarrassment.

### Latency is time-to-first-audio, not total duration

Users judge by when sound starts. Generate per sentence and stream chunks, so a five-sentence answer begins speaking after the first sentence. Two practical rules: keep sentences short (a 60-word sentence means a long wait), and choose a streaming-capable output format (a raw PCM or chunked container) rather than a format that only finalises at the end. Measure time-to-first-audio and total duration separately, and report both.

### Voice consistency is a config value, not a vibe

Pin the voice id, the model version, speaking rate and output format in config, and record them with every generated file. If you use a cloned or custom voice, treat it as sensitive: it is a biometric-adjacent asset, needs consent on file, and must never be selectable by an unauthorised user. Caching identical sentences — greetings, disclaimers, common answers — saves money and guarantees the same audio every time.

## Code

Text normalisation with a testable table, then per-sentence synthesis with streaming. The normaliser is the part that most affects perceived quality.

```python
from __future__ import annotations

import re

UNITS = {"rs": "rupees", "inr": "rupees", "kg": "kilograms", "km": "kilometres"}
ACRONYMS = {"GSTIN": "G S T I N", "UPI": "U P I", "OTP": "O T P"}

def normalise(text: str) -> str:
    """Rewrite text into something a TTS engine reads correctly. Unit-testable."""
    for k, v in ACRONYMS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    # 4,250/- or Rs. 4250 -> "rupees four thousand two hundred fifty"
    def money(m: re.Match[str]) -> str:
        amount = int(m.group("amt").replace(",", ""))
        return f"{UNITS['rs']} {amount:,}".replace(",", " thousand ")  # placeholder
    text = re.sub(r"(?:Rs\.?|INR)\s*(?P<amt>[\d,]+)(?:/-)?", money, text, flags=re.I)
    for k, v in UNITS.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

assert "U P I" in normalise("Pay by UPI today")
print(normalise("Total Rs. 4,250/- including GSTIN charges"))
```

Synthesis per sentence with a pinned voice, yielding audio chunks so playback can start immediately.

```python
from collections.abc import Iterator

from openai import OpenAI

client = OpenAI()
VOICE = "alloy"                  # pin in config; check current voice ids in docs
TTS_MODEL = "check-current-tts-model-id"
FORMAT = "pcm"                   # streaming-friendly; convert for the client

def split_sentences(text: str, max_words: int = 30) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out, buf = [], ""
    for p in parts:
        if len((buf + " " + p).split()) > max_words and buf:
            out.append(buf); buf = p
        else:
            buf = f"{buf} {p}".strip()
    return out + ([buf] if buf else [])

def speak(text: str) -> Iterator[bytes]:
    for sentence in split_sentences(normalise(text)):
        with client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=VOICE, input=sentence, response_format=FORMAT
        ) as resp:
            for chunk in resp.iter_bytes(chunk_size=4096):
                yield chunk        # first bytes arrive after sentence one, not the whole text
```

The browser side, where the chunks become sound. Queue them so playback is continuous and handle the case where the user cancels mid-stream.

```ts
export class SpeechPlayer {
  private queue: AudioBuffer[] = [];
  private ctx = new AudioContext();
  private playing = false;
  private cancelled = false;

  async push(chunk: ArrayBuffer): Promise<void> {
    const buf = await this.ctx.decodeAudioData(chunk.slice(0));
    this.queue.push(buf);
    if (!this.playing) void this.drain();
  }

  cancel(): void {
    this.cancelled = true;
    this.queue = [];
  }

  private async drain(): Promise<void> {
    this.playing = true;
    while (this.queue.length > 0 && !this.cancelled) {
      const buf = this.queue.shift();
      if (!buf) break;
      await new Promise<void>((resolve) => {
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.ctx.destination);
        src.onended = () => resolve();
        src.start();
      });
    }
    this.playing = false;
  }
}
```

## Real use case

A field-service app gives technicians repair steps by voice while their hands are inside a machine. The first version generated the whole instruction set, then played it: a four-second silence before the first word, and a robotic reading of part numbers as a single unintelligible word.

They changed two things. Instructions are synthesised sentence by sentence and streamed, so audio starts in about the time it takes to say the first clause. Part numbers go through a deterministic normaliser that spells them digit by digit and pauses with a comma, and the ten most common parts get lexicon entries. The measured improvement they cared about was time-to-first-audio, which fell by an order of magnitude in their own tests; the qualitative one was that technicians stopped replaying instructions to work out which part was meant. Cached audio for the 30 most frequent instructions cut cost and made repeat playback identical.

## Common mistakes

- **Mistake:** synthesising raw product text — **Instead:** run a deterministic normaliser and unit-test it on real strings from your app.
- **Mistake:** generating the whole response before playing — **Instead:** sentence-split and stream; report time-to-first-audio.
- **Mistake:** leaving voice id and speed implicit — **Instead:** pin them in config and record them with every audio file.
- **Mistake:** shipping a cloned voice with no consent trail — **Instead:** keep written consent, restrict access, and never expose voice selection to unauthorised users.
- **Mistake:** caching nothing — **Instead:** cache greetings, disclaimers and frequent answers; it is cheaper and consistent.

## Checkpoint

1. Write a normaliser with at least six rules and ten unit tests drawn from real strings in your product.
2. Generate the same paragraph with and without sentence streaming and measure time-to-first-audio for both.
3. Build a ten-word pronunciation list, add lexicon entries, and do a listening check with someone who has not seen the text.

**Self-check:** "Rs. 4,250/-" read badly is a normalisation bug, not a model limitation — expand it to "rupees four thousand two hundred fifty" before synthesis and test that expansion.

## Go deeper

- Next: [5. Video and Audio Processing](05-video-audio.md).
- Streaming contracts: [3. Streaming](../07-llm-apis/03-streaming.md) and [2. WebSockets](../17-ai-backend/02-websockets.md).
- Voice agent patterns: [5. Support Agent](../22-real-projects/05-support-agent.md).

---
[← Previous: 3. Speech-to-Text](03-speech-to-text.md) · [Stage index](README.md) · [Next: 5. Video and Audio Processing](05-video-audio.md)
