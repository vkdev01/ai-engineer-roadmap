# 8. Meeting Intelligence

> **Stage 22 · Real Projects** — a pipeline that turns a recorded meeting into an attributed transcript, a decision log and a list of action items with owners.
>
> **You'll be able to:** merge speaker diarisation with word-level transcription correctly, extract decisions and actions with evidence timestamps, and measure how often your action items match what humans wrote.
>
> **Prerequisites:** [Speech to Text](../14-multimodal/03-speech-to-text.md) · [Video and Audio](../14-multimodal/05-video-audio.md)

## Why this exists

Nobody wants a transcript. People want to know what was decided, who owes what, and where it was said — so they can skip the recording. A transcript without speaker attribution is close to useless for that: "we'll ship Friday" means something different from the account manager than from the engineer.

The pipeline has three genuinely hard parts and one boring one. Hard: merging diarisation segments (which come from an audio-only model, as coarse time ranges with anonymous speaker labels) with word-level transcription; resolving "she'll handle it" to a person; and telling a decision from a discussion. Boring: everything else.

Buyers are sales teams (call summaries), engineering managers (standups and reviews), and consultancies (client meetings). They pay to stop writing minutes.

Mental model: diarisation gives you *who spoke when*; transcription gives you *what was said*; the join is a time-interval problem, and the output is structured facts with timestamps as evidence.

## The mental model

```text
audio ──> [normalise: mono, 16k, denoise] ──┬──> [STT: words + timings]
                                            └──> [diarisation: turns + speaker labels]
                                                        |
                                    [join: assign each word to the turn
                                     with max temporal overlap]
                                                        |
                                    [segments: speaker + text + t_start/t_end]
                                                        |
                        +-------------------------------+-------------------------------+
                        v                               v                               v
                [summary by section]           [decision log]                 [action items]
                                                                        owner, what, when, evidence
                                                        |
                                            [human confirm step before export]
```

The join is the part that fails silently: get it wrong and every quote is attributed to the wrong person, which is worse than no attribution.

## Core concepts

### Scope and non-goals

In scope: one audio or video file, speaker-attributed transcript, section summary, decision log, action items with owner and due date, evidence timestamps, and an export to the tools teams already use.

Non-goals: live transcription, speaker *identification* across meetings (you get labels like `SPEAKER_1`, and an optional manual mapping to names), sentiment analysis, and automatic task creation without human confirmation. Cross-meeting identity is a privacy decision, not a feature.

### The join is interval arithmetic

STT returns words with `start`/`end`. Diarisation returns turns with `start`/`end` and a speaker label. Assign each word to the turn with the greatest temporal overlap; break ties by the nearer midpoint. Words that overlap no turn inherit the previous speaker. Then merge consecutive words with the same speaker into a segment, splitting on a silence gap longer than a threshold so a long monologue becomes several readable paragraphs.

Two practical guards: diarisation boundaries drift relative to STT timings, so allow a small tolerance when computing overlap; and short back-channels ("mm-hmm") should be merged into the neighbouring segment rather than starting a turn, or the transcript becomes unreadable.

### Decisions versus discussion

A decision has three properties: a commitment ("we will"), a scope, and a moment. Extract candidates with a prompt that must return the verbatim quote plus a timestamp, then filter: if the quote does not exist in the transcript at that timestamp, drop it. That is the same verification idea as [Research Assistant](01-research-assistant.md) — the model proposes, code checks. Discussion without commitment goes into the summary, not the decision log.

### Action items need an owner and a source

An action item is `{owner, action, due, quote, t_start}`. Resolve the owner from the sentence first ("Priya will…"), then from the current speaker ("I'll…" → the speaker), then leave it null and surface it for confirmation. A null owner is honest; an invented one gets someone else's work assigned to you. Show every item with its quote and a play button at the timestamp — that single feature is what makes people trust the list.

### Data model, stack, milestones, budget

| table | key fields |
|-------|-----------|
| `meeting` | id, title, started_at, duration_s, audio_key |
| `speaker` | id, meeting_id, label, display_name |
| `segment` | id, meeting_id, speaker_id, t_start, t_end, text |
| `item` | id, meeting_id, kind, owner, text, due, quote, t_start, confirmed |

Stack, loose: `ffmpeg` for normalisation, an STT API with word timings, a diarisation model (hosted or `pyannote`-style locally — check licences), Postgres, and one LLM for extraction. Store audio in object storage with a retention policy.

Four weeks: week 1 transcription and segments; week 2 the diarisation join and speaker labels; week 3 decisions and actions with evidence; week 4 confirmation UI, export, evals.

Budget, illustrative: a 45-minute meeting is priced by the minute for STT and diarisation, plus roughly 15k tokens to summarise a long transcript in sections. Extraction adds maybe 8k input and 1k output. Per meeting, STT usually dominates; per token, the summary does. Chunk the transcript by section rather than sending it whole, or long meetings will exceed context limits and cost more than they need to.

### Eval plan

- **Speaker-attributed word accuracy** — word error rate on a hand-transcribed sample, computed *per speaker*. Per-speaker is the number that matters; a good overall score can hide one badly attributed participant.
- **Decision precision and recall** against a hand-labelled set of meetings.
- **Action item match** — owner and action both correct, judged against what a human wrote in the real minutes.
- **Null-owner rate** — how often you correctly declined to guess. Track it, do not punish it.
- **Cost and turnaround time per meeting.**

## Code

The join, done properly, is the snippet worth showing in an interview.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(word: Word, turns: list[Turn], tol: float = 0.15) -> str:
    """Turn with the largest temporal overlap wins; ties go to the nearer midpoint."""
    best: tuple[float, float, str] | None = None
    mid = (word.start + word.end) / 2
    for turn in turns:
        ov = overlap(word.start - tol, word.end + tol, turn.start, turn.end)
        if ov <= 0:
            continue
        dist = abs(mid - (turn.start + turn.end) / 2)
        key = (ov, -dist, turn.speaker)
        if best is None or key > best:
            best = key
    return best[2] if best else "UNKNOWN"


def build_segments(words: list[Word], turns: list[Turn], gap: float = 1.2) -> list[Segment]:
    segments: list[Segment] = []
    for word in sorted(words, key=lambda w: w.start):
        speaker = assign_speaker(word, turns)
        if segments and segments[-1].speaker == speaker and word.start - segments[-1].end <= gap:
            segments[-1].words.append(word)
            segments[-1].end = word.end
        elif segments and len(word.text) <= 4 and word.start - segments[-1].end <= gap:
            segments[-1].words.append(word)        # back-channel: keep it in the turn
            segments[-1].end = word.end
        else:
            segments.append(Segment(speaker=speaker, start=word.start, end=word.end, words=[word]))
    return segments
```

Then extraction, with the quote check that stops invented decisions:

```python
async def extract_items(segments: list[Segment], llm) -> list[Item]:
    transcript = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in segments)
    raw = await llm.extract(transcript)          # returns candidates with quote + t_start
    joined = " ".join(s.text for s in segments)
    kept: list[Item] = []
    for cand in raw:
        if cand.quote.strip() and cand.quote.strip() in joined:   # evidence or drop
            kept.append(cand)
        # else: the model paraphrased; it is not evidence
    return kept
```

## Real use case

A ten-person engineering org records design reviews. Before: a rotating note-taker produced minutes that arrived a day late and were skimmed by nobody; decisions were re-litigated because no one could find where they were made.

After: the pipeline posts a summary with a decision log, and each decision links to the 12 seconds where it was agreed. Action items appear with owners and a quote, and anything with a null owner is flagged for the meeting owner to confirm.

The design choice that made it work is evidence timestamps on every extracted fact: reviewers could verify in one click, so they trusted the list, and trust is what changed behaviour.

Demo script, two minutes: open a meeting, show the speaker-attributed transcript with a back-channel merged correctly, click a decision and play the clip, show a null-owner action flagged for confirmation. Talking points: the interval join and its failure mode, why owners are null rather than guessed, per-speaker accuracy, and the privacy choice of not identifying speakers across meetings.

## Common mistakes

**Mistake:** assigning speakers by nearest timestamp only — **Instead:** use overlap with a tolerance; timings drift between models.
**Mistake:** letting diarisation create a turn for every "mm-hmm" — **Instead:** merge short back-channels into the current segment.
**Mistake:** accepting paraphrased evidence — **Instead:** require the verbatim quote to appear in the transcript, or drop the item.
**Mistake:** guessing an owner — **Instead:** leave it null and flag it; a wrong owner is worse than a question.
**Mistake:** summarising a long transcript in one call — **Instead:** chunk by section and merge; watch the context limit.
**Mistake:** reporting one overall word error rate — **Instead:** report per speaker; attribution errors hide inside averages.

## Checkpoint

1. Compute per-speaker word error rate on a five-minute hand transcript. Which speaker is worst, and does the join or the STT explain it?
2. Add a tolerance parameter to `assign_speaker` and sweep it. Plot the change in per-speaker accuracy.
3. Add a "decisions" review step where a human rejects an item and record the rejection rate over ten meetings. What does that tell you to fix in the prompt?

**Self-check:** Exercise 1 usually shows the worst speaker is the one with the most overlapping speech, which points at diarisation, not transcription.

## Go deeper

- [Multimodal Assistant](07-multimodal-assistant.md) — the same block model applied to single-user audio
- [Workflow Automation](10-workflow-automation.md) — pushing confirmed action items into a tracker
- [Human Evals](../15-evals/07-human-evals.md) — running a review step that people actually complete
- [PII Redaction](../20-ai-security/04-pii-redaction.md) — transcripts are personal data

---

[← Previous: Multimodal Assistant](07-multimodal-assistant.md) · [Stage index](README.md) · [Next: Knowledge Assistant](09-knowledge-assistant.md)
