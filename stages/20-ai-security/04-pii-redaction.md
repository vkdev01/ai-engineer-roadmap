# 4. PII Redaction and Detection

> **Stage 20 · AI Security** — finding personal data in free text with regex plus a model, replacing it so the pipeline still works, and choosing the one place in the pipeline where redaction belongs.
>
> **You'll be able to:** build a detector that catches the personal data your app actually sees, pseudonymise it reversibly where the task needs it, and place redaction so it cannot be bypassed.
>
> **Prerequisites:** [Data Leakage](03-data-leakage.md) · [Validation](../03-data-engineering/07-validation.md)

## Why this exists

A user pastes a customer email into your assistant to get a reply drafted. That text now contains a name, a phone number, an address, an order id and a card fragment. It goes to a model provider, into your prompt logs, into your evaluation dataset, and into a support ticket. Five copies, four of them with weaker controls than the original database.

Redaction is the discipline of removing or replacing that data before it crosses a boundary you do not control. It is not one regex. Names have no shape; addresses are wildly variable; Indian phone numbers, Aadhaar-like 12-digit numbers and UPI handles each need their own pattern; and text in Devanagari or Tamil breaks naive ASCII assumptions. A production detector is **regex for the structured cases, a model for the unstructured ones, and a policy for what to do when you are unsure.**

The mental model: **redact at the earliest boundary, not the last one, and choose between masking (irreversible) and pseudonymisation (reversible with a key held elsewhere).**

## The mental model

```text
 raw text ──► normalise ──► regex pass ──► NER pass ──► merge spans
                                                          │
                                    ┌─────────────────────┴───────────┐
                                    ▼                                 ▼
                            mask  [EMAIL]                   pseudonymise
                            (irreversible)                  PERSON_3 <-> key
                                    │                                 │
                                    ▼                                 ▼
                          model / logs / evals              model (needs the
                          never see the value               value to reason)
                                                          restore after the call
```

| Data | Detector | Notes |
|---|---|---|
| Email, URL | regex | reliable |
| Phone (India) | regex with context | `[6-9]\d{9}` plus `+91` forms; require a nearby keyword to cut false positives |
| Card number | regex + Luhn check | Luhn removes most 16-digit noise |
| 12-digit government id | regex + context | do not assume any 12-digit number is one |
| Person, org, place | NER model | imperfect; tune for your domain and language |
| Address | NER + keyword heuristic | worst recall; treat as best-effort |

## Core concepts

### Regex first, because it is explainable

Structured identifiers are the easy win and you should never use a model for them. A regex is fast, deterministic, testable and cheap to review. Two rules make it usable in practice: require **context** for short numeric patterns (a bare 10-digit number is as likely to be an order id as a phone number), and **validate the checksum** where one exists — Luhn for card numbers, format rules for known id types.

Write the patterns as named, versioned constants with a test fixture per pattern, positive and negative. A pattern without a negative test will eventually eat your order ids, and someone will notice in production.

### NER second, because names have no shape

For names, organisations, places and addresses you need a model — a small token-classification model over the sentence, run locally, returning labelled spans. Practical points: pick a model that covers the languages you serve; accept that recall on Indian names in mixed-script text is lower than on English text and say so in your risk note; keep the model small and local so redaction does not itself become a data transfer; and pin its version, because a model upgrade changes your redaction behaviour and must be re-tested.

### Merge spans and be conservative

Regex and NER overlap. Merge the span sets, then widen rather than narrow: if a name span and an email span touch, redact the union. Over-redaction costs a little answer quality; under-redaction is a breach. Then decide the fallback for low-confidence spans — for high-risk fields (government ids, card numbers) redact on any plausible match; for lower-risk fields, log for review.

### Masking vs pseudonymisation

**Masking** replaces the value with a type tag: `[EMAIL]`, `[PERSON]`. Irreversible, safest, and fine when the task does not need the value — logging, evals, analytics, embeddings you will keep.

**Pseudonymisation** replaces with a stable token and stores the mapping in a separate table with its own access control: `PERSON_3` maps to "Ravi Kumar" for this request only, or for this tenant for a day. The model can reason about a coherent document ("PERSON_3 called about ORDER_12") and you restore values in your own process after the call. Use it when the task needs entity consistency — summarising a conversation, checking whether two records refer to the same person.

Two warnings. A stable pseudonym across requests is a pseudonymous identifier and is still personal data under most regimes; treat the mapping table as production data with audit. And do not "hash" a phone number and call it safe — a 10-digit number is trivially enumerable by hashing every candidate.

### Where in the pipeline to redact

Redact **at ingress, before the first boundary you do not control**, and again at egress for anything the model produced. Concretely, in this order:

1. **At the API edge**, before prompt construction — the value never enters your prompt, your traces or your vector store.
2. **Before embedding** any document you will persist — embeddings are derived data you cannot easily unlearn.
3. **Before export** to a tracing or analytics vendor.
4. **After generation** — the model may invent or restate PII from context.

Do not redact only in the UI. The UI is one of five copies.

## Code

A detector that combines regex and a local NER model, merges spans, and returns masked text plus the spans it found. The span list is what your audit log records — counts and types, not values.

```python
from __future__ import annotations

import re
from dataclasses import dataclass

import spacy  # pip install spacy; download a small English pipeline for your locale

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
CARD = re.compile(r"(?<!\d)\d{16}(?!\d)")
UPI = re.compile(r"\b[\w.-]{2,}@(?:okhdfcbank|okicici|okaxis|oksbi|ybl|paytm)\b", re.I)

NLP = spacy.load("en_core_web_sm")  # small and local: no data leaves the process


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def regex_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in EMAIL.finditer(text):
        spans.append(Span(m.start(), m.end(), "EMAIL"))
    for m in PHONE.finditer(text):
        spans.append(Span(m.start(), m.end(), "PHONE"))
    for m in CARD.finditer(text):
        if luhn_ok(m.group()):  # checksum cuts most 16-digit false positives
            spans.append(Span(m.start(), m.end(), "CARD"))
    for m in UPI.finditer(text):
        spans.append(Span(m.start(), m.end(), "UPI"))
    return spans


def ner_spans(text: str) -> list[Span]:
    doc = NLP(text)
    keep = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC"}
    return [
        Span(e.start_char, e.end_char, keep[e.label_])
        for e in doc.ents
        if e.label_ in keep
    ]


def merge(spans: list[Span]) -> list[Span]:
    """Sort and union overlapping spans. Widening beats narrowing."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.start, -s.end)):
        if out and s.start <= out[-1].end:
            prev = out[-1]
            out[-1] = Span(prev.start, max(prev.end, s.end), prev.label)
        else:
            out.append(s)
    return out


def redact(text: str) -> tuple[str, list[Span]]:
    spans = merge(regex_spans(text) + ner_spans(text))
    parts: list[str] = []
    cursor = 0
    for s in spans:
        parts.append(text[cursor : s.start])
        parts.append(f"[{s.label}]")  # masked, irreversible
        cursor = s.end
    parts.append(text[cursor:])
    return "".join(parts), spans
```

Pseudonymisation for the case where the model needs entity consistency. The map never leaves your process; only the tokenised text is sent.

```python
from __future__ import annotations


class Pseudonymiser:
    """Stable per-call tokens: PERSON_1, PERSON_2, ... and a private reverse map."""

    def __init__(self) -> None:
        self._forward: dict[tuple[str, str], str] = {}
        self._reverse: dict[str, str] = {}

    def token(self, label: str, value: str) -> str:
        key = (label, value.lower())
        if key not in self._forward:
            name = f"{label}_{sum(1 for k in self._forward if k[0] == label) + 1}"
            self._forward[key] = name
            self._reverse[name] = value
        return self._forward[key]

    def restore(self, text: str) -> str:
        for name, value in self._reverse.items():
            text = text.replace(name, value)
        return text
```

## Real use case

**An insurance claims assistant that reads email threads.** Claimants write in with policy numbers, vehicle registrations, phone numbers, and often a photo of a document with an Aadhaar-like number. Before redaction, the assistant sent the full thread to the model provider and logged raw prompts; the team could not answer "where does claimant data live" without listing five systems.

Redaction moved to the API edge, before prompt construction. Structured fields (email, phone, card with Luhn, UPI handle, policy number via a regex from the policy schema) are masked. Names and places go through a local NER pass. Because the assistant must follow one claimant across a long thread, the team used per-request pseudonyms rather than masks, so the model sees `PERSON_1` consistently and the restored reply goes back to the agent. Traces store masked text plus span counts only. Embeddings are built from masked text, so the vector store never held the raw thread. Two of the four copies disappeared entirely and the other two became defensible.

## Common mistakes

- **Mistake:** redacting only in the frontend or only in logs — **Instead:** redact at ingress, before prompt construction, and again before any export.
- **Mistake:** a bare 10-digit phone regex with no context or checksum — **Instead:** require context or a country prefix, validate card numbers with Luhn, and write a negative fixture per pattern.
- **Mistake:** hashing identifiers and calling the result anonymous — **Instead:** treat stable pseudonyms as personal data, keep the map access-controlled and audited.
- **Mistake:** using a large hosted model to do the redaction — **Instead:** run a small local NER model; redaction should not itself be a data transfer.
- **Mistake:** measuring success by "it works on my sample" — **Instead:** keep a labelled fixture set per language you serve and report recall per field type.

## Checkpoint

1. Write positive and negative fixtures for one pattern (phone or card) and make the test fail when you remove the Luhn check.
2. Extend `redact` to return span counts per label, and log only those counts. Explain why the counts are safe when the values are not.
3. Take a real request from your app and decide, field by field, whether to mask or pseudonymise. Write one sentence of justification per field.

**Self-check:** Removing the Luhn check makes the negative fixture (an order id that looks like 16 digits) get masked, so the test fails — that is exactly the regression the check exists to prevent.

## Go deeper

- [Data Leakage](03-data-leakage.md) — the four doors this control partly closes.
- [Governance](08-governance.md) — retention, residency and what to write down about all this.
- [Validation](../03-data-engineering/07-validation.md) — the same fixture discipline applied to datasets.
- [Ship: Injection Test Suite](10-ship-injection-tests.md) — add PII leakage cases to the same CI gate.

---
[← Previous: Data Leakage](03-data-leakage.md) · [Stage index](README.md) · [Next: Sandboxing Tool Execution](05-sandboxing.md)
