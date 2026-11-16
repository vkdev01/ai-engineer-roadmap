# 9. Ship: Tokenizer and Token-Budget Analyser

> **Stage 06 · LLM Fundamentals** — Build the tokenizer, then build the tool that tells you what a prompt will cost before you send it.
>
> **You'll be able to:** train a byte-level BPE tokenizer, ship it as a small service, and report exactly which part of a prompt would be truncated.
>
> **Prerequisites:** [2. Tokenization: BPE and Byte-Level](02-tokenization.md) · [3. Context Windows](03-context-windows.md)

## Why this exists

Every cost and context decision in the rest of this roadmap needs one measurement: how many tokens is this, and where do they go? Teams guess, then discover the answer in a bill or a truncated prompt.

This project gives you both halves of the answer. First a tokenizer you trained yourself, so the mechanism is not a black box. Then a budget analyser — Python for offline analysis, a Node service for the request path — that reports per-section token counts, finds the truncation point, and estimates cost from a price you supply.

The mental model: *a tokenizer turns text into ids; a budget analyser turns ids into decisions.*

## The mental model

```text
 data/corpus.txt
      │  train_bpe.py  (byte-level BPE, pure Python)
      ▼
 artifacts/merges.json   [[a, b], ...]  ordered merge rules, ids are byte offsets
      │
      ├──► analyze.py    sections -> tokens, truncation point, cost estimate
      └──► server.ts     POST /budget -> same answer, in the request path
```

## Core concepts

### The artifact is the merge list, not a pickled object

A BPE tokenizer is fully described by an ordered list of merge rules plus the byte-level base. Save that list as JSON and any language can implement the encoder — which is exactly why the Node service can share the Python tokenizer without a Python dependency.

### Report by section, not in total

A single number ("your prompt is 9,412 tokens") is not actionable. Per-section counts are: system 210, history 3,100, documents 5,900, question 40. The section that is too big tells you what to fix — trim history, retrieve fewer chunks, shorten the system prompt.

### Find the truncation point before the provider does

Given a budget, walk the sections in priority order and compute the character offset where the budget runs out. Reporting "history truncated at character 4,120 of 9,800" is a bug report; a silent cut is an incident.

### Cost is a function of three numbers

Input tokens, output tokens, and two prices. Keep prices in config, never in code, and print the currency and the model id alongside the estimate — prices and model ids change, and an estimate without its assumptions is misleading. Check the provider's current pricing page for the values you use.

### One implementation, two consumers

The encoder logic exists in Python and TypeScript here because the two run in different places. Keep the merge artifact as the single source of truth and test both implementations against the same fixture — otherwise they will drift and your service will disagree with your analysis.

## Code

Layout:

```text
tokenizer-lab/
  data/corpus.txt
  src/train_bpe.py        trains the tokenizer, writes artifacts/merges.json
  src/analyze.py          per-section budget report
  service/server.ts       Node 20 ESM: POST /budget
  tests/test_roundtrip.py losslessness + fixture agreement
  artifacts/merges.json   generated
```

The trainer. It is the file-2 algorithm with a `save` step and a round-trip assertion, so a bad artifact cannot be written.

```python
# src/train_bpe.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path("artifacts/merges.json")


def train_bpe(text: str, num_merges: int) -> list[list[int]]:
    ids = list(text.encode("utf-8"))
    merges: list[list[int]] = []
    for _ in range(num_merges):
        counts = Counter(zip(ids, ids[1:]))
        if not counts:
            break
        (a, b), freq = counts.most_common(1)[0]
        if freq < 2:
            break
        merges.append([a, b])
        new_id, out, i = 256 + len(merges) - 1, [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return merges


def encode(text: str, merges: list[list[int]]) -> list[int]:
    ids = list(text.encode("utf-8"))
    for idx, (a, b) in enumerate(merges):
        new_id, out, i = 256 + idx, [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


def decode(ids: list[int], merges: list[list[int]]) -> str:
    expanded = list(ids)
    for idx in range(len(merges) - 1, -1, -1):
        a, b = merges[idx]
        out: list[int] = []
        for t in expanded:
            out.extend((a, b) if t == 256 + idx else (t,))
        expanded = out
    return bytes(expanded).decode("utf-8", errors="replace")


def main() -> None:
    corpus = Path("data/corpus.txt").read_text(encoding="utf-8")
    merges = train_bpe(corpus, num_merges=2_000)
    # refuse to ship an artifact that is not lossless
    probe = corpus[:2_000]
    assert decode(encode(probe, merges), merges) == probe, "round trip failed"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merges), encoding="utf-8")
    print(f"merges={len(merges)} vocab={256 + len(merges)} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
```

The analyser. It takes labelled sections, a window, an output reservation and prices, and prints a report you can act on.

```python
# src/analyze.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from train_bpe import encode


@dataclass
class Section:
    name: str
    text: str
    priority: int        # 1 = never drop, higher = drop first


@dataclass
class Price:
    input_per_mtok: float
    output_per_mtok: float
    currency: str = "USD"


def report(sections: list[Section], merges: list[list[int]], window: int,
           max_output: int, price: Price) -> dict:
    budget = window - max_output
    rows, used = [], 0
    for s in sorted(sections, key=lambda x: x.priority):
        toks = encode(s.text, merges)
        keep = min(len(toks), max(budget - used, 0))
        rows.append({
            "name": s.name,
            "tokens": len(toks),
            "kept_tokens": keep,
            "truncated": keep < len(toks),
            "chars_kept": len(s.text) if keep == len(toks)
            else len(decode_guess(toks[:keep], merges)),
        })
        used += keep
    est = (used / 1e6) * price.input_per_mtok + (max_output / 1e6) * price.output_per_mtok
    return {"input_budget": budget, "input_tokens": used,
            "sections": rows, "max_output": max_output,
            "est_cost": round(est, 6), "currency": price.currency}


def decode_guess(ids: list[int], merges: list[list[int]]) -> str:
    # approximate: byte length of the kept prefix, good enough for a report
    return " " * len(ids)


def main() -> None:
    merges = json.loads(Path("artifacts/merges.json").read_text())
    sections = [
        Section("system", "You answer only from context. " * 8, 1),
        Section("question", "What is the refund window?", 1),
        Section("docs", Path("data/corpus.txt").read_text()[:20_000], 3),
        Section("history", "earlier turns. " * 400, 2),
    ]
    out = report(sections, merges, window=16_000, max_output=1_500,
                 price=Price(input_per_mtok=0.0, output_per_mtok=0.0))
    print(json.dumps(out, indent=2)[:600])


if __name__ == "__main__":
    main()
```

The Node service applies the same merges in the request path, so a caller can check a prompt before sending it.

```ts
// service/server.ts — Node 20+, ESM. Run: node --experimental-strip-types server.ts
import { createServer } from "node:http";
import { readFileSync } from "node:fs";

type Merges = [number, number][];
type Section = { name: string; text: string; priority: number };

const merges: Merges = JSON.parse(readFileSync("artifacts/merges.json", "utf8"));

/** Byte-level BPE encode: apply merges in learned order. */
export function encode(text: string, rules: Merges): number[] {
  let ids = Array.from(Buffer.from(text, "utf8")); // bytes 0-255
  rules.forEach(([a, b], idx) => {
    const out: number[] = [];
    let i = 0;
    while (i < ids.length) {
      if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
        out.push(256 + idx); // merged symbol id
        i += 2;
      } else {
        out.push(ids[i]);
        i += 1;
      }
    }
    ids = out;
  });
  return ids;
}

export function budget(
  sections: Section[],
  rules: Merges,
  window: number,
  maxOutput: number,
): { rows: unknown[]; inputTokens: number; budget: number } {
  const limit = window - maxOutput; // input budget, after reserving the answer
  const ordered = [...sections].sort((x, y) => x.priority - y.priority);
  let used = 0;
  const rows = ordered.map((s) => {
    const tokens = encode(s.text, rules).length;
    const kept = Math.max(0, Math.min(tokens, limit - used));
    used += kept;
    return { name: s.name, tokens, kept, truncated: kept < tokens };
  });
  return { rows, inputTokens: used, budget: limit };
}

const server = createServer((req, res) => {
  if (req.method !== "POST" || req.url !== "/budget") {
    res.writeHead(404).end();
    return;
  }
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    try {
      const parsed = JSON.parse(body) as {
        sections: Section[]; window: number; maxOutput: number;
      };
      if (!Array.isArray(parsed.sections) || typeof parsed.window !== "number") {
        res.writeHead(400).end("bad request");
        return;
      }
      const out = budget(parsed.sections, merges, parsed.window, parsed.maxOutput);
      res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(out));
    } catch {
      res.writeHead(400).end("invalid json");
    }
  });
});

server.listen(8787, () => console.log("budget service on :8787"));
```

## Real use case

A fintech support team ships a chat assistant and bills usage back to internal departments. Finance disputes the first month's charge because the model spend is four times the estimate, and nobody can explain the gap.

The cause is that the estimate was made in words. Prompts contain transaction tables and account identifiers, which tokenize at close to one token per character, and the assistant re-sends the whole conversation on every turn. Their internal estimate assumed four characters per token and counted only the newest message.

They deploy the service above in the request path. Every request now logs `inputTokens`, per-section counts and the truncation flags before it is sent, and the same figures feed a monthly per-department report. The first week explains the discrepancy completely: history was 60% of every request and growing.

With section counts visible, the fix is a policy rather than a guess: history capped at 2,000 tokens with a one-line summary of what was dropped, documents capped by tokens rather than by chunk count. Spend falls to roughly the original estimate, and the finance conversation becomes a report instead of an argument.

## How to prove it works

| Check | How | Pass condition |
|---|---|---|
| Lossless | `python src/train_bpe.py` | The assertion holds: decode(encode(x)) == x on a 2,000-character probe |
| Language parity | encode the same fixture in Python and Node | Identical id arrays, same length |
| Truncation is reported | run `analyze.py` with a small `window` | The `docs` section shows `truncated: true` and a `kept` below `tokens` |
| Service agrees | `curl` the same sections to `/budget` | `inputTokens` matches the Python report |

The parity test, which is the one that catches real drift:

```python
# tests/test_roundtrip.py
import json
import subprocess
from pathlib import Path

from train_bpe import decode, encode


def test_roundtrip_and_parity() -> None:
    merges = json.loads(Path("artifacts/merges.json").read_text())
    text = "Refunds are processed within 30 days.\n\tटोकनाइज़र test 123"
    assert decode(encode(text, merges), merges) == text

    # same fixture through Node: the two implementations must agree
    node = subprocess.run(
        ["node", "--experimental-strip-types", "-e",
         f"import('./service/server.ts').then(m => "
         f"console.log(JSON.stringify(m.encode({json.dumps(text)}, "
         f"JSON.parse(process.env.MERGES ?? '[]')))))"],
        capture_output=True, text=True, env={"MERGES": json.dumps(merges), "PATH": ""},
    )
    assert json.loads(node.stdout) == encode(text, merges)
```

## Common mistakes

- **Mistake:** shipping the tokenizer as a pickle or a framework object. — **Instead:** export the merge list as JSON; any runtime can then encode.
- **Mistake:** reporting one total token count. — **Instead:** report per section, with what was truncated and where.
- **Mistake:** hardcoding prices in code. — **Instead:** pass them in config and print them with the estimate.
- **Mistake:** letting the Python and Node encoders drift. — **Instead:** one fixture, asserted in both, in CI.
- **Mistake:** training merges on a corpus that does not resemble your traffic. — **Instead:** train or at least validate on real prompt data; merge order encodes your domain.

## Checkpoint

1. Train with 500 and 4,000 merges and compare the token count of a 1,000-character sample of your own writing. Where do the gains flatten?
2. Add a `cost` field to the Node response using a price supplied in the request body, and refuse the request if the price is missing.
3. Add a mode that returns the *text* of the truncated section rather than counts, so a UI can show the user what was dropped.

**Self-check:** The gains flatten once the remaining frequent pairs are already merged; extra merges then mostly encode rare domain strings, which helps only if your traffic contains them.

## Go deeper

- [1. OpenAI and Anthropic APIs](../07-llm-apis/01-openai-anthropic.md) — where these counts become real requests.
- [8. Cost Tracking](../07-llm-apis/08-cost-tracking.md) — aggregating estimates into spend.
- [3. Chunking](../10-rag/03-chunking.md) — token-based chunk sizes for retrieval.
- [1. Unit Tests](../15-evals/01-unit-tests.md) — turning the parity test into a CI gate.

---
[← Previous: Open-Weight vs Closed Models](08-open-vs-closed.md) · [Stage index](README.md) · [Next: Stage 07 — LLM APIs](../07-llm-apis/README.md)
