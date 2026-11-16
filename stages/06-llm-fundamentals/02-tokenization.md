# 2. Tokenization: BPE and Byte-Level

> **Stage 06 · LLM Fundamentals** — The unit every model actually reads, every price is quoted in, and every context window is measured by.
>
> **You'll be able to:** train and apply byte-level BPE yourself, and explain any weird model behaviour in terms of tokens rather than words.
>
> **Prerequisites:** [1. GPT and BERT: Decoder-Only vs Encoder-Only](01-gpt-and-bert.md) · [1. PyTorch: Tensors and Modules](../05-dl-frameworks/01-pytorch.md)

## Why this exists

Models do not see text. They see integers, and the mapping from text to integers is a tokenizer that was frozen when the model was trained. Every practical oddity you will meet follows from that:

- a price per million tokens that does not match your word count;
- a model that miscounts letters in a word, because it never saw the letters individually;
- a context window that fills faster for code or Hindi than for English;
- a truncated prompt that loses its final instruction mid-word.

You cannot debug any of these by thinking in words. You have to think in tokens.

The mental model: *a tokenizer is a lossless compressor built from a fixed vocabulary; tokens are the atoms, and they are almost never words.*

## The mental model

```text
 text:      "unbelievable"
 bytes:     u  n  b  e  l  i  e  v  a  b  l  e     (12 bytes, ids 0-255)
 merges:    un | be | lie | va | ble               (learned, most frequent first)
 tokens:    ["un", "believ", "able"]  ->  [372, 8114, 512]  (vocab ids)
 decode:    reverse the merges, byte by byte, -> identical text
```

Three layers: **bytes** (always 256 possible values, so any file is representable), **merges** (learned pairs, ordered, applied in sequence), and **vocabulary ids** (the integers the model's embedding table is indexed by). Byte-level BPE means nothing is ever "unknown" — worst case, a string becomes one token per byte.

## Core concepts

### BPE learns the most frequent pairs, then merges them

Start from raw bytes. Count adjacent pairs across the corpus. Merge the most frequent pair into a new symbol. Repeat. Each merge adds one vocabulary entry. After enough merges, common words become single tokens and rare words stay as several. The merge *order* is part of the model: applying merges out of order gives different tokens.

### Byte-level means never out-of-vocabulary

Older word-level tokenizers had an `<UNK>` token and lost information. Byte-level BPE has no unknown state: a byte that has never been seen still encodes as itself. This is why a model can handle a language it barely trained on, and why emoji and rare scripts still round-trip exactly.

### Token counts are not word counts

For ordinary English prose, a rough rule of thumb is about four characters per token — but that is an average, not a promise. Code, JSON, non-Latin scripts and long identifiers tokenize far worse, often close to one token per character. Always measure with the actual tokenizer instead of estimating from words.

### The vocabulary is frozen, so the tokenizer is part of the model

You cannot swap tokenizers on a trained model: the embedding table has one row per token id. This is also why a "tokenizer-only" bug is catastrophic and silent — mismatched ids produce fluent-looking nonsense. If you fine-tune, you keep the tokenizer; if you add special tokens, you must resize the embedding table.

### Tokens explain the pricing and the failures

Billing is per token, input and output separately, with output usually priced higher because generation is sequential. Context limits are in tokens. And letter-level tasks fail because a token like `"believ"` has no internal structure the model can index into — asking "how many r's in strawberry" is asking about bytes the model never sees.

## Code

A working byte-level BPE trainer and encoder in about fifty lines. Run it on a small corpus and watch common sequences collapse into single tokens.

```python
from __future__ import annotations

from collections import Counter


def train_bpe(text: str, num_merges: int) -> list[tuple[int, int]]:
    """Learn merges from raw bytes. Ids 0-255 are bytes; 256+ are merges."""
    ids: list[int] = list(text.encode("utf-8"))       # byte-level: lossless start
    merges: list[tuple[int, int]] = []
    for _ in range(num_merges):
        counts = Counter(zip(ids, ids[1:]))           # frequency of adjacent pairs
        if not counts:
            break
        (a, b), freq = counts.most_common(1)[0]
        if freq < 2:                                  # stop when nothing repeats
            break
        merges.append((a, b))
        new_id = 256 + len(merges) - 1                # the new symbol's id
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)                    # replace the pair with one id
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return merges


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Apply merges in learned order, the same way training did."""
    ids = list(text.encode("utf-8"))
    for idx, (a, b) in enumerate(merges):
        new_id = 256 + idx
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    """Expand merges in reverse order, then read the bytes back."""
    expanded = list(ids)
    for idx in range(len(merges) - 1, -1, -1):
        a, b = merges[idx]
        target = 256 + idx
        out: list[int] = []
        for t in expanded:
            if t == target:
                out.extend((a, b))
            else:
                out.append(t)
        expanded = out
    return bytes(expanded).decode("utf-8", errors="replace")
```

Now use it, and check the property that matters — losslessness — plus the token-count effect on different kinds of text.

```python
corpus = ("the tokenizer tokenizes tokens; the model models models. "
          "low lower lowest, new newer newest, wide wider widest. ") * 20

merges = train_bpe(corpus, num_merges=40)
print("merges learned:", len(merges))
print("first three:", merges[:3])          # ids, not text: (100, 101) is 'd','e'

s = "the lowest tokenizer"
toks = encode(s, merges)
print(len(s), "chars ->", len(toks), "tokens")     # fewer tokens than bytes
assert decode(toks, merges) == s                   # lossless round trip

# tokenization is not language-neutral: same meaning, different cost
for probe in ["tokenizer", "टोकनाइज़र", "def f(x): return x + 1"]:
    print(f"{len(encode(probe, merges)):3d} tokens  {probe!r}")
```

## Real use case

A startup builds a document Q&A product and prices it at a flat monthly fee. Their prompts include retrieved passages, and they budget assuming a token is roughly a word. Two months in, the gross margin is negative on customers who upload dense technical PDFs.

The cause is measurement. Their passages contain tables, code and identifiers, which tokenize at close to one token per character rather than four characters per token. The same 2,000-word document costs several times their estimate on every question, and it is re-sent on every turn because they had no conversation trimming.

The fix has three parts, all token-based rather than word-based: a hard token budget per request measured with the real tokenizer, retrieval that returns a fixed number of tokens rather than a fixed number of chunks, and trimming of older turns when the budget is exceeded. The lesson is not "raise prices" — it is that any product decision about cost or context is a token decision, and guessing in words is how you lose money quietly.

## Common mistakes

- **Mistake:** estimating cost from word count. — **Instead:** count with the tokenizer you will actually use, on real inputs.
- **Mistake:** assuming all tokenizers agree. — **Instead:** each model family has its own vocabulary; counts differ by a wide margin on code and non-English text.
- **Mistake:** truncating by characters to fit a context window. — **Instead:** truncate by tokens, and cut at a message or sentence boundary.
- **Mistake:** adding a special token and forgetting to resize embeddings. — **Instead:** call `tokenizer.add_special_tokens(...)` and `model.resize_token_embeddings(len(tokenizer))`.
- **Mistake:** expecting a model to count letters. — **Instead:** give it the text split into characters, or use code for anything character-exact.

## Checkpoint

1. Train BPE with 5 merges and again with 200 merges on the same corpus. How does token count for a held-out sentence change, and why does it stop improving?
2. Break the round trip deliberately: apply the merges in reverse order in `encode`. Show a string that fails to decode, and explain what that tells you about merge order being part of the model.
3. Encode the same sentence in English, Hindi and a Python snippet with your tokenizer, then report tokens per character for each. Which would cost most per word of meaning?

**Self-check:** More merges means longer learned sequences, so a held-out sentence uses fewer tokens — until the merges stop matching that sentence's patterns, at which point extra vocabulary buys nothing.

## Go deeper

- [3. Context Windows](03-context-windows.md) — what these tokens are spent on.
- [9. Ship: Tokenizer and Token-Budget Analyser](09-ship-tokenizer.md) — productionise this.
- [8. Cost Tracking](../07-llm-apis/08-cost-tracking.md) — turning token counts into money.
- [8. Embeddings](../04-deep-learning/08-embeddings.md) — the table these ids index into.

---
[← Previous: GPT and BERT: Decoder-Only vs Encoder-Only](01-gpt-and-bert.md) · [Stage index](README.md) · [Next: Context Windows](03-context-windows.md)
