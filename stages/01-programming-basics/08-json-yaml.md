# 8. JSON and YAML

> **Stage 01 · Programming Basics** — the two text formats that carry every request, config and dataset in this roadmap.
>
> **You'll be able to:** read and write JSON confidently, choose JSON or YAML deliberately, and avoid the encoding and nesting traps that break pipelines.
>
> **Prerequisites:** [HTTP and REST APIs](07-http-rest-apis.md) · [Type Hints and Pydantic v2](02-type-hints-pydantic.md)

## Why this exists

JSON is what APIs send and what models are asked to produce. YAML is what humans write when a config gets big enough that braces hurt — Docker Compose files, Kubernetes manifests, CI workflows, prompt configs. Both are trees of keys and values, and both look simple enough to skip learning. Then you spend an afternoon on a trailing comma, a `null` that should have been `0`, or a YAML key that turned into a boolean.

The mental model: **JSON is for machines to exchange; YAML is for humans to edit. Both describe the same tree.**

## The mental model

```text
JSON                              YAML
{                                 name: invoice-bot
  "name": "invoice-bot",         model: gpt-class        # check provider docs
  "retries": 3,                  retries: 3
  "tags": ["ocr", "finance"],    tags:
  "limits": {"rpm": 60}            - ocr
}                                  - finance
                                  limits:
                                    rpm: 60
```

| | JSON | YAML |
|---|---|---|
| Comments | no | yes |
| Types | string, number, bool, null, array, object | same, plus dates and anchors |
| Whitespace | irrelevant | significant |
| Trailing commas | illegal | n/a |
| Best for | wire format, JSONL data | config, manifests |

## Core concepts

### JSON has exactly six kinds of value

String, number, boolean, null, array, object. No dates, no bytes, no comments, no `NaN` in strict mode. Everything else is a convention: dates become ISO strings, money becomes a number or a string (choose deliberately — floats lose cents), binary becomes base64. A key must be a string in double quotes; single quotes and unquoted keys are YAML habits that break JSON.

### `null`, missing, and empty are three different things

`{"score": null}` means the field exists and has no value. `{}` means the field is absent. `{"score": 0}` means zero. Code that reads with `.get("score", 0)` collapses all three into one and hides bugs. Decide per field: absent is an error, `null` is "unknown", `0` is a measurement.

### `json.dumps` has two flags you will use every day

`ensure_ascii=False` keeps non-Latin text readable instead of `\u0915`. `indent=2` makes output reviewable in a diff. For anything crossing a boundary, pair `dumps` with a pydantic model rather than raw dicts.

### JSONL is the format for datasets

One JSON object per line, no wrapping array. Why it wins: you can stream it (see [Generators](03-generators.md)), append to it without rewriting, and recover from a truncated last line. An array of a million objects cannot be appended to safely and cannot be parsed until it is complete.

### YAML is a superset of JSON with significant whitespace

Indentation defines nesting; two spaces is the convention, tabs are illegal. `-` starts a list item. `key: value` is a mapping. Strings often need no quotes, which is where the trouble starts.

### YAML's type guessing is the classic trap

Unquoted `yes`, `no`, `on`, `off`, `true` become booleans; `1.0` becomes a float; `2024-01-01` becomes a date; `1.10` becomes `1.1`. A version string like `version: 1.10` silently becomes `1.1`. Quote anything that is a version, an id, or a country code.

## Code

Reading and writing JSON safely, with the null-versus-missing distinction made explicit.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    # encoding="utf-8" is not the default on every platform; be explicit.
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    # Write to a temp file then replace, so a crash never leaves half a report.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def score_of(record: dict[str, Any]) -> float:
    if "score" not in record:
        raise KeyError("score missing: upstream contract broken")
    value = record["score"]
    if value is None:
        return 0.0        # explicit "unknown" -> treat as no signal
    if not isinstance(value, (int, float)):
        raise TypeError(f"score must be a number, got {type(value).__name__}")
    return float(value)
```

Streaming JSONL both ways, and the JSON-versus-YAML conversion you will need when a prompt config lives in YAML but an API wants JSON:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def yaml_file_to_json(path: Path) -> str:
    # safe_load refuses arbitrary Python objects; never use yaml.load here.
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2, ensure_ascii=False)
```

A config file written the defensive way — quoted versions, explicit types:

```yaml
# prompts/classify.yaml
name: "ticket-classifier"
version: "1.10"          # quoted: unquoted 1.10 becomes 1.1
temperature: 0.0
labels:
  - billing
  - bug
  - feature
limits:
  max_retries: 3
  timeout_seconds: 30
notes: |
  Multi-line strings keep their newlines with the | block marker.
```

## Real use case

A prompt-management service stored prompt variants in YAML so non-engineers could edit them. One variant declared `version: 1.10`. YAML parsed it as the float `1.1`, which collided with an existing variant at version `1.1`, and the loader silently overwrote the older prompt. The team spent a day chasing "the prompt changed by itself" before someone printed `type(config["version"])` and saw `<class 'float'>`.

The fix was two-part: every version, id and code in the YAML is now quoted, and the loader validates the file with a pydantic model where `version: str` is required and `extra="forbid"`. Invalid config now fails at load with the field named, and a duplicate version raises instead of overwriting. The JSON side got the same treatment: `json.dumps` with `sort_keys=True` so prompt diffs in review show only real changes.

The design choice that mattered was refusing to let the file format guess types — quoting what is text, validating what crosses the boundary.

## Common mistakes

- **Mistake:** `json.loads` on a file opened without `encoding="utf-8"`. — **Instead:** be explicit; non-ASCII text and Windows defaults will bite.
- **Mistake:** treating missing, `null` and `0` as the same thing. — **Instead:** decide the meaning per field and validate.
- **Mistake:** storing money as a float. — **Instead:** store integer minor units or a decimal string.
- **Mistake:** `yaml.load` without `Loader=yaml.SafeLoader`. — **Instead:** always `yaml.safe_load`.
- **Mistake:** an unquoted YAML version or id like `1.10` or `on`. — **Instead:** quote anything that must stay a string.
- **Mistake:** hand-editing a large JSON array dataset. — **Instead:** use JSONL so appends and streaming work.

## Checkpoint

1. Convert a YAML config to JSON with `yaml_file_to_json` and identify every value whose type changed.
2. Write a function that reads a JSONL file and reports the line number of every malformed line without stopping.
3. Build a pydantic model for the prompt config above with `version: str`, then prove that an unquoted `1.10` is rejected.

**Self-check:** the unquoted `1.10` arrives as the float `1.1`, so a `str` field raises a validation error — which is the point: it fails loudly instead of colliding silently.

## Go deeper

- [Type Hints and Pydantic v2](02-type-hints-pydantic.md) — the validation layer that sits on top of these formats.
- [Parquet and JSONL](../03-data-engineering/05-parquet-jsonl.md) — when JSONL stops being enough and columnar storage wins.
- [Structured Outputs](../07-llm-apis/04-structured-outputs.md) — getting a model to emit JSON that matches your schema.

---
[← Previous: HTTP and REST APIs](07-http-rest-apis.md) · [Stage index](README.md) · [Next: Ship a CLI Tool](09-ship-cli-tool.md)
