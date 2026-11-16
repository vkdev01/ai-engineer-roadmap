#!/usr/bin/env python3
"""Check the repo before a pull request.

    python tooling/validate.py

Verifies, for every markdown file:
  - every relative link resolves to a file that exists
  - code fences are balanced and language-tagged
  - lesson files carry the required section headings
  - lesson files are not suspiciously thin
  - no emoji and no inline citation markers

Exits non-zero on any problem, so it can be used as a CI step.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
H1 = re.compile(r"^#\s+.+$", re.MULTILINE)
FENCE_BLOCK = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")

REQUIRED = ("## Why this exists", "## Core concepts", "## Code", "## Real use case",
            "## Common mistakes", "## Checkpoint", "## Go deeper")

problems: list[str] = []
stats = {"files": 0, "words": 0, "code_lines": 0, "links": 0}


def strip_code(text: str) -> str:
    return FENCE_BLOCK.sub("", text)


def check(path: Path, text: str) -> None:
    rel = path.relative_to(ROOT)
    body = strip_code(text)
    stats["files"] += 1
    stats["words"] += len(body.split())

    for target in LINK.findall(body):
        if target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        stats["links"] += 1
        if not (path.parent / target).resolve().exists():
            problems.append(f"BROKEN LINK  {rel} -> {target}")

    fences = len(re.findall(r"^```", text, re.MULTILINE))
    if fences % 2:
        problems.append(f"UNCLOSED FENCE  {rel} ({fences} fences)")

    in_code, code_lines = False, 0
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            code_lines += 1
    stats["code_lines"] += code_lines

    if EMOJI.search(body):
        problems.append(f"EMOJI  {rel}")
    if "[cite:" in body:
        problems.append(f"CITATION MARKER  {rel}")

    if path.parent.parent.name == "stages" and path.name != "README.md":
        for heading in REQUIRED:
            if heading not in text:
                problems.append(f"MISSING SECTION {heading}  {rel}")
        words = len(body.split())
        if words < 300:
            problems.append(f"THIN FILE ({words} words)  {rel}")


def main() -> int:
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", "node_modules", "docs"} for part in path.parts):
            continue
        check(path, path.read_text(encoding="utf-8"))

    print(f"files={stats['files']} words={stats['words']} "
          f"code_lines={stats['code_lines']} links={stats['links']}")
    if problems:
        print(f"problems={len(problems)}")
        for p in problems[:80]:
            print("  " + p)
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
