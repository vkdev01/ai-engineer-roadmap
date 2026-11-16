#!/usr/bin/env python3
"""Build the static GitHub Pages site in docs/ from the markdown in this repo.

Usage:
    pip install markdown pygments
    python tooling/build_site.py

Regenerates every .html file under docs/ and the search index, reusing the
stylesheets and scripts already in docs/assets/. Nothing else is touched, so
screenshots and other static files under docs/assets survive a rebuild.

The pages this writes are plain HTML with relative links: no Jekyll, no client
framework, no build step on the server. docs/.nojekyll tells GitHub Pages to
serve the folder as-is.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("markdown is not installed. Run: pip install markdown pygments")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
ASSETS = OUT / "assets"

ROOT_PAGES = [
    ("README.md", "index.html", "Start here"),
    ("FAST-TRACK.md", "fast-track.html", "Fast track"),
    ("PROGRESS.md", "progress.html", "Progress"),
    ("GLOSSARY.md", "glossary.html", "Glossary"),
    ("CONTRIBUTING.md", "contributing.html", "Contributing"),
    ("PUBLISHING.md", "publishing.html", "Publishing"),
]

MD_LINK = re.compile(r'href="([^"]*?)\.md(#[^"]*)?"')
H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)

TEMPLATE = """<!doctype html>
<html lang="en" data-theme="{theme}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AI Engineer Roadmap</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="{rel}assets/style.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%23111'/><text x='16' y='22' font-size='14' font-family='monospace' fill='%23fff' text-anchor='middle'>AI</text></svg>">
</head>
<body>
<div id="progress"></div>
<header class="topbar">
  <button class="icon-btn" id="menuBtn" type="button" aria-label="Menu">
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 4h12M2 8h12M2 12h12"/></svg>
  </button>
  <a class="brand" href="{rel}index.html"><span class="brand-mark">AI</span><span class="brand-text">Engineer Roadmap</span></a>
  <div class="spacer"></div>
  <div class="search-wrap">
    <svg class="search-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5L14 14"/></svg>
    <input id="q" type="search" placeholder="Search lessons…" autocomplete="off" aria-label="Search lessons">
    <div id="results"></div>
  </div>
  <select id="theme" class="theme-select" aria-label="Style"></select>
</header>
<div class="layout">
  <aside class="sidebar" id="sidebar">
{sidebar}
  </aside>
  <main>
    {crumbs}
    <article class="prose">
{content}
    </article>
    {pager}
    <footer class="site-footer">
      AI Engineer Roadmap · <a href="{rel}contributing.html">Contribute</a> ·
      <a href="{rel}glossary.html">Glossary</a> ·
      <a href="{rel}progress.html">Progress</a> · MIT licensed
    </footer>
  </main>
</div>
<button id="toTop" type="button" aria-label="Back to top">
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 13V3M3 8l5-5 5 5"/></svg>
</button>
<script src="{rel}assets/app.js"></script>
</body>
</html>
"""


def title_of(text: str, fallback: str) -> str:
    m = H1.search(text)
    return re.sub(r"^\d+\.\s*", "", m.group(1)).strip() if m else fallback


def stage_title(folder: str) -> str:
    """01-programming-basics -> Programming Basics"""
    return folder.split("-", 1)[1].replace("-", " ").title()


def render(text: str) -> str:
    md = markdown.Markdown(
        extensions=["fenced_code", "codehilite", "tables", "attr_list", "sane_lists", "md_in_html", "toc"],
        extension_configs={"codehilite": {"css_class": "code-block", "guess_lang": False}},
    )
    html = md.convert(text)
    html = re.sub(r'<li>\[ \] ', '<li class="task"><input type="checkbox" disabled> ', html)
    return MD_LINK.sub(lambda m: f'href="{m.group(1)[:-6] if m.group(1).endswith("README") else m.group(1)}.html{m.group(2) or ""}"', html)


def snippet(text: str, limit: int = 700) -> str:
    body = FENCE.sub(" ", text)
    body = re.sub(r"^#.*$", " ", body, flags=re.MULTILINE)
    body = re.sub(r"[`*>|_\[\]()#]", " ", body)
    return re.sub(r"\s+", " ", body).strip()[:limit]


def collect() -> list[dict]:
    pages: list[dict] = []
    for src, out, label in ROOT_PAGES:
        p = ROOT / src
        if p.exists():
            text = p.read_text(encoding="utf-8")
            pages.append({"src": p, "out": out, "stage": "Roadmap", "folder": None,
                          "title": title_of(text, label), "depth": 0})
    for folder in sorted((ROOT / "stages").iterdir()):
        if not folder.is_dir():
            continue
        name = stage_title(folder.name)
        readme = folder / "README.md"
        if readme.exists():
            pages.append({"src": readme, "out": f"stages/{folder.name}/index.html",
                          "stage": f"Stage {folder.name[:2]} · {name}", "folder": folder.name,
                          "title": name, "depth": 2, "index": True})
        for f in sorted(folder.glob("*.md")):
            if f.name == "README.md":
                continue
            pages.append({"src": f, "out": f"stages/{folder.name}/{f.stem}.html",
                          "stage": f"Stage {folder.name[:2]} · {name}", "folder": folder.name,
                          "title": title_of(f.read_text(encoding="utf-8"), f.stem), "depth": 2})
    return pages


def sidebar(pages: list[dict], current: dict, rel: str) -> str:
    out = [f'<a class="nav-home" href="{rel}index.html">Roadmap home</a>',
           f'<a class="nav-home" href="{rel}fast-track.html">Fast track</a>',
           f'<a class="nav-home" href="{rel}progress.html">Progress tracker</a>',
           f'<a class="nav-home" href="{rel}glossary.html">Glossary</a>']
    folders = sorted({p["folder"] for p in pages if p["folder"]})
    for folder in folders:
        items = [p for p in pages if p["folder"] == folder]
        idx = next((p for p in items if p.get("index")), None)
        lessons = [p for p in items if not p.get("index")]
        is_current = current.get("folder") == folder
        out.append(f'<div class="nav-group{" open" if is_current else ""}">')
        out.append(f'<button class="nav-stage{" current" if is_current else ""}" type="button">'
                   f'<span class="num">{folder[:2]}</span><span>{stage_title(folder)}</span>'
                   '<svg class="chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">'
                   '<path d="M6 3l5 5-5 5"/></svg></button>')
        out.append('<ul class="nav-list">')
        if idx:
            active = ' class="active"' if current["out"] == idx["out"] else ""
            out.append(f'<li><a{active} href="{rel}{idx["out"]}">Stage overview</a></li>')
        for p in lessons:
            active = ' class="active"' if current["out"] == p["out"] else ""
            out.append(f'<li><a{active} href="{rel}{p["out"]}">{p["title"]}</a></li>')
        out.append("</ul></div>")
    return "\n".join(out)


def main() -> None:
    if not ASSETS.exists():
        sys.exit("docs/assets is missing — it holds style.css and app.js, which this script does not generate.")

    pages = collect()
    order = [p["out"] for p in pages]
    index: list[dict] = []

    for page in pages:
        text = page["src"].read_text(encoding="utf-8")
        rel = "../" * page["depth"]
        i = order.index(page["out"])
        prev_p = pages[i - 1] if i else None
        next_p = pages[i + 1] if i + 1 < len(pages) else None

        pager = ['<nav class="pager">']
        pager.append(f'<a class="prev" href="{rel}{prev_p["out"]}"><span class="dir">Previous</span>{prev_p["title"]}</a>'
                     if prev_p else '<span class="spacer-a"></span>')
        pager.append(f'<a class="next" href="{rel}{next_p["out"]}"><span class="dir">Next</span>{next_p["title"]}</a>'
                     if next_p else '<span class="spacer-a"></span>')
        pager.append("</nav>")

        crumbs = f'<div class="crumbs"><a href="{rel}index.html">Roadmap</a>'
        if page["folder"]:
            crumbs += (f'<span>/</span><a href="{rel}stages/{page["folder"]}/index.html">'
                       f'Stage {page["folder"][:2]} · {stage_title(page["folder"])}</a>')
        crumbs += f'<span>/</span>{page["title"]}</div>'

        target = OUT / page["out"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            TEMPLATE.format(
                theme="github-light",
                title=page["title"],
                desc=snippet(text, 160).replace('"', "&quot;"),
                rel=rel,
                sidebar=sidebar(pages, page, rel),
                crumbs=crumbs,
                content=render(text),
                pager="\n".join(pager),
            ),
            encoding="utf-8",
        )
        index.append({"t": page["title"], "u": page["out"], "s": page["stage"], "x": snippet(text)})

    (ASSETS / "search-index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"built {len(pages)} pages into {OUT}")


if __name__ == "__main__":
    main()
