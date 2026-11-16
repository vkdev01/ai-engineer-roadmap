# Publishing this repo as a GitHub Pages site

The repository ships with a pre-built static site in `docs/`. It needs no build step, no Jekyll plugins and no Node toolchain on the server — GitHub Pages serves it as plain files.

## Turn it on (2 minutes)

1. Push the repo to GitHub.
2. **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to `Deploy from a branch`.
4. Branch: `main`, folder: `/docs`. Save.
5. Wait for the first deploy (usually under a minute) and open `https://<user>.github.io/<repo>/`.

The site appears at that URL. If you rename the repo, nothing breaks — all links inside the site are relative.

`docs/.nojekyll` is already committed. It stops Jekyll from trying to process the folder, which is what you want for a pre-rendered site.

## The five styles

The site ships with a style switcher in the top bar. The choice is remembered per visitor in `localStorage`.

| Style | Look | Best for |
|---|---|---|
| **GitHub light** (default) | White, system sans, blue links, subtle borders | Reading like documentation; closest to GitHub |
| **GitHub dark** | GitHub's dark palette | Night reading, matches dark editor themes |
| **Book (paper)** | Warm cream, serif headings, narrow measure | Long-form reading, printing, tablets |
| **Slate (dark)** | Navy and indigo, tighter tracking, larger headings | Screenshots, presentations, a "product docs" feel |
| **Terminal** | Near-black, monospace everything, green accent | People who live in a terminal |

### Changing the default style

Every theme is a block of CSS variables at the top of `docs/assets/style.css`. To change what visitors see first, edit one line in the page template — the `data-theme` attribute:

```python
# build_site.py (repo tooling, not part of the published site)
theme = "github-light"      # or "github-dark", "book", "slate", "terminal"
```

Then rebuild:

```bash
python build_site.py
```

### Adding your own style

1. Copy any `html[data-theme="..."]` block in `docs/assets/style.css`.
2. Rename it, change the variables — colours, fonts, radius, measure, heading weight.
3. Add the pair to the `THEMES` array at the top of `docs/assets/app.js`:

```js
var THEMES = [
  ["github-light", "GitHub light"],
  ["github-dark", "GitHub dark"],
  ["book", "Book (paper)"],
  ["slate", "Slate (dark)"],
  ["terminal", "Terminal"],
  ["your-theme", "Your theme"]     // add this
];
```

That is the whole contract: a variable block plus a menu entry.

## What the site gives you

- Sidebar navigation for all 23 stages and 210 lessons, with the current stage expanded
- Client-side search across every lesson title and body (`/` or `Ctrl/Cmd+K` to focus)
- Copy buttons on every code block
- Previous/next navigation in reading order across the whole book
- Reading-progress bar, back-to-top, mobile drawer, print stylesheet
- Syntax highlighting, tables, task-list checkboxes on the progress page
- A `404.html` that sends people back to the roadmap

## Rebuilding after editing lessons

The site is generated from the markdown, so editing a lesson means rebuilding:

```bash
python build_site.py     # writes docs/ from stages/**/*.md and the root files
```

Commit both the markdown change and the regenerated `docs/` folder, so the published site never drifts from the source.

## Alternative: a Jekyll theme instead

If you would rather let GitHub build the site with a Jekyll theme, delete `docs/.nojekyll`, add a `_config.yml` at the repo root and point Pages at the root folder. The markdown files render fine under any Jekyll theme — `just-the-docs`, `minima`, `cayman` and `slate` all work, since the lessons use plain headings, tables and fenced code.

The trade-off is control: Jekyll themes give you less say over the reading layout and no in-page style switcher. The pre-built site in `docs/` is the better default for a book this size.

## Custom domain

Add a `CNAME` file in `docs/` containing your domain, then configure the DNS record as GitHub's Pages documentation describes. Keep the `CNAME` file in the repo or the setting resets on the next deploy.
