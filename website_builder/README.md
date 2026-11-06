# Website builder

The website builder makes the website quadriga-lib.org in the folder `website/` at the repository root. `website/` is ignored by git and is the deploy root: its content is copied to the server as it is. The site has the landing page, the three API pages, the content pages, the publications, the tutorials, the legal pages and a 404 page, plus every Markdown file of `docs/`.

It builds on the [docs builder](../docs_builder/README.md): it imports its parser, content model and Markdown writer from `docs_builder/qdocs/`, runs the docs phase in the same process and writes `docs/` too. The import goes one way only: `qdocs` never imports the website builder. The doc-block format, the content-page format, the gap markers and the Markdown twins are described in the docs-builder README; this README covers what only the website needs.

It runs on the maintainer's Linux machine and needs tools that the release builds do not have (see [Set up a build machine](#set-up-a-build-machine)). `website_builder/` is not part of the source package (`export-ignore` in `.gitattributes`) and not part of the Docker context of the Debian build (`.dockerignore`); in an unpacked source package, `make website` stops with a message saying so. The pip sdist does carry it, like every tracked file.

## Set up a build machine

The website builder needs Python 3.10 or newer with markdown-it-py, mdit-py-plugins, Pygments and Jinja2, and pandoc 3.7 or newer on `PATH`; the tests also need pytest. Set this up once per machine, in one of two ways.

1. Conda (any Linux; the documented path):

```bash
conda create -y -n quadriga-web -c conda-forge "python>=3.11" "markdown-it-py>=3.0" "mdit-py-plugins>=0.4.2" pygments "jinja2>=3.1" pytest "pandoc>=3.7"
conda activate quadriga-web
```

2. Ubuntu 26.04 or newer (older releases ship an older pandoc; use conda there):

```bash
sudo apt install pandoc python3-markdown-it python3-mdit-py-plugins python3-pygments python3-jinja2 python3-pytest
```

With conda, activate the environment before each build session. To check the set-up, run `make website`: if anything is missing, it stops before reading any source, prints one line per missing item (Python too old, no TOML reader, each missing Python package, pandoc not on `PATH`), the Python in use and both set-up lines, and exits with code 2. The build does not check versions; the minimums are what the set-up lines install. pandoc 3.7.0.2, 3.9.0.2 and 3.13 are known to work; the MathML differs in small attributes between pandoc versions, and the code colouring may change with the Pygments version. Upgrade with `conda update -n quadriga-web --all` or with the system's package updates. Nothing is installed into the working copy, and a build never downloads anything.

## Quick start

```bash
make website
python3 -m http.server --directory website
```

Then open `http://localhost:8000/`. `make website` calls `python3 website_builder/build.py website` with the `python3` of the active environment. Direct calls, from any folder:

```bash
python3 website_builder/build.py website
python3 website_builder/build.py website --release
python3 website_builder/build.py website --root /path/to/repository
```

| Option | Meaning |
|--------|---------|
| `website` | The one command: the docs phase, then the site, the checks, and writing `docs/` and `website/`. |
| `--release` | Also fail if any gap is left (see [Gaps and `--release`](#gaps-and---release)). Use it for the build that is deployed. |
| `--root DIR` | Repository root; default: the folder above `website_builder/`. Code, templates, static files and `pandoc_math.yaml` always come from the `website_builder/` next to `build.py`; sources and output are taken below `--root` (the tests use this). |

Exit codes, the diagnostics format and the build-in-memory rule are those of the docs builder (see "Scripts and options" in its README). A successful build first prints one `wrote docs/...` or `deleted docs/...` line per changed file of `docs/` and a line saying how many files of `website/` were written, deleted and left unchanged; then come the errors and warnings, the gap list and the summary `website: N errors, M warnings, K gaps` (with `--release`: `website --release: ...`). A failed build writes neither `docs/` nor `website/` and prints no file lines.

## Folder layout

| Path | Content | Lands in `website/` as |
|------|---------|------------------------|
| `build.py` | Entry point: checks the tools, then runs `qsite.cli`. | — |
| `qsite/cli.py` | The `website` command: runs the docs phase, reads the website-only pages and notebooks, renders, checks and writes. | — |
| `qsite/mdparse.py` | markdown-it set-up for doc blocks, content pages and notebook cells, with the builder's rules (`[[name]]` links, raw links, `<br>`, heading ids, note boxes, site URLs). | — |
| `qsite/render_html.py` | HTML output: pages from the templates, code colouring, the formula step (pandoc), the gap pass. | every `.html` file |
| `qsite/checks.py` | Link and anchor check of the whole site, HTML well-formedness, size budgets. | — |
| `qsite/tutorials.py` | Notebook reading, tutorial pages and their twins. | `tutorials/` |
| `qsite/assets.py` | Static files, landing figures, the preview image, the skill copy; the constants `STATIC_DIR`, `TEMPLATES_DIR`, `LANDING_FIGURES`. | — |
| `templates/base.html` | Page frame: head, skip link, header, `<main id="content">`, footer. | — |
| `templates/_header.html` | Menu (grouped), logo, name, version, GitHub link, tab row from `PAGES`. | — |
| `templates/_footer.html` | Four link columns, version line, Impressum and Privacy; "This page as Markdown" on pages with a twin. | — |
| `templates/_logo.svg` | The logo, inline (`fill="currentColor"`). | — |
| `templates/landing.html` | The landing page with its text. | `index.html` |
| `templates/api.html`, `_api_index.html`, `_fn_entry.html` | API page, its index (sidebar and "Sections and functions"), one function entry. | `cpp_api.html`, `mex_api.html`, `python_api.html` |
| `templates/text.html`, `_text_index.html` | Content page and its index. | `download.html`, `faq.html`, `ai_agents.html` |
| `templates/formats.html`, `_formats_index.html` | The formats page: intro, one section per format file, and its index. | `formats.html` |
| `templates/publications.html` | Publications. | `publications.html` |
| `templates/legal.html` | Impressum and privacy. | `impressum.html`, `privacy.html` |
| `templates/notfound.html` | Not-found page. | `404.html` |
| `templates/tutorials_index.html`, `tutorial.html` | Tutorial index and one tutorial. | `tutorials.html`, `tutorials/<stem>.html` |
| `static/css/site.css` | The one stylesheet. | `css/site.css` |
| `static/js/site.js` | Optional extras (see [Design and static parts](#design-and-static-parts)). | `js/site.js` |
| `static/fonts/` | IBM Plex Sans and Mono (seven `woff2` files) and their licence `LICENSE.txt`. | `fonts/` |
| `static/img/` | `favicon.svg`, `favicon-32.png`, `apple-touch-icon.png`, the landing figures in `img/landing/`, the preview image `og-card.png` once it exists. | `img/` |
| `static/favicon.ico`, `static/.htaccess` | Favicon for old browsers; server settings. | site root |
| `static/publications/` | PDF and Markdown versions of the publications, hand-made. | `publications/` |
| `static/quadriga_lib_material_model.md` | The material model description, hand-made. | `quadriga_lib_material_model.md` |
| `content/impressum.md`, `content/privacy.md` | Legal pages. | `impressum.html`, `privacy.html` |
| `content/404.md` | Title and sentence of the not-found page. | `404.html` |
| `content/tutorials.md` | Title, lead, placeholder note and the common text (e.g. the setup) of the tutorial index; see [The tutorial index page](#the-tutorial-index-page). | `tutorials.html`, `tutorials.md` |
| `skill/quadriga-lib/SKILL.md` | The agent skill (a placeholder for now); build variables are filled in. | `skill/SKILL.md` |
| `pandoc_math.yaml` | pandoc defaults file of the formula step. | — |
| `logo/` | Logo sources (SVG and PNG); not deployed. | — |
| `pytest.ini`, `tests/` | Tests of both builders (see [Tests](#tests)). | — |

Everything below `static/` is copied unchanged, except files named `README.md`. Every file of `docs/` except `docs/README.md` is copied byte for byte to the same path in `website/`. `website/` is fully owned by the builder: files it does not make are deleted. It never writes below `download/` or `releases/`, which exist only on the server. `make clean` keeps `website/`; `make tidy` removes it with the `__pycache__/` and `.ipynb_checkpoints/` folders below the builders and `tutorials/`, and clears the tutorials (deletes `tutorials/executed/` and the outputs of the notebooks).

The root `.gitignore` has unanchored patterns that would silently ignore new files at any depth: do not name a file `build.log`, `build*.log`, `test*.txt` or `deploy.py`, and do not name a folder `release`, `dist` or `wheelhouse`.

## Editing content

The content pages with twins (Get started, formats, FAQ, AI agents, API intros, publications) are sources of the docs builder: see "Content pages" in the [docs-builder README](../docs_builder/README.md#content-pages). The parts below are website only.

**Landing text.** The text of the landing page is HTML in `templates/landing.html`. Drafts are marked with gap markers in the template text, as on content pages (docs-builder README, "Gap markers and going live"). The section ids `why`, `paths`, `quickstart`, `apis`, `covers` and `open` are fixed; `index.html#quickstart` is an old URL that must keep working, and the "For team leads" card points to `#covers`. The quick-start code is written raw inside `{% filter highlight("python") %}...{% endfilter %}` (also `matlab`, `cpp`, `bash`), which colours it like a code fence. The template sets title and description itself.

**Legal pages and 404.** `content/impressum.md`, `content/privacy.md` and `content/404.md` have the front matter and body of a content page, without twin and without index. The privacy text must name what the site stores in the browser, the remembered code language of `static/js/site.js` (see [Design and static parts](#design-and-static-parts)); change the two together. The 404 page is served for any missing URL, so all its links are root-absolute (`/cpp_api.html`).

**Formats page.** `formats.html` is made from `docs_builder/content/formats.md` (title, lead and intro) and one file per format in `docs_builder/content/formats/` (docs-builder README, "Content pages"). Each format file becomes a section of the page, in the order of its `order` key: an `h2` with the format's title and the file stem as id (`formats.html#<stem>`), a "Markdown" link to its twin `formats/<stem>.md`, and its text, whose headings move down one level (`##` becomes `h3`, `###` `h4`). The index "On this page" has one group per format with its `##` headings. A new file in `docs_builder/content/formats/` appears on the page with the next `make website`; nothing in the templates changes. The page's "Markdown version" link points to `formats.md`, which links the format files.

**Publications** are kept in `docs_builder/content/publications.toml` (docs-builder README, "Publications data"); the files they link to go into `static/publications/`.

**Navigation.** The tab row comes from `PAGES` in `docs_builder/qdocs/config.py`. The menu groups (Start, API reference, More, Project) are written in `templates/_header.html`, the footer columns in `templates/_footer.html`, both with the helper `url(page_id)`, which gives the page's URL from the current page. The current tab and menu entry get `aria-current="page"`.

**Add a page.** For a page with a twin, follow "Add a page" in the docs-builder README, then add the page to the menu in `templates/_header.html` and to a footer column in `templates/_footer.html`. A page with a new layout also needs a template and a line in `render_site()` in `qsite/render_html.py`. A website-only text page (like the Impressum) needs only its source `website_builder/content/<id>.md` (front matter and body of a content page) and a line `("<id>", None, None)` in `PAGES` in `docs_builder/qdocs/config.py`; it is rendered with `templates/legal.html` as `<id>.html`, without tab, twin or index. Link it from the footer or a page so that visitors find it.

**Landing figures.** The landing page has two figure slots, `coverage` (next to the pitch) and `antenna` (next to the quick start); the names are `LANDING_FIGURES` in `qsite/assets.py`, the alt texts and captions are in `templates/landing.html`. Each figure is two files, `static/img/landing/<name>-light.webp` and `static/img/landing/<name>-dark.webp`, committed with the code. Rules: WebP, 1600 × 1200 px (4 : 3, the frame crops nothing), at most 150 kB each; backgrounds `#ffffff` and `#0b120e` (or transparent), text and rules in the theme's `--fg`, `--muted` and `--line` colours; a sequential colour map that does not depend on red/green; text at least 28 px high at 1600 px width; the caption names the library function that made the figure. While a variant is missing, the slot shows an empty frame with the alt text, and the gap list names the file.

**Preview image.** `static/img/og-card.png` (1200 × 630 px) is the image a messenger shows for a shared link. The `og:image` and `twitter:card` tags are written only when the file exists; until then the gap list names it.

## Tutorials

Tutorials are Jupyter notebooks `tutorials/<stem>.ipynb` at the repository root, committed without outputs. The tutorial build (`make tutorials`) runs them into executed copies `tutorials/executed/<stem>.ipynb`, which git ignores. How tutorials are written, named, run and checked, including the notebook metadata, headings, alt texts, links and outputs, is described in [tutorials/README.md](../tutorials/README.md); this section covers what the builder does with them.

The builder publishes one tutorial per notebook `tutorials/<stem>.ipynb` and reads its content from the executed copy, with the standard library; it never runs, converts or changes anything in `tutorials/`. Its messages name the notebook. Without `tutorials/executed/` the tutorial index lists no tutorial; a notebook without an executed copy is left out. Both give the warning `not-executed`, which is an error with `--release`, so a release build has every tutorial. An executed copy whose notebook was deleted is ignored.

It checks the metadata `metadata.quadriga_tutorial` (keys and types as in the tutorials README) and `metadata.kernelspec.language` (`python`, `matlab` or `octave`), and takes the alt text of a cell's images from the cell metadata `quadriga_tutorial.alt` (without it: "Figure output of cell n" and the warning `figure-alt`). Headings: `##` becomes `h2`, `###` `h3`, deeper headings `h4`; ids are made as on content pages; a `#` heading is an error, because the page title comes from the metadata. The Markdown version keeps the heading levels as written. Links in Markdown cells are site paths from the site root, as on content pages, and are made relative to the tutorial page.

What is shown: Markdown cells like a content page; code cells as "In [n]" with the code coloured; `stream` output as text (colour codes removed); of each result or display bundle the first of SVG, PNG and plain text. Images are written to `tutorials/<stem>/fig-<cell>-<output>.<ext>`. A bundle with none of the three is left out (warning `output-type`); a code cell without execution count gives `not-run`; an `error` output stops the build.

Each tutorial gives `tutorials/<stem>.html`, its Markdown version `tutorials/<stem>.md`, the executed notebook as `tutorials/<stem>.ipynb` for download and its figures; the page links "Download notebook .ipynb" and "Markdown .md". The index `tutorials.html` and its twin `tutorials.md` show the common part of `content/tutorials.md` and list the tutorials; without tutorials the list is left out. The tutorial pages and their twins exist only on the website, never in `docs/`.

### The tutorial index page

`website_builder/content/tutorials.md` is a content page in the format of the docs builder (see "Content pages" and "File format" in the docs-builder README): TOML front matter between two `+++` lines, then the Markdown body. It holds the part of the tutorial index that is common to all tutorials:

| Part | Shown |
|------|-------|
| `title` | Page title (`h1`) and title of the twin |
| `description` | Meta description of the page |
| `lead` | Lead paragraph under the title |
| `dummy_note` | Warning box above the list when every tutorial is a placeholder (`dummy` in its metadata) |
| Body | Between the lead and the list of tutorials: the setup that all tutorials need, and anything else that applies to all of them; `##` sections with ids as on other content pages (e.g. `## Setup` gets the id `setup`, so a notebook links to it as `tutorials.html#setup`) |

Notes on a single tutorial or a family of tutorials (e.g. extra packages of one series) belong in the notebooks, not here. Gap markers are allowed in all parts; each fails `--release`.

Notebook errors stop the build; each names the notebook, and the cell where there is one:

| Error | Cause | Fix |
|-------|-------|-----|
| `notebook file name must match [a-z0-9][a-z0-9_-]*.ipynb` | Upper-case letters, spaces or other characters in the file name | Rename the file |
| `not a JSON notebook: ...` | The file is not valid JSON (for example a merge conflict) | Repair or re-save it in Jupyter |
| `metadata.quadriga_tutorial: missing` | The notebook has no tutorial metadata | Add the keys listed in the tutorials README |
| `metadata.quadriga_tutorial.<key>: missing`, `not a string`, `not an integer`, `unknown key` | A key of the metadata table in the tutorials README is absent, of the wrong type or misspelt | Correct the metadata |
| `metadata.quadriga_tutorial.level: one of beginner, intermediate, advanced` | Another level | Use one of the three |
| `metadata.kernelspec.language: missing or not one of python, matlab, octave` | The notebook was saved without a kernel, or with another one | Select the Python, MATLAB or Octave kernel and save |
| `cell N: a notebook starts its sections with ##; ...` | A `#` heading in a Markdown cell | Make it `##`; the page title is `title` of the metadata |
| `cell N: the notebook was committed with a failing cell (...)` | An `error` output is stored in the executed copy | Fix the cell, run the tutorials again |
| `tutorials/executed: missing: ...`, `no executed copy ...` (with `--release`) | The tutorials have not been run, or a notebook was added since | `make tutorials` |
| `cell N: E-anchor: ...` | Two headings with the same explicit id, or an id the templates use | Change one of the ids |

## Formulas

The formula syntax is described in the docs-builder README ("Formulas"). On the website every formula becomes MathML: the build collects all formulas of all pages, converts them with one pandoc call (`pandoc -d website_builder/pandoc_math.yaml -f markdown -t html`, no call when there is no formula) and puts the result back into the pages. The defaults file `pandoc_math.yaml` selects MathML; it is used instead of a command-line flag because the flag's name changed between pandoc versions. Browsers render MathML without a script; no math fonts are shipped.

pandoc reads LaTeX math, not full LaTeX. Known to work: accents with subscripts, primes, `\pm`, `\sim`, `\mathrm{d}`, `\frac12`, `\underbrace`, `\arg`, `\lvert x \rvert`, `\cdot`, the `aligned` environment; `\mathbf{x}` comes out bold italic (vector style). These stop the build:

| Error | Cause | Fix |
|-------|-------|-----|
| `formula not converted to MathML (unknown command or unbalanced braces?): <tex>` (at the formula's file and line; in a notebook its cell) | pandoc left the formula as text: an unknown command, unbalanced braces or an environment it rejects | Correct the LaTeX or use a command from the list above |
| `pandoc: error: formula conversion failed:` followed by pandoc's own messages | pandoc printed a warning or exited with an error; it comes together with the error above for the formula concerned | As above; the messages name the command pandoc did not understand |
| `equation labels ($$...$$ (label)) are not supported` | A display formula with a label such as `$$...$$ (1)` | Remove the label; refer to the formula in words |

## Design and static parts

**Styles.** `static/css/site.css` is the only stylesheet. Every colour is a custom property on `:root`; the dark values replace them inside `@media (prefers-color-scheme: dark)`, so the site follows the device setting. Components use these tokens rather than literal colours (exceptions: white text on the green buttons, the print styles); all text pairs meet WCAG 2.2 AA contrast in both themes, so check the ratio when you change a colour. `theme-color` in `templates/base.html` repeats `--bar` of both themes (`#2e7d46`, `#24603a`). Layouts are mobile first, with breakpoints at 600 px and 960 px and a smallest supported width of 360 px; wide code blocks and tables scroll inside their box. Supported browsers: current Chrome, Edge, Firefox and Safari, desktop and mobile. Print styles hide the header, footer, index and copy buttons.

**Fonts.** IBM Plex Sans (400, 400 italic, 500, 600, 700) and IBM Plex Mono (400, 500), Latin subset, self-hosted under the SIL Open Font License 1.1; the licence file `static/fonts/LICENSE.txt` is published next to them and must stay. The Latin subset has no Greek letters and no arrows; such characters fall back to the system font.

**JavaScript.** `static/js/site.js` is optional: every page works without it. It adds copy buttons on code blocks, remembers the chosen language of the landing quick start (`localStorage` key `qlib-lang`, written only on a click), the function filter in the API index (the key `/` focuses it) and the highlight of the function being read in the sidebar. Elements with the class `js-only` (the filter field) are shown only when the script runs.

**Code colouring.** Pygments splits the code into tokens, and the builder writes a `<span>` only for coloured tokens: comments `c`, types `kt`, keywords `k`, function names `nf`, strings `s`, numbers `m`, prompts `gp`; their colours are the `--c-*` tokens of the stylesheet. The fence languages are listed in the docs-builder README ("Writing pitfalls").

**Head of every page.** Title, description, Open Graph tags with absolute URLs, the favicons, `<link rel="alternate" type="text/markdown">` on pages with a twin, the preload of the 400 Sans font, the stylesheet, the script with `defer`. `img/favicon.svg` is the logo with its own colours for light and dark; `img/favicon-32.png`, `img/apple-touch-icon.png` (180 × 180) and `favicon.ico` were rendered from it once and are committed.

## Checks

The build checks the whole site in memory before it writes anything:

- **Links and anchors.** Every `href`, `src` and `srcset` of every HTML page that is not absolute or `mailto:` must point to a file of the site; a fragment must be an `id` on an HTML target or an `<a id>` in a generated Markdown target. The same for every link and image in the generated Markdown (twins, API files, `llms.txt`, `docs/README.md`, tutorial twins); there, absolute links to `http://quadriga-lib.org/...` are checked against the site as well. Hand-made files (`publications/`, the material model) are checked as link targets only. External links and links below `download/` and `releases/` are not fetched. Errors: `link target missing`, `anchor missing`.
- **Well-formedness** of every HTML page: every element closed in order, no `id` twice, exactly one `h1`, exactly one `<main id="content">`, an `alt` on every `img`. Errors: `<x> not closed`, `</x> closes ...`, `id '...' appears twice`, `N <h1> elements`, `<img> without alt`.
- **Size budgets** in raw bytes, since the server sends no compressed responses (warning `size`): each API page 800 kB; each other HTML page 150 kB; the landing page with `css/site.css`, `js/site.js`, the four fonts it uses (Sans 400, 600, 700 and Mono 400) and the light variant of each landing figure 480 kB; each landing figure 150 kB.
- **Notebooks and formulas**: the errors and warnings of [Tutorials](#tutorials) and [Formulas](#formulas).
- **Doc-block and content errors** of the docs phase: see "Errors and warnings" in the docs-builder README.

### Gaps and `--release`

The gap markers and the gap list are described in the docs-builder README ("Gap markers and going live"); the website build adds the markers of `website_builder/content/` and `website_builder/templates/`, every missing landing figure file and a missing `img/og-card.png` to the list. On the page a marker shows as a yellow highlight, inside attributes and the `<title>` as plain text. `website_builder/build.py website --release` fails, and writes nothing, if the gap list is not empty or if any marker reached the HTML (for example one written into a doc block or a notebook). Build the deployed site with `--release`.

## Tests

The tests of both builders are in `website_builder/tests/`: `tests/docs/` tests the docs builder and imports nothing of the website builder, `tests/site/` tests the website builder. Both use the fixture repository `tests/fixtures/mini_repo/` (version `9.9.9`, a few small doc blocks per language, content files and one small notebook). Tests marked `repo` read the real repository; `tests/fixtures/old_anchors.json` holds the ids of the old website that must keep working. Run them in the active environment:

```bash
python3 -m pytest website_builder/tests
python3 -m pytest website_builder/tests/docs
```

The second line needs only Python and pytest. The tests need no network and write only below pytest's temporary folders.

## Deployment

`website/` is the deploy root: after `website_builder/build.py website --release`, its content is copied to the web server as it is. The folders `download/` (release files) and `releases/` exist only on the server; the builder never writes them, and site links to them stay absolute.

`static/.htaccess` (Apache) turns off folder listings, serves `/404.html` for missing pages, sets the content types of `.md` (`text/markdown`), `.woff2` and `.webp`, and the UTF-8 charset of HTML, CSS, JavaScript, Markdown, text and SVG files. It holds a commented-out redirect to HTTPS: switch it on only once HTTPS works at the hoster, and change `SITE_BASE_URL`, `DOWNLOAD_BASE_URL` and `RELEASES_URL` in `docs_builder/qdocs/config.py` to `https://` at the same time. Compression is not on.

After a deploy that changes the server set-up, check with `curl -sI`: `<base>/formats.md` shows `text/markdown; charset=utf-8`, `<base>/llms.txt` shows `text/plain; charset=utf-8`, and a missing page returns 404 with the new page. To try compression, add `AddOutputFilterByType DEFLATE text/html text/css application/javascript text/markdown text/plain` to the server's `.htaccess`; if `curl -sI -H 'Accept-Encoding: gzip' <base>/cpp_api.html` then shows `Content-Encoding: gzip` and a plain request still works, add the line to `static/.htaccess`.

## License

The builder code is under the Apache License 2.0, see [LICENSE](../LICENSE). The fonts in `static/fonts/` are under the SIL Open Font License 1.1, see `static/fonts/LICENSE.txt`.

## See also

- [docs_builder/README.md](../docs_builder/README.md): the doc-block and content formats, errors and warnings, gap markers, the Markdown files and the release checks.
- [CONTRIBUTING.md](../CONTRIBUTING.md): issues, pull requests, and keeping this README current with the code.
