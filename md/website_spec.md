# Quadriga-Lib website overhaul: specification

This is the specification (step 3 of `md/website_plan.md`). Implementation sub-agents (step 5) build the new documentation builder, templates, static parts and tests from it alone. It refers to the plan by decision number (D1 to D31) and section, and to `md/website_design.md` as "design" with its section number. Step 7 deletes this document; it stays in the git history.

## 1. Scope and conventions

### 1.1 What this spec covers

The new builder in `docs_builder/`, its outputs in `docs/` and `docs/website/`, the page templates, the static parts (CSS, script, fonts, images, `.htaccess`), the tutorial pipeline with one dummy notebook in `tutorials/`, the agent skill placeholder, the package-install failsafe `docs_builder/CMakeLists.txt`, the builder's tests, the new `GNUmakefile` targets, the `.gitignore` additions, the cutover checklist (D31) and the work packages for step 5. It does not cover the landing-figure scripts and images (D15), the skill content (D10), real tutorials (D11), the deploy script (D29) or any change to files D20 protects.

### 1.2 Precedence and references

Where texts differ: D24 to D31, then the rest of the plan, then the design, then the prototype in `html_builder/design_prototype/`, then this spec's reading of the prototype. Paths in the design and prototype are mapped as follows:

| Design / prototype | This spec |
|--------------------|-----------|
| `html_docu/` (output) | `docs/website/` (built site) and `docs/` (tracked Markdown) |
| `html_builder/` (builder, page sources) | `docs_builder/` |
| `md/cpp/<section>.md` | `docs/api/cpp/<section-slug>.md` (and `mex`, `python`) |
| `index.md` and other twins on every page | twins only as in D26 (section 8.6) |
| prototype `css/site.css`, `js/site.js`, `fonts/`, `img/logo.svg` | `docs_builder/static/css/site.css`, `static/js/site.js`, `static/fonts/`, `static/img/logo.svg` |

### 1.3 Terms

| Term | Meaning |
|------|---------|
| Doc block | A `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` comment block in `src/`, `api_mex/` or `api_python/` (D6). |
| Language, `lang` | One of `cpp`, `mex`, `python`; display names "C++", "MATLAB / Octave", "Python". |
| Function | One `/*!MD` block. Its anchor is its name exactly as on today's pages (section 9). |
| Family | A MATLAB/Python function pair matched by file stem, plus its C++ counterpart(s) (section 9.4). |
| Fragment | A list of markdown-it-py block tokens after the builder's own rules ran (section 5.6); the leaf content of the model. |
| Twin | The Markdown version of an HTML page (D26). |
| Gap | A content placeholder written with the gap marker (section 6.4) or a missing figure (section 12.6). |
| Build variable | A value filled in at build time, written `{{ name }}` (section 6.3). |
| Site root | `docs/website/`; URLs in this spec are relative to it unless absolute. |
| Repository root | The folder holding `CMakeLists.txt`; default is the parent of `docs_builder/`. |
| Structural error | A defect that stops the build with file and line (D7). |
| Style warning | A defect that prints a warning and does not stop the build (D7). |

### 1.4 Notation

`path:line` means a file path relative to the repository root and a 1-based line number. "Must" is a requirement; "should" is a default an implementer may deviate from only with a reason in the hand-over report.

## 2. File tree

### 2.1 New folders and files

Everything below is new. "Tracked" means committed by the maintainer (D19); "ignored" means listed in `.gitignore` (section 2.4); "generated" means written by the builder; "hand-made" means written once by a step-5 package or by the maintainer and then maintained by hand.

| Path | Kind | Purpose |
|------|------|---------|
| `docs/` | tracked, generated | Markdown tree (D8, D24): twins, per-section API files, full API files, `llms.txt`, `README.md`. Fully owned by the builder: files in it that the builder does not write are deleted (section 3.6), except `docs/website/`. |
| `docs/website/` | ignored, generated | Built site and deploy root (D24). Fully owned by the builder. |
| `docs_builder/build.py` | tracked, hand-made | Entry point (section 3). Standard library only at import time. |
| `docs_builder/qdocs/` | tracked, hand-made | Builder package (section 3.2). |
| `docs_builder/templates/` | tracked, hand-made | Jinja2 templates (section 8). |
| `docs_builder/katex_render.mjs` | tracked, hand-made | Node script that renders all formulas of one build (section 11). |
| `docs_builder/site.toml` | tracked, hand-made | Site settings, page list, navigation (section 3.3). |
| `docs_builder/content/` | tracked, hand-made | Prose pages (`*.md`), structured pages (`landing.toml`, `publications.toml`), `llms.toml`, API intros, `api_matching.toml` (sections 6, 7, 9.4). |
| `docs_builder/static/` | tracked, hand-made | Files copied unchanged into the site root (section 12). |
| `docs_builder/skill/quadriga-lib/SKILL.md` | tracked, hand-made | Skill placeholder (section 14). |
| `docs_builder/figures/` | tracked, hand-made | Landing-figure scripts (D15; written in a later session) and `og-card-placeholder.svg` (section 12.5). |
| `docs_builder/tests/` | tracked, hand-made | Test suite and fixtures (section 17). |
| `docs_builder/pytest.ini` | tracked, hand-made | pytest settings (section 17.1). |
| `docs_builder/requirements.txt` | tracked | Build packages, minimum versions (section 4). |
| `docs_builder/requirements-test.txt` | tracked | Test packages, minimum versions. |
| `docs_builder/requirements-tutorials.txt` | tracked | Tutorial packages, minimum versions. |
| `docs_builder/package.json` | tracked | npm packages, minimum versions (section 4.3). No lockfile is committed (D4 as amended). |
| `docs_builder/.htmlvalidate.json`, `docs_builder/.pa11yci.json` | tracked, hand-made | Optional validator settings (section 15.5). |
| `docs_builder/CMakeLists.txt` | tracked, hand-made | Package install and failsafe (section 16). |
| `docs_builder/node_modules/` | ignored | npm packages of the venv set-up (section 4.4); the conda set-up installs them into the environment instead. |
| `docs_builder/.venv/` | ignored | Virtual environment of the venv set-up, made by `make docs-setup-venv`. |
| `docs_builder/package-lock.json` | ignored | Written by npm in the venv set-up if `--no-package-lock` is forgotten; never committed. |
| `docs_builder/.cache/` | ignored, generated | Builder cache: `tutorials/` (section 13.4), `staging/` (section 3.6). |
| `tutorials/` | tracked | Notebooks without outputs (D28); one dummy notebook `tutorials/pipeline_demo.ipynb` (section 13.7). |
| `docs_builder/README.md` | tracked | Written in step 7, not step 5. |

The old builder (`html_builder/`), `html_docu/` and every file D20 protects stay unchanged. The only files outside the three new folders that step 5 changes are `GNUmakefile` (new targets, section 3.5) and `.gitignore` (section 2.4).

### 2.2 Builder package `docs_builder/qdocs/`

Module names are fixed so that work packages own disjoint files (section 19). No file in `docs_builder/` may be named `build*.log`, `test*.txt`, `deploy.py`, or sit in a folder named `release`, `dist` or `build` (they would collide with existing ignore patterns; section 2.4 neutralises the first four, the rule keeps the tree clear).

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Version constant `FORMAT_VERSION = 1` (bumped when output formats change). |
| `cli.py` | Argument parsing, command dispatch, exit codes (section 3.1). |
| `env.py` | Python version check, package import checks, Node and kernel checks, the plain messages of section 3.4. |
| `diag.py` | Error and warning collection and printing (section 3.7). |
| `config.py` | Loading and checking `site.toml`; reading the library and Armadillo versions once (section 3.8). |
| `schema.py` | The small schema checker used for `site.toml`, front matter and the TOML data files (section 7.1). |
| `docblocks.py` | Block extraction and the block-level grammar (section 5.1 to 5.5). |
| `model.py` | Dataclasses of the content model (section 5.7). |
| `mdparse.py` | markdown-it-py set-up and the builder's markdown-it rules (section 5.6). |
| `params.py` | Parameter-list recognition (section 5.4). |
| `links.py` | Link targets, `[[...]]` resolution, per-output URL rendering (section 8.7). |
| `anchors.py` | Readable ids, legacy hash ids, GitHub heading slugs (section 9). |
| `matching.py` | Cross-language families and pills (section 9.4). |
| `prose.py` | Front matter, build variables, prose page model (section 6). |
| `structured.py` | Landing and publications data files (section 7). |
| `gaps.py` | Gap marker rule, gap scan and list (section 6.4). |
| `render_md.py` | Markdown renderer: API files, prose twins, publications twin, `llms.txt`, `docs/README.md` (section 8.6 to 8.9). |
| `mdformat_ext.py` | mdformat plugin with renderers for the builder's token types (section 8.6). |
| `render_html.py` | HTML renderer: Jinja2 environment, fragment rendering, page assembly (section 8.1 to 8.5). |
| `highlight.py` | Span-free Pygments formatter (section 10). |
| `katex.py` | Formula collection and the single Node call (section 11). |
| `tutorials.py` | Notebook metadata, execution, cache, fragment and Markdown output (section 13). |
| `skill.py` | Skill zip (section 14). |
| `staticfiles.py` | Static copy, extra sources until the cutover, figure slots, stamp file (section 12). |
| `checks.py` | Link and anchor check, well-formedness, size budgets (section 15). |
| `output.py` | In-memory output tree, writing, stale-file removal, check mode (section 3.6). |

### 2.3 Output files

All paths of generated files. `<lang>` is `cpp`, `mex` or `python`; `<slug>` is a section slug (section 9.2); `<stem>` is a notebook file stem; `<v>` is the library version.

Tracked Markdown in `docs/` (written by `make docs`):

| Path | Content |
|------|---------|
| `docs/README.md` | Index of `docs/` for GitHub (section 8.9). |
| `docs/llms.txt` | Agent entry point (section 8.8). |
| `docs/quadriga_lib_api_<lang>.md` | Full API per language (section 8.6.2). URLs of D3. |
| `docs/api/<lang>/general-usage-notes.md` | The API intro per language. |
| `docs/api/<lang>/<slug>.md` | One file per API section (section 8.6.3). Measured today: 10 C++, 9 MATLAB, 9 Python sections. |
| `docs/download.md`, `docs/formats.md`, `docs/faq.md`, `docs/ai_agents.md`, `docs/publications.md` | Twins of the content pages (D26). |

Site in `docs/website/` (written by `make website`):

| Path | Content |
|------|---------|
| `index.html` | Landing (no twin). |
| `download.html`, `formats.html`, `faq.html`, `ai_agents.html` | Text pages. `ai_agents.html` is the new AI page (D9). |
| `cpp_api.html`, `mex_api.html`, `python_api.html` | API pages. |
| `publications.html` | Publications. |
| `tutorials.html`, `tutorials.md` | Tutorial index and its twin (website only, D28). |
| `tutorials/<stem>.html`, `tutorials/<stem>.md`, `tutorials/<stem>.ipynb`, `tutorials/<stem>/<figure files>` | One tutorial page, its twin, the notebook for download, its figures. |
| `impressum.html`, `privacy.html` | Legal pages (D17); no twins. |
| `404.html` | Not-found page; no twin. |
| every file of `docs/` except `README.md` | Copied byte for byte at the same relative path (D24). |
| `skill/quadriga-lib-skill-<v>.zip` | Skill placeholder zip (section 14). |
| `css/site.css`, `js/site.js`, `fonts/*`, `img/*`, `favicon.ico`, `.htaccess` | Static parts (section 12). |
| `publications/**`, `quadriga_lib_material_model.md` | Hand-made files, read from `html_docu/` until the cutover (section 12.7). |
| `docs-stamp.txt` | Version stamp for the package failsafe (section 16.2). |

The builder must never write a path that starts with `download/` or `releases/` in `docs/website/` (plan section 11); `output.py` rejects such a path with an internal error, and a test checks it.

### 2.4 `.gitignore` additions

Append this block at the end of the root `.gitignore`, unchanged:

```
# Documentation builder (docs_builder/README.md). The "!" lines keep the
# unanchored patterns above (release/, dist/, test*.txt, build*.log) out of
# the new folders; the lines after them ignore what must not be committed.
!/docs/**
!/docs_builder/**
!/tutorials/**
/docs/website/
/docs_builder/node_modules/
/docs_builder/package-lock.json
/docs_builder/.venv/
/docs_builder/.cache/
/tutorials/**/.ipynb_checkpoints/
**/__pycache__/
```

Why: the existing unanchored patterns `release/`, `dist/`, `test*.txt`, `build*.log` and `__pycache__/` match at any depth (measured with `git check-ignore`: `docs_builder/tests/fixtures/test_a.txt`, `docs_builder/tests/fixtures/release/x` and `docs_builder/dist/x` are ignored today). The negations re-include everything below the three new folders; later lines re-exclude the generated and installed parts, because the last matching pattern wins. `.venv/`, `node_modules/` and `package-lock.json` exist only in the venv set-up; the conda set-up keeps all tools outside the repository and needs no ignore line (section 4.4). Measured in a scratch repository with a copy of the current `.gitignore`: with this block, fixture files named `test_a.txt` or inside `release/` are visible to git, while `docs/website/`, `.venv/`, `node_modules/`, `.cache/` and `__pycache__/` stay ignored. `build.py` itself is not caught by any pattern (`build/*` and `build_*/*` are anchored to the root). A test (section 17.2, `test_gitignore.py`) keeps this true.

## 3. Builder entry point, CLI and make targets

### 3.1 Command line

`python3 docs_builder/build.py <command> [options]`, run from any folder (paths are resolved from the repository root).

| Command | Does |
|---------|------|
| `docs` | Parses all sources, renders the tracked Markdown tree and writes `docs/` (section 3.6). Python packages of section 4.1 "docs" only; never starts Node. |
| `docs --check` | Same rendering in memory; writes nothing; compares with `docs/` (D23). Prints each differing, missing or extra file. Exit 3 if anything differs. |
| `website` | Runs `docs` (writes `docs/`), then renders the site, runs the checks of section 15 and writes `docs/website/`. Starts Node only if a page has a formula (D21). |
| `website --release` | Strict mode (plan section 11): additionally fails on remaining gaps, missing figures, the placeholder preview image and tutorials whose cache entry is missing or stale. |
| `tutorials [--force] [STEM ...]` | Executes notebooks whose cache key changed (all with `--force`; only the named ones if stems are given) and writes their cache entries (section 13). |
| `gaps [--release]` | Prints the gap list (section 6.4). Exit 0; with `--release`, exit 1 if the list is not empty. |
| `validate` | Runs the optional npm validators on `docs/website/` (section 15.5). |
| `setup-node [--global] [--validators]` | Installs the npm packages of `docs_builder/package.json` with their minimum-version specs: with `--global` into the active Node installation (`npm install -g`, the conda set-up), otherwise into `docs_builder/node_modules/` (`npm install --no-package-lock --prefix docs_builder`, the venv set-up); `--validators` adds the `devDependencies`. Sets `PUPPETEER_SKIP_DOWNLOAD=1` for the npm call, so no browser is downloaded (section 15.5). The specs are read from `package.json`, so they are written once. |

Options for all commands: `-v` / `--verbose` (print every warning in full and phase timings), `-q` / `--quiet` (print only errors and the summary line), `--root DIR` (repository root; used by tests). Unknown options or commands: usage message, exit 2.

Exit codes: 0 success (warnings allowed); 1 build failed (structural errors, failed checks, strict-mode failures, notebook errors); 2 environment or usage problem (wrong Python, missing package or tool, bad arguments); 3 `docs --check` found differences.

### 3.2 Code layout and imports

`build.py` must run under any Python 3 far enough to print the version message: it contains no syntax newer than Python 3.6, checks `sys.version_info >= (3, 11)` first (message as in section 3.4), then puts its own folder on `sys.path` and calls `qdocs.cli.main()`. Third-party packages are imported inside the functions that need them, through `env.require(...)` (section 3.4), so that `make docs` never imports Pygments, Jinja2 or nbclient.

Code, templates and `katex_render.mjs` are always taken from the folder of `build.py`. Everything else is taken relative to `--root`: sources (`src/`, `api_mex/`, `api_python/`, `include/`, `CMakeLists.txt`), `docs_builder/site.toml`, `docs_builder/content/`, `docs_builder/static/`, `docs_builder/skill/`, `tutorials/`, the outputs `docs/` and `docs/website/`, and the cache `docs_builder/.cache/`. Tests use this to build small fixture repositories.

### 3.3 `docs_builder/site.toml`

The page list and the navigation live here and nowhere else (plan 5.1). Schema-checked like the data files (section 7.1): unknown keys and missing keys are structural errors. Initial content (step 5 writes exactly this; values are settings, not wording, except the navigation labels, which come from the design):

```toml
[site]
name = "Quadriga-Lib"
base_url = "http://quadriga-lib.org"             # no trailing slash; http until HTTPS works (plan section 10)
download_base_url = "http://quadriga-lib.org/download"   # D25: the one setting for release package links
releases_url = "http://quadriga-lib.org/releases/"       # index of older releases (D25)
github_url = "https://github.com/stjaeckel/quadriga-lib"
issues_url = "https://github.com/stjaeckel/quadriga-lib/issues"
license_url = "https://github.com/stjaeckel/quadriga-lib/blob/main/LICENSE"
og_image = "img/og-card.png"
og_image_placeholder = "img/og-card-placeholder.png"
theme_color_light = "#2e7d46"
theme_color_dark = "#24603a"

[sources]
version_header = "include/quadriga_lib.hpp"
armadillo_cmake = "CMakeLists.txt"
cpp = "src"
mex = "api_mex"
python = "api_python"

# Hand-made files the old site also needs; read from their old place until the cutover (D24).
[[static_extra]]
from = "html_docu/publications"
to = "publications"
[[static_extra]]
from = "html_docu/quadriga_lib_material_model.md"
to = "quadriga_lib_material_model.md"

[[page]]
id = "index"
file = "index.html"
template = "landing"
source = "content/landing.toml"
tab = "Home"

[[page]]
id = "download"
file = "download.html"
template = "text"
source = "content/download.md"
twin = "download.md"
tab = "Get started"

[[page]]
id = "cpp_api"
file = "cpp_api.html"
template = "api"
source = "api:cpp"
twin = "quadriga_lib_api_cpp.md"
tab = "C++ API"

[[page]]
id = "mex_api"
file = "mex_api.html"
template = "api"
source = "api:mex"
twin = "quadriga_lib_api_mex.md"
tab = "MATLAB API"

[[page]]
id = "python_api"
file = "python_api.html"
template = "api"
source = "api:python"
twin = "quadriga_lib_api_python.md"
tab = "Python API"

[[page]]
id = "tutorials"
file = "tutorials.html"
template = "tutorials_index"
source = "content/tutorials.md"
twin = "tutorials.md"
twin_website_only = true
tab = "Tutorials"

[[page]]
id = "formats"
file = "formats.html"
template = "text"
source = "content/formats.md"
twin = "formats.md"
tab = "Formats"

[[page]]
id = "publications"
file = "publications.html"
template = "publications"
source = "content/publications.toml"
twin = "publications.md"
tab = "Publications"

[[page]]
id = "ai_agents"
file = "ai_agents.html"
template = "text"
source = "content/ai_agents.md"
twin = "ai_agents.md"
tab = "AI agents"

[[page]]
id = "faq"
file = "faq.html"
template = "text"
source = "content/faq.md"
twin = "faq.md"
tab = "FAQ"

[[page]]
id = "impressum"
file = "impressum.html"
template = "legal"
source = "content/impressum.md"

[[page]]
id = "privacy"
file = "privacy.html"
template = "legal"
source = "content/privacy.md"

[[page]]
id = "404"
file = "404.html"
template = "notfound"
source = "content/404.md"

[nav]
tabs = ["index", "download", "cpp_api", "mex_api", "python_api", "tutorials", "formats", "publications", "ai_agents", "faq"]

[[nav.menu]]
title = "Start"
pages = ["index", "download", "tutorials"]
[[nav.menu]]
title = "API reference"
pages = ["cpp_api", "mex_api", "python_api"]
[[nav.menu]]
title = "More"
pages = ["formats", "publications", "ai_agents", "faq"]
[[nav.menu]]
title = "Project"
links = [{ label = "GitHub", url = "github_url" }]

[[nav.footer]]
title = "Documentation"
items = [{ page = "download", label = "Get started" }, { page = "cpp_api", label = "C++ API" }, { page = "mex_api", label = "MATLAB / Octave API" }, { page = "python_api", label = "Python API" }, { page = "formats", label = "Data formats" }]
[[nav.footer]]
title = "Learn"
items = [{ page = "tutorials", label = "Tutorials" }, { page = "publications", label = "Publications" }, { page = "faq", label = "Questions" }, { page = "ai_agents", label = "AI agents" }]
[[nav.footer]]
title = "Project"
items = [{ url = "github_url", label = "GitHub" }, { url = "issues_url", label = "Issues" }, { href = "faq.html#cite", label = "How to cite" }, { url = "license_url", label = "Apache-2.0 licence" }]
[[nav.footer]]
title = "For agents"
items = [{ href = "llms.txt", label = "llms.txt" }, { twin = true, label = "This page as Markdown" }]
```

Rules: `page.id` unique; `file` and `twin` unique; `template` one of `landing`, `text`, `api`, `tutorials_index`, `publications`, `legal`, `notfound`; `source` is a path below `docs_builder/` or `api:<lang>`; `twin` is optional (pages without it have no twin, D26); `twin_website_only` (default false) keeps the twin out of `docs/` (D28). In navigation items, `url` names a key of `[site]`, `href` is a site-relative link (checked by section 15), `page` names a page id, and `twin = true` renders only on pages that have a twin (D26). The tutorial pages are not listed: the builder adds one page per notebook (section 13). Legal pages and 404 have no `tab`; the menu in the header lists `tab` labels, the footer its own labels (design 4.1, 4.2).

### 3.4 Missing tools: plain messages

`env.require(command, packages)` imports each package; on `ImportError` it prints one message per missing package and exits 2, without a traceback (D21). Messages (`<python>` is `sys.executable`):

| Missing | Message |
|---------|---------|
| Python < 3.11 | `error: Python 3.11 or newer is needed (found 3.X at <python>). Activate the tool environment: conda activate quadriga-web (or: source docs_builder/.venv/bin/activate); to create one, see make docs-setup-conda or make docs-setup-venv` |
| A package of `requirements.txt` | `error: the Python package '<name>' (minimum <version>) is missing for <python>. Activate the tool environment (conda activate quadriga-web, or source docs_builder/.venv/bin/activate) or install it there: <python> -m pip install -r docs_builder/requirements.txt` |
| A package of `requirements.txt` older than its minimum | `error: the Python package '<name>' is <found>, at least <minimum> is needed. Upgrade: <python> -m pip install -U -r docs_builder/requirements.txt` |
| A package of `requirements-tutorials.txt` | the same two messages with `requirements-tutorials.txt` |
| `node` on `PATH` (only when a formula exists) | `error: Node.js is needed because <file>:<line> contains a formula. Activate the tool environment (conda activate quadriga-web, which holds Node.js), or for the venv set-up install Node.js 22 or newer from nodejs.org or with conda; then run: python3 docs_builder/build.py setup-node (add --global in the conda environment)` |
| katex (neither in `docs_builder/node_modules/` nor in `npm root -g`) | `error: the npm package 'katex' is missing. Run in the active environment: python3 docs_builder/build.py setup-node --global (conda) or python3 docs_builder/build.py setup-node (venv)` |
| Node older than 22 | `error: Node.js 22 or newer is needed (found vX.Y.Z).` |
| Jupyter kernel of a notebook | warning, not error: `warning: tutorials/<stem>.ipynb needs the Jupyter kernel '<name>', which is not installed; skipped` (D28: MATLAB notebooks run only where a kernel exists) |
| `html-validate` or `pa11y-ci` for `validate` | `error: the validators are not installed. Run: python3 docs_builder/build.py setup-node --validators (add --global in the conda environment)` |
| Chrome for pa11y | `error: pa11y needs Chrome; set QDOCS_CHROME to its path (default /usr/bin/google-chrome)` |

`env.require` also compares each package's installed version (`importlib.metadata.version`) with the minimum in the requirements file it belongs to (parsed by `env.py`; `packaging` is not needed: versions are compared as tuples of their leading integers). The `docs` command needs only `markdown_it`, `mdit_py_plugins`, `mdformat`, `mdformat_gfm`; the `website` command adds `pygments`, `jinja2`; the `tutorials` command adds `nbformat`, `nbclient`, `jupyter_client` and the website packages.

### 3.5 `GNUmakefile` targets

Rule for finding the tools: the targets use the `python3` and `node` that are on `PATH`; the user activates the tool environment first (`conda activate quadriga-web`, or `source docs_builder/.venv/bin/activate`). A make variable overrides the Python: `make docs DOCS_PY=/path/to/python3`. Why: one rule serves both set-ups and also a direct call of `build.py`; make needs no detection logic, which would have to know conda's layout or guess between two environments that may both exist; and `build.py` reports `sys.executable` in every missing-tool message, so a wrong environment is obvious. The builder finds katex and the validators in `docs_builder/node_modules/` first and then in the active Node installation (`npm root -g`, and the validators' commands on `PATH`), so the same rule covers Node in both set-ups.

Added after the `documentation` target, which stays unchanged (D21, D31):

```make
# New documentation builder (docs_builder/README.md). Activate the tool
# environment first: conda activate quadriga-web (preferred), or
# source docs_builder/.venv/bin/activate.
DOCS_PY ?= python3

docs:
	$(DOCS_PY) docs_builder/build.py docs

docs-check:
	$(DOCS_PY) docs_builder/build.py docs --check

website:
	$(DOCS_PY) docs_builder/build.py website

website-release:
	$(DOCS_PY) docs_builder/build.py website --release

tutorials:
	$(DOCS_PY) docs_builder/build.py tutorials

docs-test:
	$(DOCS_PY) -m pytest -c docs_builder/pytest.ini docs_builder/tests

docs-validate:
	$(DOCS_PY) docs_builder/build.py validate

# Tool environment, conda (preferred): everything in the env "quadriga-web"
docs-setup-conda:
	conda create -y --name quadriga-web -c conda-forge "python>=3.11" "nodejs>=22" pip
	conda run --name quadriga-web python -m pip install -r docs_builder/requirements.txt -r docs_builder/requirements-test.txt -r docs_builder/requirements-tutorials.txt
	conda run --name quadriga-web python docs_builder/build.py setup-node --global --validators

# Tool environment, venv: Python packages in docs_builder/.venv, Node.js from the system
docs-setup-venv:
	python3 -m venv docs_builder/.venv
	docs_builder/.venv/bin/python3 -m pip install -r docs_builder/requirements.txt -r docs_builder/requirements-test.txt
	docs_builder/.venv/bin/python3 docs_builder/build.py setup-node

docs-setup-tutorials-venv: docs-setup-venv
	docs_builder/.venv/bin/python3 -m pip install -r docs_builder/requirements-tutorials.txt

.PHONY: docs docs-check website website-release tutorials docs-test docs-validate docs-setup-conda docs-setup-venv docs-setup-tutorials-venv
```

The comment style follows the existing note on the `quadriga-pip` conda environment in `GNUmakefile` (PyPI packaging). The conda set-up installs the tutorial packages too, because it is the maintainer's full environment; the venv set-up keeps them optional. Upgrading the tools: `python -m pip install -U -r docs_builder/requirements.txt` (and the other files) and `python3 docs_builder/build.py setup-node [--global]` again, in the active environment.

`make website` runs the `docs` phase itself inside one process (the builder parses the sources once), which is how it "depends on make docs" (D21): it always rewrites `docs/` before the site. `make tutorials` is separate and never runs as part of `docs` or `website` (D28).

### 3.6 Output handling (`output.py`)

- Every command builds the complete output set in memory first: a mapping from relative path to bytes, for `docs/` and (for `website`) for `docs/website/`. Iteration over files, sections, functions and pages is always sorted (section 3.9).
- No file is written while any structural error exists. Errors from all inputs are collected first, then the build stops with exit 1.
- `docs`: writes each file of the set whose bytes differ from the file on disk (unchanged files keep their modification time), then deletes every file under `docs/` that is not in the set, except the subtree `docs/website/`, and removes empty folders. `--check` writes and deletes nothing.
- `website`: deletes `docs/website/docs-stamp.txt` before anything else. Renders the site into memory, writes it into `docs_builder/.cache/staging/` (cleared first), runs the checks of section 15 on the staging tree, and only if they pass synchronises `docs/website/` with it (write changed files, delete files not in the set, remove empty folders) and finally writes `docs-stamp.txt`. A failed build therefore leaves the previous site in place but without its stamp, so the package failsafe (section 16) refuses it.
- All text files are UTF-8 without BOM with `\n` line ends and exactly one final newline.

### 3.7 Diagnostics (`diag.py`)

Format, one per line: `<path>:<line>: error: <message>` and `<path>:<line>: warning: <message>` (`:<line>` omitted when there is no line, for example for a missing file). Summary line at the end: `docs: N errors, M warnings` (or `website:` and so on). Errors are always printed in full. Warnings are grouped by kind; without `-v`, the builder prints the first three of each kind and a line `... and K more '<kind>' warnings (use -v)`. Each warning kind has a short fixed id (for example `param-separator`, `param-item`, `fence-language`, `html-output`) shown in brackets at the end of the line, so the later doc-block touch-up (plan section 7) can grep for them.

### 3.8 Versions read once

`config.py` reads, once per run: the library version from `include/quadriga_lib.hpp` with the regular expression `#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"` (the same one the old builder and `pyproject.toml` use), and the Armadillo version from `CMakeLists.txt` with `set\(armadillo_version\s+"([^"]+)"\)`. A missing file or no match is a structural error naming the file. Both values are passed down; no other module reads them.

### 3.9 Byte-identical output

The same inputs give byte-identical `docs/` and `docs/website/` trees (plan 5.1). Rules: sorted file and directory listings (`sorted()` on POSIX path strings); no dates, times, user names, host names or absolute paths in any output; JSON written with `sort_keys=True`; Jinja2 rendering with a fixed environment (section 8.1); the skill zip as in section 14; tutorial outputs read from the cache, which is itself deterministic given the same notebook and package versions (section 13.5). A test builds the real repository twice into separate roots and compares all bytes (section 17.2). Byte-identity holds for the same tool versions. The tools have minimum versions only (section 4), so an upgrade may change the output without any source change (measured: the Markdown of a doc block rendered with markdown-it-py 3.0.0, mdit-py-plugins 0.4.2 and with 4.2.0, 0.6.1 was identical; the span-free C++ page was 693,763 bytes with Pygments 2.17.2 and 694,128 bytes with 2.21.0). Tool versions are never written into any output file. When `docs --check` finds differences it prints, after the file list, the versions that affect the tracked Markdown: `Python <x.y.z>, markdown-it-py <v>, mdit-py-plugins <v>, mdformat <v>, mdformat-gfm <v>`, and the line `If the sources did not change, the differences come from these tool versions; regenerate with make docs and commit docs/.` (D4 as amended, D23).

## 4. Dependencies

The tools come from PyPI and npm, not from Ubuntu packages (D4 as amended). The requirement files and `package.json` give minimum versions only (`>=`), so the tools can be upgraded. There is no exact pinning and no committed `package-lock.json`; this amends plan 5.1 ("a Node package file with lockfile"). The minimums are the lowest versions the spec's code needs, measured where cheap; "tested with" marks versions that were run but whose lower limit was not searched.

### 4.1 Python

Minimum Python 3.11 (`tomllib`; plan section 11). Measured on the build machine: Python 3.14.4.

`docs_builder/requirements.txt`:

```
markdown-it-py>=3.0.0
mdit-py-plugins>=0.4.2
mdformat>=1.0.0
mdformat-gfm>=1.0.0
Pygments>=2.17.2
Jinja2>=3.1.2
```

| Package | Minimum | Basis | Needed by |
|---------|---------|-------|-----------|
| markdown-it-py | 3.0.0 | measured: the parser set-up of section 5.6 (CommonMark, tables, raw HTML) and the mdformat round trip work; mdit-py-plugins 0.4 needs markdown-it-py 3 | docs, website |
| mdit-py-plugins | 0.4.2 | measured: `dollarmath` (with `allow_space`, `allow_digits`, `double_inline`) and `container` give the tokens of section 5.6 | docs, website |
| mdformat | 1.0.0 | measured; the renderer API of section 8.6 (`MDRenderer`, `PARSER_EXTENSIONS["tables"]`) as used; 1.0.0 is also the newest | docs |
| mdformat-gfm | 1.0.0 | measured together with mdformat 1.0.0 (it brings wcwidth) | docs |
| Pygments | 2.17.2 | measured: the span-free formatter of section 10 runs on the prototype page; tested with 2.21.0 as well | website |
| Jinja2 | 3.1.2 | tested with 3.1.2 and 3.1.6 (installs MarkupSafe) | website |

Measurement: a scratch venv with exactly these minimums (`/home/stephan/quadriga-lib-agents/website_step3/venv_min/`) ran the parser and renderer prototypes; their Markdown output was byte-identical to the one with the newest versions (markdown-it-py 4.2.0, mdit-py-plugins 0.6.1). A later major version (mdformat 2, markdown-it-py 5) may change an API the builder uses; `docs-test` shows it.

`docs_builder/requirements-test.txt`: `pytest>=8.0` (tested with 9.1.1).

`docs_builder/requirements-tutorials.txt`, installed into the same environment as the build packages: `nbformat>=5.10`, `nbclient>=0.10`, `jupyter-client>=8.0`, `ipykernel>=6.29`, `matplotlib>=3.8`, `numpy>=1.26`, `quadriga-lib>=0.12.2` (the released pip package, D28). Not measured at their lower limits; the current releases (nbformat 5.11.1, nbclient 0.11.0, jupyter-client 8.10.0, ipykernel 7.4.0, matplotlib 3.11.2, numpy 2.5.4, quadriga-lib 0.12.2) are the ones step 5 tests with. quadriga-lib 0.12.2 has wheels for CPython 3.10 to 3.14 on manylinux x86_64.

`make docs` needs only the first four packages of `requirements.txt` and Python (D21).

### 4.2 Node

Minimum Node 22 (`engines` of html-validate needs `^22.22.0 || >=24.8.0`, pa11y-ci `^22.13.0 || >=24`; katex itself would run on older Node). Measured: system Node v22.22.1 and npm 9.2.0 on the build machine; conda-forge offers `nodejs` 22.x (newest 22.23.4) up to 26.12.1 (`conda search -c conda-forge --override-channels "nodejs>=22"`, 2026-11-02). Node is needed only by `make website` when a page has a formula, by `make tutorials` when a notebook has a formula, and by `make docs-validate`.

### 4.3 `docs_builder/package.json`

```json
{
  "name": "quadriga-lib-docs-builder",
  "private": true,
  "description": "Build-time tools of the Quadriga-Lib website",
  "license": "Apache-2.0",
  "type": "module",
  "engines": { "node": ">=22" },
  "dependencies": { "katex": ">=0.16.0" },
  "devDependencies": { "html-validate": ">=11.0.0", "pa11y-ci": ">=5.0.0" }
}
```

katex: measured with 0.16.0 and 0.18.11 under Node 22.22.1 (MathML output and per-item parse errors as in section 11; 0.16.0 gives 441 bytes for the test formula, 0.18.11 581 bytes, another example of tool versions changing output). html-validate and pa11y-ci: minimums are the major versions current on 2026-11-02, not measured lower (tested with: not yet run; step 5 runs them).

### 4.4 The two tool set-ups

Both must work (D4 as amended); `docs_builder/README.md` (step 7) documents both.

| | Conda (preferred) | Virtual environment |
|---|-------------------|---------------------|
| Environment | conda env `quadriga-web` from conda-forge: Python, pip, Node.js and npm | `docs_builder/.venv/` with the system `python3`; Node.js from the system (any source, version 22 or newer) |
| Python packages | `pip install -r ...` inside the env | `pip install -r ...` inside the venv |
| npm packages | `build.py setup-node --global [--validators]`: into the env's Node prefix (`npm root -g` lies inside the env), not into the repository | `build.py setup-node [--validators]`: into `docs_builder/node_modules/` (ignored) |
| Create | `make docs-setup-conda` | `make docs-setup-venv` (tutorials: `make docs-setup-tutorials-venv`) |
| Use | `conda activate quadriga-web`, then `make docs` and so on | `source docs_builder/.venv/bin/activate`, then `make docs` and so on |
| Files in the repository | none | `.venv/`, `node_modules/` (both ignored) |

The tutorial kernel `python3` is the one of the active environment (ipykernel installed there, section 13.3).

## 5. Doc-block parser

### 5.1 Files read

For each language the builder reads every file `<folder>/*.cpp` directly in the source folder (`src/`, `api_mex/`, `api_python/`; not recursive, as today), in sorted order, as UTF-8 (a decoding error is a structural error). A doc-block delimiter in any other file of these folders (`.hpp`, `.h`, `.cu`, `.md`) gives a warning `ignored-file` and the file is skipped. Measured: 75 files in `src/`, 85 in `api_mex/`, 80 in `api_python/` contain blocks; all are `.cpp`, none is in a subfolder.

### 5.2 Block grammar

A block starts with a line that is exactly `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` and ends with the next line that is exactly `SECTION!*/`, `SECTION_DESC!*/` or `MD!*/` of the same kind. Trailing whitespace on a delimiter line is a style warning and accepted.

Structural errors (file and line of the opening delimiter unless stated):

| Id | Condition |
|----|-----------|
| E-delim-open | An opening delimiter without its closing delimiter before the next opening delimiter or the end of the file. |
| E-delim-inline | A delimiter text (`/*!MD`, `MD!*/`, and the others) that is not alone on its line (line of the text). |
| E-section-count | A file with `/*!MD` or `/*!SECTION_DESC` blocks and no `/*!SECTION` block, or with more than one `/*!SECTION` block. The old builder silently dropped such blocks. Measured: no file today. |
| E-section-title | A `/*!SECTION` block whose content is not exactly one non-empty line. |
| E-name | The first line of an `/*!MD` block does not match `^# (\.?[A-Za-z_][A-Za-z0-9_.]*)(<\+\+>)?$`. `<++>` is accepted and ignored (plan section 11). |
| E-brief | Line 2 of an `/*!MD` block is empty or starts with `#`, `-`, `|` or three backticks. |
| E-fence | A code fence (a line whose stripped text starts with three backticks) without a closing fence in the same block or block part (line of the opening fence). |
| E-heading | A Markdown heading other than the `## ` sub-block headings of section 5.3 outside code fences, in an `/*!MD` or `/*!SECTION_DESC` block (line of the heading). Measured: none today. |
| E-required | A required sub-block is missing (section 5.3). |
| E-html | Raw HTML other than `<br>` (also `<br/>`, `<br />`) and `<a href="...">...</a>` outside code (line of the tag); or an `<a>` without `href`, or without its `</a>` on the same line. Measured: only `<br>` and `<a>` occur. |
| E-link | An unresolved `[[...]]` link (section 8.7.2). |
| E-anchor | Two functions with the same anchor on one page, or an anchor equal to a section id or legacy id on the same page. Measured: no duplicates today, also none case-insensitively. |
| E-gap | A gap marker opened and not closed on the same line (section 6.4). |

Line 3 of an `/*!MD` block should be empty; otherwise a style warning `blank-line` (measured: always empty today).

### 5.3 Parts of an `/*!MD` block

The block is split at the source-line level, outside code fences, into: name (line 1), brief (line 2), intro (from line 3 to the first sub-block heading), and sub-blocks. A sub-block heading is a line starting with `## `. Its label is the heading text with a trailing `<br>`, then a trailing `:` or `;`, removed and whitespace stripped (measured variants: `## Inputs:`, `## Attributes:<br>`, `## Function 'duplicate';`, `## In/out (launch configuration, ...):`). An empty label is error E-heading.

Each sub-block gets a kind from its label:

| Kind | Labels (case-sensitive, after normalisation) | Rendering |
|------|----------------------------------------------|-----------|
| `code` | `Declaration`, `Declarations`, `Usage`, `Example`, `Examples` | Fragment; its code fences use the page language when they have no info string. |
| `params` | `Input`, `Inputs`, `Output`, `Outputs`, `Attributes`, `Arguments`, `Input Arguments`, `Output Argument`, `Output Arguments`, `Optional outputs`, and any label starting with `Inputs (`, `Outputs (` or `In/out` | Parameter lists (section 5.4). |
| `plain` | everything else (for example `Returns`, `See also`, `Description`, `Simple member functions`, `Data buffer layout (EM mode)`) | Fragment. |

Required sub-blocks (D7, E-required), measured against all blocks so that today's sources pass:

- C++: a `Declaration`, `Declarations` or `Usage` sub-block, except for class blocks, which are blocks with an `Attributes` sub-block (measured: `arrayant`, `channel`, `path`).
- MATLAB and Python: a `Usage` sub-block. A block without it is error E-required, also when it has a `Declaration` sub-block instead (measured before the doc-block fixes: one block, `api_python/qpy_write_png.cpp`; it is fixed on `main` before the spec review, section 21.1, C1; expected count after the fixes: zero).

The display name is the name from line 1 without `<++>`; on the MATLAB page it is lower-cased (as today, design open point 7; the sources write MATLAB names in upper case).

### 5.4 Parameter lists

In a `params` sub-block, each top-level bullet list item is tested against this pattern on its first source line (`NAME` is `` `[^`]+`(?:, (?:\.\.\.|`[^`]+`))* ``):

```
^- (?P<names>\*\*NAME\*\*(?:(?:, | / )\*\*NAME\*\*)*)(?:\s+(?P<qual>\*\([^)]*\)\*|\([^)]*\)))?(?P<sep>\s+[—–-]\s+|<br>\s*$|\s*$)
```

- A matching item becomes a parameter row: `names` (the code-span texts with their separators as written, for example `` `Dx`, `Dy`, `Dz`, `*D` `` or `` `V_re` / `V_im` ``), `qualifier` (the text inside the parentheses, for example `optional`, `in/out`, `optional, in/out`, `single-frequency only`, `input`; empty if absent) and `description` (the rest of the item as a fragment, including continuation lines, nested lists and tables).
- The separator ` — ` is the SPEC form. ` - ` and ` – ` are accepted with the style warning `param-separator`; `<br>` at the end of the line or nothing means the description starts on the next line.
- A non-matching item becomes a note row (rendered across both columns) with the style warning `param-item`.
- Blocks in a `params` sub-block that are not bullet lists (paragraphs, tables) are rendered as ordinary fragments in place.

Measured over all `params` sub-blocks of the three languages (541 sub-blocks, 2641 top-level items): 2616 items match (99.1 %); separators: 2593 em dash, 17 hyphen, 6 `<br>`; 25 items do not match (for example `` - `input`: Input data ... `` in `src/interp.cpp`, `` - `**result**`— ... `` in `api_mex/point_inside_mesh.cpp`); 12 lists contain at least one non-matching item. The pattern covers all variants of plan section 11: `*(optional)*` (567) and `(optional)` (22), `*(in/out)*` (3), `*(optional, in/out)*` (2), several names in one bold span (8) and several bold names (`**`origN`**, **`destN`**`).

### 5.5 Differences between the languages

| | C++ | MATLAB | Python |
|---|-----|--------|--------|
| Folder | `src/` | `api_mex/` | `api_python/` |
| Display names | as written; member functions start with `.` | lower case | as written; may contain `.` (`sos.acf_2d`) |
| Page language for fences without info | `cpp` | `matlab` | `python` |
| Required sub-block | `Declaration`/`Declarations`/`Usage` (not for class blocks) | `Usage` | `Usage` |
| File stem for families | — | `<stem>.cpp` | `qpy_<stem>.cpp` (or `<stem>.cpp` without prefix) |

### 5.6 markdown-it set-up and the builder's rules (`mdparse.py`)

One factory, `make_parser(kind)`, with `kind` in `docblock`, `prose`, `notebook`. All kinds: `MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False}).enable("table")`, plus `dollarmath_plugin(allow_space=False, allow_digits=False, double_inline=True)` (D5; `$5 and $6` stays text), plus the gap rule (section 6.4). The builder's own rules:

| Rule | Kinds | Effect |
|------|-------|--------|
| `wikilink` (inline rule) | docblock | `[[name]]` outside code becomes a link token whose `meta["target"]` is an unresolved API target; a `.` directly before `[[` makes it the member form `.[[name]]` with the target name `.name` (C++ member functions such as `.calc_directivity_dBi`, as in the old builder). There is no syntax for a link to another language; such references are written as raw links (section 8.7.3). |
| `raw_a` (core rule) | docblock | Converts `html_inline` `<a href="U" ...>` + text + `</a>` into `link_open`/`text`/`link_close` with `href=U`; `target` and `rel` attributes are dropped, so every link opens in the same tab (decided, section 21.2, Q6). |
| `raw_br` (core rule) | docblock | Keeps `<br>` as `html_inline` `<br>`; removes `<br>` tokens that are followed only by other `<br>` or soft breaks until the end of a paragraph or heading (these served as paragraph spacing in the old pages). |
| `no_html` (core rule) | prose, notebook | Any `html_inline` or `html_block` is a structural error `E-html` with file and line (D27: hand-written HTML is not used for content). |
| `heading_attrs` (core rule) | prose, notebook | A trailing `{...}` on a heading sets its id and legacy ids (section 6.2) and is removed from the heading text. |
| `note` | prose | `container_plugin` for `note` and `warning` (section 6.5). |
| `site_url` (core rule) | all | Links whose URL starts with `http://quadriga-lib.org/`, `https://quadriga-lib.org/` or the `www.` forms are turned into site-relative links (so that `http://quadriga-lib.org/formats.html` in doc blocks resolves locally and in twins; the doc blocks themselves stay unchanged, plan section 10). |

Code spans and fences are never touched by these rules. Token `.map` line ranges plus the part's first source line give the line numbers for diagnostics.

### 5.7 Content model (`model.py`)

```python
@dataclass
class Fragment:            # block tokens after the rules of section 5.6
    tokens: list           # markdown_it.token.Token
    source: str            # file path for diagnostics
    line: int              # source line of the first token

@dataclass
class Param:
    names: str             # as written, e.g. "`Dx`, `Dy`"
    qualifier: str         # "" if none
    description: Fragment
    is_note: bool          # True for a non-matching item (section 5.4)

@dataclass
class SubBlock:
    label: str             # normalised label, e.g. "Inputs"
    kind: str              # "code" | "params" | "plain"
    content: list          # Fragment items, and for "params" lists of Param

@dataclass
class Function:
    lang: str
    name: str              # display name (MATLAB lower case)
    anchor: str            # == name (section 9.1)
    source: str; line: int
    brief: Fragment        # inline content of line 2
    intro: Fragment | None
    subblocks: list[SubBlock]
    family: str | None     # section 9.4

@dataclass
class Section:
    lang: str
    title: str
    slug: str              # section 9.2
    legacy_id: str         # section 9.3
    desc: list[Fragment]   # SECTION_DESC blocks in file-name order
    functions: list[Function]   # sorted (section 5.8)

@dataclass
class ApiPage:
    lang: str
    intro: "ProsePage"     # content/api_intro_<lang>.md (section 6.7)
    sections: list[Section]     # sorted by title
```

Prose pages, structured pages and tutorials have their own model classes (`ProsePage` with front matter and one `Fragment`; `Landing`, `Publications`; `Tutorial`), defined in the modules that own them. One parser and one model feed two renderers (plan 5.1): `render_html.py` and `render_md.py` take these objects; neither re-parses source text.

### 5.8 Ordering

Sections are sorted by title (Python string order), functions within a section by `(name.startswith("."), name)` on the display name, as on today's pages. Several `SECTION_DESC` blocks of one section are kept in file-name order. The general usage notes come first on each API page.

## 6. Prose pages

Prose pages are formats, FAQ, Get started, the AI page, the legal pages, the tutorial index text, the 404 sentence and the three API intros (D27). Each is one Markdown file in `docs_builder/content/`.

### 6.1 File format and front matter

A prose file starts with a TOML front matter between two lines that are exactly `+++`; the Markdown body follows. TOML, not YAML, because the builder already reads TOML with the standard library.

| Key | Type | Required | Inline Markdown | Use |
|-----|------|----------|-----------------|-----|
| `title` | string | yes | no | `h1`, `<title>` ("`title` · Quadriga-Lib"), `og:title`, twin heading |
| `description` | string | yes | no | meta description and `og:description` |
| `lead` | string | no | yes | lead paragraph under the `h1` and in the twin |
| `dummy_note` | string | only in `tutorials.md` | yes | note shown while all notebooks are dummies (section 13.6) |

Unknown keys, missing required keys and wrong types are structural errors with file and key (section 7.1). Markdown dialect of the body: CommonMark with GFM tables, `$...$` and `$$...$$` math, note boxes, explicit heading ids, build variables and gap markers. Not allowed (structural errors): raw HTML, `#` (h1) headings (the `h1` comes from `title`), headings deeper than `####`.

### 6.2 Heading ids

Every `##` and `###` heading gets an id. Default: the readable slug of its text (section 9.2), made unique within the page by appending `-2`, `-3`, ... in document order. An explicit id is written at the end of the heading line: `### How do I cite Quadriga-Lib? {#cite}`. Legacy ids (old hash anchors and old entry ids, D3) are added with `legacy="..."`, repeatable: `## Material Properties {#material-properties legacy="9f6922f9"}`. Grammar of the braces: `{` then `#id` (id: `[A-Za-z][A-Za-z0-9_-]*`) then zero or more ` legacy="text"` (text without `"`) then `}`. Explicit and legacy ids must be unique within the page (E-anchor). In HTML the id goes on the heading and each legacy id on an empty `<span id="...">` as the first child of the heading. In twins the braces are removed and anchors use GitHub slugs (section 9.5).

### 6.3 Build variables

Written `{{ name }}` (spaces optional) in prose bodies, front matter strings and TOML text fields; not in doc blocks or notebooks. Substituted on the source text before parsing, also inside code. Names:

| Name | Value |
|------|-------|
| `version` | library version (section 3.8) |
| `armadillo_version` | bundled Armadillo version (section 3.8) |
| `site_base_url` | `[site] base_url` |
| `download_base_url` | `[site] download_base_url` (D25) |
| `releases_url` | `[site] releases_url` |
| `github_url` | `[site] github_url` |

An unknown name is a structural error. Build variables are values; gap markers (section 6.4) are content placeholders: the two syntaxes differ on purpose (plan section 11).

### 6.4 Gap markers

One marker for all sources (D27): `%%GAP: text%%` on one line. `text` is either a description of missing content (`%%GAP: N%%`, `%%GAP: performance numbers and hardware%%`) or a draft that still needs the maintainer's approval (`%%GAP: Realistic radio channels for mobile and Wi-Fi network simulation%%`). Removing the marker (keeping or replacing the text) is the approval.

- Where it may appear: prose files and their front matter, TOML data files (`landing.toml`, `publications.toml`, `llms.toml`), doc blocks, notebook Markdown cells. An opening `%%GAP:` without `%%` on the same line is error E-gap.
- HTML: in body text an inline rule turns the marker into `<span class="ph">text</span>` (the `--ph-bg`/`--ph-fg` colours of the prototype). In plain-text contexts (`<title>`, meta description, `alt`, `og:*`) the text is used without the marker.
- Markdown twins: the marker is kept verbatim, so that a reader of `docs/` sees it.
- Missing figures are gaps too (section 12.6).
- The gap list: `gaps.py` scans the raw text of all sources listed above with the pattern `%%GAP:\s*(.*?)%%` (doc blocks only inside blocks), plus the figure and preview-image checks, and prints `Gaps (N):` followed by one line `<path>:<line>: <text>` per gap, sorted by path and line. `docs` and `website` print the count and the first ten; `gaps` prints all. `website --release` and `gaps --release` fail (exit 1) if the list is not empty.

### 6.5 Note boxes

```
::: note Optional title
Body Markdown.
:::

::: warning Placeholder
Body Markdown.
:::
```

HTML: `<div class="note-box">` or `<div class="note-box warn">`, with `<b>title</b>` first (default title "Note" or "Warning"), then the rendered body (design 6). Twin: a GitHub alert, `> [!NOTE]` or `> [!WARNING]`, then `> **title**` and the body lines prefixed with `> `.

### 6.6 One-time conversion of the old prose sources

Step 5 (work package WP4) converts `html_builder/formats/*.md`, `html_builder/faq/*.md` and `html_builder/download/*.md` into `docs_builder/content/formats.md`, `faq.md` and `download.md`. The old files stay unchanged until the cutover; until then an edit to these pages is made in both places (D24). Rules:

- Each old `/*!SECTION` becomes an `##` heading with an explicit readable id and the old hash as legacy id. Old hashes (measured, section 9.3): formats "QuaDRiGa Array Antenna Exchange Format (QDANT)" `6cab4884`, "Material Properties" `9f6922f9`; FAQ "About the Library" `e63d3e38`, "Bug Reports and Contact" `59d38498`, "Distribution License" `89373af1`, "Supporting the Project" `d8094d72`; download "Download" `7c9ba8cc`, "Installation from Binaries" `7fc9f392`, "Installation from Source" `904e3979`, "Quick Start" `fdc96a45`.
- `/*!SECTION_DESC` text becomes the body of the section; each old `/*!MD` block becomes a `###` heading (its name line) followed by its brief and body, with its `## X:` sub-headings as `#### X`. The old entry ids `<arrayant>` and `<layout>` of the formats page are kept as legacy ids. Dropped (decided, section 21.2, Q3): the four entry ids with spaces (`1. Material Table`, `2. Built-in EM Materials`, `3. Principled BSDF`, `4. References`), which are not valid HTML ids and which no page or doc block links to (measured), and the ids `…` and `1`, which came from unescaped example XML (meaningless, D3).
- FAQ and download: each bold-labelled list item (`- **How do I cite Quadriga-Lib?**<br>`) becomes a `###` heading with its text as body. The question "How do I cite Quadriga-Lib?" gets the explicit id `cite` (plan section 10). The download quick-start labels (`**Python**`, `**MATLAB / Octave**`, `**C++**`) become `###` headings and their fences get the info strings `python`, `matlab`, `cpp`; the compile line becomes a `bash` fence that uses `{{ armadillo_version }}` instead of `15.6.2`.
- Raw HTML is converted to Markdown: `<a href="U">T</a>` to `[T](U)` (dropping `target` and `rel`), `<br>` removed, `&lt;`/`&gt;`/`&quot;` and the umlaut entities to characters (escaping `<` in text as `\<`), `[` in text escaped as `\[`.
- `download.md.in`: `__VERSION__` becomes `{{ version }}`; `download/<file>` links become `{{ download_base_url }}/<file>`; `releases/` becomes `{{ releases_url }}`.
- Order of sections: formats as in the prototype (QDANT first, then materials); FAQ and download in the old file order (alphabetical by file name, as the old builder read them).
- Front matter: `title` "Data formats", "Questions", "Get started"; `description` and `lead` taken from the prototype where it has them, wrapped in gap markers (design section 9: meta descriptions are gaps).

### 6.7 API intro files

`docs_builder/content/api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md`, with front matter `title` ("C++ API", "MATLAB / Octave API", "Python API"; design open point 7), `description` (gap-marked draft from the prototype's meta description for C++; a gap for the others), `lead` (the prototype's C++ lead adapted per language, with `{{ version }}`, gap-marked). The body is the "General usage notes" list. Text (decided, section 21.2, Q5): the C++ and MATLAB lists from `get_preamble()` in `html_builder/extract_md.py`, and in the C++ units line the example `*_3dB` from the HTML copy added after `*_deg` (as in `md/SPEC_CPP_API.md`), with `[arrayant_generate](#arrayant_generate)` written as `[[arrayant_generate]]`; for API intros the `wikilink` rule is enabled as in doc blocks. The Python body is one line: `%%GAP: General usage notes for the Python API (import, numpy conventions, index base, errors)%%` (content gap for go-live, plan section 11). The notes become the first section of each API page (id `sec-general-usage-notes`) and the file `docs/api/<lang>/general-usage-notes.md`.

### 6.8 Other content files written in step 5

| File | Initial content |
|------|-----------------|
| `content/ai_agents.md` | Draft text of the prototype `ai_agents.html`, with the file table adapted to section 2.3 paths (`api/<lang>/<section>.md`, `quadriga_lib_api_<lang>.md`, `llms.txt`, twins), the skill download link `skill/quadriga-lib-skill-{{ version }}.zip`, and a `::: warning Placeholder` box saying the skill is a placeholder (plan 5.4). First line of the body: `%%GAP: whole page text is a draft (design section 9)%%`. |
| `content/impressum.md` | `title` "Impressum"; body: `::: warning Placeholder` with `%%GAP: Impressum text from the maintainer (D17); no address in git until settled%%`. |
| `content/privacy.md` | `title` "Privacy"; body: `::: warning Placeholder` with `%%GAP: privacy text from the maintainer: the hoster and its server logs; no cookies, no analytics, no third-party requests; the browser stores the chosen code language under the localStorage key qlib-lang, only after a click (D18)%%`. |
| `content/tutorials.md` | `title` "Tutorials"; `lead` and `dummy_note` from the prototype `tutorials.html`, gap-marked. |
| `content/404.md` | `title` "Page not found"; `description` "Page not found"; body: the prototype sentence, gap-marked (design section 9). |
| `content/llms.toml` | `summary` (one paragraph for `llms.txt`, gap-marked draft from the landing lead). Schema: `summary` string, required. |

## 7. Structured pages

### 7.1 Schema checker (`schema.py`)

A schema is a nested Python dict: for each key its type (`str`, `int`, `bool`, `table`, `list[str]`, `list[table]`), whether it is required, whether its strings allow inline Markdown, and for lists the allowed count (`exactly n`, `at least n`). The checker reports every problem, not only the first, as `<file>: error: <key path>: <problem>` (TOML gives no line numbers; the key path, for example `why.card[2].title`, locates it): unknown key, missing required key, wrong type, wrong count. Inline-Markdown strings are rendered with `renderInline` of the `prose` parser (links, code, emphasis, math, gap markers; no block syntax); a newline in an inline string is a structural error. Plain strings are HTML-escaped.

### 7.2 `content/landing.toml`

Section ids in the template are fixed: `why`, `paths`, `quickstart` (D3 keeps `index.html#quickstart`; the prototype's `qs` is replaced), `apis`, `covers`, `open`.

| Key | Type | Req. | Inline MD | Notes |
|-----|------|------|-----------|-------|
| `meta.title` | str | yes | no | `<title>` and `og:title` ("Quadriga-Lib") |
| `meta.description` | str | yes | no | meta description |
| `pitch.headline` | str | yes | yes | `h1` |
| `pitch.lead` | str | yes | yes | |
| `pitch.primary` | table `{label, href}` | yes | no | filled button |
| `pitch.secondary` | table `{label, href}` | yes | no | outlined button |
| `pitch.proof` | list[str], exactly 3 | yes | yes | proof lines with dots |
| `pitch.figure` | str | yes | no | id of a `figure` entry |
| `why.heading` | str | yes | no | |
| `why.intro` | str | yes | yes | |
| `why.card` | list[table], exactly 3 | yes | | each `title` (str, no MD), `text` (str, MD) |
| `paths.heading` | str | yes | no | visually hidden `h2` |
| `paths.card` | list[table], exactly 2 | yes | | each `label` (str; template appends " →"), `href` (str), `text` (str, MD) |
| `quickstart.heading` | str | yes | no | |
| `quickstart.intro` | str | yes | yes | |
| `quickstart.install` | str | yes | no | shown as a `bash` code block |
| `quickstart.python`, `quickstart.matlab`, `quickstart.cpp` | str (multi-line) | yes | no | code of the three language tabs |
| `quickstart.note` | str | yes | yes | note line below the tabs |
| `quickstart.figure` | str | yes | no | id of a `figure` entry |
| `apis.heading`, `apis.intro` | str | yes | no / yes | |
| `apis.card` | list[table], exactly 3 | yes | | `label`, `href`, `text` (MD) |
| `covers.heading`, `covers.intro` | str | yes | no / yes | |
| `covers.feature` | list[table], at least 1 | yes | | `title` (str), `text` (str, MD), `badge` (str, optional; only `"experimental"`, D12) |
| `open.heading`, `open.intro` | str | yes | no / yes | |
| `open.card` | list[table], exactly 3 | yes | | `label`, `href`, `text` (MD) |
| `figure` | list[table], at least 1 | yes | | `id` (str), `base` (str, for example `img/landing/coverage`; files `<base>-light.webp`, `<base>-dark.webp`), `alt` (str), `caption` (str, MD), `placeholder` (str: `coverage` or `antenna`, which placeholder drawing to show while files are missing), `lazy` (bool, optional, default false) |

`quickstart.python`, `.matlab`, `.cpp` are multi-line TOML strings (`"""..."""`); the newline right after the opening quotes is dropped by TOML, and the builder strips one final newline. Step 5 fills the file from the prototype `index.html`: every text the design lists as a gap (design section 9) is wrapped in a gap marker, every `<span class="ph">[...]</span>` becomes `%%GAP: ...%%`, and the figure entries are `coverage` (pitch, `img/landing/coverage`, not lazy) and `antenna` (quick start, `img/landing/antenna`, lazy), with the prototype's alt texts and captions. The second path card ("For team leads") keeps the prototype's `href = "#covers"`: it points into the landing page, not to a separate page (D30, plan section 10).

### 7.3 `content/publications.toml`

| Key | Type | Req. | Inline MD | Notes |
|-----|------|------|-----------|-------|
| `page.title` | str | yes | no | "Publications" |
| `page.description` | str | yes | no | |
| `page.lead` | str | yes | yes | links to `faq.html#cite` |
| `group` | list[table], at least 1 | yes | | in display order |
| `group[].id` | str | yes | no | heading id |
| `group[].title` | str | yes | no | `h2` |
| `group[].intro` | str | no | yes | paragraph before the list |
| `group[].outro` | str | no | yes | paragraph after the list |
| `group[].entry` | list[table], at least 1 | yes | | in display order (newest first, as written) |
| `entry.year` | int | yes | | |
| `entry.title` | str | yes | no | |
| `entry.authors` | str | yes | no | |
| `entry.venue` | str | yes | no | |
| `entry.link` | list[table], at least 1 | yes | | `label` (str), `url` (str: absolute URL or site-relative path such as `publications/x.pdf`) |

Step 5 fills it from `html_builder/parts/publications.html.part` in the grouping and order of the prototype `publications.html` (two groups, ids `quadriga-lib` and `quadriga-background`). The publication texts are existing content, not gaps; `page.description` is a gap-marked draft.

## 8. Renderers and templates

### 8.1 Jinja2 environment

`jinja2.Environment(loader=FileSystemLoader(templates), autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, undefined=StrictUndefined)`. Fragments rendered by `render_html.py` are passed as `markupsafe.Markup`. Templates emit compact markup for repeated structures (no indentation inside index lists, parameter lists and pills), as in the prototype. Every internal URL in a template is written `{{ root }}<path>`: `root` is `""` for pages in the site root, `"../"` for `tutorials/<stem>.html`, and `"/"` for `404.html` (root-absolute links, plan section 11, because the server shows it at any path).

### 8.2 Templates

| File | Used by | Structure (prototype file in brackets) |
|------|---------|----------------------------------------|
| `base.html` | all pages | head (section 8.3), skip link, header, `<main id="content">` block, footer (`index.html`) |
| `_header.html` | base | bar with menu `<details class="menu">`, inline logo SVG, name, version, GitHub link; tab row (design 4.1) |
| `_footer.html` | base | four columns from `[[nav.footer]]`, bottom line "Quadriga-Lib v`version`", Impressum, Privacy (design 4.2) |
| `_logo.svg` | header, 404 | the logo of `static/img/logo.svg` inline, `aria-hidden="true" focusable="false"` |
| `landing.html` | `index` | sections of design 5.1 (`index.html`) |
| `api.html` | API pages | page head, side index, mobile index, general notes section, sections with entries (`cpp_api.html`) |
| `_api_index.html` | api | the index list (used twice: sidebar and mobile `<details>`) |
| `_fn_entry.html` | api | one function entry (section 8.4) |
| `text.html` | text pages | breadcrumb, `h1`, lead, page tools ("Markdown version"), mobile index, prose, side index "On this page" (`formats.html`, `ai_agents.html`) |
| `legal.html` | Impressum, privacy | breadcrumb, `h1`, prose; no index (`impressum.html`) |
| `tutorials_index.html` | tutorials | breadcrumb, `h1`, lead, dummy note, card grid (`tutorials.html`) |
| `tutorial.html` | each tutorial | breadcrumb, `h1`, tags, downloads, fragment, back link (`tutorial_sample.html`) |
| `publications.html` | publications | breadcrumb, `h1`, lead, groups with entry lists (`publications.html`) |
| `notfound.html` | 404 | centred logo 96 px, `h1`, sentence, buttons Home, C++ API, MATLAB API, Python API (design 5.8; the design names "the API pages", so all three) |

Breadcrumbs: "Home › `title`" on text, legal, publications and tutorial index pages; "Home › Tutorials › `title`" on tutorial pages; "Home › API reference › C++" (and "MATLAB / Octave", "Python") on API pages, with "API reference" linking to `index.html#apis`. The current page's tab and menu entry get `aria-current="page"`; tutorial pages mark "Tutorials".

### 8.3 Head of every page (design 4.3)

In this order: `<!doctype html>`, `<html lang="en" class="no-js">`, `<meta charset="utf-8">`, viewport, `<title>`, meta description, two `theme-color` metas (light and dark, from `[site]`), `og:type` `website`, `og:title`, `og:description`, `og:url` (absolute: `base_url` + `/` + page path), `og:image` (absolute: `base_url` + `/` + the preview image path, section 12.5), `twitter:card` `summary_large_image`, `<link rel="icon" href="img/favicon.svg" type="image/svg+xml">`, `<link rel="icon" href="img/favicon-32.png" sizes="32x32" type="image/png">`, `<link rel="apple-touch-icon" href="img/apple-touch-icon.png">`, `<link rel="alternate" type="text/markdown" href="...">` only on pages with a twin (D26), `<link rel="preload" href="fonts/ibm-plex-sans-latin-400-normal.woff2" as="font" type="font/woff2" crossorigin>`, `<link rel="stylesheet" href="css/site.css">`, `<script>document.documentElement.className = "js";</script>`, `<script src="js/site.js" defer></script>`. Titles: "`title` · Quadriga-Lib"; landing: `meta.title`.

### 8.4 API page

Follows the prototype `cpp_api.html` and design 5.2:

- Page head: breadcrumb; `h1` = intro `title`; lead = intro `lead`; page tools: "Same API in" links to the other two pages, and "Markdown version" to `quadriga_lib_api_<lang>.md`.
- Index (`nav.doc-side.idx`, `aria-label="<display> API index"`; mobile copy in `details.doc-index` with summary "Sections and functions" and `aria-label="<display> API index (collapsible)"`): filter `<input class="idx-filter js-only" type="search" placeholder="Filter functions   /" aria-label="Filter functions">`; link "General usage notes" to `#sec-general-usage-notes`; per section a closed `<details>` with summary title and `<span class="count">` (number of functions), then "Section overview" (`#sec-<slug>`) and one `<li class="fn">` per function; below, `.idx-other` with links to the other two API pages and the full Markdown file.
- General usage notes: `<section class="api-section" id="sec-general-usage-notes"><h2>General usage notes</h2>` + intro body.
- Section: `<section class="api-section" id="sec-<slug>"><h2 id="<legacy hash>">title</h2><div class="section-desc">...</div>` + entries.
- Entry `_fn_entry.html`: `<article class="fn-entry" id="<anchor>">`; `<div class="fn-head"><h3>name</h3><a class="anchor" href="#<anchor>" aria-label="Link to name">#</a></div>`; `<div class="fn-langs"><span class="lbl">Also in</span>` + three pills in the order C++, MATLAB, Python: the current language `<a class="pill" aria-current="page" href="#<anchor>">C++</a>`, a counterpart `<a class="pill" href="mex_api.html#<anchor>">MATLAB</a>`, a missing counterpart `<span class="pill none" title="No MATLAB counterpart">MATLAB –</span>`; then `<a class="md" href="api/<lang>/<slug>.md">Markdown</a></div>`; `<p class="fn-brief">`; intro fragment; per sub-block `<h4>label</h4>` and its content.
- Parameter list: `<dl class="params">` with one `<div>` per row: `<dt>names<span class="opt">qualifier</span></dt><dd>description</dd>` (names: code-span texts with their separators, no `<code>` wrapper; `.opt` only when a qualifier exists); a note row is `<div class="pnote"><dd>...</dd></div>` (CSS: spans both columns).

### 8.4a Other templates

- Text page (`text.html`, prototype `formats.html` and `ai_agents.html`): the index lists the page's `h2` headings; an `h2` with `h3` headings below it becomes `<li><details open><summary><span>h2 text</span></summary><ul><li><a href="#h2-id">Overview</a></li><li class="fn"><a href="#h3-id">h3 text</a></li>...</ul></details></li>`, an `h2` without `h3` a plain `<li><a>`. The list appears twice, as on API pages: `nav.doc-side.idx` with `aria-label="Page sections"` and inside `details.doc-index` (summary "On this page", `aria-label="Page sections (collapsible)"`). A page with fewer than two `h2` headings gets no index and uses the single-column layout of `legal.html`. Page tools: "Markdown version" link to the twin.
- Landing, publications, tutorial index, tutorial page and 404: the markup of the prototype files named in section 8.2, with all texts from the data files, `{{ root }}` URLs, the fixed section ids of section 7.2, and the figure markup of section 12.6. The landing language tabs keep the prototype structure (`div.ltabs[data-lang-tabs]`, radio inputs `name="qs"`, ids `qs-python`, `qs-matlab`, `qs-cpp`, values `python`, `matlab`, `cpp`, Python checked); their code is coloured by `highlight.py`. Inline `style` attributes of the prototype are replaced by classes in `site.css` (section 12.2).

### 8.5 Fragment rendering to HTML

markdown-it's renderer with these overrides: `fence` and `code_block` through `highlight.py` (section 10) giving `<pre><code class="language-<lang>">...</code></pre>`; `table_open`/`table_close` wrapped in `<div class="table-wrap">`; `link_open` with the URL from `links.py` for this output location; `math_inline` and `math_block` replaced after the KaTeX run (section 11), block math wrapped in `<div class="math-block">`; note containers as section 6.5; gap tokens as `<span class="ph">`; headings in prose with ids (section 6.2); prose `##` renders as `h2`, so the page keeps its single `h1` from the template.

### 8.6 Markdown renderer (`render_md.py`, `mdformat_ext.py`)

Fragments are rendered with `mdformat.renderer.MDRenderer().render(tokens, options, env)` with `options = {"parser_extension": [mdformat.plugins.PARSER_EXTENSIONS["tables"], mdformat_ext], "mdformat": {"wrap": "keep", "number": False, "end_of_line": "lf"}}`, after the link tokens got the URLs for the output location. `mdformat_ext` provides renderers for `math_inline` (`$...$`), `math_inline_double` and `math_block` (`$$` blocks), `container_note_*` and `container_warning_*` (GitHub alerts, section 6.5) and `gap` (verbatim marker). Tested in a scratch venv: mdformat renders the doc-block tokens back to clean Markdown, including tables inside list items (`api_mex/get_channels_ieee_indoor.cpp`). The structure around fragments (headings, index tables, parameter items, separators) is written by `render_md.py` directly. Parameter rows are written in the SPEC form `` - **`name`** *(qualifier)* — description `` (the names part as in the source); note rows as their original item. Sub-block labels become `### Label` without the colon. Code fences get the language as info string.

#### 8.6.1 Twin header

Every twin starts with `# title`, an empty line, and the line `Quadriga-Lib <version> · Web page: <absolute URL of the HTML page>` (for API section files: the page URL with `#sec-<slug>`). Then, if the page has at least two `##` headings, an index table (D26) of its `##` headings (columns `Section` and, for prose pages with `###` headings, `Topics`, a comma-separated list of links to its `###` headings). Then the lead, then the body. Exception: the full API files (section 8.6.2).

#### 8.6.2 Full API files `docs/quadriga_lib_api_<lang>.md`

The old layout, minus the build date and the "Line" column (plan section 11):

```
---
title: "C++ API Documentation for Quadriga-Lib v0.12.4"
author: "Stephan Jaeckel"
lang: en-US
---

# General usage notes
<intro body>

# Function Index

| Function | Section | Summary |
| --- | --- | --- |
| [arrayant](#arrayant) | Array antenna class | Class for storing and manipulating array antenna models |
...

---

# Array antenna class

<section description>

---
## arrayant
<brief>

Also in: [MATLAB](quadriga_lib_api_mex.md#...) · [Python](...)

<intro>

### Declaration
...
```

Titles: "C++ API Documentation for Quadriga-Lib v`version`", "MATLAB / Octave API Documentation for Quadriga-Lib v`version`", "Python API Documentation for Quadriga-Lib v`version`" (as today). The "Summary" column holds the brief; it replaces "Line" so that an agent can pick functions from the index. The "Also in" line lists existing counterparts only and is omitted when there are none.

#### 8.6.3 Per-section files `docs/api/<lang>/<slug>.md`

Twin header with title = section title; index table `| Function | Summary |`; section description; then the functions as in the full file. `general-usage-notes.md` holds the intro. The largest section file is expected near 128 kB (measured: the C++ "Site-specific simulation tools" part of today's full file is 128,481 bytes); this is accepted: a section is the unit an agent asks for, and splitting it would need a second, arbitrary unit (plan section 11).

### 8.7 Links

#### 8.7.1 Link targets

`links.py` turns every link into a target object, then each renderer asks for its URL at a given output location. Kinds:

| Kind | Example in a source | HTML page in root | HTML page in `tutorials/` | Twin in `docs/` (and its copy) |
|------|---------------------|-------------------|---------------------------|--------------------------------|
| API function | `[[x]]`, `cpp_api.html#x` | `cpp_api.html#x` (same page: `#x`) | `../cpp_api.html#x` | full file: `#<gh-slug>` or `quadriga_lib_api_<lang>.md#<gh-slug>`; section file: `#<gh-slug>`, `<slug>.md#<gh-slug>` or `../<lang>/<slug>.md#<gh-slug>` |
| Page with twin | `faq.html#cite` | `faq.html#cite` | `../faq.html#cite` | relative path of the twin plus the GitHub slug of the heading that carries the id (`faq.md#how-do-i-cite-quadriga-lib`) |
| Page without twin | `index.html#quickstart`, `impressum.html` | relative | relative | absolute: `<base_url>/index.html#quickstart` |
| Twin file | `quadriga_lib_api_cpp.md` | relative | relative | relative |
| Site file outside `docs/` | `publications/x.pdf`, `quadriga_lib_material_model.md`, `skill/...zip`, `tutorials/...` | relative | relative | absolute: `<base_url>/<path>` (plan section 11) |
| External | `https://...` | unchanged | unchanged | unchanged |
| Same-page anchor | `#files` | unchanged | unchanged | GitHub slug of the target heading |

A link to a function on another API page, written as a raw link in a doc block (`<a href="cpp_api.html#ray_state_update">ray_state_update</a> (C++ API)`, section 8.7.3) or as a Markdown link in prose, is the kind "API function": the builder recognises `cpp_api.html`, `mex_api.html` and `python_api.html` with an anchor that is a function anchor of that page, renders it relative in HTML, and in twins points it to the other language's section file and GitHub slug (from a section file `../cpp/<slug>.md#<gh-slug>`, from a full file `quadriga_lib_api_cpp.md#<gh-slug>`). An anchor that is not a function, section or legacy id of that page is error E-link at parse time. Links in sources are written relative to the site root (as all current links are). A site-relative link whose path is neither a page, a twin nor a file of the site, or whose anchor does not exist, is caught by the link check (section 15.1); in prose and data files it is also a structural error at parse time when the target is a page or twin of the site.

#### 8.7.2 Resolving `[[name]]`

`[[name]]` in a block of language L (and in the API intro of L) links to the function of L whose display name equals `name` (MATLAB: compared in lower case); `.[[name]]` links to the C++ member function `.name`. Nothing else is tried: no other language, no file-stem match. An unresolved link is error E-link (D7, decided strict, section 21.1, C1). A reference to a function of another language is written as a raw link (section 8.7.3).

Measured before the doc-block fixes: C++ 637 links, all resolve; MATLAB 391 links with 5 unresolved, Python 358 with 6 unresolved (the eleven `[[...]]` defects of the fix list for `main`). Expected after the fixes and the merge of `main` into `website` (D20 as amended): zero. The measurements of this section and of Appendix A are re-run on the merged tree before the spec review.

#### 8.7.3 Raw `<a href>` in doc blocks

Converted by `raw_a` (section 5.6), then classified like any other link (section 8.7.1). Cross-language references are raw links to another API page with the API named in the text, for example `<a href="cpp_api.html#ray_state_update">ray_state_update</a> (C++ API)`, as the doc-block fixes on `main` write them; the link check (section 15.1) verifies their targets in HTML and in twins. The other current targets are `formats.html#6cab4884` (resolves to the legacy id on the formats page), `quadriga_lib_material_model.md` and `publications/...` (site files outside `docs/`: relative in HTML, absolute in twins), `http://quadriga-lib.org/formats.html` (made site-relative by `site_url`), and IEEE and other external URLs.

### 8.8 `llms.txt`

Tracked in `docs/` and copied to the site root (D9, plan 5.4). Format of llmstxt.org; all links absolute (`base_url`), because the file is fetched on its own:

```
# Quadriga-Lib

> <content/llms.toml summary>

Version <version>. Every page of quadriga-lib.org has a Markdown version; the API reference exists per language as one full file and as one file per section.

## Documentation

- [Get started](<base>/download.md): installation and quick start
- [Data formats](<base>/formats.md): QDANT antenna files and the material table
- [Questions](<base>/faq.md): FAQ, how to cite
- [Publications](<base>/publications.md): papers and IEEE 802.11 contributions
- [AI agents](<base>/ai_agents.md): how to use these files and the agent skill
- [Tutorials](<base>/tutorials.md): worked examples as Jupyter notebooks

## API reference

- [C++ API, complete](<base>/quadriga_lib_api_cpp.md)
- [MATLAB / Octave API, complete](<base>/quadriga_lib_api_mex.md)
- [Python API, complete](<base>/quadriga_lib_api_python.md)

## API sections

- [C++: General usage notes](<base>/api/cpp/general-usage-notes.md)
- [C++: <section title>](<base>/api/cpp/<slug>.md): <n> functions
- ... (all sections, C++ then MATLAB / Octave then Python)

## Optional

- [Agent skill (placeholder)](<base>/skill/quadriga-lib-skill-<version>.zip)
```

The list texts are fixed in `render_md.py`; they are navigation labels, and the maintainer can change them there.

### 8.9 `docs/README.md`

Not copied to the site. Content:

```
# Quadriga-Lib documentation in Markdown

Generated by `make docs` from the documentation comments in `src/`, `api_mex/` and `api_python/` and from `docs_builder/content/`; do not edit these files. Version <version>. quadriga-lib.org shows the documentation of the latest release; these files show this branch as of the last `make docs`.

| File | Content |
| --- | --- |
| [llms.txt](llms.txt) | Entry point for AI agents |
| [quadriga_lib_api_cpp.md](quadriga_lib_api_cpp.md) | C++ API, complete |
| [quadriga_lib_api_mex.md](quadriga_lib_api_mex.md) | MATLAB / Octave API, complete |
| [quadriga_lib_api_python.md](quadriga_lib_api_python.md) | Python API, complete |
| [download.md](download.md) | Get started |
| [formats.md](formats.md) | Data formats |
| [faq.md](faq.md) | Questions |
| [publications.md](publications.md) | Publications |
| [ai_agents.md](ai_agents.md) | AI agents |

## API sections

| Section | C++ | MATLAB / Octave | Python |
| --- | --- | --- | --- |
| General usage notes | [notes](api/cpp/general-usage-notes.md) | [notes](api/mex/general-usage-notes.md) | [notes](api/python/general-usage-notes.md) |
| <title> | [<n> functions](api/cpp/<slug>.md) | – | ... |
```

Rows: the union of section titles of the three languages, sorted; `–` where a language has no such section.

## 9. Anchors and URLs

### 9.1 Function anchors (D3)

The `id` of a function entry is its display name exactly, as on today's pages: C++ `arrayant_interpolate_multi`, `.calc_directivity_dBi`; MATLAB lower case `arrayant_generate`; Python `sos.acf_2d`. In `href` attributes the anchor is percent-encoded only where needed (none of today's names needs it).

### 9.2 Readable ids and slugs

`slug(text)`: lower-case; every run of characters outside `[a-z0-9]` becomes one `-`; leading and trailing `-` removed. Section ids on API pages: `sec-` + `slug(title)` (the prefix keeps them apart from function names); section file names: `slug(title)` + `.md` (for example "Miscellaneous / Tools" gives `sec-miscellaneous-tools` and `miscellaneous-tools.md`). Prose heading ids: `slug(text)` without prefix (section 6.2). A slug collision between two section titles of one language is error E-anchor.

### 9.3 Old hash ids

The old builder made section ids with `hashlib.shake_256(title.encode("utf-8")).hexdigest(4)` (8 hex characters; read from `html_builder/extract_html.py` and confirmed against the ids in `html_docu/*.html`). The new builder computes the same value for every API section title and puts it on the section's `h2` (section 8.4). Prose pages carry their old hash ids as explicit legacy ids (section 6.6). Measured values:

| Section title | Id |
|---------------|----|
| Array antenna class | `3a0c780f` |
| Array antenna functions | `e091e809` |
| Channel class | `4f5f509b` |
| Channel functions | `eab50272` |
| Channel generation functions | `6f9553d7` |
| Channel statistics | `c2d4d113` |
| Geometry-based stochastic model | `85ec7f8f` |
| Math functions | `40522dfe` |
| Miscellaneous / Tools | `b11e2d16` |
| Site-specific simulation tools | `386370db` |
| Sum-of-sinusoids random number generator | `2d15ed9b` |

Renaming a section changes its legacy id, as it did before.

### 9.4 Language pills: matching rule (design open point 1)

Families:

1. MATLAB and Python functions pair by file stem: `api_mex/<stem>.cpp` and `api_python/qpy_<stem>.cpp` (plan section 11). Measured: 85 MATLAB files, 80 Python files, 78 pairs; MATLAB only: `calc_rotation_matrix`, `fast_sincos`, `generate_diffraction_paths`, `get_channels_irs`, `interp`, `ray_mesh_interact`, `subdivide_triangles`; Python only: `components`, `mitsuba_xml_file_write`. Every MATLAB or Python file is a family (87 in total).
2. A family's C++ counterpart by name: the C++ function whose name equals the MATLAB name or the Python name, compared in lower case with `.` replaced by `_` (so `sos.acf_2d` matches `sos_acf_2d`). Measured: 68 of 87 families.
3. The mapping file `docs_builder/content/api_matching.toml` gives the C++ counterparts that names cannot find. It overrides rule 2 for the families it lists. Rule for its entries: the C++ counterparts of a family are the public C++ functions its MATLAB and Python wrappers call to carry out their operation, not the helper calls for validation, sizing or setup (such as `arrayant_is_valid_multi`, `hdf5_read_layout`, `.set_size`, and the `.copy_element` and `.rotate_pattern` calls with which `arrayant_generate` sets up its patterns).

```toml
# Family (MATLAB/Python file stem) = C++ functions. The first one is the target
# of the C++ pill on the MATLAB and Python pages; every listed C++ function shows
# pills to this family's MATLAB and Python functions.
[families]
arrayant_calc_beamwidth = [".calc_beamwidth_deg"]
arrayant_calc_directivity = [".calc_directivity_dBi"]
arrayant_combine_pattern = [".combine_pattern", "arrayant_combine_pattern_multi"]
arrayant_concat = ["arrayant_concat_multi", ".append"]
arrayant_copy_element = [".copy_element", "arrayant_copy_element_multi"]
arrayant_export_obj_file = [".export_obj_file"]
arrayant_generate = ["generate_arrayant_3GPP", "generate_arrayant_custom", "generate_arrayant_dipole", "generate_arrayant_half_wave_dipole", "generate_arrayant_multibeam", "generate_arrayant_omni", "generate_arrayant_ula", "generate_arrayant_xpol"]
arrayant_interpolate = [".interpolate", "arrayant_interpolate_multi"]
arrayant_qdant_read = ["qdant_read", "qdant_read_multi"]
arrayant_qdant_write = [".qdant_write", "qdant_write_multi"]
arrayant_rotate_pattern = [".rotate_pattern", "arrayant_rotate_pattern_multi"]
calc_angular_spread = ["calc_angular_spreads_sphere"]
channel_export_obj_file = [".write_paths_to_obj_file"]
hdf5_create_file = ["hdf5_create"]
hdf5_write_channel = ["hdf5_write"]
interp = ["interp_2D"]
cart2geo = ["fast_cart2geo"]
geo2cart = ["fast_geo2cart"]
```

Checked against the wrappers (calls outside the doc block, also unqualified calls through the `CALL_QD` macro): every listed C++ function is called by the family's MATLAB wrapper; the only entry not called by the Python wrapper is `.copy_element` in `arrayant_copy_element` (`api_python/qpy_arrayant_copy_element.cpp` calls only `arrayant_copy_element_multi`); the MATLAB wrapper calls both. For all 68 families matched by name, both wrappers call the matched C++ function. Measured with this file: 85 of 87 families have a C++ counterpart (left without: `components`, `version`, which have no C++ block), and 99 of 137 C++ functions show at least one pill to another language. Rules for the file (decided, section 21.2, Q4): it lists the C++ functions the family's wrappers call to carry out their operation, and the order in the file is the rule for the pill target: the first listed function is the target of the C++ pill on the MATLAB and Python pages (a member function before its `_multi` counterpart, as in the table). Structural errors in the file: an unknown family stem, an unknown C++ name, a C++ name listed in two families. Pills: a function's pill for language X links to the family's function in X (for C++ from MATLAB or Python: the first listed C++ function); no counterpart gives the dashed "X –" pill. In twins the same counterparts appear in the "Also in" line (section 8.6.2).

### 9.5 Anchors in twins

GitHub computes heading anchors itself; twins therefore never use HTML ids. `anchors.gh_slug()` reproduces GitHub's algorithm (github-slugger): lower-case the heading's plain text (code spans contribute their content, emphasis marks and link URLs are dropped); keep characters whose Unicode category starts with `L`, `M` or `N`, the category `Pc` (`_`), `-` and space; replace each space by `-`; a slug seen before in the same file gets `-1`, `-2`, ... in document order. Test vectors: `.calc_directivity_dBi` gives `calc_directivity_dbi`; `sos.acf_2d` gives `sosacf_2d`; "How do I cite Quadriga-Lib?" gives `how-do-i-cite-quadriga-lib`; "Miscellaneous / Tools" gives `miscellaneous--tools`; the second "Inputs" heading in a file gives `inputs-1`. The renderer computes slugs over all headings of each output file in order, then fills in link anchors.

### 9.6 URL inventory (D3)

Kept: `index.html` with `#quickstart`; `cpp_api.html`, `mex_api.html`, `python_api.html` with every function anchor (measured: 137, 85, 80) and every section hash (10, 9, 9); `formats.html` with `#6cab4884`, `#9f6922f9` and the entry ids `<arrayant>` and `<layout>` (section 6.6); `faq.html` with its four hashes; `download.html` with its four hashes; `publications.html`; `/quadriga_lib_api_{cpp,mex,python}.md`; `/quadriga_lib_material_model.md`; `/publications/...`. Dropped (decided, section 21.2, Q3 and Q9): all `quadriga-lib_*` layout ids of the old pages; the formats ids `…`, `1`, `1. Material Table`, `2. Built-in EM Materials`, `3. Principled BSDF`, `4. References`; the files `style2.css`, `quadriga-lib-logo80.png`, `quadriga-lib-q_letter64.png`. New: `ai_agents.html`, `tutorials.html`, `tutorials/...`, `impressum.html`, `privacy.html`, `404.html`, the twins, `api/...`, `llms.txt`, `skill/...`. A test compares the new pages with the id inventory of today's `html_docu/` (section 17.2).

## 10. Code colouring

`highlight.py` implements its own small formatter over `pygments.lex()`; it does not use `HtmlFormatter`. Each token type is mapped to the first matching entry of this table (a type matches if it is the listed type or a subtype); tokens without an entry produce plain escaped text and no `<span>` (design open point 4); adjacent tokens with the same class are merged into one `<span>`.

| Token type | Class | Colour (site.css) |
|------------|-------|-------------------|
| `Comment` (including `Comment.Preproc`, `Comment.PreprocFile`) | `c` | `--c-cm`, italic |
| `Keyword.Type` | `kt` | `--c-ty` |
| `Keyword` | `k` | `--c-kw` |
| `Name.Class`, `Name.Namespace` | `kt` | `--c-ty` |
| `Name.Function`, `Name.Builtin` | `nf` | `--c-fn` |
| `String` | `s` | `--c-st` |
| `Number` | `m` | `--c-nu` |
| `Generic.Prompt` | `gp` | `--muted`, `user-select: none` |

Lexers by info string: `cpp`, `c++` → `CppLexer`; `matlab`, `octave` → `MatlabLexer`; `python`, `py` → `PythonLexer`; `bash`, `sh`, `shell` → `BashLexer`; `console` → `BashSessionLexer`; `text`, `none` → no colouring. An unknown info string gives the style warning `fence-language` and no colouring. Lexer options `stripnl=False, ensurenl=False`; the fence's final newline is removed before lexing. Escaping: `&`, `<`, `>`.

Size (measured, plan section 11): the prototype C++ page is 1,042,598 bytes raw. Re-colouring its 152 code blocks with this formatter in a scratch folder (`/home/stephan/quadriga-lib-agents/website_step3/spanfree.py`, Pygments 2.21.0) gives 694,128 bytes; the same page with no colouring at all is 647,202 bytes, so colouring costs about 47 kB. Of the 694 kB, code blocks are 111 kB, the pill rows 45 kB, tables 31 kB, one copy of the index 11.6 kB; the rest is the text of 137 functions (today's full C++ Markdown file is 447,013 bytes). Budgets are in section 15.4.

## 11. Formulas

- Syntax: `$...$` inline and `$$...$$` display (D5), parsed by `dollarmath` (section 5.6) in doc blocks, prose, data-file inline strings and notebook Markdown cells. Today no site source contains a formula (plan 5.1; measured: no `$` outside code in any doc block).
- Output: KaTeX with `output: "mathml"`. Reasons: MathML Core is supported by all browsers of plan 5.2 (current Chrome, Edge, Firefox and Safari), it needs no KaTeX stylesheet and no KaTeX fonts (about 300 kB of `woff2` files and 25 kB of CSS in katex 0.18.11), it is accessible to screen readers without the extra hidden copy that `htmlAndMathml` adds, and it is small (measured: a sum formula is 581 bytes as MathML, 3,936 bytes as KaTeX HTML). Trade-off: the glyphs come from the device's math font (CSS: `math { font-family: "Latin Modern Math", "STIX Two Math", "Cambria Math", math; }`, as in the prototype), so formulas look slightly different across systems. This satisfies D5 (rendered at build time, no math script in the browser, nothing from third parties); with MathML output there is nothing to self-host.
- One Node process per build (plan 5.1): `katex.py` collects every formula of the build (text and display flag, with its source file and line), removes duplicates, and only if at least one exists runs `node docs_builder/katex_render.mjs <katex directory>` once, with a JSON document on standard input and a timeout of 120 s. The katex directory is `docs_builder/node_modules/katex` if it exists, else `<npm root -g>/katex` (section 3.5); the script loads it with `createRequire(import.meta.url)(<katex directory>)`, so it works for both set-ups. Input: `{"items": [{"tex": "...", "display": true}, ...]}`. Output on standard output: `{"results": [{"ok": true, "html": "..."} | {"ok": false, "error": "..."}, ...]}` in the same order. The script calls `katex.renderToString(tex, {displayMode, output: "mathml", throwOnError: true, strict: "error", trust: false})` for each item and catches errors per item. A failed item is a structural error at its source location with KaTeX's message; a non-zero exit of Node or invalid JSON is an error naming the script. Tested in a scratch folder with katex 0.18.11 and Node 22.22.1: valid formulas give `<span class="katex"><math ...>`, `\frac{1}{2` gives "KaTeX parse error: Unexpected end of input ...".
- Twins keep the `$...$` source (GitHub renders it, D8).

## 12. Static parts

### 12.1 `docs_builder/static/`

Copied unchanged into the site root, with the same relative paths. Contents after step 5:

| Path | Source |
|------|--------|
| `css/site.css` | Port of the prototype `css/site.css` (section 12.2) |
| `js/site.js` | Port of the prototype `js/site.js` (section 12.3) |
| `fonts/ibm-plex-sans-latin-{400-normal,400-italic,500-normal,600-normal,700-normal}.woff2`, `fonts/ibm-plex-mono-latin-{400,500}-normal.woff2`, `fonts/LICENSE.txt` | Copied from the prototype `fonts/` (seven files, 152 kB, `@fontsource` 5.2.5 Latin subset; plan section 11: committed, not fetched from npm) |
| `img/logo.svg` | The prototype `img/logo.svg` (`fill="currentColor"`) |
| `img/favicon.svg` | The logo SVG with an embedded `<style>svg { color: #2e7d46 } @media (prefers-color-scheme: dark) { svg { color: #7fd197 } }</style>`, so that its `currentColor` fills are green in light and light green in dark browser themes |
| `img/favicon-32.png`, `img/apple-touch-icon.png` (180 × 180), `favicon.ico` (16 and 32) | Rendered once from `img/favicon.svg` with Inkscape and ImageMagick (both installed: `/usr/bin/inkscape`, `/usr/bin/convert`); committed |
| `img/og-card-placeholder.png` | 1200 × 630, rendered once from `docs_builder/figures/og-card-placeholder.svg` (the logo and "Quadriga-Lib" on `--bar` green) |
| `img/landing/` | Empty folder with `README.md` (one line: the figures of D15 go here, file names of design section 8); filled by the figure session |
| `.htaccess` | Section 12.4 |

`static/img/landing/README.md` is not copied (files named `README.md` in `static/` are skipped).

### 12.2 `site.css`

Port of the prototype stylesheet: every token value, breakpoint, size and colour stays (design 3, "the spec may restructure it but keeps the values"). Changes:

- Font URLs stay `../fonts/...` (the stylesheet is in `css/`).
- Remove the unused header search styles (`.search`, design 4.1: no search field).
- Code colour selectors replaced by the class set of section 10: `pre .k { color: var(--c-kw) }`, `pre .kt { color: var(--c-ty) }`, `pre .nf { color: var(--c-fn) }`, `pre .s { color: var(--c-st) }`, `pre .m { color: var(--c-nu) }`, `pre .c { color: var(--c-cm); font-style: italic }`, `pre .gp { color: var(--muted); user-select: none }`.
- Add `.params > .pnote { grid-template-columns: minmax(0, 1fr); }` (note rows over both columns at every width).
- Add `.section-desc` (already in the prototype as `.api-section > .section-desc`), `.math-block` and `math` rules (kept), `.prose h4` (`font-size: var(--fs-md)`; `margin-top: var(--s4)`), `.note-box` content spacing for multi-paragraph bodies (`.note-box > * + * { margin-top: var(--s2) }`).
- Print: also hide `.doc-side` and `.idx-filter`.
- Replace the prototype's inline `style` attributes (page padding, `margin-top` on headings and notes, the centred tutorial column) with classes: `.page` (`padding-top: var(--s6); padding-bottom: var(--s8)`), `.page-narrow` (`margin: 0 auto`), and spacing rules on `.page > h1`, `.page > .lead`, `.page > .note-box`, `.page > .grid-2` with the prototype's values, and `max-width: 46em` on the tutorial index lead and note. Column alignment of Markdown tables stays as markdown-it renders it (`style="text-align:right"` on cells), as in the prototype.
- `.ph` stays (gap marker), with the prototype's colours.

### 12.3 `site.js`

Port of the prototype script, behaviour as design section 7, about 4 kB: copy buttons, remembered language (`localStorage` key `qlib-lang`, written only in the `change` handler of the language tabs, D18), function filter with `/` shortcut, current-function highlight. Selectors match the markup of section 8: `.idx-filter`, `.idx`, `li.fn`, `.doc-side a[href^='#']`, `.fn-entry, .api-section, .prose h2[id], .prose h3[id]`, `[data-lang-tabs]`. Radio values of the landing tabs: `python`, `matlab`, `cpp`; Python first and checked in the HTML.

### 12.4 `.htaccess`

```
Options -Indexes
ErrorDocument 404 /404.html
AddType text/markdown .md
AddType font/woff2 .woff2
AddType image/webp .webp
AddCharset utf-8 .html .css .js .md .txt .svg
# Switch on only after HTTPS works at the hoster (plan section 10):
# RewriteEngine On
# RewriteCond %{HTTPS} off
# RewriteRule ^ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
```

`.md` files are served as `text/markdown; charset=utf-8` and `llms.txt` as `text/plain; charset=utf-8` (plan section 11; `.txt` is `text/plain` by default). Compression: measured on 2026-11-02, the server answers `Accept-Encoding: gzip, br` with an uncompressed `Content-Type: text/html` and no `Content-Encoding`, and serves `.md` without a content type. Whether `mod_deflate` or pre-compressed files can be enabled through `.htaccess` at this hoster cannot be tested from here; the site uses raw-byte budgets (section 15.4; decided, section 21.2, Q7). Whether `mod_deflate` works is tested after the first deploy (go-live checks, section 18.4); step 5 does not add any compression directive. Whether the hoster honours `AddType` and `AddCharset` in `.htaccess` is checked after the first deploy with `curl -sI <base>/formats.md` and `curl -sI <base>/llms.txt` (go-live checklist, section 18.4).

### 12.5 Preview image (design open point 5)

`og:image` is `<base_url>/img/og-card.png` if `docs_builder/static/img/og-card.png` exists, otherwise `<base_url>/img/og-card-placeholder.png` and the gap "preview image og-card.png missing". The real image is made in the figure session (design section 8). Favicon PNG fallback: section 12.1.

### 12.6 Landing figure slots

For each `figure` entry of `landing.toml` the template emits, if both `<base>-light.webp` and `<base>-dark.webp` exist under `docs_builder/static/`:

```html
<figure><div class="frame"><picture><source media="(prefers-color-scheme: dark)" srcset="img/landing/coverage-dark.webp"><img src="img/landing/coverage-light.webp" width="1600" height="1200" alt="..." decoding="async"></picture></div><figcaption>...</figcaption></figure>
```

with `loading="lazy"` when `lazy = true`. If either file is missing, the frame holds the placeholder drawing of the prototype (`placeholder = "coverage"` or `"antenna"`, the inline SVGs of the prototype `index.html` with `role="img"` and the entry's `alt` as `aria-label`), and the gap list gets "landing figure <id>: <base>-light.webp / -dark.webp missing". The figure scripts live in `docs_builder/figures/` (D24; design open point 6), use the released pip package (D15), run by hand (`python3 docs_builder/figures/<script>.py`, with the tutorial requirements, which include quadriga-lib, numpy and matplotlib) and write their WebP files directly into `docs_builder/static/img/landing/`; the images are committed (D15). Step 5 writes no figure script.

### 12.7 Hand-made files until the cutover

`[[static_extra]]` in `site.toml` copies `html_docu/publications/` to `publications/` and `html_docu/quadriga_lib_material_model.md` to `quadriga_lib_material_model.md` (D24). A missing source is a structural error. At the cutover these files move to `docs_builder/static/` with `git mv` and the `[[static_extra]]` entries are deleted (section 18). The old logo PNGs and `style2.css` are not copied; their URLs are dropped (decided, section 21.2, Q9). Logo sources in `html_builder/img/` are not read by the new builder (the cleaned `logo.svg` replaces them); they move to `docs_builder/logo/` at the cutover.

### 12.8 Settings for URLs

`base_url` (site base URL, `http://` until HTTPS works, plan section 10) and `download_base_url` (D25) are the only places these URLs are written; templates and content use the build variables of section 6.3. Moving the downloads to GitHub Releases changes `download_base_url` and `releases_url` only (D25).

## 13. Tutorial pipeline (D28, plan 5.3)

### 13.1 Notebooks

`tutorials/<stem>.ipynb`, `<stem>` matching `[a-z0-9][a-z0-9_-]*`, nbformat 4, stored without outputs: every code cell has `outputs == []` and `execution_count == null`. A notebook with outputs is a structural error in `tutorials` and `website` (`tutorials/<stem>.ipynb: error: notebook contains outputs; clear them before committing (Jupyter: Edit > Clear Outputs of All Cells)`); the builder never rewrites notebooks (plan 5.1).

### 13.2 Metadata

Notebook-level `metadata.quadriga_tutorial` (schema-checked, section 7.1):

| Key | Type | Req. | Meaning |
|-----|------|------|---------|
| `title` | str | yes | `h1`, card title, `<title>` |
| `summary` | str | yes | one line on the card and meta description; inline Markdown |
| `level` | str | yes | `beginner`, `intermediate` or `advanced`; shown capitalised |
| `duration_min` | int | yes | shown as "about N minutes" |
| `order` | int | yes | index order (ascending, then by title) |
| `dummy` | bool | no (false) | marks a pipeline test notebook (section 13.6) |

Language and kernel come from `metadata.kernelspec`: `name` is the kernel to run (for example `python3`, `jupyter_matlab_kernel`, `octave`); `language` gives the tag (`python` → "Python", `matlab` → "MATLAB", `octave` → "Octave") and the Pygments lexer. Cell-level `metadata.quadriga_tutorial.alt` (str) is the alt text for image outputs of that cell; missing gives the warning `figure-alt` and the alt text "Figure output of cell <n>".

### 13.3 Running (`make tutorials`)

For each notebook (sorted by stem): compute the cache key (section 13.4); if a cache entry with this key exists and `--force` is not given, skip. Otherwise, if the kernel is not installed (`jupyter_client.kernelspec.KernelSpecManager().find_kernel_specs()`; ipykernel in the same venv provides `python3`), skip with the warning of section 3.4. Otherwise execute with `nbclient.NotebookClient(nb, kernel_name=<name>, timeout=600, resources={"metadata": {"path": <empty work folder in the cache>}})`. A cell error stops this notebook with an error naming the cell index and the exception text; its old cache entry is deleted; the command continues with the next notebook and exits 1 at the end. A successful run converts the executed notebook (section 13.5) and replaces the cache entry. Cache entries of notebooks that no longer exist are deleted. The run uses the released pip package installed from `requirements-tutorials.txt` into the active tool environment (D28, section 4.4), not the library built in the repository.

### 13.4 Cache

Location `docs_builder/.cache/tutorials/<stem>/` (ignored, section 2.4). Contents: `key.txt`, `meta.json` (title, summary, level, duration, order, dummy, language, kernel), `fragment.html`, `page.md`, `figures/<files>`. The key is the SHA-256 hex digest of the concatenation of: the notebook file bytes; the library version (section 3.8); the kernel name and the installed versions (`importlib.metadata`, in the active environment) of quadriga-lib, numpy, matplotlib, ipykernel, nbclient, Pygments and markdown-it-py, written as `name==version` lines in sorted order; `qdocs.FORMAT_VERSION`. So a notebook is re-run when its source, the library version (in the repository or of the installed pip package), an upgraded tool or the output format change (D28). Since the requirement files hold minimums only (section 4), the installed versions, not the file, are the right key.

### 13.5 Conversion of an executed notebook

`fragment.html` is the final HTML of the page body below the downloads (KaTeX and Pygments already applied, one Node process per `make tutorials` run if any notebook has a formula):

- Markdown cell: parsed with the `notebook` parser (section 5.6; raw HTML is an error, h1 headings give the warning `notebook-h1` and are rendered as h2), rendered like prose, heading ids as section 6.2.
- Code cell: `<div class="cell"><div class="cell-label">In [n]</div><pre><code class="language-<lang>">...</code></pre>` then one `<div class="out">` per output, then `</div>`.
- Outputs, by type: `stream` → `<pre>` with the text (ANSI escape sequences removed); `execute_result` and `display_data` → the first available of `image/svg+xml`, `image/png`, `image/jpeg` (written to `figures/fig-<cell>-<output>.<ext>`, referenced as `<img src="<stem>/fig-<cell>-<output>.<ext>" alt="..." loading="lazy">`), `text/markdown` (rendered with the notebook parser), `text/latex` (display formula), `text/plain` (`<pre>`); `text/html` without another usable type is replaced by its `text/plain` with the warning `html-output`; `error` cannot occur (execution stops).
- `page.md`: the twin, same text (D26): twin header (section 8.6.1; web page URL `<base_url>/tutorials/<stem>.html`), a line with the two downloads, then Markdown cells verbatim (gap markers kept), code cells as fenced blocks with the language, text outputs as fenced `text` blocks, images as `![alt](<stem>/fig-<cell>-<output>.<ext>)`, Markdown outputs inline, LaTeX outputs as `$$` blocks. Image paths are relative to the twin, which sits next to the HTML page, so the same path works in both (D28: how figures are referenced).
- Determinism: matplotlib's inline backend writes PNG without timestamps; the cache stores the executed notebook only in memory; cell execution metadata (timestamps) is never written to the outputs.

### 13.6 Site build with tutorials

`website` reads, for every notebook in `tutorials/`, the cache entry whose `key.txt` equals the current key. A notebook without a current entry (never run, or stale) is left out with the warning `tutorial-missing` (`tutorials/<stem>.ipynb: warning: not built or out of date; run make tutorials`); `--release` turns it into an error. With no current entries at all the index shows no cards and a short line "No tutorials yet." (plan 5.3: site and Markdown builds work when the tutorial target has never run). Output per included notebook: `tutorials/<stem>.html` (template `tutorial.html` around `fragment.html`), `tutorials/<stem>.md` (`page.md`), `tutorials/<stem>.ipynb` (copy of the source notebook, without outputs, D28), `tutorials/<stem>/<figures>`. Download links (design 5.5 and 6): "Download notebook .ipynb" and "Markdown .md". The index (`tutorials.html` and its twin `tutorials.md`, both website only) lists cards in `order`; each card shows language tag, level tag, title, summary and links to the page. If every included notebook has `dummy = true`, the `dummy_note` of `content/tutorials.md` is shown as a warning note box (plan 5.3). Tutorial pages, their twins and figures never go into `docs/` (D28).

### 13.7 The dummy notebook

`tutorials/pipeline_demo.ipynb`, kernel `python3`, metadata `title` "Pipeline demo", `summary` "Placeholder notebook that tests the tutorial pipeline.", `level` "beginner", `duration_min` 5, `order` 1, `dummy` true. Cells (D28, plan 5.3: text, code, text output, a figure, a table, a formula; fixed seed):

1. Markdown: `## Set-up` and one sentence.
2. Code: `import numpy as np`, `import quadriga_lib`, `rng = np.random.default_rng(1)`, `print(quadriga_lib.version())`. Text output.
3. Markdown: `## An antenna` and the inline formula `$D = 10 \log_{10}(\max G)$`.
4. Code: `ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)` and `print(round(float(quadriga_lib.arrayant.calc_directivity(ant, 0)), 2))`.
5. Code with cell metadata `alt` "Line plot of 50 random values": matplotlib figure of `rng.standard_normal(50)` with `%matplotlib inline`. Image output.
6. Markdown: a three-row table and a display formula `$$\sigma^2 = \frac{1}{N}\sum_{n=1}^{N} (x_n - \bar x)^2$$`.

The exact function names follow the released package's Python API (the quick-start example of the design uses the same calls); step 5 checks them against quadriga-lib 0.12.2.

## 14. Agent skill placeholder (D10)

- Folder `docs_builder/skill/quadriga-lib/` with `SKILL.md`:

```
---
name: quadriga-lib
description: Placeholder for the Quadriga-Lib agent skill (radio channel modelling library with C++, MATLAB/Octave and Python APIs). Until the skill exists, read http://quadriga-lib.org/llms.txt.
---

# Quadriga-Lib skill (placeholder)

This skill is a placeholder. The documentation for agents starts at http://quadriga-lib.org/llms.txt.
```

  The URL is written with `{{ site_base_url }}` and substituted at build time (section 6.3). The placeholder is not a gap: the site may go live with it (plan 5.4 asks for a placeholder note on the AI page instead).
- `website` packs every file below `docs_builder/skill/` (sorted paths, so the zip contains `quadriga-lib/SKILL.md`) into `docs/website/skill/quadriga-lib-skill-<version>.zip` with `zipfile.ZipFile(..., compression=ZIP_STORED)`; each entry is a `ZipInfo` with `date_time=(1980, 1, 1, 0, 0, 0)`, `external_attr = 0o644 << 16`, `create_system = 3`. Stored, not deflated, so that the bytes do not depend on the zlib version (plan section 11: fixed timestamps, byte-identical).
- The AI page links it (section 6.8); `llms.txt` lists it (section 8.8).

## 15. Checks after the build

All checks run on the staging tree (section 3.6) of `website`; the Markdown checks also run for `docs` on the in-memory tree. A failed check is an error (exit 1) unless stated.

### 15.1 Links and anchors

For every HTML file: every `href` and `src` (also `srcset`) that is not absolute (`scheme:`) or `mailto:` is resolved against the file's folder (root-absolute paths against the site root); the target file must exist in the tree; if a fragment is given and the target is HTML, an element with that `id` must exist (after percent-decoding); if the target is a `.md` file, a heading with that GitHub slug (section 9.5) must exist. For every Markdown file (twins, section files, full files, `llms.txt`, `docs/README.md`, tutorial twins): every link and image URL (collected from markdown-it tokens) is checked the same way, relative to the file; absolute URLs that start with `base_url` are mapped to the tree and checked too (so links in twins and in `llms.txt` are checked although they are absolute). External URLs are not fetched. Example the check must catch (plan section 10): a FAQ without the `cite` id makes the footer link `faq.html#cite` fail on every page, and the twin link `faq.md#...` to the cite heading fail.

### 15.2 HTML well-formedness

`checks.py` parses each HTML file with `html.parser.HTMLParser` and a tag stack: every non-void element is closed in the right order (void elements: `area base br col embed hr img input link meta source track wbr`); no `id` occurs twice in a file; exactly one `h1`; exactly one `<main id="content">`; every `<img>` has an `alt` attribute. The `math` subtree from KaTeX is checked the same way (MathML elements are not void).

### 15.3 Twin consistency

Every page with `twin` in `site.toml` has its twin in the tree, and the `<link rel="alternate">` and footer link point to it; pages without `twin` have neither (D26).

### 15.4 Size budgets (warnings)

Raw bytes, because the server does not compress (section 12.4). Each budget is above today's measured or estimated size, so that growth is noticed early:

| Item | Budget | Measured or estimated basis |
|------|--------|-----------------------------|
| `cpp_api.html` | 800,000 | 694,128 for the prototype page with span-free colouring (section 10) |
| `mex_api.html`, `python_api.html` | 600,000 | about 460,000 and 445,000, scaled from today's pages (378,442 and 366,583 against 573,395 for C++) |
| other HTML pages | 150,000 | prototype `formats.html` 56,301 |
| landing total: `index.html` + `site.css` + `site.js` + all seven fonts + the larger variant of each landing figure | 550,000 | 23,536 + 25,614 + 3,828 + 152,012 + 2 × 150,000 (design 8: at most 150 kB per figure) = 505,000 |
| `index.html` alone | 40,000 | 23,536 |
| one per-section Markdown file | 150,000 | 128,481 (section 8.6.3) |
| `site.css` | 40,000 | 25,614 |
| `site.js` | 8,000 | 3,828 |

Each exceeded budget prints `warning: <file> is <n> bytes, budget <b> [size]`.

### 15.5 Optional validators (plan 8.3)

`make docs-validate` (`build.py validate`) needs the validators (`build.py setup-node --validators`, with `--global` in the conda set-up) and is never part of `docs` or `website`. It runs `html-validate --config docs_builder/.htmlvalidate.json "docs/website/**/*.html"` and `pa11y-ci --config docs_builder/.pa11yci.json`, each taken from `docs_builder/node_modules/.bin/` if present, else from `PATH`. `.htmlvalidate.json` extends `html-validate:recommended` and `html-validate:a11y`; rules that conflict with the design are switched off with a one-line reason each in the hand-over report (for example `no-inline-style` is not needed: the templates use no inline styles). `.pa11yci.json` lists `file://` URLs of one page per template (index, cpp_api, formats, publications, tutorials, one tutorial if built, impressum, 404), standard `WCAG2AA`, runners `axe` and `htmlcs`, and `chromeLaunchConfig.executablePath` from the environment variable `QDOCS_CHROME` (default `/usr/bin/google-chrome`, present on the build machine); `validate` passes it through. No legacy id contains whitespace (section 6.6), so the `valid-id` rule stays on. Neither tool needs Java (plan section 11). Findings are printed; the command exits 1 if either tool reports errors.

## 16. Package install and failsafe (D22)

### 16.1 `docs_builder/CMakeLists.txt`

Works in two modes: standalone (`cmake -S docs_builder`, for tests in step 5) and as a subdirectory of the top-level project (after the cutover, one line `add_subdirectory(docs_builder)`). Behaviour:

1. If `CMAKE_CURRENT_SOURCE_DIR` equals `CMAKE_SOURCE_DIR` (standalone): `cmake_minimum_required(VERSION 3.18)` and `project(QuadrigaLibDocs LANGUAGES NONE)`.
2. `option(ENABLE_DOCS_INSTALL "Install the prebuilt documentation from docs/website" OFF)` (decided, section 21.2, Q2: OFF by default; the package builds switch it ON, section 16.4) and `set(QDOCS_INSTALL_DIR "share/doc/quadriga-lib/html" CACHE STRING "Install destination of the documentation, relative to the install prefix")`.
3. Repository root `_qd_root` = absolute path of `${CMAKE_CURRENT_SOURCE_DIR}/..`. Cache variables for tests: `QDOCS_WEBSITE_DIR` (default `${_qd_root}/docs/website`) and `QDOCS_VERSION_HEADER` (default `${_qd_root}/include/quadriga_lib.hpp`).
4. Return without doing anything (no check, no install) when: `ENABLE_DOCS_INSTALL` is OFF; or `SKBUILD_PLATLIB_DIR` is defined (pip builds ship no documentation, D22); or `CMAKE_INSTALL_PREFIX` equals `_qd_root` (local layout of `make`, where `docs/website/` is already in place).
5. Version: `QUADRIGA_VERSION` if defined by the parent (top-level `CMakeLists.txt` lines 8 to 17 define it), else parsed from `QDOCS_VERSION_HEADER` with the same regular expression as the top-level file.
6. Stamp: `${QDOCS_WEBSITE_DIR}/docs-stamp.txt`. Read with `file(STRINGS ... REGEX "^version=")`; found version = the text after `version=`, or `none` if the folder or the file is missing.
7. If the found version differs from the version being built: `message(FATAL_ERROR ...)` with exactly this text (CMake adds the prefix):

```
The documentation in docs/website/ is missing or was built for another version (found: <found>, building: <version>). Run "make website" in the repository root, or configure with -DENABLE_DOCS_INSTALL=OFF to build the package without documentation.
```

8. Otherwise `install(DIRECTORY "${QDOCS_WEBSITE_DIR}/" DESTINATION "${QDOCS_INSTALL_DIR}")` (the default is the system-install path used today, top-level `CMakeLists.txt` line 620) and `message(STATUS "Documentation: installing docs/website (version <version>)")`.

The check runs while CMake processes this file, at configure time, before anything is compiled. After the cutover the `add_subdirectory` line sits right after the version block of the top-level file (line 18), so the check comes seconds after the start (D22). With the option OFF by default, the check and the install happen only where a package build switches it ON; section 16.4 lists how each path does that.

### 16.2 Stamp file

`docs/website/docs-stamp.txt`, written by `website` as the last step of a successful build (section 3.6), content exactly `version=<version>\n`. Deployed and installed with the site; it holds no date, so builds stay byte-identical.

### 16.3 Testing it on its own

`docs_builder/tests/test_cmake.py` (skipped if `cmake` is not on `PATH`; measured: cmake 4.2.3 on the build machine) configures `cmake -S docs_builder -B <tmp>/b -DCMAKE_INSTALL_PREFIX=<tmp>/inst -DQDOCS_WEBSITE_DIR=<tmp>/site -DQDOCS_VERSION_HEADER=<tmp>/quadriga_lib.hpp` with a fixture header `#define QUADRIGA_LIB_VERSION_STR "1.2.3"`, and with `-DENABLE_DOCS_INSTALL=ON` unless stated, and checks: (a) no site folder: configure fails and the output contains `found: none, building: 1.2.3`; (b) stamp `version=1.2.2`: fails with `found: 1.2.2`; (c) stamp `version=1.2.3` and one file `index.html`: configure succeeds, `cmake --install <tmp>/b` copies `index.html` and the stamp to `<tmp>/inst/share/doc/quadriga-lib/html/`; (d) case (a) with `-DENABLE_DOCS_INSTALL=OFF`: succeeds, installs nothing; (e) case (a) with `-DCMAKE_INSTALL_PREFIX=<repository root>`: succeeds, installs nothing; (f) case (a) with `-DSKBUILD_PLATLIB_DIR=<tmp>/x`: succeeds; (g) case (a) without any `ENABLE_DOCS_INSTALL` argument (default OFF): succeeds, installs nothing; (h) case (c) with `-DQDOCS_INSTALL_DIR=documentation`: installs into `<tmp>/inst/documentation/`. The top-level `CMakeLists.txt` is not touched before the cutover (D22).

### 16.4 How each build path uses the failsafe

Applied at the cutover (section 18.3); step 5 only provides `docs_builder/CMakeLists.txt`. The check logic exists once, in that file; the paths below call it, either through the top-level project or through a standalone configure of `docs_builder`.

| Path | Today | After the cutover |
|------|-------|-------------------|
| Debian package | `GNUmakefile` `deb-*` → `docker build` with `Dockerfile.ubuntu` (`COPY . .`) → `dpkg-buildpackage` → `debian/rules` `override_dh_auto_configure` with two configure runs, both `-DCMAKE_INSTALL_PREFIX=/usr`; build 1 (gcc13) installs `html_docu/`, build 2 (gcc11, MATLAB MEX only; only its `.mexa64` files are copied into the package) | `COPY . .` brings the ignored `docs/website/` into the container (`.dockerignore` keeps out `docs_builder/node_modules/`, `docs_builder/.venv/`, `docs_builder/.cache/` and the publication PDFs). Build 1 gets `-DENABLE_DOCS_INSTALL=ON` and so runs the check and installs the site; build 2 stays at the default OFF. |
| Early fail before Docker | none | A new `GNUmakefile` target `docs-install-check` runs `cmake -S docs_builder -B build_docs_check -DENABLE_DOCS_INSTALL=ON -DCMAKE_INSTALL_PREFIX=/usr` (configure only: the same check, no install). `deb-jammy`, `deb-noble` and `deb-resolute` get it as their first prerequisite, with the Docker build moved into their recipe (`deb-jammy: docs-install-check` and recipe `$(MAKE) release/$(DEB_JAMMY)`), so that under `make -j` the check finishes before any Docker build starts; a missing or stale site stops in seconds. |
| Release | `release: documentation package`, then `$(MAKE) deb`, `$(MAKE) pip` | Recipe of sequential steps: `$(MAKE) docs-check`, `$(MAKE) website-release`, `$(MAKE) docs-install-check`, `$(MAKE) package`, `$(MAKE) deb`, `$(MAKE) pip`. The release rebuilds the site first, so the check runs right after that rebuild, the earliest point where it can pass, and before the source package and any Docker build. Sequential recipe lines instead of prerequisites, so that `make -j release` keeps the order. |
| Windows package | `Makefile` `all` configures with `CMAKE_INSTALL_PREFIX=%CD%` (local layout: the check would never run); `release-win` runs `cmake --install ... --prefix $(DIST_DIR)` on that build without re-configuring and copies `html_docu\*` by `xcopy` to a hard-coded `0.11.5` path | `release-win` starts with a standalone configure `cmake -S docs_builder -B $(CMAKE_BUILD_DIR)\docs $(GENERATOR) -D ENABLE_DOCS_INSTALL=ON -D QDOCS_INSTALL_DIR=documentation -D CMAKE_INSTALL_PREFIX=%CD%\$(DIST_DIR)` (fails before anything is copied or zipped), and after the existing `cmake --install` of the library runs `cmake --install $(CMAKE_BUILD_DIR)\docs`; the `xcopy html_docu` line is deleted. The site goes to `documentation\` at the top of the zip: on Windows the FHS path `share\doc\quadriga-lib\html` means nothing, a top-level folder is found at once, and the name says what it holds without being confused with the repository's `docs/` (which holds only the Markdown). `README_win.md` is updated at the cutover. |
| pip wheels and sdist (`SKBUILD_PLATLIB_DIR`) | no documentation | unchanged: skipped (D22) |
| `make` and `cmake --install` into the local layout (`CMAKE_INSTALL_PREFIX` = repository root) | `html_docu/` stays in place | unchanged: skipped; `docs/website/` is already in place |
| A user's `cmake --install` into a system prefix | installs `html_docu/` | installs no documentation unless `-DENABLE_DOCS_INSTALL=ON` is given, which then runs the check |

## 17. Tests

### 17.1 Framework and running

pytest (version in section 4.1). Location `docs_builder/tests/`; fixtures in `docs_builder/tests/fixtures/<package id>/` (one subfolder per work package, section 19, so that no two packages own the same fixture). `docs_builder/pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    node: needs Node.js and katex (in docs_builder/node_modules or the active environment)
    cmake: needs cmake
    kernel: needs the tutorial packages and the python3 Jupyter kernel
    repo: reads the real repository sources
```

Run all: `make docs-test` (section 3.5) in the active tool environment; the suite runs the same in the conda and the venv set-up, and finds katex in either place. Tests marked `node`, `cmake` or `kernel` skip themselves with a reason when the tool is missing. The suite must run in under two minutes on the build machine without the `kernel` tests; it starts no GPU work and needs no network. `conftest.py` provides a fixture `mini_repo(tmp_path)` that copies `fixtures/wp1/mini_repo/` (a minimal repository: `include/quadriga_lib.hpp` with version `9.9.9`, `CMakeLists.txt` with `set(armadillo_version "1.0.0")`, two or three small blocks per language, a reduced `docs_builder/site.toml`, small content files) and a helper `run_build(root, *args)` that calls `qdocs.cli.main([... , "--root", root])` and returns exit code, stdout and stderr.

### 17.2 Test files and what they cover

| File | Package | Covers |
|------|---------|--------|
| `test_cli.py` | WP1 | commands and options; exit codes 0, 1, 2, 3; `--root`; usage message |
| `test_env.py` | WP1 | Python version message (simulated); missing package message without traceback (simulated `ImportError`), exit 2; `docs` imports no Pygments, Jinja2, nbclient (checked with `sys.modules`); a package below its minimum (simulated) gives the upgrade message; `docs --check` prints the tool-version lines of section 3.9 on a difference; `setup-node` builds the right npm command for `--global` and local (command checked, npm not run) |
| `test_config.py` | WP1 | `site.toml` schema errors with key paths; version and Armadillo version read once; missing header error |
| `test_output.py` | WP1 | writing only changed files; stale files in `docs/` and `docs/website/` removed, `docs/website/` untouched by `docs`; `--check` writes nothing and exits 3 on a changed, a missing and an extra file, 0 when equal; refusal of paths under `download/` and `releases/`; stamp deleted at the start of `website` and written last; no write when a structural error exists |
| `test_gitignore.py` | WP1 | (`repo`) `git check-ignore --no-index` reports none of the files the builder tracks (every file under `docs/` except `website/`, every committed file under `docs_builder/`, `tutorials/*.ipynb`, fixture names such as `test_a.txt` or `release/x` below `docs_builder/tests/fixtures/`) and reports `docs/website/x`, `docs_builder/.venv/x`, `docs_builder/node_modules/x`, `docs_builder/.cache/x`, `docs_builder/qdocs/__pycache__/x.pyc` |
| `test_docblocks.py` | WP2 | extraction and line numbers; every error id of section 5.2 with its file and line (one fixture each); `<++>` ignored; sub-block labels and kinds; required sub-blocks including the class-block exemption, and E-required for a MATLAB or Python block with `## Declaration:` instead of `## Usage:`; ignored non-`.cpp` files |
| `test_params.py` | WP2 | the item pattern on every variant of section 5.4 (single name, several names in one bold, `, ` and ` / ` between bolds, `...`, each qualifier form, separators em dash, en dash, hyphen, `<br>`, none); note rows and the `param-item` warning; nested table and list in a description |
| `test_mdparse.py` | WP2 | `wikilink` including the member form `.[[name]]` (also after a word, `x.[[name]]`, which is then unresolved in MATLAB and Python); `raw_a` (drops `target`, `rel`); `raw_br` (trailing breaks removed, others kept, `<br>` in table cells kept); `no_html` error in prose; `heading_attrs`; `site_url`; `dollarmath` not triggered by `$5 and $6`; no rule touches code spans or fences |
| `test_links.py` | WP2 | `[[...]]` resolution on the own page only, E-link for a name of another language or a file stem; raw cross-language links (`cpp_api.html#x` from a MATLAB block) in HTML and in both twin kinds, and E-link for a raw link to a missing function anchor; URL of every link kind of section 8.7.1 for each output location (root HTML, `tutorials/` HTML, full file, own section file, other section file, other language) |
| `test_anchors.py` | WP2 | `slug()`; legacy hashes equal the table of section 9.3; `gh_slug()` test vectors of section 9.5 with duplicates; E-anchor cases |
| `test_matching.py` | WP2 | file-stem families; name rule including `sos.` names; mapping overrides; mapping errors (unknown stem, unknown C++ name, name in two families); pills for primary and further C++ functions |
| `test_prose.py` | WP3 | front matter errors (missing, unknown key, wrong type, no `+++`); build variables including the unknown-name error; h1 and too-deep headings rejected; note boxes; explicit and legacy ids |
| `test_structured.py` | WP3 | `landing.toml` and `publications.toml` schemas: unknown key, missing key, wrong type, wrong count, each reported with file and key path; inline Markdown rendered, newline rejected |
| `test_gaps.py` | WP3 | marker rendering (HTML span, plain text in meta, verbatim in twins); unclosed marker error; gap list with paths and lines, sorted; missing figure and placeholder preview image listed; `gaps --release` and `website --release` exit 1 with gaps, 0 without |
| `test_render_md.py` | WP3 | golden files from `mini_repo` for: a full API file (front matter without date, index without "Line", "Also in" line), a section file, a prose twin (index table, alert boxes, variables substituted, gap kept, heading braces removed, GitHub-slug links), the publications twin, `llms.txt`, `docs/README.md`; parameter rows in SPEC form |
| `test_content.py` | WP4 | (`repo`) the real content files parse without errors; `faq.md` has the id `cite`; the formats, FAQ and download legacy ids of sections 6.6 and 9.3 are present; the landing and publications files pass their schemas |
| `test_render_html.py` | WP5 | golden snippets from `mini_repo`: head tags of section 8.3 (alternate link and footer twin link only with a twin), header tabs with `aria-current`, function entry, parameter list with qualifier and note row, pills (current, link, missing), section ids with legacy ids, index lists (sidebar and mobile), prose heading ids with legacy spans, tables wrapped, note boxes, gap spans; `404.html` uses only root-absolute internal URLs; tutorial pages use `../` |
| `test_highlight.py` | WP5 | class mapping table of section 10; no `<span>` for names, operators, punctuation, whitespace; adjacent merge; escaping; unknown language warning |
| `test_katex.py` | WP5 | (`node`) one Node process for several formulas (counted by patching `subprocess.run`), none when there is no formula; MathML output; a parse error reported at the source line; missing `node` message (simulated) |
| `test_checks.py` | WP5 | link check: missing file, missing HTML id, missing Markdown heading slug, absolute `base_url` link in a twin, and the `faq.html#cite` case of plan section 10 (fixture FAQ without `cite`) in HTML and in Markdown; well-formedness: unclosed tag, wrong nesting, duplicate id, two `h1`, `img` without `alt`; size warnings |
| `test_static.py` | WP6 | (`repo`) `site.css` keeps every token value of design 3.1 (light and dark tables parsed from the CSS and compared with a fixture copy of the design tables); the seven fonts and `LICENSE.txt` exist; `.htaccess` has the lines of section 12.4; `site.js` contains the `qlib-lang` key and writes it only inside the `change` handler (text check) |
| `test_tutorials.py` | WP7 | metadata schema errors; notebook with outputs rejected; cache key changes with notebook bytes, library version, an installed package version (simulated), format version; `website` without cache: no tutorials, warning, exit 0; with a fake cache entry: page, twin, notebook copy and figures written, index card, dummy note; stale entry left out with warning and `--release` error; output conversion of each output type from a fixture executed notebook (no kernel needed); (`kernel`) running `pipeline_demo.ipynb` end to end |
| `test_skill.py` | WP8 | zip entries, `ZIP_STORED`, fixed timestamps and attributes; two builds give identical bytes |
| `test_cmake.py` | WP8 | (`cmake`) cases (a) to (h) of section 16.3 |
| `test_real_repo.py` | WP9 | (`repo`) `docs` on the real repository into a temporary copy of the outputs: zero errors; function counts 137, 85, 80; section counts 10, 9, 9; every id of `fixtures/wp9/old_anchors.json` (captured once from `html_docu/*.html`, without the dropped ids of section 9.6) exists on the matching new page, and `index.html#quickstart`; byte-identical output of two `website` runs into separate roots (without tutorials); `docs --check` passes right after `docs` |

### 17.3 Golden files

Golden outputs live in `fixtures/<package>/golden/`. A test compares bytes and, on a difference, prints a unified diff. Updating golden files is a deliberate act (`QDOCS_UPDATE_GOLDEN=1 make docs-test`), never automatic.

## 18. Cutover checklist (D31)

Done after go-live and the maintainer's sign-off, not in step 5. Step 7 copies this section into `docs_builder/README.md`. Order matters where noted.

### 18.1 Files that move (`git mv`)

1. `html_docu/publications/` → `docs_builder/static/publications/`.
2. `html_docu/quadriga_lib_material_model.md` → `docs_builder/static/quadriga_lib_material_model.md`.
3. `html_builder/extract_matlab_comments.py` → `docs_builder/extract_matlab_comments.py`. It keeps to the Python standard library and runs on Python 3.10 (deb build in a jammy container, CMake on Windows); it must not import `qdocs`.
4. `html_builder/img/*` → `docs_builder/logo/` (logo sources, not deployed).
5. Then delete the two `[[static_extra]]` entries in `docs_builder/site.toml`.

### 18.2 Files that are deleted

`html_builder/` (all remaining files, including `parts/`, `formats/`, `faq/`, `download/`, the scripts and `README.md`), `html_docu/` (all remaining files). The design prototype is already gone in step 7.

### 18.3 Files that change

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `add_subdirectory(docs_builder)` after the version block (after line 18); the option default stays OFF (section 16.4). Delete `set(INSTALL_DOCDIR html_docu)` (line 608), `set(INSTALL_DOCDIR share/doc/quadriga-lib/html)` (line 620) and `install(DIRECTORY html_docu/ DESTINATION ${INSTALL_DOCDIR})` (line 629). Change the path in the `mex_documentation` command (line 499) to `${CMAKE_SOURCE_DIR}/docs_builder/extract_matlab_comments.py`. |
| `GNUmakefile` | Delete `HB = html_builder` and the `documentation` target. Add `docs-install-check` and change `deb-jammy`, `deb-noble`, `deb-resolute` and `release` as in section 16.4. `package_dirty`: replace `-x 'html_docu/releases/*'` with `-x 'docs_builder/node_modules/*' -x 'docs_builder/.venv/*' -x 'docs_builder/.cache/*'`. `clean`: no change (keeps `docs/website/` and `docs_builder/.cache/`). `tidy` (decided, section 21.2, Q10): add `- rm -rf docs/website` and `- rm -rf docs_builder/.cache` (the tutorial cache included); `docs_builder/.venv/` and `docs_builder/node_modules/` are installed tools, not build products, and stay. This changes the existing `tidy` target, so it waits for the cutover. |
| `Makefile` (Windows) | `release-win` as in section 16.4: standalone configure of `docs_builder` as the first command, its install after the library install, and the line `if exist html_docu xcopy ... "release\quadriga_lib_0.11.5_win64\html_docu\"` (line 95) deleted. |
| `Dockerfile.ubuntu` | No change needed: `COPY . .` copies the ignored `docs/website/` from the working tree; build 1 of `debian/rules` then runs the check and installs it. Check after the first deb build that `share/doc/quadriga-lib/html/index.html` is in the package listing printed in step 5 of the Dockerfile. |
| `.dockerignore` | Replace `html_docu/releases/` and `html_docu/publications/` with `docs/website/publications/` (keeps the 13 MB of PDFs out of the deb, as today), `docs_builder/node_modules/`, `docs_builder/.venv/`, `docs_builder/package-lock.json`, `docs_builder/.cache/`, `docs_builder/static/publications/` (the first three exist only in the venv set-up; the conda environment is outside the repository). |
| `debian/stamp-changelog.sh` | No change: "HTML API documentation included." stays true, since the deb build installs `docs/website/`. |
| `debian/rules` | Build 1 (`dh_auto_configure`, gcc13): add `-DENABLE_DOCS_INSTALL=ON`. Build 2 (gcc11, MATLAB MEX): no change (default OFF). |
| `.gitattributes` | `html_docu/publications/ export-ignore` becomes `docs_builder/static/publications/ export-ignore`. |
| `.gitignore` | Delete `html_docu/releases/`. |
| `pyproject.toml` | Check only: pip builds skip the documentation install (option OFF by default and `SKBUILD_PLATLIB_DIR`, D22). Optional: `[tool.scikit-build] sdist.exclude = ["docs_builder/static/publications/"]` to keep the PDFs out of the source distribution. |
| `README.md` | Make-target table: replace `make documentation` with `make docs`, `make website`, `make tutorials` (step 7 already adds the new targets); repository-layout table: replace the `html_docu/` and `html_builder/` rows with `docs/` (Markdown documentation, generated, tracked), `docs_builder/` (documentation builder, page sources, site settings) and `tutorials/` (Jupyter notebooks). |
| `README_win.md` | Package-contents table: `html_docu/` becomes `documentation/` (section 16.4). |
| `md/SPEC_CPP_API.md` | Line 13: "the pages in `html_docu/`" becomes "the pages of quadriga-lib.org". Lines 48, 97, 243: `html_docu/quadriga_lib_api_cpp.md` becomes `docs/quadriga_lib_api_cpp.md`. Line 208: `make documentation` becomes `make docs`. Lines 244 and 245: generated by `make docs`, regenerated by hand and may lag behind `main` between regenerations (D8); replace the "Line" column advice with: read the function index of the full file, then open only the section file `docs/api/cpp/<section>.md` that holds the function. |
| `md/SPEC_MATLAB_API.md` | Lines 12 to 14 and 387: `make documentation` → `make docs`, `html_docu/quadriga_lib_api_*.md` → `docs/quadriga_lib_api_*.md`, "may lag behind" as above. Line 16: the help script path `docs_builder/extract_matlab_comments.py`. |
| `md/SPEC_PYTHON_API.md` | Lines 15 to 18 and 554: as for the MATLAB file. |
| `md/BUG_SCAN.md` | Line 68: "A stale `html_docu/` alone" becomes "Stale generated Markdown in `docs/` alone" (`make docs-check` shows it). |
| `md/acoustic_modeling_guidelines.md` (line 5), `md/acoustic_engine_notes.md` (line 15), `md/materials_kickoff.md` (line 33), `md/materials_manager.md` (line 189), `md/gbsm_integration_handoff.md` (line 50) | `html_docu/quadriga_lib_material_model.md` becomes `docs_builder/static/quadriga_lib_material_model.md`; `html_docu/` after `make documentation` becomes `docs/` after `make docs`. Some of these are working notes; the maintainer may instead leave or delete them. |
| `tools/update_armadillo.py` | Remove `html_builder/download/quick_start.md` from `VERSION_FILES` and the comment above it: the Get started page now takes the Armadillo version from `CMakeLists.txt` through `{{ armadillo_version }}`. |
| `tools/acoustic/check_docs.py` | `PAGE` becomes `docs_builder/static/quadriga_lib_material_model.md`; `FORMATS` becomes `docs_builder/content/formats.md`. Run it once: the formats file now uses `###`/`####` headings instead of `/*!MD` blocks, so its parser may need a small change. |
| `tools/acoustic/README.md` (lines 4, 10), `tools/acoustic/band_probe.cpp` (line 4), `tools/acoustic/gzlib.py` (line 4), `tools/acoustic/paris_cf.py` (line 4), `tools/acoustic/txlib.py` (line 4), `tools/sos_port/SOS_PAPER_REPRODUCTION.md` (line 5), `tests/catch2_tests/test_ieee_indoor_publication.cpp` (line 7) | Comments naming `html_docu/quadriga_lib_material_model.md` or `html_docu/publications/...`: point to `docs_builder/static/...` or to the URL on quadriga-lib.org. Only comments change. |

### 18.4 Checks after the cutover

1. `make docs-check` passes; `make website-release` passes; `make docs-test` passes.
2. `cmake -S . -B build_check -DENABLE_DOCS_INSTALL=ON -DCMAKE_INSTALL_PREFIX=/tmp/qd_inst` stops with the failsafe message when `docs/website/` is missing, and succeeds after `make website`; `cmake --install` puts the site into `/tmp/qd_inst/share/doc/quadriga-lib/html/`.
3. A MEX build still writes the `.m` help files (`make` with MATLAB or Octave on).
4. A deb build contains `share/doc/quadriga-lib/html/index.html`; a Windows package contains `documentation\index.html`; `make docs-install-check` with a stale stamp stops before any Docker build.
5. `git grep -n 'html_docu\|html_builder'` finds nothing outside the git history and `docs_builder/README.md`.

Go-live checks (before or at go-live, after the first deploy; plan section 7): `curl -sI <base>/formats.md` shows `Content-Type: text/markdown; charset=utf-8`; `curl -sI <base>/llms.txt` shows `text/plain; charset=utf-8`; `curl -sI <base>/no-such-page` returns 404 with the new page; compression test (section 21.2, Q7): add `AddOutputFilterByType DEFLATE text/html text/css application/javascript text/markdown text/plain` to `.htaccess` on the server, then `curl -sI -H 'Accept-Encoding: gzip' <base>/cpp_api.html` must show `Content-Encoding: gzip` and a normal request must still return the page; if it does, the line goes into `docs_builder/static/.htaccess`, otherwise it is removed again; `python3 docs_builder/build.py gaps --release` exits 0.

## 19. Work packages for step 5

Each package owns the files listed; no file has two owners. Shared interfaces are the dataclasses of section 5.7 (owned by WP2) and the function signatures named below; a package that needs a change in a file it does not own asks the manager.

| Id | Package | Owns | Depends on |
|----|---------|------|------------|
| WP1 | Foundation | `build.py`; `qdocs/__init__.py`, `cli.py`, `env.py`, `diag.py`, `config.py`, `schema.py`, `output.py`; `site.toml`; `requirements*.txt` (three files); `package.json`; `pytest.ini`; `tests/conftest.py`; `tests/fixtures/wp1/`; `test_cli.py`, `test_env.py`, `test_config.py`, `test_output.py`, `test_gitignore.py`; the `.gitignore` block; the `GNUmakefile` targets | none |
| WP2 | Doc-block parser and model | `qdocs/docblocks.py`, `model.py`, `mdparse.py`, `params.py`, `links.py`, `anchors.py`, `matching.py`; `content/api_matching.toml`; `tests/fixtures/wp2/`; `test_docblocks.py`, `test_params.py`, `test_mdparse.py`, `test_links.py`, `test_anchors.py`, `test_matching.py` | WP1 |
| WP3 | Markdown output, prose and data pages | `qdocs/prose.py`, `structured.py`, `gaps.py`, `render_md.py`, `mdformat_ext.py`; `tests/fixtures/wp3/`; `test_prose.py`, `test_structured.py`, `test_gaps.py`, `test_render_md.py` | WP2 |
| WP4 | Content conversion | `content/formats.md`, `faq.md`, `download.md`, `ai_agents.md`, `impressum.md`, `privacy.md`, `tutorials.md`, `404.md`, `api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md`, `landing.toml`, `publications.toml`, `llms.toml`; `tests/fixtures/wp4/`; `test_content.py` | WP3 (formats of sections 6 and 7) |
| WP5 | HTML output and checks | `qdocs/render_html.py`, `highlight.py`, `katex.py`, `checks.py`; `katex_render.mjs`; `templates/*`; `.htmlvalidate.json`, `.pa11yci.json`; `tests/fixtures/wp5/`; `test_render_html.py`, `test_highlight.py`, `test_katex.py`, `test_checks.py` | WP2, WP3; WP6 for class names (fixed by this spec, so WP5 and WP6 can run in parallel) |
| WP6 | Static parts | `docs_builder/static/**` (CSS, JS, fonts, images, `.htaccess`, `img/landing/README.md`); `qdocs/staticfiles.py`; `figures/og-card-placeholder.svg`; `tests/fixtures/wp6/`; `test_static.py` | WP1 |
| WP7 | Tutorials | `qdocs/tutorials.py`; `tutorials/pipeline_demo.ipynb`; `tests/fixtures/wp7/`; `test_tutorials.py` | WP5 (fragment rendering, highlighting, KaTeX), WP3 |
| WP8 | Skill and package failsafe | `docs_builder/skill/quadriga-lib/SKILL.md`; `qdocs/skill.py`; `docs_builder/CMakeLists.txt`; `tests/fixtures/wp8/`; `test_skill.py`, `test_cmake.py` | WP1 |
| WP9 | Integration | `tests/fixtures/wp9/` (including `old_anchors.json`); `test_real_repo.py`; the full-build run and its report | all others |

Order: WP1; then WP2, WP6 and WP8 in parallel; then WP3; then WP4 and WP5 in parallel; then WP7; then WP9. The `website` command wires the modules together in `cli.py` (WP1) through these entry points, which the owning packages implement with exactly these names: `docblocks.load_api(root, lang, config, diag) -> ApiPage`, `prose.load_page(path, config, diag) -> ProsePage`, `structured.load_landing(...)`, `structured.load_publications(...)`, `render_md.render_docs(model, config) -> dict[str, bytes]`, `render_html.render_site(model, config, diag) -> dict[str, bytes]`, `staticfiles.collect(root, config, diag) -> dict[str, bytes]`, `tutorials.collect_for_site(root, config, diag) -> list[Tutorial]`, `skill.build_zip(root, version) -> bytes`, `checks.run(tree, config, diag)`, `gaps.scan(root, config, model) -> list[Gap]`. Until a package delivers, WP1 uses stubs that return empty results.

Acceptance criteria (each package also passes its own tests and leaves the tests of earlier packages green):

| Id | Done when |
|----|-----------|
| WP1 | `make docs`, `make website`, `make docs-check` run end to end with stubs; exit codes and messages of sections 3.1 and 3.4 shown by tests; `.gitignore` test passes; both tool set-ups work from scratch: `make docs-setup-conda` creates `quadriga-web` with Python, Node and all packages and leaves nothing in the repository, and `make docs-setup-venv` creates `docs_builder/.venv/` and `docs_builder/node_modules/`; in each, `make docs`, `make website` (with a fixture formula) and `make docs-test` pass; a package below its minimum gives the message of section 3.4; `docs --check` prints the tool versions on a difference. |
| WP2 | All real sources of the merged tree (D20 as amended) parse with zero errors (an unmerged tree shows exactly the twelve errors of the doc-block fix list); warning counts reported in the hand-over (expected near: 17 `param-separator`, 25 `param-item`); the measured numbers of sections 5.4 and 9.4 reproduced by tests. |
| WP3 | Golden-file tests pass; `make docs` on the real repository writes the files of section 2.3 with zero errors; `docs --check` passes right after; `docs` output contains no date. |
| WP4 | Real content passes all schemas; the legacy ids and `cite` exist; every draft text from the prototype and every design-section-9 item is marked as a gap; `make docs` output reviewed by the manager against the old pages for lost text (a script compares the plain text of old and new formats, FAQ and download pages and lists differences). |
| WP5 | `make website` builds all pages; link, anchor and well-formedness checks pass on the real site; C++ page within budget; the pages match the prototype at 375, 768 and 1280 px in the browser pane (manager's check, step 6 repeats it). |
| WP6 | CSS values equal the prototype's (test); the favicon, PNG fallbacks and preview placeholder exist with the stated sizes; `.htaccess` as section 12.4. |
| WP7 | With the tutorial requirements installed, `make tutorials` runs `pipeline_demo.ipynb`, a second run skips it, and `make website` shows it with figure, table, formula and downloads; without the cache, `make website` passes with the warning. |
| WP8 | CMake cases (a) to (h) pass; zip byte-identical across runs. |
| WP9 | `test_real_repo.py` passes; the manager gets a report with the gap list, warning counts, page sizes against budgets and the run time of `make docs` and `make website`. |

## 20. Traceability

Every item of design section 11, plan section 10 and plan section 11, with the section of this spec that answers it.

| Source | Item | Answered in |
|--------|------|-------------|
| Design 11.1 | Matching rule for the language pills | 9.4 (rules, mapping file, measured counts), 8.4 (pill markup), 21.2 Q4 |
| Design 11.2 | Section anchors: readable ids and old hash ids | 9.2, 9.3, 8.4 (legacy id on the `h2`), 6.2 and 6.6 (prose pages) |
| Design 11.3 | Names and paths of the per-section Markdown files, footer "This page as Markdown" | 2.3, 8.6.3, 8.4 ("Markdown" link per function), 3.3 (footer item `twin = true`) |
| Design 11.4 | Pygments without spans for uncoloured tokens; API page budget | 10, 15.4 |
| Design 11.5 | Favicon PNG fallback, preview image `og-card.png` | 12.1, 12.5, 8.3 |
| Design 11.6 | Where the figure scripts live, how images are committed | 12.6, 2.1 |
| Design 11.7 | MATLAB and Python pages: labels, lower-case names | 5.3, 5.5, 6.7, 3.3 |
| Design 11.8 | Placeholder marking, look on the page, build list | 6.4, 12.5, 12.6, 3.1 (`gaps`, `--release`) |
| Plan 10 | Paths of the design mapped to D24 | 1.2, 2.3 |
| Plan 10 | FAQ provides `#cite`; link check catches such gaps | 6.6, 15.1, 17.2 (`test_checks.py`, `test_content.py`) |
| Plan 10 | Page size: budget per API page | 10, 15.4 |
| Plan 10 | Old section anchors keep working | 9.3, 6.6, 9.6, 17.2 (`test_real_repo.py`) |
| Plan 10 | HTTPS: site base URL setting, `http://` until SSL, commented redirect, doc-block links stay | 3.3, 12.4, 12.8, 5.6 (`site_url` rule leaves the blocks unchanged) |
| Plan 10 | "For team leads" points into the landing page | 7.2 |
| Plan 10 | Design open points answered using D24 to D31 | this table (design rows) |
| Plan 11, cutover | `CMakeLists.txt`: install lines replaced by `docs_builder/CMakeLists.txt`; help script path; Python 3.10 and standard library | 18.1, 18.3, 16.1 |
| Plan 11, cutover | `Makefile`, `Dockerfile.ubuntu`, `.dockerignore`, `debian/stamp-changelog.sh`, `.gitattributes`, `GNUmakefile` (`documentation`, `release`, `package_dirty`, `clean`, `tidy`) | 18.3, 16.4 (build paths of the failsafe) |
| Plan 11, cutover | Text naming old paths: SPEC files ("Line" column, "may lag behind"), `md/BUG_SCAN.md`, `md/acoustic_modeling_guidelines.md`, `md/gbsm_integration_handoff.md`, `README.md`, `README_win.md`, `tools/update_armadillo.py`; further users found | 18.3 (also `md/acoustic_engine_notes.md`, `md/materials_kickoff.md`, `md/materials_manager.md`, `tools/acoustic/*`, `tools/sos_port/SOS_PAPER_REPRODUCTION.md`, `tests/catch2_tests/test_ieee_indoor_publication.cpp`, `.gitignore`, `debian/rules`, `pyproject.toml`) |
| Plan 11 | Full API files at `docs/quadriga_lib_api_<lang>.md`, no build date, no "Line" column; per-section files under `docs/api/<lang>/`; largest section about 127 kB accepted or split | 8.6.2, 8.6.3 (accepted, with reason), 2.3 |
| Plan 11 | Links in twins to files outside `docs/`; raw `<a href>` in doc blocks from every output location | 8.7.1, 8.7.3, 5.6 |
| Plan 11 | One Markdown intro per language in `docs_builder/content/`; Python intro is a content gap | 6.7, 21.2 Q5 |
| Plan 11 | Build variables and gap markers with different syntax; strict mode fails on gaps and missing figures | 6.3, 6.4, 3.1 (`--release`), 12.5, 12.6 |
| Plan 11 | Explicit heading ids (`#cite`) stripped from twins | 6.2, 9.5 |
| Plan 11 | Fonts committed in `docs_builder/static/fonts/`; KaTeX HTML or MathML chosen with reason | 12.1, 11 |
| Plan 11 | Cross-language matching: MATLAB and Python by file name; C++ rule or mapping file | 9.4 |
| Plan 11 | Doc-block grammar: `*(optional)*`, `(optional)`, `*(in/out)*`, several names, raw HTML, `<++>`, `## Usage:` | 5.2, 5.3, 5.4, 5.6 |
| Plan 11 | Pygments: uncoloured tokens without `<span>` | 10 |
| Plan 11 | Byte-identical output: sorted lists, fixed zip timestamps, no dates | 3.9, 14, 13.5, 17.2 |
| Plan 11 | Server without compression and without `.md` type: raw budgets or compression; content types in `.htaccess` | 12.4, 15.4, 21.2 Q7 |
| Plan 11 | `404.html` with root-absolute links; never write `download/` or `releases/` | 8.1, 2.3, 17.2 (`test_output.py`, `test_render_html.py`) |
| Plan 11 | Python 3.11 (`tomllib`); no Java validators | 4.1, 3.2, 15.5 |
| Plan 11 | Privacy page mentions the storage key | 6.8, 12.3 |
| Plan 11 | Generated `docs/README.md` | 8.9 |

## 21. Open questions for the maintainer

### 21.1 Conflicts found

**C1. D7 (strict links and required headings) against D20 (doc blocks unchanged). Resolved.** The new builder's checks found twelve doc-block defects on `main` (eleven `[[...]]` links that name no function on their own page, one Python block with `## Declaration:` instead of `## Usage:`), so D7 applied literally would stop the first build while D20 forbade fixing the blocks. Decided by the maintainer on 2026-11-02: D20 is amended; the twelve blocks are fixed on `main` in a separate session (fix list `/home/stephan/quadriga-lib-agents/website/docblock_fixes.md`), `main` is merged into `website` before the spec review, and D7 stays strict: no fallbacks, an unresolved `[[name]]` is E-link, a MATLAB or Python block without `## Usage:` is E-required (sections 5.3, 5.6, 8.7.2). Cross-language references are raw links to the other API page (section 8.7.3). Expected error count on the merged tree: zero; the measurements are re-run then.

### 21.2 Product decisions

**Q1. Decided:** see C1 (strict D7, blocks fixed on `main`).

**Q2. Decided: `ENABLE_DOCS_INSTALL` is OFF by default; the package builds switch it ON** (Debian build 1, the early check before Docker, the release target, the Windows package; section 16.4). Source users who install into a system prefix get no error and no documentation unless they ask for it.

**Q3. Decided: drop** the four formats entry ids with spaces (`1. Material Table`, `2. Built-in EM Materials`, `3. Principled BSDF`, `4. References`); sections 6.6, 9.6, 15.5 and the anchor fixture of `test_real_repo.py`.

**Q4. Decided: C++ mapping and pill target (section 9.4).** The mapping file lists the C++ functions the family's wrappers call (evidence: `api_mex/cart2geo.cpp:104` calls `fast_cart2geo`, `api_mex/geo2cart.cpp:139` `fast_geo2cart`, `api_mex/interp.cpp` `interp_2D`, `api_mex/calc_angular_spread.cpp` `calc_angular_spreads_sphere`, `api_mex/hdf5_create_file.cpp` `hdf5_create`; all entries checked against the wrappers). Where a family maps to several C++ functions, the first one listed is the pill target, member function before `_multi`; the order in the file is the rule.

**Q5. Decided: merged usage notes** (section 6.7): the Markdown copy from `extract_md.py` plus the `*_3dB` example in the C++ units line.

**Q6. Decided: drop `target="_blank"`** (and `rel`); all links open in the same tab (sections 5.6, 6.6).

**Q7. Decided: raw-byte budgets now** (sections 12.4, 15.4); `mod_deflate` through `.htaccess` is tested after the first deploy (go-live checks, section 18.4), not in step 5.

**Q8. Decided:** FAQ questions and Get started items become `###` headings, the quick-start labels too, and the formats page follows the prototype's order (QDANT first) (section 6.6).

**Q9. Decided: drop** `style2.css`, `quadriga-lib-logo80.png` and `quadriga-lib-q_letter64.png` (sections 9.6, 12.7).

**Q10. Decided: `make tidy` deletes `docs/website/` and `docs_builder/.cache/`** (the tutorial cache included); `make clean` keeps both; `tidy` keeps `docs_builder/.venv/` and `docs_builder/node_modules/`, which are installed tools, not build products (to be confirmed by the maintainer); the conda environment `quadriga-web` is outside the repository and is never touched by any make target except `docs-setup-conda`. It changes an existing target and is therefore a cutover item (section 18.3).

## Appendix A. Measurements

All measured on 2026-11-02 in the working copy (branch `website`, before the doc-block fixes and the merge of `main`), with scripts and venvs in `/home/stephan/quadriga-lib-agents/website_step3/`, nothing written to the repository. The rows on doc blocks are re-run on the merged tree before the spec review (D20 as amended); expected changes: zero unresolved links, every MATLAB and Python block with `## Usage:`.

| What | Result | How |
|------|--------|-----|
| Files with doc blocks | 75 `src/`, 85 `api_mex/`, 80 `api_python/`, all `.cpp`, none in subfolders; one `/*!SECTION` per file; no `/*!MD` without `/*!SECTION`; every delimiter alone on its line | `survey.py`: regular expressions over all files of the source folders |
| Blocks and sections | `/*!MD`: 137 C++, 85 MATLAB, 80 Python; sections 10, 9, 9; `SECTION_DESC`: 2 per language | `survey2.py` |
| Sub-block headings | 26 distinct C++ labels, 19 MATLAB, 19 Python (lists in section 5.3) | `survey2.py` |
| Raw HTML outside code | only `<br>` and `<a>`: 260 and 192 tags (opening and closing counted) in the API blocks and the old prose sources together | `survey3.py` |
| Parameter items | 2616 of 2641 match the pattern of section 5.4; separators 2593 em dash, 17 hyphen, 6 `<br>`; qualifiers 567 `*(optional)*`, 22 `(optional)`, 3 `*(in/out)*`, 2 `*(optional, in/out)*`, 4 `*(single-frequency only)*`, 1 `(input)` | `params.py`, `survey4.py` |
| `[[...]]` links | 637 C++ (all resolve on the page), 391 MATLAB (5 unresolved), 358 Python (6 unresolved); expected after the fixes: 0 unresolved | `links.py` |
| Required sub-blocks | C++ blocks without `Declaration(s)`/`Usage`: `arrayant`, `channel`, `path` (class blocks); MATLAB all have `Usage`; Python: one `Declaration` instead (`qpy_write_png.cpp`, on the fix list) | inline script |
| Duplicate function names | none per language, also none case-insensitively | inline script |
| Old hash anchors | `shake_256(title).hexdigest(4)`; matches the ids in `html_docu/*.html` (for example `6cab4884`) | `html_builder/extract_html.py`, `grep` |
| Old ids per page | `cpp_api.html` 147, `mex_api.html` 94, `python_api.html` 89, `formats.html` 10, `faq.html` 4, `download.html` 4, `index.html` 1, `publications.html` 0 (without `quadriga-lib_*` layout ids) | `grep -o ' id="..."'` |
| Cross-language families | 78 MATLAB/Python pairs; C++ by name 68 of 87 families; with the mapping file 85 of 87; 99 of 137 C++ functions with a pill; every mapped and name-matched C++ function is called by the family's wrappers (one exception: `.copy_element`, called by the MATLAB wrapper only) | `match.py`, `match2.py`, wrapper-call scan (inline script) |
| Span-free colouring | prototype C++ page 1,042,598 → 694,128 bytes; no colouring 647,202; gzip 146,909 → 135,571 | `spanfree.py` with Pygments 2.21.0 |
| Section sizes in today's full Markdown files | largest: C++ "Site-specific simulation tools" 128,481 bytes | split of `html_docu/quadriga_lib_api_*.md` at section headings |
| KaTeX | katex 0.18.11 under Node 22.22.1: MathML 581 bytes vs HTML 3,936 bytes for one sum formula; parse errors reported per item | `node/render.mjs` |
| mdformat | MDRenderer renders doc-block tokens back to Markdown, including tables in list items | `proto_md2.py`, `proto_md3.py` |
| Live server | no compression for `Accept-Encoding: gzip, br`; `.md` served without content type; HTTPS handshake fails; `releases/` lists files, `download/` gives 404 | `curl -sI` |
| `.gitignore` | current patterns ignore `test*.txt`, `release/`, `dist/` below new folders; the block of section 2.4 fixes it | `git check-ignore` in a scratch repository |
| Tools | Python 3.14.4, Node v22.22.1, npm 9.2.0, cmake 4.2.3, Inkscape and ImageMagick installed, Chrome at `/usr/bin/google-chrome`; conda at `/home/stephan/miniconda3`, conda-forge `nodejs` 22.23.4 to 26.12.1 available | `--version`, `which`, `conda search` |
| Minimum versions | markdown-it-py 3.0.0, mdit-py-plugins 0.4.2, mdformat 1.0.0, mdformat-gfm 1.0.0, Pygments 2.17.2, Jinja2 3.1.2 run the prototypes; Markdown output identical to the newest versions; katex 0.16.0 renders MathML | `venv_min/`, `node_min/` |
| Package versions | as in section 4 | PyPI JSON API, `npm view`, scratch installs |
