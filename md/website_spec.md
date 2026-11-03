# Quadriga-Lib website overhaul: specification

This is the specification (step 3 of `md/website_plan.md`). Implementation sub-agents (step 5) build the two new builders (D39), templates, static parts and tests from it alone, and carry out the cutover (D35). It refers to the plan by decision number (D1 to D39) and section, and to `md/website_design.md` as "design" with its section number. Step 7 deletes this document; it stays in the git history.

## 1. Scope and conventions

### 1.1 What this spec covers

The two builders of D39: `docs_builder/` (the tracked Markdown in `docs/`, standard library only, part of every release build) and `website_builder/` (the site in `website/`, maintainer's Linux machine only); the page templates, the static parts, the tutorial conversion with one dummy notebook in `tutorials/`, the agent skill placeholder, the builders' tests, the new `GNUmakefile` lines, the `.gitignore` and `.dockerignore` additions, the cutover work items (D35: moves, deletions, CMake, packaging, READMEs, SPEC files) and the work packages for step 5. It does not cover the landing-figure scripts and images (D15), the skill content (D10), real tutorials (D11), running and refreshing notebooks before a release (D34) or the deploy script (D29).

The code is meant to stay lean: fewer modules, fewer settings, conventions instead of configuration. Where this spec leaves an internal detail open (function names, data structures inside a module), the implementer chooses the simplest form.

### 1.2 Precedence and references

Where texts differ: D32 to D39, then D24 to D31, then the rest of the plan, then the design, then the prototype in `html_builder/design_prototype/`. Paths in the design and prototype map as follows:

| Design / prototype | This spec |
|--------------------|-----------|
| `html_docu/` (output) | `website/` (built site) and `docs/` (tracked Markdown) |
| `html_builder/` (builder, page sources) | `docs_builder/` (Markdown) and `website_builder/` (site) |
| `md/cpp/<section>.md` | `docs/api/cpp/<section-slug>.md` (and `mex`, `python`) |
| `index.md` and other twins on every page | twins only as in D26 |
| prototype `css/site.css`, `js/site.js`, `fonts/`, `img/logo.svg` | `website_builder/static/css/site.css`, `static/js/site.js`, `static/fonts/`, `templates/_logo.svg` |

### 1.3 Terms

| Term | Meaning |
|------|---------|
| Doc block | A `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` comment block in `src/`, `api_mex/` or `api_python/` (D6). |
| Language, `lang` | One of `cpp`, `mex`, `python`; display names "C++", "MATLAB / Octave", "Python". |
| Function | One `/*!MD` block. Its anchor is its name exactly as on today's pages (section 9). |
| Family | A MATLAB/Python function pair matched by file stem, plus its C++ counterpart(s) (section 9.4). |
| Fragment | A list of markdown-it-py block tokens made by the website builder from a part's source lines after its own rules ran (section 5.6); used for HTML only. |
| Twin | The Markdown version of an HTML page (D26), written from the source text (section 8.6). |
| Gap | A content placeholder written with the gap marker (section 6.4), a missing landing figure or a missing preview image (section 12). |
| Build variable | A value filled in at build time, written `{{ name }}` (section 6.3). |
| Site root | `website/`; URLs in this spec are relative to it unless absolute. |
| Repository root | The folder holding `CMakeLists.txt`; default is the parent of the folder of the `build.py` that runs. |
| Docs path | Everything `docs_builder/build.py docs` runs: standard library only, line-based (D39). |
| Structural error | A defect that stops the build with file and line (D7). |
| Warning | A defect that is printed and does not stop the build (D7). |

## 2. File tree

### 2.1 New folders and files

"Tracked" means committed by the maintainer (D19); "ignored" means listed in `.gitignore` (section 2.4); "generated" means written by a builder; "hand-made" means written in step 5 or by the maintainer.

| Path | Kind | Purpose |
|------|------|---------|
| `docs/` | tracked, generated | Markdown tree (D8, D24, D33). Fully owned by `docs_builder`: files it does not write are deleted (section 3.6). |
| `website/` | ignored, generated | Built site and deploy root (D33, D29). Fully owned by `website_builder`. |
| `docs_builder/build.py` | tracked | Entry point of the docs builder (section 3). |
| `docs_builder/qdocs/` | tracked | Docs builder package (section 2.2). |
| `docs_builder/content/` | tracked | Every source of `docs/` besides the doc blocks: `download.md`, `formats.md`, `faq.md`, `ai_agents.md`, `api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md`, `publications.toml`, `api_matching.toml` (sections 6, 7.2, 9.4). |
| `docs_builder/extract_matlab_comments.py` | tracked | MATLAB help-file script moved from `html_builder/` (section 18.1); called by CMake. |
| `docs_builder/README.md` | tracked | Written in step 7: Python requirement, make targets, content formats. |
| `website_builder/build.py` | tracked | Entry point of the website builder (section 3). |
| `website_builder/qsite/` | tracked | Website builder package (section 2.2). |
| `website_builder/content/` | tracked | Sources that feed only the website: `impressum.md`, `privacy.md`, `404.md`, `tutorials.md` (section 6.8). |
| `website_builder/templates/` | tracked | Jinja2 templates, including the landing text (sections 7.1, 8). |
| `website_builder/static/` | tracked | Files copied unchanged into the site root, including `publications/` and `quadriga_lib_material_model.md` (section 12); read from the code folder (section 3.2). |
| `website_builder/logo/` | tracked | Logo sources moved from `html_builder/img/` (section 18.1); not deployed. |
| `website_builder/skill/quadriga-lib/SKILL.md` | tracked | Skill placeholder (section 14); it feeds only the site. |
| `website_builder/pandoc_math.yaml` | tracked | pandoc defaults file for the formula step (section 11). |
| `website_builder/pytest.ini`, `website_builder/tests/` | tracked | Tests of both builders and the fixture repository `tests/fixtures/mini_repo/` (section 17). |
| `website_builder/README.md` | tracked | Section "Set up a build machine" in step 5 (section 4.2); the rest in step 7. |
| `tutorials/` | tracked | Notebooks committed with their outputs (D34); one dummy notebook `tutorials/pipeline_demo.ipynb` (section 13.5). |

`website_builder/` is left out of the source package and of the Docker context (sections 16, 18.1). The tests live there because the fixture repository holds copies of `CMakeLists.txt`, `include/quadriga_lib.hpp` and source files that do not belong in a source package, and because both builders share that fixture; `docs_builder/` then holds only what release builds run.

Nothing is installed into the working copy: the website tools come from a conda environment or from Ubuntu packages, outside the repository (section 4, D38).

Naming rule: no file in the new folders may be named `build*.log`, `test*.txt` or `deploy.py`, or sit in a folder named `release`, `dist` or `build`. These names match unanchored patterns of the existing `.gitignore` at any depth and would be silently ignored.

### 2.2 Builder packages

A guide, not a contract; the implementer may merge further where it keeps the code simpler. Imports go one way only: `qsite` imports `qdocs`, never the reverse (plan 5.1: one parser). Every `qdocs` module uses only the standard library; markdown-it, Pygments, Jinja2 and pandoc are used only in `qsite`.

`docs_builder/qdocs/`:

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Argument parsing, the `docs` command, diagnostics (section 3.7), output writing and check mode (section 3.6). |
| `config.py` | Site constants and page list (section 3.3), library and Armadillo versions (section 3.8), the TOML import (section 3.4). |
| `docblocks.py` | Block extraction, block grammar, parameter lists, content model (section 5). |
| `links.py` | Link targets and URLs per output location, `[[...]]` resolution, `slug`, ids, legacy hashes, cross-language matching (sections 8.7, 9). |
| `prose.py` | Prose pages, front matter, heading ids, build variables, publications data, the gap scan (sections 6, 7.2). |
| `render_md.py` | Markdown output: the source-text twin writer and its link scanner, API files, `llms.txt`, `docs/README.md` (sections 8.6 to 8.8). |
| `checks.py` | Markdown link and anchor check (section 15.1). |

`website_builder/qsite/`:

| Module | Responsibility |
|--------|----------------|
| `cli.py` | The `website` command: runs the `docs` phase through `qdocs`, then renders, checks and writes the site (sections 3.1, 3.6). |
| `mdparse.py` | markdown-it set-up and the builder's rules (section 5.6). |
| `render_html.py` | HTML output: `render_site(model) -> dict[str, bytes]`, called by `cli.py`, which then calls `checks.check_site(mapping)`; Jinja2, fragment rendering, span-free code colouring, the pandoc formula step, the HTML gap pass (sections 8.1 to 8.5, 10, 11). |
| `checks.py` | Link and anchor check of the whole site, well-formedness, size budgets (section 15). |
| `tutorials.py` | Notebook reading and conversion, tutorial pages and twins (section 13). |
| `assets.py` | Static files, landing figures, skill copy; the constants `STATIC_DIR`, `TEMPLATES_DIR`, `LANDING_FIGURES` (sections 3.3, 12, 14). |

### 2.3 Output files

`<lang>` is `cpp`, `mex` or `python`; `<slug>` is a section slug (section 9.2); `<stem>` is a notebook file stem.

Tracked Markdown in `docs/` (written by `make docs`; D33: nothing else lives in `docs/`):

| Path | Content |
|------|---------|
| `docs/README.md` | Index of `docs/` for GitHub (section 8.8). |
| `docs/llms.txt` | Agent entry point (section 8.8). |
| `docs/quadriga_lib_api_<lang>.md` | Full API per language (section 8.6.2). URLs of D3. |
| `docs/api/<lang>/general-usage-notes.md` | The API intro per language. |
| `docs/api/<lang>/<slug>.md` | One file per API section (section 8.6.3); 10 C++, 9 MATLAB, 9 Python sections today. |
| `docs/download.md`, `docs/formats.md`, `docs/faq.md`, `docs/ai_agents.md`, `docs/publications.md` | Twins of the content pages (D26). |

Site in `website/` (written by `make website`):

| Path | Content |
|------|---------|
| `index.html` | Landing (no twin). |
| `download.html`, `formats.html`, `faq.html`, `ai_agents.html` | Text pages; `ai_agents.html` is the new AI page (D9). |
| `cpp_api.html`, `mex_api.html`, `python_api.html` | API pages. |
| `publications.html` | Publications. |
| `tutorials.html`, `tutorials.md` | Tutorial index and its twin (website only, D28). |
| `tutorials/<stem>.html`, `tutorials/<stem>.md`, `tutorials/<stem>.ipynb`, `tutorials/<stem>/<figure files>` | One tutorial page, its twin, the notebook for download, its figures. |
| `impressum.html`, `privacy.html`, `404.html` | Legal pages (D17) and not-found page; no twins. |
| every file of `docs/` except `README.md` | Copied byte for byte at the same relative path (D24). |
| `skill/SKILL.md` | Skill placeholder (section 14, D37). |
| `css/`, `js/`, `fonts/`, `img/`, `favicon.ico`, `.htaccess`, `publications/**`, `quadriga_lib_material_model.md` | Static parts (section 12). |

The website builder writes nothing below `download/` or `releases/` of the site root (plan section 11): none of its outputs has such a path. These folders exist only on the server, never in the local `website/`, so the deletion rule of section 3.6 cannot touch them.

### 2.4 `.gitignore` and `.dockerignore` additions

Append to the root `.gitignore`:

```
/website/
.ipynb_checkpoints/
```

Append `website/` to `.dockerignore`, so the built site never enters the Docker context of the deb build (D33); `website_builder/` leaves it through the line of section 18.1. `docs_builder/` and `docs/` stay in the context: the deb build runs the docs check (section 16). The existing `__pycache__/` line of `.gitignore` already covers byte-code. The naming rule of section 2.1 keeps the new files clear of the existing unanchored patterns.

## 3. Builder entry points, CLI and make targets

### 3.1 Command line

Two entry points, run from any folder:

| Command | Does |
|---------|------|
| `python3 docs_builder/build.py docs` | Parses all sources, renders the tracked Markdown tree and writes `docs/` (section 3.6). |
| `python3 docs_builder/build.py docs --check` | Same rendering in memory; writes nothing; compares with `docs/` (D23); prints each differing, missing or extra file. |
| `python3 website_builder/build.py website` | Runs the `docs` phase through `qdocs` (builds `docs/` in memory, section 3.6), then renders the site, runs the checks of section 15 and writes `website/`. |
| `python3 website_builder/build.py website --release` | The one strict switch (plan section 11): additionally fails on any remaining gap; a failure writes nothing. |

Option for both: `--root DIR` (repository root; used by tests). Exit codes: 0 success (warnings allowed); 1 build failed (structural errors, failed checks, `--check` differences, strict-mode failures); 2 environment or usage problem (wrong Python, missing tool, bad arguments).

### 3.2 Code layout

Each `build.py` uses only the standard library until its check of section 3.4 has passed: it checks first (before anything imports `tomllib`, `tomli` or a third-party package), then puts the code folders on `sys.path` (its own; for `website_builder/build.py` also `../docs_builder`, the sibling folder), imports `qdocs.cli` or `qsite.cli` and calls `main()`. Code, `templates/`, `static/` and `pandoc_math.yaml` are always taken from the builder folders next to the `build.py` that runs (absolute paths, independent of the current folder): `static/` is part of the site's code like the templates, so the test fixture needs no copy of it. Everything else is taken relative to `--root`: sources, `docs_builder/content/`, `website_builder/content/`, `website_builder/skill/`, `tutorials/` and the outputs.

### 3.3 Site constants and page conventions (`qdocs/config.py`)

Constants, not a settings file:

| Constant | Value |
|----------|-------|
| `SITE_BASE_URL` | `http://quadriga-lib.org` (no trailing slash; http until HTTPS works, plan section 10) |
| `DOWNLOAD_BASE_URL` | `http://quadriga-lib.org/download` (D25) |
| `RELEASES_URL` | `http://quadriga-lib.org/releases/` (D25) |
| `GITHUB_URL` | `https://github.com/stjaeckel/quadriga-lib` |
| `PAGES` | Tuple of `(id, tab, twin)` in tab order; `file` is `<id>.html`; `tab` is the label in the tab row and menu (`None`: not in them); `twin` is the twin path (`None`: no twin, D26). It lists the website-only pages too, since twins link to them (section 8.7.1). |

The website constants are in `qsite/assets.py`: `STATIC_DIR` and `TEMPLATES_DIR` (`static/` and `templates/` in `website_builder/`, section 3.2; module constants, so tests may monkeypatch them to a copy below `tmp_path`, for `og:image` and the landing size warning) and `LANDING_FIGURES` (`("coverage", "antenna")`: figure names; the files are `img/landing/<name>-light.webp` and `-dark.webp`, section 12.4).

Initial `PAGES`: `index` (Home), `download` (Get started, `download.md`), `cpp_api` (C++ API, `quadriga_lib_api_cpp.md`), `mex_api` (MATLAB API, `quadriga_lib_api_mex.md`), `python_api` (Python API, `quadriga_lib_api_python.md`), `tutorials` (Tutorials, `tutorials.md`), `formats` (Formats, `formats.md`), `publications` (Publications, `publications.md`), `ai_agents` (AI agents, `ai_agents.md`), `faq` (FAQ, `faq.md`), `impressum`, `privacy`, `404`, the last three without tab and twin.

Template and source per page id (`D/` is `docs_builder/content/`, `W/` is `website_builder/content/`):

| Page id | Template | Source |
|---------|----------|--------|
| `index` | `landing.html` | the template itself (section 7.1) |
| `cpp_api`, `mex_api`, `python_api` | `api.html` | doc blocks of the language, intro `D/api_intro_<lang>.md` |
| `publications` | `publications.html` | `D/publications.toml` |
| `tutorials` | `tutorials_index.html` | `W/tutorials.md` and the notebooks |
| `impressum`, `privacy` | `legal.html` | `W/<id>.md` |
| `404` | `notfound.html` | `W/404.md` |
| any other id | `text.html` | `D/<id>.md` |

A missing source file of this table is a structural error: `docs` checks the files in `D/`, `website` also those in `W/`.

Source paths are constants too (`src/`, `api_mex/`, `api_python/`, `include/quadriga_lib.hpp`, `CMakeLists.txt`). The tutorial twins are website only (D28): `tutorials.md` and the tutorial pages never enter `docs/`. One tutorial page per notebook is added by the website builder (section 13). The header menu, tab row and footer are written in the templates with a helper `url(page_id)` (section 8.2).

### 3.4 Checks before any work

`docs_builder/build.py` checks that Python is 3.10 or newer and that a TOML reader is importable: `tomllib` (standard library from 3.11), else `tomli` (on Ubuntu 22.04 the system package `python3-tomli`, version 1.2.2 there). `qdocs/config.py` does the one import, `try: import tomllib` / `except ImportError: import tomli as tomllib`; the code uses only `tomllib.loads(text)` and `tomllib.TOMLDecodeError`, which tomli 1.2 has too (checked in the jammy build image). Nothing else is checked: the docs path needs no other tool.

`website_builder/build.py` checks the same, and also that `markdown_it`, `mdit_py_plugins`, `pygments` and `jinja2` are importable and that `pandoc` is on `PATH` (`shutil.which`). There are no lazy imports and no version comparisons.

Each collects everything missing, prints it in one run, one line per item, and exits 2 without a traceback (D21). A missing TOML reader reads: use Python 3.11 or newer, or on 3.10 install `python3-tomli` (Ubuntu) or `pip install tomli`. `build.py` uses no syntax newer than Python 3.6 and imports `qdocs` or `qsite` only after its checks, so an older Python also gets the plain message. The message names the Python in use (`sys.executable`); the website message also says to activate the conda environment `quadriga-web`, gives both set-up lines of section 4.2 and points to "Set up a build machine" in `website_builder/README.md`. No message starts an installation; a build never downloads anything. There is no `env` command and no `make docs-env` (D38).

### 3.5 `GNUmakefile` lines

`GNUmakefile` is a lean wrapper around CMake; the builders add only short callers:

```make
# Documentation: docs_builder/README.md and website_builder/README.md
docs:
	python3 docs_builder/build.py docs
docs-check:
	python3 docs_builder/build.py docs --check
website:
	@test -f website_builder/build.py || { echo "make website: website_builder/ is not part of the source package; build the website from a git clone"; exit 1; }
	python3 website_builder/build.py website
.PHONY: docs docs-check website
```

And two lines at the end of the existing `tidy` recipe (section 3.10). Everything else is called directly (`website_builder/build.py website --release`, the tests of section 17). `python3` is the one from `PATH`: for `docs` and `docs-check` any Python 3.10 or newer (no conda environment needed), for `website` the one of the active environment. `make website` runs the `docs` phase itself in one process, which is how it "depends on make docs" (D21). The old `documentation` target goes in package D (section 18.3).

### 3.6 Output handling

- Every command builds the complete output as an in-memory mapping from relative path to bytes. Iteration over files, sections, functions and pages is always sorted.
- No file is written while any structural error exists; errors from all inputs, including the formula step of section 11, are collected first, and only then are `docs/` and `website/` written.
- `docs`: writes each file whose content differs from the file on disk, then deletes every file under `docs/` not in the mapping and removes empty folders. `--check` writes and deletes nothing.
- `website` writes `docs/` and `website/` only after all errors and checks have passed. It renders the site into memory; runs the checks of section 15 on the in-memory mapping; only if they pass, writes changed files into `website/`, deletes files not in the mapping and removes empty folders.
- All text files are UTF-8 without BOM with `\n` line ends and one final newline.

### 3.7 Diagnostics

One per line: `<path>:<line>: error: <message>` or `<path>:<line>: warning: <message> [<kind>]` (`:<line>` omitted when there is none). Every warning is printed, each with a short fixed kind id (for example `param-separator`, `param-item`, `fence-language`, `output-type`), so the later doc-block touch-up (plan section 7) can grep for them. `docs` prints the gaps of its own sources, `website` the full gap list (section 6.4). Paths are relative to the repository root and written with `/` on every system. Last line: `<command>: N errors, M warnings, K gaps`.

### 3.8 Versions read once

`config.py` reads once per run the library version from `include/quadriga_lib.hpp` (`#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"`, the expression the old builder and `pyproject.toml` use) and the Armadillo version from `CMakeLists.txt` (`set\(armadillo_version\s+"([^"]+)"\)`). A missing file or no match is a structural error.

### 3.9 Byte-identical output

The same inputs give byte-identical `docs/` and `website/` trees (plan 5.1): sorted listings; no dates, user names or absolute paths in any output; tool versions are never written into outputs. The tracked Markdown is written from the source text by `qdocs` alone (section 8.6), so `docs` and `docs --check` (D23) give the same result on Linux, on Windows, in the deb containers and with any Python from 3.10 on; no tool version is involved. The HTML in `website/` may change with the pandoc or Pygments version.

Rules on the docs path that keep Linux and Windows identical: `pathlib` for paths; every file is read with `encoding="utf-8"` in text mode, so a checkout with `\r\n` line ends gives the same text, and `docs --check` and the "differs" test of section 3.6 compare the files in `docs/` read the same way; every file is written with `encoding="utf-8", newline="\n"`; relative paths in outputs, diagnostics and sort keys are `as_posix()` strings (never `Path` objects, which compare without case on Windows); no POSIX-only module (and no subprocess, section 3.11); `build.py` calls `reconfigure(errors="backslashreplace")` on `sys.stdout` and `sys.stderr`, so a non-ASCII character in a diagnostic cannot stop a run on a Windows console.

### 3.10 What the builders leave in the working copy, `clean` and `tidy`

The builders write `docs/` (tracked) and `website/`; running them or the tests leaves `__pycache__/` below `docs_builder/` and `website_builder/`; editing notebooks in Jupyter leaves `.ipynb_checkpoints/`. Tests write only below pytest's `tmp_path`.

`make tidy` restores the working copy to the state of a clean clone (D4, D33); `make clean` keeps `website/`. No `git clean`: it would also delete the maintainer's ignored local files, such as the deploy script. Step 5 appends to the existing `tidy` recipe:

```make
	- rm -rf website
	- find docs_builder website_builder tutorials \( -name __pycache__ -o -name .ipynb_checkpoints \) -type d -prune -exec rm -rf {} +
```

The conda environment is outside the repository and never touched.

### 3.11 Speed

Time targets on the idle build machine (1-minute load below 4): `make docs` under 5 s; `make website` under 15 s (with the `docs` phase and the checks, formulas included). One pandoc call converted 44 formulas in 0.05 s (manager's measurement, pandoc 3.13). Rules that keep the targets:

- One Python process per command; the sources are read and parsed once.
- No subprocess in `docs`; in `website` at most one pandoc call per build, and none when no page has a formula (section 11); no network access.
- Files are compared before they are written; the checks parse each output once.

## 4. Dependencies and tool set-up (D38, D39)

### 4.1 Tools and minimum versions

`docs_builder` needs Python 3.10 or newer and nothing else (on 3.10 also `tomli`, section 3.4); `make docs`, `make docs-check` and the release builds run with the system Python. The rest of this section is for `website_builder` and the tests, on the maintainer's Linux machine: the tools come from conda-forge (documented path, any Linux) or from Ubuntu packages; no pip, no virtual environment, no requirement files (D38). No Node.js (D5).

| Tool | Minimum | Why |
|------|---------|-----|
| Python | 3.10 | as `docs_builder`; the conda line installs 3.11 or newer, which needs no `tomli` |
| markdown-it-py | 3.0 | parser API used in section 5.6 |
| mdit-py-plugins | 0.4.2 | `dollarmath_plugin` with the options of section 5.6 |
| Pygments | none | lexers of section 10 |
| Jinja2 | 3.1 | environment options of section 8.1 |
| pandoc | 3.7 | formula step (section 11); tested with 3.7.0.2 (the version of Ubuntu 26.04), 3.9.0.2 and 3.13: the defaults file works and only invalid formulas print a warning; the MathML differs in small attributes between versions (section 3.9) |
| pytest | none | tests (section 17); also on Windows for the docs tests after the merge (section 18.4) |

Notebooks are read with the standard library (`json`); the website builder needs no Jupyter package (D34). Minimums are not checked by the build (section 3.4); they are the versions the set-up lines below install.

### 4.2 README section "Set up a build machine"

Written in step 5 as the first section of `website_builder/README.md`; step 7 adds the rest of that README. Once per machine, one of two variants:

1. Conda (any Linux; the documented path):

```bash
conda create -y -n quadriga-web -c conda-forge "python>=3.11" "markdown-it-py>=3.0" "mdit-py-plugins>=0.4.2" pygments "jinja2>=3.1" pytest "pandoc>=3.7"
conda activate quadriga-web
```

2. Ubuntu 26.04 or newer (older releases ship an older pandoc; use conda there):

```bash
sudo apt install pandoc python3-markdown-it python3-mdit-py-plugins python3-pygments python3-jinja2 python3-pytest
```

With conda, activate the environment before each build session. To check the set-up, run `make website`: if anything is missing it stops with the plain message of section 3.4. Upgrade with `conda update -n quadriga-web --all` or the system's package updates. Nothing is installed into the working copy.

## 5. Doc-block parser

### 5.1 Files read

For each language every `<folder>/*.cpp` directly in `src/`, `api_mex/` or `api_python/` (not recursive, as today), sorted, read as UTF-8. Today 75, 85 and 80 files contain blocks; no other file type does.

### 5.2 Block grammar and structural errors

All of this section runs on the docs path, line by line (D39). A block starts with a line that is exactly `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` and ends with the next line that is exactly `SECTION!*/`, `SECTION_DESC!*/` or `MD!*/` of the same kind. Structural errors (file and line):

| Id | Condition |
|----|-----------|
| E-delim-open | An opening delimiter without its closing delimiter before the next opening delimiter or the end of the file. |
| E-section-count | A file with `/*!MD` or `/*!SECTION_DESC` blocks and no `/*!SECTION` block, or with more than one, or a `/*!SECTION` block that is not exactly one non-empty line. (The old builder silently dropped such blocks.) |
| E-name | The first line of an `/*!MD` block does not match `^# (\.?[A-Za-z_][A-Za-z0-9_.]*)(<\+\+>)?$`. `<++>` is accepted and ignored (plan section 11). |
| E-fence | A code fence (a line whose stripped text starts with three backticks) without a closing fence in the same block part. |
| E-required | A required sub-block is missing (section 5.3). |
| E-link | An unresolved `[[...]]` link (section 8.7.2). |
| E-anchor | Two anchors equal on one page (functions, section ids, legacy ids, prose heading ids). |
| E-gap | A gap marker opened and not closed on the same line (section 6.4). |
| E-heading | A heading that markdown-it parses and that becomes `h2` or `h3` in a prose page or notebook cell, but that the line scan of section 6.2 does not see (a setext or indented heading); found by the website builder. |

Other malformed content (stray raw HTML, odd headings) is caught by the HTML well-formedness check (section 15.2).

### 5.3 Parts of an `/*!MD` block

Split at the source-line level, outside code fences: name (line 1), brief (line 2), intro (up to the first sub-block heading), sub-blocks. Each part keeps its source lines, which the twin writer copies (section 8.6). A sub-block heading is a line starting with `## `; its label is the heading text with a trailing `<br>`, then a trailing `:` or `;`, removed (variants today: `## Inputs:`, `## Attributes:<br>`, `## Function 'duplicate';`).

| Kind | Labels | HTML rendering |
|------|--------|----------------|
| `code` | `Declaration`, `Declarations`, `Usage`, `Example`, `Examples` | Fragment; fences without info string use the page language. |
| `params` | `Input`, `Inputs`, `Output`, `Outputs`, `Attributes`, `Arguments`, `Input Arguments`, `Output Argument`, `Output Arguments`, `Optional outputs`, and labels starting with `Inputs (`, `Outputs (` or `In/out` | Parameter lists (section 5.4). |
| `plain` | everything else | Fragment. |

Required sub-blocks (D7): C++ blocks need `Declaration`, `Declarations` or `Usage`, except class blocks (blocks with an `Attributes` sub-block: `arrayant`, `channel`, `path`); MATLAB and Python blocks need `Usage`. The current sources satisfy this rule (zero errors).

Display name: the name without `<++>`; on the MATLAB page lower-cased (as today, design open point 7).

### 5.4 Parameter lists

In a `params` sub-block, outside code fences, a top-level item starts at a line that begins with `- ` in column 0. It takes the following lines that are indented, blank, inside a fence, or unindented directly after a non-blank line (lazy continuation); it ends before the next top-level item line or before an unindented non-blank line that follows a blank line. On today's sources this gives the same items, start and end lines, as markdown-it's list parsing in all 541 parameter sub-blocks (measured). Each item is tested on its first source line (`NAME` is `` `[^`]+`(?:, (?:\.\.\.|`[^`]+`))* ``):

```
^- (?P<names>\*\*NAME\*\*(?:(?:, | / )\*\*NAME\*\*)*)(?:\s+(?P<qual>\*\([^)]*\)\*|\([^)]*\)))?(?P<sep>\s+[—–-]\s+|<br>\s*$|\s*$)
```

- A matching item is a parameter row: names as written (for example `` `Dx`, `Dy`, `Dz`, `*D` `` or `` `V_re` / `V_im` ``), qualifier (text in the parentheses: `optional`, `in/out`, `optional, in/out`, ...; empty if absent) and description (the rest of the item, including continuation lines, nested lists and tables). The description's source lines, which the website builder parses into a fragment, are the first line without the matched `- **names** (qual) — ` prefix and the following lines dedented by the item's content indent (2).
- ` — ` is the SPEC separator; ` - ` and ` – ` give the warning `param-separator`; `<br>` or nothing means the description starts on the next line.
- A non-matching item is a note row across both columns, with the warning `param-item`.
- Other blocks in a `params` sub-block (paragraphs, tables) render in place.

Expected on today's sources: about 17 `param-separator` and 25 `param-item` warnings; the pattern covers every variant of plan section 11 (`*(optional)*`, `(optional)`, `*(in/out)*`, several names). The structure is for the HTML page only; twins keep the item text as written.

### 5.5 Differences between the languages

| | C++ | MATLAB | Python |
|---|-----|--------|--------|
| Folder | `src/` | `api_mex/` | `api_python/` |
| Display names | as written; member functions start with `.` | lower case | as written; may contain `.` (`sos.acf_2d`) |
| Language of fences without info | `cpp` | `matlab` | `python` |
| Required sub-block | `Declaration`/`Declarations`/`Usage` (not for class blocks) | `Usage` | `Usage` |
| File stem for families | — | `<stem>.cpp` | `qpy_<stem>.cpp` |

### 5.6 markdown-it set-up and the builder's rules (`qsite/mdparse.py`)

markdown-it renders HTML only, in the website builder; twins and everything else on the docs path never pass through it (section 8.6). One factory for three kinds: `docblock`, `prose`, `notebook`. All: `MarkdownIt("commonmark").enable("table")` with `dollarmath_plugin(allow_space=False, allow_digits=False, double_inline=True)` (D5; `$5 and $6` stays text). `html` is on for doc blocks only; in prose and notebooks raw HTML is not interpreted and shows as text (D27: hand-written HTML is not used for content). The builder's rules:

| Rule | Kinds | Effect |
|------|-------|--------|
| `wikilink` | docblock, and the API intros (prose plus this rule, with the language of the page) | `[[name]]` outside code becomes a link to an API target; a `.` directly before `[[` makes it the member form `.name` (as in the old builder). |
| `raw_a` | docblock | `<a href="U" ...>text</a>` becomes a link to `U`; `target` and `rel` are dropped, so every link opens in the same tab. |
| `raw_br` | docblock | `<br>` is kept, except trailing `<br>` at the end of a paragraph or heading (the old pages used them as spacing). |
| `heading_attrs` | prose, notebook | A trailing `{...}` on a heading sets its id and legacy ids (section 6.2) and is removed from the text. |
| `alert` | prose, notebook | A blockquote whose first line is exactly `[!NOTE]` or `[!WARNING]` becomes a note box (section 6.5). |
| `site_url` | all | Links starting with `http://` or `https://` plus `quadriga-lib.org/` (also `www.`) become site-relative, so the absolute links in doc blocks (plan section 10) resolve locally and in twins. URLs that start with `DOWNLOAD_BASE_URL + "/"` or `RELEASES_URL` (which ends in `/`) stay absolute and count as external: the builder never writes those folders (section 2.3). |

These rules never touch code spans or fences. Token line maps plus the part's first source line give the line numbers.

### 5.7 Content model

The model is built by `qdocs` on the docs path. Leaf content is the part's source lines with source file and first line; the website builder parses them into a fragment (section 5.6) when it renders HTML. A function has language, display name (= anchor), source and line, brief, intro, sub-blocks (label, kind, content; `params` content as rows with names, qualifier, description, note flag) and family. A section has language, title, slug, legacy id, descriptions (several `SECTION_DESC` blocks in file-name order) and functions. An API page has the intro (a prose page) and its sections. Prose pages, publications and tutorials have their own small classes. One parser feeds both renderers; the sources are read once (plan 5.1).

Ordering: sections by title, functions by `(name.startswith("."), name)` on the display name (as today); the general usage notes come first.

## 6. Prose pages

Formats, FAQ, Get started, the AI page and the three API intros (D27), each one Markdown file in `docs_builder/content/`; the legal pages, the tutorial index text and the 404 sentence, which feed only the website, in `website_builder/content/` (section 2.1). Both kinds use the same format and the same `qdocs/prose.py` code.

### 6.1 File format

The file starts with TOML front matter between two lines that are exactly `+++`; the Markdown body follows. Keys: `title` (required; `h1`, `<title>` "`title` · Quadriga-Lib", `og:title`, twin heading), `description` (required; meta and `og:description`), `lead` (optional, inline Markdown; lead paragraph on the page and in the twin), `dummy_note` (only `tutorials.md`; section 13.4), `llms_summary` (only `ai_agents.md`; the summary paragraph of `llms.txt`, section 8.8). A missing required key, an unknown key or a value that is not a string is a structural error naming file and key. Body: CommonMark with tables, math, note boxes, heading ids, build variables and gap markers; headings from `##` to `####` (the `h1` comes from `title`).

### 6.2 Heading ids

Headings are found line by line outside fences: a line in column 0 that matches `^(#{1,6}) (.+)$`. Every heading that becomes an `h2` or `h3` gets an id (`##` and `###` in prose; `#` and `##` in notebook cells, section 13.3), computed once by `qdocs/prose.py` in the content model and used by both renderers: the slug (section 9.2) of the heading's plain text, unique within the page by appending `-2`, `-3`. Plain text: the heading text without a trailing `{...}`; `[[x]]` and `.[[x]]` become `x`; a link or image `[t](u)` or `![t](u)` becomes `t`; `*` and backticks are removed; a backslash before an ASCII punctuation character is removed; entities are decoded (`html.unescape`). On the 41 distinct headings of the old formats, FAQ and download pages, and on test headings with code spans, links, `[[x]]`, escapes and entities, this gives the same ids as markdown-it's plain text (measured). The website builder takes each heading's id from the model by source line; a heading token that becomes `h2` or `h3` without one is E-heading. An explicit id is written at the end of the heading: `### How do I cite Quadriga-Lib? {#cite}`. Legacy ids (old anchors, D3) are added with `legacy="..."`, repeatable: `## Material Properties {#material-properties legacy="9f6922f9"}`. In HTML the id goes on the heading, each legacy id on an empty `<span id="...">` as its first child. In twins the braces are removed and the same ids are written as explicit anchors (section 9.5).

### 6.3 Build variables

`{{ name }}` (regex `\{\{\s*([a-z_]+)\s*\}\}`) in prose bodies, front matter, `publications.toml` strings and the skill stub (not in doc blocks or notebooks), substituted in the source text before parsing, so HTML and twin get the same text, also inside code: `version`, `armadillo_version` (section 3.8), `site_base_url`, `download_base_url`, `releases_url`, `github_url` (section 3.3). An unknown name is a structural error, also inside code: a literal `{{` followed by a lowercase name in code must be avoided (substitution inside code is needed for the compile line of section 6.6). Templates get the same values as Jinja2 context variables. Build variables are values; gap markers are placeholders; the two syntaxes differ on purpose (plan section 11).

### 6.4 Gap markers

One marker: `%%GAP: text%%` on one line. `text` describes missing content (`%%GAP: performance numbers and hardware%%`) or is a draft awaiting approval (`%%GAP: Realistic radio channels ...%%`); removing the marker is the approval.

- It may appear only in `docs_builder/content/` and `website_builder/content/` (prose files, their front matter, `publications.toml`) and `website_builder/templates/` (the landing text, section 7.1); not in doc blocks or notebooks.
- HTML: one pass over each rendered page replaces markers in element text with `<span class="ph">text</span>`, and markers inside tags (attribute values such as `alt`, `content`) and inside `<title>` with plain `text`.
- Twins and every other file in `docs/` (API files, section files, `llms.txt`, `docs/README.md`) leave markers out: each `%%GAP: text%%` is removed with its text; a line or list item that held only a marker is dropped; a paragraph left empty disappears; a field (title, lead, summary) left empty is omitted. This is the one deliberate exception to D26's "same text" (section 8.6.1). The website-only tutorial twins follow the same rule for simplicity. The HTML keeps the markers as placeholders, and the gap list and `website --release` are unchanged.
- The gap list: all markers found by a plain text scan (`path:line: text`, sorted). `docs` scans `docs_builder/content/` and prints those gaps; `website` scans all three folders and adds missing landing figures (section 12.4) and a missing `img/og-card.png` (section 12.3), and prints the full list. The HTML pass counts the markers it replaces; `website --release` fails if the gap list is not empty or that count is not zero (this also catches a marker in a doc block or notebook).

### 6.5 Note boxes

GitHub alert syntax (it renders on GitHub as is):

```
> [!NOTE]
> **Optional title**
> Body Markdown.
```

`[!NOTE]` or `[!WARNING]`. HTML: `<div class="note-box">` or `<div class="note-box warn">` with `<b>title</b>` first; the title is the first body line if that line is entirely bold, otherwise "Note" or "Warning". Twins keep the blockquote as written.

### 6.6 One-time conversion of the old prose sources

Step 5 converts `html_builder/formats/*.md`, `faq/*.md` and the download sources `download.md.in`, `install_binaries.md`, `install_source.md`, `quick_start.md` (not the generated `download/download.md`) into `docs_builder/content/formats.md`, `faq.md` and `download.md`. The old files are deleted with `html_builder/` in package D (section 18.2).

- Each old `/*!SECTION` becomes an `##` heading with an explicit id and the old hash as legacy id: formats "QuaDRiGa Array Antenna Exchange Format (QDANT)" `6cab4884`, "Material Properties" `9f6922f9`; FAQ "About the Library" `e63d3e38`, "Bug Reports and Contact" `59d38498`, "Distribution License" `89373af1`, "Supporting the Project" `d8094d72`; download "Download" `7c9ba8cc`, "Installation from Binaries" `7fc9f392`, "Installation from Source" `904e3979`, "Quick Start" `fdc96a45`.
- `/*!SECTION_DESC` text becomes the section body; each `/*!MD` block becomes a `###` heading with its brief and body, its `## X:` sub-headings becoming `#### X`. The formats entry ids `<arrayant>` and `<layout>` are kept as legacy ids. The old ids with spaces (`1. Material Table`, `2. Built-in EM Materials`, `3. Principled BSDF`, `4. References`) and the ids `…` and `1` are dropped: they are not valid HTML ids or are meaningless, and nothing links to them (D3).
- FAQ and download: each bold-labelled item (`- **How do I cite Quadriga-Lib?**<br>`) becomes a `###` heading with its text as body; the cite question gets the id `cite` (plan section 10). The quick-start labels (`**Python**`, `**MATLAB / Octave**`, `**C++**`) become `###` headings and their fences get `python`, `matlab`, `cpp`; the compile line becomes a `bash` fence with `{{ armadillo_version }}` instead of `15.6.2`.
- Fences with shell commands get the info string `bash`; other unlabelled fences stay uncoloured (section 10).
- Raw HTML becomes Markdown: links to `[T](U)` without `target` and `rel`; `<br>` removed; entities to characters (`<` in text escaped as `\<`; `[` as `\[`).
- `download.md.in`: `__VERSION__` becomes `{{ version }}`; `download/<file>` links become `{{ download_base_url }}/<file>`; `releases/` becomes `{{ releases_url }}`.
- Section order: formats as in the prototype (QDANT first); FAQ and download in the order of the old pages, which sorted sections by title (`html_builder/extract_html.py`).
- Front matter: titles "Data formats", "Questions", "Get started"; descriptions and leads from the prototype where it has them, gap-marked (design section 9).

### 6.7 API intro files

`docs_builder/content/api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md`: `title` "C++ API", "MATLAB / Octave API", "Python API" (design open point 7); `description` and `lead` from the prototype's C++ page adapted per language, with `{{ version }}`, gap-marked. Body: the "General usage notes" list of `get_preamble()` in `html_builder/extract_md.py`, with the example `*_3dB` added after `*_deg` in the C++ units line (as in `md/SPEC_CPP_API.md`) and `[arrayant_generate](#arrayant_generate)` written as `[[arrayant_generate]]` (intros are parsed as prose plus the `wikilink` rule of their language, section 5.6). The Python body is one gap marker: `%%GAP: General usage notes for the Python API (import, numpy conventions, index base, errors)%%` (plan section 11). The notes become the first section of each API page (`sec-general-usage-notes`) and `docs/api/<lang>/general-usage-notes.md`.

### 6.8 Other content files written in step 5

| File | Initial content |
|------|-----------------|
| `docs_builder/content/ai_agents.md` | The prototype's draft text, with the file table adapted to section 2.3 paths, the skill link `skill/SKILL.md`, and a `> [!WARNING]` box titled **Placeholder** saying the skill is a placeholder (plan 5.4, D37); first line `%%GAP: whole page text is a draft (design section 9)%%`; `llms_summary` a gap-marked draft from the landing lead. |
| `website_builder/content/impressum.md` | A `> [!WARNING]` box titled **Placeholder** with `%%GAP: Impressum text from the maintainer (D17); no address in git until settled%%`. |
| `website_builder/content/privacy.md` | A `> [!WARNING]` box titled **Placeholder** with `%%GAP: privacy text: the hoster and its server logs; no cookies, no analytics, no third-party requests; the browser stores the chosen code language under the localStorage key qlib-lang, only after a click (D18)%%`. |
| `website_builder/content/tutorials.md` | `lead` and `dummy_note` from the prototype, gap-marked. |
| `website_builder/content/404.md` | `title` and `description` "Page not found"; the prototype sentence, gap-marked. |

## 7. Landing and publications

### 7.1 Landing text in `website_builder/templates/landing.html` (D36)

The landing text is written in the template as HTML, ported from the prototype `index.html`: every design-section-9 gap and every `<span class="ph">` becomes a gap marker in the template text. Section ids are fixed: `why`, `paths`, `quickstart` (D3 keeps `index.html#quickstart`), `apis`, `covers`, `open`. The template sets title and meta description itself. The "For team leads" card keeps `href="#covers"` (D30); acoustics carries the "experimental" badge (D12). The quick-start code is written raw (unescaped, without spans) in the template inside `{% filter highlight("python") %}...{% endfilter %}` (and `matlab`, `cpp`, `bash`), the colouring of section 10. Figures: a macro `figure(name, alt, caption)` renders the slot of section 12.4; the builder passes `figures`, a dict from each name of `LANDING_FIGURES` to whether both variants exist. Figures `coverage` (pitch) and `antenna` (quick start) keep the prototype's alt texts and captions.

### 7.2 `docs_builder/content/publications.toml`

`page.title`, `page.description`, `page.lead` (inline Markdown, links `faq.html#cite`); `group` (at least 1, in display order) with `id` (the `h2` id and the twin anchor), `title`, optional `intro` and `outro` (inline Markdown) and `entry` (at least 1, in display order: newest first, as written) with `year` (int), `title`, `authors`, `venue` and `link` (at least 1 of `{label, url}`; `url` absolute or site-relative such as `publications/x.pdf`). A plain check in `qdocs/prose.py` reports a missing required key, an unknown key or an empty list as a structural error with file and key path (for example `group[1].entry[2].year: missing`). On the page, inline-Markdown strings are rendered with `renderInline` of the prose parser (website builder); other strings are escaped. Step 5 fills it from `html_builder/parts/publications.html.part` in the grouping of the prototype (groups `quadriga-lib` and `quadriga-background`); only `page.description` is a gap. The twin `docs/publications.md` is written from the same data: per group `## title` with intro, one list item per entry (year, title, authors, venue, links), outro.

## 8. Renderers and templates

Sections 8.1 to 8.5 are the website builder (`qsite`); sections 8.6 to 8.8 are the docs path (`qdocs`), which the website builder also calls for the tutorial twins.

### 8.1 Jinja2 environment

`Environment(loader=FileSystemLoader(templates), autoescape=True, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, undefined=StrictUndefined)`. Fragments are passed as `Markup`. Templates emit compact markup for repeated structures (index lists, parameter lists, pills), as in the prototype. Every internal URL is `{{ root }}<path>`: `root` is `""` for pages in the site root, `"../"` for `tutorials/<stem>.html`, `"/"` for `404.html` (root-absolute, plan section 11).

### 8.2 Templates

| File | Used by | Structure (prototype file) |
|------|---------|----------------------------|
| `base.html` | all | head (section 8.3), skip link, header, `<main id="content">`, footer (`index.html`) |
| `_header.html` | base | menu `<details class="menu">` with the groups of design 4.1, inline logo, name, version, GitHub link; tab row from `PAGES` |
| `_footer.html` | base | the four columns of design 4.2, bottom line "Quadriga-Lib v`version`", Impressum, Privacy; "This page as Markdown" only on pages with a twin (D26) |
| `_logo.svg` | header, 404 | the prototype logo (`fill="currentColor"`) inline, `aria-hidden="true" focusable="false"` |
| `landing.html` | landing | design 5.1 with the landing text (section 7.1; `index.html`) |
| `api.html`, `_api_index.html`, `_fn_entry.html` | API pages | section 8.4 (`cpp_api.html`) |
| `text.html` | text pages | breadcrumb, `h1`, lead, "Markdown version" link, index, prose (`formats.html`, `ai_agents.html`) |
| `legal.html` | Impressum, privacy | breadcrumb, `h1`, prose; no index (`impressum.html`) |
| `tutorials_index.html`, `tutorial.html` | tutorials | section 13.4 (`tutorials.html`, `tutorial_sample.html`) |
| `publications.html` | publications | groups with entry lists (`publications.html`) |
| `notfound.html` | 404 | centred logo 96 px, `h1`, sentence, buttons Home, C++ API, MATLAB API, Python API (design 5.8) |

Menu groups, footer columns and their labels are written in the templates (design 4.1 and 4.2) with the helper `url(page_id)`, which returns `root` plus the page's file; external links use the URL constants. Breadcrumbs: "Home › `title`"; "Home › Tutorials › `title`" on tutorial pages; "Home › API reference › C++" (and "MATLAB / Octave", "Python") with "API reference" linking to `index.html#apis`. The current tab and menu entry get `aria-current="page"`; tutorial pages mark "Tutorials". Text pages: the index lists the `h2` headings; an `h2` with `h3` headings below it becomes an open `<details>` with "Overview" and the `h3` links; the list appears in `nav.doc-side.idx` (`aria-label="Page sections"`) and in `details.doc-index` ("On this page"); a page with fewer than two `h2` headings has no index. Other templates follow their prototype files; the landing language tabs keep the prototype structure (`div.ltabs[data-lang-tabs]`, radios `name="qs"`, values `python`, `matlab`, `cpp`, Python checked). Inline `style` attributes of the prototype become classes in `site.css`.

### 8.3 Head of every page (design 4.3)

`<!doctype html>`, `<html lang="en" class="no-js">`, charset, viewport, `<title>`, description, `theme-color` `#2e7d46` (light) and `#24603a` (dark) written literally in `base.html`, `og:type`, `og:title`, `og:description`, `og:url` (absolute), and, only if `static/img/og-card.png` exists, `og:image` (absolute) and `twitter:card` `summary_large_image` (section 12.3); icons `img/favicon.svg`, `img/favicon-32.png`, `img/apple-touch-icon.png`; `<link rel="alternate" type="text/markdown">` only on pages with a twin (D26); preload of `fonts/ibm-plex-sans-latin-400-normal.woff2`; `css/site.css`; the inline `document.documentElement.className = "js"`; `js/site.js` with `defer`.

### 8.4 API page

Follows the prototype `cpp_api.html` and design 5.2:

- Page head: breadcrumb; `h1` and lead from the intro file; "Same API in" links to the other two pages; "Markdown version" to `quadriga_lib_api_<lang>.md`.
- Index (`nav.doc-side.idx`, `aria-label="<display> API index"`; copy in `details.doc-index` "Sections and functions"): the js-only filter input; "General usage notes"; per section a closed `<details>` with title and `<span class="count">`, then "Section overview" and one `<li class="fn">` per function; below, links to the other API pages and the full Markdown file.
- Section: `<section class="api-section" id="sec-<slug>"><h2 id="<legacy hash>">title</h2><div class="section-desc">...</div>` and its entries; the general usage notes as `sec-general-usage-notes` without legacy id.
- Entry: `<article class="fn-entry" id="<anchor>">`; `fn-head` with `h3` name and `#` anchor link; `fn-langs` with "Also in" and three pills (C++, MATLAB, Python): the current one `aria-current="page"`, a counterpart as link to `<lang>_api.html#<anchor>`, a missing one `<span class="pill none" title="No MATLAB counterpart">MATLAB –</span>`; then "Markdown" to `api/<lang>/<slug>.md`; `p.fn-brief`; intro; per sub-block `<h4>label</h4>` and content.
- Parameters: `<dl class="params">`, one `<div>` per row with `<dt>names<span class="opt">qualifier</span></dt><dd>description</dd>`; a note row is `<div class="pnote"><dt></dt><dd>...</dd></div>` (the empty `dt` keeps the `dl` valid; CSS lets the `dd` span both columns).

### 8.5 Fragment rendering to HTML

markdown-it's renderer with overrides: fences through the span-free colouring (section 10) as `<pre><code class="language-<lang>">`; tables wrapped in `<div class="table-wrap">`; link URLs from `links.py` for the output location; math converted to MathML (section 11), `math_block` in `<div class="math-block">`; note boxes (section 6.5); heading ids (section 6.2). Gap spans come from the pass of section 6.4.

### 8.6 Markdown output

#### 8.6.1 Twin writer and twin layout

Twins are written from the source text, not from markdown-it tokens. Each part (name, brief, intro, sub-blocks, prose body, notebook Markdown cell) is copied line by line, and only these things change:

- Doc-block sub-block headings `## X:` are demoted to the twin's sub-block level (`### X:`, text otherwise unchanged).
- Prose and notebook headings lose their trailing `{...}`; the ids are written as anchors (section 9.5). Notebook headings move down one level (`#` → `##`, `##` → `###`, deeper levels → `####`), as in HTML (section 13.3).
- Fences without info string in doc blocks get the page language (section 5.5).
- Outside code spans and fences: `[[name]]` and `.[[name]]` become Markdown links; raw `<a href="U" ...>T</a>` becomes `[T](U')`; a trailing `<br>` at the end of a line becomes a Markdown hard break (`\` at the line end) when the next line continues the paragraph, and is removed at the end of a paragraph; the URLs of Markdown links and images and of raw links are mapped for the twin's location (section 8.7.1, including the `site_url` rule).

The writer keeps a fence map per part (which lines lie inside a fence) and splits the other lines at backtick runs, carrying an open code span across the lines of a paragraph until its closing run (a blank line ends the paragraph), so code spans stay untouched. A backtick run left open at the end of a paragraph is literal text (CommonMark); the writer then gives the warning `code-span` and leaves the rest of that paragraph unrewritten. Gap markers are removed (section 6.4): the one deliberate exception to D26's "same text"; it is part of the same rewrite, so `docs --check` compares the output without markers. Math, tables and note boxes pass through verbatim; parameter items keep their source form. Links and images are found by one scanner, which takes the inline forms `[t](u)` and `![t](u)` (the sources use no reference-style links) and is also used by the Markdown link check (section 15.1). Structure around the parts (headings, anchors, index tables, separators) is written directly. Heading ids come from the content model (section 6.2), matched by source line.

Every twin starts with `# title`, an empty line, and `Quadriga-Lib <version> · Web page: <absolute URL>` (for a section file `<base_url>/<lang>_api.html#sec-<slug>`); then, if the page has at least two `##` headings, an index table (D26) of its `##` headings (with a `Topics` column of `###` links on prose pages); then the lead and the body. Exception: the full API files.

#### 8.6.2 Full API files `docs/quadriga_lib_api_<lang>.md`

The old layout, without build date and without the "Line" column (plan section 11): YAML front matter with `title` ("C++ API Documentation for Quadriga-Lib v`version`" and the MATLAB / Octave and Python forms), `author`, `lang`; `# General usage notes`; `# Function Index` with the columns Function, Section, Summary (the brief); then per section `---`, anchors, `# Section title`, description, and per function `---`, its anchor, `## name`, brief, a counterpart line that names both other languages, with the same information as the HTML pills: "Also in: [C++](…) · Not in: Python"; with no counterparts "Not in: C++, Python"; with both "Also in: [C++](…) · [Python](…)"; the line is never omitted; the labels are those of the pills, "C++", "MATLAB", "Python"; then intro and the sub-blocks (`### X:`).

#### 8.6.3 Per-section files `docs/api/<lang>/<slug>.md`

Twin header with the section title, index table `| Function | Summary |`, description, the functions as in the full file (with the same counterpart line). The largest (C++ "Site-specific simulation tools", about 128 kB) is accepted: a section is the unit an agent asks for (plan section 11).

### 8.7 Links

#### 8.7.1 Link targets

Every link becomes a target; each renderer asks for its URL at its output location. Twins carry the same ids as the HTML (section 9.5), so the twin URL only changes the path.

| Kind | Example in a source | HTML in root | HTML in `tutorials/` | Twin in `docs/` |
|------|---------------------|--------------|----------------------|-----------------|
| API function or section | `[[x]]` in a block; `cpp_api.html#x` in prose | `cpp_api.html#x` (same page: `#x`) | `../cpp_api.html#x` | full file: `#x`; section file: `#x` or `<slug>.md#x`; prose twin: `api/<lang>/<slug>.md#x` (the section file holding id `x`) |
| Page with twin | `faq.html#cite` | `faq.html#cite` | `../faq.html#cite` | twin path, same fragment (`faq.md#cite`) |
| Page without twin | `index.html#quickstart` | relative | relative | absolute: `<base_url>/index.html#quickstart` |
| Twin file | `quadriga_lib_api_cpp.md` | relative | relative | relative |
| Site file outside `docs/` | `publications/x.pdf`, `quadriga_lib_material_model.md`, `skill/SKILL.md` | relative | relative | absolute: `<base_url>/<path>` (plan section 11) |
| External | `https://...` | unchanged | unchanged | unchanged |
| Same-page anchor | `#files` | unchanged | unchanged | unchanged |
| Counterpart line "Also in" (twins only, section 8.6.2; "Not in" entries are plain text) | counterpart in language X | — | — | full file: `quadriga_lib_api_<X>.md#<anchor>`; section file: `../<X>/<slug>.md#<anchor>` |
| Page whose twin is website only (`tutorials.html`, `tutorials/<stem>.html`) | `tutorials.html` | relative | relative | absolute twin URL: `<base_url>/tutorials.md` |

Twins that exist only in `website/` (`tutorials.md` in the root, `tutorials/<stem>.md`) follow the column "Twin in `docs/`", with relative paths taken from their own folder (`../` in `tutorials/`); site files outside `docs/` stay relative there, since they are on the same site.

Links in sources are written relative to the site root. Doc blocks link only within their own API page (`[[name]]`; `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`); their raw links point to site pages such as `formats.html#6cab4884`, site files or external URLs. A wrong anchor in any site-relative link is caught by the link check (section 15.1).

#### 8.7.2 Resolving `[[name]]`

`[[name]]` in a block or intro of language L links to the function of L whose display name equals `name` (MATLAB: compared in lower case); `.[[name]]` links to the member function `.name`. Nothing else is tried; an unresolved link is E-link (D7). Zero unresolved links on the current tree (637 C++, 387 MATLAB, 354 Python).

### 8.8 `llms.txt` and `docs/README.md`

One function writes both, with absolute links (`base_url`) for `llms.txt`, which is fetched on its own, and relative links for `docs/README.md`, which is read on GitHub (D9, plan 5.4). Format of llmstxt.org, shown for `llms.txt` (`<base>` is `base_url`; `docs/README.md` has the same lines with relative links):

```
# Quadriga-Lib

> <llms_summary of docs_builder/content/ai_agents.md>

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
- [Agent skill (placeholder)](<base>/skill/SKILL.md)
```

`docs/README.md` is the same list after one intro sentence: generated by `make docs` from the documentation comments and `docs_builder/content/`, do not edit; quadriga-lib.org shows the latest release, these files this branch as of the last `make docs`. In `docs/README.md`, links to files outside `docs/` (tutorials, skill) are absolute. It is not copied to the site.

## 9. Anchors and URLs

### 9.1 Function anchors (D3)

The `id` of a function entry is its display name exactly, as on today's pages: `arrayant_interpolate_multi`, `.calc_directivity_dBi`, `arrayant_generate` (MATLAB, lower case), `sos.acf_2d`.

### 9.2 Readable ids and slugs

`slug(text)`: lower-case; every run of characters outside `[a-z0-9]` becomes one `-`; leading and trailing `-` removed. API section ids: `sec-` + slug (apart from function names); section file names: slug + `.md` ("Miscellaneous / Tools" gives `sec-miscellaneous-tools` and `miscellaneous-tools.md`). Prose heading ids: slug without prefix.

### 9.3 Old hash ids

The old builder made section ids with `hashlib.shake_256(title.encode("utf-8")).hexdigest(4)` (8 hex characters, confirmed against `html_docu/`). The new builder puts the same value on each API section's `h2`; prose pages carry theirs as explicit legacy ids (section 6.6). Renaming a section changes its legacy id, as before. Expected values: Array antenna class `3a0c780f`, Array antenna functions `e091e809`, Channel class `4f5f509b`, Channel functions `eab50272`, Channel generation functions `6f9553d7`, Channel statistics `c2d4d113`, Geometry-based stochastic model `85ec7f8f`, Math functions `40522dfe`, Miscellaneous / Tools `b11e2d16`, Site-specific simulation tools `386370db`, Sum-of-sinusoids random number generator `2d15ed9b`.

### 9.4 Language pills: matching rule (design open point 1)

1. Each MATLAB or Python file is a family; MATLAB and Python pair by file stem: `api_mex/<stem>.cpp` and `api_python/qpy_<stem>.cpp` (plan section 11).
2. A family's C++ counterpart by name: the C++ function whose name equals the MATLAB or Python name, compared in lower case with `.` replaced by `_` (`sos.acf_2d` matches `sos_acf_2d`).
3. `docs_builder/content/api_matching.toml` lists the C++ counterparts that names cannot find, and overrides rule 2 for its families. Its entries are the C++ functions the family's wrappers call to carry out their operation (not helper calls for validation, sizing or set-up). The first listed function is the target of the C++ pill on the MATLAB and Python pages (member function before `_multi`); the order in the file is the rule.

```toml
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

An unknown stem, an unknown C++ name or a C++ name in two families is a structural error. A pill for language X links to the family's function in X; no counterpart gives the dashed "X –" pill. Expected: 78 MATLAB/Python pairs; 85 of 87 families with a C++ counterpart (without: `components`, `version`); 99 of 137 C++ functions with at least one pill.

### 9.5 Anchors in twins

Twins write explicit anchors with the ids the HTML uses, so one id works in both formats and no GitHub heading slug is computed. On its own line directly before the element: `<a id="<anchor>"></a>` before each function heading; `<a id="sec-<slug>"></a><a id="<legacy hash>"></a>` before each API section heading; before each prose or notebook heading, one `<a id>` for its id and one per legacy id. Links in twins use these ids (section 8.7.1).

GitHub keeps such anchors with the prefix `user-content-` and its page script resolves `#x` to them. This must be checked once on GitHub after the first push (a function link with a `.` such as `#.calc_directivity_dBi`, a section link, `faq.md#cite`). If it fails, the fallback is a slug function that follows GitHub's heading anchors (github-slugger: lower case, punctuation dropped, spaces to `-`, `-1`, `-2` for repeats), used for links in twins only. The HTML ids stay as they are either way.

### 9.6 URL inventory (D3)

Kept: `index.html#quickstart`; `cpp_api.html`, `mex_api.html`, `python_api.html` with every function anchor (137, 85, 80 today) and section hash (10, 9, 9); `formats.html` with `#6cab4884`, `#9f6922f9`, `#<arrayant>`, `#<layout>`; `faq.html` and `download.html` with their four hashes each; `publications.html`; `/quadriga_lib_api_{cpp,mex,python}.md`; `/quadriga_lib_material_model.md`; `/publications/...`. Dropped (D3 as amended): the `quadriga-lib_*` layout ids of the old pages, the formats ids of section 6.6, `style2.css`, `quadriga-lib-logo80.png`, `quadriga-lib-q_letter64.png`. New: `ai_agents.html`, `tutorials.html`, `tutorials/...`, `impressum.html`, `privacy.html`, `404.html`, the twins, `api/...`, `llms.txt`, `skill/SKILL.md`.

## 10. Code colouring

`qsite/render_html.py` has its own small formatter over `pygments.lex()` (no `HtmlFormatter`). Each token type maps to the first matching entry; tokens without an entry produce plain escaped text and no `<span>` (design open point 4); adjacent tokens of one class share one `<span>`. The same function is the template filter `highlight(lang)` (section 7.1).

| Token type (and subtypes) | Class | Colour (site.css) |
|---------------------------|-------|-------------------|
| `Comment` | `c` | `--c-cm`, italic |
| `Keyword.Type`, `Name.Class`, `Name.Namespace` | `kt` | `--c-ty` |
| `Keyword` | `k` | `--c-kw` |
| `Name.Function`, `Name.Builtin` | `nf` | `--c-fn` |
| `String` | `s` | `--c-st` |
| `Number` | `m` | `--c-nu` |
| `Generic.Prompt` | `gp` | `--muted`, `user-select: none` |

Lexers: `cpp`, `c++` → C++; `matlab`, `octave` → MATLAB; `python`, `py` → Python; `bash`, `sh`, `shell` → Bash; `console` → Bash session; `text`, `none` and, in prose, a fence without info string → no colouring; another info string gives the warning `fence-language` and no colouring. Span-free colouring brings the prototype C++ page from about 1,043 kB to about 694 kB.

## 11. Formulas

- Syntax: `$...$` and `$$...$$` (D5) in doc blocks, prose, inline data strings and notebook Markdown cells. No site source has a formula today.
- Converter: pandoc, LaTeX to MathML (D38). `website` collects every formula of the build first (the `math_inline` and `math_block` tokens, and `math_inline_double` (`$$...$$` inside a paragraph, which `double_inline=True` produces) as a display formula, from all fragments, with source file and line), then runs pandoc once, and only if there is at least one formula: `pandoc -d website_builder/pandoc_math.yaml -f markdown -t html` (the defaults file by its absolute path, section 3.2) with the document on stdin and the HTML on stdout. `pandoc_math.yaml` holds `html-math-method:` / `  method: mathml`; the defaults file is used instead of a command-line flag because `--mathml` prints a deprecation warning from pandoc 3.11 on and `--math-method` does not exist before 3.11, while the defaults key works in 3.7 to 3.13 without a warning.
- A `math_block_label` token (`$$...$$ (label)`) is a structural error: equation labels are not supported.
- Batching: the document has, per formula in order, a marker paragraph `QDOCSMATH<n>` and then the formula as its own paragraph (`$tex$` or `$$tex$$`). The output is split at the `<p>QDOCSMATH<n></p>` lines. Marker paragraphs are plain words, which pandoc passes through unchanged, so the split cannot drift, and one process handles any number of formulas.
- Checks (D7): a formula whose chunk is not exactly one `<p>` holding a single `<math>` element is a structural error at its source file and line; this catches unknown commands (pandoc leaves raw TeX) and unbalanced braces (pandoc's reader does not take them as math). A non-zero exit status or any text on stderr (for example "Could not convert TeX math ...") is one more structural error that prints pandoc's stderr verbatim.
- Output: the `<math>` element as pandoc writes it (with its `<semantics>` and TeX annotation), inline in the text flow; a `math_block` formula inside `<div class="math-block">`; a `math_inline_double` formula is pandoc's `<math display="block">` without that wrapper, since it sits inside a `<p>`. Measured by the manager with pandoc 3.13: accents with subscripts, primes, `\pm` and `\sim` spacing, upright `\mathrm{d}`, `aligned`, `\frac12`, `\underbrace` and `\arg` render correctly; `\mathbf{x}` comes out as bold-italic Unicode (ISO vector style, accepted). All browsers of plan 5.2 render MathML Core; no math CSS beyond the prototype's `math { font-family: "Latin Modern Math", "STIX Two Math", "Cambria Math", math; }` and no math fonts are shipped.
- Twins keep the `$...$` source (GitHub renders it, D8).

## 12. Static parts

### 12.1 `website_builder/static/`

Copied unchanged into the site root (files named `README.md` are skipped):

| Path | Source |
|------|--------|
| `css/site.css` | Port of the prototype `css/site.css` (section 12.2) |
| `js/site.js` | Port of the prototype `js/site.js`: copy buttons, remembered language (`localStorage` key `qlib-lang`, written only in the `change` handler of the language tabs, D18), function filter with `/`, current-function highlight (design 7); selectors match section 8 |
| `fonts/` | The seven `woff2` files and `LICENSE.txt` of the prototype (committed, not from npm; plan section 11) |
| `img/favicon.svg` | The logo with `<style>svg { color: #2e7d46 } @media (prefers-color-scheme: dark) { svg { color: #7fd197 } }</style>` |
| `img/favicon-32.png`, `img/apple-touch-icon.png` (180 × 180), `favicon.ico` | Rendered once from `img/favicon.svg` (Inkscape, ImageMagick) and committed (design open point 5) |
| `img/landing/README.md` | One line: the figures of D15 go here, with the file names of design section 8 |
| `.htaccess` | Section 12.5 |
| `publications/`, `quadriga_lib_material_model.md` | Hand-made files moved from `html_docu/` in package 0 (section 18.1) |

### 12.2 `site.css`

A port of the prototype: every token value, breakpoint, size and colour stays (design 3). Changes: remove `.search`; code colours for the classes of section 10; `.params > .pnote` spans both columns; `.prose h4` and `.note-box > * + *` spacing; print also hides `.doc-side` and `.idx-filter`; the prototype's inline `style` attributes become classes (`.page`, `.page-narrow` and spacing rules with the prototype's values); `.ph` stays for gap markers.

### 12.3 Preview image (design open point 5)

`og:image` and `twitter:card` are emitted only if `static/img/og-card.png` exists; otherwise the gap list names it. The image is made in the figure session (design section 8).

### 12.4 Landing figures

For each name of `LANDING_FIGURES`, with `<base>` = `img/landing/<name>`: if `static/<base>-light.webp` and `-dark.webp` exist, `<figure><div class="frame"><picture><source media="(prefers-color-scheme: dark)" srcset="<base>-dark.webp"><img src="<base>-light.webp" width="1600" height="1200" alt="..." decoding="async"></picture></div><figcaption>...</figcaption></figure>`; otherwise the same frame, empty, with the alt text as a centred label, and the gap list names the missing files. Every figure except the first on the page gets `loading="lazy"`. The figure scripts go into `website_builder/figures/` in the figure session (D15, D39); they write the WebP files into `static/img/landing/`, where the images are committed.

### 12.5 `.htaccess`

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

`.md` is served as `text/markdown; charset=utf-8`, `llms.txt` as `text/plain; charset=utf-8` (plan section 11). The hoster serves no compressed responses today, so the budgets of section 15.3 are in raw bytes; `mod_deflate` is tested after the first deploy (section 18.4), not in step 5.

## 13. Tutorials (D28, D34, plan 5.3)

### 13.1 Notebooks

`tutorials/<stem>.ipynb`, `<stem>` matching `[a-z0-9][a-z0-9_-]*`, committed with their outputs (D34). The website builder reads them with `json` from the standard library, never executes and never rewrites them. Running and refreshing the notebooks before a release is a separate project (D34). A code cell without an execution count gives the warning `not-run`.

### 13.2 Metadata

Notebook-level `metadata.quadriga_tutorial`, checked like the publications data (section 7.2): `title` (str), `summary` (str, inline Markdown), `level` (`beginner`, `intermediate`, `advanced`), `duration_min` (int), `order` (int; index order, then title), `dummy` (bool, optional). Language from `metadata.kernelspec.language` (`python` → "Python", `matlab` → "MATLAB", `octave` → "Octave"). A stem that does not match section 13.1 or a missing `kernelspec.language` is a structural error. A tutorial page's description and `og:description` are `summary` as plain text. Cell-level `metadata.quadriga_tutorial.alt` is the alt text of that cell's images; missing gives the warning `figure-alt` and "Figure output of cell <n>".

### 13.3 Conversion

- Markdown cells: rendered like prose (notebook kind). All heading levels move down one: `#` → `h2`, `##` → `h3`, deeper → `h4`; the twin does the same (section 8.6.1); ids as for prose headings (section 6.2).
- Code cells: `<div class="cell"><div class="cell-label">In [n]</div><pre><code class="language-<lang>">...</code></pre>` and one `<div class="out">` per output (`n` is the execution count).
- Outputs: `stream` text as `<pre>` (ANSI codes removed). An `execute_result` or `display_data` bundle shows one representation, the first present of `image/svg+xml` (stored as text in the notebook, written as is), `image/png` (base64, decoded), `text/plain` (as `<pre>`); images go to `<stem>/fig-<cell>-<output>.<ext>` (`<cell>` is the 1-based position of the cell among all cells, `<output>` the 1-based position of the output in the cell) and are shown as `<img alt loading="lazy">`. Only a bundle with none of the three gives the warning `output-type` and is left out. An `error` output is a structural error (the notebook was committed with a failing cell).
- Twin `tutorials/<stem>.md` (D26): twin header (web page `<base_url>/tutorials/<stem>.html`), a line with the two downloads, Markdown cells through the twin writer (section 8.6.1), code cells as fenced blocks, text outputs as `text` fences, images as `![alt](<stem>/fig-<cell>-<output>.<ext>)`. Twin and page sit side by side, so one relative image path works in both (D28).

### 13.4 Pages

Per notebook: `tutorials/<stem>.html` (template `tutorial.html` with `root` `"../"`), `tutorials/<stem>.md`, `tutorials/<stem>.ipynb` (the source notebook, D28), the figures. Downloads: "Download notebook .ipynb" and "Markdown .md". The twin `tutorials.md` lists each tutorial in the same order with link, language, level and summary. The index (`tutorials.html`, twin `tutorials.md`, both website only) shows cards in `order` with language tag, level tag, title and summary; if every notebook is a dummy, `dummy_note` is shown as a warning note box; with no notebook the index shows "No tutorials yet." (plan 5.3).

### 13.5 The dummy notebook

`tutorials/pipeline_demo.ipynb`, kernel `python3`, metadata `title` "Pipeline demo", `summary` "Placeholder notebook that tests the tutorial pipeline.", `level` beginner, `duration_min` 5, `order` 1, `dummy` true. Cells (fixed seed): (1) Markdown `## Set-up` and a sentence; (2) code: `import numpy as np`, `import quadriga_lib`, `rng = np.random.default_rng(1)`, `print(quadriga_lib.version())`; (3) Markdown `## An antenna` with `$D = 10 \log_{10}(\max G)$`; (4) code: `ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)` and `print(round(quadriga_lib.arrayant.calc_directivity(ant, 0)[0], 2))` (indexing, not `float()` on the array, so numpy prints no DeprecationWarning); (5) code with `alt` "Line plot of 50 random values": a matplotlib plot of `rng.standard_normal(50)` with `%matplotlib inline`, ending with `plt.show()` so the cell has no `execute_result` showing an object address; (6) Markdown with a three-row table and `$$\sigma^2 = \frac{1}{N}\sum_{n=1}^{N} (x_n - \bar x)^2$$`.

Step 5 runs it once to store the outputs: `jupyter nbconvert --to notebook --execute --inplace tutorials/pipeline_demo.ipynb` in any environment with `quadriga-lib` (0.12.2 or newer from pip), `numpy`, `matplotlib` and `jupyter` installed, for example a scratch conda environment in the agents folder. That environment is not part of the project. The stored outputs must contain no warning text.

## 14. Agent skill placeholder (D10, D37)

`website_builder/skill/quadriga-lib/SKILL.md`:

```
---
name: quadriga-lib
description: Placeholder for the Quadriga-Lib agent skill (radio channel modelling library with C++, MATLAB/Octave and Python APIs). Until the skill exists, read {{ site_base_url }}/llms.txt.
---

# Quadriga-Lib skill (placeholder)

This skill is a placeholder. The documentation for agents starts at {{ site_base_url }}/llms.txt.
```

`website` copies it to `skill/SKILL.md` after substituting build variables (section 6.3); the AI page and `llms.txt` link it. No zip (D37). The placeholder is not a gap: the site may go live with it (plan 5.4 asks for a note on the AI page).

## 15. Checks after the build

`website` runs these on the in-memory output (section 3.6), in `qsite/checks.py`; the Markdown link check is `qdocs/checks.py`, which `docs` runs limited as in section 15.1 and `website` runs on all generated Markdown. A failed check is an error unless stated.

### 15.1 Links and anchors

For every HTML file: every `href`, `src` and `srcset` that is not absolute or `mailto:` is resolved against the file (root-absolute against the site root); the target must exist in the output; a fragment on an HTML target must be an `id` there, on a `.md` target an `<a id>` there (section 9.5). For every Markdown file the builders generate (twins, API files, `llms.txt`, `docs/README.md`, tutorial twins): every link and image URL found by the twin writer's scanner (section 8.6.1), and every `<a id>` as a target, line by line; resolved the same way; absolute URLs that start with `base_url` are mapped to the output and checked too. Hand-made files copied into the site (`publications/**`, `quadriga_lib_material_model.md`) are not checked inside; links into them are checked for existence only. URLs that start with `DOWNLOAD_BASE_URL + "/"` or `RELEASES_URL` and all other external URLs are not fetched. In the `docs` phase only links whose target lies in `docs/` are checked; absolute site links and links to files outside `docs/` are checked in `website`. It must catch the case of plan section 10: a FAQ without `cite` breaks the footer link `faq.html#cite` on every page and the twin link `faq.md#cite`.

### 15.2 HTML well-formedness

`html.parser` with a tag stack: every non-void element closed in order; no duplicate `id`; exactly one `h1`; exactly one `<main id="content">`; every `<img>` has `alt`.

### 15.3 Size budgets (warnings)

Raw bytes (section 12.5), per page: API pages 800 kB (the C++ page is expected near 700 kB); other HTML pages 150 kB. Landing total (plan 5.2): `index.html` + `css/site.css` + `js/site.js` + every font file the landing page uses (IBM Plex Sans 400, 600 and 700 and IBM Plex Mono 400 for the quick start, per the CSS) + the light variant of each existing landing figure at most 480 kB (prototype: 24 + 26 + 4 kB, fonts 23 + 24 + 23 + 15 kB, plus at most 2 × 150 kB of figures: 439 kB, with about 10 % headroom); each landing WebP at most 150 kB (design section 8). Each exceeded budget is a warning `[size]`.

## 16. Documentation in packages and the release (D32, D39)

The built site goes into no package. Every binary package carries the tracked Markdown of `docs/`, which holds all the documentation:

| Package | How `docs/` gets in | Changed in package D (section 18.3) |
|---------|---------------------|-------------------------------------|
| System install (`cmake --install` with a prefix other than the source folder, deb build 1) | `CMakeLists.txt`, system-install branch: `install(DIRECTORY docs/ DESTINATION share/doc/quadriga-lib)` in place of `install(DIRECTORY html_docu/ DESTINATION ${INSTALL_DOCDIR})` | both `set(INSTALL_DOCDIR ...)` lines go; the local layout installs nothing (the tree already has `docs/`) |
| Debian package | through the system install of build 1 (`COPY . .` in `Dockerfile.ubuntu` brings `docs/` and `docs_builder/`; `.dockerignore` keeps `website/` and `website_builder/` out) | `debian/rules`: `override_dh_compress:` with recipe `dh_compress -X.md -Xllms.txt`; `debian/control`: "and HTML API documentation" (it spans a line break in the description) becomes "and the API documentation as Markdown files", with the description lines reflowed; `debian/stamp-changelog.sh` line 53: "HTML API documentation included." becomes "Markdown API documentation included." |
| Windows zip | `Makefile` `release-win`: `xcopy /E /I "docs" "$(DIST_DIR)\docs"` in place of the `xcopy html_docu` line. `cmake --install` does not do it: `all` configures with `CMAKE_INSTALL_PREFIX=%CD%`, the local layout, so the system-install branch never runs for this build. | `README_win.md`: the `html_docu/` row becomes `docs/` "Documentation as Markdown files (the content of quadriga-lib.org)" |
| pip wheels | none (as today) | — |
| pip sdist (scikit-build-core) | contains every file git does not ignore, so `docs/` (today `html_docu/`); it also carries `website_builder/`, as it carries the publications today (`export-ignore` does not apply to it) | — |
| Source zip (`make package`, `git archive HEAD`) | `docs/` and `docs_builder/` are tracked; `website_builder/` is `export-ignore` (section 18.1) | — |

`dh_compress` (debhelper 13, compat 13 here) gzips every file below `usr/share/doc` larger than 4 kB, except a fixed list of extensions (`.htm*`, `.css`, `.js`, `.svg`, `.pdf`, images, archives, fonts and a few others); `.md` is not on that list, so without `-X.md` the larger Markdown files would become `.md.gz` and every relative link to them would break; `-Xllms.txt` keeps `llms.txt`, which `ai_agents.md` links relatively. `-X` excludes files whose name contains the given text. Checked against the `dh_compress` man page and its source on salsa.debian.org (the `find` expression with `-size +4k`).

Every release build runs the docs check (D39) with a plain Python 3.10 or newer, no conda environment and no website tools; a stale `docs/` stops the build before anything is packaged:

| Release path | Where the check runs | Why there |
|--------------|----------------------|-----------|
| `make release` (`GNUmakefile`, D32) | First line of the recipe: `test -z "$$(git status --porcelain --untracked-files=no; git status --porcelain -- docs)" \|\| { echo "uncommitted changes: commit before a release"; exit 1; }`, which refuses any uncommitted change to a tracked file (the packages are made from `HEAD` and from the working tree, which must agree) and any new file in `docs/` not yet committed (a new section file); then `python3 docs_builder/build.py docs --check` (the committed Markdown is current); then `$(MAKE) package`, `$(MAKE) deb`, `$(MAKE) pip`. `release:` has no prerequisites and its own line `.PHONY: release` (the folder `release/` exists, so without it make reports the target up to date and runs nothing). | Before the source zip, which is the first package made. |
| Each deb container (`Dockerfile.ubuntu`, also when `make deb-jammy` and the others run on their own) | A step `RUN python3 docs_builder/build.py docs --check` directly after `COPY . .` and before the changelog stamp; `python3-tomli` is added to the existing `apt-get install` line (needed on 22.04, harmless on newer releases). | It checks the tree the container packages, and it fails in seconds, before the long compile of `dpkg-buildpackage`. The deb recipes pipe `docker build` into `tee`; `GNUmakefile` gets `SHELL := /bin/bash` and `.SHELLFLAGS := -o pipefail -c` (section 18.3), so the failure stops make. |
| Windows zip (`Makefile`, `release-win`) | First command of the recipe: `python docs_builder\build.py docs --check`. | Before the old dist folder is removed and anything is copied, so a failure leaves nothing half-made; `release-win` is the step that copies `docs` into the zip. The Windows build machine needs Python on `PATH` for this step (section 18.3, `README.md`). |

The website is built (`make website`) and deployed independently of the release (D29).

## 17. Tests

pytest, on the maintainer's machine only (the builders never need it). All tests are in `website_builder/tests/` (section 2.1): `tests/docs/` tests `qdocs` and imports nothing else; `tests/site/` tests `qsite`. `website_builder/pytest.ini` sets `testpaths = tests`, `addopts = -p no:cacheprovider` and the marker `repo` (reads the real repository); `tests/conftest.py` inserts `docs_builder/` and `website_builder/` into `sys.path` (no `pythonpath` setting, so pytest 6.2 of Ubuntu 22.04 works too). Test file names are unique across both folders (they have no `__init__.py`). Run all with `python3 -m pytest website_builder/tests` in the active environment; the docs tests alone with `python3 -m pytest website_builder/tests/docs`, which needs only Python and pytest. The suite runs in under two minutes, needs no network and writes only below `tmp_path`.

`tests/conftest.py` provides `mini_repo(tmp_path)`, a copy of `tests/fixtures/mini_repo/` (a minimal repository: header with version `9.9.9`, `CMakeLists.txt` with `set(armadillo_version "1.0.0")`, two or three small blocks per language, `docs_builder/content/`, `website_builder/content/`, `website_builder/skill/` and a small notebook with outputs in `tutorials/`); `run_docs(root, *args)`, which calls `qdocs.cli.main` with `--root` and returns exit code and output; and `run_site(root, *args)`, the same for `qsite.cli.main`, importing `qsite` inside the function so the docs tests never import it. Rendering tests assert substrings of the `mini_repo` output, not whole files; for real data the tracked `docs/` plus `docs --check` is the reference output.

| File | Package | Covers |
|------|---------|--------|
| `docs/test_docs_cli.py` | A | `docs` on `mini_repo` exits 0; a structural error gives 1 and writes nothing; every `import` in `docs_builder/**/*.py` names a module of `sys.stdlib_module_names`, `qdocs`, `tomllib` or `tomli` (`tomllib` is not in that set on 3.10), never `qsite` |
| `docs/test_docs_output.py` | A | Only changed files written; stale files removed; `--check` writes nothing and exits 1 on a changed, missing or extra file; a copy of `mini_repo` whose sources and `docs/` have `\r\n` line ends gives the same output, and `--check` passes on it |
| `docs/test_docblocks.py` | A | Line numbers; each error id of section 5.2 found on the docs path (one small fixture each); sub-block kinds; required sub-blocks with the class-block exemption; parameter items (lazy continuation, end at a blank line, nested lists and tables kept in the item), pattern variants of section 5.4, note rows, warnings |
| `docs/test_links.py` | A | `[[...]]` resolution and E-link; URLs of each link kind per output location; `slug`, legacy hashes (section 9.3 values); E-anchor; families, mapping order and mapping errors |
| `docs/test_prose.py` | A | Front matter errors; publications key check with key paths; build variables and the unknown-name error; heading ids with the plain-text rule of section 6.2 (code span, link, `[[x]]`, escape, entity, `{...}`); gap markers removed from twins (marker-only line or list item dropped, empty paragraph and empty field gone); gap scan of `docs_builder/content/`; a missing content file is a structural error |
| `docs/test_render_md.py` | A | Twin writer: sub-block demotion, `{...}` stripped, fence language added, `[[x]]`, raw `<a href>`, trailing `<br>` and site URLs rewritten outside code and unchanged inside code spans and fences; anchors before functions, sections and headings; full API file (front matter without date, index without "Line", counterpart line with "Also in" links and, for a function with a missing counterpart, "Not in"), a section file, a prose twin (index table, alert kept), `llms.txt` and `docs/README.md` (absolute and relative links) |
| `docs/test_md_checks.py` | A | Missing file, missing `<a id>` in a Markdown target, the `faq.md#cite` case; links outside `docs/` skipped in the `docs` phase |
| `docs/test_docs_content.py` | C | (`repo`) the real `docs_builder/content/` parses; `cite` and the legacy ids of sections 6.6 and 9.3 exist; publications pass their key check |
| `docs/test_docs_real_repo.py` | C | (`repo`) `docs` on the real repository into a temporary root: zero errors; 137, 85, 80 functions; 10, 9, 9 sections; `docs --check` passes after `docs` |
| `site/test_site_cli.py` | B | All missing tools (Python packages and `pandoc`) reported in one run, exit 2 before reading sources: `sys.executable -S website_builder/build.py website --root <mini_repo>` with `PATH` empty (`-S` hides site-packages, the empty `PATH` hides pandoc); `website` writes `docs/` and `website/`; gap list of all three folders |
| `site/test_mdparse.py` | B | `wikilink` with member form; `raw_a`; `raw_br`; raw HTML in prose shown as text; `heading_attrs`; `alert`; `site_url`; `$5 and $6` not math; code untouched; E-heading for a setext heading |
| `site/test_render_html.py` | B | Substrings: head tags (alternate link and Markdown footer link only with a twin, `og:image` only with the image), tabs with `aria-current`, function entry, parameter rows, pills, section and legacy ids, text-page index, note boxes with default and bold titles, span-free code colouring and the `highlight` filter; `404.html` root-absolute; tutorial pages with `../`; gap markers kept as placeholders in HTML (page text, `<title>` and attributes) and `website --release` failing on gaps or replaced markers; `.htaccess` copied to the site root; the landing size warning |
| `site/test_math.py` | B | Inline (`math_inline`) and display (`math_block`, `math_inline_double`) MathML (`display` attribute); `math_block_label` a structural error; several formulas give exactly one pandoc call and a page set without formulas none (counted by patching `subprocess.run`); a formula with an unknown command and one with unbalanced braces: structural errors at their source lines; twins keep `$...$` |
| `site/test_tutorials.py` | B | Metadata errors; output conversion (stream; SVG before PNG before `text/plain` in one bundle; a bundle with none of them gives `output-type`; `error` output rejected; `not-run`; figure numbering); heading levels moved down in page and twin; page, twin, notebook copy, figures, card and dummy note in the `website` output |
| `site/test_site_checks.py` | B | Missing file, missing id, absolute `base_url` link in a twin, the `faq.html#cite` case in HTML; unclosed tag, duplicate id, two `h1`, `img` without `alt`; size warnings |
| `site/test_site_real_repo.py` | C | (`repo`) every id of `fixtures/old_anchors.json` (captured once from `html_docu/*.html` without the dropped ids of section 9.6) and `index.html#quickstart` exist in the `website` output; two `website` runs give identical bytes; `pipeline_demo.ipynb` converts without warnings (these cases skip themselves while `website_builder/templates/base.html` is missing) |

Windows is checked after the merge, on the `main` checkout, which is the only tree the Windows VM sees (section 18.4).

## 18. Cutover work items (D35)

Package 0 does section 18.1 at the start of step 5, so the builders know only the new paths and has no `html_docu` constants. Package D does sections 18.2 and 18.3 at the end of step 5; section 18.4 is its acceptance. D20's ban on CMake and packaging changes does not apply to these items (D35).

### 18.1 Moves (package 0)

1. `git mv html_docu/publications website_builder/static/publications`; in `.gitattributes` the line `html_docu/publications/ export-ignore` becomes `website_builder/ export-ignore` (the whole website builder leaves the source package, D39); in `.dockerignore` `html_docu/publications/` becomes `website_builder/`.
2. `git mv html_docu/quadriga_lib_material_model.md website_builder/static/quadriga_lib_material_model.md`; `tools/acoustic/check_docs.py`: `PAGE` points to the new path.
3. `git mv html_builder/img website_builder/logo` (logo sources, not deployed).
4. `git mv html_builder/extract_matlab_comments.py docs_builder/extract_matlab_comments.py`; in `CMakeLists.txt` (`mex_documentation`) the path becomes `${CMAKE_SOURCE_DIR}/docs_builder/extract_matlab_comments.py`. The script keeps to the standard library and Python 3.10 (deb build in a jammy container, CMake on Windows) and does not import `qdocs`.

### 18.2 Deletions (package D)

`html_docu/` (all remaining files) and `html_builder/` except `html_builder/design_prototype/`, which step 6 still compares against and step 7 deletes.

### 18.3 Changes (package D)

| File | Change |
|------|--------|
| `CMakeLists.txt` | As in section 16: delete both `set(INSTALL_DOCDIR ...)` lines; `install(DIRECTORY docs/ DESTINATION share/doc/quadriga-lib)` replaces the `html_docu` install line. |
| `GNUmakefile` | Add `SHELL := /bin/bash` and `.SHELLFLAGS := -o pipefail -c` near the top, so the deb recipes, which pipe `docker build` into `tee`, fail when the build fails (section 16); delete `HB = html_builder` and the `documentation` target; `release` and its `.PHONY: release` line as in section 16 (with plain `python3`, no environment); in `package_dirty` the exclusion `-x 'html_docu/releases/*'` becomes `-x 'website/*'`; `clean` unchanged (it keeps `website/`); `tidy` already has the lines of section 3.10. |
| `Makefile` (Windows) | `release-win` as in section 16: the docs check as its first command, the `xcopy` of `docs`. |
| `debian/rules`, `debian/control` | As in section 16. |
| `Dockerfile.ubuntu` | As in section 16: `python3-tomli` in the `apt-get install` line (with a comment: the docs check on 22.04), the `RUN python3 docs_builder/build.py docs --check` step after `COPY . .`. |
| `debian/stamp-changelog.sh` | As in section 16. |
| `.dockerignore` | Delete `html_docu/releases/`. |
| `.gitignore` | Delete `html_docu/releases/`. |
| `README.md` | Make-target table: the `make documentation` row becomes `make docs` (tracked Markdown in `docs/`), `make docs-check`, `make website` (site in `website/`); repository-layout table: the `html_docu/` and `html_builder/` rows become `docs/` (documentation as Markdown, generated by `make docs`), `docs_builder/` (makes `docs/` and the MATLAB help files; Python standard library), `website_builder/` (makes the website; not in the source package) and `tutorials/`; Windows build step 1: for `nmake release-win`, Python 3.11 or newer (3.10 with `tomli`), for its docs check; `numpy` and `pytest` stay optional for the tests. |
| `README_win.md` | As in section 16. |
| `md/SPEC_CPP_API.md` | "the pages in `html_docu/`" becomes "the pages of quadriga-lib.org"; `html_docu/quadriga_lib_api_cpp.md` becomes `docs/quadriga_lib_api_cpp.md`; `make documentation` becomes `make docs`; the "Line" column advice becomes: read the function index of the full file, then open only the section file `docs/api/cpp/<section>.md`; "may lag behind" becomes: regenerated by hand, may lag behind `main` (D8); document the doc-block extensions of D6: `$...$` and `$$...$$` math; line 208 "compiled to HTML and Markdown (`make documentation`)" becomes "rendered into the website and the Markdown files in `docs/` (`make website`, `make docs`)". |
| `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` | `make documentation`, `html_docu/quadriga_lib_api_*.md` and "may lag behind" as for the C++ file; script path `docs_builder/extract_matlab_comments.py`. |
| `md/BUG_SCAN.md` | "A stale `html_docu/` alone" becomes "Stale generated Markdown in `docs/` alone" (`make docs-check` shows it). |
| `md/acoustic_modeling_guidelines.md`, `md/acoustic_engine_notes.md`, `md/materials_kickoff.md`, `md/materials_manager.md`, `md/gbsm_integration_handoff.md` | `html_docu/quadriga_lib_material_model.md` becomes `website_builder/static/quadriga_lib_material_model.md`, `make documentation` becomes `make docs`; in `md/gbsm_integration_handoff.md` line 50 "`html_docu/` after `make documentation`" becomes "`docs/` after `make docs`". |
| `tools/update_armadillo.py` | Delete `html_builder/download/quick_start.md` from `VERSION_FILES` and its comment: the Get started page takes the Armadillo version from `CMakeLists.txt`. |
| `tools/acoustic/check_docs.py` | `FORMATS` becomes `docs_builder/content/formats.md`; the docstring lines 8 and 14 name the new paths of both files; run it once, its parser may need a small change for the new headings. |
| `tools/acoustic/README.md`, `band_probe.cpp`, `gzlib.py`, `paris_cf.py`, `txlib.py`, `tools/sos_port/SOS_PAPER_REPRODUCTION.md`, `tests/catch2_tests/test_ieee_indoor_publication.cpp` | Comments naming `html_docu/...` point to `website_builder/static/...` or the URL on quadriga-lib.org. |

### 18.4 Acceptance of the cutover and checks at go-live

After package D: `make docs-check`, `make website` and the tests pass; `git grep -n 'html_docu\|html_builder'` finds nothing outside `md/website_*.md` and `html_builder/design_prototype/`; a configure with `-DCMAKE_INSTALL_PREFIX=<tmp>` and `cmake --install` puts `docs/` into `<tmp>/share/doc/quadriga-lib/`; a MEX build (`ENABLE_MEX_DOC=ON`) still writes the `.m` help files; the deb built in Docker (`make deb-noble`, and `make deb-jammy` for Python 3.10 with `tomli`) shows the docs check passing in its log and contains `usr/share/doc/quadriga-lib/quadriga_lib_api_cpp.md` uncompressed and `api/cpp/*.md`, and lintian shows no new tags; in a scratch copy of the tree with a stale `docs/`, `make deb-jammy` exits non-zero; a source zip made with `git archive --format=zip HEAD` after the maintainer's commit, unpacked into a scratch folder, contains no `website_builder/`, passes `make docs-check` there, and `make website` there stops with the message of section 3.5. These builds compile the library; they follow the machine rules (load, thread count) and are run by the manager or with the maintainer's agreement.

After the merge, on the `main` checkout (the Windows VM sees only that tree), with the first `nmake release-win`: `python docs_builder\build.py docs --check` and `python -m pytest website_builder\tests\docs` pass, and `release-win` runs the docs check first and gives a zip with `docs\quadriga_lib_api_cpp.md`. Before the merge the docs builder is checked on Linux and on Python 3.10 in the jammy container only (section 19, package A).

At go-live, after the first deploy (plan section 7): `curl -sI <base>/formats.md` shows `text/markdown; charset=utf-8`; `curl -sI <base>/llms.txt` shows `text/plain; charset=utf-8`; a missing page returns 404 with the new page; compression test: add `AddOutputFilterByType DEFLATE text/html text/css application/javascript text/markdown text/plain` to the server's `.htaccess`; if `curl -sI -H 'Accept-Encoding: gzip' <base>/cpp_api.html` then shows `Content-Encoding: gzip` and a plain request still works, the line goes into `website_builder/static/.htaccess`, otherwise it is removed; the twin anchors work on GitHub (section 9.5); `website_builder/build.py website --release` passes (no gaps left).

## 19. Work packages for step 5

Each package owns the files listed; a package that needs a change in a file it does not own asks the manager.

| Id | Package | Owns | Depends on |
|----|---------|------|------------|
| 0 | Moves | the `git mv` and line changes of section 18.1 | none |
| A | Docs builder | `docs_builder/build.py`; `docs_builder/qdocs/` (all modules of section 2.2); `docs_builder/content/api_matching.toml`; `website_builder/pytest.ini`; `website_builder/tests/conftest.py`; `tests/fixtures/mini_repo/` except its `website_builder/` and `tutorials/` folders; the tests marked A; the `GNUmakefile` lines of sections 3.5 and 3.10; the lines of section 2.4 | 0 |
| B | Website builder | `website_builder/build.py`; `website_builder/qsite/`; `pandoc_math.yaml`; `templates/` (with the landing text of section 7.1); `static/` except the moved files; `website_builder/README.md` section "Set up a build machine"; `mini_repo/website_builder/` and `mini_repo/tutorials/`; the tests marked B | A |
| C | Content | `docs_builder/content/` except `api_matching.toml`; `website_builder/content/`; `website_builder/skill/quadriga-lib/SKILL.md`; `tutorials/pipeline_demo.ipynb` with outputs; `tests/fixtures/old_anchors.json`; the tests marked C | A; B for acceptance |
| D | Cutover | the deletions and changes of sections 18.2 and 18.3 | A, B, C accepted |

Order: 0; A (the docs builder first); then B and C in parallel; B's checks on the real repository run after C is in; then D.

Interface from A to B: `qsite` uses only these names of `qdocs` and keeps no copy of their logic; A chooses their arguments and documents them in docstrings, and changes them only through the manager:

| Name | Gives |
|------|-------|
| `cli.build_docs(root, diag)` | The whole docs phase in memory: the content model (API pages, prose pages, publications) and the `docs/` mapping (sections 5 to 9, 15.1). |
| `cli.Diagnostics`, `cli.write_tree(folder, mapping)` | Error, warning and gap collection and printing (section 3.7); writing a mapping with the rules of section 3.6. |
| `config` | The constants and `PAGES` (section 3.3), the versions (section 3.8), `tomllib`. |
| `prose.read_page(path, diag)`, `prose.substitute(text)`, `prose.gap_scan(folders)` | A website-only prose page with front matter and heading ids (sections 6.1, 6.2); build variables (section 6.3); the gap scan (section 6.4). |
| `prose.heading_ids(lines, level_shift)` | The heading ids of section 6.2 for a list of source lines; used by `read_page` (shift 0) and for notebook Markdown cells (shift 1, section 13.3). |
| `links.url(target, location)`, `links.slug(text)` | URLs per output location (section 8.7.1), ids (section 9.2). |
| `render_md.twin_lines(lines, ...)` | The twin writer for one part, used for notebook Markdown cells (sections 8.6.1, 13.3). |
| `render_md.twin_header(title, web_url, headings)` | The twin header with the index table (section 8.6.1), used for the tutorial twins. |
| `render_md.prose_twin(page)` | The whole twin of a prose page, gap removal in fields included (sections 6.4, 8.6.1), used for `tutorials.md`. |
| `checks.check_markdown(mapping, scope)` | The Markdown link check (section 15.1). |

Acceptance (each package also passes its tests and leaves the others green):

| Id | Done when |
|----|-----------|
| 0 | `git status` shows the four moves as renames; after a configure with `ENABLE_PYTHON=ON` and `ENABLE_MEX_DOC=ON` (without Python, `CMakeLists.txt` switches the MEX help off), `cmake --build <dir> --target mex_documentation` writes the `.m` help files (CMake does not check the script path at configure time; the target needs Python and Octave or MATLAB found by the configure). |
| A | `python3 docs_builder/build.py docs` and `docs --check`, with `--root <copy of mini_repo>`, work with the system `python3` outside any conda environment and in a throwaway container of the local image `quadriga-deb-jammy` with `python3-tomli` installed (Python 3.10, tomli 1.2): `docker run --rm --user $(id -u):$(id -g) -e PYTHONDONTWRITEBYTECODE=1`, the tree mounted read-only and the run made on a copy under `/tmp` in the container, optionally with `python3 -m pytest website_builder/tests/docs` there; a scratch script that calls `qdocs.docblocks` on the real doc blocks (parser only; the real `content/` arrives with C) finds zero errors and about 17 `param-separator` and 25 `param-item` warnings; no output contains a date; after `make tidy`, `git status --porcelain --ignored` shows nothing new in the working copy. |
| B | On `mini_repo`: the tests marked B pass. The conda set-up following only the README section succeeds and `make website` runs in it (the Ubuntu line is checked where a 26.04 machine is at hand); without `pandoc` on `PATH` the build stops with the message of section 3.4. After C is in: `make website` builds all pages with link, anchor and well-formedness checks passing and the C++ page within budget; the dummy tutorial shows code, text output, figure, table, formula and downloads; manual: the CSS token values equal the prototype's, and the pages match the prototype at 375, 768 and 1280 px in the browser pane. |
| C | `make docs` on the real repository writes the files of section 2.3 with zero errors and `make docs-check` passes right after; publications pass their key check; `cite` and the legacy ids exist; every prototype draft and design-section-9 item is a gap; `pipeline_demo.ipynb` is committed with outputs and without warning text; `test_docs_real_repo.py` and `test_site_real_repo.py` pass; a plain-text comparison of the old and new formats, FAQ and download pages shows no lost text. |
| D | Section 18.4, first paragraph. |

The manager's acceptance run after all packages: the full build on the real repository; one timed run each of `make docs` and `make website` (`/usr/bin/time`, wall and CPU) with the 1-minute load average recorded before and after, compared with the targets of section 3.11 (a run above load 4 is reported as provisional); manual check that `make tidy` after `docs`, `website` and the tests leaves the state of a clean clone and that `make clean` keeps `website/`; and a report with these timings, the gap list, the warning counts and the page sizes against the budgets.

## 20. Traceability

Every item of design section 11, plan section 10, plan section 11 and decisions D32 to D39, with the section that answers it. This section is deleted after the spec review.

| Source | Item | Answered in |
|--------|------|-------------|
| Design 11.1 | Matching rule for the language pills | 9.4, 8.4 |
| Design 11.2 | Section anchors: readable ids and old hash ids | 9.2, 9.3, 8.4, 6.2, 6.6 |
| Design 11.3 | Names and paths of the per-section Markdown files; footer "This page as Markdown" | 2.3, 8.6.3, 8.4, 8.2 |
| Design 11.4 | Pygments without spans for uncoloured tokens; API page budget | 10, 15.3 |
| Design 11.5 | Favicon PNG fallback, preview image `og-card.png` | 12.1, 12.3, 8.3 |
| Design 11.6 | Where the figure scripts live, how images are committed | 12.4 |
| Design 11.7 | MATLAB and Python pages: labels, lower-case names | 5.3, 5.5, 6.7, 3.3 |
| Design 11.8 | Placeholder marking, look on the page, build list | 6.4, 12.3, 12.4, 3.1 (`--release`) |
| Plan 10 | Paths of the design mapped to D24 and D33 | 1.2, 2.3 |
| Plan 10 | FAQ provides `#cite`; link check catches such gaps | 6.6, 15.1, 17 (`test_md_checks.py`, `test_site_checks.py`, `test_docs_content.py`) |
| Plan 10 | Page size: budget per API page | 10, 15.3 |
| Plan 5.2 | Landing-page size budget, fonts and images included | 15.3 |
| Plan 10 | Old section anchors keep working | 9.3, 6.6, 9.6, 17 (`test_site_real_repo.py`) |
| Plan 10 | HTTPS: site base URL, `http://` until SSL, commented redirect, doc-block links stay | 3.3, 12.5, 5.6 (`site_url`) |
| Plan 10 | "For team leads" points into the landing page | 7.1 |
| Plan 10 | Design open points answered using D24 to D38 | the design rows above |
| Plan 11, cutover | `CMakeLists.txt`: install lines, help script path, Python 3.10 and standard library | 18.1, 16, 18.3 |
| Plan 11, cutover | `Makefile`, `Dockerfile.ubuntu`, `.dockerignore`, `debian/stamp-changelog.sh`, `.gitattributes`, `GNUmakefile` (`documentation`, `release`, `package_dirty`, `clean`, `tidy`) | 16, 18.1, 18.3, 3.10 |
| Plan 11, cutover | Text naming old paths (SPEC files, `md/BUG_SCAN.md`, `md/acoustic_modeling_guidelines.md`, `md/gbsm_integration_handoff.md`, `README.md`, `README_win.md`, `tools/update_armadillo.py`); further users found | 18.3 |
| Plan 11 | Full API files, no build date, no "Line" column; per-section files; largest section accepted or split | 8.6.2, 8.6.3, 2.3 |
| Plan 11 | Links in twins to files outside `docs/`; raw `<a href>` in doc blocks | 8.7.1, 8.6.1, 5.6 |
| Plan 11 | One Markdown intro per language; Python intro is a content gap | 6.7 |
| Plan 11 | Build variables and gap markers with different syntax; strict mode fails on gaps and missing figures | 6.3, 6.4, 3.1, 12.3, 12.4 |
| Plan 11 | Explicit heading ids stripped from twins | 6.2, 8.6.1, 9.5 |
| Plan 11 | Fonts committed; MathML output chosen with reason (D38: pandoc) | 12.1, 11 |
| Plan 11 | Cross-language matching | 9.4 |
| Plan 11 | Doc-block grammar variants | 5.2, 5.3, 5.4, 5.6 |
| Plan 11 | Pygments: uncoloured tokens without `<span>` | 10 |
| Plan 11 | Byte-identical output: sorted lists, no dates (no zip any more, D37) | 3.9, 17 (`test_docs_real_repo.py`, `test_site_real_repo.py`) |
| Plan 11 | Server without compression and `.md` type: raw budgets; content types in `.htaccess` | 12.5, 15.3, 18.4 |
| Plan 11 | `404.html` root-absolute; never write `download/` or `releases/` | 8.1, 2.3 |
| Plan 11 | Python version (3.10 with the `tomli` fallback, D39); no Java validators (validators dropped) | 3.4, 4.1 |
| Plan 11 | Privacy page mentions the storage key | 6.8, 12.1 |
| Plan 11 | Generated `docs/README.md` | 8.8 |
| D32 | No site in packages; packages install `docs/`; `dh_compress -X.md -Xllms.txt`; Windows zip; release checks `docs/` current and committed | 16, 18.3 |
| D33 | Site in top-level `website/`, ignored, removed by `make tidy`, out of the Docker context; `docs/` only tracked Markdown | 2.1, 2.3, 2.4, 3.6, 3.10 |
| D34 | Notebooks committed with outputs; builder converts only; no `make tutorials`, cache or kernels; dummy run once in step 5; demo without DeprecationWarning | 13, 4.1, 3.5 |
| D35 | Cutover inside step 5: moves first, deletions and updates last, checks as acceptance; no edits in two places | 18, 19, 6.6 |
| D36 | Landing text in the template; publications TOML with a plain key check; `llms.txt` summary in the AI page front matter | 7.1, 7.2, 6.1, 8.8 |
| D37 | AI page links the stub `SKILL.md`; no zip | 14, 6.8, 8.8 |
| D38 | Tools from conda-forge or Ubuntu packages, no pip, venv or requirement files; one presence check incl. `pandoc`, no `env`; formulas by one pandoc call, its warnings stop the build; HTML may vary with tool versions | 4, 3.4, 3.5, 3.9, 3.11, 11, 17 |
| D39 | Two builders: `docs_builder/` (standard library, Python 3.10 with `tomli`, Linux and Windows, in the source package, line-based docs path, MATLAB help script) and `website_builder/` (tools, templates, static files, publications, figures, tests; imports `qdocs`; `export-ignore`; `make website` message on package contents); docs check in `make release`, the deb containers and `release-win`; same check result everywhere; work packages re-cut | 1.1, 2.1, 2.2, 2.4, 3.1 to 3.5, 3.9, 4.1, 5.2, 5.4, 5.6, 5.7, 6.2, 6.4, 15.1, 16, 17, 18.1, 18.3, 18.4, 19 |
