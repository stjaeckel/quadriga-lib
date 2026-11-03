# Quadriga-Lib website overhaul: specification

This is the specification (step 3 of `md/website_plan.md`). Implementation sub-agents (step 5) build the new documentation builder, templates, static parts and tests from it alone. It refers to the plan by decision number (D1 to D31) and section, and to `md/website_design.md` as "design" with its section number. Step 7 deletes this document; it stays in the git history.

## 1. Scope and conventions

### 1.1 What this spec covers

The new builder in `docs_builder/`, its outputs in `docs/` and `docs/website/`, the page templates, the static parts, the tutorial pipeline with one dummy notebook in `tutorials/`, the agent skill placeholder, the package-install failsafe `docs_builder/CMakeLists.txt`, the builder's tests, the new `GNUmakefile` lines, the `.gitignore` additions, the cutover checklist (D31) and the work packages for step 5. It does not cover the landing-figure scripts and images (D15), the skill content (D10), real tutorials (D11), the deploy script (D29) or any change to files D20 protects.

The code is meant to stay lean: fewer modules, fewer settings, conventions instead of configuration. Where this spec leaves an internal detail open (function names, data structures inside a module), the implementer chooses the simplest form.

### 1.2 Precedence and references

Where texts differ: D24 to D31, then the rest of the plan, then the design, then the prototype in `html_builder/design_prototype/`. Paths in the design and prototype map as follows:

| Design / prototype | This spec |
|--------------------|-----------|
| `html_docu/` (output) | `docs/website/` (built site) and `docs/` (tracked Markdown) |
| `html_builder/` (builder, page sources) | `docs_builder/` |
| `md/cpp/<section>.md` | `docs/api/cpp/<section-slug>.md` (and `mex`, `python`) |
| `index.md` and other twins on every page | twins only as in D26 |
| prototype `css/site.css`, `js/site.js`, `fonts/`, `img/logo.svg` | `docs_builder/static/css/site.css`, `static/js/site.js`, `static/fonts/`, `static/img/logo.svg` |

### 1.3 Terms

| Term | Meaning |
|------|---------|
| Doc block | A `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` comment block in `src/`, `api_mex/` or `api_python/` (D6). |
| Language, `lang` | One of `cpp`, `mex`, `python`; display names "C++", "MATLAB / Octave", "Python". |
| Function | One `/*!MD` block. Its anchor is its name exactly as on today's pages (section 9). |
| Family | A MATLAB/Python function pair matched by file stem, plus its C++ counterpart(s) (section 9.4). |
| Fragment | A list of markdown-it-py block tokens after the builder's own rules ran (section 5.6). |
| Twin | The Markdown version of an HTML page (D26). |
| Gap | A content placeholder written with the gap marker (section 6.4), a missing landing figure or a missing preview image (section 12). |
| Build variable | A value filled in at build time, written `{{ name }}` (section 6.3). |
| Site root | `docs/website/`; URLs in this spec are relative to it unless absolute. |
| Repository root | The folder holding `CMakeLists.txt`; default is the parent of `docs_builder/`. |
| Structural error | A defect that stops the build with file and line (D7). |
| Warning | A defect that is printed and does not stop the build (D7). |

## 2. File tree

### 2.1 New folders and files

"Tracked" means committed by the maintainer (D19); "ignored" means listed in `.gitignore` (section 2.4); "generated" means written by the builder; "hand-made" means written in step 5 or by the maintainer.

| Path | Kind | Purpose |
|------|------|---------|
| `docs/` | tracked, generated | Markdown tree (D8, D24). Fully owned by the builder: files it does not write are deleted (section 3.6), except `docs/website/`. |
| `docs/website/` | ignored, generated | Built site and deploy root (D24). Fully owned by the builder. |
| `docs_builder/build.py` | tracked | Entry point (section 3). Standard library only at import time. |
| `docs_builder/qdocs/` | tracked | Builder package (section 2.2). |
| `docs_builder/templates/` | tracked | Jinja2 templates (section 8). |
| `docs_builder/site.toml` | tracked | Site URLs and page list (section 3.3). |
| `docs_builder/content/` | tracked | Prose pages (`*.md`), `landing.toml`, `publications.toml`, API intros, `api_matching.toml` (sections 6, 7, 9.4). |
| `docs_builder/static/` | tracked | Files copied unchanged into the site root (section 12). |
| `docs_builder/skill/quadriga-lib/SKILL.md` | tracked | Skill placeholder (section 14). |
| `docs_builder/tests/` | tracked | Test suite and the fixture repository `tests/fixtures/mini_repo/` (section 17). |
| `docs_builder/pytest.ini` | tracked | pytest settings (section 17). |
| `docs_builder/requirements.txt`, `requirements-test.txt`, `requirements-tutorials.txt` | tracked | Python packages, minimum versions (section 4). The whole toolchain is Python; there is no Node.js (D4, D5 as amended). |
| `docs_builder/CMakeLists.txt` | tracked | Package install and failsafe (section 16). |
| `docs_builder/README.md` | tracked | Section "Set up a build machine" in step 5 (section 4.3); the rest (make targets, content formats, failsafe, cutover checklist) in step 7. |
| `docs_builder/.cache/` | ignored, generated | Tutorial cache (section 13.4). |
| `tutorials/` | tracked | Notebooks without outputs (D28); one dummy notebook `tutorials/pipeline_demo.ipynb` (section 13.6). |

Nothing is installed into the working copy (D4 as amended): the tool environments with their packages live outside the repository (section 4.2). The old builder, `html_docu/` and every file D20 protects stay unchanged. Outside the three new folders, step 5 changes only `GNUmakefile` (section 3.5) and `.gitignore` (section 2.4).

Naming rule: no file in the new folders may be named `build*.log`, `test*.txt` or `deploy.py`, or sit in a folder named `release`, `dist` or `build`. These names match unanchored patterns of the existing `.gitignore` at any depth and would be silently ignored.

### 2.2 Builder package `docs_builder/qdocs/`

A guide, not a contract; the implementer may merge further where it keeps the code simpler.

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Argument parsing, command dispatch, diagnostics (section 3.7), output writing and check mode (section 3.6). |
| `config.py` | `site.toml`, page conventions (section 3.3), library and Armadillo versions (section 3.8), the small schema checker (section 7.1). |
| `env.py` | Tool presence check and the `env` command (section 3.4). |
| `docblocks.py` | Block extraction, block grammar, parameter lists, content model (section 5). |
| `mdparse.py` | markdown-it set-up and the builder's rules (section 5.6). |
| `links.py` | Link targets and URLs per output location, `[[...]]` resolution, ids, legacy hashes, GitHub slugs, cross-language matching (sections 8.7, 9). |
| `prose.py` | Prose pages, front matter, build variables, landing and publications data, gap markers and the gap list (sections 6, 7). |
| `render_md.py` | Markdown output: API files, twins, `llms.txt`, `docs/README.md`; the mdformat extension (section 8.6 to 8.8). |
| `render_html.py` | HTML output: Jinja2, fragment rendering, span-free code colouring, LaTeX to MathML (sections 8.1 to 8.5, 10, 11). |
| `checks.py` | Link and anchor check, well-formedness, size budgets (section 15). |
| `tutorials.py` | Notebook execution, cache, tutorial pages (section 13). |
| `assets.py` | Static files, hand-made files until the cutover, landing figures, skill zip, stamp (sections 12, 14, 16.2). |

### 2.3 Output files

`<lang>` is `cpp`, `mex` or `python`; `<slug>` is a section slug (section 9.2); `<stem>` is a notebook file stem; `<v>` is the library version.

Tracked Markdown in `docs/` (written by `make docs`):

| Path | Content |
|------|---------|
| `docs/README.md` | Index of `docs/` for GitHub (section 8.8). |
| `docs/llms.txt` | Agent entry point (section 8.8). |
| `docs/quadriga_lib_api_<lang>.md` | Full API per language (section 8.6.2). URLs of D3. |
| `docs/api/<lang>/general-usage-notes.md` | The API intro per language. |
| `docs/api/<lang>/<slug>.md` | One file per API section (section 8.6.3); 10 C++, 9 MATLAB, 9 Python sections today. |
| `docs/download.md`, `docs/formats.md`, `docs/faq.md`, `docs/ai_agents.md`, `docs/publications.md` | Twins of the content pages (D26). |

Site in `docs/website/` (written by `make website`):

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
| `skill/quadriga-lib-skill-<v>.zip` | Skill placeholder zip (section 14). |
| `css/`, `js/`, `fonts/`, `img/`, `favicon.ico`, `.htaccess` | Static parts (section 12). |
| `publications/**`, `quadriga_lib_material_model.md` | Hand-made files, read from `html_docu/` until the cutover (section 12.5). |
| `docs-stamp.txt` | Version stamp for the package failsafe (section 16.2). |

The builder writes nothing below `download/` or `releases/` of the site root (plan section 11): none of its outputs has such a path.

### 2.4 `.gitignore` additions

Append to the root `.gitignore`:

```
/docs/website/
/docs_builder/.cache/
.ipynb_checkpoints/
```

The existing `__pycache__/` line already covers byte-code. The naming rule of section 2.1 keeps the new files clear of the existing unanchored patterns.

## 3. Builder entry point, CLI and make targets

### 3.1 Command line

`python3 docs_builder/build.py <command> [options]`, run from any folder.

| Command | Does |
|---------|------|
| `docs` | Parses all sources, renders the tracked Markdown tree and writes `docs/` (section 3.6). |
| `docs --check` | Same rendering in memory; writes nothing; compares with `docs/` (D23); prints each differing, missing or extra file and the hint of section 3.9. |
| `website` | Runs the `docs` phase (writes `docs/`), then renders the site, runs the checks of section 15 and writes `docs/website/`. |
| `website --release` | The one strict switch (plan section 11): additionally fails on any remaining gap and on tutorials whose cache entry is missing or stale. |
| `tutorials [--force] [STEM ...]` | Executes notebooks whose cache key changed (all with `--force`; only the named ones if stems are given) (section 13). |
| `env` | Checks the whole toolchain and prints one line per tool (section 3.4). |

Option for all commands: `--root DIR` (repository root; used by tests). Exit codes: 0 success (warnings allowed); 1 build failed (structural errors, failed checks, `--check` differences, strict-mode failures, notebook errors); 2 environment or usage problem (wrong Python, missing or too old tool, bad arguments).

### 3.2 Code layout and imports

`build.py` contains no syntax newer than Python 3.6, so that it can print the version message of section 3.4 under an old Python; it then puts its own folder on `sys.path` and calls `qdocs.cli.main()`. Third-party packages are imported inside the functions that need them, so that `make docs` never imports Pygments, Jinja2 or nbclient.

Code and templates are always taken from the folder of `build.py`. Everything else is taken relative to `--root`: sources, `docs_builder/site.toml`, `docs_builder/content/`, `docs_builder/static/`, `docs_builder/skill/`, `tutorials/`, the outputs and the cache.

### 3.3 `docs_builder/site.toml` and page conventions

```toml
[site]
base_url = "http://quadriga-lib.org"                     # no trailing slash; http until HTTPS works (plan section 10)
download_base_url = "http://quadriga-lib.org/download"   # D25
releases_url = "http://quadriga-lib.org/releases/"       # D25
github_url = "https://github.com/stjaeckel/quadriga-lib"

[[page]]
id = "index"
file = "index.html"
tab = "Home"
# ... one entry per page, in tab order
```

Each `[[page]]` has `id`, `file`, optional `tab` (label in the tab row and menu; pages without `tab` are not in them) and optional `twin` (path of the twin; pages without it have no twin, D26). List order is tab order. Initial list: `index` (Home), `download` (Get started, twin `download.md`), `cpp_api` (C++ API, twin `quadriga_lib_api_cpp.md`), `mex_api` (MATLAB API, twin `quadriga_lib_api_mex.md`), `python_api` (Python API, twin `quadriga_lib_api_python.md`), `tutorials` (Tutorials, twin `tutorials.md`), `formats` (Formats, twin `formats.md`), `publications` (Publications, twin `publications.md`), `ai_agents` (AI agents, twin `ai_agents.md`), `faq` (FAQ, twin `faq.md`), `impressum`, `privacy`, `404`; `file` is `<id>.html`. Unknown or missing keys are structural errors (section 7.1).

Conventions in `config.py`, not settings:

| Page id | Template | Source |
|---------|----------|--------|
| `index` | `landing.html` | `content/landing.toml` |
| `cpp_api`, `mex_api`, `python_api` | `api.html` | doc blocks of the language, intro `content/api_intro_<lang>.md` |
| `publications` | `publications.html` | `content/publications.toml` |
| `tutorials` | `tutorials_index.html` | `content/tutorials.md` and the notebooks |
| `impressum`, `privacy` | `legal.html` | `content/<id>.md` |
| `404` | `notfound.html` | `content/404.md` |
| any other id | `text.html` | `content/<id>.md` |

Source paths are constants in `config.py` (`src/`, `api_mex/`, `api_python/`, `include/quadriga_lib.hpp`, `CMakeLists.txt`). The tutorial twins are website only (D28): a coded exception keeps `tutorials.md` and the tutorial pages out of `docs/`. One tutorial page per notebook is added by the builder (section 13). The header menu, tab row and footer are written in the templates with a helper `url(page_id)` (section 8.2).

### 3.4 Tool check and the `env` command

The toolchain check lives in the builder, not in CMake: `docs_builder/CMakeLists.txt` runs on the packaging machines (Docker, Windows), which have no website tools; it checks only the built site's stamp (section 16).

Each command checks, before it reads any source, that everything it needs is present (importable), collects every missing tool, prints all of them in one run, one message per tool, and exits 2 without a traceback (D21). Build commands check presence only; version comparisons are the job of `env`.

| Command | Checked up front (present) |
|---------|----------------------------|
| `docs`, `docs --check` | Python 3.11+; `markdown-it-py`, `mdit-py-plugins`, `mdformat`, `mdformat-gfm` |
| `website` | the `docs` set; `Pygments`, `Jinja2`, `latex2mathml` |
| `tutorials` | the `website` set; the packages of `requirements-tutorials.txt`; the kernels the notebooks name (a missing kernel is a warning, section 13.3) |

`build.py env` (make target `docs-env`) checks the whole toolchain: every package of the three requirement files and the kernels the notebooks name. It also compares installed versions (`importlib.metadata.version`) with the minimums in the requirement files, as tuples of their leading integers. It prints one line per tool with found version (or "missing"), minimum and status (ok, missing, too old, optional), then a summary. Exit 2 if anything that `docs` or `website` needs is missing or too old; tutorial packages and kernels are reported but optional.

Each message must contain the tool, "missing" (or for `env` the found version and the minimum), the Python in use (`sys.executable`), and the remedy: activate the tool environment, or the exact `pip install -r ...` command of section 4.3, with a pointer to "Set up a build machine" in `docs_builder/README.md`. No message starts an installation; a build never downloads anything (D4 as amended).

### 3.5 `GNUmakefile` lines

`GNUmakefile` is a lean wrapper around CMake; the builder adds only one-line callers, after the `documentation` target, which stays unchanged (D21, D31):

```make
# Documentation builder; set-up and other commands: docs_builder/README.md
docs:
	python3 docs_builder/build.py docs
docs-check:
	python3 docs_builder/build.py docs --check
website:
	python3 docs_builder/build.py website
tutorials:
	python3 docs_builder/build.py tutorials
docs-env:
	python3 docs_builder/build.py env
.PHONY: docs docs-check website tutorials docs-env
```

And two lines at the end of the existing `tidy` recipe (section 3.10). Everything else is called directly (`build.py website --release`, `python3 -m pytest docs_builder/tests`). The tools are those of the active environment: `python3` from `PATH`; another Python is used by calling `build.py` with it. `make website` runs the `docs` phase itself in one process, which is how it "depends on make docs" (D21); `make tutorials` never runs as part of `docs` or `website` (D28).

### 3.6 Output handling

- Every command builds the complete output as an in-memory mapping from relative path to bytes. Iteration over files, sections, functions and pages is always sorted.
- No file is written while any structural error exists; errors from all inputs are collected first.
- `docs`: writes each file whose bytes differ from the file on disk, then deletes every file under `docs/` not in the mapping (except `docs/website/`) and removes empty folders. `--check` writes and deletes nothing.
- `website`: deletes `docs/website/docs-stamp.txt` first; renders the site into memory; runs the checks of section 15 on the in-memory mapping; only if they pass, writes changed files, deletes files not in the mapping, removes empty folders, and finally writes `docs-stamp.txt`. A failed build leaves the previous site without its stamp, so the package failsafe (section 16) refuses it.
- All text files are UTF-8 without BOM with `\n` line ends and one final newline.

### 3.7 Diagnostics

One per line: `<path>:<line>: error: <message>` or `<path>:<line>: warning: <message> [<kind>]` (`:<line>` omitted when there is none). Every warning is printed, each with a short fixed kind id (for example `param-separator`, `param-item`, `fence-language`, `output-type`), so the later doc-block touch-up (plan section 7) can grep for them. `docs` and `website` print the full gap list (section 6.4). Last line: `<command>: N errors, M warnings, K gaps`.

### 3.8 Versions read once

`config.py` reads once per run the library version from `include/quadriga_lib.hpp` (`#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"`, the expression the old builder and `pyproject.toml` use) and the Armadillo version from `CMakeLists.txt` (`set\(armadillo_version\s+"([^"]+)"\)`). A missing file or no match is a structural error.

### 3.9 Byte-identical output

The same inputs and the same tool versions give byte-identical `docs/` and `docs/website/` trees (plan 5.1): sorted listings; no dates, user names or absolute paths in any output; the skill zip as in section 14. Tool versions are never written into outputs. Because the tools have minimum versions only (section 4), an upgrade may change the output without a source change; when `docs --check` finds differences it prints the fixed hint `If the sources did not change, a tool upgrade may be the cause; regenerate with make docs.` (D23, D4 as amended).

### 3.10 What the builder leaves in the working copy, `clean` and `tidy`

The builder writes `docs/` (tracked), `docs/website/` and `docs_builder/.cache/`; running the builder or the tests leaves `__pycache__/` below `docs_builder/`; editing notebooks in Jupyter leaves `.ipynb_checkpoints/`; after the cutover the early check in the `deb` recipe leaves `build_docs_check/` (section 16.3). Tests write only below pytest's `tmp_path`.

`make tidy` restores the working copy to the state of a clean clone (D4 as amended); `make clean` keeps `docs/website/` and the cache. No `git clean`: it would also delete the maintainer's ignored local files, such as the deploy script. Step 5 appends to the existing `tidy` recipe, which changes nothing it does today:

```make
	- rm -rf docs/website docs_builder/.cache
	- find docs_builder tutorials \( -name __pycache__ -o -name .ipynb_checkpoints \) -type d -prune -exec rm -rf {} +
```

`build_docs_check/` falls under the existing `rm -rf build*` of `tidy`. The tool environments are outside the repository and never touched.

### 3.11 Speed

Time targets on the idle build machine (1-minute load below 4), measured as wall and CPU time with the load average recorded before and after: `make docs` under 5 s; `make website` under 15 s (with the `docs` phase and the checks, without notebook execution, formulas included); `make tutorials` with every cache entry current under 5 s. The prototypes measured about 2.5 s for the `docs` work and about 4 s for the `website` work, most of it mdformat (about 1.7 s per pass over all doc blocks); converting 1,091 formulas to MathML took 0.12 s. Rules that keep the targets:

- One Python process per command; the sources are read and parsed once; both renderers use the same token lists.
- mdformat runs once per fragment, not once per output file: link URLs, which differ between the full API file and the section file, are rendered as placeholder strings and filled in per file.
- No subprocess in `docs` and `website`; no network access.
- `website` reads tutorials from the cache and never executes a notebook.
- Files are compared before they are written; the checks parse each output once.

## 4. Dependencies and tool set-up

### 4.1 Packages

The tools come from PyPI, not from Ubuntu packages, with minimum versions only, so that they can be upgraded; no exact pins (D4 as amended). Python 3.11 or newer (`tomllib`); no Node.js (D5 as amended). The minimums below were run with the prototypes and are the lowest versions known to work.

`docs_builder/requirements.txt`:

```
markdown-it-py>=3.0.0
mdit-py-plugins>=0.4.2
mdformat>=1.0.0
mdformat-gfm>=1.0.0
Pygments>=2.17.2
Jinja2>=3.1.2
latex2mathml>=3.81.1
```

`make docs` needs only the first four (D21). `requirements-test.txt`: `pytest>=8.0`. `requirements-tutorials.txt`: `nbformat>=5.10`, `nbclient>=0.10`, `jupyter-client>=8.0`, `ipykernel>=6.29`, `matplotlib>=3.8`, `numpy>=1.26`, `quadriga-lib>=0.12.2` (the released pip package, D28). `latex2mathml` 3.81.1 is the version tested (section 11); older versions give different MathML for about a tenth of the test formulas.

### 4.2 The two tool set-ups

Both must work (D4 as amended); both are part of the build machine's set-up, done once per machine, outside the repository.

| | Conda (preferred) | Virtual environment |
|---|-------------------|---------------------|
| Environment | conda env `quadriga-web` from conda-forge: Python and pip | venv at a path the user chooses, default `~/.venvs/quadriga-web` |
| Use | `conda activate quadriga-web` | `source ~/.venvs/quadriga-web/bin/activate` |
| Files in the repository | none | none |

The tutorial kernel `python3` is the one of the active environment.

### 4.3 README section "Set up a build machine"

Written in step 5 as the first section of `docs_builder/README.md`; step 7 adds the rest of that README, and the mention of the new targets in the top-level `README.md` stays a step-7 item. A manual, once-per-machine procedure, run from the repository root:

1. Prerequisites: conda (Miniconda or Miniforge), or a system Python 3.11+.
2. Conda set-up (preferred):

```bash
conda create -y --name quadriga-web -c conda-forge "python>=3.11" pip
conda activate quadriga-web
python -m pip install -r docs_builder/requirements.txt -r docs_builder/requirements-test.txt -r docs_builder/requirements-tutorials.txt
```

3. Venv set-up:

```bash
python3 -m venv ~/.venvs/quadriga-web
source ~/.venvs/quadriga-web/bin/activate
python -m pip install -r docs_builder/requirements.txt -r docs_builder/requirements-test.txt
# optional: tutorials
python -m pip install -r docs_builder/requirements-tutorials.txt
```

4. Activate the environment before each build session.
5. Verify with `make docs-env`: every tool for `docs` and `website` must be ok (it also compares versions with the minimums).
6. Upgrade: repeat the install lines with `pip install -U`; then run `python3 docs_builder/build.py tutorials --force` (the tutorial cache key does not see tool or builder versions, section 13.4), and expect `make docs-check` to show differences that come from the new versions (section 3.9).
7. Nothing is installed into the working copy; removing the environment removes the tools.

## 5. Doc-block parser

### 5.1 Files read

For each language every `<folder>/*.cpp` directly in `src/`, `api_mex/` or `api_python/` (not recursive, as today), sorted, read as UTF-8. Today 75, 85 and 80 files contain blocks; no other file type does.

### 5.2 Block grammar and structural errors

A block starts with a line that is exactly `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` and ends with the next line that is exactly `SECTION!*/`, `SECTION_DESC!*/` or `MD!*/` of the same kind. Structural errors (file and line):

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

Other malformed content (stray raw HTML, odd headings) is caught by the HTML well-formedness check (section 15.2).

### 5.3 Parts of an `/*!MD` block

Split at the source-line level, outside code fences: name (line 1), brief (line 2), intro (up to the first sub-block heading), sub-blocks. A sub-block heading is a line starting with `## `; its label is the heading text with a trailing `<br>`, then a trailing `:` or `;`, removed (variants today: `## Inputs:`, `## Attributes:<br>`, `## Function 'duplicate';`).

| Kind | Labels | Rendering |
|------|--------|-----------|
| `code` | `Declaration`, `Declarations`, `Usage`, `Example`, `Examples` | Fragment; fences without info string use the page language. |
| `params` | `Input`, `Inputs`, `Output`, `Outputs`, `Attributes`, `Arguments`, `Input Arguments`, `Output Argument`, `Output Arguments`, `Optional outputs`, and labels starting with `Inputs (`, `Outputs (` or `In/out` | Parameter lists (section 5.4). |
| `plain` | everything else | Fragment. |

Required sub-blocks (D7): C++ blocks need `Declaration`, `Declarations` or `Usage`, except class blocks (blocks with an `Attributes` sub-block: `arrayant`, `channel`, `path`); MATLAB and Python blocks need `Usage`. The current sources satisfy this rule (zero errors).

Display name: the name without `<++>`; on the MATLAB page lower-cased (as today, design open point 7).

### 5.4 Parameter lists

In a `params` sub-block, each top-level bullet item is tested on its first source line (`NAME` is `` `[^`]+`(?:, (?:\.\.\.|`[^`]+`))* ``):

```
^- (?P<names>\*\*NAME\*\*(?:(?:, | / )\*\*NAME\*\*)*)(?:\s+(?P<qual>\*\([^)]*\)\*|\([^)]*\)))?(?P<sep>\s+[—–-]\s+|<br>\s*$|\s*$)
```

- A matching item is a parameter row: names as written (for example `` `Dx`, `Dy`, `Dz`, `*D` `` or `` `V_re` / `V_im` ``), qualifier (text in the parentheses: `optional`, `in/out`, `optional, in/out`, ...; empty if absent) and description (the rest of the item, including continuation lines, nested lists and tables).
- ` — ` is the SPEC separator; ` - ` and ` – ` give the warning `param-separator`; `<br>` or nothing means the description starts on the next line.
- A non-matching item is a note row across both columns, with the warning `param-item`.
- Other blocks in a `params` sub-block (paragraphs, tables) render in place.

Expected on today's sources: about 17 `param-separator` and 25 `param-item` warnings; the pattern covers every variant of plan section 11 (`*(optional)*`, `(optional)`, `*(in/out)*`, several names).

### 5.5 Differences between the languages

| | C++ | MATLAB | Python |
|---|-----|--------|--------|
| Folder | `src/` | `api_mex/` | `api_python/` |
| Display names | as written; member functions start with `.` | lower case | as written; may contain `.` (`sos.acf_2d`) |
| Language of fences without info | `cpp` | `matlab` | `python` |
| Required sub-block | `Declaration`/`Declarations`/`Usage` (not for class blocks) | `Usage` | `Usage` |
| File stem for families | — | `<stem>.cpp` | `qpy_<stem>.cpp` |

### 5.6 markdown-it set-up and the builder's rules (`mdparse.py`)

One factory for three kinds: `docblock`, `prose`, `notebook`. All: `MarkdownIt("commonmark").enable("table")` with `dollarmath_plugin(allow_space=False, allow_digits=False, double_inline=True)` (D5; `$5 and $6` stays text) and the gap rule (section 6.4). `html` is on for doc blocks only; in prose and notebooks raw HTML is not interpreted and shows as text (D27: hand-written HTML is not used for content). The builder's rules:

| Rule | Kinds | Effect |
|------|-------|--------|
| `wikilink` | docblock, and the API intros (prose plus this rule, with the language of the page) | `[[name]]` outside code becomes a link to an API target; a `.` directly before `[[` makes it the member form `.name` (as in the old builder). |
| `raw_a` | docblock | `<a href="U" ...>text</a>` becomes a link to `U`; `target` and `rel` are dropped, so every link opens in the same tab. |
| `raw_br` | docblock | `<br>` is kept, except trailing `<br>` at the end of a paragraph or heading (the old pages used them as spacing). |
| `heading_attrs` | prose, notebook | A trailing `{...}` on a heading sets its id and legacy ids (section 6.2) and is removed from the text. |
| `note` | prose | `container_plugin` for `note` and `warning` (section 6.5). |
| `site_url` | all | Links starting with `http://` or `https://` plus `quadriga-lib.org/` (also `www.`) become site-relative, so the absolute links in doc blocks (plan section 10) resolve locally and in twins. URLs that start with `download_base_url` or `releases_url` stay absolute and count as external: the builder never writes those folders (section 2.3). |

These rules never touch code spans or fences. Token line maps plus the part's first source line give the line numbers.

### 5.7 Content model

Leaf content is a fragment (token list, source file, first line). A function has language, display name (= anchor), source and line, brief, intro, sub-blocks (label, kind, content; `params` content as rows with names, qualifier, description, note flag) and family. A section has language, title, slug, legacy id, descriptions (several `SECTION_DESC` blocks in file-name order) and functions. An API page has the intro (a prose page) and its sections. Prose pages, landing, publications and tutorials have their own small classes. One parser and one model feed both renderers; neither re-parses source text (plan 5.1).

Ordering: sections by title, functions by `(name.startswith("."), name)` on the display name (as today); the general usage notes come first.

## 6. Prose pages

Formats, FAQ, Get started, the AI page, the legal pages, the tutorial index text, the 404 sentence and the three API intros (D27), each one Markdown file in `docs_builder/content/`.

### 6.1 File format

The file starts with TOML front matter between two lines that are exactly `+++`; the Markdown body follows. Keys: `title` (required; `h1`, `<title>` "`title` · Quadriga-Lib", `og:title`, twin heading), `description` (required; meta and `og:description`), `lead` (optional, inline Markdown; lead paragraph on the page and in the twin), `dummy_note` (only `tutorials.md`; section 13.5). Unknown keys, missing required keys and wrong types are structural errors (section 7.1). Body: CommonMark with tables, math, note boxes, heading ids, build variables and gap markers; headings from `##` to `####` (the `h1` comes from `title`).

### 6.2 Heading ids

Every `##` and `###` heading gets an id: the slug of its text (section 9.2), unique within the page by appending `-2`, `-3`. An explicit id is written at the end of the heading: `### How do I cite Quadriga-Lib? {#cite}`. Legacy ids (old anchors, D3) are added with `legacy="..."`, repeatable: `## Material Properties {#material-properties legacy="9f6922f9"}`. In HTML the id goes on the heading, each legacy id on an empty `<span id="...">` as its first child. In twins the braces are removed and anchors are GitHub slugs (section 9.5).

### 6.3 Build variables

`{{ name }}` in prose bodies, front matter and TOML text fields (not in doc blocks or notebooks), substituted before parsing, also inside code: `version`, `armadillo_version` (section 3.8), `site_base_url`, `download_base_url`, `releases_url`, `github_url` (from `[site]`). An unknown name is a structural error. Build variables are values; gap markers are placeholders; the two syntaxes differ on purpose (plan section 11).

### 6.4 Gap markers

One marker for all sources (D27): `%%GAP: text%%` on one line. `text` describes missing content (`%%GAP: performance numbers and hardware%%`) or is a draft awaiting approval (`%%GAP: Realistic radio channels ...%%`); removing the marker is the approval.

- It may appear in prose files and their front matter, `landing.toml`, `publications.toml`, doc blocks and notebook Markdown cells.
- HTML: `<span class="ph">text</span>` in body text; plain `text` in `<title>`, meta, `alt` and `og:*`.
- Twins keep the marker verbatim.
- The gap list: all markers found in the sources (`path:line: text`, sorted), plus missing landing figures (section 12.4) and a missing `img/og-card.png` (section 12.3). `docs` and `website` print it in full; `website --release` fails if it is not empty.

### 6.5 Note boxes

```
::: note Optional title
Body Markdown.
:::
```

`note` or `warning`. HTML: `<div class="note-box">` or `<div class="note-box warn">` with `<b>title</b>` first (default "Note" or "Warning"). Twin: a GitHub alert, `> [!NOTE]` or `> [!WARNING]`, then `> **title**` and the body.

### 6.6 One-time conversion of the old prose sources

Step 5 converts `html_builder/formats/*.md`, `faq/*.md` and `download/*.md` into `content/formats.md`, `faq.md` and `download.md`. The old files stay until the cutover; until then an edit to these pages is made in both places (D24).

- Each old `/*!SECTION` becomes an `##` heading with an explicit id and the old hash as legacy id: formats "QuaDRiGa Array Antenna Exchange Format (QDANT)" `6cab4884`, "Material Properties" `9f6922f9`; FAQ "About the Library" `e63d3e38`, "Bug Reports and Contact" `59d38498`, "Distribution License" `89373af1`, "Supporting the Project" `d8094d72`; download "Download" `7c9ba8cc`, "Installation from Binaries" `7fc9f392`, "Installation from Source" `904e3979`, "Quick Start" `fdc96a45`.
- `/*!SECTION_DESC` text becomes the section body; each `/*!MD` block becomes a `###` heading with its brief and body, its `## X:` sub-headings becoming `#### X`. The formats entry ids `<arrayant>` and `<layout>` are kept as legacy ids. The old ids with spaces (`1. Material Table`, `2. Built-in EM Materials`, `3. Principled BSDF`, `4. References`) and the ids `…` and `1` are dropped: they are not valid HTML ids or are meaningless, and nothing links to them (D3).
- FAQ and download: each bold-labelled item (`- **How do I cite Quadriga-Lib?**<br>`) becomes a `###` heading with its text as body; the cite question gets the id `cite` (plan section 10). The quick-start labels (`**Python**`, `**MATLAB / Octave**`, `**C++**`) become `###` headings and their fences get `python`, `matlab`, `cpp`; the compile line becomes a `bash` fence with `{{ armadillo_version }}` instead of `15.6.2`.
- Fences with shell commands get the info string `bash`; other unlabelled fences stay uncoloured (section 10).
- Raw HTML becomes Markdown: links to `[T](U)` without `target` and `rel`; `<br>` removed; entities to characters (`<` in text escaped as `\<`; `[` as `\[`).
- `download.md.in`: `__VERSION__` becomes `{{ version }}`; `download/<file>` links become `{{ download_base_url }}/<file>`; `releases/` becomes `{{ releases_url }}`.
- Section order: formats as in the prototype (QDANT first); FAQ and download in the order of the old pages, which sorted sections by title (`html_builder/extract_html.py`).
- Front matter: titles "Data formats", "Questions", "Get started"; descriptions and leads from the prototype where it has them, gap-marked (design section 9).

### 6.7 API intro files

`content/api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md`: `title` "C++ API", "MATLAB / Octave API", "Python API" (design open point 7); `description` and `lead` from the prototype's C++ page adapted per language, with `{{ version }}`, gap-marked. Body: the "General usage notes" list of `get_preamble()` in `html_builder/extract_md.py`, with the example `*_3dB` added after `*_deg` in the C++ units line (as in `md/SPEC_CPP_API.md`) and `[arrayant_generate](#arrayant_generate)` written as `[[arrayant_generate]]` (intros are parsed as prose plus the `wikilink` rule of their language, section 5.6). The Python body is one gap marker: `%%GAP: General usage notes for the Python API (import, numpy conventions, index base, errors)%%` (plan section 11). The notes become the first section of each API page (`sec-general-usage-notes`) and `docs/api/<lang>/general-usage-notes.md`.

### 6.8 Other content files written in step 5

| File | Initial content |
|------|-----------------|
| `content/ai_agents.md` | The prototype's draft text, with the file table adapted to section 2.3 paths, the skill link `skill/quadriga-lib-skill-{{ version }}.zip`, and a `::: warning Placeholder` box saying the skill is a placeholder (plan 5.4); first line `%%GAP: whole page text is a draft (design section 9)%%`. |
| `content/impressum.md` | `::: warning Placeholder` with `%%GAP: Impressum text from the maintainer (D17); no address in git until settled%%`. |
| `content/privacy.md` | `::: warning Placeholder` with `%%GAP: privacy text: the hoster and its server logs; no cookies, no analytics, no third-party requests; the browser stores the chosen code language under the localStorage key qlib-lang, only after a click (D18)%%`. |
| `content/tutorials.md` | `lead` and `dummy_note` from the prototype, gap-marked. |
| `content/404.md` | `title` and `description` "Page not found"; the prototype sentence, gap-marked. |

## 7. Structured pages

### 7.1 Schema checker

A schema is a nested dict: per key its type (`str`, `int`, `bool`, `table`, list of either), whether it is required, whether strings allow inline Markdown, and for lists an exact or minimum count. The checker reports every problem as `<file>: error: <key path>: <problem>` (for example `why.card[2].title: missing`). Inline-Markdown strings are rendered with `renderInline` of the prose parser; other strings are escaped. Used for `site.toml`, front matter, `landing.toml`, `publications.toml` and notebook metadata.

### 7.2 `content/landing.toml`

Section ids are fixed in the template: `why`, `paths`, `quickstart` (D3 keeps `index.html#quickstart`), `apis`, `covers`, `open`.

| Key | Type | Inline MD | Notes |
|-----|------|-----------|-------|
| `meta.title`, `meta.description` | str | no | `<title>`, `og:title`, meta |
| `llms.summary` | str | no | the summary paragraph of `llms.txt` (section 8.8) |
| `pitch.headline`, `pitch.lead` | str | yes | `h1`, lead |
| `pitch.primary`, `pitch.secondary` | `{label, href}` | no | filled and outlined button |
| `pitch.proof` | list[str], exactly 3 | yes | proof lines |
| `pitch.figure` | str | no | id of a `figure` entry |
| `why.heading`, `why.intro` | str | no / yes | |
| `why.card` | exactly 3 of `{title, text}` | text | |
| `paths.heading` | str | no | visually hidden `h2` |
| `paths.card` | exactly 2 of `{label, href, text}` | text | template appends " →" to the label |
| `quickstart.heading`, `quickstart.intro`, `quickstart.note` | str | no / yes / yes | |
| `quickstart.install` | str | no | `bash` code block |
| `quickstart.python`, `.matlab`, `.cpp` | multi-line str | no | the three language tabs; one final newline stripped |
| `quickstart.figure` | str | no | id of a `figure` entry |
| `apis.heading`, `apis.intro` | str | no / yes | |
| `apis.card`, `open.card` | exactly 3 of `{label, href, text}` | text | |
| `covers.heading`, `covers.intro` | str | no / yes | |
| `covers.feature` | at least 1 of `{title, text, badge?}` | text | `badge` only `"experimental"` (D12) |
| `open.heading`, `open.intro` | str | no / yes | |
| `figure` | at least 1 of `{id, base, alt, caption}` | caption | files `<base>-light.webp`, `<base>-dark.webp` (section 12.4) |

All keys are required except `badge`. Step 5 fills the file from the prototype `index.html`: every design-section-9 gap and every `<span class="ph">` becomes a gap marker; figures `coverage` (`img/landing/coverage`, pitch) and `antenna` (`img/landing/antenna`, quick start) with the prototype's alt texts and captions; the "For team leads" card keeps `href = "#covers"` (D30); `llms.summary` is a gap-marked draft from the landing lead.

### 7.3 `content/publications.toml`

`page.title`, `page.description`, `page.lead` (inline MD, links `faq.html#cite`); `group` (at least 1, in display order) with `id`, `title`, optional `intro` and `outro` (inline MD) and `entry` (at least 1, in display order: newest first, as written) with `year` (int), `title`, `authors`, `venue` and `link` (at least 1 of `{label, url}`; `url` absolute or site-relative such as `publications/x.pdf`). Step 5 fills it from `html_builder/parts/publications.html.part` in the grouping of the prototype (groups `quadriga-lib` and `quadriga-background`); only `page.description` is a gap.

## 8. Renderers and templates

### 8.1 Jinja2 environment

`Environment(loader=FileSystemLoader(templates), autoescape=True, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, undefined=StrictUndefined)`. Fragments are passed as `Markup`. Templates emit compact markup for repeated structures (index lists, parameter lists, pills), as in the prototype. Every internal URL is `{{ root }}<path>`: `root` is `""` for pages in the site root, `"../"` for `tutorials/<stem>.html`, `"/"` for `404.html` (root-absolute, plan section 11).

### 8.2 Templates

| File | Used by | Structure (prototype file) |
|------|---------|----------------------------|
| `base.html` | all | head (section 8.3), skip link, header, `<main id="content">`, footer (`index.html`) |
| `_header.html` | base | menu `<details class="menu">` with the groups of design 4.1, inline logo, name, version, GitHub link; tab row from the page list |
| `_footer.html` | base | the four columns of design 4.2, bottom line "Quadriga-Lib v`version`", Impressum, Privacy; "This page as Markdown" only on pages with a twin (D26) |
| `_logo.svg` | header, 404 | the logo inline, `aria-hidden="true" focusable="false"` |
| `landing.html` | landing | design 5.1 (`index.html`) |
| `api.html`, `_api_index.html`, `_fn_entry.html` | API pages | section 8.4 (`cpp_api.html`) |
| `text.html` | text pages | breadcrumb, `h1`, lead, "Markdown version" link, index, prose (`formats.html`, `ai_agents.html`) |
| `legal.html` | Impressum, privacy | breadcrumb, `h1`, prose; no index (`impressum.html`) |
| `tutorials_index.html`, `tutorial.html` | tutorials | section 13.5 (`tutorials.html`, `tutorial_sample.html`) |
| `publications.html` | publications | groups with entry lists (`publications.html`) |
| `notfound.html` | 404 | centred logo 96 px, `h1`, sentence, buttons Home, C++ API, MATLAB API, Python API (design 5.8) |

Menu groups, footer columns and their labels are written in the templates (design 4.1 and 4.2) with the helper `url(page_id)`, which returns `root` plus the page's `file`; external links use the `[site]` URLs. Breadcrumbs: "Home › `title`"; "Home › Tutorials › `title`" on tutorial pages; "Home › API reference › C++" (and "MATLAB / Octave", "Python") with "API reference" linking to `index.html#apis`. The current tab and menu entry get `aria-current="page"`; tutorial pages mark "Tutorials". Text pages: the index lists the `h2` headings; an `h2` with `h3` headings below it becomes an open `<details>` with "Overview" and the `h3` links; the list appears in `nav.doc-side.idx` (`aria-label="Page sections"`) and in `details.doc-index` ("On this page"); a page with fewer than two `h2` headings has no index. Other templates follow their prototype files with all texts from the data files; the landing language tabs keep the prototype structure (`div.ltabs[data-lang-tabs]`, radios `name="qs"`, values `python`, `matlab`, `cpp`, Python checked). Inline `style` attributes of the prototype become classes in `site.css`.

### 8.3 Head of every page (design 4.3)

`<!doctype html>`, `<html lang="en" class="no-js">`, charset, viewport, `<title>`, description, `theme-color` `#2e7d46` (light) and `#24603a` (dark) written literally in `base.html`, `og:type`, `og:title`, `og:description`, `og:url` (absolute), and, only if `static/img/og-card.png` exists, `og:image` (absolute) and `twitter:card` `summary_large_image` (section 12.3); icons `img/favicon.svg`, `img/favicon-32.png`, `img/apple-touch-icon.png`; `<link rel="alternate" type="text/markdown">` only on pages with a twin (D26); preload of `fonts/ibm-plex-sans-latin-400-normal.woff2`; `css/site.css`; the inline `document.documentElement.className = "js"`; `js/site.js` with `defer`.

### 8.4 API page

Follows the prototype `cpp_api.html` and design 5.2:

- Page head: breadcrumb; `h1` and lead from the intro file; "Same API in" links to the other two pages; "Markdown version" to `quadriga_lib_api_<lang>.md`.
- Index (`nav.doc-side.idx`, `aria-label="<display> API index"`; copy in `details.doc-index` "Sections and functions"): the js-only filter input; "General usage notes"; per section a closed `<details>` with title and `<span class="count">`, then "Section overview" and one `<li class="fn">` per function; below, links to the other API pages and the full Markdown file.
- Section: `<section class="api-section" id="sec-<slug>"><h2 id="<legacy hash>">title</h2><div class="section-desc">...</div>` and its entries; the general usage notes as `sec-general-usage-notes` without legacy id.
- Entry: `<article class="fn-entry" id="<anchor>">`; `fn-head` with `h3` name and `#` anchor link; `fn-langs` with "Also in" and three pills (C++, MATLAB, Python): the current one `aria-current="page"`, a counterpart as link to `<lang>_api.html#<anchor>`, a missing one `<span class="pill none" title="No MATLAB counterpart">MATLAB –</span>`; then "Markdown" to `api/<lang>/<slug>.md`; `p.fn-brief`; intro; per sub-block `<h4>label</h4>` and content.
- Parameters: `<dl class="params">`, one `<div>` per row with `<dt>names<span class="opt">qualifier</span></dt><dd>description</dd>`; a note row is `<div class="pnote"><dd>...</dd></div>`.

### 8.5 Fragment rendering to HTML

markdown-it's renderer with overrides: fences through the span-free colouring (section 10) as `<pre><code class="language-<lang>">`; tables wrapped in `<div class="table-wrap">`; link URLs from `links.py` for the output location; math converted to MathML (section 11), block math in `<div class="math-block">`; note boxes (section 6.5); gap spans; heading ids (section 6.2).

### 8.6 Markdown output

#### 8.6.1 Renderer and twin layout

Fragments are rendered with `mdformat.renderer.MDRenderer` (options `wrap: keep`, `number: false`, `end_of_line: lf`, the `tables` extension) plus a small extension with renderers for math (`$...$`, `$$` blocks), note boxes (GitHub alerts) and gap markers (verbatim). Structure around the fragments is written directly. Parameter rows are written in the SPEC form `` - **`name`** *(qualifier)* — description ``; note rows as their item. Sub-block labels become `### Label`; fences get the language.

Every twin starts with `# title`, an empty line, and `Quadriga-Lib <version> · Web page: <absolute URL>` (for a section file `<base_url>/<lang>_api.html#sec-<slug>`); then, if the page has at least two `##` headings, an index table (D26) of its `##` headings (with a `Topics` column of `###` links on prose pages); then the lead and the body. Exception: the full API files.

#### 8.6.2 Full API files `docs/quadriga_lib_api_<lang>.md`

The old layout, without build date and without the "Line" column (plan section 11): YAML front matter with `title` ("C++ API Documentation for Quadriga-Lib v`version`" and the MATLAB / Octave and Python forms), `author`, `lang`; `# General usage notes`; `# Function Index` with the columns Function, Section, Summary (the brief); then per section `---`, `# Section title`, description, and per function `---`, `## name`, brief, an "Also in:" line with links to existing counterparts (omitted when there are none), intro and the sub-blocks.

#### 8.6.3 Per-section files `docs/api/<lang>/<slug>.md`

Twin header with the section title, index table `| Function | Summary |`, description, the functions as in the full file. The largest (C++ "Site-specific simulation tools", about 128 kB) is accepted: a section is the unit an agent asks for (plan section 11).

### 8.7 Links

#### 8.7.1 Link targets

Every link becomes a target; each renderer asks for its URL at its output location.

| Kind | Example in a source | HTML in root | HTML in `tutorials/` | Twin in `docs/` |
|------|---------------------|--------------|----------------------|-----------------|
| API function | `[[x]]` in a block; `cpp_api.html#x` in prose | `cpp_api.html#x` (same page: `#x`) | `../cpp_api.html#x` | full file: `#<gh-slug>`; section file: `#<gh-slug>` or `<slug>.md#<gh-slug>`; prose twin: `api/<lang>/<slug>.md#<gh-slug>` |
| Page with twin | `faq.html#cite` | `faq.html#cite` | `../faq.html#cite` | twin path plus the GitHub slug of the heading with that id (`faq.md#how-do-i-cite-quadriga-lib`) |
| Page without twin | `index.html#quickstart` | relative | relative | absolute: `<base_url>/index.html#quickstart` |
| Twin file | `quadriga_lib_api_cpp.md` | relative | relative | relative |
| Site file outside `docs/` | `publications/x.pdf`, `quadriga_lib_material_model.md`, `skill/...zip` | relative | relative | absolute: `<base_url>/<path>` (plan section 11) |
| External | `https://...` | unchanged | unchanged | unchanged |
| Same-page anchor | `#files` | unchanged | unchanged | GitHub slug of the target heading |
| "Also in" (twins only, section 8.6.2) | counterpart in language X | — | — | full file: `quadriga_lib_api_<X>.md#<gh-slug>`; section file: `../<X>/<slug>.md#<gh-slug>` |

Links in sources are written relative to the site root. Doc blocks link only within their own API page (`[[name]]`; `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`); their raw links point to site pages such as `formats.html#6cab4884`, site files or external URLs. A prose link to an API page with a function anchor (for example `cpp_api.html#obj_file_read` on the formats page) is the kind "API function". A wrong anchor in any site-relative link is caught by the link check (section 15.1).

#### 8.7.2 Resolving `[[name]]`

`[[name]]` in a block or intro of language L links to the function of L whose display name equals `name` (MATLAB: compared in lower case); `.[[name]]` links to the member function `.name`. Nothing else is tried; an unresolved link is E-link (D7). Zero unresolved links on the current tree (637 C++, 387 MATLAB, 354 Python).

### 8.8 `llms.txt` and `docs/README.md`

One function writes both, with absolute links (`base_url`) for `llms.txt`, which is fetched on its own, and relative links for `docs/README.md`, which is read on GitHub (D9, plan 5.4). Format of llmstxt.org, shown for `llms.txt` (`<base>` is `base_url`; `docs/README.md` has the same lines with relative links):

```
# Quadriga-Lib

> <landing.toml llms.summary>

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
3. `content/api_matching.toml` lists the C++ counterparts that names cannot find, and overrides rule 2 for its families. Its entries are the C++ functions the family's wrappers call to carry out their operation (not helper calls for validation, sizing or set-up). The first listed function is the target of the C++ pill on the MATLAB and Python pages (member function before `_multi`); the order in the file is the rule.

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

GitHub makes heading anchors itself; twins use them, never HTML ids. `gh_slug()` follows github-slugger: lower-case the heading's plain text; keep characters of Unicode categories `L*`, `M*`, `N*`, `Pc`, plus `-` and space; spaces become `-`; a repeated slug in the same file gets `-1`, `-2`, ... in document order. Test vectors: `.calc_directivity_dBi` → `calc_directivity_dbi`; `sos.acf_2d` → `sosacf_2d`; "How do I cite Quadriga-Lib?" → `how-do-i-cite-quadriga-lib`; "Miscellaneous / Tools" → `miscellaneous--tools`; the second "Inputs" → `inputs-1`.

### 9.6 URL inventory (D3)

Kept: `index.html#quickstart`; `cpp_api.html`, `mex_api.html`, `python_api.html` with every function anchor (137, 85, 80 today) and section hash (10, 9, 9); `formats.html` with `#6cab4884`, `#9f6922f9`, `#<arrayant>`, `#<layout>`; `faq.html` and `download.html` with their four hashes each; `publications.html`; `/quadriga_lib_api_{cpp,mex,python}.md`; `/quadriga_lib_material_model.md`; `/publications/...`. Dropped (D3 as amended): the `quadriga-lib_*` layout ids of the old pages, the formats ids of section 6.6, `style2.css`, `quadriga-lib-logo80.png`, `quadriga-lib-q_letter64.png`. New: `ai_agents.html`, `tutorials.html`, `tutorials/...`, `impressum.html`, `privacy.html`, `404.html`, the twins, `api/...`, `llms.txt`, `skill/...`.

## 10. Code colouring

`render_html.py` has its own small formatter over `pygments.lex()` (no `HtmlFormatter`). Each token type maps to the first matching entry; tokens without an entry produce plain escaped text and no `<span>` (design open point 4); adjacent tokens of one class share one `<span>`.

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
- Output: MathML, converted at build time by the pure-Python package `latex2mathml` (D5 as amended): `latex2mathml.converter.convert(tex, display="block")` for `$$...$$`, `display="inline"` for `$...$`; the result is a `<math>` element inserted as is (block formulas inside `<div class="math-block">`). All browsers of plan 5.2 render MathML Core; no math CSS beyond the prototype's `math { font-family: "Latin Modern Math", "STIX Two Math", "Cambria Math", math; }` and no math fonts are shipped; glyphs come from the device's math font.
- Evaluated on 1,091 formulas (the material model, the four publication Markdown files and the tutorial formulas): none raises; the elements it emits are all MathML Core; the visible text equals KaTeX's MathML for all but 12 formulas, which differ in accents (`\bar` as a combining macron), `‖` for `\|` and two failures below; bold, script and double-struck letters come out as Unicode math letters (𝐇, 𝒩, ℂ), which render in every browser, where KaTeX's `mathvariant` attributes are ignored by Chrome. Rendered side by side with KaTeX's MathML in the browser pane, both look alike.
- The converter does not raise on unknown commands or unbalanced braces; it passes them through. So the builder treats as a structural error at the formula's source file and line (D7): unbalanced `{` `}` in the source, any backslash left in the text of the MathML output (an unsupported command), and any exception of the converter. Found in the evaluation: `\arg` (unsupported; write `\operatorname{arg}`) and `\_` inside `\text{}`. Environments the converter does not handle (`aligned`, `gather`, `gathered`, `eqnarray`, `multline`, `alignat`; tested) are a structural error when `\begin{<name>}` appears in a formula; `align`, `split`, `array`, the matrix environments, `cases` and `\tag` work.
- Twins keep the `$...$` source (GitHub renders it, D8).

## 12. Static parts

### 12.1 `docs_builder/static/`

Copied unchanged into the site root (files named `README.md` are skipped):

| Path | Source |
|------|--------|
| `css/site.css` | Port of the prototype `css/site.css` (section 12.2) |
| `js/site.js` | Port of the prototype `js/site.js`: copy buttons, remembered language (`localStorage` key `qlib-lang`, written only in the `change` handler of the language tabs, D18), function filter with `/`, current-function highlight (design 7); selectors match section 8 |
| `fonts/` | The seven `woff2` files and `LICENSE.txt` of the prototype (committed, not from npm; plan section 11) |
| `img/logo.svg` | The prototype logo (`fill="currentColor"`) |
| `img/favicon.svg` | The logo with `<style>svg { color: #2e7d46 } @media (prefers-color-scheme: dark) { svg { color: #7fd197 } }</style>` |
| `img/favicon-32.png`, `img/apple-touch-icon.png` (180 × 180), `favicon.ico` | Rendered once from `img/favicon.svg` (Inkscape, ImageMagick) and committed (design open point 5) |
| `img/landing/README.md` | One line: the figures of D15 go here, with the file names of design section 8 |
| `.htaccess` | Section 12.6 |

### 12.2 `site.css`

A port of the prototype: every token value, breakpoint, size and colour stays (design 3). Changes: remove `.search`; code colours for the classes of section 10; `.params > .pnote` spans both columns; `.prose h4` and `.note-box > * + *` spacing; print also hides `.doc-side` and `.idx-filter`; the prototype's inline `style` attributes become classes (`.page`, `.page-narrow` and spacing rules with the prototype's values); `.ph` stays for gap markers.

### 12.3 Preview image (design open point 5)

`og:image` and `twitter:card` are emitted only if `static/img/og-card.png` exists; otherwise the gap list names it. The image is made in the figure session (design section 8).

### 12.4 Landing figures

For each `figure` entry: if `static/<base>-light.webp` and `-dark.webp` exist, `<figure><div class="frame"><picture><source media="(prefers-color-scheme: dark)" srcset="<base>-dark.webp"><img src="<base>-light.webp" width="1600" height="1200" alt="..." decoding="async"></picture></div><figcaption>...</figcaption></figure>`; otherwise the same frame, empty, with the alt text as a centred label, and the gap list names the missing files. Every figure except the first on the page gets `loading="lazy"`. The figure scripts go into `docs_builder/figures/` in the figure session (D15, D24); they write the WebP files into `static/img/landing/`, where the images are committed.

### 12.5 Hand-made files until the cutover

`assets.py` copies, as constants until the cutover, `html_docu/publications/` to `publications/` and `html_docu/quadriga_lib_material_model.md` to `quadriga_lib_material_model.md` (D24); a missing source is a structural error. At the cutover the files move to `static/` and the constants go (section 18).

### 12.6 `.htaccess`

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

## 13. Tutorial pipeline (D28, plan 5.3)

### 13.1 Notebooks

`tutorials/<stem>.ipynb`, `<stem>` matching `[a-z0-9][a-z0-9_-]*`, stored without outputs (every code cell has no outputs and no execution count; otherwise a structural error telling the author to clear outputs). The builder never rewrites notebooks.

### 13.2 Metadata

Notebook-level `metadata.quadriga_tutorial`, schema-checked: `title` (str), `summary` (str, inline MD), `level` (`beginner`, `intermediate`, `advanced`), `duration_min` (int), `order` (int; index order, then title), `dummy` (bool, optional). Language and kernel from `metadata.kernelspec` (`name`; `language`: `python` → "Python", `matlab` → "MATLAB", `octave` → "Octave"). Cell-level `metadata.quadriga_tutorial.alt` is the alt text of that cell's images; missing gives the warning `figure-alt` and "Figure output of cell <n>".

### 13.3 Running (`make tutorials`)

For each notebook, sorted: if a cache entry with the current key exists and `--force` is not given, skip. If the kernel is not installed (`jupyter_client.kernelspec`), skip with a warning (D28: MATLAB notebooks run only where a kernel exists). Otherwise execute with `nbclient.NotebookClient(nb, kernel_name=..., timeout=600)` in an empty work folder in the cache. A cell error is an error naming the cell and the exception; the old entry is deleted and the run continues, exiting 1 at the end. Entries of removed notebooks are deleted. The run uses the released pip package of the active environment (D28).

### 13.4 Cache

`docs_builder/.cache/tutorials/<stem>/` with `key.txt`, `fragment.html`, `page.md` and the figures. Key: SHA-256 of the notebook bytes, the installed quadriga-lib version (`importlib.metadata.version("quadriga-lib")`, the released package the notebook runs with) and the kernel name. The key does not see the builder's code or other tools: after a tool upgrade or a change of the tutorial rendering, run `tutorials --force` (section 4.3).

### 13.5 Output

- Markdown cells: rendered like prose (notebook kind; a `#` heading is rendered as `h2`), heading ids as section 6.2.
- Code cells: `<div class="cell"><div class="cell-label">In [n]</div><pre><code class="language-<lang>">...</code></pre>` and one `<div class="out">` per output.
- Output types: `stream` and `text/plain` as `<pre>` (ANSI codes removed); `image/png` and `image/svg+xml` written to `<stem>/fig-<cell>-<output>.<ext>` and shown as `<img alt loading="lazy">`. Any other type falls back to its `text/plain` with the warning `output-type`.
- `page.md` (the twin, D26): twin header (web page `<base_url>/tutorials/<stem>.html`), a line with the two downloads, Markdown cells verbatim, code cells as fenced blocks, text outputs as `text` fences, images as `![alt](<stem>/fig-<cell>-<output>.<ext>)`. Twin and page sit side by side, so one relative image path works in both (D28).
- Interface: `tutorials.load_tutorials(root)` returns one entry per notebook with a current cache entry: `stem`, `meta` (title, summary, level, duration_min, order, dummy, language), `fragment` (HTML), `page_md` (the twin text) and `files` (relative path → bytes: the notebook copy and the figures). Package A ships `tutorials.py` as a stub returning `[]`; package D replaces it. `render_html.py` (B) renders `tutorial.html` with `root` (`"../"`), `tut` (the entry) and `fragment`, and `tutorials_index.html` with `page` (front matter and lead of `content/tutorials.md`), `tutorials` (the entries, sorted) and `show_dummy_note`; `assets.py` (B) writes `page_md` and `files`; `render_md.py` (A) writes the index twin `tutorials.md`.
- `website` uses the cache entry whose key matches; otherwise the notebook is left out with the warning `tutorial-missing` ("run make tutorials"), an error with `--release`. Without any entry the index shows "No tutorials yet." (plan 5.3). Per included notebook: `tutorials/<stem>.html` (template around `fragment.html`), `tutorials/<stem>.md`, `tutorials/<stem>.ipynb` (the source notebook, D28), the figures. Metadata for the page and the index is read from the notebook. Downloads: "Download notebook .ipynb" and "Markdown .md". The index (`tutorials.html`, twin `tutorials.md`, both website only) shows cards in `order` with language tag, level tag, title and summary; if every included notebook is a dummy, `dummy_note` is shown as a warning note box.

### 13.6 The dummy notebook

`tutorials/pipeline_demo.ipynb`, kernel `python3`, metadata `title` "Pipeline demo", `summary` "Placeholder notebook that tests the tutorial pipeline.", `level` beginner, `duration_min` 5, `order` 1, `dummy` true. Cells (fixed seed): (1) Markdown `## Set-up` and a sentence; (2) code: `import numpy as np`, `import quadriga_lib`, `rng = np.random.default_rng(1)`, `print(quadriga_lib.version())`; (3) Markdown `## An antenna` with `$D = 10 \log_{10}(\max G)$`; (4) code: `ant = quadriga_lib.arrayant.generate('3gpp', M=2, N=2, freq=3.7e9)` and `print(round(float(quadriga_lib.arrayant.calc_directivity(ant, 0)), 2))`; (5) code with `alt` "Line plot of 50 random values": a matplotlib plot of `rng.standard_normal(50)` with `%matplotlib inline`; (6) Markdown with a three-row table and `$$\sigma^2 = \frac{1}{N}\sum_{n=1}^{N} (x_n - \bar x)^2$$`. Step 5 checks the calls against quadriga-lib 0.12.2.

## 14. Agent skill placeholder (D10)

`docs_builder/skill/quadriga-lib/SKILL.md`:

```
---
name: quadriga-lib
description: Placeholder for the Quadriga-Lib agent skill (radio channel modelling library with C++, MATLAB/Octave and Python APIs). Until the skill exists, read {{ site_base_url }}/llms.txt.
---

# Quadriga-Lib skill (placeholder)

This skill is a placeholder. The documentation for agents starts at {{ site_base_url }}/llms.txt.
```

The placeholder is not a gap: the site may go live with it (plan 5.4 asks for a note on the AI page). `website` packs every file below `skill/` into `skill/quadriga-lib-skill-<version>.zip` with `ZIP_STORED` (bytes independent of the zlib version), sorted entries, `date_time=(1980, 1, 1, 0, 0, 0)`, `external_attr = 0o644 << 16`, `create_system = 3`, after substituting build variables.

## 15. Checks after the build

`website` runs these on the in-memory output (section 3.6); the Markdown link check also runs in `docs`, limited as in section 15.1. A failed check is an error unless stated.

### 15.1 Links and anchors

For every HTML file: every `href`, `src` and `srcset` that is not absolute or `mailto:` is resolved against the file (root-absolute against the site root); the target must exist in the output; a fragment on an HTML target must be an `id` there, on a `.md` target a GitHub slug of a heading there. For every Markdown file the builder generates (twins, API files, `llms.txt`, `docs/README.md`, tutorial twins): every link and image URL, the same way; absolute URLs that start with `base_url` are mapped to the output and checked too. Hand-made files copied into the site (`publications/**`, `quadriga_lib_material_model.md`) are not checked inside; links into them are checked for existence only. URLs that start with `download_base_url` or `releases_url` and all other external URLs are not fetched. In the `docs` phase only links whose target lies in `docs/` are checked; absolute site links and links to files outside `docs/` are checked in `website`. It must catch the case of plan section 10: a FAQ without `cite` breaks the footer link `faq.html#cite` on every page and the twin link to the cite heading.

### 15.2 HTML well-formedness

`html.parser` with a tag stack: every non-void element closed in order; no duplicate `id`; exactly one `h1`; exactly one `<main id="content">`; every `<img>` has `alt`.

### 15.3 Size budgets (warnings)

Raw bytes (section 12.6): API pages 800 kB (the C++ page is expected near 700 kB); other HTML pages 150 kB; landing total 550 kB, counting `index.html`, `site.css`, `site.js`, the font files and, per landing figure that exists, the larger of its two theme variants (a browser loads one). Each exceeded budget is a warning `[size]`.

## 16. Package install and failsafe (D22)

### 16.1 `docs_builder/CMakeLists.txt`

Works standalone (`cmake -S docs_builder`) and, after the cutover, as `add_subdirectory(docs_builder)`:

1. Standalone (`CMAKE_CURRENT_SOURCE_DIR` equals `CMAKE_SOURCE_DIR`): `cmake_minimum_required(VERSION 3.18)`, `project(QuadrigaLibDocs LANGUAGES NONE)`.
2. `option(ENABLE_DOCS_INSTALL "Install the prebuilt documentation from docs/website" OFF)` and the cache string `QDOCS_INSTALL_DIR` (default `share/doc/quadriga-lib/html`). OFF by default, the package builds switch it ON (D22 as amended; section 16.3).
3. If the option is OFF: return. This is the only skip condition (pip builds, the local layout of `make` and users' own installs leave it OFF).
4. Version: `QUADRIGA_VERSION` from the parent, or parsed standalone from `../include/quadriga_lib.hpp` as the top-level file does.
5. Stamp: `../docs/website/docs-stamp.txt`, line `version=<v>`; found version, or `none` if the folder or file is missing.
6. If the found version differs: `message(FATAL_ERROR ...)`. The message must name `docs/website/`, the found and the expected version, `make website` as the fix, and `-DENABLE_DOCS_INSTALL=OFF` as the way to build without documentation.
7. Otherwise `install(DIRECTORY ../docs/website/ DESTINATION ${QDOCS_INSTALL_DIR})`.

The check runs at configure time, before anything is compiled; after the cutover the `add_subdirectory` line sits right after the version block of the top-level file, so it comes seconds after the start (D22).

### 16.2 Stamp file

`docs/website/docs-stamp.txt`, written last by a successful `website` (section 3.6), content `version=<version>\n`; no date, so builds stay byte-identical.

### 16.3 Build paths (applied at the cutover)

| Path | After the cutover |
|------|-------------------|
| Debian package | `GNUmakefile` `deb-*` → `docker build` (`COPY . .` includes the ignored `docs/website/`) → `debian/rules`. Build 1 (gcc13, installs the docs) gets `-DENABLE_DOCS_INSTALL=ON`; build 2 (MATLAB MEX only) stays OFF. |
| Early fail before Docker | First line of the `deb` recipe: `cmake -S docs_builder -B build_docs_check -DENABLE_DOCS_INSTALL=ON` (configure only), then the existing `$(MAKE) -j deb-...`. One line, no new target; a direct `make deb-jammy` is still protected by the check inside the container, only later. |
| Release | `release:` without prerequisites, with the recipe `python3 docs_builder/build.py docs --check`, `python3 docs_builder/build.py website --release`, `$(MAKE) package`, `$(MAKE) deb`, `$(MAKE) pip`: the checks run before the source package is made. |
| Windows package | `release-win` starts with `cmake -S docs_builder -B $(CMAKE_BUILD_DIR)\docs $(GENERATOR) -D ENABLE_DOCS_INSTALL=ON -D QDOCS_INSTALL_DIR=documentation -D CMAKE_INSTALL_PREFIX=%CD%\$(DIST_DIR)` and, after the library install, runs `cmake --install $(CMAKE_BUILD_DIR)\docs`; the `xcopy html_docu` line goes. Today `all` configures the local layout and `release-win` installs that build with `--prefix`, so a check in the top-level project would never run there. The site goes to `documentation\` at the top of the zip: the FHS path means nothing on Windows, and the name is not confused with the repository's `docs/`. |
| pip, `make` (local layout), users' system installs | option OFF: no check, no documentation install, as with pip today (D22). |

### 16.4 Test

`test_cmake.py` (skipped without `cmake`) copies `docs_builder/CMakeLists.txt` into a temporary tree with `include/quadriga_lib.hpp` (version `1.2.3`) and `docs/website/`, and configures with `-DENABLE_DOCS_INSTALL=ON -DCMAKE_INSTALL_PREFIX=<tmp>/inst` unless stated: (a) no site: configure fails naming `none` and `1.2.3`; (b) stamp `version=1.2.2`: fails naming `1.2.2`; (c) stamp `version=1.2.3` and `index.html`: succeeds, and `cmake --install` puts both into `<tmp>/inst/share/doc/quadriga-lib/html/`; (d) no site, option not given: succeeds, installs nothing. The top-level `CMakeLists.txt` is not touched before the cutover (D22).

## 17. Tests

pytest; run with `python3 -m pytest docs_builder/tests` in the active environment. `docs_builder/pytest.ini` sets `testpaths = tests`, `addopts = -p no:cacheprovider` and the markers `cmake`, `kernel` (tests that need cmake or the python3 kernel skip themselves without them) and `repo` (reads the real repository). The suite runs in under two minutes without `kernel` tests, needs no network and writes only below `tmp_path`. `tests/conftest.py` provides `mini_repo(tmp_path)`, a copy of `tests/fixtures/mini_repo/` (a minimal repository: header with version `9.9.9`, `CMakeLists.txt` with `set(armadillo_version "1.0.0")`, two or three small blocks per language, a reduced `site.toml` and content), and `run_build(root, *args)`, which calls `qdocs.cli.main` with `--root` and returns exit code and output. Rendering tests assert substrings of the `mini_repo` output, not whole files; for real data the tracked `docs/` plus `docs --check` is the reference output.

| File | Package | Covers |
|------|---------|--------|
| `test_cli.py` | A | Smoke: `docs` on `mini_repo` exits 0; a structural error gives 1 and writes nothing; a `website` run with a fixture formula starts no subprocess (checked by patching `subprocess`) |
| `test_env.py` | A | All missing tools reported in one run, exit 2 before reading sources; the per-command sets; build commands do not compare versions; `env` compares versions and gives the right lines and exit codes (2 for a `docs` or `website` tool missing or too old, 0 when only optional tools are missing) |
| `test_output.py` | A | Only changed files written; stale files removed; `docs` leaves `docs/website/` alone; `--check` writes nothing, exits 1 on a changed, missing or extra file and prints the hint; stamp deleted first and written last |
| `test_docblocks.py` | A | Line numbers; each error id of section 5.2 (one small fixture each); sub-block kinds; required sub-blocks with the class-block exemption; parameter pattern variants of section 5.4, note rows, warnings |
| `test_mdparse.py` | A | `wikilink` with member form; `raw_a`; `raw_br`; raw HTML in prose shown as text; `heading_attrs`; `site_url`; `$5 and $6` not math; code untouched |
| `test_links.py` | A | `[[...]]` resolution and E-link; URLs of each link kind per output location; `slug`, legacy hashes (section 9.3 values), `gh_slug` vectors; E-anchor; families, mapping order and mapping errors |
| `test_prose.py` | A | Front matter and schema errors with key paths; build variables and the unknown-name error; note boxes; heading ids; gap markers in HTML, meta and twins; gap list and `website --release` failing on gaps |
| `test_render_md.py` | A | Substrings of the full API file (front matter without date, index without "Line", "Also in"), a section file, a prose twin (index table, alert, braces removed, GitHub-slug links), `llms.txt` and `docs/README.md` (absolute and relative links) |
| `test_render_html.py` | B | Substrings: head tags (alternate link and Markdown footer link only with a twin, `og:image` only with the image), tabs with `aria-current`, function entry, parameter rows, pills, section and legacy ids, text-page index, note boxes, gap spans, span-free code colouring; `404.html` root-absolute; tutorial pages with `../` |
| `test_math.py` | B | Inline and block MathML (`display` attribute); a formula with an unknown command, one with unbalanced braces: structural errors at their source lines; twins keep `$...$` |
| `test_checks.py` | A | Missing file, missing id, missing Markdown slug, absolute `base_url` link in a twin, the `faq.html#cite` case in HTML and Markdown; unclosed tag, duplicate id, two `h1`, `img` without `alt`; size warnings |
| `test_content.py` | C | (`repo`) the real content parses; `cite` and the legacy ids of sections 6.6 and 9.3 exist; landing and publications pass their schemas |
| `test_real_repo.py` | C | (`repo`) `docs` on the real repository into a temporary root: zero errors; 137, 85, 80 functions; 10, 9, 9 sections; every id of `fixtures/old_anchors.json` (captured once from `html_docu/*.html` without the dropped ids of section 9.6) and `index.html#quickstart` exist; two `website` runs give identical bytes; `docs --check` passes after `docs` |
| `test_tutorials.py` | D | Metadata errors; notebook with outputs rejected; cache key changes; `website` without cache (warning, no cards) and with a fake entry (page, twin, notebook, figures, card, dummy note); stale entry and `--release`; output conversion from a fixture executed notebook; (`kernel`) `pipeline_demo.ipynb` end to end |
| `test_cmake.py` | D | (`cmake`) cases (a) to (d) of section 16.4 |

## 18. Cutover checklist (D31)

Done after go-live and the maintainer's sign-off, not in step 5. Step 7 copies this section into `docs_builder/README.md`.

### 18.1 Files that move (`git mv`)

1. `html_docu/publications/` → `docs_builder/static/publications/`.
2. `html_docu/quadriga_lib_material_model.md` → `docs_builder/static/quadriga_lib_material_model.md`.
3. `html_builder/extract_matlab_comments.py` → `docs_builder/extract_matlab_comments.py`; it keeps to the standard library and Python 3.10 (deb build in a jammy container, CMake on Windows) and does not import `qdocs`.
4. `html_builder/img/*` → `docs_builder/logo/` (logo sources, not deployed).
5. Then delete the two hand-made-file constants in `assets.py` (section 12.5).

### 18.2 Files that are deleted

`html_builder/` and `html_docu/` (all remaining files).

### 18.3 Files that change

| File | Find | Change |
|------|------|--------|
| `CMakeLists.txt` | the version block; `INSTALL_DOCDIR`; `install(DIRECTORY html_docu/ ...)`; the `mex_documentation` command | Add `add_subdirectory(docs_builder)` after the version block; delete both `set(INSTALL_DOCDIR ...)` lines and the `html_docu` install line; point the help script path to `docs_builder/extract_matlab_comments.py`. |
| `GNUmakefile` | `HB = html_builder`, the `documentation` target, `deb:`, `release:`, `package_dirty:` | Delete `HB` and `documentation`; the `deb` and `release` recipes as in section 16.3; in `package_dirty` replace the `html_docu/releases` exclusion with `docs_builder/.cache`. |
| `Makefile` (Windows) | `release-win`, the `xcopy html_docu` line | As in section 16.3. |
| `debian/rules` | `override_dh_auto_configure`, build 1 | Add `-DENABLE_DOCS_INSTALL=ON` to build 1. |
| `Dockerfile.ubuntu`, `debian/stamp-changelog.sh` | — | No change; check after the first deb build that `share/doc/quadriga-lib/html/index.html` is in the package listing. |
| `.dockerignore` | `html_docu/releases/`, `html_docu/publications/` | Replace with `docs/website/publications/` (keeps the PDFs out of the deb, as today), `docs_builder/.cache/`, `docs_builder/static/publications/`. |
| `.gitattributes` | `html_docu/publications/ export-ignore` | `docs_builder/static/publications/ export-ignore`. |
| `.gitignore` | `html_docu/releases/` | Delete. |
| `pyproject.toml` | — | Check only: pip builds install no documentation (option OFF). Optional `sdist.exclude` for `docs_builder/static/publications/`. |
| `README.md` | make-target table, repository-layout table | `make documentation` becomes `make docs`, `make website`, `make tutorials`; the `html_docu/` and `html_builder/` rows become `docs/`, `docs_builder/`, `tutorials/`. |
| `README_win.md` | `html_docu/` in the package contents | `documentation/`. |
| `md/SPEC_CPP_API.md` | "the pages in `html_docu/`"; `html_docu/quadriga_lib_api_cpp.md`; `make documentation`; the "Line" column advice and "may lag behind" | "the pages of quadriga-lib.org"; `docs/quadriga_lib_api_cpp.md`; `make docs`; read the function index of the full file, then open only the section file `docs/api/cpp/<section>.md`; regenerated by hand, may lag behind `main` (D8); document the doc-block extensions of D6: `$...$` and `$$...$$` math and the gap marker `%%GAP: text%%`. |
| `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` | `make documentation`, `html_docu/quadriga_lib_api_*.md`, "may lag behind", the help script path | As for the C++ file; script path `docs_builder/extract_matlab_comments.py`. |
| `md/BUG_SCAN.md` | "A stale `html_docu/` alone" | "Stale generated Markdown in `docs/` alone" (`make docs-check` shows it). |
| `md/acoustic_modeling_guidelines.md`, `md/acoustic_engine_notes.md`, `md/materials_kickoff.md`, `md/materials_manager.md`, `md/gbsm_integration_handoff.md` | `html_docu/quadriga_lib_material_model.md`, `make documentation` | `docs_builder/static/quadriga_lib_material_model.md`, `make docs`; working notes may instead be left or deleted. |
| `tools/update_armadillo.py` | `html_builder/download/quick_start.md` in `VERSION_FILES` | Delete the entry and its comment: the Get started page takes the Armadillo version from `CMakeLists.txt`. |
| `tools/acoustic/check_docs.py` | `PAGE`, `FORMATS` | `docs_builder/static/quadriga_lib_material_model.md`, `docs_builder/content/formats.md`; run it once, its parser may need a small change for the new headings. |
| `tools/acoustic/README.md`, `band_probe.cpp`, `gzlib.py`, `paris_cf.py`, `txlib.py`, `tools/sos_port/SOS_PAPER_REPRODUCTION.md`, `tests/catch2_tests/test_ieee_indoor_publication.cpp` | comments naming `html_docu/...` | Point to `docs_builder/static/...` or the URL on quadriga-lib.org. |

### 18.4 Checks after the cutover and at go-live

After the cutover: `make docs-check`, `build.py website --release` and the tests pass; `cmake -S . -B build_check -DENABLE_DOCS_INSTALL=ON -DCMAKE_INSTALL_PREFIX=/tmp/qd_inst` fails without `docs/website/` and succeeds after `make website`, and `cmake --install` puts the site into `share/doc/quadriga-lib/html/`; a MEX build still writes the `.m` help files; a deb contains `share/doc/quadriga-lib/html/index.html` and a Windows package `documentation\index.html`; `make deb` with a stale stamp stops at its first line; `git grep -n 'html_docu\|html_builder'` finds nothing outside `docs_builder/README.md`.

At go-live, after the first deploy (plan section 7): `curl -sI <base>/formats.md` shows `text/markdown; charset=utf-8`; `curl -sI <base>/llms.txt` shows `text/plain; charset=utf-8`; a missing page returns 404 with the new page; compression test: add `AddOutputFilterByType DEFLATE text/html text/css application/javascript text/markdown text/plain` to the server's `.htaccess`; if `curl -sI -H 'Accept-Encoding: gzip' <base>/cpp_api.html` then shows `Content-Encoding: gzip` and a plain request still works, the line goes into `static/.htaccess`, otherwise it is removed; `build.py website --release` passes (no gaps left).

## 19. Work packages for step 5

Each package owns the files listed; a package that needs a change in a file it does not own asks the manager.

| Id | Package | Owns | Depends on |
|----|---------|------|------------|
| A | Core and Markdown output | `build.py`; `qdocs/cli.py`, `config.py`, `env.py`, `docblocks.py`, `mdparse.py`, `links.py`, `prose.py`, `render_md.py`, `checks.py`, the stub `tutorials.py`; `site.toml`; `content/api_matching.toml`; the requirement files; `pytest.ini`; `tests/conftest.py`; `tests/fixtures/mini_repo/`; the tests marked A; the `GNUmakefile` lines and `.gitignore` lines of sections 2.4, 3.5 and 3.10; `docs_builder/README.md` section "Set up a build machine" | none |
| B | HTML output and static parts | `qdocs/render_html.py`, `assets.py`; `templates/`; `static/`; the tests marked B | A |
| C | Content conversion | `content/` except `api_matching.toml`; `tests/fixtures/old_anchors.json`; the tests marked C | A (B for the visual check) |
| D | Tutorials, skill, CMake | `qdocs/tutorials.py` (replacing A's stub); `tutorials/pipeline_demo.ipynb`; `skill/quadriga-lib/SKILL.md`; `docs_builder/CMakeLists.txt`; the tests marked D | A, B |

Order: A; then B, C and the CMake part of D in parallel; then the rest of D.

Acceptance (each package also passes its tests and leaves the others green):

| Id | Done when |
|----|-----------|
| A | `docs`, `docs --check` and `env` work on `mini_repo`; the real doc blocks parse with zero errors and about 17 `param-separator` and 25 `param-item` warnings; no output contains a date; for both set-ups, a fresh set-up following only the README section succeeds, `make docs-env` passes, and after `make tidy`, `git status --porcelain --ignored` shows nothing new in the working copy. |
| B | `make website` builds all pages with link, anchor and well-formedness checks passing and the C++ page within budget; manual: the CSS token values equal the prototype's, and the pages match the prototype at 375, 768 and 1280 px in the browser pane. |
| C | `make docs` on the real repository writes the files of section 2.3 with zero errors and `docs --check` passes right after; the real content passes all schemas; `cite` and the legacy ids exist; every prototype draft and design-section-9 item is a gap; `test_real_repo.py` passes; a plain-text comparison of the old and new formats, FAQ and download pages shows no lost text. |
| D | `make tutorials` runs `pipeline_demo.ipynb`, a second run skips it, and `make website` shows it with figure, table, formula and downloads; without the cache, `make website` passes with the warning; the skill zip is byte-identical across runs; the CMake cases pass. |

The manager's acceptance run after all packages: the full build on the real repository; the time targets of section 3.11 met (`/usr/bin/time`, wall and CPU, three runs each, the 1-minute load average recorded before and after; runs above load 4 repeated or reported as provisional); manual check that `make tidy` after `docs`, `website` and the tests leaves the state of a clean clone (the clean-clone state is checked after `make tidy`, not after a build) and that `make clean` keeps the site and the cache; and a report with these timings, the gap list, the warning counts and the page sizes against the budgets.

## 20. Traceability

Every item of design section 11, plan section 10 and plan section 11, with the section that answers it. This section is deleted after the spec review.

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
| Plan 10 | Paths of the design mapped to D24 | 1.2, 2.3 |
| Plan 10 | FAQ provides `#cite`; link check catches such gaps | 6.6, 15.1, 17 (`test_checks.py`, `test_content.py`) |
| Plan 10 | Page size: budget per API page | 10, 15.3 |
| Plan 10 | Old section anchors keep working | 9.3, 6.6, 9.6, 17 (`test_real_repo.py`) |
| Plan 10 | HTTPS: site base URL, `http://` until SSL, commented redirect, doc-block links stay | 3.3, 12.6, 5.6 (`site_url`) |
| Plan 10 | "For team leads" points into the landing page | 7.2 |
| Plan 10 | Design open points answered using D24 to D31 | the design rows above |
| Plan 11, cutover | `CMakeLists.txt`: install lines, help script path, Python 3.10 and standard library | 18.1, 18.3, 16.1 |
| Plan 11, cutover | `Makefile`, `Dockerfile.ubuntu`, `.dockerignore`, `debian/stamp-changelog.sh`, `.gitattributes`, `GNUmakefile` (`documentation`, `release`, `package_dirty`, `clean`, `tidy`) | 18.3, 16.3, 3.10 |
| Plan 11, cutover | Text naming old paths (SPEC files, `md/BUG_SCAN.md`, `md/acoustic_modeling_guidelines.md`, `md/gbsm_integration_handoff.md`, `README.md`, `README_win.md`, `tools/update_armadillo.py`); further users found | 18.3 |
| Plan 11 | Full API files, no build date, no "Line" column; per-section files; largest section accepted or split | 8.6.2, 8.6.3, 2.3 |
| Plan 11 | Links in twins to files outside `docs/`; raw `<a href>` in doc blocks | 8.7.1, 5.6 |
| Plan 11 | One Markdown intro per language; Python intro is a content gap | 6.7 |
| Plan 11 | Build variables and gap markers with different syntax; strict mode fails on gaps and missing figures | 6.3, 6.4, 3.1, 12.3, 12.4 |
| Plan 11 | Explicit heading ids stripped from twins | 6.2, 9.5 |
| Plan 11 | Fonts committed; MathML output chosen with reason (D5 as amended: Python converter) | 12.1, 11 |
| Plan 11 | Cross-language matching | 9.4 |
| Plan 11 | Doc-block grammar variants | 5.2, 5.3, 5.4, 5.6 |
| Plan 11 | Pygments: uncoloured tokens without `<span>` | 10 |
| Plan 11 | Byte-identical output: sorted lists, fixed zip timestamps, no dates | 3.9, 14, 17 (`test_real_repo.py`) |
| Plan 11 | Server without compression and `.md` type: raw budgets; content types in `.htaccess` | 12.6, 15.3, 18.4 |
| Plan 11 | `404.html` root-absolute; never write `download/` or `releases/` | 8.1, 2.3 |
| Plan 11 | Python 3.11 (`tomllib`); no Java validators (validators dropped) | 4.1 |
| Plan 11 | Privacy page mentions the storage key | 6.8, 12.1 |
| Plan 11 | Generated `docs/README.md` | 8.8 |
