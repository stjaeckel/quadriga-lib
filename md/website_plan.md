# Quadriga-Lib website overhaul: project plan

This plan is the input for every later step of the website overhaul: the design session (step 2), the spec (step 3), the spec review (step 4), the implementation (step 5), the implementation review (step 6) and the cleanup (step 7). It records the goals, the decisions taken in the planning session on 2026-11-02, the requirements, and the process. It is a temporary document: step 7 deletes it, and it stays in the git history.

Sessions that read this plan: the decisions in section 3 are settled. Do not reopen them; if one turns out to be impossible or contradicts another, stop and report it to the maintainer.

## 0. Status

| Step | What | Output | State |
|------|------|--------|-------|
| 1 | Project plan | `md/website_plan.md` | signed off 2026-11-02 |
| 2 | Design session | `md/website_design.md`, `html_builder/design_prototype/` | signed off 2026-11-02 |
| 3 | Spec | `md/website_spec.md` | done 2026-11-02 |
| 3a | Doc-block fixes on `main`, merge of `main` into `website`, re-check (D20) | fixed doc blocks; spec measurements updated | done 2026-11-02: fixed on `main`, merged, re-check passed |
| 4 | Spec review (1 to 3 rounds) | updated spec | round 1 done 2026-11-02 (21 findings fixed); waiting for sign-off |
| 5 | Implementation | builder, static parts, pages | not started |
| 6 | Implementation review (1 to 3 rounds) | fixes | not started |
| 7 | Documentation, cleanup, merge into `main` | `docs_builder/README.md` with the cutover checklist | not started |
| after go-live | Cutover (D31) | old builder and `html_docu/` removed | not started |

The manager session (steps 3 to 7) keeps this table current.

## 1. Goals and audiences

The site at quadriga-lib.org is built from sources by a Python builder (`make documentation`). Its design came from the Armadillo site, looks dated, and does not work on phones. The overhaul replaces the design and the builder, keeps the principle "a builder makes static pages from source files", and adds tutorials and support for AI agents.

Audiences, in order of traffic:

1. **Engineers using the API**, mostly on a desktop. They need fast navigation to a function, its arguments, shapes and units, in C++, MATLAB/Octave or Python.
2. **Decision makers** (team leads, managers, standardisation delegates), often on a phone after a conference pitch. Within one screen they must understand what the library does, why it is credible, and how their team starts. The aim is that they keep the page open and pass it on to their team.
3. **Curious technical people** who want an idea of what the project is about.

The site is for developers and technically versed people, not the general public.

Success criteria:

- The landing page reads well on a 360 px wide phone: no horizontal scrolling, no zooming, the key message and a path for engineers and for decision makers visible without long scrolling.
- An engineer reaches any function of any language in at most two page navigations from any page (opening an index group does not count), and can jump to the same function in another language where one exists.
- The build stops with file and line number on broken doc blocks, instead of silently making wrong pages.
- Every page has a Markdown version for AI agents, and the site explains how to use the library with an AI agent.

## 2. Current state (2026-11-02, commit 4b0cfe4)

- `make documentation` (in `GNUmakefile`) runs `sed` on `html_builder/download/download.md.in`, `extract_version.py` three times (it rewrites line 1 of tracked page starts in `html_builder/parts/`), `extract_html.py` eight times and `extract_md.py` three times.
- `extract_html.py` (about 400 lines) and `extract_md.py` (about 350 lines) convert the doc blocks with regular expressions and contain parallel copies of the same logic. They do not report errors.
- Each HTML page is the shared header, a hand-written page start from `parts/`, content from `/*!SECTION`, `/*!SECTION_DESC` and `/*!MD` blocks, and the shared footer. Pages: `index`, `cpp_api`, `mex_api`, `python_api`, `formats`, `faq`, `publications`, `download`.
- API pages are one long page per language (C++ about 570 kB, about 137 functions; MATLAB and Python about 370 kB each). There is no search and no link between the languages.
- `style2.css` has one breakpoint at 800 px; the header and menu are built from tables and divs. No JavaScript, no external fonts.
- `html_docu/` is tracked in git. Hand-made files in it: `style2.css`, `.htaccess`, logo PNGs, `publications/` (about 13 MB of PDFs, Markdown versions and figures), `quadriga_lib_material_model.md`. `html_docu/releases/` is ignored and lives only on the server side.
- Other users of `html_docu/`: the API SPEC files (`md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`) tell coding sessions to read `html_docu/quadriga_lib_api_*.md`; `README.md`, `README_win.md` (the Windows binary package ships `html_docu/` as offline documentation); `make package` uses `git archive HEAD`, so the source zip contains whatever is tracked.
- `html_builder/extract_matlab_comments.py` is called by CMake (`CMakeLists.txt`, target `mex_documentation`) to make the MATLAB help files. It is not part of `make documentation` and must keep working.
- Deployment: a local, git-ignored script mirrors `html_docu/` and `release/` to the Strato web space by sftp. Strato serves static files with Apache (`.htaccess` is honoured).
- Formulas: the API blocks contain a few one-line formulas as inline code (`sos_calc_mse`, `sos_acf_approx`). The publication Markdown versions and `quadriga_lib_material_model.md` contain LaTeX (`$...$`, `$$...$$`), which the site does not render; they are only linked as raw files.
- There is no Impressum and no privacy page.

## 3. Decisions

| Id | Topic | Decision |
|----|-------|----------|
| D1 | JavaScript | Every page works fully without JavaScript. Small optional scripts may add comfort: search over function names, copy buttons on code, remembering the chosen API language. Nothing may break or disappear when JavaScript is off. No JavaScript from third parties. |
| D2 | API page structure | One page per language, as now, with a navigation sidebar on wide screens and a collapsible section/function index on phones. |
| D3 | URLs | All current website file names and anchors stay valid: `index.html` (including `#quickstart`), `cpp_api.html`, `mex_api.html`, `python_api.html` with every function anchor (for example `cpp_api.html#arrayant_interpolate_multi`), `formats.html`, `faq.html`, `publications.html`, `download.html`, and the hash section anchors of all pages (doc blocks link for example `formats.html#6cab4884`). Old hash anchors are kept as explicit ids next to the new readable ids; meaningless ids of the old pages (for example `id="1"`) are dropped. New pages get new names. Amended by the maintainer on 2026-11-02: four old formats ids with spaces, `style2.css` and the two old logo PNGs are dropped; nothing links to them. The full API Markdown files keep their URLs `/quadriga_lib_api_{cpp,mex,python}.md`. |
| D4 | Build tools | Python 3 with pinned pip packages (for example markdown-it-py, Pygments, Jinja2) at build time (Node.js removed by the D5 amendment). No server-side processing on the web server. Amended by the maintainer on 2026-11-02: the tools come from PyPI (not from system packages), installed either into a conda environment named `quadriga-web` (preferred; everything goes there) or into a virtual environment; both ways must work. The requirement files give minimum versions only, so the tools can be upgraded; an upgrade may change the tracked Markdown, which is then regenerated. The environments (conda or venv) are part of the build machine's setup and live outside the repository; nothing is installed into the working copy, and a build never downloads packages. `make docs` and `make website` run in seconds. `make tidy` restores the working copy to the state of a clean clone, so it also deletes `docs/website/` and the builder's cache; `make clean` keeps both. |
| D5 | Formulas | LaTeX (`$...$`, `$$...$$`) rendered at build time. No math JavaScript in the browser. Amended by the maintainer on 2026-11-02: formulas are converted to MathML by a pure-Python converter instead of KaTeX; browsers draw MathML themselves, so no stylesheet, fonts or Node.js are needed. |
| D6 | Doc-block syntax | The `/*!SECTION`, `/*!SECTION_DESC`, `/*!MD` syntax stays. Small extensions are allowed (for example math, notes) if the SPEC files are updated in the same project. The lean-API rule stays: API blocks hold layout, ranges, units and errors, not derivations. |
| D7 | Builder strictness | Structural errors stop the build with file and line (unclosed code fence, unresolved `[[link]]`, missing name line, missing `## Declaration:` (`## Usage:` in MATLAB and Python blocks) where required, duplicate anchors). Style issues print a warning. The builder has its own test suite. |
| D8 | Generated output in git | Generated Markdown is tracked in `docs/` (D24): the full API files, the per-section API files, the other Markdown twins (D26; tutorials excepted, D28) and `llms.txt`. Git users and agents read an API reference on GitHub (which renders Markdown and `$...$` math) without building anything. It is regenerated by hand with `make docs` (D21, D23); falling behind `main` between regenerations is accepted. The output copies in `docs/website/` (HTML, CSS, fonts, KaTeX output, images) are not tracked; their sources in `docs_builder/static/` are. Until the cutover (D31) `html_docu/` and the old builder stay tracked and working unchanged. |
| D9 | AI agents | The builder writes Markdown twins of the content pages (D26), per-section API Markdown files, the three full API Markdown files, and `llms.txt`. A new page explains how to use the library with AI agents. |
| D10 | Agent skill | A placeholder now: a folder with a stub `SKILL.md`, packed by the builder into a versioned zip that the AI page offers for download. The skill content is designed with the maintainer in a separate session (section 7). |
| D11 | Tutorials | Jupyter notebooks for Python and MATLAB (MathWorks Jupyter MATLAB kernel; Octave kernel possible). Amended by D28: the builder runs the notebooks and makes the fragments itself. This project builds the pipeline, the tutorial index and one dummy notebook that exercises it; writing real tutorials is a follow-up. |
| D12 | Acoustics | Listed as a feature marked "experimental"; its API functions appear as now. Not part of the landing-page pitch. |
| D13 | Publications | Better-looking list with links to IEEE, PDF and Markdown versions, as now. No HTML rendering of publications. |
| D14 | Landing-page claims | Standards work (3GPP TR 38.901, IEEE 802.11 TGn/ac/ax/ah, IEEE 802.11 contributions), lineage and track record (successor of QuaDRiGa), measured performance. Wording and numbers come from the maintainer; nothing is invented. No named users or organisations. |
| D15 | Landing-page visuals | Figures made by scripts that use the released pip package (design section 8 chooses them). The scripts and the images are made in a separate session before going live; step 5 builds only the figure slots with placeholders. The images are committed in `docs_builder/static/img/landing/` (hand-made static files, not generated by the build). |
| D16 | Logo | The general shape of the current logo stays; tweaks and colour changes are allowed. |
| D17 | Legal pages | An Impressum page and a short privacy page, linked from every footer. Their text comes from Markdown files the maintainer fills in. No address goes into git until the maintainer has settled it; until then the Impressum file holds a placeholder and going live is blocked. |
| D18 | Privacy and analytics | No analytics, no cookies, no third-party requests (fonts, scripts, images all self-hosted). The privacy page covers the hoster and its server logs. Storing the chosen API language in the browser happens only after the visitor clicks a language. |
| D19 | Commits | The maintainer makes every commit. Sessions give the `git add` line and the commit message. |
| D20 | Branch and working copy | Branch `website`, working copy `/home/stephan/quadriga-lib-website`, made from `main` at 4b0cfe4. The branch stays on its own: `main` is not merged into it. The branch adds `docs/`, `docs_builder/`, `tutorials/` and the new make targets (D21); it does not change library sources, doc blocks, SPEC files, CMake files, packaging files or the old builder. Touch-ups of the doc-block format, if needed, are a later step. Local only, not pushed until the merge (D31). Amended by the maintainer on 2026-11-02: the doc-block defects that the new builder's checks found are fixed in a separate session on `main`, and `main` is then merged into `website` once, before the spec review; the merged tree is checked again. |
| D21 | Make targets and toolchain | `make docs` builds the tracked Markdown in `docs/` with Python alone (no Node; pip packages only where the spec shows them to be needed). `make website` depends on `make docs` and builds the full site in `docs/website/` with the full toolchain (D4), on the maintainer's Linux machine only. `make tutorials` runs the notebooks (D28). A missing tool stops the build with a plain message naming the tool and how to install it, not with a stack trace. The old `make documentation` stays unchanged until the cutover (D31). |
| D22 | Packages | The website is built only on Linux; no website tooling on Windows or in the Docker image. CMake installs the prebuilt `docs/website/` through its own `docs_builder/CMakeLists.txt` and never builds the site. Failsafe: at CMake configure time, seconds after the start and before compiling, if documentation install is on and `docs/website/` is missing or its stamp file names a different library version than the one being built, configure stops with an error that says to run `make website`; an explicit option builds the package without documentation. This covers the deb build (Docker copies the working tree), the Windows package (built from the same tree through the shared drive) and `make install`. Wheels ship no documentation (pip builds skip the documentation install), so they need no check. Step 5 writes `docs_builder/CMakeLists.txt` and tests it on its own (`cmake -S docs_builder`); the top-level `CMakeLists.txt` includes it with one line at the cutover. Amended by the maintainer on 2026-11-02: the documentation install option is OFF by default; the Debian and Windows package builds and the release target switch it ON, so the failsafe guards every package, while a user's own system install from a clone or source zip installs no documentation unless the option is given. |
| D23 | Staleness check | `make docs` has a check mode (for example `build.py --check`) that rebuilds the tracked Markdown in memory and fails if it differs from the committed files. It runs on demand and before a release; regeneration is otherwise manual. |

Decisions D24 to D31 were added after the design session (2026-11-02) and its review. Order of precedence where texts differ: D24 to D31, then the rest of this plan, then `md/website_design.md`, then the prototype. The design and the prototype use the old paths (`html_builder/`, `html_docu/`, `md/cpp/...`) and show twins on every page; the spec maps paths to D24 and twins to D26.

| Id | Topic | Decision |
|----|-------|----------|
| D24 | Folder layout | `docs/` holds the tracked Markdown tree in the same layout the website uses (for example `docs/llms.txt`, `docs/api/...`, `docs/formats.md`), with relative links, so the files work as a browsable reference on GitHub and as the website's Markdown twins. `docs/website/` holds the built site, is ignored as a whole and is the deploy root; the builder copies the Markdown tree into it. `docs_builder/` holds everything that makes the documentation: builder code, templates, `content/` (prose and data files), `static/` (CSS, fonts, logo, `.htaccess`, `publications/`, `quadriga_lib_material_model.md`), tests, the landing-figure scripts (D15), `CMakeLists.txt` for the package install (D22) and the skill placeholder. `html_builder/` and `html_docu/` stay until the cutover (D31). Until then the new builder reads the hand-made files the old site also needs (`publications/`, `quadriga_lib_material_model.md`, logo sources in `html_builder/img/`) from their current places; the cutover moves them with `git mv`, together with the MATLAB help script. The prose sources of formats, FAQ and download are converted into `docs_builder/content/`; until the cutover an edit to these pages is made in both places. A top-level `tutorials/` folder holds the notebooks (D28). Exact sub-paths are for the spec. A side effect: `docs/` on GitHub shows the documentation of `main` as of the last `make docs`, quadriga-lib.org the documentation of the latest release. |
| D25 | Releases | For now release packages stay on Strato as today (latest version in the download folder, older versions in `releases/`); the download page links there. Moving to GitHub Releases and filling in older versions is a separate design step after this project (section 7). The spec keeps the switch easy: the base URL of the download links is one setting, and nothing in the build depends on where packages are hosted. No step of this project is blocked by it. |
| D26 | Markdown twins | Twins exist for content pages: the API pages (full file and per-section files), formats, FAQ, Get started (`download.html`), the AI page, publications, the tutorial index and each tutorial (the tutorial twins exist on the website only, D28). No twin for the landing page (`llms.txt` takes that role), the legal pages and the 404 page. Rule: HTML and Markdown come from the same content model and carry the same text, tables and code; each format adds its own navigation (sidebar in HTML; index table at the top of Markdown). The footer link "This page as Markdown" appears only on pages with a twin. |
| D27 | Content format | Three kinds of sources. API documentation: the doc blocks in the sources (D6). Prose pages (formats, FAQ, Get started, AI page, legal pages): plain Markdown files with a small header (title, description), tables, math (D5) and a note-box syntax; their headings make the page index and the anchors (for example `faq.html#cite`). The current `/*!SECTION` blocks in `html_builder/formats/`, `faq/` and `download/` are converted once. Structured pages (landing, publications): TOML data files with fixed fields (pitch, proof lines, cards, features; publication entries), whose text fields allow inline Markdown; templates turn them into the designed layout. Every data file is checked against a schema: unknown keys and missing fields stop the build with file and key. One placeholder marker for all sources; the build lists every remaining placeholder. Hand-written HTML is not used for content. |
| D28 | Tutorial pipeline | `tutorials/` holds all notebooks, stored without outputs. A separate make target (not part of the Markdown or site build) runs each notebook with its kernel and makes an HTML fragment and a Markdown file from the executed notebook; it re-runs only notebooks whose source or the library version changed. The site build wraps the fragments in the tutorial template (design section 5.5: code cells with copy buttons, rendered figures, tables and formulas) and copies the notebook and the Markdown file next to the page for download, so a user can run the notebook on their own installation. The tutorial index is made from metadata in each notebook. Only the notebooks are tracked: tutorial HTML, Markdown and figures are built for the website only and never go into `docs/`. One dummy Python notebook that uses the released pip package tests the pipeline; MATLAB notebooks are supported by the same code but tested only where a MATLAB kernel is installed. The spec settles the metadata fields, the cache location and how figures are referenced from the Markdown. |
| D29 | Deployment | The maintainer adapts his local deploy script (outside the repository) to `docs/website/`; release packages are mirrored as today (D25). |
| D30 | Landing page for team leads | No separate overview page. The "For team leads" card points to sections of the landing page, as in the design. |
| D31 | Coexistence and cutover | The new system is built next to the old one, from the same tree: `make documentation` (old, unchanged) and `make docs` / `make website` (new). Going live with the new site does not need the old one removed. The cutover comes after go-live and the maintainer's sign-off: it deletes `html_builder/` and `html_docu/`, moves the shared hand-made files and the MATLAB help script, includes `docs_builder/CMakeLists.txt`, and updates every user of the old paths (section 11). Step 7 writes its checklist into `docs_builder/README.md`, so it survives the deletion of the planning documents. When the branch is merged into `main` is decided at step 7. |

## 4. Site content

Navigation and grouping are fixed in `md/website_design.md`; this is the content inventory. Source kinds follow D27.

| Page | File | Source | Twin (D26) | Notes |
|------|------|--------|------------|-------|
| Landing | `index.html` | TOML data | no (`llms.txt`) | Pitch, paths, quick start, visuals (D14, D15, D30) |
| C++ API | `cpp_api.html` | `src/` | full + per section | D2, D3 |
| MATLAB/Octave API | `mex_api.html` | `api_mex/` | full + per section | names in lower case, as now |
| Python API | `python_api.html` | `api_python/` | full + per section | |
| Data formats | `formats.html` | Markdown | yes | may contain formulas (D5) |
| Tutorials | `tutorials.html` + one page per notebook | `tutorials/` (D28) | yes | one dummy notebook |
| AI agents | new name | Markdown | yes | D9, D10 |
| Publications | `publications.html` | TOML data | yes | D13 |
| Questions | `faq.html` | Markdown | yes | `#cite` anchor for "How to cite" |
| Get started | `download.html` | Markdown | yes | version and download base URL filled in at build time (D25) |
| Impressum, privacy | new names | maintainer's Markdown files | no | D17, D18 |
| Not found | `404.html` | template | no | served through `.htaccess` |
| `llms.txt`, skill zip | | generated | | D9, D10 |
| Static files | `publications/`, logos, fonts, images, `.htaccess`, `quadriga_lib_material_model.md` | `docs_builder/static/` | | moved from `html_docu/` (D24) |

Landing-page content (the wording is drafted in the design session, numbers come from the maintainer):

- What it is, in one sentence a manager understands, and one for engineers.
- Why it is credible: standards models, QuaDRiGa lineage, measured performance.
- What it covers: channel models, antennas, site-specific simulation, data formats; acoustics marked experimental.
- How to start: `pip install quadriga-lib`, a three-line example, links to the API references and tutorials.
- Where it lives: GitHub, licence, how to cite.

## 5. Requirements

### 5.1 Builder

- One entry point (for example `python3 docs_builder/build.py`) called by `make docs`, `make website` and `make tutorials` (D21), with the page list and navigation in one place instead of fifteen make lines.
- One parser for doc blocks feeding one internal model; HTML and Markdown are two renderers of that model. No duplicated conversion logic.
- A real Markdown parser for prose pages (formats, FAQ, download, AI page, legal pages, tutorial text) instead of regular expressions; TOML data files with schema checks for the structured pages (D27).
- The library version is read once and inserted at build time. The new builder modifies no source files; it writes only `docs/`, `docs/website/` and its cache.
- Code colouring at build time (C++, Python, MATLAB, shell) with CSS classes; no JavaScript.
- Formulas converted to MathML at build time by a Python package (D5). Today no site source contains a formula.
- Clean output: the builder writes the complete `docs/` Markdown tree and `docs/website/`, and removes stale files from earlier builds.
- Checks after the build: every internal link and anchor resolves; generated HTML is well formed; size warnings for pages over a budget the spec sets.
- Errors and warnings as in D7. A test suite for the builder with small fixture blocks covers the parser, both renderers and the error cases.
- Same input and the same tool versions give byte-identical output (D4).
- Dependencies with minimum versions (D4): Python requirements files in `docs_builder/`, no lockfile. The build runs on Linux; the spec states the minimum Python version, and the extra dependencies of the tutorial target (Jupyter, kernels).
- `docs_builder/CMakeLists.txt` installs `docs/website/` with the configure-time failsafe of D22; `make website` writes the stamp file it checks.
- The old builder, `html_docu/`, `extract_matlab_comments.py` and all their users stay untouched until the cutover (D31). The spec describes the cutover changes (section 11) so that `docs_builder/README.md` can carry them as a checklist.
- `.gitignore` additions: `docs/website/`, the tutorial cache; everything else in `docs/` is tracked Markdown (D8, D24). Existing unanchored patterns (`test*.txt`, `dist/`, `release/`, `build*`) must not catch files inside `docs_builder/`.
- Tracked Markdown is written so that parallel branches conflict as little as possible: stable ordering, no build dates or other content that changes on every build.

### 5.2 Pages and static parts

- Mobile first: layouts from 360 px wide upwards, no horizontal page scrolling (wide code blocks and tables scroll inside their box).
- The API pages: a sidebar with sections and functions on wide screens, a collapsible index on phones that works without JavaScript; per function, links to the same function in the other two languages where the builder can match it (the spec settles the matching rule); a link to the Markdown version.
- Accessibility: semantic HTML, keyboard navigation, visible focus, WCAG 2.2 AA contrast, alt texts.
- Browser support: current versions of Chrome, Edge, Firefox and Safari (desktop and mobile), released in the last two years. Older browsers get readable pages, not necessarily the full layout.
- Performance: the landing page loads in one go on a phone; the spec sets a size budget (fonts and images included).
- Link previews: title, description and preview image tags so that a shared link shows a card in messengers.
- Dark mode, print styles and similar details are design decisions (step 2).

### 5.3 Tutorials

- The pipeline of D28: run, convert, wrap, copy notebook and Markdown for download.
- One dummy Python notebook in `tutorials/` (with a fixed random seed) that exercises everything the template shows: text, code cells, text output, a figure, a table, a formula. The tutorial index is built from the notebooks present; while only the dummy exists, a note says so.
- The site and Markdown builds work when the tutorial target has never run (no tutorials shown, a warning printed).

### 5.4 AI agents

- `llms.txt` at the site root: title, one-paragraph summary, links to the Markdown twins, the per-section API files, the full API files, formats, tutorials (website URLs, D28) and the skill.
- A Markdown twin next to every content page (D26), linked from the page.
- Per-section API Markdown files per language, small enough for an agent to fetch one section.
- The AI page explains these files and how to install the skill, with a clear "placeholder" note until the skill exists.

## 6. Out of scope

- Writing the agent skill content (section 7).
- Writing real tutorials (only the dummy notebook is in scope, D28).
- Changing API doc-block content beyond what the builder's new checks reveal as broken; such fixes are listed and made only with the maintainer's agreement.
- The deployment script, the web hosting setup and the move to GitHub Releases (D25, D29).
- Changes to library sources, doc blocks, SPEC files, CMake and packaging files, and the old builder (D20); the cutover (D31).
- The landing-figure scripts and images (D15).
- Compiling the library. The landing-page figure scripts use the released pip package.

## 7. Follow-ups after this project

- Agent skill: a separate session designs the skill content with the maintainer (what an agent needs, recipes, conventions, how it is tested and versioned), then fills the placeholder from D10.
- Tutorial notebooks: writing the real tutorials (D28).
- Releases on GitHub: a separate design step sets up GitHub Releases (tags, release notes, assets, a make target that prepares the release command), fills in the older versions (some releases have no git tag, for example v0.12.4), and then switches the download base URL (D25).
- Registering the library with documentation indexes for agents (for example Context7).
- Cutover (D31) after go-live, following the checklist in `docs_builder/README.md`.
- Doc-block format touch-ups, if the new builder's checks ask for them (D20).
- Development docs: a GitHub Action builds the site on every push to `main` and publishes it on GitHub Pages, clearly labelled as development documentation, while quadriga-lib.org keeps the release documentation. Decided after the new site is live.
- Going live: HTTPS enabled at Strato (section 10), Impressum address settled and text filled in, privacy text filled in, proof-point numbers and landing texts supplied, landing figures made, the deploy script adapted (D29), build and deploy.

## 8. Process

### 8.1 Steps and sign-off

Each step ends with the maintainer's sign-off; the next step starts only after it. Notes from other sessions are information, not a go-ahead.

1. **Plan** (this document). Planning session with the maintainer. The maintainer commits it.
2. **Design.** A design session works out the design with the maintainer (section 9) and writes `md/website_design.md`. The maintainer commits it.
3. **Spec.** Written by a sub-agent of the manager session from the plan and the design. Covers the builder, the static parts, the page templates, the tests and the checks. Output `md/website_spec.md`.
4. **Spec review.** An independent reviewer (a fresh sub-agent that gets only the plan, the design, the spec and the repository, not the writer's conversation) looks for gaps, contradictions and points the implementer would have to guess. One round is mandatory, at most three. After each round the spec is updated and the maintainer commits it.
5. **Implementation** by sub-agents of the manager, from the spec only. The maintainer commits.
6. **Implementation review.** An independent reviewer checks the code and the built site against the spec, the design and this plan, including the phone, tablet and desktop checks of section 8.3. One round is mandatory, at most three; fixes after each round.
7. **Documentation and cleanup.** Write `docs_builder/README.md` (how to build, the make targets, the content formats, the failsafe, and the cutover checklist of D31); mention the new targets in `README.md`; delete `md/website_plan.md`, `md/website_design.md`, `md/website_spec.md` and `html_builder/design_prototype/`. When the branch is merged into `main` is decided here (D31).

### 8.2 Manager session (steps 3 to 7)

- The maintainer talks only to the manager. The manager spawns sub-agents for writing, reviewing and implementing, gives each a complete brief (this plan, the design, the spec, the files it owns, what done looks like), and checks their results before reporting.
- The manager works in `/home/stephan/quadriga-lib-website` on branch `website` and keeps the status table in section 0 current.
- For each step the manager reports what was done, what is open, and the commit command (`git add` line and message) for the maintainer. Commit messages describe the change in plain words, without step numbers or document names.
- Reviews are independent: the reviewer never sees the author's conversation. Each finding names the place, what is wrong, and why it matters; the manager sorts findings into "fix", "decide" (goes to the maintainer) and "reject" (with a reason).
- Sub-agents follow the machine rules in the maintainer's global instructions (shared machine, thread limits). A documentation build is light; the builder's tests need no GPU.
- Temporary files (briefs, scratch output) go to the session's scratchpad or `/home/stephan/quadriga-lib-agents`, not into the repository.

### 8.3 Verification tools

- The built-in browser pane of the desktop app opens the built pages, emulates phone (375 px), tablet (768 px) and desktop widths, and takes screenshots. Reviews use it at all three widths for every page template.
- Link and anchor check and HTML well-formedness check in the builder (section 5.1).
- An HTML validator and an accessibility checker, run by the reviewer of step 6 from a scratch folder; they are not part of the project (decided 2026-11-02).
- The maintainer checks the landing page on a real phone before sign-off of steps 2 and 6.

### 8.4 Prerequisites

- The Python packages from the requirements files, installed in the conda environment `quadriga-web` or in a virtual environment (D4) before step 5.

## 9. Design session (step 2)

### 9.1 What the design session decides

- Visual identity: colours (light and dark if chosen), typography (self-hosted fonts with an open licence), spacing, logo tweaks (D16).
- Navigation: top-level menu, the API sidebar and its phone version, footer.
- Page templates: landing, API page (function entry, section index, language links), text page (formats, FAQ, download, AI page, legal), tutorial index and tutorial page, publications list, 404.
- Components: code block, formula, table, note box, cards, buttons, download links.
- The landing page in detail: text drafts with placeholders for the maintainer's numbers, the figures to make (D15), the phone layout of the first screen.

### 9.2 How the design session works

The maintainer is not a web designer; the session leads, offers concrete choices, and explains trade-offs in plain words. No tools need to be installed. The workflow:

1. **References.** The maintainer names two to four documentation sites he likes or dislikes. The session opens them in the built-in browser pane at phone and desktop width and asks what works for him on each.
2. **Directions.** The session makes two or three clearly different design directions on a claude.ai Design canvas artifact (private by default): for each, the landing page at phone and desktop width and an excerpt of an API page. The maintainer comments on the canvas or in the chat, and can open the link on his own phone.
3. **Choice.** The maintainer picks one direction, possibly with parts from another. Quick A/B choices (colour, typeface, header layout) use inline mockups or multiple-choice questions with previews.
4. **Prototype.** The session turns the chosen direction into a static HTML/CSS prototype of the page templates with real content (a real API section, the formats page, the landing text), in `html_builder/design_prototype/` on the branch. It checks the prototype in the browser pane at 375, 768 and 1280 px; the maintainer checks it on his phone. The prototype is reference material for the spec; step 7 deletes it.
5. **Freeze.** The session writes `md/website_design.md`.

### 9.3 What `md/website_design.md` must contain

- Design tokens: colours with their use, fonts and sizes, spacing scale, breakpoints, radii, shadows.
- Every page template and component from 9.1, with its behaviour at phone, tablet and desktop width, referring to the prototype files.
- Screenshots or descriptions precise enough that a spec writer who has not seen the conversation can specify the HTML and CSS.
- The landing-page text drafts and figure list, with the open numbers marked.
- Which optional JavaScript extras (D1) the design uses, and how each page behaves without them.
- Open points for the spec, if any.

## 10. Notes for the spec from the design review

The planning session reviewed `md/website_design.md` and the prototype at 375 and 1280 px on 2026-11-02. The design is accepted; these points go into the spec:

- Paths: the design and its prototype use `html_builder/`, `html_docu/` and `md/cpp/<section>.md`; the spec maps them to D24 (for example the per-section files under `docs/api/`).
- "How to cite" links to `faq.html#cite`, which the current FAQ does not have. The FAQ conversion (D27) must provide it, and the link check (section 5.1) must catch such gaps.
- Page size: the prototype C++ page is about 1 MB raw (146 kB compressed, current page 125 kB), mostly from colour spans on uncoloured tokens (design section 6). The spec sets a budget per API page.
- Old section anchors: the current pages use hash ids for sections; they keep working next to the readable ids (D3, design open point 2).
- HTTPS does not work on quadriga-lib.org today (TLS handshake failure, checked 2026-11-02); HTTP answers without a redirect. Absolute URLs (preview image, `llms.txt`, links in twins) come from one site base URL setting, `http://` until SSL is enabled at Strato. The HTTP-to-HTTPS redirect is a separate, commented-out `.htaccess` line, switched on only after HTTPS is confirmed. The `http://quadriga-lib.org` links inside doc blocks stay (D20).
- "For team leads" keeps pointing into the landing page (D30).
- The design's open points (design section 11) are answered in the spec, using D24 to D31 where they apply.

## 11. Readiness review (2026-11-02): items for the spec

An independent reviewer checked the plan and the design against the repository. Its decisions are in D3, D8, D15, D20 to D23 and D31; the remaining items go to the spec.

Cutover changes (D31), to be listed in the spec and in the checklist of `docs_builder/README.md`, not made in step 5:

- `CMakeLists.txt`: the `install(DIRECTORY html_docu/ ...)` lines in the system-install branch are replaced by including `docs_builder/CMakeLists.txt`; the `extract_matlab_comments.py` path changes. The script and anything it imports keep to the Python standard library and Python 3.10 (deb build in a jammy container, CMake on Windows).
- `Makefile` (Windows package: copies `html_docu\*` to a hard-coded `0.11.5` path), `Dockerfile.ubuntu` and `.dockerignore` (`docs/website/` must reach the container; the tutorial cache must not), `debian/stamp-changelog.sh`, `.gitattributes` (`html_docu/publications/ export-ignore` moves with the publications), `GNUmakefile` (`documentation`, `release`, `package_dirty`, `clean`, `tidy`).
- Text that names old paths: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md` (including the "Line" column use and "may lag behind"), `md/BUG_SCAN.md`, `md/acoustic_modeling_guidelines.md`, `md/gbsm_integration_handoff.md`, `README.md`, `README_win.md`, `tools/update_armadillo.py`.

Builder and pages:

- Full API Markdown files: kept as `docs/quadriga_lib_api_<lang>.md` (URLs of D3); no build date in the front matter; the function index drops the "Line" column, which shifts on every inserted function. Per-section files under `docs/api/<lang>/`; the largest C++ section is about 127 kB, which the spec accepts or splits.
- Links in twins to files that are not in `docs/` (publication PDFs, the material model, figures, skill zip, notebooks, packages) and raw `<a href="quadriga_lib_material_model.md">` links in doc blocks must work from every output location: the builder rewrites them, or uses absolute site URLs.
- API page intro ("General usage notes"): today two differing copies (`html_builder/parts/*_api.html.part` and `extract_md.py`); the Python HTML copy is outdated. One Markdown intro per language in `docs_builder/content/`; the Python intro is a content gap for go-live.
- Build variables (version, download base URL, site base URL) and content-gap markers use different syntax. A strict mode (for example `--release`) fails on remaining gap markers and missing figures.
- Text-page anchors: explicit heading ids (for example `#cite`) through a syntax that is stripped from the twins.
- Fonts: the seven IBM Plex files are committed in `docs_builder/static/fonts/`, not fetched from npm. KaTeX output: HTML with its fonts, or MathML (`output: "mathml"`, no extra CSS or fonts); the spec chooses and says why.
- Cross-language matching: MATLAB and Python sources match by file name (`api_mex/<x>.cpp` and `api_python/qpy_<x>.cpp`, 78 of 80); C++ needs a rule or a mapping file in `docs_builder/content/`, since doc blocks are not changed.
- Doc-block grammar the parser handles: `*(optional)*` and `(optional)`, `*(in/out)*`, several names in one parameter line, raw HTML (`<br>`, `<a href>`), `<++>` (ignored), `## Usage:` instead of `## Declaration:`.
- Pygments: uncoloured token types mapped to plain text so they get no `<span>`.
- Byte-identical output: sorted file lists, fixed timestamps in the skill zip, no dates.
- Web server (checked 2026-11-02): Strato serves no compressed responses and no content type for `.md`. Size budgets are set in raw bytes, or the spec checks whether `.htaccess` can enable compression; `.htaccess` sets `text/markdown; charset=utf-8` for `.md` and `text/plain; charset=utf-8` for `llms.txt`.
- `404.html` uses root-absolute asset links. The builder never writes `download/` or `releases/` into `docs/website/`, so the deploy mirror can leave those server folders alone.
- Tools: Python 3.11 or newer (`tomllib`); validators that need Java are avoided.
- The privacy page mentions the browser storage key used for the remembered language (D18).
- A generated `docs/README.md` makes `docs/` easy to browse on GitHub.
