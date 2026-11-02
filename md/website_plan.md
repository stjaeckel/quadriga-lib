# Quadriga-Lib website overhaul: project plan

This plan is the input for every later step of the website overhaul: the design session (step 2), the spec (step 3), the spec review (step 4), the implementation (step 5), the implementation review (step 6) and the cleanup (step 7). It records the goals, the decisions taken in the planning session on 2026-11-02, the requirements, and the process. It is a temporary document: step 7 deletes it, and it stays in the git history.

Sessions that read this plan: the decisions in section 3 are settled. Do not reopen them; if one turns out to be impossible or contradicts another, stop and report it to the maintainer.

## 0. Status

| Step | What | Output | State |
|------|------|--------|-------|
| 1 | Project plan | `md/website_plan.md` | signed off 2026-11-02 |
| 2 | Design session | `md/website_design.md` | not started |
| 3 | Spec | `md/website_spec.md` | not started |
| 4 | Spec review (1 to 3 rounds) | updated spec | not started |
| 5 | Implementation | builder, static parts, pages | not started |
| 6 | Implementation review (1 to 3 rounds) | fixes | not started |
| 7 | Documentation, cleanup, merge into `main` | README, `html_builder/README.md` | not started |

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
- An engineer reaches any function of any language in at most two clicks from any page, and can jump to the same function in another language where one exists.
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
| D3 | URLs | All current file names and function anchors stay valid (`cpp_api.html#arrayant_generate`, `mex_api.html#...`, `python_api.html#...`, `formats.html`, `faq.html`, `publications.html`, `download.html`, `index.html`). Section anchors may change only if old ones keep working. New pages get new names. |
| D4 | Build tools | Python 3 with pinned pip packages (for example markdown-it-py, Pygments, Jinja2) and Node.js at build time. No server-side processing on the web server. |
| D5 | Formulas | LaTeX (`$...$`, `$$...$$`) rendered to HTML at build time with KaTeX; KaTeX stylesheet and fonts self-hosted. No math JavaScript in the browser. |
| D6 | Doc-block syntax | The `/*!SECTION`, `/*!SECTION_DESC`, `/*!MD` syntax stays. Small extensions are allowed (for example math, notes) if the SPEC files are updated in the same project. The lean-API rule stays: API blocks hold layout, ranges, units and errors, not derivations. |
| D7 | Builder strictness | Structural errors stop the build with file and line (unclosed code fence, unresolved `[[link]]`, missing name line, missing `## Declaration:` where required, duplicate anchors). Style issues print a warning. The builder has its own test suite. |
| D8 | Generated output in git | Generated Markdown stays tracked: the three full API files at their current paths, the per-section API files and `llms.txt`. Git users and agents then read a current API reference on GitHub (which renders Markdown and `$...$` math) without building anything, and the SPEC files keep working. Generated HTML, CSS, fonts and KaTeX output are no longer tracked; hand-made files move from `html_docu/` into `html_builder/`. Reasons: generated HTML in git conflicts between parallel branches that edit doc blocks and goes stale when a rebuild is forgotten. Supporting rules: the Markdown outputs build with Python only (D21); packages build their own HTML (D22); a check finds stale Markdown (D23). |
| D9 | AI agents | The builder writes a Markdown twin of every page, per-section API Markdown files, the three full API Markdown files, and `llms.txt`. A new page explains how to use the library with AI agents. |
| D10 | Agent skill | A placeholder now: a folder with a stub `SKILL.md`, packed by the builder into a versioned zip that the AI page offers for download. The skill content is designed with the maintainer in a separate session (section 7). |
| D11 | Tutorials | Jupyter notebooks for Python and MATLAB (MathWorks Jupyter MATLAB kernel; Octave kernel possible). Pipeline (outside this project): notebooks are run, converted to HTML and Markdown, and the builder wraps them in the site layout; the `.ipynb` and `.md` are offered for download. This project builds a tutorial index with 3 to 4 dummy cards and one sample tutorial page from a hand-made dummy export, which fixes the page template and the input format. |
| D12 | Acoustics | Listed as a feature marked "experimental"; its API functions appear as now. Not part of the landing-page pitch. |
| D13 | Publications | Better-looking list with links to IEEE, PDF and Markdown versions, as now. No HTML rendering of publications. |
| D14 | Landing-page claims | Standards work (3GPP TR 38.901, IEEE 802.11 TGn/ac/ax/ah, IEEE 802.11 contributions), lineage and track record (successor of QuaDRiGa), measured performance. Wording and numbers come from the maintainer; nothing is invented. No named users or organisations. |
| D15 | Landing-page visuals | Figures made by scripts that use the library (for example an antenna pattern, a ray-tracing scene, a channel response). The design session chooses them. The scripts run against the released pip package, outside `make documentation`; their images are committed as static files. |
| D16 | Logo | The general shape of the current logo stays; tweaks and colour changes are allowed. |
| D17 | Legal pages | An Impressum page and a short privacy page, linked from every footer. Their text comes from plain text files the maintainer fills in. No address goes into git until the maintainer has settled it; until then the Impressum file holds a placeholder and going live is blocked. |
| D18 | Privacy and analytics | No analytics, no cookies, no third-party requests (fonts, scripts, images all self-hosted). The privacy page covers the hoster and its server logs. Storing the chosen API language in the browser happens only after the visitor clicks a language. |
| D19 | Commits | The maintainer makes every commit. Sessions give the `git add` line and the commit message. |
| D20 | Branch and working copy | Branch `website`, working copy `/home/stephan/quadriga-lib-website` (made from `main`, fast-forwarded to `main` when it moves ahead and the branch has no merges pending). Local only, not pushed until the merge in step 7. |
| D21 | Toolchain by output | The tracked Markdown outputs build with Python alone, with no Node and no pip packages beyond what the spec shows to be needed, through a fast separate target (for example `make documentation-md`). The full HTML site needs the full toolchain (D4) and is built by the maintainer for deployment and packages. A missing tool stops the build with a plain message naming the tool and how to install it, not with a stack trace. The spec checks whether pages without formulas (the API pages) can build without Node and keeps them so if possible. |
| D22 | Packages | `make package` and the Windows binary package build the HTML site and add it to the archive, so offline documentation does not depend on what is tracked. |
| D23 | Staleness check | A check mode of the builder (for example `build.py --check`) rebuilds the tracked Markdown in memory and fails if it differs from the committed files. It runs before a release and is ready for later use in CI. |

## 4. Site content

The design session settles navigation and grouping; this is the content inventory.

| Page | File | Source | Notes |
|------|------|--------|-------|
| Landing | `index.html` | hand-written | Pitch for decision makers, paths for engineers, quick start, visuals (D14, D15) |
| C++ API | `cpp_api.html` | `src/` | D2, D3 |
| MATLAB/Octave API | `mex_api.html` | `api_mex/` | names in lower case, as now |
| Python API | `python_api.html` | `api_python/` | |
| Data formats | `formats.html` | `html_builder/formats/` | may contain formulas (D5) |
| Tutorials | `tutorials.html` + one sample page | dummy (D11) | new |
| AI agents | new name | hand-written | new (D9, D10) |
| Publications | `publications.html` | hand-written | D13 |
| Questions | `faq.html` | `html_builder/faq/` | |
| Download and setup | `download.html` | `html_builder/download/` | version filled in at build time |
| Impressum, privacy | new names | maintainer's text files | D17, D18 |
| Not found | `404.html` | hand-written | served through `.htaccess` |
| `llms.txt`, Markdown twins, per-section API Markdown, skill zip | | generated | D9, D10 |
| Static files | `publications/`, logos, fonts, images, `.htaccess`, `quadriga_lib_material_model.md` | `html_builder/` | moved from `html_docu/` (D8) |

Landing-page content (the wording is drafted in the design session, numbers come from the maintainer):

- What it is, in one sentence a manager understands, and one for engineers.
- Why it is credible: standards models, QuaDRiGa lineage, measured performance.
- What it covers: channel models, antennas, site-specific simulation, data formats; acoustics marked experimental.
- How to start: `pip install quadriga-lib`, a three-line example, links to the API references and tutorials.
- Where it lives: GitHub, licence, how to cite.

## 5. Requirements

### 5.1 Builder

- One entry point that `make documentation` calls (for example `python3 html_builder/build.py`), with the page list and navigation in one place instead of fifteen make lines.
- One parser for doc blocks feeding one internal model; HTML and Markdown are two renderers of that model. No duplicated conversion logic.
- A real Markdown parser for prose pages (formats, FAQ, download, AI page, tutorials) instead of regular expressions.
- The library version is read once and inserted at build time. The builder does not rewrite tracked files (the current `extract_version.py` and `sed` steps go away).
- Code colouring at build time (C++, Python, MATLAB, shell) with CSS classes; no JavaScript.
- Formulas through KaTeX at build time (D5), with one Node process per build, not one per formula.
- Clean output: the builder writes the complete `html_docu/` and removes stale files from earlier builds.
- Checks after the build: every internal link and anchor resolves; generated HTML is well formed; size warnings for pages over a budget the spec sets.
- Errors and warnings as in D7. A test suite for the builder with small fixture blocks covers the parser, both renderers and the error cases.
- Same input gives byte-identical output.
- Pinned dependencies: a Python requirements file and a Node package file with lockfile in `html_builder/`. The build runs on Linux; the spec states the minimum Python and Node versions.
- `extract_matlab_comments.py` keeps its CMake contract. It may reuse the new parser.
- Every user of `html_docu/` listed in section 2 keeps working or is changed in this project: the SPEC files, `README.md`, `README_win.md`, the Windows binary package, `make package`, `make release`. The spec lists each one and what happens to it (D8, D22).
- `.gitignore` ignores the generated HTML and assets in `html_docu/` but not the tracked Markdown outputs (D8). The spec decides whether the tracked Markdown stays inside `html_docu/` or moves, keeping the current API file paths valid.
- Tracked Markdown is written so that parallel branches conflict as little as possible: stable ordering, no build dates or other content that changes on every build.

### 5.2 Pages and static parts

- Mobile first: layouts from 360 px wide upwards, no horizontal page scrolling (wide code blocks and tables scroll inside their box).
- The API pages: a sidebar with sections and functions on wide screens, a collapsible index on phones that works without JavaScript; per function, links to the same function in the other two languages where the builder can match it (the spec settles the matching rule); a link to the Markdown version.
- Accessibility: semantic HTML, keyboard navigation, visible focus, WCAG 2.2 AA contrast, alt texts.
- Browser support: current versions of Chrome, Edge, Firefox and Safari (desktop and mobile), released in the last two years. Older browsers get readable pages, not necessarily the full layout.
- Performance: the landing page loads in one go on a phone; the spec sets a size budget (fonts and images included).
- Link previews: title, description and preview image tags so that a shared link shows a card in messengers.
- Dark mode, print styles and similar details are design decisions (step 2).

### 5.3 Tutorials placeholder

- Tutorial index with 3 to 4 dummy cards (title, language, one-line summary, level).
- One sample tutorial page from a hand-made dummy export: text, a code cell, an output figure, a formula, and download links for `.ipynb` and `.md`.
- The spec defines the input format the future notebook pipeline must deliver (for example an HTML fragment, a Markdown file and a small metadata file per tutorial) and where these files live in the repository.

### 5.4 AI agents

- `llms.txt` at the site root: title, one-paragraph summary, links to the Markdown twins, the per-section API files, the full API files, formats, tutorials and the skill.
- A Markdown twin next to every HTML page, linked from the page.
- Per-section API Markdown files per language, small enough for an agent to fetch one section.
- The AI page explains these files and how to install the skill, with a clear "placeholder" note until the skill exists.

## 6. Out of scope

- Writing the agent skill content (section 7).
- Running notebooks and the notebook-to-HTML pipeline.
- Changing API doc-block content beyond what the builder's new checks reveal as broken; such fixes are listed and made only with the maintainer's agreement.
- The deployment script and the web hosting setup.
- Compiling the library. The landing-page figure scripts use the released pip package.

## 7. Follow-ups after this project

- Agent skill: a separate session designs the skill content with the maintainer (what an agent needs, recipes, conventions, how it is tested and versioned), then fills the placeholder from D10.
- Tutorial notebooks and their pipeline (D11).
- Registering the library with documentation indexes for agents (for example Context7).
- Development docs: a GitHub Action builds the site on every push to `main` and publishes it on GitHub Pages, clearly labelled as development documentation, while quadriga-lib.org keeps the release documentation. Decided after the new site is live.
- Going live: Impressum address settled and text filled in, privacy text filled in, proof-point numbers supplied, build and deploy.

## 8. Process

### 8.1 Steps and sign-off

Each step ends with the maintainer's sign-off; the next step starts only after it. Notes from other sessions are information, not a go-ahead.

1. **Plan** (this document). Planning session with the maintainer. The maintainer commits it.
2. **Design.** A design session works out the design with the maintainer (section 9) and writes `md/website_design.md`. The maintainer commits it.
3. **Spec.** Written by a sub-agent of the manager session from the plan and the design. Covers the builder, the static parts, the page templates, the tests and the checks. Output `md/website_spec.md`.
4. **Spec review.** An independent reviewer (a fresh sub-agent that gets only the plan, the design, the spec and the repository, not the writer's conversation) looks for gaps, contradictions and points the implementer would have to guess. One round is mandatory, at most three. After each round the spec is updated and the maintainer commits it.
5. **Implementation** by sub-agents of the manager, from the spec only. The maintainer commits.
6. **Implementation review.** An independent reviewer checks the code and the built site against the spec, the design and this plan, including the phone, tablet and desktop checks of section 8.3. One round is mandatory, at most three; fixes after each round.
7. **Documentation, cleanup and merge.** Update `html_builder/README.md`, `README.md`, `README_win.md` and the SPEC files where they mention the builder or `html_docu/`; delete `md/website_plan.md`, `md/website_design.md` and `md/website_spec.md`; delete design prototypes left on the branch; merge `website` into `main` after the maintainer's sign-off.

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
- Optional, because Node is available: an HTML validator and an accessibility checker run from npm, called by a separate make target, not by `make documentation`.
- The maintainer checks the landing page on a real phone before sign-off of steps 2 and 6.

### 8.4 Prerequisites

- Node.js and npm on the build machine (Ubuntu package `nodejs`; the spec states the minimum version) before step 5.
- The Python packages from the requirements file, installed in a virtual environment or with the user's pip before step 5.

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
