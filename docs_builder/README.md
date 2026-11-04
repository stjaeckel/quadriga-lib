# Docs builder

The docs builder writes the documentation of Quadriga-Lib as Markdown files into `docs/`: the full API reference per language, one file per API section, the general usage notes, the Markdown versions of the content pages (Get started, formats, FAQ, publications, AI agents), `llms.txt` for agents and `docs/README.md` as an index on GitHub. Its sources are the documentation comments in `src/`, `api_mex/` and `api_python/` and the files in `docs_builder/content/`. The same folder holds `extract_matlab_comments.py`, which CMake runs to write the MATLAB / Octave help files.

It runs everywhere the library is built: on Linux and Windows, in the release builds (the Debian containers and the Windows zip) and from an unpacked source package. It needs Python 3.10 or newer and nothing outside the standard library, except a TOML reader on Python 3.10 (see below). The website is built by a second builder, which reuses this one's parser: see [website_builder/README.md](../website_builder/README.md).

## Set-up

Nothing to install beyond Python 3.10 or newer. Python 3.11 and newer read TOML with the standard library (`tomllib`). On Python 3.10 install `tomli`: `sudo apt install python3-tomli` on Ubuntu 22.04, or `pip install tomli`. If Python is too old or no TOML reader is found, `build.py` prints one line per missing item, names the Python in use and exits with code 2.

## Quick start

| Command | Does |
|---------|------|
| `make docs` | Builds `docs/` from the sources: writes the files that changed, deletes files the build no longer makes. Run it after editing a doc block or a content page, and commit `docs/` with the change. |
| `make docs-check` | Builds `docs/` in memory and compares it with the files on disk; writes nothing. Exits with 1 and names each file that differs, is missing or is extra. |

Both call `docs_builder/build.py` with the `python3` on `PATH`; no conda environment is needed. Direct calls, from any folder:

```bash
python3 docs_builder/build.py docs
python3 docs_builder/build.py docs --check
python3 docs_builder/build.py docs --root /path/to/repository
```

`make website` runs the same docs phase first and writes `docs/` too.

## Folder layout

| Path | Content |
|------|---------|
| `docs_builder/build.py` | Entry point: checks Python and the TOML reader, then runs `qdocs.cli`. |
| `docs_builder/qdocs/cli.py` | The `docs` command: runs the whole build in memory, writes `docs/` or compares it (`--check`), prints the diagnostics. |
| `docs_builder/qdocs/config.py` | Site URLs, the page list `PAGES`, the content pages `TEXT_PAGES`, source folders, build variables, the version reader, the TOML import. |
| `docs_builder/qdocs/docblocks.py` | Doc-block parser: block grammar, sub-blocks, parameter lists, the table and wrap checks. |
| `docs_builder/qdocs/prose.py` | Content pages: front matter, heading ids, build variables, `publications.toml`, the gap scan. |
| `docs_builder/qdocs/links.py` | Slugs, cross-language matching, `[[name]]` resolution, the URL of every link for each output file. |
| `docs_builder/qdocs/render_md.py` | Markdown output: the twin writer, the full and per-section API files, `publications.md`, `llms.txt`, `docs/README.md`. |
| `docs_builder/qdocs/checks.py` | Link and anchor check of the generated Markdown. |
| `docs_builder/content/` | Every source of `docs/` besides the doc blocks (see [Content pages](#content-pages)). |
| `docs_builder/extract_matlab_comments.py` | MATLAB / Octave help files from the MEX doc blocks (see [Scripts and options](#scripts-and-options)). |
| `docs/` | Generated and tracked. Never edit it by hand: the next `make docs` overwrites the change, and files the build does not make are deleted. |

Every `qdocs` module uses only the standard library. The website builder imports `qdocs`; `qdocs` never imports the website builder. The tests of this builder live in `website_builder/tests/docs/` (see [Checks, tests and the release](#checks-tests-and-the-release)).

How the sources land in `docs/` (`<lang>` is `cpp`, `mex` or `python`; `<slug>` is the section slug, see [Anchors](#anchors)):

| File in `docs/` | Made from |
|-----------------|-----------|
| `quadriga_lib_api_<lang>.md` | All doc blocks of the language (`src/`, `api_mex/`, `api_python/`) and `content/api_intro_<lang>.md` |
| `api/<lang>/general-usage-notes.md` | `content/api_intro_<lang>.md` |
| `api/<lang>/<slug>.md` | The doc blocks of one section |
| `download.md`, `faq.md`, `ai_agents.md` | `content/<same name>.md` |
| `formats.md` | The index of the data formats: title, lead and intro of `content/formats.md`, a table of the format files |
| `formats/<name>.md` | `content/formats/<name>.md`, one file per data format |
| `publications.md` | `content/publications.toml` |
| `llms.txt` | The page list, the API sections and `llms_summary` of `content/ai_agents.md`; absolute links |
| `README.md` | The same list with relative links, for GitHub; not copied to the website |

Every file also takes the library version from `include/quadriga_lib.hpp` (`QUADRIGA_LIB_VERSION_STR`); the Armadillo version for the build variables comes from `CMakeLists.txt` (`set(armadillo_version "...")`). The cross-language links come from `content/api_matching.toml` and the names.

## Scripts and options

### `build.py`

```
python3 docs_builder/build.py docs [--check] [--root DIR]
```

| Option | Meaning |
|--------|---------|
| `docs` | The one command. |
| `--check` | Compare the build with `docs/` and write nothing. |
| `--root DIR` | Repository root; default: the folder above `docs_builder/`. The code always runs from the `docs_builder/` next to `build.py`; sources and output are taken below `--root` (the tests use this). |

| Exit code | Meaning |
|-----------|---------|
| 0 | Success; warnings and gaps are allowed. |
| 1 | Build failed: a structural error, a failed link check, or a difference found by `--check`. |
| 2 | Environment or usage problem: Python too old, no TOML reader, bad arguments, `--root` not a folder. |

The build runs in memory. Errors from all sources are collected first; while any error exists, nothing is written. Without errors, `docs` writes only the files whose content changed, deletes every other file below `docs/` and removes empty folders. All output is UTF-8 with `\n` line ends and one final newline; a checkout with `\r\n` line ends gives the same result.

Diagnostics, one per line, paths relative to the repository root with `/`:

```
<path>:<line>: error: <message>
<path>:<line>: warning: <message> [<kind>]
<path>:<line>: <gap text>
docs: <N> errors, <M> warnings, <K> gaps
```

`:<line>` is left out where there is no line. Errors of the doc-block grammar start their message with an id such as `E-table:`; each warning ends with its kind in brackets, so all cases of one kind can be found with `grep`. The gap list follows the errors and warnings, sorted; the last line is the summary. Ids and kinds are listed in [Errors and warnings](#errors-and-warnings).

### `extract_matlab_comments.py`

```
python3 docs_builder/extract_matlab_comments.py api_mex/<name>.cpp <build folder>/+quadriga_lib/<name>.m
```

CMake calls it once per MEX source in the target `mex_documentation` (`CMakeLists.txt`, section "MEX Documentation", with `ENABLE_MEX_DOC=ON`). It takes the first `/*!MD` block of the file, removes backticks, asterisks, `<br>` and raw links, prefixes each line with `%` and appends the licence text. The result is the MATLAB / Octave `help` text. The script uses only the standard library, runs on Python 3.10 (the Ubuntu 22.04 Debian build) and on Windows, and does not import `qdocs`. Keep it that way.

## Content pages

The prose sources of `docs/` live in `docs_builder/content/`:

| File | Page | Twin in `docs/` |
|------|------|-----------------|
| `download.md` | Get started (`download.html`) | `download.md` |
| `formats.md` | Title, description, lead and intro of the Data formats page (`formats.html`) | `formats.md` (with the table of formats) |
| `formats/<name>.md` | One data format: one section of `formats.html` (see [Data formats](#data-formats)) | `formats/<name>.md` |
| `faq.md` | Questions (`faq.html`) | `faq.md` |
| `ai_agents.md` | AI agents (`ai_agents.html`) | `ai_agents.md` |
| `api_intro_cpp.md`, `api_intro_mex.md`, `api_intro_python.md` | Title, lead and "General usage notes" of each API page | in the full API file and `api/<lang>/general-usage-notes.md` |
| `publications.toml` | Publications (`publications.html`) | `publications.md` |
| `api_matching.toml` | Cross-language buttons (see [Cross-language buttons](#cross-language-buttons)) | none |

The website-only pages (Impressum, privacy, 404, tutorial index) use the same format but live in `website_builder/content/`; see the website README.

### File format

A content page starts with TOML front matter between two lines that are exactly `+++`, followed by the Markdown body:

```
+++
title = "Questions"
description = "Answers to frequent questions about Quadriga-Lib."
lead = "Citing, licence, bug reports."
+++

## About the Library

### How do I cite Quadriga-Lib? {#cite}

Text ...
```

| Key | Required | Use |
|-----|----------|-----|
| `title` | yes | Page heading (`h1`), browser title "title · Quadriga-Lib", preview title, twin heading. |
| `description` | yes | Meta description and preview text. |
| `lead` | no | Lead paragraph below the title (inline Markdown), also in the twin. |
| `llms_summary` | only in `ai_agents.md` | The summary paragraph of `llms.txt` and `docs/README.md`. |
| `dummy_note` | only in `website_builder/content/tutorials.md` | See the website README. |

A missing required key, an unknown key or a value that is not a string stops the build with file and key.

The body is CommonMark with tables, formulas, note boxes, heading ids, build variables and gap markers. Headings run from `##` to `####`; the `h1` comes from `title`. Write each paragraph and each list item on one line; do not wrap prose. Raw HTML is not interpreted on content pages and shows as text: write links and formatting in Markdown.

**Heading ids.** Every `##` and `###` heading gets an id: the slug of its text (lower case, every run of other characters than `a-z` and `0-9` becomes `-`), made unique on the page with `-2`, `-3`. Links and `[[x]]` count with their text only; `*` and backticks are dropped; escapes and entities are resolved. A heading must start in column 0 with `## `; a text line underlined with `=` or `-` (a setext heading) and an indented heading would get no id, so they stop the build (`E-heading`), and so does a `#` heading in the body of a content page, whose title comes from the front matter. Leave a blank line before a `---` rule. To give a heading another id, add it at the end of the heading: `### How do I cite Quadriga-Lib? {#cite}`; the braces never show. Links point to these ids (the footer of every page links `faq.html#cite`), so changing a heading's text changes its id and can break a link; the link check reports it. A text page shows an index of its `##` headings (with the `###` headings below) when it has at least two.

**Build variables.** `{{ name }}` is replaced in the source text before anything else, also inside code, in the body, the front matter and the strings of `publications.toml` (not in doc blocks or notebooks):

| Variable | Value |
|----------|-------|
| `version` | Library version from `include/quadriga_lib.hpp` |
| `armadillo_version` | Armadillo version from `CMakeLists.txt` |
| `site_base_url` | `http://quadriga-lib.org` |
| `download_base_url` | `http://quadriga-lib.org/download` |
| `releases_url` | `http://quadriga-lib.org/releases/` |
| `github_url` | `https://github.com/stjaeckel/quadriga-lib` |

The URL values are constants in `docs_builder/qdocs/config.py`. An unknown name is an error, also inside code: a literal `{{` followed by a lowercase name cannot be written on a content page.

**Note boxes.** GitHub's alert syntax, which GitHub also renders:

```
> [!NOTE]
> **Optional title**
> Body text.
```

The first line is exactly `[!NOTE]` or `[!WARNING]`. If the next line is entirely bold, it is the title; otherwise the title is "Note" or "Warning". The twins keep the block quote as written.

**Formulas and gap markers** follow the same rules as in doc blocks: see [Formulas](#formulas) and [Gap markers and going live](#gap-markers-and-going-live).

### Data formats

The Data formats page has one section per file format, so the formats are not one long file:

- `docs_builder/content/formats.md` holds the page's front matter (`title`, `description`, `lead`) and an optional intro text.
- `docs_builder/content/formats/<name>.md` holds one format. Its front matter has three keys, all required: `title` (the section heading and the title of its Markdown file), `summary` (one line for the table of formats in `docs/formats.md`, `llms.txt` and `docs/README.md`) and `order` (an integer; the sections appear in this order, then by file name). The body starts with `##` headings, which sit one level below the format's title.
- The section id of a format on `formats.html` is its file name without `.md`: `formats.html#qdant` for `formats/qdant.md`. Heading ids are unique across the whole page: a heading that repeats one of an earlier section gets `-2`.
- `make docs` writes `docs/formats/<name>.md` per format and `docs/formats.md` with the table of formats. A link to `formats.html#<name>` becomes a link to `formats/<name>.md` in the Markdown files, a link to a heading in a format to `formats/<name>.md#<id>`.

To add a format: write `docs_builder/content/formats/<name>.md` with the three keys, choose `order`, run `make docs` and commit the new files of `docs/`.

### Edit a page

Edit the file in `docs_builder/content/`, run `make docs`, check the changed files in `docs/` and commit both. Run `make website` to see the HTML page (website README). Links to other pages are written relative to the site root as on the website, for example `[Questions](faq.html#cite)` or `[QDANT](formats.html#qdant)`; the builder rewrites them for each output file (`faq.md#cite` in a twin).

### Add a page

1. Add a line `(id, tab label, twin path)` to `PAGES` in `docs_builder/qdocs/config.py`, in tab order; `None` as tab label keeps the page out of the tab row. The page file is `<id>.html`.
2. For a text page with a twin: add the id to `TEXT_PAGES` in the same file and write `docs_builder/content/<id>.md`. To list it in `llms.txt` and `docs/README.md`, add it to `DOC_LIST` in `docs_builder/qdocs/render_md.py`.
3. Add the page to the menu and footer templates of the website: see "Add a page" in the website README.
4. Run `make docs` and `make website`.

### Publications data

`docs_builder/content/publications.toml` holds the publications page:

| Key | Content |
|-----|---------|
| `page.title`, `page.description`, `page.lead` | Page heading, meta description, lead (inline Markdown). All three are required. |
| `[[group]]` | One per group, in display order; at least one. |
| `group.id`, `group.title` | Heading id (also the twin anchor) and heading. |
| `group.intro`, `group.outro` | Optional inline Markdown before and after the list. |
| `[[group.entry]]` | One per publication, in display order (newest first); at least one per group. |
| `entry.year` (integer), `entry.title`, `entry.authors`, `entry.venue` | The entry text. |
| `entry.link` | At least one `{ label = "...", url = "..." }`; the URL is absolute or relative to the site root, such as `publications/x.pdf`. |

A missing key, an unknown key, a wrong type or an empty list stops the build with the key path, for example `group[1].entry[2].year: missing`. The PDF and Markdown versions of the publications are files of the website (`website_builder/static/publications/`).

## API documentation from the sources

This section is the format of the doc blocks: what the builder parses and what it checks. What to write in them (layout, sizes, ranges, units, argument names, language conventions) is in `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md`.

### Files read

| | C++ | MATLAB / Octave | Python |
|---|-----|-----------------|--------|
| Files | `src/*.cpp` | `api_mex/*.cpp` | `api_python/*.cpp` |
| Function names on the page | as written; member functions start with `.` | lower case | as written; may contain `.` (`sos.acf_2d`) |
| Language of code fences without info string | `cpp` | `matlab` | `python` |
| Required sub-block | `## Declaration:`, `## Declarations:` or `## Usage:`, not for class blocks | `## Usage:` | `## Usage:` |
| File stem for cross-language matching | none | `<stem>.cpp` | `qpy_<stem>.cpp` |

Only `.cpp` files directly in these folders are read (not subfolders), in file-name order, as UTF-8.

### Block grammar

A block starts with a line that is exactly `/*!SECTION`, `/*!SECTION_DESC` or `/*!MD` and ends with the next line that is exactly `SECTION!*/`, `SECTION_DESC!*/` or `MD!*/` of the same kind. Nothing else may stand on a delimiter line, not even leading spaces.

```
/*!SECTION
Array antenna functions
SECTION!*/

/*!SECTION_DESC
Text that describes the whole section.
SECTION_DESC!*/

/*!MD
# function_name
One-line summary

Description ...

## Usage:
...
MD!*/
```

- `/*!SECTION`: exactly one per file that has blocks; it holds one non-empty line, the section title. Files with the same title form one section. Sections are sorted by title on the page; a new title makes a new section.
- `/*!SECTION_DESC`: optional, any number per section; on the page they follow each other in file-name order below the section heading.
- `/*!MD`: one function. Line 1 is the name line, `# name` with nothing else: the name matches `\.?[A-Za-z_][A-Za-z0-9_.]*`, a leading `.` marks a member function (`# .calc_directivity_dBi`), a trailing `<++>` is accepted and ignored. Line 2 is the one-line summary (the brief): it appears in the function tables of the Markdown files and below the function heading. The description follows up to the first sub-block heading. Functions are sorted by name within a section, member functions after the free functions.

### Sub-blocks

A sub-block starts at a line beginning with `## ` outside code fences. Its label is the heading text without a trailing `<br>` and then a trailing `:` or `;` (`## Inputs:`, `## Attributes:<br>`, `## Function 'duplicate';`). The label decides how the content is shown on the website:

| Kind | Labels | Shown as |
|------|--------|----------|
| code | `Declaration`, `Declarations`, `Usage`, `Example`, `Examples` | Markdown; fences without info string get the page language |
| parameters | `Input`, `Inputs`, `Output`, `Outputs`, `Attributes`, `Arguments`, `Input Arguments`, `Output Argument`, `Output Arguments`, `Optional outputs`, and labels starting with `Inputs (`, `Outputs (` or `In/out` | A two-column parameter list (see below) |
| plain | all other labels (`Returns`, `See also`, ...) | Markdown |

Required: C++ blocks need `Declaration`, `Declarations` or `Usage`, except class blocks (blocks with an `Attributes` sub-block); MATLAB and Python blocks need `Usage`. The label may hold inline Markdown such as a code span. In the Markdown files the sub-block heading moves one level down (`### Inputs:`) and is otherwise kept as written.

### Parameter lines

In a parameter sub-block, each list item that starts with `- ` in column 0 is one entry. Write it as

```
- **`name`** — Description; facts separated by semicolons; `[size]`
- **`name`** *(optional)* — Description
- **`Dx`, `Dy`, `Dz`** — Several names in one entry
- **`V_re`** / **`V_im`** — Real and imaginary part
- **`name`**<br>
  Description on the next line
```

- Names are bold code spans; several names are separated by `, ` or ` / ` (also `` `a`, ... `` inside one bold run).
- An optional qualifier in parentheses follows the names: `*(optional)*`, `(optional)`, `*(in/out)*`, `*(optional, in/out)*`; the website shows it below the name.
- The separator is ` — ` (space, em dash, space). ` - ` or ` – ` work but give the warning `param-separator`. `<br>` or nothing after the names means the description starts on the next line.
- The entry takes the following lines that are indented, blank, inside a fence, or unindented directly after a non-blank line. It ends before the next `- ` line in column 0, or before an unindented line that follows a blank line. Indent continuation lines, nested lists and tables by 2 spaces.
- An item that does not have this form is shown as a note across both columns and gives the warning `param-item`.
- Paragraphs and tables between the items are shown in place.

The Markdown files keep every item as written; only the website splits it into name, qualifier and description.

### Links

- `[[name]]` links to the function `name` on the same API page, by the name exactly as in its name line (`[[arrayant_generate]]`, `[[sos.val]]`), without a `quadriga_lib.` prefix; `.[[name]]` links to the member function `.name` (C++). MATLAB names are compared in lower case. Nothing else is tried: a name that matches no function on that page stops the build (`E-link`). Links go only to functions of the same language; a function that exists only in another API cannot be linked from a doc block.
- Links to other pages and files are written relative to the site root, as on the website: `formats.html#qdant`, `faq.html#cite`, `publications/x.pdf`. Absolute links to `http://quadriga-lib.org/...` (with or without `www.`, http or https) are treated the same; links below `download/` and `releases/` stay absolute. Every such link and its anchor is checked: a wrong anchor stops the build.
- A raw `<a href="U">text</a>` in a doc block works like `[text](U)`; `target` and `rel` are dropped. A `<br>` at the end of a line becomes a line break when the next line continues the paragraph, and is dropped at the end of a paragraph.
- Inside code spans and fences nothing is rewritten.

### Anchors

The id of a function is its name on the page, exactly: `arrayant_interpolate_multi`, `.calc_directivity_dBi`, `arrayant_generate` (MATLAB, lower case), `sos.acf_2d`. A section's id is `sec-<slug>`, the slug of its title: "Miscellaneous / Tools" gives `sec-miscellaneous-tools` and the file `api/<lang>/miscellaneous-tools.md`. Headings on content pages get the slug of their text unless `{#id}` sets another one (see [File format](#file-format)); there are no other ids. Renaming a section or a function changes its ids and breaks links to it from outside. Two equal ids on one page stop the build (`E-anchor`).

### Cross-language buttons

Each function entry shows buttons "C++", "MATLAB", "Python" (on the website) and the line "Also in: ... · Not in: ..." (in the Markdown files). The counterparts are found like this:

1. A MATLAB and a Python function form a family when their files have the same stem: `api_mex/<stem>.cpp` and `api_python/qpy_<stem>.cpp`.
2. The family's C++ counterpart is the C++ function with the same name, compared in lower case with `.` read as `_` (`sos.acf_2d` matches `sos_acf_2d`).
3. `docs_builder/content/api_matching.toml` lists the C++ functions of families whose names do not match, and replaces rule 2 for the families it lists:

```toml
[families]
arrayant_calc_directivity = [".calc_directivity_dBi"]
arrayant_combine_pattern = [".combine_pattern", "arrayant_combine_pattern_multi"]
```

Add an entry when a new MATLAB or Python wrapper calls a C++ function with a different name. List the C++ functions the wrappers call to carry out their operation, not helpers for validation, sizing or set-up. The first function listed is the target of the C++ button on the MATLAB and Python pages; put the member function before the `_multi` variant. All listed C++ functions link back to the family. An unknown stem, an unknown C++ name or a C++ function in two families stops the build.

### Formulas

Write a formula in LaTeX between `$...$` (inline) or `$$...$$` (display), in doc blocks, content pages, inline data strings and notebook Markdown cells. Inline formulas need no space after the opening `$` or before the closing `$`, and no digit directly before the opening or after the closing `$`, so `$5 and $6` stays text. Equation labels (`$$...$$ (1)`) are not supported. The Markdown files keep the source; GitHub renders it. The website converts it to MathML; a formula pandoc cannot convert stops the website build (see "Formulas" in the website README).

### Writing pitfalls

These follow from how Markdown reads the text, and the builder checks the first three:

- **Tables need blank lines around them** (`E-table`). A table is a line starting with `|` followed by a delimiter row. The line after its last row must be blank or start a block of its own (list item, `>`, heading, fence, rule), and the line before its header must be blank, a table row, or a line whose content does not start further right than the table; otherwise the text runs into the table.
- **A wrapped line must not start with `>` or `+ `** (`E-wrap`): Markdown would read a quote or a list item. Move the line break, for example before the word in front of the `>`.
- **Code fences close on a backtick run at least as long as the opener** and only on a line that holds nothing but backticks (`E-fence`). To show a fence inside a fence, open the outer one with four backticks.
- **No `*` or `|` in running text**: they are emphasis and table syntax. Write `·` for multiplication and `abs()` for absolute values; inside a formula use `\cdot` and `\lvert x \rvert`.
- **A backtick run needs its closing run in the same paragraph** (warning `code-span`); otherwise the rest of the paragraph is not rewritten in the Markdown files. Escape a literal backtick as `` \` ``.
- **Line length in doc blocks**: up to 120 characters in all three languages (a soft limit; tables and usage lines may be longer). Continuation lines of a list item are indented by 2 spaces. Content pages are not wrapped: one line per paragraph and per list item.
- Fence info strings that colour code: `cpp`, `c++`, `matlab`, `octave`, `python`, `py`, `bash`, `sh`, `shell`, `console`; `text` and `none` give plain text. Another info string gives the warning `fence-language` in the website build.

## Errors and warnings

Errors stop the build; nothing is written. Warnings are printed and do not stop it. "Found by" names the build that reports the case: `docs` means `make docs`, `make docs-check` and `make website`; `website` means only `make website`.

| Id or kind | Type | Found by | Cause | Fix |
|------------|------|----------|-------|-----|
| `E-delim-open` | error | docs | An opening delimiter has no closing delimiter of its kind before the next opening delimiter or the end of the file. | Add the closing line (`MD!*/`, `SECTION!*/`, `SECTION_DESC!*/`) on its own, without spaces. |
| `E-section-count` | error | docs | A file with `/*!MD` or `/*!SECTION_DESC` blocks has no `/*!SECTION` block or more than one, or the `/*!SECTION` block is not exactly one non-empty line. | One `/*!SECTION` block per file, holding only the section title. |
| `E-name` | error | docs | The first line of an `/*!MD` block is not `# name`. | Write `# name` (member: `# .name`) and nothing else on the line. |
| `E-fence` | error | docs | A code fence is not closed within its part (the summary line, the body of an `/*!MD` block, an `/*!SECTION_DESC` block). | Close it with a line of at least as many backticks as the opener. |
| `E-required` | error | docs | A required sub-block is missing. | Add `## Declaration:` (C++) or `## Usage:` (MATLAB, Python). |
| `E-link` | error | docs | `[[name]]` matches no function on the API page of the same language. | Correct the name, use `.[[name]]` for a member function, or remove the link. |
| `E-anchor` | error | docs | Two equal ids on one page: function names, section ids, heading ids or publication group ids (on the formats page across all format files). | Rename one, or give a heading an explicit `{#id}`. |
| `E-gap` | error | docs | A gap marker `%%GAP:` is not closed with `%%` on the same line. | Close it on the same line. |
| `E-table` | error | docs | A table and the text next to it have no blank line between them. | Add the blank line. |
| `E-wrap` | error | docs | A wrapped line starts with `>` or `+ `. | Move the line break. |
| `E-heading` | error | docs | In a doc block or on a content page, a text line underlined with `=` or `-` (setext heading) or a `#` heading that does not start in column 0; on a content page also a `#` heading in the body. The website build also reports a heading in a notebook cell that is not written as `## text` in column 0. | Write it as `## text` or `### text` at the start of the line; put a blank line before a `---` rule; the page title belongs in the front matter. |
| `param-item` | warning | docs | An item in a parameter sub-block does not have the form ``- **`name`** — text``. | Rewrite the item; otherwise it is shown as a note row. |
| `param-separator` | warning | docs | A parameter entry uses ` - ` or ` – ` instead of ` — `. | Use the em dash. |
| `code-span` | warning | docs | A backtick run has no closing run in its paragraph. | Close the code span or escape the backtick. |
| `fence-language` | warning | website | A code fence has an info string the colouring does not know. | Use one of the names in [Writing pitfalls](#writing-pitfalls), or `text`. |
| `size` | warning | website | A page or the landing page exceeds its size budget. | See "Checks" in the website README. |
| `not-run` | warning | website | A notebook code cell has no execution count. | Run the notebook and commit it with its outputs. |
| `output-type` | warning | website | A notebook output has no SVG, PNG or plain-text representation; it is left out. | Change the cell so that it produces one of these. |
| `figure-alt` | warning | website | A notebook image has no alt text. | Set the cell metadata `quadriga_tutorial.alt`. |

Errors without an id name the file and, where possible, the line or key:

| Message | Cause and fix |
|---------|---------------|
| `content file missing` | A file of `docs_builder/content/` (or `website_builder/content/`, the skill stub) that the page list needs is not there. |
| `file not found`, `no match for ...` | The version could not be read from `include/quadriga_lib.hpp` or `CMakeLists.txt`. |
| `the file must start with TOML front matter ...`, `TOML: ...`, `front matter: <key>: ...` | Front matter missing, not valid TOML, or a key missing, unknown or not a string. |
| `unknown build variable '{{ x }}'` | See [build variables](#file-format). |
| `<key path>: missing`, `unknown key`, `empty list`, `not an integer` | `publications.toml` or notebook metadata does not match its keys. |
| `unknown stem`, `unknown C++ function`, `... is in two families` | `api_matching.toml` names a file or C++ function that does not exist, or one function twice. |
| `link target missing`, `anchor missing`, `link leaves the site` | A link in the generated files points to a file or id that does not exist. The `docs` build checks only links inside `docs/`; the website build checks all of them. |
| `differs from the build output`, `missing, the build writes it`, `extra, the build does not write it` | `docs --check` found `docs/` out of date: run `make docs` and commit. |

The website build adds its own errors (HTML checks, notebooks, formulas, `--release`); they are listed in the website README.

## Gap markers and going live

A gap marker marks content that is still missing or a draft that waits for approval: `%%GAP: performance numbers and hardware%%`. It must open and close on one line. Removing the marker (keeping or replacing its text) is the approval.

- Markers may appear in `docs_builder/content/` and `website_builder/content/` (bodies, front matter, `publications.toml`) and in the website templates (the landing text). Not in doc blocks or notebooks.
- Nothing in `docs/` contains a marker: each `%%GAP: text%%` is removed with its text; a line or list item that held only a marker is dropped, and a field (title, lead, summary) left empty is omitted. So `docs/` never shows draft text that is still gap-marked.
- `make docs` prints the gaps of `docs_builder/content/` as a list (`path:line: text`). The website build prints the full list and shows each marker highlighted on the page.
- The site goes live only when the list is empty: `website_builder/build.py website --release` fails on any remaining gap (see "Gaps and `--release`" in the website README).

## Checks, tests and the release

**Checks.** `make docs-check` tells whether the committed `docs/` matches the sources. Every build also checks every link and anchor in the generated Markdown that points into `docs/`.

**Tests.** The tests of this builder are in `website_builder/tests/docs/`. They live in the website builder's folder because their fixture repository (`website_builder/tests/fixtures/mini_repo/`, which holds copies of `CMakeLists.txt`, the header and small sources) must not ship in the source package. They need only Python and pytest:

```bash
python3 -m pytest website_builder/tests/docs
```

They import nothing of the website builder and write only below pytest's temporary folders. Tests marked `repo` read the real repository. `test_readme.py` checks that the table in [Errors and warnings](#errors-and-warnings) lists exactly the ids and kinds the code defines: update the table when you add or remove one.

**Byte-identical output.** The same sources give byte-identical files in `docs/` on Linux and Windows, in the Debian containers and with any Python from 3.10: listings are sorted, and no date, user name, absolute path or tool version is written. So `docs --check` gives the same answer everywhere.

**Release.** Each release path refuses a stale `docs/`, with plain Python and no website tools:

| Release path | Check |
|--------------|-------|
| `make release` | First refuses any uncommitted change to a tracked file and any new file in `docs/`, then runs `python3 docs_builder/build.py docs --check`, then makes the source zip, the Debian packages and the pip packages. |
| Debian build (`Dockerfile.ubuntu`, also `make deb-jammy` and the others on their own) | `RUN python3 docs_builder/build.py docs --check` right after copying the tree, before the long compile. |
| `nmake release-win` (`Makefile`) | First command: `python docs_builder\build.py docs --check`. The Windows build machine needs Python on `PATH`. |

The packages carry `docs/`: the system install (`cmake --install` with a prefix) puts it into `share/doc/quadriga-lib/`, the Debian package keeps the `.md` files and `llms.txt` uncompressed (`dh_compress -X.md -Xllms.txt` in `debian/rules`, so the relative links work), and the Windows zip has a `docs\` folder. The source zip (`git archive`) contains `docs/` and `docs_builder/`, but not `website_builder/`.

## Twins and agents

Every page of the website that has a Markdown version (a twin) links to it in the footer and in the page head. Twins are written from the source text, not from the HTML:

- Each twin starts with `# title`, then `Quadriga-Lib <version> · Web page: <URL>`, then an index table of its `##` headings when it has at least two (with a column of `###` links on content pages), then the lead and the body.
- The text is copied line by line. Only these change: sub-block headings move one level down; `{...}` after headings is removed; fences without info string get the page language; `[[name]]`, raw `<a href>` links and trailing `<br>` become Markdown; link URLs are rewritten for the twin's location; gap markers are removed. Formulas, tables, note boxes and parameter items stay as written.
- Anchors: every function, section and heading gets an explicit `<a id="..."></a>` line before it, with the same ids as the HTML page (for a section `<a id="sec-<slug>"></a>`). So `#arrayant_generate`, `#sec-channel-functions` and `faq.md#cite` work in both formats, and links between twins use these ids. When you change how anchors are written, check on GitHub that a function link with a `.` (such as `#.calc_directivity_dBi`) still resolves.
- Every function has the line "Also in: [C++](...) · [Python](...)" or "Not in: C++, Python" with the same information as the buttons on the website. "Not in" names a language without a counterpart; the line is never left out.
- `quadriga_lib_api_<lang>.md` has YAML front matter (title with the version, author, language), the general usage notes, a "Function Index" table (Function, Section, Summary) and then every section and function. Agents should read the index and then fetch only the section file `api/<lang>/<slug>.md`.
- Links to website pages without a twin (the landing page, the Impressum) and to site files outside `docs/` (publications, the skill, the material model, the tutorials) are absolute `http://quadriga-lib.org/...` links in `docs/`.
- `llms.txt` (format of llmstxt.org) lists the content pages, every data format, the three full API files and every API section with its number of functions, with absolute links; it is served at `http://quadriga-lib.org/llms.txt`. `docs/README.md` is the same list with relative links and a note that the files are generated.

## License

Apache License 2.0, see [LICENSE](../LICENSE).

## See also

- [website_builder/README.md](../website_builder/README.md): the website, its tools, templates, tutorials and deployment.
- `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`: what to write in a doc block.
- [CONTRIBUTING.md](../CONTRIBUTING.md): issues, pull requests, and keeping this README current with the code.
