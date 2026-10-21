# Brief — the material part of the Data Formats page

This brief is the whole task description. Read it in full before doing anything. Findings go into
`md/formats_doc_report.md` (create it). One stop: before the commit.

## Why

`tools/data_formats_md/format_materials.md` is the source of the material part of the published Data Formats page
(`html_docu/formats.html`). It was written before the acoustic work and still describes a model that no longer exists. The
task-3 report (`md/cleanup_doc_report.md` 0.7) lists 18 contradictions with the code, for example:
- a transmission gate (light→dense / dense→light) that does not exist;
- `m` as a slope;
- "no scalar refraction";
- the one-parameter rigid mapping;
- a clamp of the reflected power;
- the quarter-wave "phaseless tracer" argument;
- `fRef` 0.875 GHz fixed for every acoustic material;
- an acoustic `concrete` default row without `tf`.

Its column table lacks 6 of the 26 columns.

Since 2026-10-21 the public description of the model is `html_docu/quadriga_lib_material_model.md`, rewritten from the
code (commit 23ad1b6). The Data Formats page should not describe the model a second time.

## Decisions (Stephan, 2026-10-21)

- **A lean format reference.** The material part of the Data Formats page keeps:
  - what belongs to the file format: the material table, its CSV form and how `obj_file_read` reads it;
  - a column table of all 26 columns (key, unit, default, a one-line meaning, wave media / mass-law / both);
  - the built-in EM table;
  - the visual BSDF part.

  All model prose is replaced by a link to the material-model page: the formulas, mechanisms, the EM interpretation
  beyond the table, and the whole acoustic interpretation. There is one description of the physics, so nothing can drift.
- **The "Acoustic Default Materials" section is removed.** One sentence points to the preliminary starting rows of
  the page (section 11.4). `data/materials_ac.csv` and `tools/acoustic_parameter_mapping/` stay untouched; they belong
  to the parameter project.
- **Three neighboring fixes**, comment-only:
  1. The C++ `obj_file_read` API block (`src/obj_file_read.cpp`, about line 400) says a `.obj` material is matched
     "by exact name, then by name with a trailing Blender `.NNN` suffix removed". The code matches the part before
     the first dot (`base_name`, line 89; `concrete.gray` → `concrete`), as the MATLAB and Python blocks say.
     Correct the C++ sentence.
  2. The sentence "For a detailed description of the material model see Data Formats" appears in the three
     `obj_file_read` blocks (`src/obj_file_read.cpp`, `api_python/qpy_obj_file_read.cpp`, `api_mex/obj_file_read.cpp`).
     It should name both: the material-model page for the model, Data Formats for the table format.
  3. Section 9 of the material-model page says the built-in table is "valid 1–40 GHz, ground 1–10 GHz", while
     `format_materials.md` gives per-material upper limits up to 100 GHz (its "max fGHz" column). Check both against
     Rec. ITU-R P.2040-3, Table 3. Correct the one that is wrong, and keep the per-material ranges in the formats
     table if they hold.
- **One stop**, before the commit. The report lists every removed or changed statement, with its reason and the code
  line it rests on.

## Facts at the start

- Last commit 32c55b2 (the `.gitignore` change after 23ad1b6). `git status` should be clean apart from
  `md/acoustic_manager.md` and this brief. If anything under `src/` or `tests/` is modified, stop and ask Stephan.
- Counts (`md/wave.md` 0, 12 threads): `[wave]` 163474 / 178, `[acoustic]` 42741 / 59 (`QD_DIAG_RAY=1`), whole
  suite 1344398 / 1264 (`QD_DIAG=1`). `python3 tools/acoustic/check_docs.py` passes.
- `format_materials.md` (307 lines) has the following blocks:
  - `/*!SECTION` (title "Material Properties");
  - `/*!SECTION_DESC` (the section's introduction, with the "transmission partition" paragraph and a list of
    related functions and Wikipedia links);
  - five `/*!MD` blocks: "1. Material Model" (column table, frequency laws, mechanisms), "2. Electromagnetic
    Interpretation" (with the default table), "3. Acoustic Interpretation" (lines 128–232), "4. Principled BSDF",
    "5. Background and References".
- How the page is built: `make documentation` (in `GNUmakefile`) runs
  `python3 tools/extract_html.py -o html_docu/formats.html -p tools/html_parts/formats.html.part -d tools/data_formats_md/`.
  - `extract_html.py` reads the `/*!SECTION`, `/*!SECTION_DESC` and `/*!MD` blocks of every `.md` file in that
    folder. `format_qdant.md` is the other file there and is not touched.
  - An MD block's first line `# N. Title` is its entry name, and the second line its one-line description. Entries
    are sorted by name, which the numbers keep in order.
  - The converter knows a small Markdown subset: inline code, bold, italic, lists, tables, `<br>`, HTML links. It has
    **no LaTeX math**, so formulas stay plain text in code spans, as today.
- The column facts are in the code:
  - `MaterialCols` (`src/ray_mesh_interact.cpp` 257–374): names, defaults, checks;
  - the table section of the material-model page (section 4), written from the same code;
  - the CSV reader (`src/obj_file_read.cpp` 95–170): only `name` is required, empty numeric cells parse as 0, a
    non-numeric cell is an error;
  - the built-in table `DEFAULT_MATERIAL_CSV` (`src/obj_file_read.cpp` 23–49).
- Check the claim "`air` is row 0 of the table, the transparent fallback" against the code: unmatched faces get index
  0 (no material), and the built-in `air` row is a row of its own.
- The other places that link to `formats.html`:
  - the `ray_mesh_interact` API block (its "CSV format of the material table" pointer, kept as decided in task 3);
  - the three `obj_file_read` blocks;
  - `html_docu/cpp_api.html` and the other generated pages, which Stephan regenerates.

## Hard rules

- **Documents only.** Under `src/` and `api_*` only the sentences named above change, and only inside their
  `/*!MD … MD!*/` blocks. Checked with `git diff -U0` (every changed line inside an MD block), a build, the whole
  suite (counts unchanged), and `tools/extract_md.py -a cpp`, `-a python` and `-a mex` into scratch.
- **Written from the code.** Every column, unit, default and reader rule is checked against the source line, and the
  report names it. The material-model page is the reference for wording and meaning. Where the page and the code
  differ, the code decides and the report says so (expected: none, apart from the frequency range).
- **The P.2040-3 ranges come from the recommendation itself.** If it cannot be read, report that, leave both
  statements as they are, and do not guess.
- **A public document.** No internal labels, no `md/` files, no history ("removed on …", "the former …"), American
  English, plain statements, numbers with units.
- **The generated page is read.** Build `formats.html` into scratch with the command above (the `-o` path in scratch)
  and read the material part as HTML. Check that no raw Markdown or LaTeX is left, that the tables render, and that the
  links resolve (`formats.html` sits next to `quadriga_lib_material_model.md` in `html_docu`, so the relative link
  works).
- **`tools/acoustic/check_docs.py`** also checks `format_materials.md`: every `MaterialCols` column is a row of its
  column table, no internal labels, American spelling. Its README row says so.
- **Resources** (another session shares the machine): at most 12 threads (`cmake --build build_linux --parallel 12`,
  `OMP_NUM_THREADS=12` or fewer), one suite at a time, no heavy tools.
- **No new open items** (Stephan's rule): a side finding is one line in the report, nothing more.
- Do not commit; Stephan commits.

## The work

1. **Baseline**: build, the whole suite with `QD_DIAG=1`, the counts; `check_docs.py`.
2. **Inventory** (report section 1): every statement of `format_materials.md`, kept / changed / removed, with the
   reason (the code line, or "model prose, now on the page"). The 18 points of `md/cleanup_doc_report.md` 0.7 are
   covered by name.
3. **The new file.**
   - `SECTION_DESC`: what the material table is (one row per material, a name-keyed table, `csv_prop` as a map of
     columns), where the model is described (the page link), the related functions.
   - "1. Material Table": the CSV form and reader rules; the 26-column table (key, unit, default, meaning,
     wave / mass-law / both), with `m > 0` named as the switch to mass-law materials.
   - "2. Built-in EM Materials": the table with its per-material frequency range, checked against P.2040-3.
   - "3. Principled BSDF": unchanged unless it contradicts `obj_file_read`.
   - "4. References": the standards (P.2040-3, P.833-9, 3GPP TR 38.901) and the page. The Wikipedia list on the
     physics goes; the page has the references.
   - The acoustic default section goes, replaced by one sentence pointing to the page's starting rows.
   - The numbering of the MD blocks may change; keep the order.
4. **The three neighboring fixes**, then the checks of the hard rules.
5. **`check_docs.py`** extended, and it passes.
6. **`md/acoustic_manager.md`** is the manager's file; leave it alone.
7. **Report section 2 and stop:**
   - the line counts before and after;
   - the diff scope;
   - the counts;
   - the P.2040-3 finding with its source;
   - the list of what went where.

## Traps

- `extract_html.py` folds list continuation lines into the item and treats an indented `|` line as a table row. Write
  tables at column 0 or follow the existing idioms, and look at the generated HTML.
- `<br><br>` is the paragraph break inside `SECTION_DESC` (the converter does not make paragraphs there).
- `obj_file_read` has two material systems, the visual `.mtl` side and the EM/acoustic table; the BSDF part is the
  visual one. Do not mix them.
- The built-in table defines only `a`, `b`, `c`, `d`, `att` (`fRef` 1 GHz, `μ` 1); every other column is absent and
  takes its default.
- The material-model page uses LaTeX (it is read as Markdown); the formats page cannot. Formulas there stay in code
  spans.
- Run the test binary from the repository root; `python3 -I` puts no script directory on `sys.path`; the PDF of an
  ITU recommendation may need a manual download (`itu.int/rec/R-REC-P.2040`). If it cannot be fetched, report that.
- Scratch is not kept: the report states every fact the page relies on.
