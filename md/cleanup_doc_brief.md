# Brief — the material-model page (task 3 of 3)

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/cleanup_doc_report.md` (create it), one section per phase.

## Why

`html_docu/quadriga_lib_material_model.md` is the public companion document of the material model and the ray-surface
physics. It is linked from the API blocks of `ray_mesh_interact` and `ray_state_update`, so a user (or an AI) reading
the C++ API lands on it. It was written before the acoustic work and the two cleanup tasks (last edited 807a1d6,
2026-10-13) and is stale throughout. Examples: dispatch tables and macros that no longer exist (`APPLY_TRANSITION`),
internals renamed in task 1 (38 hits against the rename table in `md/cleanup_src_report.md` 1.6), the old transmission
gate, and none of the mass law, the plate rule, the finite panel, the peaked face or the leaf route. What users need
today (the rules of use and the limits) is spread over two internal, transitory documents, `md/wave.md` and
`md/acoustic.md`, which nobody outside this work reads and which will be deleted after the parameter project.

Task 1 (the source cleanup, 29e1f05) and task 2 (the ray-test cleanup, dc47ff6) are done. This is the last task of
the cleanup series.

## Decisions (Stephan, 2026-10-20 and 2026-10-21)

- **Rewrite** the page to match what is implemented. The source is the normative reference. Formulas are given as
  implemented.
- **One page, two parts.**
  - A **user part**: the physics; every material column with its unit, default and meaning; the rules of use; the
    limits; the acoustic mapping; worked examples.
  - A **developer part** of about 100 lines: a map of `src/ray_mesh_interact.cpp` (the per-event call sequence, the
    seven sections, the state words and the ledger, where the hit-code tables are) plus **how a change is checked**
    (the test tags, `QD_TEST_DIAG`, the baselines, the stored twins, `rowdiff`, and the rules: EM and wave rows first,
    energy). That part must survive the deletion of `md/wave.md` (its sections 10 and 11 move here). The source's
    84-line header overview and its section comments are the basis; do not copy them, explain them.
- **Format: Markdown with LaTeX math** (GitHub syntax: `$…$` inline, `$$…$$` display), as the old page. Same file
  name and place (`html_docu/quadriga_lib_material_model.md`): it is hand-written, `make documentation` does not
  generate it, and the API blocks link to it by that name.
- **EM first** (the library's primary use). Acoustics comes second, as the mapping plus the mass-column switch.
- **The acoustic mapping: two-parameter only** ($\varepsilon = n z$, $\mu = n/z$, $z = Z_\text{air}/Z$,
  $n = c_\text{air}/c$; `md/acoustic.md` 1). The one-parameter form ($\varepsilon = (Z_\text{air}/Z)^2$, $\mu = 1$)
  gets one sentence: older tables use it, its phase at a rigid half-space is wrong ($2\theta$), and a porous layer on
  it absorbs up to 0.39–0.53 too much. This settles the pending M8; note it in `md/acoustic.md` 10 in phase 2.
- **Worked examples.**
  - EM: two or three materials of the built-in ITU table (`obj_file_read`, `map_to_itu_materials`), at one or two
    frequencies.
  - Acoustics: three constructions, none touched by a pending decision:
    1. a **single-leaf wall** (masonry or gypsum: the mass law, the finite panel `side`);
    2. a **glass pane or window** (`m` with `tf`, the `res*` / `coi*` columns);
    3. a **porous absorber on a rigid wall** (a wave-medium layer in the two-parameter mapping).
  - Column values come from the literature review's data and the decided rules (below), every number is **measured
    with the library** (not computed by hand), and the deviation from the measured reference data is stated.
- **API blocks**: where a `/*!MD … MD!*/` block in `src/ray_mesh_interact.cpp` disagrees with the code or the page,
  the worker lists it at the plan stop (old text, new text, reason). After Stephan's OK it is corrected, **comment-only
  inside the MD blocks** (the page link lines included). No other change under `src/`.
- **Not in this task**: the column reference `tools/data_formats_md/format_materials.md`. Every place where it
  contradicts the code goes into the report, one line each, and nothing more. Also out: the MATLAB and Python API
  documentation, the generated `html_docu` files (Stephan regenerates them), and the `md/` documents beyond the small
  updates of phase 2.

## Facts at the start

- Last commit dc47ff6. `git status` should show only `md/acoustic_manager.md`, the review-path line in three
  `tools/acoustic` files, and this brief. If anything under `src/` or `tests/` is modified, stop and ask Stephan.
- Counts (`md/wave.md` 0, with `QD_TEST_DIAG=1`, 12 threads): `[wave]` 163474 assertions in 178 cases,
  `[acoustic]` 42741 in 59, whole suite 1343848 in 1234 (generic build). `python3 tools/acoustic/check_docs.py`
  passes.
- `src/ray_mesh_interact.cpp`, 4725 lines:
  - the header overview (lines 5–88);
  - seven sections in one anonymous namespace;
  - the five public functions after their API blocks: `medium_gain` (line 2445), `interface_gain`, `refractive_index`,
    `ray_mesh_interact` (2598), `ray_state_update` (3211).
  - The material columns are read in `MaterialCols` (line 257): 27 in all, `fRef` plus `a b c d e f g h thk att attB
    alpha alphaB m resF resQ resS coiF coiQ tf tfB tfP tfF tfQ side`.
  - The page is linked from the "See also" lines of the `ray_mesh_interact` and `ray_state_update` blocks (2733,
    3409). The `ray_mesh_interact` block also sends "a detailed description of the material model" to the Data
    Formats page (`formats.html`, generated from `format_materials.md`).
- The old page, 738 lines: sections 1–10. The physics in 2–6 is partly reusable after checking it against the code;
  the state machine and dispatch tables in 7–8 are stale; 182 lines carry LaTeX.
- The sources to absorb:
  - `md/wave.md` (581 lines): 0 scope, 1–7 the conventions, 8 the accepted limitations, 9 the attachment points,
    10 the rules for changes, 11 the test instruments.
  - `md/acoustic.md` (1478 lines): 1 the mapping, 2.1–2.8 the machine as built, 5 what stays out of reach,
    7 the accepted limitations, 9 the acceptance test; the decisions in 10.
  - The rules of use are scattered: search both files for "rule of use" and read every "Accepted limitation".
- Data for the examples:
  - The literature review is at `/sjc/accoustic_research` (moved 2026-10-21): `review.md`,
    `data/material_properties.csv`, `references/index.csv` (keys to full citations), the PDFs in `references/`.
  - `tools/acoustic`:
    - `tl_refs.json`: the 13 transmission constructions with measured octaves, e.g. glass 4 / 6 mm, 4-16-4 double
      glazing, brick 95 / 203, concrete 152, block 190, gypsum 12.5 (a model, not measured).
    - `txfit.py` fits a mass-law material's columns; `txlib.py` is the formula copy.
    - `sabine_to_random.py` converts chamber absorption to random incidence (the decided tiered rule A13; Dupont's
      50 mm glass wool is the validated class-4 case).
    - `paris_cf.py` gives a face's random-incidence absorption.
    - `band_probe.cpp` sends one ray through `ray_progress` with a band of frequencies (README).
  - The decided rules for converting data to columns are in `md/acoustic.md` 9 and 10: the A13 tiered rule, A16
    tolerances, A19 revised (`f₀` measured, else 60 / 50), the fit on the 0–90° average, the default `side` 3.29 m.
- **Not references**: `data/materials_ac.csv` and `tools/acoustic_parameter_mapping/` (including its
  `modelling_guidelines.md`) are outdated, from before the acoustic work. Do not cite them or take values from them.
- How the pages are published: the `documentation` target in `GNUmakefile` runs `tools/extract_html.py` (HTML pages
  from `tools/html_parts` and the `/*!MD` / `/*!SECTION` blocks) and `tools/extract_md.py` (the API `.md` files).
  The material-model page is a static file next to them.

## Hard rules

- **Documents only.** No code change under `src/`, `include/`, `api_*`, `tests/`. The only edits under `src/` are the
  agreed API-block corrections, inside `/*!MD … MD!*/`. Checked with `git diff -U0 src/`, a build, `[wave]` and
  `[acoustic]` (`rowdiff` identical), and `tools/extract_md.py -a cpp` into scratch.
- **Written from the code.** Every formula, default, unit and statement is checked against the source, not copied
  from the old page or the `md/` files (they can be wrong or stale). Where the page and an API block would differ,
  the code decides, and the API block goes onto the correction list.
- **Every number is measured**: a coefficient, an angle, a level or an accuracy comes from a run of the library (a
  scratch probe, `band_probe`, or a test case's printed row), never from hand arithmetic. Accuracy statements quote
  what a test asserts or what the worker measured, with the reference named.
- **A public document.**
  - No internal labels: item ids (A13, M8, N7, P9, D2-b …), "round 1", rigs, reports, briefs or `md/` files. The
    page outlives them.
  - Literature is cited in full (author, title, venue, year; ITU-R P.2040 with its edition), not by review key.
  - Test case names are allowed, in italics, where a statement is asserted (mostly in the developer part).
  - American English, plain statements, numbers with units, no emphasis adverbs.
- **What goes in.**
  - Rules of use: all of them; each is one thing a user must do or avoid, with the reason in a sentence.
  - Limits: only those a user can meet in a typical scene (indoor rooms, site models in local coordinates) or that
    change what they should model. Each gets one line with its size. Fringe edge cases stay in `md/` and are listed
    as dropped at the plan stop.
  - **No new open items** (Stephan's rule): a side finding is one line in the report, nothing more.
- **Resources** (another session shares the machine): at most 12 threads for compiler and tests alike
  (`cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or fewer); `OMP_NUM_THREADS=1` per worker for numpy
  pools of at most 12 workers; single-ray probes on one thread; never two suites at once; no heavy tools (ASAN, AFL).
- Do not commit; Stephan commits. The plan stop and the final stop are the two stops.

## Phase 0 — inventory and plan (no edit outside the report; stop)

1. **Baseline**: build, the three suites with the 12-thread commands and `QD_TEST_DIAG=1`, the counts above;
   `check_docs.py`.
2. **The old page against the code**: per section, what is still right, what is stale (with the code's statement),
   what is gone.
3. **The absorption table**: every section of `md/wave.md` and of `md/acoustic.md` 1, 2, 5, 7, 9, each mapped to
   a place in the new page or "dropped" with a reason (internal history, a decision record, a fringe case, a
   candidate). Every rule of use is listed by name. Every accepted limitation is marked kept (one line, its size) or
   dropped (why).
4. **The outline**: the sections of both parts with their contents in a line each and the expected lines per
   section.
   - User part, suggested order, adjust as needed: what the model is and which functions apply it; scene and mesh
     conventions; the interaction types; the columns (one table: column, unit, default, meaning, which media); the
     formulas as implemented; interface physics and energy; sheets, stacks and traced solids; what `ray_state_update`
     adds for a user (in-medium loss and phase, the series sum, `thin_slab_threshold`); EM use and examples;
     acoustic use (the mapping, wave media, mass-law materials, leaves) and examples; rules of use; limits;
     references.
   - Developer part: as in the decisions.
5. **The worked examples**: for each of the three acoustic constructions and the EM materials, the source data (file,
   row or construction, reference), the rule that turns it into columns (fit, conversion, measured `f₀`), the columns
   expected, what will be measured (normal and random incidence, which octaves), with which instrument, and the
   accuracy expected from the decided tolerances. A construction that needs a pending decision is replaced, with the
   reason.
6. **The API-block correction list**: every disagreement between an API block and the code or the planned page:
   block, line, old text, new text, reason. Include whether the "see Data Formats" sentence and the "See also" link
   text of the two blocks should point to the new page or change wording, and whether the three short blocks
   (`medium_gain`, `interface_gain`, `refractive_index`) get the link.
7. **`format_materials.md` contradictions**, one line each (report only).
8. **The checks to add to `tools/acoustic/check_docs.py`** for the page:
   - every `MaterialCols` column appears in the column table;
   - no internal labels or `md/` references;
   - American spelling and table column counts, as for the `md/` files;
   - `$` balanced per paragraph;
   - in-page anchors resolve.
9. **Stop** for Stephan's OK on the outline, the absorption table, the example plan and the correction list.

## Phase 1 — writing (after the OK)

- The worked examples first: measure them in scratch and write the numbers into the report (section 1). The page then
  quotes the report's measured values.
- Write the page part by part. Every formula is checked against the source line it comes from (note the function in
  the report).
- Extend `check_docs.py` as agreed (and its README line); it passes.
- Apply the agreed API-block corrections; run the checks of the hard rules.

## Phase 2 — final checks and the `md/` updates (stop before the commit)

- `md/acoustic.md`: A9 closed in 6 (the page is written); M8 decided in 10 (two-parameter only on the page, Stephan,
  2026-10-21); the one-line pointer that the user-facing rules now live in the page. `md/wave.md` line 495 (the page
  "out of date") updated. Nothing else in those files.
- A last read of the page against the code. Spot-check ten formulas or defaults chosen across the page and list them.
- Counts: page lines per part; the API-block diff (`git diff --stat`, MD-only); suites unchanged (the counts above,
  `rowdiff` identical); `check_docs.py` passes.
- Report section 2 and stop. Stephan reviews the page (also rendered on GitHub), regenerates `html_docu`, and
  commits; the manager writes the commit message.

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9: run binaries from the repository root; one suite at a
  time; `pkill -f` kills its own shell; `python3 -I`; a probe that includes a test file runs every case unless given
  a test spec; a case name with a comma needs `\,`; a material identical to air is air (a box of air needs `μ`
  1.000001); `make clean` removes `build_linux/scratch`.
- GitHub math:
  - `_` and `*` inside `$…$` can be read as emphasis; use the dollar-backtick form (`` $`…`$ ``) or a `$$` block where
    a formula has several of them.
  - A `|` inside math in a table cell needs `\|`.
  - `$` next to a letter or digit may not open math.
  - No `\newcommand`.
  - The old page's LaTeX rendered on GitHub and is the model to follow.
- Frequencies: the API takes Hz, the columns and `fRef` are in GHz, and the acoustic mapping scales by
  $c_\text{light}/c_\text{sound}$ (1 kHz ≈ 0.874 GHz). An acoustic example states both scales.
- A mass-law material's `ε`, `μ` set only its faces' reflection and critical angle; its transmission comes from the
  mass term, and no wave is traced in it (the held wave). Do not describe its inside as a medium.
- `tf` is the transmission factor of a face, not a transmission coefficient. It has two forms: an ideal transformer
  at an open face and a pad at a closed one. The old page and `format_materials.md` describe older forms.
- Old statements from `md/acoustic.md` 4 (candidates) are not the machine; only 2 (as built) and 10 (decided) are.
- The interaction types 1 and 4 keep the direction while `path_dirN` reports the Snell direction. Read the API block
  before describing the outputs.
- Scratch is not kept: the report states every number the page relies on, with the probe's input.
