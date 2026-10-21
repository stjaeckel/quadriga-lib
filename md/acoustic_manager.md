# The acoustic work — manager handoff

This file is the manager's memory. A manager session reads it first, works from it, and updates it after every step
(state, roadmap, pending decisions). When a manager's context fills up, a new manager reads this file and continues;
nothing depends on one chat. Last updated: 2026-10-21, after commit 32c55b2 (cleanup tasks 1–3 done, task 3 committed 23ad1b6; the Data Formats
material part done and reviewed, waiting for Stephan's commit; `main` is the acoustic work's branch again).

## 1. Roles

- **Stephan** (radio and acoustic engineer, maintains the library) decides every open point, checks results himself,
  and commits. He often takes the manager's recommendation; the manager still lays out the options.
- **The manager** writes briefs and kick-off prompts for worker sessions, reviews their results, prepares Stephan's
  decisions and the commit messages, and keeps this file and the roadmap current. It does not edit `src/` or tests and
  does not run experiments beyond a quick read-only check of a worker's claim. Its own edits: this file, briefs
  (`md/*_brief.md`), small document corrections Stephan asks for, `tools/acoustic`.
- **Stephan's veto (2026-10-19): no new items.** The goal is a good, usable system, not a perfect one. Side findings
  are recorded in the report (at most a one-line limitation in the documents) and dropped; no new open item, study or
  roadmap entry from them, and no work on fringe edge cases. Size every effect for typical scenes before proposing work.
- **Workers** (one fresh session per item) follow a brief in phases, write `md/<item>_report.md`, stop after each
  phase, and never commit.

## 2. The workflow

1. **Brief.** The manager writes `md/<item>_brief.md` (structure: Why; Facts at the start; Hard rules; Phase 0 analysis
   in scratch; Phase 1 fixtures; Phase 2 source; Phase 3 documents; Traps) and a one-line kick-off prompt:
   `Read md/<item>_brief.md in full and follow it. Start with the baseline check and phase 0; stop after phase 0 with
   the report in md/<item>_report.md.`
2. **Phases with stops.** analysis → Stephan's decisions → fixtures first (they fail on today's library, nothing else
   moves) → source → documents. Each phase ends with a report and a stop.
3. **Review.** Stephan says a phase is done (or pastes the worker's summary). The manager reads the report from disk
   and the diff (`git diff --stat`, `git diff src/`), runs the checklist (section 3), and answers with: what checks out,
   what does not, the decisions in the format of section 2.1, and a paste-ready message for the worker.
4. **Commit.** After the last phase the manager writes the commit message (style below; saved in the scratchpad for
   `git commit -F`) and the `git add` line; Stephan commits. The worker regenerates the baselines before the commit. The worker session can then be archived.
5. **Update this file**: state, roadmap, pending decisions.

**2.1 Decisions are presented as:** what Stephan is deciding; the options; trade-offs and implications; what is correct
(physics first, then the library's rules); a recommendation. One table or list per decision, a paste-ready message at
the end.

**2.2 Commit messages:** title "<area>: <what changed>" (e.g. "Acoustic: …", "Ray tests: …"), body in plain physical and code terms with the measured
numbers, the line count of `src/ray_mesh_interact.cpp`, the three counts, and the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Stephan keeps it). **No internal labels or documents**
(Stephan, 2026-10-17): no item ids (M10, D2-b, A12, N4 …), no `md/` files, briefs, reports or their sections, no
"round 1"; they are transitory and will be deleted. Test case names, source and tool paths are fine.

**2.3 Working with Stephan (lessons, 2026-10-14 to 21):**
- Explain decisions in plain physical terms first (what the quantity is, what happens, how big it is in typical
  scenes: indoor rooms, site models in local coordinates); he asked twice for a plainer explanation before deciding.
  Size every effect for typical scenes; say plainly when something is a fringe case (his veto, section 1).
- Before setting up a new task, ask him the open choices (scope, strictness, order, stops) with the question tool, a
  recommendation first, then write the brief.
- Verify a worker's factual claims that a decision rests on (a "dead" helper was in fact called; a claimed cause was a
  test-fixture artefact): read the code, rerun the suites, recompute key numbers.
- Cleanup and document tasks use a plan stop (inventory, what changes, what is removed and why) and a stop before the
  commit, not the fixtures-first phases.
- When a worker's context nears full, a fresh worker continues the next phase from the brief and the report on disk.
- In the desktop app, a decision question in the next-step box ends the turn and hides any text after it: write the
  explanation and the paste-ready message in the reply first, then ask.
- Every brief and prompt carries the resource rule (section 3) and "no new items".
- Public Markdown pages: plain `$…$` math (Stephan's viewer shows the dollar-backtick form raw) and one line per
  paragraph or list item (2026-10-21).

## 3. The review checklist

Run on every worker phase; say which items were checked.

- **Baselines**: the worker started from the committed counts; a plain what-if copy reproduced them (only the seed
  line differs).
- **`[wave]` identical** (`test_wave.txt` line for line, timing lines excluded) and the stored twins
  (`tests/data/wave_twins.tsv`) unchanged and asserted. A change that moves a `[wave]` row stops the phase.
- **Every moved row accounted for**: in `[acoustic]` and the whole suite only the intended cases' rows; the counts add
  up (new assertions = the difference).
- **References independent**: fixtures write their reference from formulas in the test (`lw::stack`, the limp mass
  law, a column definition), never from library code; bounds 2 to 3 times the measured maxima.
- **No grid artifacts**: a peak or dip quoted as a frequency comes from a fine sweep (`lw::peak_f`, 1/96 octave), not
  from the nearest grid point. (A18's "160 Hz at 30°" was 148.6; "315 Hz at 45°" was 298.8.)
- **Run time**: `--durations yes` as CPU time or alternated runs with the load stated (do not wait for an idle
  machine: another coding session runs in parallel and races for it too); a new case well under 0.5 s; the whole
  suite about 59 s summed per case.
- **Resources** (Stephan, 2026-10-19): at most 12 threads for this work, compiler and tests alike
  (`cmake --build build_linux --parallel 12`, `OMP_NUM_THREADS=12` or fewer; numpy pools `OMP_NUM_THREADS=1` per
  worker, at most 12 workers); never two suites at once; ASAN, AFL and other heavy tools only when the task justifies
  them, never by default, on the resource-saving end. Every brief and prompt carries this.
- **Walkers**: collect a path in air only outside the structure (`lw::walk` does; `bf::walk` / `cf::walk` do not and
  can hide energy creation, `md/wave.md` 11); a followed series reports its dropped power; recursion needs
  `ulimit -s unlimited` or `lw::walk`.
- **Conditions on hit codes are complete**: compare against the `out_typeN` table in `src/ray_mesh_interact.cpp` (every
  code that starts inside a body: 1, 5, 7, 13, 21, 29, +32 for total reflection; shared codes 69 / 71 need the
  re-derived sheet-ness).
- **EM first**: a change in shared code is measured on EM types too; if a behavior is decided for EM, EM rows are
  asserted (A21 is the example: EM created R + T 42.19 before the kill).
- **Energy**: R + T ≤ 1 from a lossless side on every new configuration, both tracers, summed and followed.
- **Documents**: `python3 tools/acoustic/check_docs.py` passes (tables, source keys, case names, counts against the
  baselines, spelling); decisions in `md/acoustic.md` 10, open items in 6, candidate status in 4; American English and
  the style of the file (plain statements, numbers with units, test names in italics, no emphasis adverbs).
- **Scratch is not kept** (`make clean` removes `build_linux/scratch`): every number a decision rests on is stated in a
  report or a document; reusable instruments go to `tools/acoustic`.
- **One writer** for `md/acoustic.md` and `md/wave.md` at a time (the active worker in its document phase, or the
  manager), never two sessions at once.

## 4. Current state (2026-10-21)

- Last commit **32c55b2** (`.gitignore`: `__pycache__/` everywhere; the stray `tools/acoustic/__pycache__` file was
  removed from the index in 23ad1b6). Before it **23ad1b6**, task 3: the material-model page rewritten from the code,
  seven API-block edits, `check_docs.py` checks the page. Before them, e34dd2a. On 2026-10-21 three commits from the fixes manager (branch `fixes`,
  `/sjc/quadriga-lib-fixes`) landed on `main`; from then on `main` is ours again, and the fixes work touches it only
  for an agreed merge:
  - **f0d093d**, merge of `fixes`: `ray_triangle_intersect` (all kernels) reports hits at equal distance in index
    order (the AVX2 kernel used to swap them), the AVX2 kernel forms its vectors in double before rounding, a ray in
    a bounding-box plane is no longer rejected, `fbs` / `sbs` may alias `orig` / `dest`; also `xpr_update`,
    `path_to_tube`, and documentation of `calc_diffraction_gain` (scalar mode takes the EM-equivalent frequency in
    the whole C++ core). None of the ray files or ray tests changed. Checked by the manager: generic `[wave]` and
    `[acoustic]` identical to the baselines; AVX2 `[wave]` identical before and after the merge; AVX2 `[acoustic]`
    moved 146 values, all in the three-face tie of *Sheet between two bodies - nothing is created and the wave left
    in air inside the far body is killed* (before: box exit + wall entry, R 0.9998; now the index-order pair, as
    generic), so the AVX2 `[acoustic]` rows now equal the generic baseline number for number. AVX2 `[wave]` still
    differs from generic in 18 `nh` values (hits reported on the segment, 3 instead of 2 at 50°) of *Unentered
    branches - a corner with a sheet is the sheet alone once the state is kept*, every coefficient identical; this
    was already so before the merge.
  - **a7f1954**, **e34dd2a**: `QD_TEST_DIAG` replaced by per-area switches in `tests/catch2_tests/qd_test_diag.hpp`
    (`QD_DIAG_RAY` the ray tests, `QD_DIAG_SOS`, `QD_DIAG_CONF`, `QD_DIAG` all areas; set and not `0` = on);
    `ray_test_helpers.hpp` uses `QD_TEST_DIAG_SWITCH("QD_DIAG_RAY")`; the commands in `md/wave.md`, `md/acoustic.md`,
    this file and the task-3 brief were updated by the fixes manager.
- Before them: dc47ff6 (ray-test cleanup), 71b894b (AVX2 test fix: the three-face tie asserted for
  every face pair), a5bdfb7 (merge of the `fixes` branch from another session), 29e1f05 (source cleanup of
  `src/ray_mesh_interact.cpp`), 6307a64 (traced-leg precision), de60194 (finite panel, `side` column), bc24b0c (`m` a
  switch, `coiA` removed), 2c5be17 (peaked face as series impedance), 238444d, 04b92bd, 4720a92 (round 1).
- `src/ray_mesh_interact.cpp` 4731 lines (4725 before the task-3 API-block edits), restructured: an 84-line header overview, seven sections in one anonymous
  namespace, the five public API blocks shortened (`md/cleanup_src_report.md` 1.6 has the rename table old → new).
- Counts: `[wave]` 163474 / 178, `[acoustic]` 42741 / 59, whole suite 1344398 / 1264 (generic build; the merge
  added 30 cases outside the ray tests); AVX2 build 1346180 / 1265. Baselines are generated **with `QD_DIAG_RAY=1`** for `[wave]` / `[acoustic]` and `QD_DIAG=1` for the
  whole suite (the diagnostic tables print only with these environment variables; a plain run prints nothing; the
  variables were renamed from `QD_TEST_DIAG` on 2026-10-21, see `tests/catch2_tests/qd_test_diag.hpp`): commands in `md/wave.md` 0, 12 threads.
  `python3 tools/acoustic/check_docs.py` passes. Checked by the manager 2026-10-21: `[wave]` and `[acoustic]` rerun
  at dc47ff6 and again at e34dd2a, `rowdiff` identical. `test.txt` regenerated at e34dd2a with `QD_DIAG=1`: the
  printed lines are unchanged by the merge (23821), only its summary moved to 1344398 / 1264 (the quoted count in
  `md/wave.md` 0 updated with it; before, `check_docs.py` passed only because both were stale).
- Builds: `build_linux` (generic, the suite's build); an AVX2 build for checks at `build_linux/scratch/avx2`
  (`cmake -S . -B build_linux/scratch/avx2 -DCMAKE_CXX_COMPILER=g++-12 -DENABLE_AVX2=ON -DENABLE_TESTS=ON`; scratch,
  may be gone after `make clean`).
- Manager's verification routine for a worker's claim: rebuild with `--parallel 12`, rerun `[wave]` and `[acoustic]`
  one after the other with `QD_DIAG_RAY=1 OMP_NUM_THREADS=12`, compare against the baselines with
  `python3 -I tools/acoustic/rowdiff.py test_wave.txt <new run>` (number by number), and check the diff scope
  (`git diff --stat`, `git diff --quiet -- src include api_python api_mex`).
- Documents (internal, transitory; they stay until the parameter project): `md/acoustic.md` (the acoustic reference),
  `md/wave.md` (the wave reference), briefs and reports per item (`md/*_brief.md`, `md/*_report.md`; the latest:
  `cleanup_src_*`, `cleanup_tests_*`, `precision_*`, `grazing_*`, `m10_m15_*`, `m6_*`).
- Tools: `tools/acoustic` (`rowdiff.py` two runs number by number, `cases.py` moved rows per case, `check_docs.py`,
  `txlib.py` / `txfit.py` / `tl_refs.json` the transmission formula copy and fit, `gzlib.py` finite-plate reference
  models, `paris_cf.py`, `sabine_to_random.py`, `band_probe.cpp`; README).
- Literature review: `/sjc/accoustic_research/review.md` (moved there 2026-10-21, the same disk location as
  `~/sjc/accoustic_research`; a local git repository, read-only for the manager and workers), data
  `data/material_properties.csv`, keys in `references/index.csv` (every key `md/acoustic.md` cites resolves there;
  `check_docs.py` has no list of known gaps any more).
- Left for Stephan, no item: regenerate `html_docu/` after the API-block changes; the MATLAB `ray_mesh_interact`
  wrapper and its test will be rewritten later (its test at line 113 expects the field `|T|²`).

## 5. Decided (pointers; the text is in `md/acoustic.md` 10)

P1–P9 (P6 reflect side first; P9 scattering out of scope); N1–N7 (round 1: the plate rule, nothing reflected back into
a mass-law body, `res*` and `coi*` in `X₀`, the two-parameter mapping); A10, A11; M5 (resonator law of the `tf` peak);
M6 (a peaked face is a locally reacting series impedance; over-damped default, `tfQ < 0` under-damped); M9 (an
unpartitioned built construction is a wave-medium leaf on a box of near-air material); M10 (`m` > 0 switches the mass
law on at 20 dB per decade); `coiA` removed; M11, M13 dropped; M7 dropped; M14 closed (superseded by the finite panel:
Davy's forced radiation efficiency, column `side` = √(panel area), default 3.29 m, 0 the infinite plate); transmission
references fitted on the 0–90° average; traced legs count the relaunch offset (rule of use: scenes in local
coordinates near the origin); A13 (tiered rule for `tf` references), A14, A15, A16, A17, A19 revised, A21; M8
(Stephan, 2026-10-21: the material-model page teaches the two-parameter mapping only, the one-parameter form in one
sentence; recorded in `md/acoustic.md` 10 by the task-3 worker).

## 6. Pending decisions (for the parameter project, none blocks anything)

A13 refinement (a conversion whose fitted σ is off a measured σ by more than about 1.5 times is marked); A14 (two
Scrosati rows out); A17′ (class 8 as the family of resonant absorbers); M15 (the bending loss as a resistance; decide where `coiQ` comes from); M16 (a relaxation term on
`μ` for curtains).

## 7. Open items without a decision (`md/acoustic.md` 6; no work planned, Stephan's veto)

A8 (low priority), A20 (a wall's power-law `tf` behind a porous fill), A21 remainder (the three-face tie: since f0d093d every
kernel reports hits at equal distance in index order, so generic and AVX2 agree; both outcomes stay asserted since
71b894b).

## 8. Roadmap (next first)

Done in this cleanup series: task 1, the source cleanup (29e1f05; `md/cleanup_src_report.md`); task 2, the ray-test
cleanup (dc47ff6; `md/cleanup_tests_report.md`: four redundant cases removed by per-case coverage, diagnostics behind
`QD_TEST_DIAG`, the ray cases 24.9 → 5.9 CPU-s, the whole suite 38 → 19 s); task 3, the material-model page
(23ad1b6; `md/cleanup_doc_report.md`; details in item 1 below, kept for the record).

1. **Task 3: the material-model document** — **done, committed 23ad1b6 (2026-10-21)**. Brief written 2026-10-21 (`md/cleanup_doc_brief.md`, report to come
   in `md/cleanup_doc_report.md`), **approved by Stephan 2026-10-21**; a fresh worker runs phase 0 with the kick-off prompt
   `Read md/cleanup_doc_brief.md in full and follow it. Start with the baseline check and phase 0; stop after phase 0
   with the report in md/cleanup_doc_report.md. At most 12 threads, one suite at a time, no new items.`
   Phase 0 done 2026-10-21 (`md/cleanup_doc_report.md` 0): baseline reproduced at f33c8cb; outline user part about
   760 lines in 14 sections plus developer part about 100; 16 rules of use; examples EM `itu_concrete` 200 mm,
   `itu_glass` / `irr_glass` 6 mm at 3.5 and 28 GHz (P.2040 slab closed form, 3GPP TR 38.901 losses), acoustic
   concrete 152 mm and glass 4 mm (the decided fit rows, second references Mahn 2023 CON150 and Quirt 1981), glass
   wool 50 mm (Komatsu at the measured σ, Dupont 2023); seven API-block edits; 18 `format_materials.md`
   contradictions. Manager's review: the code claims behind the API edits checked and right (the pole needs
   `resQ > 0` and `m = 0`, `eta()`; a held wave leaving a plate into a wave medium is refracted as from air,
   `interact_with` 827–844; `prev_embedded` written for a solid, 4492; a sheet composite carries `qd_mass_amp` and
   `stack_layer_phase`); the brief's `map_to_itu_materials` belongs to `mitsuba_xml_file_write` (the manager's
   error); the U+2212 delimiter row in `md/acoustic.md` 2.7 (line 383) confirmed. Review notes: user part long
   (760 against the old page's 738 in all; Q7 proposed: about 600); D2 without the machine and team rules (12 threads,
   shared machine, `pkill`); no "U1" / "D1" labels on the page (the label check would flag them).
   Decided by Stephan (2026-10-21): Q1 keep the fit rows with `coiQ` (if M15 changes `coiQ`'s meaning, the two
   mass-law examples are re-measured); Q2 keep the Data Formats pointer (worker's text, the CSV format), no edit of
   `format_materials.md` in this task (a separate session follows, roadmap item 2); Q3 the 13 fitted constructions
   go on the page as preliminary starting rows with a note that they will change (also the starting point of the
   parameter project), worst deviations re-evaluated with `txlib.py`, no refit; Q4 title "The quadriga-lib Material
   Model and Ray-Surface Physics"; Q5 both EM references (3GPP differences stated, not asserted); Q6 every limit
   size from today's library: the `QD_DIAG_RAY` diagnostic output first, else a scratch probe, else dropped; Q7 user
   part about 600 lines as a guideline.
   Phases 1–2 done and reviewed 2026-10-21 (report 1, 2): the page 538 lines (one line per paragraph, Stephan's
   request; Part I 490, Part II 48), plain `$…$` math (the dollar-backtick form showed raw in Stephan's viewer);
   seven API-block edits, all inside `/*!MD` (checked line by line), `src/ray_mesh_interact.cpp` 4731 lines;
   `check_docs.py` checks the page (columns 26 of 26: the brief's "27" was wrong, labels, math, anchors). Manager's
   checks: rebuild, `[wave]` / `[acoustic]` rowdiff identical (141128 / 112451), `check_docs.py` passes, the page
   read in full, the mass-term factors and the `tf` peak checked against `mass_X0` / `Material::tf`. Two nits fixed by the
   manager on Stephan's request: `k0 = 2π·10⁹ f / c0` (f in GHz, as the section states); reference [16] with its
   year (2022, the copyright year in library catalogs; released 2021). Side findings recorded: `band_probe.cpp` has no `attB` column (every fit row carries one; for the parameter
   project); `tools/acoustic/__pycache__/txlib.cpython-314.pyc` was committed by accident in bc24b0c.
   **If `coiQ`'s meaning changes (M15), the page's 11.1, 11.2 and 11.4 are re-measured.**
   Stephan's answers before the brief (2026-10-21): **Markdown with LaTeX** (GitHub math, same file and name, not
   generated); **API blocks** may be corrected comment-only inside `/*!MD` after his OK at the plan stop; **developer
   part** about 100 lines, a source map plus how a change is checked (takes over `md/wave.md` 10–11); **worked
   examples**: EM from the ITU table, acoustic a single-leaf wall, a glass pane or window and a porous absorber on a
   rigid wall, from the review's data by the decided rules, measured with the library; **mapping** two-parameter
   only (M8 decided). Phases: 0 inventory and plan (absorption table of `md/wave.md` / `md/acoustic.md`, outline,
   example plan, API-block correction list, `format_materials.md` contradictions report-only, `check_docs.py`
   extended to the page) → stop; 1 examples measured, page written, corrections applied; 2 `md/` updates (A9 closed,
   M8 in 10, `md/wave.md` line 495) and final checks → stop before the commit. Decided by Stephan (2026-10-20):
   - the file `html_docu/quadriga_lib_material_model.md` (738 lines, hand-written, last edited 2026-10-13 (807a1d6), before the
     acoustic work, so stale throughout) is rewritten to match what is implemented;
   - **one page, two sections**: a user part (the physics, every material column with its unit and meaning, the rules
     of use, the limits) and a short developer part (how `src/ray_mesh_interact.cpp` is structured, after task 1: its
     header overview and sections are the basis);
   - **published** with the other `html_docu` pages (find how the pages are generated: `tools/extract_html.py`,
     `tools/extract_md.py`, `tools/html_parts`; Stephan regenerates `html_docu`);
   - it **absorbs what users need from `md/acoustic.md` and `md/wave.md`** (the rules of use and the limits, for EM and
     for acoustics), so those transitory files are no longer needed for that; the `md/` files themselves stay until the
     parameter project;
   - the column reference `tools/data_formats_md/format_materials.md` is **not** in this task;
   - workflow as for tasks 1–2: a plan stop (outline, what comes from where, what is dropped) and a stop before the
     commit; no source or test change; written from the code (the source is the normative reference), with the
     formulas as implemented; EM first (the library's primary use), acoustics as the mapping and the mass-column
     switch; the API blocks in the source are the public reference for the five functions, the page must agree with
     them.
2. **`tools/data_formats_md/format_materials.md`** (Stephan, 2026-10-21: a separate session after task 3 updates
   the published column reference): 18 contradictions with the code are listed in `md/cleanup_doc_report.md` 0.7
   (missing columns `thk`, `tf*`; the transmission gate that no longer exists; `m` as a slope; no scalar refraction;
   the one-parameter rigid family; the acoustic `concrete` default row without `tf`). **Brief written 2026-10-21**
   (`md/formats_doc_brief.md`, report to come in `md/formats_doc_report.md`), waiting for Stephan's review. Decided
   by Stephan before the brief (2026-10-21): a **lean format reference** (the table format, all 26 columns with unit,
   default and a one-line meaning, the built-in EM table, the BSDF part; all model prose replaced by a link to the
   page); the **acoustic default section removed** (a pointer to the page's starting rows; `data/materials_ac.csv`
   and `tools/acoustic_parameter_mapping/` untouched until the parameter project); **three neighboring fixes**,
   comment-only (the C++ `obj_file_read` block's name matching: the code takes the part before the first dot, line
   89; the "detailed description … Data Formats" pointer in the three `obj_file_read` blocks also names the page;
   the page's section 9 range "valid 1–40 GHz" against the per-material ranges up to 100 GHz, checked against
   P.2040-3 Table 3); **one stop**, before the commit. Kick-off prompt: `Read md/formats_doc_brief.md in full and
   follow it. Start with the baseline check; stop before the commit with the report in md/formats_doc_report.md. At
   most 12 threads, one suite at a time, no new items.`
   Done and reviewed 2026-10-21 (`md/formats_doc_report.md`): `format_materials.md` 307 → 179 lines (the table, its
   CSV form and reader rules, 26 columns, the built-in EM table with range and source per row, BSDF, references);
   the three fixes inside the MD blocks (checked hunk by hunk by the manager); P.2040-3 Table 3 read from
   `references/itu_rec_p2040_3.pdf` (printed p. 24, PDF page 26) by the manager too: the page's "1–40 GHz" was
   wrong, the per-row ranges right, floorboard 50–100 GHz; seven built-in rows (`textiles` … `water_ice`) are
   library estimates (commit 75f43b9), `itu_vegetation` 0.03–30 GHz (P.833-9 Fig. 2). Manager's checks: whole suite
   1344398 / 1264, output identical apart from the seed; `check_docs.py` passes (formats 26 of 26); `formats.html`
   built into scratch, no raw Markdown. Side findings outside the acoustic work (report 2.6), proposed for the fixes
   manager: `include/quadriga_tools.hpp` 111 calls `calc_diffraction_gain`'s `mtl_ind` 0-based (1-based in code);
   `obj_file_write` block defaults (`side` 3.29 written) and `thk` among the extra columns; `std::stod` accepts
   `1.5x`, `nan`, `inf`; the `.mtl` reader fails on CRLF files and indented keywords. Five other API blocks keep the
   old "see Data Formats" sentence (`api_mex/ray_mesh_interact.cpp`, `api_mex/calc_diffraction_gain.cpp`,
   `api_python/qpy_calc_diffraction_gain.cpp`, `api_python/qpy_obj_file_write.cpp`,
   `api_python/qpy_mitsuba_xml_file_write.cpp`); page section 9 says `irr_glass` follows TR 38.901 (only its `att`).
3. **The parameter session** — to become a project of its own (Stephan, 2026-10-20); kept here only so its context is
   not lost; not to be started from this file: the material table from the review's data with the A13 / A14 / A19
   rules (`tools/acoustic/sabine_to_random.py`, `txfit.py`), the acceptance classes of `md/acoustic.md` 9, defaults for
   `side`, and the open decisions of section 6.

## 9. Traps (collected)

- `make clean` removes `build_linux/scratch`; rigs named in documents may be gone.
- Run a relinked test binary from the repository root; never two suites at once; `timeout` on probes; check the load
  (`~/.claude/CLAUDE.md`), single-threaded probes under load.
- A probe that includes a test file runs every case of it unless given a test spec; a case name with a comma needs `\,`.
- `pkill -f <pattern>` from a shell whose command line contains the pattern kills that shell; kill by pid.
- A glued sheet lies in front of the body it is glued on; an isolated sheet about its mesh plane.
- A material identical to air is air: a box of air needs `μ` 1.000001.
- A rigid backing (`r = 1`) is not the reference for a two-parameter wall: use its half-space (or the pad on it).
- `ray_progress` allows 255 interactions: a meshed gap behind a leaf is out of reach; with the threshold at 1.0 some
  geometries run away (A21 before the kill: 12.7 GB, no end).
- A followed series of a lossy body between strong reflectors diverges (round trip above one): only summed rows count.
- `python3 -I` puts no script directory on `sys.path`.
- Pasted worker summaries are shortened; read the report and the diff.
- A baseline with the right counts can still be a plain run: check that `test.txt` carries the ray tables (about
  1.9 MB with `QD_TEST_DIAG=1`, 34 KB without).
- `data/materials_ac.csv` and `tools/acoustic_parameter_mapping/` are from before the acoustic work: never a
  reference.
