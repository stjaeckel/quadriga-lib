# The acoustic work — manager handoff

This file is the manager's memory. A manager session reads it first, works from it, and updates it after every step
(state, roadmap, pending decisions). When a manager's context fills up, a new manager reads this file and continues;
nothing depends on one chat. Last updated: 2026-10-16, after commit 2c5be17 (6307a64; cleanup task 1 complete, waiting for Stephan's commit).

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
4. **Commit.** After the last phase the manager writes the commit message (style below) and the `git add` line;
   Stephan commits and regenerates the three baselines (`md/wave.md` 0). The worker session can then be archived.
5. **Update this file**: state, roadmap, pending decisions.

**2.1 Decisions are presented as:** what Stephan is deciding; the options; trade-offs and implications; what is correct
(physics first, then the library's rules); a recommendation. One table or list per decision, a paste-ready message at
the end.

**2.2 Commit messages:** title "Acoustic: <what changed>", body in plain physical and code terms with the measured
numbers, the line count of `src/ray_mesh_interact.cpp`, the three counts, and the trailer
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (Stephan keeps it). **No internal labels or documents**
(Stephan, 2026-10-17): no item ids (M10, D2-b, A12, N4 …), no `md/` files, briefs, reports or their sections, no
"round 1"; they are transitory and will be deleted. Test case names, source and tool paths are fine.

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

## 4. Current state (2026-10-16)

- Last commit **6307a64** (traced-leg precision). Before it: de60194 (finite panel, `side`), bc24b0c (M10, `coiA`
  removed), 2c5be17 (M6), 4035ac4 (this file), 238444d (M5, A21), 4720a92 (round 1 built).
- `src/ray_mesh_interact.cpp` 5211 lines.
- Counts: `[wave]` 165594 / 181, `[acoustic]` 42391 / 60, whole suite 1317534 / 1148 (baselines current, 12 threads);
  `check_docs.py` passes.
- Documents: `md/acoustic.md` (the reference), `md/wave.md`, `md/acoustic_round1_report.md`,
  `md/m9_fixtures_report.md`, `md/m5_a21_report.md`, `md/m6_report.md`; briefs `md/m9_fixtures_brief.md`,
  `md/m5_a21_brief.md`, `md/m6_brief.md`, `md/m10_m15_brief.md`, report `md/m10_m15_report.md`; `md/grazing_brief.md`, report `md/grazing_report.md`; `md/precision_brief.md`, report `md/precision_report.md`; `md/cleanup_src_brief.md`, report `md/cleanup_src_report.md` (uncommitted).
- Tools: `tools/acoustic` (`sabine_to_random.py`, `band_probe.cpp` with the peak columns, `paris_cf.py` the closed-form
  Paris value of a locally reacting face, `check_docs.py`, and with M10: `txlib.py` / `txfit.py` / `tl_refs.json` the transmission formula copy
  and fit, `cases.py` moved rows per case; README).
- Literature review: `/home/stephan/accoustic_research/review.md`, data `data/material_properties.csv`, keys in
  `references/index.csv` (four keys the review cites are missing from its index; `check_docs.py` knows them).

## 5. Decided (pointers; the text is in `md/acoustic.md` 10)

P1–P9 (P6 reflect side first; P9 scattering out of scope, a separate project only on very strong evidence); N1–N7
(round 1: the plate rule R1h, M2, `res*` and `coi*` in `X₀`, the two-parameter mapping); A10, A11; M5 (built), M10 (built: `m` a switch), D2-b (`coiA` removed), M11 / M13 dropped, M15 proposal, M14 open (study),
0–90° fit rule, M6 (built:
locally reacting peaks, D1–D5 in `md/acoustic.md` 10; over-damped default, `tfQ < 0` under-damped), M7 dropped, M9 (the route for
unpartitioned built constructions: a wave-medium leaf glued on a box of near-air material on a mass-law wall); A13
(tiered rule for `tf` references), A14 (class 4 on Dupont's 50 mm glass wool), A15 (Miki / Komatsu), A16 (two-tier
tolerances), A17 (classes 8 panel absorber, 9 curtain, 10 MPP; carpet a row of class 4; no corner-trap class), A19
revised (measured `f₀`, else 60 empty / 50 filled; no framed factor); A21 (the kill after a glued exit, every type);

## 6. Pending decisions (proposed in `md/acoustic.md`, not yet decided)

- **A13 refinement**: a conversion whose fitted σ is off a measured σ by more than about 1.5 times is marked, not used.
- **A14**: the two Scrosati rows out until a polyester model and a framed-sample model exist.
- **A17′**: class 8 as the family of resonant absorbers (panel and membrane; perforated and slotted panels; MPP).
- **M8**: retire the one-parameter mapping for materials (documentation only).
- **M15**: the bending loss as a resistance; decide with the parameter session (where `coiQ` comes from).
- **M14**: a limiting angle; after the grazing-incidence study (roadmap item 1).
- **M16**: a relaxation term on `μ` for fabrics (curtains).

## 7. Open items without a decision (`md/acoustic.md` 6)

A8 (low priority), A9 (`html_docu/quadriga_lib_material_model.md`, last), A12 (closed for locally reacting faces by M6; a flat `tf`
keeps the hard wall's phase), A20 (closed for the peak; open for a wall's power-law `tf` behind a porous fill: the pad on
the fill's immittance against a wall impedance fixed as seen from air), A21 remainder
(the three-face tie is decided by mesh order; a full fix needs the intersector or `ray_progress` to report the third
face; how often the kill fires in a real EM scene with glued sheets is unmeasured).

## 8. Roadmap (next first)

Done: **M10–M15** (committed bc24b0c, 2026-10-17; `md/m10_m15_report.md`): M10 built, `coiA` removed (D2-b, with the
unused helper header), M11 / M13 dropped, M15 a proposal, M14 open (item 1), fit rule 0–90°. Left from it: regenerate
`html_docu/` (Stephan); the MATLAB `ray_mesh_interact` wrapper and test to be rewritten (Stephan, later; its test at
line 113 expects the field `|T|²`, the library returns `1 − R`); the two "1 = never" comments on the slab threshold
(`calc_diffraction_gain`'s description, `include/quadriga_tools.hpp` `thin_slab_threshold`) were not in that round's
scope and are still open, comment only.

Done: **grazing incidence** (committed de60194, 2026-10-19; `md/grazing_report.md`): Davy's forced radiation
efficiency of a finite panel in the mass term, the column `side` = √(panel area) (default 3.29 m, 0 the infinite
plate), M14 closed; 13 constructions within 2.70 dB at 0–90°; `tools/acoustic/gzlib.py`. Left: regenerate
`html_docu/` (Stephan).

Done: **traced-leg precision** (committed 6307a64, 2026-10-20; `md/precision_report.md`): the in-medium distance counts
the relaunch offset (exact in double far from the origin), the float bend check in mass-law media, the rule of use
"local coordinates near the origin". Dropped (Stephan, 2026-10-20, veto): the `res*` cavity-spring item (1–2.6 dB,
inside tolerance).

Next: three cleanup tasks (Stephan, 2026-10-20), in this order; each with a plan stop (inventory: what changes,
what is removed and why) and a stop before the commit. The parallel `fixes` checkout (`/sjc/quadriga-lib-fixes`) does
not touch the ray files or the ray tests (checked 2026-10-20). Resources: 12 threads, one suite at a time.

1. **Source cleanup of `src/ray_mesh_interact.cpp`** (brief `md/cleanup_src_brief.md`, approved 2026-10-20; phase 0
   (inventory and plan) done, `md/cleanup_src_report.md`; Stephan decided 0.6 with the worker (option C: pure and
   shared helpers at file scope, per-ray event logic as flat lambdas with explicit captures; every step bit-identical;
   `tools/acoustic/rowdiff.py`; dead-code items 1–6; API statements corrected); manager's review 2026-10-20: plan
   sound, name suggestions for the state-word bits (one name per bit and word) and a few lambdas, taken; phase 1 done
   and reviewed 2026-10-20: 4726 lines (from 5211), 769 comment-only (from 1227), an 84-line header overview; the
   manager rebuilt and reran `[wave]` and `[acoustic]` with 12 threads, `rowdiff` against the baselines: every number
   identical; `tools/extract_md.py -a cpp` extracts. Decided (Stephan, 2026-10-20): A (one normalize form) dropped,
   B `half_pi` the exact double nearest π/2 (moves 1326 printed rounding-noise values below 1e-9, every assertion
   passes), C `k0()` in `davy_sigma`. Phase 2 with a new worker (the first one's context was 85 % full); phase 2 done and reviewed 2026-10-20 (B and C
   applied, exactly the measured values moved, baselines regenerated, CPU 0.983–1.000, 4725 lines, 768 comment-only;
   the manager reran `[acoustic]` against the new baselines: identical); commit message given, waiting for Stephan's
   commit). Remove stale comments; comments down to the necessary minimum
   without losing readability; an agent that reads only this file must understand what it does (a header overview);
   readable structure, correct scoping (anonymous namespace, `const`, the narrowest scope), sensible names. **One
   file**, with the large functions (`ray_state_update` about 2000 lines of lambdas, `ray_mesh_interact` about 1000)
   split into named helper functions grouped in sections (material model, interface physics, slab / stack solver,
   `ray_mesh_interact`, `ray_state_update`). **The public API blocks (`/*!MD … */`) are shortened to the essentials**:
   they compile into the public docs, so they describe what a user or an AI sees from the whole-library perspective;
   internal workings only where needed for context. **Tests must not move**: every assertion passes and the stored
   twins hold their 1e-12 tolerance; printed rows may drift in their last digits only (Stephan chose this over
   bit-identity) — the worker reports every printed value that changes. No change to `ray_progress.cpp` or the
   tests in this task.
2. **Test cleanup of the ray tests** (`[wave]` and `[acoustic]` files and `ray_test_helpers.hpp`, about 33,800 lines;
   other test areas and the Python / MATLAB suites out of scope). Remove **redundant** tests only: print-only cases
   without assertions, duplicates, tests of retired behavior, cases whose every assertion another case already makes;
   every mechanism and every EM statement keeps at least one guard; each removal listed with its reason at the plan
   stop. Remove stale comments; comments to the necessary minimum. **As fast as sensible** (no number; the two big
   cases, *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer* 9.8 s and
   *Measurement - the mirror behind the ray after a back reflection* 9.6 s of the whole suite's 38 s, first).
   **Diagnostic output off by default, switchable** (one environment variable, no rebuild), so the printed-row
   comparison stays available for future analysis.
3. **Material model document**: `html_docu/quadriga_lib_material_model.md` rewritten to match what is implemented,
   **two sections in one page**: a user part (the physics, every column, the rules of use, the limits) and a short
   developer part (how the code is structured, after task 1); **published** with the other html_docu pages. It
   **absorbs what users need from `md/acoustic.md` and `md/wave.md`** (rules of use, limits). The transitory `md/`
   files stay until the parameter project (Stephan's choice); the column reference `format_materials.md` is not in
   this task.

4. **The parameter session** — to become a project of its own (Stephan, 2026-10-20); kept here only so its
   context is not lost; not to be started from this file: the material table from the review's data with the A13 / A14 / A19 rules
   (`tools/acoustic/sabine_to_random.py`), the acceptance classes of `md/acoustic.md` 9.
(A9, the material-model page, is task 3 above.)

Decisions in section 6 can be taken whenever Stephan wants; none blocks item 1.

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
