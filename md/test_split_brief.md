# Brief — split the ray tests into wave and acoustic, and clean them up

This brief is the whole task description. Read it in full before doing anything, and re-read it at
the start of every phase. Findings go into `md/test_split_report.md` (create it), one section per
phase, so the next session can pick them up.

## Why

The thin-sheet / traced-solid work (plan `md/sheet_solid_fix_plan_v46.md`, 45 rounds) has two kinds
of physics in one engine, and the tests mix them:

- **Wave physics** — every material *without* the mass column (`m = 0`): Fresnel faces, the
  solver's recursion, `tf` as transformer and pad, the slab series, the recovered cosine, the
  relaunch geometry, a resonance as part of the medium. EM types (0, 1, 2) and scalar types (3, 4,
  5) alike; porous absorbers are wave media. It is settled: no open item of the plan reaches it.
- **Lumped plate physics, the acoustic add-on** — every material *with* the mass column (`m > 0`):
  the closed mass-law layer, the held wave, the pair rule at mass-law faces, the plate rule and its
  thickness code, the mass cosine. This part is going to be redesigned in later sessions.

The tests are to be split along that line, so that the wave tests guard against regressions while
the acoustic part is rebuilt, and both become readable and faster to compile.

**The boundary is the mass column, decided per test case by measurement (see phase 0), never by
interaction type and never by reading.** A case in which no material with `m > 0` is ever evaluated
is a wave case. A case in which one is evaluated is an acoustic case. A case with both is mixed and is
split by section where that is clean, otherwise it goes to acoustics. A rigid permittivity
(`ε ≪ 1`) *without* the mass column is a wave medium for this purpose.

Names: **wave** and **acoustic** (files, baselines, documents).

## Facts at the start (measured 2026-10-14)

- Suite: `All tests passed (1297934 assertions in 1074 test cases)`, about 3 min 13 s wall on an idle
  machine. `test.txt` in the repository root is the baseline, 19,870 lines, nearly all of it
  diagnostic tables. It is byte-identical to the committed one.
- The three large files: `tests/catch2_tests/test_ray_state_update.cpp` 20,350 lines, 102 cases;
  `test_ray_progress.cpp` 5,487 lines, 54 cases; `test_ray_mesh_interact.cpp` 3,354 lines, 33 cases.
- Compile time of each translation unit alone with the project flags: 55 s and 1.1 GB, 16 s, 9 s.
- Other test files that touch the same functions or the material columns (the inventory has to
  check each): `test_sheet_vs_solid.cpp` (1,157 lines, has `QD_TEST_VERBOSE` tables),
  `test_acoustic_interactions.cpp` (608, old, tests the same statements as newer cases),
  `test_gains.cpp`, `test_calc_diffraction_gain.cpp`, `test_ray_commit.cpp`,
  `test_ray_subdivide_flag.cpp`, `test_get_channels_multifreq.cpp`, `test_ray_init.cpp`,
  `test_obj_file_write.cpp`, `test_baseband_freq_response_multi.cpp`.
- CMake globs `tests/catch2_tests/*.cpp` (`CONFIGURE_DEPENDS`), so new test files need no edit of
  `CMakeLists.txt`.
- `QD_TEST_VERBOSE` is defined in `test_ray_state_update.cpp`, `test_ray_mesh_interact.cpp`,
  `test_ray_progress.cpp` and `test_sheet_vs_solid.cpp`.
- The plan's section 1, *Diagnostic infrastructure*, is the map of the helpers and of every case's
  instruments (the reference builders `diag_*`, `qd_*_ref`, `DiagLayer`, `LumpMtl`, the namespace
  `ub`, the walkers `bf::walk` and `cf::walk`, the chain runners copied into several cases, the
  one-thread guard). Read it there; do not edit the plan.

## Hard rules

- No change to `src/` or to anything outside `tests/` and `md/` — the library is not touched in
  this task. No change to the physics a test asserts.
- Do not commit; the user commits manually.
- Measure, don't guess. Every statement about what a case covers, costs or duplicates comes from a
  run or a search. If a command does not run, say so and stop.
- Phases end with a report and a stop. Phase 1 and phase 2 start only after the user's OK on the
  previous report.
- Before anything else: `cmake --build build_linux --parallel`, then `./build_linux/test_bin >
  test.txt` from the repository root, and check that the last line reports 1297934 assertions in
  1074 test cases and that `git status` shows `test.txt` unchanged.

## Phase 0 — inventory and analysis (no edits to tests)

Deliver in `md/test_split_report.md`, section *Phase 0*:

1. **Inventory.** Every TEST_CASE in the files above: file, line range, lines, assertions, run time
   (`--durations yes`), output lines in `test.txt`, and which library functions it calls.
2. **Classification by measurement.** A marker build: copy `src/ray_mesh_interact.cpp` into a fresh
   scratch folder `build_linux/scratch/ts1`, add a counter where a `Material` with `m > 0` is
   constructed from the columns (that covers every evaluation in that file, the public
   `medium_gain` and `interface_gain` included), compile it with the flags of
   `build_linux/CMakeFiles/quadriga_static.dir/flags.make` and relink `test_bin` from
   `build_linux/CMakeFiles/test_bin.dir/link.txt` with the patched object in front of
   `libquadriga.a`. A Catch2 listener (`CATCH_REGISTER_LISTENER`, `testCaseEnded`) calls a dump
   function of the marked copy, which prints and clears the counter per case. Templates:
   `build_linux/scratch/i47/mk47.py` (marker `QD_MARK`, `qdmk_dump`), `listener.cpp`, `build.sh`.
   First check that the marked copy with the marker off reproduces `test.txt`. Result: every case
   wave / acoustic / mixed, with the count. Also mark cases that use scalar types only on wave media
   (porous absorbers) — they are wave cases.
3. **Overlap.** Which cases assert the same statement twice: `test_acoustic_interactions.cpp`
   against the newer cases, `test_sheet_vs_solid.cpp` against the sheet-against-solid tables of
   the other files, and repeated copies of one table. For each pair, which one to keep and why.
4. **Diagnostics, table by table** (every `# diag` table and every unnamed table in `test.txt`):
   carries an assertion (which), is a "twins with every digit" regression guard, is recorded only,
   or describes a superseded rule or an earlier state of the code (headers that still describe a
   rule that is gone, `qd_mass_TL_clipped`, ...). Its output lines. Whether the plan cites it.
5. **Helpers.** Which helpers are shared by several cases and belong in one test header; which are
   dead (no caller — `diag_tf_port_rare` is one, per the plan); which copies of a chain runner or
   walker can collapse into one; which quadriga_lib functions the fixtures could call instead of
   their own code (mesh building, `obj_file_read`, the public `medium_gain` / `interface_gain`,
   path utilities). Read the public headers in `include/` for that.
6. **Proposal.** The file layout (for example `test_wave_*.cpp` and `test_acoustic_*.cpp` in a few
   files of similar compile time, plus one shared header such as
   `tests/catch2_tests/ray_test_helpers.hpp`), the header's contents, the list of what to delete or
   merge with what each item guarded, the diagnostics to keep, and target sizes and compile times.

Proposed policy for the diagnostics, to be confirmed or changed by the user after phase 0:

- **Wave baseline:** keep only tables that carry assertions or are regression twins; turn as many
  twins as possible into assertions against stored values, so a wave regression fails an assertion
  instead of hiding in a diff. Target: a few thousand lines.
- **Acoustic baseline:** may stay looser for now; most of its rows will move in the redesign.
- **Recorded-only and superseded tables:** delete, once phase 0 shows nothing references them.
- `QD_TEST_VERBOSE` stays a switch; the baselines are generated with the kept tables on.

Stop and report.

## Phase 1 — pure move (after the user's OK)

The shared header and the split files, with no logic change: cases, sections, assertions, bounds
and printed tables move verbatim. Old files that end up empty are removed.

Acceptance:
- the same 1074 cases and 1297934 assertions, all passing;
- the diagnostic output identical row by row, compared **keyed by case and table** (the order
  changes with the files); rows that move only in their residual columns at about 1e-16 are listed
  apart — changing a translation unit is known to shift 14 to 27 such rows by up to 3e-16 (plan,
  *Rows that move without a library change*);
- compile time of each new translation unit, and of the whole test build.

Stop and report.

## Phase 2 — cleanup (after the user's OK on the phase-0 list)

Only what the user approved: dead helpers removed, duplicate cases merged, copies of runners
collapsed onto the header, recorded-only and superseded tables deleted, stale headers and comments
corrected, twins turned into assertions. Every dropped assertion is listed with the statement it
guarded and the case that still asserts that statement; a statement that would lose its last guard
is not dropped.

Stop and report.

## Phase 3 — baselines

Generate `test_wave.txt` and `test_acoustic.txt` from the repository root (the binary's test-spec
filters or tags select each half; Catch2 tags `[wave]` and `[acoustic]` on every case are the
simplest way), with their last lines. Keep `test.txt` as the whole-suite baseline only if the user
wants it. Write in the report the new "first step" instruction for future sessions: the build, the
two baseline commands, and the expected counts of each half.

## Rig and traps

- A relinked binary has to be run from the repository root: some cases open files under
  `tests/data` by a relative path.
- Never run two suites at once; check with `ps` that no other job loads the machine; wrap every
  probe in `timeout`.
- Fixtures that call the library one ray at a time need the one-thread guard (`ub::OneThread`, or
  a scoped `omp_set_num_threads(1)`); without it such a case takes twenty times as long.
- A case name must not contain a comma (the tables are CSV and the name is used as a filter).
- `arma::accu(arma::square(...))` links against a BLAS wrapper the test binary does not have; sum by
  hand. `arma::abs` of an xprmat is per real and imaginary row, not per complex entry.
- Catch2 prints a failed comparison of long doubles on three lines; count failures by case.
- `build_linux/scratch/i47/build.sh` writes its output relative to `build_linux` unless the output
  path is absolute.
- `pkill -f <pattern>` kills the calling shell when the pattern is in its own command line; kill by
  pid (`pgrep -x`).
- Keep scratch files in `build_linux/scratch/ts1`; the scratch folders of earlier items (`i38` to
  `i47`) are read-only references.

## What happens afterwards (not part of this task)

With the split in place, two documents replace the plan: `wave.md` (the settled wave conventions,
each tied to the case that asserts it, and the accepted limitations) and `acoustic.md` (the
mass-column machine: what exists, what is entangled, the open changes). The classification and the
per-case list of what is asserted, from phase 0, are their input.
