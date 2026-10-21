# Brief — cleanup of the ray tests (task 2 of 3)

This brief is the whole task description. Read it in full before doing anything, and re-read it at the start of
every phase. Findings go into `md/cleanup_tests_report.md` (create it), one section per phase.

## Why

The ray tests grew with months of physics rounds: 11 test files tagged `[wave]` or `[acoustic]` plus
`tests/catch2_tests/ray_test_helpers.hpp`, about 33,900 lines, 4,657 of them comment-only; 241 cases (181 `[wave]`,
60 `[acoustic]`) that sum to about 29 s of the whole suite (12 threads). They run on every future task. Stephan wants
them lean and fast without losing what they guard. Task 1 (the source cleanup of `src/ray_mesh_interact.cpp`,
committed 29e1f05) is done; task 3 (the material-model document) follows.

**Goals (Stephan, 2026-10-20):**

- **Remove redundant tests only**: print-only cases without assertions, duplicates, tests of retired behavior, and
  cases whose every assertion another case already makes. **Every mechanism and every EM statement keeps at least one
  guard.** Each removal is listed with its reason at the plan stop, before it happens.
- **Remove every stale comment**; reduce comments to the necessary minimum without losing readability. A test's
  leading comment says what it asserts and why that is the expected physics, in plain words. Comments do not cite
  internal labels (A21, M10, N4, D2-b …), `md/` documents, reports, rounds or commit hashes, and do not tell how the
  code used to behave: those documents are transitory and will be deleted.
- **As fast as sensible** (no number): the large cases first (by `--durations`, 12 threads, today:
  *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer* about 9.8 s, *Measurement -
  the mirror behind the ray after a back reflection* about 9.6 s, *Measurement - ray_progress and the mirror behind a
  back reflection* 1.8 s, *Bare tf - …* 1.3 s, *Resonant medium - …* 0.8 s): fewer sweep points where a sweep states
  the same thing at every point, one OpenMP thread where a case runs one ray (`ub::OneThread`), shared setup.
- **Diagnostic output off by default, switchable**: the `QD_DIAG` tables (today on by `#define QD_TEST_VERBOSE` in
  `ray_test_helpers.hpp`, about 23,800 printed lines in a whole-suite run) print only when an environment variable is
  set (for example `QD_TEST_DIAG=1`), with no rebuild, so the printed-row comparison (`tools/acoustic/rowdiff.py`)
  stays available for future analysis.
- Test comments that name internals renamed in task 1 are updated from the rename table
  (`md/cleanup_src_report.md` 1.6; about 30 places).

## Facts at the start

- Last commit 71b894b (the AVX2 test fix). `git status` should be clean apart from this brief and
  `md/acoustic_manager.md`; if anything under `src/` or `tests/` is modified, stop and ask Stephan.
- Baselines (`md/wave.md` 0, 12-thread commands, regenerated after the `fixes` merge): `[wave]` 165594 assertions in
  181 cases, `[acoustic]` 42741 in 60, whole suite 1345968 in 1238 (generic build). An AVX2 build
  (`build_linux/scratch/avx2`, `-DENABLE_AVX2=ON`) passes 1346299 in 1238. `tests/data/wave_twins.tsv` holds the
  stored every-digit values, asserted at 1e-12 relative; `QD_TWIN_WRITE` rewrites it (`md/wave.md` 0).
- The files in scope: `test_wave_mesh_interact.cpp`, `test_wave_progress.cpp`, `test_wave_sheet_vs_solid.cpp`,
  `test_wave_state_update_{spec,sheet,chain,branch}.cpp`, `test_acoustic_mesh_interact.cpp`,
  `test_acoustic_progress.cpp`, `test_acoustic_state_update.cpp`, `test_acoustic_leaf.cpp`, and
  `ray_test_helpers.hpp`. Every other test file (other library areas, the Python and MATLAB suites) is out of scope.
- `md/wave.md` and `md/acoustic.md` cite many case names (their appendices; `tools/acoustic/check_docs.py` checks that
  every `[acoustic]` case is named in `md/acoustic.md`). They stay until the parameter project; a removed or renamed
  case is removed or renamed there too, nothing more.
- Instruments: `tools/acoustic/rowdiff.py` (two runs number by number), `tools/acoustic/cases.py` (moved rows and
  failed assertions per case between two `--durations` runs). Task 1 built a coverage build of
  `src/ray_mesh_interact.cpp` (`-O0 --coverage`, its object linked into a scratch copy of the test binary, `gcov -b`;
  `md/cleanup_src_report.md` 0.3): the same instrument measures what the tests reach.

## Hard rules

- **Tests only.** No change under `src/`, `include/`, `api_*`. The library must not need a change to make a test
  pass; if a test fails for a reason in the library, stop and report.
- **Nothing that is guarded today loses its guard.** Measured, not argued:
  - line and branch coverage of `src/ray_mesh_interact.cpp` and `src/ray_progress.cpp` by the ray tests, before and
    after: no line or branch that a ray test reaches today may become unreached, unless the plan names it and Stephan
    agrees;
  - every kept case's assertions keep their bounds, unless the plan names a change and its reason (a shorter sweep
    whose measured maximum is lower may keep the old bound);
  - the stored twins: rows of kept tables unchanged; rows of a removed table go only with that table, by
    `QD_TWIN_WRITE`, listed in the plan.
- **Generic and AVX2 builds both pass** after every step (the AVX2 build in scratch).
- **Comments**: no internal labels, no `md/` references, no history; names from the task-1 rename table.
- **Resources** (another session shares the machine): at most 12 threads, compiler and tests alike
  (`cmake --build … --parallel 12`, `OMP_NUM_THREADS=12` or fewer); never two suites at once; the coverage build is
  justified here, other heavy tools (ASAN, AFL) are not; timings as CPU time or alternated runs with the load stated,
  without waiting for an idle machine.
- No new open items (Stephan's rule): a side finding is one line in the report, nothing more.
- Do not commit; Stephan commits. The plan stop and the final stop are the two stops.

## Phase 0 — inventory and plan (no test edit; stop)

1. Baseline: build, the three suites with the 12-thread commands, the counts above; the AVX2 build's whole suite.
2. The coverage baseline: line and branch coverage of `src/ray_mesh_interact.cpp` and `src/ray_progress.cpp` by the
   `[wave]` and `[acoustic]` cases alone, and per case (which lines and branches a case reaches that no other case
   does). This is the instrument for "redundant".
3. The inventory, per file and per case: what it asserts (one line), its assertions and run time (`--durations`), the
   lines and branches only it reaches, its diagnostic output, its comments by kind (stale, needed, redundant), internal
   labels and `md/` references, names to update from the rename table.
4. The removal list: each candidate case with its kind (print-only, duplicate, retired behavior, covered by another),
   the case that keeps its guard, and the coverage it would lose (expected none). Cases that are slow but unique go
   to the speed list instead.
5. The speed list: each slow case, what makes it slow, the change (fewer points, one thread, shared setup, a smaller
   ray set), its expected time, and what its assertions keep.
6. The diagnostic switch: how `QD_DIAG` and the other print helpers become runtime-switchable (one place, read once),
   and how the baselines are generated (with the switch on, so `rowdiff` keeps the printed rows; the `md/wave.md` 0
   commands updated in the final phase).
7. The plan: the order of steps (for example: the diagnostic switch; comment cleanup per file; removals; speed-ups),
   the expected counts (cases, assertions, lines, comment lines, run time), and the risks.
8. Stop for Stephan's OK on the plan, the removal list and the speed list.

## Phase 1 — the cleanup (after the OK)

- The steps of the plan, file by file. After each step: build (generic and AVX2), the three suites (generic, with the
  diagnostic switch on), the AVX2 whole suite; `rowdiff` against the baselines for the kept cases' rows; the counts;
  the coverage check after the removals and the speed-ups.
- Update `md/acoustic.md` and `md/wave.md` where they name a removed or renamed case (names only), and the
  `md/wave.md` 0 commands for the diagnostic switch; `tools/acoustic/check_docs.py` passes.
- Regenerate the three baselines (with the switch on) at the end.

## Phase 2 — final checks (stop before the commit)

- Both builds pass; the coverage of both source files by the ray tests is unchanged against phase 0 (or every loss
  named and agreed); the twins as planned.
- Run time of the ray cases, CPU time, before and after (alternated, load stated); the whole suite's time.
- The final counts: cases, assertions, lines and comment lines per file, printed lines with the switch on and off.
- Report section 2 and stop. Stephan commits; the manager writes the commit message.

## Traps

- The traps of `md/wave.md` 11 and `md/acoustic_manager.md` 9: run binaries from the repository root, one suite at a
  time, `pkill -f` kills its own shell, `python3 -I`, a probe that includes a test file runs every case of it unless
  given a test spec, a case name with a comma needs `\,`.
- Some cases assert nothing directly but feed the twins or a shared table that another case asserts: check before
  calling a case print-only.
- Some sweeps are long because the statement is "at every angle" or "no step at the critical angle": a shorter sweep
  must still contain the points that carry the statement (the critical angle, grazing, a resonance).
- `ub::OneThread` makes a single-ray case faster but does not change results; a case that runs many rays may need the
  threads.
- The diagnostic switch must not change any assertion: a value computed only for printing must not feed a check.
- Removing a case changes the printed rows of no other case; renaming a case changes `check_docs.py`'s index and the
  `md/` appendices.
- Scratch is not kept: the report states every number it relies on.
