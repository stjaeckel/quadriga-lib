# Report — cleanup of the ray tests (task 2 of 3)

Brief: `md/cleanup_tests_brief.md`. One section per phase. Scratch is `build_linux/scratch/ct0` and `ct1` (not kept; every
number this report relies on is stated here).

## Phase 0 — inventory and plan (2026-10-20, no test edit)

### 0.1 Baseline

- `git status`: clean apart from `md/acoustic_manager.md` (modified) and `md/cleanup_tests_brief.md` (new); nothing
  under `src/` or `tests/` modified. Last commit 71b894b. The eleven test files and `ray_test_helpers.hpp`: 33,887
  lines.
- `cmake --build build_linux --parallel 12`: up to date. The suites with the 12-thread commands of `md/wave.md` 0, run
  from the repository root, one at a time (a script waits until no other `test_bin` runs):

  | suite | result | wall / user CPU | load at start / end |
  | --- | --- | --- | --- |
  | `[wave]` | All tests passed (165594 assertions in 181 test cases) | 24.1 s / 27.6 s | 6.3 / 5.3 |
  | `[acoustic]` | All tests passed (42741 assertions in 60 test cases) | 0.9 s / 1.3 s | 5.3 / 5.3 |
  | whole suite | All tests passed (1345968 assertions in 1238 test cases) | 38.1 s / 74.1 s | 5.3 / 4.3 |
  | whole suite, AVX2 (`build_linux/scratch/avx2`, up to date) | All tests passed (1346299 assertions in 1238 test cases) | 38.7 s / 74.1 s | 4.0 / 3.1 |

  The counts are the brief's. Printed lines (without Catch2's `--durations` lines): `[wave]` 13,972, `[acoustic]`
  9,017, whole suite 23,834; the 241 ray cases print 22,979 of them (139 cases print anything).
- The five largest cases (`--durations`, 12 threads): *ray_progress - the reflect port of a sheet lies where its
  transmit port puts the layer* 9.58 s, *Measurement - the mirror behind the ray after a back reflection* 9.27 s,
  *Measurement - ray_progress and the mirror behind a back reflection* 1.74 s, *Bare tf - …* 1.23 s, *Resonant medium -
  …* 0.75 s: 22.6 of the 24.8 s the 241 ray cases take.

### 0.2 Instruments

All in scratch; the methods, so that phase 1 can rebuild them:

- **Marked binary.** A Catch2 listener (`CATCH_REGISTER_LISTENER`) linked into a copy of `test_bin` prints a line before
  and after every case: name, `file:line`, passed and failed assertions, wall time, process CPU time (`std::clock`).
  The printed output of a case is the lines between the two marks. The `[wave],[acoustic]` run: 208335 assertions in
  241 cases, 24.9 s, the total of the two plain-binary suites.
- **Coverage build.** `src/ray_mesh_interact.cpp` and `src/ray_progress.cpp` compiled `-O0 --coverage` (`g++-12`, the
  library's flags otherwise), replaced in a copy of `libquadriga.a`, linked with `CMakeFiles/test_bin.dir/link.txt` and
  a listener that, at the end of every case, sets `GCOV_PREFIX` to a per-case folder, calls `__gcov_dump()` and
  `__gcov_reset()`. One run of `[wave],[acoustic]` (all passed, 96.8 s) gives one `.gcda` pair per case;
  `gcov-12 -j -t -b` per case. Checked: *Lossy verdict - a lossy second medium has no critical angle* from the full run
  against the same case run alone, identical lines (1236) and branch keys (1508).
  Definitions: a **line** is reached when its count is above zero; a **branch** is a key (line, function, index of the
  branch on that line), throw edges excluded, and it is reached when taken at least once. Template instances and the
  OpenMP outlined bodies are separate functions in gcov's JSON, so a branch of the `float` and of the `double`
  instance are two keys.
- **Section timers.** A scratch copy of a test file with `std::clock()` stamps at its section boundaries (and per row
  where needed), compiled with the test flags and linked in place of that file's object; the cases run with
  `ub::OneThread`, so CPU time is their run time.
- **What-if build.** The same, with a planned change applied to scratch copies, as a plain binary (times) and as a
  coverage binary (reach).
- `tools/acoustic/rowdiff.py` and `tools/acoustic/cases.py` as tracked.

### 0.3 The coverage baseline

What the 241 ray cases reach:

| file | executable lines | reached | branch keys | taken |
| --- | --- | --- | --- | --- |
| `src/ray_mesh_interact.cpp` | 1783 | 1679 | 4536 | 3685 |
| `src/ray_progress.cpp` | 474 | 461 | 579 | 524 |

(gcov's own per-file summary of a single case reads "Lines executed … of 1783" for `ray_mesh_interact.cpp`, the same
line total; task 1's 2458 was the file before its cleanup.)

Per case:
- **40 cases reach something no other ray case reaches** (93 lines and 405 branch keys in all). They are the
  input-validation cases (*ray_progress - input validation* 26 lines / 56 branches, *ray_state_update - input
  validation* 21 / 75, the delegated-intersection and delegated-flag validation cases), *Ray-Mesh Interact - Air to Air*
  (14 / 17), *Unentered branches - a corner with a sheet is the sheet alone …* (9 / 23), *ray_state_update - type parity
  between double and float* (0 / 27, the `float` instance), and 32 cases with one to four lines or a few branches each
  (the list is column "unique L / B" of appendix A). Every one of them is kept.
- **48 cases reach nothing that a single other case does not also reach**, mostly short unit cases of the classic
  `Ray-Mesh Interact` / `ray_progress` / `ray_state_update` sets whose assertions differ from their superset's; that
  alone is not redundancy.
- Two cases reach neither file: *ray_state_update - state and code encoding* (the bit helpers) and *Layered stack - the
  characteristic-matrix reference against the single-layer oracles* (the test's own references against each other).
  Both are kept.

### 0.4 Inventory

Per file (comment lines: full-line comments including the three licence lines, and trailing comments; "flagged
stale": lines that contain an internal label, a document or commit reference, a history phrase or a task-1 old name,
found by pattern; banner lines: `// ------` and `// ======` rules):

| file | lines | cases | assertions | wall s (12 thr) | CPU s | printed lines | comment lines full / trailing | flagged stale | banner lines | `QD_DIAG` lines | `#ifdef QD_TEST_VERBOSE` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `test_wave_mesh_interact.cpp` | 2879 | 38 | 3415 | 0.01 | 0.14 | 375 | 368 / 122 | 29 | 4 | 40 | 0 |
| `test_wave_progress.cpp` | 4462 | 52 | 69837 | 11.93 | 15.82 | 2514 | 359 / 169 | 7 | 50 | 86 | 26 |
| `test_wave_sheet_vs_solid.cpp` | 454 | 2 | 3 | 0.00 | 0.00 | 0 | 82 / 7 | 8 | 0 | 0 | 1 |
| `test_wave_state_update_spec.cpp` | 1487 | 28 | 737 | 0.00 | 0.02 | 0 | 73 / 114 | 5 | 0 | 0 | 0 |
| `test_wave_state_update_sheet.cpp` | 5206 | 32 | 7346 | 0.03 | 0.33 | 1678 | 1109 / 204 | 66 | 64 | 304 | 0 |
| `test_wave_state_update_chain.cpp` | 4263 | 12 | 15484 | 9.49 | 10.52 | 3065 | 510 / 131 | 6 | 98 | 338 | 0 |
| `test_wave_state_update_branch.cpp` | 3334 | 17 | 68772 | 2.49 | 2.54 | 6335 | 391 / 208 | 4 | 56 | 326 | 0 |
| `test_acoustic_mesh_interact.cpp` | 331 | 2 | 174 | 0.00 | 0.01 | 20 | 61 / 20 | 3 | 0 | 5 | 0 |
| `test_acoustic_progress.cpp` | 842 | 5 | 3190 | 0.14 | 0.25 | 1009 | 64 / 21 | 12 | 2 | 14 | 4 |
| `test_acoustic_state_update.cpp` | 6649 | 47 | 34911 | 0.49 | 0.89 | 6970 | 890 / 399 | 89 | 118 | 483 | 0 |
| `test_acoustic_leaf.cpp` | 690 | 6 | 4466 | 0.26 | 0.26 | 1013 | 86 / 14 | 10 | 0 | 26 | 0 |
| `ray_test_helpers.hpp` | 3290 | – | – | – | – | – | 696 / 155 | 26 | 22 | 2 | 1 |
| total | 33887 | 241 | 208335 | 24.83 | 30.78 | 22979 | 4689 / 1564 | 265 | 414 | 1624 | 32 |

(My full-line count is 4,689 against the brief's 4,657: block-comment lines and the licence lines are counted here.)
The CPU column is the process CPU time per case, all OpenMP threads: the 10 single-ray cases without `ub::OneThread`
take 5.3 CPU-s for 0.46 s of wall time (0.4).

Per case: appendix A (assertions, wall and CPU time, printed lines, comment lines, the leading comment's length,
flagged lines by kind, unique coverage, action). What each case asserts is its name, which states it; the
leading comment says the rest, and phase 1 shortens it to that.

**Comments by kind.** Pattern flags find the certain cases (265 lines); reading finds more, which phase 1 takes file by
file:

- *Stale, internal labels and documents* (removed or replaced by the physics they stand for): `test_acoustic_state_update.cpp`
  N1 ×14, M6 ×7, M10 ×6, N2 ×5, A7 ×4, R1h, N3, N4, N7, A12 ×3 each, A2, A3, A10, D2-b ×2, B4s, M4a, M5; `md/acoustic.md`
  ×24, `md/precision_report.md` ×3, `md/grazing_report.md` ×2, "report 1.11", "report 3.10", "Round 1", "decision
  2026-10-19" ×3, the commit `bb336dc` ×12 ("At bb336dc the plate passed …"). `test_acoustic_progress.cpp`: M6 ×6, N1,
  M5 ×2, M10, A10, "issue 15", `md/acoustic.md` ×3, `md/m6_report.md`, "report 3.8", bb336dc. `test_acoustic_leaf.cpp`:
  M9, A18, A21 ×2, "item 3/4/5", `md/m9_fixtures_report.md` ×2, `md/acoustic.md` ×4, `md/wave.md`.
  `test_acoustic_mesh_interact.cpp`: M10 ×2, N1, `md/acoustic.md`, "Test 3". `test_wave_mesh_interact.cpp`: D2-b ×3,
  "Test 4 / 6 / 8 / 9 / 10". `test_wave_state_update_spec.cpp`: D2-b. `test_wave_state_update_sheet.cpp`: "issue 1 / 7 /
  9", "Symptom three". `test_wave_progress.cpp` and `test_wave_state_update_chain.cpp`: "Plan item 1" ×4,
  `md/precision_report.md`. `ray_test_helpers.hpp`: M10 ×2, N1, "issue 3", `md/acoustic.md` ×2,
  `md/precision_report.md`, "decision 2026-10-19". (The chain file's "A1." / "A2." are the case's own table rows, not
  labels.)
- *Stale, history* (how the code behaved or how the test came about): "post-VBS refactor", "Model of the refactored
  function", "the former test_ray_*.cpp" (helpers); "now engine-computed outputs", "xprmat now [8, n_rayN]", "after the
  refactor" ×3, "now the lossless Fresnel term only; the α loss moved to medium_gain()", "pre-mu behavior", "Tripwire:
  if the Fresnel term reverts", the convention-change narrative of *Diagnostic - Transmission convention baseline* (30
  lines), "had never seen a magnetic material", "the plan sets for it", "What the clamp was said to be guarding against"
  (mesh-interact file); "In the first run that corrupted three rows", "pre-refactor result", "Higher lods were used in
  the first run", "is currently an open question" (sheet vs solid); "the acceptance test for removing the Tsp * corr
  pair … ratio_pre is what the code produced BEFORE that change", "Two corrections over the first run", "material 4 was
  tried first", "the case that would have caught the exit-angle defect", "A measurement, taken before any decision"
  (sheet file, about 25 such passages); the *accumulated path length* NOTE "fails until ray_progress writes the length"
  (the case passes); "5 (scalar refraction) is now valid", "Engagement tripwires for the new material features", a
  commented-out `WARN(...)` line (spec file); "Before the change the summed series created energy (R 42 …)" (leaf);
  "Before M6 …", "At bb336dc …" (acoustic files).
- *Stale, wrong*: *ray_state_update - lossy medium feeds |phi| from medium_gain* says the mass column cannot be used and
  calls its check "the mass column took effect", but the material carries `alpha`; `QdMassRef` describes the
  composite before the plate rule ("the transmit port is the product of the two raw Fresnel transmission amplitudes
  times the square root of the mass gain") and two of its three fields are unused; *Acoustic - interface_gain …* says
  the gain is "clamped at >= 0 dB" (task 1 replaced the clamp by validation); *Measurement - …* "Nothing is asserted
  beyond sanity" (they assert the wall's matrix and the inherited verdict); *ray_state_update - resolve clamp near the
  Airy pole* says "the resolve is refused" where its trailing comments say "floored, not refused" (to be read against
  the code in phase 1).
- *Task-1 old names*: 62 comment lines and one case name (list in 0.4.1).
- *Redundant*: the 414 banner lines; 70 cases whose leading comment runs past 10 lines (631 lines beyond the tenth,
  longest 34); passages that describe a printed column ("Recorded: …", "Rows that must not move are printed with every
  digit"); trailing comments that restate the line.
- *Needed*: the rest — what a case asserts and why that is the physics; the palette legends (trailing `// 1 C1 …`
  rows); the conventions block of the helpers, once its stale parts are gone.

#### 0.4.1 Names from the task-1 rename table

Comment lines that name an internal of `src/ray_mesh_interact.cpp` that task 1 renamed (old → new, from
`md/cleanup_src_report.md` 1.6):

| old | new | where (file:line) |
| --- | --- | --- |
| `sheet_case` (×14) | `SheetCase` (`none, isolated, glued_entry, glued_exit, stack, duplicate, corner`) | sheet 233, 234, 503, 504, 512, 515, 516, 534, 535, 690, 691, 1961 (case name), 1965, 1970 … |
| `SCALE_BY_MEDIUM` (×8) | `close_leg` (`leg_exact_factor()`) | sheet_vs_solid 312, helpers 1057, sheet 353, 366, 369, … |
| `SHEET_EVENT` (×7), `SHEET_RESOLVE` (×4) | `sheet_event`, `sheet_at_tracked_media` | helpers 786, sheet 1965, 1978, 2007, … |
| `SLAB_AIRY_FACTOR` (×6) | `sum_series` | sheet |
| `Material::interact_with` (×13 of the 17 `Material::` lines) | `interact_with(A, B, …)` | mesh-interact 1714, 1800; helpers 53, 947; sheet |
| `thetaF` (×3) | `theta_fbs` | helpers 2530, 2562 |
| `fresnel_r` (×2) | `slab_face_reflection` | helpers 904 |
| `REPLACE_BY_VBS` (×2), `EXACT_IN_SLAB_COSINE` (×2), `TRANSPARENT_TO`, `KILL_RAY`, `APPLY_TRANSITION` | `face_at_vbs`, `in_medium_cosine` / `recover_cosine()`, `same_eps_mu()`, `kill`, `transition` | helpers 786, sheet, acoustic state update 2458, 2915 |

`Material::medium_gain`, `Material::eta` and `qd_relaunch_offset` kept their names in task 1 and stay. The case
*Layered stack - sheet_case 4 resolves rather than killing the ray* is the one case name with an old name (decision
D6).

#### 0.4.2 Diagnostic output

- `QD_DIAG(...)` is `std::printf` while `ray_test_helpers.hpp` defines `QD_TEST_VERBOSE` (line 35), 1,624 lines in the
  ray files. Five uses are inside a conditional expression (`cond ? QD_DIAG(…) : QD_DIAG(…)`), so the macro has to
  stay an expression.
- 31 `#ifdef QD_TEST_VERBOSE` blocks with plain `std::printf` / `std::snprintf`: 26 in `test_wave_progress.cpp` (the
  two *Measurement* cases, the reflect port case, …), 4 in `test_acoustic_progress.cpp`, and
  `test_wave_sheet_vs_solid.cpp`, which does not include the helpers and already reads an environment variable
  `QD_TEST_VERBOSE` at run time (unset in the baselines: it prints nothing today).
- Values computed only for printing: the 30-step bisection `rho_lib` of *Measurement - the mirror behind the ray
  after a back reflection* (inside the `QD_DIAG` arguments: 0.19 s), three of its four path walks per row in section B
  (5.4 s), the reference walks `Z60` of both *Measurement* cases, the gain-floor table of the ray_progress one. Section
  0.6 says which become print-only.
- `twin()` asserts every-digit rows against `tests/data/wave_twins.tsv` (736 rows, all from
  `test_wave_state_update_branch.cpp`: bare tf E / T, closed face H, flat branch H, rare sheet E / F, scalar closed
  face C, sheet tf H, stack geometry C1 / C2) whether or not the rows print. No planned change touches a twin.
- Other test files (`test_sos_*`, `test_conf_*`, `test_scenpar_validate.cpp`) carry their own `QD_TEST_VERBOSE` /
  `QD_DIAG` defines; out of scope, untouched.

#### 0.4.3 Dead code and duplication

- `diag_res` (helpers 1181) is used nowhere; `QdMassRef::base` and `::refl` are never read; *ray_progress - refracting
  tracer: … - the mass-law part* carries the beam/vertex setup, `find`, `angles`, `TOL` and the forced-flag plumbing
  of its wave half, none of which it uses; several classic `Ray-Mesh Interact` cases declare unused locals
  (`rad2deg`, `normal_vecN`, `fbs_angleN`, …). About 100 lines.
- One ray through `ray_progress` until none is left, collected in air behind or in front of a wall, is written out as
  a local lambda 12 times (`test_wave_progress.cpp` ×6, `test_acoustic_progress.cpp` ×4, `test_acoustic_leaf.cpp` ×1,
  with its own float plane-mesh `build` each), 45–70 lines a copy; they differ in the generation limit, the
  interaction limit, the gain floor, and whether they sum powers, complex fields with the in-wall phase, or keep the
  strongest path. A shared `rp::walk` with those as options would save about 450 lines (decision D5).

### 0.5 The removal list

Each candidate, measured with the per-case coverage (lines and branch keys of both files reached by the remaining 237
cases: 6349 of 6349, nothing lost):

| | case | kind | what keeps its guard | coverage it alone reaches | assertions, lines |
| --- | --- | --- | --- | --- | --- |
| R1 | *Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more than it received - the mass-law part* (`test_acoustic_state_update.cpp:4720`) | print-only: 0 assertions; prints a "twins" table of one row set but calls no `twin()`; its `inside` lambda is unused | nothing to keep; the rare mass-law sheet's physics is asserted by *Rare sheet - …* (wave half) for the coherent sheet and by the plate-rule cases for the mass-law sheet | 0 / 0 | 0, 74 |
| R2 | *Diagnostic - Derived lengths and the tangency limit* (`test_wave_mesh_interact.cpp:2480`) | duplicate + test of retired behavior: its first loop asserts the test's own arithmetic (no library call; `d·c = thk`, `thk·cos ≤ thk`, `c_slab > 0.9`), about a removed grazing clamp; its second loop is *Thin sheet - grazing incidence does not depend on the face* (same sheet, same eps 5.24 − 0.63j, same references) at 89.9, 89.99, 89.999 and 89.9999 deg | *Thin sheet - grazing incidence does not depend on the face*, with 89.9999 deg added to its sweep (+10 assertions) | 0 / 0 | 42, 62 |
| R3 | *Thin sheet - mesh subdivision invariance* (`test_wave_mesh_interact.cpp:2424`) | duplicate: the same statement — the triangle a sheet is cut from does not enter the composite, which sits on the analytic slab at grazing — as the grazing case, which compares a 565.7 m face with a 0.057 m one (this case: 565.7 m with 8.7 m) at the same angles and on the same references; identical coverage | *Thin sheet - grazing incidence does not depend on the face* | 0 / 0 | 49, 55 |
| R4 | *Acoustic - interface_gain is the penetration loss alone; a coiA column is ignored* (`test_wave_mesh_interact.cpp:2585`) | covered by others: "a table with coiF, coiQ, coiA gives the bare gain" is asserted by *Coincidence column - a table that carries coiA loads and the column changes nothing* (interface gain bitwise against the bare table, and both tracers on a solid and a sheet); the att power law and the attB doubling by *Ray-Mesh Interact - Public helpers: …* (att 6 dB at 2 GHz, attB 1 → 1e-3 at 10 GHz) and *Ray-Mesh Interact - Penetration loss frequency scaling*; its comment describes the removed clamp | the three cases named | 0 / 0 | 55, 42 |

`md/acoustic.md` names R1 in its appendix table (line 1445); it goes there. R2–R4 are named in no `md/` document
other than the task reports. No twin row belongs to them.

Not on the list, with the reason:
- The 48 coverage-subsumed cases (0.3): each asserts a contract its superset does not (the classic unit cases).
- The `ray_progress - …` cases that mirror a `ray_state_update` case of the same statement: they guard the tracer's
  own mechanisms (float, a band on the base frequency's geometry, the generation loop) and reach `ray_progress.cpp`.
- *Measurement - …* (three cases): their sections assert the inherited verdict, the verdict handed in, the wall's matrix
  at the default threshold, and the band; the sanity-only sections become print-only instead (S2, S3).
- *Bare tf - … - the mass-law part*: asserts the closed face's event exactly (R = 1 − tf, T = tf at 1e-12); the plate
  rule cases assert the wall at 0.003 dB, not the event.
- *Held exit - a non-rigid mass-law body …* (coverage inside *Mass law - …*): asserts tracer 5 = tracer 4 on a
  non-rigid mass-law body, which nothing else does.

### 0.6 The speed list

Measured section times (CPU time of a single-thread case; the scratch section-timer copies, load 1–3), then what-if
builds with S1–S4 applied (one binary, `[wave],[acoustic]`, 12 threads, load 5.6 / 5.3): **the ray cases 24.83 s →
6.08 s wall, 30.78 → 12.0 CPU-s**, every assertion that remains passing, and the coverage union of both files unchanged
(6349 of 6349; the what-if coverage run, all passed).

| | case | what makes it slow (measured) | change | time before → after (what-if) | what its assertions keep |
| --- | --- | --- | --- | --- | --- |
| S1 | *ray_progress - the reflect port of a sheet lies where its transmit port puts the layer* | section B (sheet 6 mm, air gap 30 mm, host 100 mm, cavity followed to 110 interactions and −200 dB): 9.55 of 9.58 s; of it the lossless host at 70 deg, 6 GHz: 7.20 s (618,300 paths) and 3 GHz: 1.64 s | drop those two rows (lossless host, 70 deg, 3 and 6 GHz) | 9.565 → 0.660 s; −10 assertions | B keeps 28 rows: both hosts, 10/40/70 deg, 0.3–6 GHz on the lossy host, 0.3–1 GHz at 70 deg on the lossless one. The dropped rows discriminate the least (u0/ut 14.5 deg from the centred layer at 70 deg against 43 deg at 10 deg, 6 GHz) and held the largest residual (R 3e-4 dB, T −0.012 deg); the bounds stay |
| S2 | *Measurement - the mirror behind the ray after a back reflection* | B (every path, 4 walks per row, 320 rows): 8.9 s, of which only walk L is asserted (`nT ≥ 1`); I, U, Z and the HSH reference Z60 are printed only (5.4 s). A: `rho_lib` bisection inside the `QD_DIAG` arguments (0.19 s). C (near coating, 64 rows): asserts finiteness only | B and C become print-only (run with the switch on, assert nothing); their 320 + 128 sanity assertions go. A and D unchanged | 9.176 → 0.219 s (0.03 s with the switch off, the bisection gone); −448 assertions | A: the verdict is inherited (`x1 == x0`), the followed mark and the unchanged previous-medium word at every back reflection; D: the verdict handed in, both ways. B and C print the same rows with the switch on |
| S3 | *Measurement - ray_progress and the mirror behind a back reflection* | A 1.24 s (the asserted traces; HSH at threshold 0.15 is 0.83 s), B (gain-floor table, asserts `nT ≥ 1`) 0.40 s, D (near coating, asserts `nR ≥ 1`; the same rows are asserted against the matrix in S1's case, section A) 0.002 s, the HSH reference Z60 printed only | B and D print-only (−36 − 32 sanity assertions); Z60 computed only with the switch on | 1.730 → 1.260 s; −68 assertions | A: the default threshold and every series summed are one wall, 10 walls × 2 tracers × 16 angles; C: the band, both frequencies on the matrix |
| S4 | *Bare tf - a bare face with tf is a transformer where it is open and a flat leak where it is closed* | A (one solid with tf, 7 media × 3 tf × 4 types × 3 thicknesses × 6 angles, summed and followed): 1.24 s, the followed walks 1.17 s | followed rows at 0, 45 and 85 deg (the summed rows keep all six angles) | 1.213 → 0.693 s; −1458 assertions | "passes and returns no more than it received, in each channel" on every medium, tf, type and thickness, summed at six angles and followed at three including grazing. The one line only this case reaches (`vbs_intersect`'s near-parallel decline, `ray_mesh_interact.cpp` 2113) is still reached (what-if coverage) |
| S5 | ten single-ray cases without `ub::OneThread`: *Measurement - ray_progress follows the coupling …*, *ray_progress - a declined series at a sheet …*, *ray_progress - refracting tracer: …*, *Stack with tf - the scattering form …*, *Exits and verdicts - …*, *Sheet reflection - a declined series …*, *Refracting tracer - …*, *Phase reference - …*, *Measurement - the mirror behind a second solid …*, *Leak direction - a wave held in a mass-law body …* | a thread team per one-ray call: 5.3 CPU-s for 0.46 s wall | `ub::OneThread` | measured 12 → 1 thread: nine cases 0.18 → 0.12 s wall, 2.18 → 0.12 CPU-s; *Measurement - … coupling* 0.28 → 0.45 s wall, 3.4 → 0.45 CPU-s (it fans out many rays per call; decision D3) | unchanged (results do not depend on the thread count) |

Considered and not proposed: *Resonant medium - …* (0.75 s: B 0.37 s, C 0.31 s, D 0.10 s; every row is the
statement at its medium, configuration, thickness, tracer and angle); trimming the asserted angles of S3's section A
(16 → 9 angles would save about 0.5 s; the sweep carries where the default threshold flips from summing to following,
decision D4); fewer frequencies in *ray_progress - the plate rule and the absorption peak …* and the leaf cases (all
under 0.1 s).

Expected after S1–S5 and the switch off: the 237 ray cases about 6 s wall and 7 CPU-s at 12 threads (from 24.8 s and
30.8 CPU-s; the what-if's 6.08 s less the bisection, plus S5), measured alternated in phase 2; the whole suite from
38 s to about 19–20 s.

### 0.7 The diagnostic switch

- One place, read once, in `ray_test_helpers.hpp`:

  ```cpp
  // Diagnostic tables print when the environment variable QD_TEST_DIAG is set (and not "0"); the assertions run either way.
  static inline bool qd_diag()
  {
      static const bool on = [] { const char *v = std::getenv("QD_TEST_DIAG"); return v && v[0] && v[0] != '0'; }();
      return on;
  }
  #define QD_DIAG(...) (qd_diag() ? (void)std::printf(__VA_ARGS__) : (void)0)
  ```

  An expression, so the five ternary uses stay as they are; the arguments are evaluated only when it is on.
  `QD_TEST_VERBOSE` disappears from the ray files.
- The 31 `#ifdef QD_TEST_VERBOSE` blocks become `if (qd_diag())` blocks (or their `std::printf` calls `QD_DIAG`).
  `test_wave_sheet_vs_solid.cpp` reads `QD_TEST_DIAG` in its `verbose()` (it does not include the helpers); with the
  switch on its two tables print for the first time in a baseline (about 25 lines).
- Print-only computations go inside `if (qd_diag())`: S2's sections B and C, S3's B, D and Z60. Nothing inside such a
  block feeds a `CHECK`; each block is read for that before it is moved.
- The name `QD_TEST_DIAG` and not `QD_TEST_VERBOSE`: the out-of-scope files define a macro of that name, and an
  environment variable of the same name invites the assumption that it switches them too.
- Baselines: generated with the switch on, `QD_TEST_DIAG=1 OMP_NUM_THREADS=12 ./build_linux/test_bin "[wave]" >
  test_wave.txt` (and `[acoustic]`, whole suite), so `rowdiff` compares the same printed rows as today; the commands of
  `md/wave.md` 0 and the instrument paragraph of `md/wave.md` 11 are updated in the final phase. A run without the
  switch prints nothing from the ray cases (checked by counting).

### 0.8 The plan

Order of steps; after each one: build (generic and the AVX2 scratch build, `--parallel 12`), the three suites with
`QD_TEST_DIAG=1` (12 threads, one at a time), the AVX2 whole suite, `rowdiff` against the phase 0 runs for the kept
rows, the counts. The coverage check (the per-case build of 0.2) after steps 2, 3 and 6.

1. **The switch** (0.7) without S2/S3: every row identical with the switch on (`rowdiff` 0 differences, the printed
   line counts as today, `test_wave_sheet_vs_solid.cpp`'s tables added), the ray cases printing nothing with it off;
   counts unchanged.
2. **Speed**: S1–S5 (S2/S3 as print-only blocks). Expected rows that move: the S1 rows removed; *bare tf A*'s
   followed summary rows (worst, at, above, least over fewer angles). Counts as below.
3. **Removals** R1–R4, 89.9999 deg into the grazing case; `md/acoustic.md` (R1) updated; `check_docs.py` passes.
4. **Comments**, file by file (helpers first, then the twelve files): stale and redundant comments out, leading
   comments to what is asserted and why it is the physics, labels and document references replaced by the physics they
   stand for, the old names of 0.4.1, the case rename D6 (`md/` updated where named), the dead code of 0.4.3. Code is
   not touched otherwise: every printed row identical (`rowdiff`), counts unchanged.
5. **Optional, D5**: the shared `rp::walk` (and float plane mesh) for the 12 copies; every row identical.
6. **Baselines** regenerated with the switch on; `md/wave.md` 0 and 11.

Expected counts (assertions from the what-if runs; lines and comments estimated, measured at the end):

| | today | after |
| --- | --- | --- |
| `[wave]` | 165594 in 181 cases | 163474 in 178 (−146 R2–R4, +10 grazing, −1984 S1–S4) |
| `[acoustic]` | 42741 in 60 | 42741 in 59 |
| whole suite | 1345968 in 1238 | 1343848 in 1234 (AVX2 1344179) |
| lines of the 12 files | 33,887 | about 31,900 (comments −1,500, removals −233, dead code −100, switch +40); about 31,450 with D5 |
| full-line comments | 4,689 | about 3,000–3,300 |
| ray cases, wall / CPU at 12 threads | 24.8 s / 30.8 s | about 6 s / 7 s |
| printed lines, switch on / off | 22,979 / – | about 22,980 (−45 removed rows, −2 S1 rows, +25 sheet vs solid) / 0 |

Risks:
- **Comment edits that touch code**: a comment block moved or a dead lambda removed shifts no number, but a slip that
  changes an expression would; `rowdiff` (every printed digit) and the twins catch it at every step.
- **The switch hides a check**: only blocks without a `CHECK` / `REQUIRE` / `twin` go behind it, and the assertion
  counts are compared with the switch on and off after step 1 and step 2 (they must be equal).
- **AVX2**: the AVX2 build differs by 331 assertions (kernel-dependent pairs); S1–S4 do not touch those cases, but the
  AVX2 suite is the check, every step.
- **Coverage noise**: two OpenMP threads can race on a counter; a reached line stays reached (a lost update keeps it
  non-zero). The comparison is reached / not reached.
- **Docs**: R1 is in `md/acoustic.md`'s appendix; a missed name fails `check_docs.py`.

### 0.9 Side findings (one line each, no items)

- *ray_progress - accumulated path length*: its NOTE says the first `CHECK` fails until a fix; the fix is in, the case
  passes, the NOTE is stale (goes in step 4).
- `QdMassRef` (acoustic mesh interact) still describes the composite before the plate rule; its case uses only the
  in-slab cosine (step 4).
- *ray_state_update - lossy medium feeds |phi| from medium_gain* describes a mass-column material while using `alpha`
  (step 4).
- The coverage union of the ray cases leaves 104 lines of `ray_mesh_interact.cpp` and 13 of `ray_progress.cpp`
  unreached; task 1 listed the ones that are guards on input or numerics.
- The ten single-ray cases of S5 used 5.3 CPU-s at 12 threads for 0.46 s of wall time; the measure matters more under
  load than the wall time shows.

### 0.10 Decisions for Stephan

- **D1** the removal list R1–R4 (0.5), with 89.9999 deg moved into the grazing case.
- **D2** the speed list S1–S4 (0.6), including the sanity-only sections of the two *Measurement* cases becoming
  print-only (−448 and −68 sanity assertions).
- **D3** S5: `ub::OneThread` on the ten cases, including *Measurement - ray_progress follows the coupling …* (+0.16 s
  wall, −3 CPU-s), or the nine without it.
- **D4** S3's asserted sweep at 16 angles (proposed: keep) or 9 (−0.5 s).
- **D5** the shared `ray_progress` walker (step 5, about −450 lines, rows identical) or not.
- **D6** rename *Layered stack - sheet_case 4 resolves rather than killing the ray* → *Layered stack - an ordered stack
  of two sheets resolves rather than killing the ray*.
- **D7** the environment variable name `QD_TEST_DIAG`.
- **D8** the dead helper `diag_res` removed (named in `md/wave.md` 11's helper list, which would lose the word), or
  kept.
- Case-name prefixes "Diagnostic -" and "Measurement -" stay (renaming them is doc churn for no guard).

Stopped here for the OK on the plan, the removal list and the speed list.

## Phase 1 — the cleanup (2026-10-20/21, decisions D1–D8 of 0.10 as decided)

Scratch `build_linux/scratch/ct1` (not kept). After every step: the generic build and the AVX2 scratch build
(`--parallel 12`, 0 warnings in both), `[wave]`, `[acoustic]` and the whole suite with `QD_TEST_DIAG=1` (12 threads,
one at a time), the AVX2 whole suite, the ray cases with the switch off, `rowdiff` of all four runs against the step
before (against the phase 0 runs for steps 1–3), and the counts. The coverage check (0.2: the per-case gcov build of
both source files, the ray cases only) after steps 2, 3, 4 and 5.

### 1.1 Steps and checks

| step | change | `[wave]` / `[acoustic]` / whole / AVX2 | printed rows against the step before | coverage |
| --- | --- | --- | --- | --- |
| 1 switch | `QD_DIAG` reads `QD_TEST_DIAG` once (0.7); every `#ifdef QD_TEST_VERBOSE` block of the twelve files became `QD_DIAG` or `if (qd_diag())`; `test_wave_sheet_vs_solid.cpp`'s `verbose()` reads the same variable | 165594/181, 42741/60, 1345968/1238, 1346299/1238 | identical; with the switch off the ray cases print nothing, the assertion count is the same | – |
| 2 speed | S1–S4 as planned; S5: `ub::OneThread` on the ten cases | 163610/181 | the two S1 rows gone; *bare tf A*: its header (`of 18 (followed: 9)`) and 46 numbers of the followed summary rows (fewer angles) | unchanged |
| 3 removals | R1–R4; 89.9999 deg added to *Thin sheet - grazing incidence does not depend on the face*; `qd_quad_x_sub` (only R3 used it) | 163474/178, 42741/59, 1343848/1234, 1344179/1234 | the four removed tables; one row added (89.9999 deg) | unchanged |
| 4 comments | file by file (1.2); D6; the dead code of 1.3 | unchanged | identical except the grazing angle column, printed `%.4f` instead of `%.3f` so that 89.9999 does not print as 90.000 (2 numbers "within") | unchanged |
| 5 D5 | `rp::walk`, `rp::Opt`, `rp::planes` in the helpers (1.4) | unchanged | identical (256,771 numbers, whole suite; 256,963 AVX2) | unchanged |
| 6 headers | three printed headers that carried an item label, a `md/` reference and an old name (1.2) | unchanged | the three header lines; every number identical | – (strings only) |

Coverage of the ray cases, phase 0 and after steps 2, 3, 4 and 5: `ray_mesh_interact.cpp` 1679 lines and 3685 branch
keys, `ray_progress.cpp` 461 lines and 524 branch keys; lost none, gained none (237 cases after step 3). The stored
twins: `tests/data/wave_twins.tsv` unchanged, its rows asserted at 1e-12 as before (no twin row belonged to a removed
table). Assertion counts with the switch on and off: equal after every step (after step 6: 206215 in 237 ray cases
both ways).

### 1.2 What changed, per file

- `ray_test_helpers.hpp`: the switch; banners to one line; comments rewritten; `qd_quad_x_sub` and `diag_tf_port`
  removed (1.3); `namespace rp` added (1.4).
- `test_wave_mesh_interact.cpp`: R2, R3, R4 removed; the grazing sweep with 89.9999 deg (+10 assertions); comments;
  unused locals (`rad2deg`, `normal_vecN`, `fbs_angleN`, …).
- `test_wave_progress.cpp`: S1, S3 (sections B and D and the Z60 reference inside `if (qd_diag())`), S5 on three
  cases; eight walker copies and five plane-mesh lambdas replaced (1.4); comments; the stale NOTE of 0.9.
- `test_wave_state_update_chain.cpp`: S2 (sections B and C inside `if (qd_diag())`; their `REQUIRE` on the walk
  became a `continue`, so the rows print as before); S5 on five cases; comments; unused lambdas.
- `test_wave_state_update_branch.cpp`: S4 (the followed rows at 0, 45 and 85 deg); comments.
- `test_wave_state_update_sheet.cpp`: D6 (*Layered stack - an ordered stack of two sheets resolves rather than killing
  the ray*; `md/` named it nowhere but in the task reports); S5 on *Stack with tf*; comments (the old event names
  replaced by what they are: a glued entry, exit or isolated sheet); the header "sheet_case 4" (step 6).
- `test_wave_state_update_spec.cpp`, `test_wave_sheet_vs_solid.cpp`: comments; a commented-out `WARN`, an unused
  constant, the unread `MatDef` fields.
- `test_acoustic_mesh_interact.cpp`: `QdMassRef` replaced by `qd_slab_cos` (the same expression; the only part the case
  used); comments.
- `test_acoustic_progress.cpp`: four walker copies and one plane-mesh lambda replaced (1.4); the mass-law part of the
  refracting-tracer case reduced to its own `ray_progress` calls and checks (same arguments, same assertions); the
  header "on M6" (step 6); comments.
- `test_acoustic_state_update.cpp`: R1 removed; S5 on *Leak direction*; comments; unused lambdas and locals; the
  header's `md/acoustic.md` reference (step 6).
- `test_acoustic_leaf.cpp`: one walker copy replaced (1.4); comments.

Comments: no internal labels, `md/` references, rounds, commit hashes or history are left in any comment of the
twelve files. Searched for the label patterns, `md/`, "round", "commit", "used to", "previously", "no longer",
"until the fix": what remains are material names (A20, B20 …), medium names (M1, M2), section letters of a case's own
tables, and physics wording ("no longer collapse onto", "the commit stage"). Printed
headers keep their section letters; the three that named an item, a document or an old name were rewritten (step 6).

### 1.3 Dead code, re-checked (D8)

Every claim of 0.4.3 was checked again with calls through other helpers counted:
- `diag_res` is used (`lump_res` calls it; `lump_res` is used in `test_acoustic_state_update.cpp` and
  `test_wave_state_update_sheet.cpp`). It stays, and so does its entry in `md/wave.md` 11. My phase 0 claim was wrong.
- `diag_tf_port` has no call anywhere (helpers, the eleven files and every other test file): removed, and its word in
  `md/wave.md` 11's list with it.
- `qd_quad_x_sub`: R3 was its only user; removed in step 3.
- `QdMassRef::base` and `::refl` unread: the struct went (1.2).
- The unused setup of the refracting tracer's mass-law part: removed with the rewrite (1.2).
- Unused locals and lambdas: three compiler passes (`-Wall -Wextra`, the eleven files) until clean, 74 + 10 + 1
  sites.

### 1.4 The shared walker (D5)

`rp::walk(mesh, mi, pal, cf, theta, opt, take)` sends one ray from (−1, 0.37 − tan θ, 0), launched 0.1 m out, through
`ray_progress` until no ray is left or `opt.calls` calls are made, and calls `take(ray)` for every ray in air that
the last call extended; `rp::Ray` gives the direction test, the first and last points, the coefficients and the
polyline length, written with the expressions of the copies (so every printed digit is the same). `rp::Opt` holds the
arguments the copies differed in (calls 30–130, interaction limit 12–60, reflection and transmission limits, gain
floor, thin-slab threshold, tracer, scalar). `rp::planes` makes the faces in planes x = const (the 400 m square of
every copy).

Converted, 13 loops: in `test_wave_progress.cpp` the coupling and declined-series cases, the mirror and reflect-port
cases, the lossy medium, the coherent sheet with tf, the traced solid with tf, the resonant solid; in
`test_acoustic_progress.cpp` the wall behind a traced layer, the bonded plate, the plate rule and the Paris integral;
in `test_acoustic_leaf.cpp` the band case. Each keeps its own arithmetic in its `take`. `rp::planes` replaced five
identical `build` lambdas in `test_wave_progress.cpp`, one in `test_acoustic_progress.cpp`, and six inline plane
meshes.

Kept as they are, with the reason: the rare sheet and the tunnelling sheet (they wait for the one expected path and
read the EM channels), the wall with a stack inside it (it asserts the number of rays launched over all calls), the
ordered stack and the refracting tracer of `test_wave_progress.cpp` (tubes, a forced subdivision, every call
recorded), and the mass-law part of the refracting tracer (its own records per call).

### 1.5 Docs

- `md/wave.md` 0: the three commands with `QD_TEST_DIAG=1`, the new counts, one sentence on the switch. 11: the switch
  sentence replaces the `QD_TEST_VERBOSE` one (with the four print-only sections named); an `rp::walk` entry; the
  sentence on the per-case `trace` copies shortened; `diag_tf_port` out of the list. The line naming R4 now names
  *Ray-Mesh Interact - Public helpers: …*, one of the cases that keeps its guard.
- `md/acoustic.md`: R1's appendix row removed; the first-step command with `QD_TEST_DIAG=1` and 59 cases.
- `tools/acoustic/check_docs.py`: all checks pass (counts against the regenerated baselines included).
- Baselines regenerated with the switch on (the commands of `md/wave.md` 0): `test_wave.txt` 163474 in 178,
  `test_acoustic.txt` 42741 in 59, `test.txt` 1343848 in 1234; `rowdiff` of `test.txt` against the step-6 run:
  identical.

## Phase 2 — final checks (2026-10-21)

### 2.1 Builds, coverage, twins

- Generic and AVX2 builds: 0 warnings; whole suite generic 1343848 assertions in 1234 cases, AVX2 1344179 in 1234 (the
  331 difference of phase 0 unchanged), all passed.
- Coverage of both files by the ray cases: unchanged against phase 0 (1.1). No loss to agree.
- Twins as planned: no row removed, `tests/data/wave_twins.tsv` unmodified.
- Nothing under `src/`, `include/` or `api_*` changed (`git status`).

### 2.2 Run time

CPU time (user + system), `/usr/bin/time`, `flock /tmp/bench.lock`, the old binary (phase 0 tree, prints always) and
the new one alternated, three rounds:

| run | round 1 | round 2 | round 3 | median |
| --- | --- | --- | --- | --- |
| ray cases, 1 thread, before | 24.63 s | 24.93 s | 24.97 s | 24.93 s |
| ray cases, 1 thread, after, switch off | 5.86 s | 5.89 s | 5.84 s | 5.86 s |
| ray cases, 1 thread, after, switch on | 15.69 s | 15.72 s | 15.67 s | 15.69 s |

Paired ratio after (off) / before: 0.238, 0.236, 0.234. With the switch on the print-only sections of S2 and S3 run,
hence the 15.7 s. Whole suite, 12 threads, two rounds alternated: before 38.37 / 38.69 s wall (78.1 / 77.4 CPU-s),
after (switch off) 19.38 / 19.26 s wall (54.2 / 53.4 CPU-s). The ray cases alone at 12 threads, switch off: 5.9 s
wall, 7.0 CPU-s (phase 0: 24.8 s, 30.8 CPU-s). Load during the timings 2.4–4.4 (the whole-suite rounds reached 4.4):
the CPU-time ratios are paired and stand; the 12-thread wall times are provisional.

### 2.3 Final counts

| file | lines | full-line comments / trailing | cases | assertions | printed lines (switch on) |
| --- | --- | --- | --- | --- | --- |
| `test_wave_mesh_interact.cpp` | 2879 → 2651 | 368 / 122 → 273 / 119 | 38 → 35 | 3415 → 3279 | 364 → 342 |
| `test_wave_progress.cpp` | 4462 → 4063 | 359 / 169 → 298 / 164 | 52 | 69837 → 69759 | 2484 → 2482 |
| `test_wave_sheet_vs_solid.cpp` | 454 → 418 | 82 / 7 → 52 / 5 | 2 | 3 | 0 → 27 |
| `test_wave_state_update_spec.cpp` | 1487 → 1484 | 73 / 114 → 71 / 113 | 28 | 737 | 0 |
| `test_wave_state_update_sheet.cpp` | 5206 → 4938 | 1109 / 204 → 841 / 204 | 32 | 7346 | 1612 |
| `test_wave_state_update_chain.cpp` | 4263 → 4118 | 510 / 137 → 399 / 131 | 12 | 15484 → 15036 | 3009 |
| `test_wave_state_update_branch.cpp` | 3334 → 3297 | 391 / 208 → 315 / 208 | 17 | 68772 → 67314 | 6268 |
| `test_acoustic_mesh_interact.cpp` | 331 → 293 | 61 / 20 → 47 / 17 | 2 | 174 | 19 |
| `test_acoustic_progress.cpp` | 842 → 623 | 64 / 21 → 52 / 15 | 5 | 3190 | 1004 |
| `test_acoustic_state_update.cpp` | 6649 → 6271 | 890 / 399 → 693 / 382 | 47 → 46 | 34911 | 6869 → 6854 |
| `test_acoustic_leaf.cpp` | 690 → 656 | 86 / 14 → 84 / 14 | 6 | 4466 | 1005 |
| `ray_test_helpers.hpp` | 3290 → 3300 | 695 / 155 → 631 / 158 | – | – | – |
| total | 33,887 → 32,112 | 4,688 / 1,564 → 3,756 / 1,530 | 241 → 237 | 208335 → 206215 | 22,634 → 22,622 |

Lines and comments are counted with the counter of 0.4 (full-line comments include block-comment and licence lines),
on the committed files and on the working tree; on the committed files it gives 4,688 (0.4 quoted 4,689). Printed lines
here are non-blank lines other than Catch2's own, per file (`-#` file tags); the old binary printed always, its
`test_wave_sheet_vs_solid.cpp` tables only under `QD_TEST_VERBOSE` (0.1 counted with blank lines, hence its 22,979).
With the switch off the ray cases print nothing; the whole suite then prints 853 lines, all from out-of-scope files.
Suites: `[wave]` 163474 in 178, `[acoustic]` 42741 in 59, whole 1343848 in 1234 (AVX2 1344179). Against the plan
(0.8): assertions as expected; lines 32,112 against "about 31,450 with D5" and full-line comments 3,756 against
"3,000–3,300": the leading comments of the cases were rewritten rather than cut, and fewer comments were stale than
estimated.

### 2.4 Side findings (one line each, no items)

- `md/test_split_report.md` still names the case under its old name (D6); it is a report of an earlier task and was
  left.
- The helpers grew by 10 lines: `namespace rp` (99 lines) against the removed helpers, banners and comments.

Stopped here before the commit. Stephan commits; the manager writes the commit message.

## Appendix A — per case

Columns: the case's `TEST_CASE` line; assertions; wall and process CPU time (12 threads, phase 0 marked run);
printed lines; full-line / trailing comment lines (leading comment included); the leading comment's length; lines
flagged stale by pattern (internal label, document or commit reference, history phrase, task-1 old name); lines /
branch keys of the two source files only this case reaches; the action (remove R*, speed S*, rename, comments: a
leading comment over ten lines or a flagged line; keep: nothing beyond the general pass of step 4).

**test_wave_mesh_interact.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | Ray-Mesh Interact - Air to Air (x-z plane) | 23 | 0.000 | 0.000 | 0 | 10 / 4 | 1 | 0, 0, 1, 0 | 14 / 17 | comments |
| 149 | Ray-Mesh Interact - Air to Dielectric Medium (x-z plane) | 22 | 0.000 | 0.000 | 0 | 18 / 8 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 292 | Ray-Mesh Interact - Dielectric Medium to Air (x-y plane, float) | 26 | 0.000 | 0.000 | 0 | 20 / 8 | 0 | 0, 0, 0, 0 | 0 / 21 | keep |
| 453 | Ray-Mesh Interact - Medium to Medium (x-y plane, double) | 14 | 0.000 | 0.000 | 0 | 14 / 4 | 1 | 0, 0, 1, 0 | 0 / 0 | comments |
| 571 | Ray-Mesh Interact - Conductive to Dielectric (x-y plane, double) | 9 | 0.000 | 0.000 | 0 | 18 / 8 | 1 | 0, 0, 1, 0 | 0 / 0 | comments |
| 685 | Ray-Mesh Interact - fRef parameterization equivalence | 4 | 0.000 | 0.000 | 0 | 6 / 0 | 0 | 0, 0, 1, 0 | 0 / 0 | comments |
| 734 | Ray-Mesh Interact - Penetration loss frequency scaling | 1 | 0.000 | 0.000 | 0 | 5 / 0 | 0 | 0, 0, 1, 0 | 0 / 0 | comments |
| 770 | Ray-Mesh Interact - Alpha in-medium absorption (medium_gain) | 2 | 0.000 | 0.000 | 0 | 6 / 2 | 0 | 0, 0, 2, 0 | 0 / 0 | comments |
| 817 | Ray-Mesh Interact - Permeability defaults to 1 (backward compatible) | 13 | 0.000 | 0.000 | 0 | 1 / 4 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 863 | Ray-Mesh Interact - Permeability decouples reflection from refraction | 5 | 0.000 | 0.000 | 0 | 3 / 2 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 913 | Ray-Mesh Interact - Permeability impedance-matches a dense medium | 3 | 0.000 | 0.000 | 0 | 1 / 5 | 0 | 0, 0, 1, 0 | 0 / 0 | comments |
| 955 | Ray-Mesh Interact - eps<->mu swap leaves reflected power invariant | 2 | 0.000 | 0.000 | 0 | 3 / 3 | 3 | 0, 0, 1, 0 | 0 / 0 | comments |
| 993 | Ray-Mesh Interact - Permeability drives in-medium loss (medium_gain) | 3 | 0.000 | 0.000 | 0 | 5 / 3 | 0 | 0, 0, 2, 0 | 0 / 0 | comments |
| 1040 | Ray-Mesh Interact - path_dirN direction contract (45° dielectric, x-z plane) | 14 | 0.000 | 0.000 | 0 | 8 / 5 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1118 | Ray-Mesh Interact - ray_indN compaction round-trip | 9 | 0.000 | 0.001 | 0 | 2 / 4 | 0 | 0, 0, 0, 0 | 1 / 1 | keep |
| 1183 | Ray-Mesh Interact - Scalar interactions (types 3/4/5) gains, invariant, interface gain | 11 | 0.000 | 0.000 | 0 | 6 / 5 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1268 | Ray-Mesh Interact - Transmission factor splits reflection/transmission energy | 12 | 0.000 | 0.009 | 0 | 4 / 0 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1335 | Ray-Mesh Interact - Ray-tube TIR straddle forces whole-tube pass-through | 6 | 0.000 | 0.000 | 0 | 3 / 4 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1389 | Ray-Mesh Interact - path_dirN collapses to undeviated under spine TIR | 11 | 0.000 | 0.000 | 0 | 3 / 1 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1439 | Ray-Mesh Interact - Spherical ray-tube keeps undeviated directions on pass-through | 2 | 0.000 | 0.000 | 0 | 2 / 1 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1476 | Ray-Mesh Interact - Public helpers: refractive_index, interface_gain, medium_gain | 11 | 0.000 | 0.000 | 0 | 6 / 7 | 1 | 1, 0, 0, 0 | 0 / 0 | comments |
| 1537 | Ray-Mesh Interact - out_typeN bit encoding | 11 | 0.000 | 0.000 | 0 | 4 / 10 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1657 | Ray-Mesh Interact - compact = false keeps missed rays in place | 15 | 0.000 | 0.000 | 0 | 5 / 1 | 0 | 0, 0, 0, 0 | 1 / 2 | keep |
| 1712 | Diagnostic - Interface energy split per polarization | 112 | 0.000 | 0.000 | 17 | 18 / 5 | 12 | 0, 0, 2, 1 | 0 / 0 | comments |
| 1796 | Diagnostic - Transmission convention baseline, single interface | 1672 | 0.002 | 0.024 | 197 | 38 / 2 | 28 | 0, 1, 2, 1 | 0 / 0 | comments |
| 1958 | Diagnostic - Thin sheet composite vs analytic slab | 640 | 0.001 | 0.012 | 51 | 41 / 2 | 22 | 0, 0, 4, 0 | 0 / 0 | comments |
| 2133 | Thin sheet - permeability and the lumped columns through ray_mesh_interact | 364 | 0.001 | 0.012 | 53 | 24 / 0 | 13 | 0, 0, 0, 0 | 0 / 0 | comments |
| 2305 | Diagnostic - In-slab cosine projection, two ways | 28 | 0.000 | 0.000 | 10 | 7 / 1 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2370 | Thin sheet - grazing incidence does not depend on the face | 90 | 0.000 | 0.000 | 13 | 13 / 0 | 10 | 0, 0, 0, 0 | 0 / 0 | +89.9999 (R2) |
| 2424 | Thin sheet - mesh subdivision invariance | 49 | 0.004 | 0.048 | 10 | 10 / 0 | 10 | 0, 0, 0, 0 | 0 / 0 | remove R3 |
| 2480 | Diagnostic - Derived lengths and the tangency limit | 42 | 0.000 | 0.000 | 16 | 15 / 3 | 12 | 0, 0, 3, 0 | 0 / 0 | remove R2, comments |
| 2543 | Resonance - the public index and in-medium gain carry the permittivity pole | 9 | 0.000 | 0.000 | 8 | 3 / 2 | 3 | 0, 1, 0, 0 | 0 / 0 | comments |
| 2585 | Acoustic - interface_gain is the penetration loss alone; a coiA column is ignored | 55 | 0.000 | 0.000 | 0 | 9 / 0 | 4 | 1, 0, 0, 0 | 0 / 0 | remove R4, comments |
| 2625 | Coincidence column - a table that carries coiA loads and the column changes nothing | 96 | 0.001 | 0.023 | 0 | 6 / 3 | 5 | 1, 0, 0, 0 | 0 / 0 | comments |
| 2680 | Acoustic - convergence with acoustic terms absent | 5 | 0.000 | 0.002 | 0 | 5 / 0 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2720 | Acoustic - scalar total internal reflection | 12 | 0.000 | 0.000 | 0 | 8 / 8 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2774 | Acoustic - pass-through calibration | 10 | 0.000 | 0.000 | 0 | 10 / 3 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2848 | Acoustic - embedded material counted once | 2 | 0.000 | 0.009 | 0 | 4 / 4 | 0 | 0, 0, 0, 0 | 0 / 1 | keep |

**test_wave_progress.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 13 | ray_progress - single reflection | 38 | 0.000 | 0.000 | 0 | 5 / 5 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 55 | ray_progress - single transmission | 33 | 0.000 | 0.000 | 0 | 1 / 3 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 85 | ray_progress - reflection and transmission in one call | 37 | 0.000 | 0.000 | 0 | 4 / 1 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 118 | ray_progress - no hit terminates the trace | 21 | 0.000 | 0.000 | 0 | 3 / 0 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 142 | ray_progress - every ray hits but none survives the gain gate | 20 | 0.000 | 0.000 | 0 | 3 / 1 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 165 | ray_progress - max_no_interactions = 0 disables tracing | 17 | 0.000 | 0.000 | 0 | 0 / 0 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 185 | ray_progress - reflection and transmission passes can be switched off | 60 | 0.000 | 0.000 | 0 | 0 / 0 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 223 | ray_progress - per-ray counters gate the next generation | 40 | 0.000 | 0.000 | 0 | 6 / 2 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 290 | ray_progress - beam subdivision | 84 | 0.000 | 0.000 | 0 | 6 / 5 | 0 | 0, 0, 1, 0 | 0 / 0 | comments |
| 346 | ray_progress - subdivision requires beam data and a positive limit | 10 | 0.000 | 0.011 | 0 | 0 / 3 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 393 | ray_progress - every ray subdivided | 127 | 0.000 | 0.000 | 0 | 2 / 1 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 422 | ray_progress - beam buffers survive the assembly | 105 | 0.000 | 0.000 | 0 | 3 / 1 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 462 | ray_progress - compaction preserves per-ray results | 26414 | 0.002 | 0.013 | 0 | 14 / 3 | 10 | 0, 0, 0, 0 | 0 / 0 | keep |
| 532 | ray_progress - the reported hit count is independent of subdivision | 4 | 0.001 | 0.011 | 0 | 4 / 2 | 3 | 0, 1, 0, 0 | 0 / 0 | comments |
| 566 | ray_progress - multi-frequency | 25 | 0.000 | 0.000 | 0 | 8 / 6 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 635 | ray_progress - scalar layout | 25 | 0.000 | 0.000 | 0 | 0 / 2 | 0 | 0, 0, 0, 0 | 1 / 2 | keep |
| 659 | ray_progress - medium state across a slab | 15 | 0.000 | 0.000 | 0 | 4 / 3 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 694 | ray_progress - sub-mesh partitioning does not change the result | 138 | 0.000 | 0.000 | 0 | 2 / 1 | 0 | 0, 0, 0, 0 | 2 / 3 | keep |
| 738 | ray_progress - a trace terminates and respects its limits | 3334 | 0.001 | 0.012 | 0 | 0 / 2 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 779 | ray_progress - repeated calls are deterministic | 1031 | 0.000 | 0.000 | 0 | 2 / 0 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 821 | ray_progress - accumulated path length | 9 | 0.000 | 0.000 | 0 | 14 / 2 | 9 | 0, 0, 2, 0 | 0 / 0 | comments |
| 867 | ray_progress - delegated intersection matches the internal one | 4722 | 0.001 | 0.023 | 0 | 7 / 2 | 3 | 0, 0, 1, 0 | 0 / 0 | comments |
| 968 | ray_progress - delegated intersection across a multi-generation trace | 3831 | 0.001 | 0.012 | 0 | 3 / 1 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1007 | ray_progress - delegated intersection ignores the acceleration structure | 458 | 0.000 | 0.000 | 0 | 2 / 2 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1035 | ray_progress - delegated intersection with max_no_interactions = 0 | 18 | 0.000 | 0.000 | 0 | 1 / 1 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1055 | ray_progress - delegated intersection validation | 11 | 0.000 | 0.000 | 0 | 1 / 0 | 0 | 0, 0, 0, 0 | 6 / 16 | keep |
| 1114 | ray_progress - delegated subdivision flag matches the internal one | 7996 | 0.002 | 0.024 | 0 | 5 / 11 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1247 | ray_progress - the delegated flag count is the reported n_subdiv | 176 | 0.000 | 0.000 | 0 | 3 / 0 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1284 | ray_progress - the delegated flag overrides the geometry | 72 | 0.000 | 0.000 | 0 | 2 / 4 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1328 | ray_progress - delegated subdivision flag across a multi-generation trace | 12046 | 0.003 | 0.036 | 0 | 3 / 0 | 3 | 0, 0, 0, 0 | 0 / 2 | keep |
| 1369 | ray_progress - delegated subdivision flag validation | 10 | 0.000 | 0.000 | 0 | 1 / 1 | 0 | 0, 0, 0, 0 | 3 / 6 | keep |
| 1425 | ray_progress - a flagged ray that misses the mesh is still subdivided | 53 | 0.000 | 0.000 | 0 | 4 / 5 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1466 | ray_progress - the end-of-trace signal accounts for subdivision | 58 | 0.000 | 0.000 | 0 | 2 / 2 | 2 | 0, 0, 0, 0 | 1 / 3 | keep |
| 1502 | ray_progress - a flagged ray past its subdivision limit is still subdivided | 14 | 0.000 | 0.000 | 0 | 2 / 2 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1543 | ray_progress - a flagged ray inside a medium inherits its propagation state | 27 | 0.000 | 0.000 | 0 | 8 / 5 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1589 | ray_progress - the outside-a-medium branch still recomputes and clears | 30 | 0.000 | 0.011 | 0 | 3 / 3 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1626 | ray_progress - hit and subdivision counts are independent | 560 | 0.000 | 0.000 | 0 | 4 / 4 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1683 | ray_progress - input validation | 44 | 0.001 | 0.001 | 0 | 0 / 14 | 0 | 0, 0, 0, 0 | 26 / 56 | keep |
| 1872 | ray_progress - the material count is limited by the 14-bit state index | 2 | 0.001 | 0.012 | 0 | 0 / 0 | 0 | 0, 0, 0, 0 | 1 / 2 | keep |
| 1889 | Measurement - ray_progress follows the coupling between the series of a layered wall | 864 | 0.269 | 3.228 | 398 | 16 / 7 | 10 | 1, 0, 0, 0 | 0 / 0 | S5, comments |
| 2130 | ray_progress - a declined series at a sheet is followed and not dropped | 184 | 0.053 | 0.636 | 69 | 17 / 4 | 8 | 0, 0, 0, 0 | 0 / 0 | S5 |
| 2333 | Measurement - ray_progress and the mirror behind a back reflection | 1628 | 1.730 | 1.784 | 529 | 25 / 4 | 6 | 1, 0, 0, 0 | 0 / 0 | S3, comments |
| 2644 | ray_progress - the reflect port of a sheet lies where its transmit port puts the layer | 979 | 9.565 | 9.564 | 249 | 25 / 5 | 8 | 0, 0, 0, 0 | 0 / 0 | S1 |
| 2880 | ray_progress - a lossy medium has no critical angle and a band takes one path | 480 | 0.039 | 0.039 | 127 | 14 / 6 | 13 | 0, 0, 0, 0 | 0 / 0 | comments |
| 3083 | ray_progress - an ordered stack leaves the ray on the line it arrived on | 320 | 0.009 | 0.009 | 129 | 15 / 5 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3329 | ray_progress - a coherent sheet with tf that is rarer than the invariant passes no more... | 528 | 0.008 | 0.008 | 124 | 5 / 8 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3435 | ray_progress - a coherent sheet rarer than the invariant without tf tunnels | 344 | 0.008 | 0.008 | 64 | 8 / 8 | 6 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3572 | ray_progress - a coherent sheet with tf creates nothing and has no gate on the invariant | 752 | 0.102 | 0.102 | 228 | 11 / 6 | 10 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3725 | ray_progress - a traced solid with tf creates nothing | 908 | 0.089 | 0.089 | 233 | 8 / 5 | 7 | 0, 0, 0, 0 | 0 / 5 | keep |
| 3863 | ray_progress - a wall with a stack inside it is on its matrix with the paths of a singl... | 72 | 0.029 | 0.029 | 29 | 6 / 1 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3978 | ray_progress - a resonant solid is its medium at every frequency of a band | 240 | 0.002 | 0.002 | 44 | 6 / 1 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 4091 | ray_progress - refracting tracer: the segment out of a lossy medium is the direction th... | 823 | 0.013 | 0.152 | 291 | 31 / 9 | 12 | 0, 0, 0, 0 | 0 / 18 | S5, comments |

**test_wave_sheet_vs_solid.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 386 | Collapsed duplicate equals a single sheet | 2 | 0.000 | 0.000 | 0 | 9 / 1 | 3 | 0, 0, 1, 0 | 0 / 0 | comments |
| 428 | thk = 0 is identical to no sheet at all | 1 | 0.000 | 0.000 | 0 | 4 / 0 | 4 | 0, 0, 1, 0 | 0 / 0 | comments |

**test_wave_state_update_spec.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | ray_state_update - state and code encoding | 17 | 0.000 | 0.000 | 0 | 2 / 10 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 33 | ray_state_update - dispatch: o-i entry family | 25 | 0.000 | 0.000 | 0 | 2 / 2 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 68 | ray_state_update - dispatch: i-o exit family | 35 | 0.000 | 0.000 | 0 | 5 / 4 | 2 | 0, 0, 0, 0 | 0 / 1 | keep |
| 116 | ray_state_update - dispatch: material-to-material family | 23 | 0.000 | 0.000 | 0 | 2 / 5 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 150 | ray_state_update - dispatch: corner families | 29 | 0.000 | 0.000 | 0 | 2 / 0 | 1 | 0, 0, 0, 0 | 0 / 3 | keep |
| 190 | ray_state_update - dispatch: no crossing and unmatched codes | 21 | 0.000 | 0.000 | 0 | 0 / 1 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 215 | ray_state_update - dispatch: reflection pass | 31 | 0.000 | 0.000 | 0 | 1 / 3 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 254 | ray_state_update - flag semantics | 22 | 0.000 | 0.011 | 0 | 2 / 4 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 302 | ray_state_update - Airy factor against the closed form | 33 | 0.000 | 0.000 | 0 | 3 / 1 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 356 | ray_state_update - survival gate thresholds on eps | 40 | 0.000 | 0.000 | 0 | 2 / 1 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 411 | ray_state_update - lossy medium feeds \|phi\| from medium_gain | 3 | 0.000 | 0.000 | 0 | 7 / 4 | 4 | 0, 0, 1, 0 | 0 / 0 | comments |
| 457 | ray_state_update - transmission factor folds into the resolved gain | 4 | 0.000 | 0.000 | 0 | 4 / 3 | 3 | 0, 0, 1, 0 | 0 / 0 | comments |
| 485 | ray_state_update - resolve clamp near the Airy pole | 19 | 0.000 | 0.000 | 0 | 2 / 6 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 542 | ray_state_update - cross-pass invariance of the resolve decision | 28 | 0.000 | 0.000 | 0 | 2 / 3 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 588 | ray_state_update - a resolved ray passes an exit transparently | 8 | 0.000 | 0.000 | 0 | 2 / 2 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 605 | ray_state_update - gain consistency and batch determinism | 44 | 0.000 | 0.000 | 0 | 4 / 1 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 672 | ray_state_update - optional arguments | 34 | 0.000 | 0.000 | 0 | 3 / 3 | 0 | 0, 0, 0, 0 | 0 / 4 | keep |
| 759 | ray_state_update - gain operation semantics and probe handling | 23 | 0.000 | 0.000 | 0 | 0 / 3 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 809 | ray_state_update - energy conservation of a lossless slab | 14 | 0.000 | 0.000 | 0 | 4 / 11 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 853 | ray_state_update - energy safety of a lossy slab | 4 | 0.000 | 0.000 | 0 | 2 / 2 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 895 | ray_state_update - scalar refraction (interaction type 5) | 17 | 0.000 | 0.000 | 0 | 0 / 2 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |
| 924 | ray_state_update - path direction correction (VBS) | 7 | 0.000 | 0.000 | 0 | 3 / 2 | 2 | 0, 0, 0, 0 | 0 / 1 | keep |
| 966 | ray_state_update - accumulated distance ledger | 11 | 0.000 | 0.000 | 0 | 4 / 16 | 0 | 0, 0, 0, 0 | 0 / 2 | keep |
| 1042 | ray_state_update - permeability affects the refractive index | 3 | 0.000 | 0.000 | 0 | 1 / 2 | 1 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1059 | ray_state_update - a resonance shifts the slab factor, coincidence columns leave a wave... | 5 | 0.000 | 0.000 | 0 | 3 / 2 | 1 | 1, 0, 1, 0 | 0 / 0 | comments |
| 1101 | ray_state_update - type parity between double and float | 70 | 0.000 | 0.000 | 0 | 0 / 0 | 0 | 0, 0, 0, 0 | 0 / 27 | keep |
| 1107 | ray_state_update - ray_ind mapping and read-only inputs | 29 | 0.000 | 0.000 | 0 | 6 / 5 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1183 | ray_state_update - input validation | 138 | 0.001 | 0.012 | 0 | 1 / 16 | 1 | 0, 0, 1, 0 | 21 / 75 | comments |

**test_wave_state_update_sheet.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | Diagnostic - Slab factor per polarization | 24 | 0.000 | 0.000 | 10 | 32 / 1 | 22 | 0, 0, 2, 0 | 0 / 0 | comments |
| 101 | Diagnostic - Entry boundary evaluated at two angles | 15 | 0.000 | 0.000 | 15 | 40 / 2 | 22 | 0, 0, 1, 0 | 0 / 0 | comments |
| 227 | Diagnostic - Series collapse when the previous medium matches the slab | 32 | 0.000 | 0.000 | 19 | 14 / 0 | 13 | 0, 0, 0, 2 | 0 / 0 | comments |
| 282 | Diagnostic - Host layer carried through a matched sheet | 20 | 0.000 | 0.000 | 7 | 23 / 1 | 13 | 0, 0, 2, 1 | 0 / 0 | comments |
| 351 | Diagnostic - Excess-phase walk-off convention | 29 | 0.000 | 0.000 | 9 | 43 / 3 | 28 | 2, 0, 0, 3 | 0 / 0 | comments |
| 448 | Thin sheet - the redo sits on the analytic slab at grazing incidence | 27 | 0.000 | 0.000 | 12 | 16 / 2 | 13 | 0, 0, 0, 1 | 0 / 0 | comments |
| 499 | Glued sheet - the near-to-far flux conversion | 473 | 0.001 | 0.007 | 70 | 45 / 14 | 27 | 0, 0, 0, 8 | 0 / 0 | comments |
| 670 | Sheet in a medium - the composite at the exact in-medium cosine | 400 | 0.001 | 0.012 | 27 | 84 / 16 | 33 | 0, 0, 4, 8 | 0 / 0 | comments |
| 904 | Exit boundary - the port at the recovered angle | 84 | 0.000 | 0.000 | 33 | 29 / 6 | 16 | 0, 0, 1, 1 | 0 / 0 | comments |
| 1016 | Traced solid - air \| slab \| air chain against the analytic slab | 144 | 0.000 | 0.011 | 32 | 28 / 2 | 16 | 0, 0, 0, 0 | 0 / 0 | comments |
| 1124 | Permeability - the traced solid and a magnetic layer in a stack | 188 | 0.000 | 0.000 | 41 | 69 / 11 | 24 | 0, 0, 3, 2 | 0 / 0 | comments |
| 1420 | Layered stack - the characteristic-matrix reference against the single-layer oracles | 161 | 0.000 | 0.000 | 74 | 21 / 4 | 12 | 0, 0, 0, 2 | 0 / 0 | comments |
| 1537 | Permeability - the sheet, stack and glued paths | 336 | 0.000 | 0.011 | 90 | 71 / 10 | 21 | 0, 1, 0, 3 | 0 / 0 | comments |
| 1831 | Layered stack - a host layer bounded by a sheet | 98 | 0.000 | 0.000 | 23 | 43 / 0 | 27 | 2, 0, 1, 2 | 0 / 0 | comments |
| 1961 | Layered stack - sheet_case 4 resolves rather than killing the ray | 70 | 0.000 | 0.000 | 13 | 29 / 7 | 20 | 0, 0, 0, 4 | 0 / 0 | rename, comments |
| 2084 | Lumped columns - interface gain on the sheet, glued and embedded paths | 270 | 0.001 | 0.011 | 25 | 43 / 8 | 27 | 0, 0, 0, 1 | 0 / 0 | comments |
| 2251 | Lumped columns - the transmission factor on a sheet, both ports | 860 | 0.003 | 0.025 | 110 | 53 / 8 | 31 | 0, 0, 0, 1 | 0 / 0 | comments |
| 2500 | Lumped columns - a permittivity resonance is part of the medium on the sheet paths | 170 | 0.001 | 0.012 | 38 | 27 / 6 | 20 | 0, 1, 1, 0 | 0 / 0 | comments |
| 2679 | Neighbour tf - each face of a sheet is owed by the pair of media that form it | 260 | 0.002 | 0.024 | 97 | 25 / 8 | 15 | 0, 0, 0, 3 | 0 / 0 | comments |
| 2861 | Stack with tf - the scattering form against its reference, produce mode and itself | 758 | 0.005 | 0.060 | 176 | 30 / 19 | 14 | 0, 0, 0, 0 | 0 / 1 | S5, comments |
| 3128 | Reciprocity - glued entry against glued exit | 137 | 0.001 | 0.001 | 23 | 22 / 6 | 20 | 0, 0, 0, 0 | 0 / 0 | comments |
| 3242 | Lined on both faces - a glued exit behind a sheet exits into air | 356 | 0.002 | 0.024 | 120 | 35 / 10 | 12 | 0, 0, 0, 0 | 0 / 0 | comments |
| 3496 | Layered wall - a glued exit behind a bare-entered second layer exits into air | 808 | 0.003 | 0.036 | 177 | 44 / 16 | 21 | 0, 0, 0, 0 | 0 / 0 | comments |
| 3880 | Recovery behind a sheet - the leg after a sheet crossing at the exact in-medium cosine | 96 | 0.001 | 0.012 | 65 | 32 / 3 | 16 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4056 | Measurement - a host entered from a lossy solid across a bare face | 432 | 0.001 | 0.012 | 60 | 26 / 1 | 14 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4171 | Recovery behind a lossy solid - EM, a second transition and a sheet inside the host | 336 | 0.002 | 0.024 | 80 | 19 / 2 | 12 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4374 | Recovery behind a lossy solid - a face that was evaluated at a real angle keeps the sol... | 104 | 0.000 | 0.000 | 22 | 20 / 1 | 15 | 0, 0, 0, 0 | 0 / 1 | comments |
| 4470 | Sheet inside a host - the mirror behind the ray is backed by the host | 256 | 0.001 | 0.012 | 70 | 12 / 3 | 11 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4546 | ray_state_update - the material index is 14 bits | 6 | 0.001 | 0.012 | 0 | 4 / 1 | 1 | 0, 0, 0, 0 | 1 / 2 | keep |
| 4580 | Lumped columns - an ordered stack: what the core carries, by matrix and by scattering | 278 | 0.001 | 0.012 | 51 | 41 / 16 | 18 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4829 | Total reflection of the outer pair - the sheet stays in the result | 72 | 0.000 | 0.011 | 37 | 50 / 10 | 34 | 0, 0, 0, 3 | 0 / 0 | comments |
| 5001 | Leak chain - each face at the cosine the wave has and the series on a closed far face | 46 | 0.000 | 0.000 | 52 | 35 / 7 | 15 | 0, 0, 0, 0 | 0 / 0 | comments |

**test_wave_state_update_chain.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | Recovery gate - the hit code a traced run carries | 437 | 0.003 | 0.026 | 98 | 37 / 6 | 13 | 0, 0, 0, 0 | 0 / 0 | comments |
| 269 | Exits and verdicts - every exit at the cosine the wave has there | 409 | 0.005 | 0.060 | 162 | 38 / 13 | 7 | 2, 0, 0, 0 | 0 / 0 | S5, comments |
| 734 | Sheet reflection - a declined series is followed and not dropped | 528 | 0.011 | 0.133 | 134 | 24 / 9 | 11 | 0, 0, 0, 0 | 0 / 0 | S5, comments |
| 1006 | Measurement - the mirror behind the ray after a back reflection | 3464 | 9.176 | 9.234 | 791 | 47 / 16 | 20 | 1, 0, 0, 0 | 0 / 0 | S2, comments |
| 1493 | Sheet reflect port - where the layer lies and what the echo pays | 1374 | 0.177 | 0.177 | 335 | 78 / 14 | 19 | 0, 0, 0, 0 | 0 / 0 | comments |
| 1998 | Direction - a ray leaves a parallel slab parallel to how it arrived | 364 | 0.002 | 0.014 | 115 | 35 / 4 | 22 | 0, 0, 0, 0 | 0 / 0 | comments |
| 2226 | Refracting tracer - a traced bend is known from the state and the leg is exact | 4524 | 0.021 | 0.252 | 292 | 66 / 11 | 20 | 0, 0, 0, 0 | 0 / 0 | S5, comments |
| 2660 | Phase reference - a traversal carries the phase at the point the polyline leaves at | 1370 | 0.017 | 0.193 | 394 | 59 / 11 | 18 | 0, 0, 0, 0 | 0 / 1 | S5, comments |
| 3060 | Leak direction - the direction a closed face writes for a wave launched inside a solid | 348 | 0.002 | 0.024 | 79 | 12 / 2 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3224 | Measurement - the mirror behind a second solid and behind a sheet inside a wall | 660 | 0.025 | 0.300 | 170 | 40 / 9 | 18 | 1, 0, 0, 0 | 0 / 0 | S5, comments |
| 3598 | Lossy verdict - a lossy second medium has no critical angle | 1670 | 0.010 | 0.065 | 379 | 63 / 35 | 20 | 0, 0, 0, 0 | 0 / 0 | comments |
| 4226 | Far from the origin - a thin dielectric slab transmits what it does at the origin, summ... | 336 | 0.040 | 0.040 | 116 | 7 / 1 | 7 | 0, 1, 1, 0 | 0 / 0 | comments |

**test_wave_state_update_branch.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | Unentered branches - a corner with a sheet is the sheet alone once the state is kept | 346 | 0.000 | 0.011 | 36 | 6 / 9 | 6 | 0, 0, 0, 0 | 9 / 23 | keep |
| 94 | Unentered branches - the displacement budget caps what a sheet takes back out of its host | 1692 | 0.008 | 0.056 | 193 | 20 / 2 | 12 | 0, 0, 0, 0 | 0 / 0 | comments |
| 260 | Unentered branches - the core's loss cap is a floor on the transmit port and leaves the... | 224 | 0.001 | 0.001 | 51 | 6 / 4 | 4 | 0, 0, 0, 0 | 0 / 0 | keep |
| 348 | Unentered branches - sheet and stack and slab series at float against double | 576 | 0.007 | 0.007 | 198 | 4 / 4 | 4 | 0, 0, 0, 0 | 0 / 0 | keep |
| 414 | Unentered branches - the followed mark and the resolved flag across abutting and nested... | 227 | 0.003 | 0.003 | 41 | 15 / 23 | 10 | 0, 0, 1, 0 | 0 / 1 | comments |
| 551 | Unentered branches - a verdict handed in on a leg known not parallel and near a pole | 60 | 0.000 | 0.000 | 23 | 6 / 6 | 6 | 0, 0, 0, 0 | 0 / 0 | keep |
| 648 | Scalar closed face - a lossy box on a rigid wall reflects no more than it received and ... | 834 | 0.046 | 0.046 | 182 | 31 / 4 | 19 | 0, 0, 0, 0 | 0 / 0 | comments |
| 785 | One solver - a sheet and a stack are one sum and no mirror is clamped | 1248 | 0.014 | 0.014 | 299 | 15 / 16 | 6 | 1, 0, 0, 0 | 0 / 0 | comments |
| 1019 | Budget - a sheet or a stack inside a host is one wall on every port whether it is cappe... | 412 | 0.036 | 0.036 | 98 | 4 / 1 | 4 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1087 | Resonant medium - a permittivity resonance is part of the medium a wave crosses and tra... | 4256 | 0.750 | 0.750 | 1744 | 19 / 1 | 7 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1256 | Unentered branches - ray_mesh_interact's own glued composite against the redo | 194 | 0.001 | 0.001 | 81 | 11 / 18 | 10 | 0, 0, 0, 0 | 0 / 0 | keep |
| 1404 | Stack geometry - an ordered stack leaves the ray on the line it arrived on | 1872 | 0.015 | 0.015 | 398 | 30 / 22 | 12 | 0, 0, 1, 0 | 0 / 0 | comments |
| 1638 | Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more t... | 1752 | 0.022 | 0.022 | 648 | 40 / 26 | 19 | 1, 0, 0, 0 | 0 / 0 | comments |
| 1956 | Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the chara... | 1732 | 0.025 | 0.025 | 514 | 58 / 24 | 24 | 0, 0, 0, 0 | 0 / 0 | comments |
| 2370 | Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the ... | 37038 | 0.281 | 0.281 | 1028 | 72 / 32 | 34 | 0, 0, 0, 0 | 0 / 0 | comments |
| 2864 | Bare tf - a bare face with tf is a transformer where it is open and a flat leak where i... | 7051 | 1.213 | 1.213 | 645 | 8 / 11 | 7 | 0, 0, 0, 0 | 1 / 2 | S4 |
| 3106 | Closed face - a coating met from inside a dense solid | 9258 | 0.063 | 0.063 | 156 | 42 / 5 | 27 | 0, 0, 0, 0 | 0 / 0 | comments |

**test_acoustic_mesh_interact.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 127 | Thin sheet - mass-law angle dependence | 168 | 0.001 | 0.007 | 20 | 27 / 6 | 6 | 1, 0, 0, 0 | 0 / 0 | comments |
| 279 | Acoustic - medium_gain in-medium loss | 6 | 0.000 | 0.000 | 0 | 10 / 6 | 0 | 0, 0, 0, 0 | 0 / 0 | keep |

**test_acoustic_progress.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 9 | ray_progress - the wall behind a traced layer is met at the incidence angle at every fr... | 1080 | 0.043 | 0.100 | 403 | 12 / 5 | 11 | 0, 0, 0, 0 | 0 / 0 | comments |
| 183 | ray_progress - the bonded plate at any depth | 1968 | 0.056 | 0.056 | 534 | 21 / 7 | 9 | 4, 0, 0, 0 | 0 / 40 | comments |
| 424 | ray_progress - refracting tracer: the segment out of a lossy medium is the direction th... | 4 | 0.000 | 0.000 | 6 | 6 / 7 | 3 | 1, 0, 0, 0 | 0 / 0 | comments |
| 597 | ray_progress - the plate rule and the absorption peak at every frequency of a band | 120 | 0.003 | 0.027 | 44 | 8 / 1 | 7 | 4, 0, 0, 0 | 0 / 0 | comments |
| 710 | ray_progress - the random-incidence absorption of a peaked lining is the Paris integral... | 18 | 0.035 | 0.065 | 22 | 13 / 1 | 10 | 3, 0, 0, 0 | 0 / 0 | comments |

**test_acoustic_state_update.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 53 | ray_state_update - mass-law material is excluded from slab resolution | 4 | 0.000 | 0.000 | 0 | 8 / 1 | 4 | 1, 0, 0, 0 | 0 / 0 | comments |
| 95 | Lumped columns - a bounding sheet: tf is a layer, a mass-law sheet a half-space | 241 | 0.001 | 0.001 | 55 | 30 / 10 | 27 | 0, 0, 2, 1 | 0 / 0 | comments |
| 217 | Stack with a mass-law layer - resolved by segments, not killed | 292 | 0.002 | 0.022 | 64 | 26 / 18 | 21 | 1, 1, 0, 0 | 0 / 0 | comments |
| 381 | Leak reciprocity - a mass-law panel on a rigid wall as a sheet and as two traced solids | 34 | 0.000 | 0.000 | 20 | 35 / 10 | 15 | 1, 0, 0, 0 | 0 / 0 | comments |
| 535 | Bonded mass - sheet on sheet, sheet on solid and solid on solid as one plate | 208 | 0.002 | 0.024 | 121 | 70 / 38 | 27 | 3, 0, 0, 0 | 0 / 0 | comments |
| 961 | Ports of a mass-law face from inside - the reflect port is the other direction's | 84 | 0.000 | 0.000 | 11 | 16 / 10 | 16 | 1, 1, 0, 0 | 0 / 0 | comments |
| 1039 | Bonded mass - a mass-law sheet inside a mass-law wall | 72 | 0.003 | 0.033 | 25 | 18 / 17 | 16 | 2, 0, 0, 0 | 0 / 0 | comments |
| 1157 | Leak direction - a closed face hands the wall behind a traced layer the direction the w... | 1639 | 0.024 | 0.081 | 568 | 53 / 27 | 21 | 0, 0, 0, 0 | 0 / 18 | comments |
| 1583 | Bonded mass - the plate at any depth | 1864 | 0.042 | 0.042 | 1023 | 49 / 30 | 20 | 6, 1, 0, 0 | 1 / 4 | comments |
| 2088 | Unentered branches - a sheet without the mass column of its mass-law host's permittivit... | 88 | 0.001 | 0.001 | 29 | 7 / 10 | 7 | 1, 0, 0, 0 | 0 / 1 | comments |
| 2159 | Unentered branches - a stack of a coherent and a mass-law layer against the laminate me... | 120 | 0.003 | 0.003 | 32 | 10 / 4 | 6 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2219 | Mass cosine - a mass-law sheet behind a refracting layer forms its term at the cosine t... | 1054 | 0.018 | 0.018 | 273 | 34 / 14 | 8 | 1, 0, 0, 0 | 1 / 4 | comments |
| 2451 | Held lining - a mass-law sheet behind a wave held in a mass-law wall is the same wall a... | 1292 | 0.022 | 0.022 | 438 | 59 / 24 | 23 | 1, 1, 0, 1 | 0 / 3 | comments |
| 2788 | Lumped columns - an ordered stack: what the core carries, by matrix and by scattering -... | 36 | 0.000 | 0.000 | 8 | 8 / 12 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 2887 | Total reflection of the outer pair - the sheet stays in the result - the mass-law part | 25 | 0.000 | 0.000 | 10 | 23 / 3 | 2 | 1, 0, 0, 1 | 0 / 0 | comments |
| 2994 | Leak chain - each face at the cosine the wave has and the series on a closed far face -... | 2 | 0.000 | 0.005 | 6 | 12 / 0 | 2 | 1, 1, 0, 0 | 0 / 0 | comments |
| 3044 | Direction - a ray leaves a parallel slab parallel to how it arrived - the mass-law part | 22 | 0.000 | 0.000 | 16 | 10 / 1 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3154 | Refracting tracer - a traced bend is known from the state and the leg is exact - the ma... | 55 | 0.000 | 0.000 | 30 | 16 / 3 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3337 | Phase reference - a traversal carries the phase at the point the polyline leaves at - t... | 84 | 0.002 | 0.035 | 30 | 11 / 7 | 2 | 1, 0, 0, 0 | 0 / 0 | comments |
| 3507 | Leak direction - a wave held in a mass-law body keeps its direction and a closed face h... | 2568 | 0.016 | 0.192 | 297 | 27 / 8 | 19 | 0, 0, 0, 0 | 0 / 0 | S5, comments |
| 3741 | Measurement - the mirror behind a second solid and behind a sheet inside a wall - the m... | 30 | 0.001 | 0.001 | 22 | 10 / 5 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 3936 | Lossy verdict - a lossy second medium has no critical angle - the mass-law part | 88 | 0.001 | 0.012 | 43 | 15 / 28 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 4194 | Unentered branches - ray_mesh_interact's own glued composite against the redo - the mas... | 70 | 0.000 | 0.011 | 30 | 2 / 12 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 4260 | Closed plate - a mass-law layer the wave is evanescent in passes the plate rule of its ... | 1718 | 0.019 | 0.051 | 325 | 56 / 20 | 18 | 2, 1, 0, 0 | 0 / 4 | comments |
| 4570 | Stack geometry - an ordered stack leaves the ray on the line it arrived on - the mass-l... | 1024 | 0.008 | 0.008 | 189 | 13 / 21 | 2 | 0, 0, 1, 0 | 0 / 0 | comments |
| 4720 | Rare sheet - a coherent sheet with tf that is rarer than the invariant passes no more t... | 0 | 0.000 | 0.000 | 16 | 7 / 13 | 3 | 1, 0, 0, 0 | 0 / 0 | remove R1, comments |
| 4795 | Flat branch - a coherent sheet rarer than the invariant without tf tunnels on the chara... | 18 | 0.001 | 0.001 | 22 | 11 / 1 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 4874 | Sheet tf - the face of a coherent sheet with tf creates nothing and has no gate on the ... | 12 | 0.001 | 0.001 | 39 | 20 / 20 | 2 | 0, 0, 0, 0 | 0 / 0 | keep |
| 5049 | Bare tf - a bare face with tf is a transformer where it is open and a flat leak where i... | 48 | 0.000 | 0.000 | 28 | 3 / 2 | 3 | 0, 0, 0, 0 | 0 / 0 | keep |
| 5104 | Closed face - tf at a closed face is a pad and the two ports of a sheet on it add up | 16026 | 0.250 | 0.250 | 626 | 57 / 10 | 31 | 0, 0, 0, 0 | 0 / 0 | comments |
| 5568 | Plate rule - a rigid mass-law panel passes the incoherent series of its face and its pl... | 320 | 0.004 | 0.004 | 164 | 6 / 1 | 6 | 2, 1, 0, 0 | 0 / 0 | comments |
| 5616 | Plate rule - a wall bonded of two mass-law bodies passes the same in both directions | 36 | 0.001 | 0.001 | 21 | 4 / 0 | 4 | 1, 1, 0, 0 | 0 / 0 | comments |
| 5659 | Plate rule - a traced mass-law body passes in one path: nothing is reflected back into it | 24 | 0.001 | 0.001 | 15 | 4 / 0 | 4 | 1, 1, 0, 0 | 0 / 0 | comments |
| 5690 | Resonance - on a mass-law material res* is the plate's resonance and leaves the permitt... | 168 | 0.002 | 0.002 | 87 | 5 / 1 | 5 | 1, 2, 0, 0 | 2 / 8 | comments |
| 5731 | Coincidence - on a mass-law material coiF and coiQ are the bending stiffness of the plate | 40 | 0.001 | 0.001 | 43 | 3 / 0 | 3 | 2, 1, 0, 0 | 0 / 0 | comments |
| 5760 | Coincidence column - a table that carries coiA loads and the column changes nothing - t... | 95 | 0.001 | 0.001 | 0 | 5 / 3 | 4 | 1, 0, 0, 0 | 0 / 0 | comments |
| 5813 | Mass law - m switches the limp mass law on | 320 | 0.004 | 0.004 | 195 | 10 / 2 | 10 | 1, 2, 0, 0 | 0 / 0 | comments |
| 5856 | Finite plate - a mass-law panel of side s radiates with Davy's forced radiation efficiency | 1362 | 0.015 | 0.015 | 686 | 14 / 0 | 12 | 0, 1, 0, 0 | 0 / 0 | comments |
| 5914 | Finite plate - side 0 is the infinite plate, a table without the column is side 3.29, a... | 151 | 0.004 | 0.004 | 56 | 11 / 4 | 7 | 0, 2, 0, 0 | 1 / 12 | comments |
| 5997 | Two-parameter mapping - air \| water, a hard wall's phase, a porous layer on it and ever... | 53 | 0.000 | 0.000 | 39 | 10 / 1 | 6 | 1, 2, 0, 0 | 0 / 0 | comments |
| 6105 | Held exit - a lining behind a rigid wall is the same as its traced layer, for every tracer | 48 | 0.006 | 0.006 | 19 | 5 / 0 | 5 | 1, 1, 0, 0 | 0 / 0 | comments |
| 6149 | Held path - alpha on a lumped plate is charged over the path the held wave takes | 20 | 0.001 | 0.001 | 13 | 3 / 0 | 3 | 1, 1, 0, 0 | 0 / 0 | comments |
| 6180 | Held exit - a non-rigid mass-law body holds every tracer and passes at every angle | 8 | 0.000 | 0.000 | 7 | 4 / 1 | 4 | 1, 1, 0, 0 | 0 / 0 | comments |
| 6242 | Absorption peak - tfP, tfF and tfQ add a resonant peak to the room-side absorption of a... | 160 | 0.002 | 0.002 | 84 | 7 / 0 | 7 | 3, 1, 0, 0 | 0 / 0 | comments |
| 6283 | Absorption peak - a peaked face is its series impedance in magnitude and phase from 0 t... | 2460 | 0.024 | 0.024 | 814 | 21 / 1 | 19 | 4, 0, 0, 0 | 0 / 7 | comments |
| 6545 | Far from the origin - a traced mass-law wall passes what it passes at the origin, its l... | 756 | 0.006 | 0.006 | 256 | 7 / 1 | 7 | 0, 1, 1, 0 | 0 / 0 | comments |
| 6600 | Far from the origin - in single precision a mass-law leg is not taken for a bent one | 72 | 0.002 | 0.002 | 75 | 6 / 0 | 6 | 0, 1, 0, 0 | 0 / 0 | comments |

**test_acoustic_leaf.cpp**

| line | case | assert. | wall s | CPU s | printed | comments full / trail | lead | flagged stale (label, md, history, old name) | unique L / B | action |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 199 | Leaf - a wave-medium sheet in the two-parameter mapping is a limp leaf | 350 | 0.001 | 0.001 | 73 | 5 / 0 | 5 | 0, 0, 0, 0 | 0 / 0 | keep |
| 233 | Leaf - a leaf on a box of air is the transfer matrix of leaf and cavity at every angle | 2042 | 0.106 | 0.106 | 465 | 12 / 1 | 8 | 1, 0, 0, 0 | 0 / 0 | comments |
| 328 | Leaf - two leaves on a box of air resonate at the mass-air-mass frequency | 568 | 0.025 | 0.025 | 122 | 7 / 0 | 5 | 1, 0, 0, 0 | 0 / 0 | comments |
| 387 | ray_progress - a leaf on a box of air at every frequency of a band | 840 | 0.021 | 0.021 | 195 | 8 / 0 | 6 | 1, 0, 0, 0 | 0 / 5 | comments |
| 521 | Leaf - a porous fill at the back of the box is the transfer matrix of leaf, air, fill a... | 208 | 0.104 | 0.104 | 107 | 8 / 0 | 8 | 1, 1, 0, 0 | 0 / 0 | comments |
| 555 | Sheet between two bodies - nothing is created and the wave left in air inside the far b... | 458 | 0.004 | 0.004 | 51 | 20 / 5 | 13 | 1, 1, 1, 0 | 0 / 0 | comments |
