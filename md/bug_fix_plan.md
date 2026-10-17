# Bug fix plan for branch `fixes`

- Source: `md/bugs.md` (bug hunt of 2026-10-15 at commit 4720a92, 118 items B-001 to B-118).
- This plan is maintained by the manager session. Workers read it but do not edit it.
- Status of this file: see "Status" at the end; it is updated after every work package.

## 1. Roles and loop

**Manager session** (one, long-lived). Does not change code, tests or wrappers. It:
1. writes the prompt for the next worker from the template in section 6, filled with the work package of section 5
   and the decisions of section 4;
2. after the worker reports, reads the worker log `md/fix_logs/WP-xx.md`, checks the result independently (section 3),
   and tells the user what it found, item by item;
3. after the user's approval and commit, updates `md/bugs.md` (section 3.3) and this plan (status, any changed scope),
   and moves to the next work package.

**Worker session** (one per work package, fresh each time). Fixes the items of exactly one work package, test first,
writes `md/fix_logs/WP-xx.md`, and stops. It does not touch `md/bugs.md` or this plan and does not commit.

**User.** Answers the decisions (WP-00), passes "WP-xx done" from the worker to the manager, reviews the diff and the
manager's check, commits (one commit per work package), and starts the next worker with the prompt from the manager.

Loop per work package:

```
manager: prompt WP-xx  ->  user starts worker  ->  worker: fixes + tests + md/fix_logs/WP-xx.md
  ->  user: "WP-xx done" to manager  ->  manager: checks (section 3), report to user
  ->  user: reviews, commits  ->  manager: updates md/bugs.md and this plan  ->  next WP
```

Only one worker runs at a time: work packages share source files, the Catch2 suite must not run twice in parallel,
and MATLAB/GPU runs are serialised anyway. If a check fails, the manager writes a follow-up prompt for the same work
package (a second worker session, "WP-xx, round 2"), not for the next one.

## 2. Rules for every worker

- Work in `/sjc/quadriga-lib-fixes`, branch `fixes`. Change only what the work package needs: the listed source files,
  their tests, wrappers and documentation blocks. No unrelated reformatting, no drive-by fixes of other items (note
  them in the log instead).
- Test first: for each item add the test named in "Test for the fix session" of `md/bugs.md`, show that it fails
  before the fix (or state why it cannot, e.g. ASan-only), then fix. Catch2 tests in `tests/catch2_tests/`, MEX tests
  in `tests/mex_tests/`, Python tests in `tests/python_tests/`.
- Follow the decisions of section 4. If an item needs a decision that is not there, stop on that item, record the
  question in the log, and continue with the others.
- Builds: rebuild what the work package touches, in the build directories of the Baseline section of `md/bugs.md`
  (`build_linux` with `cmake --install build_linux` for `lib/` and `+quadriga_lib/`, `build_generic`, `build_asan`).
  The repro scripts in `build_linux/scratch/hunt/final/` link against these libraries.
- Machine rules of `~/.claude/CLAUDE.md` apply (load check, thread counts, `flock /tmp/bench.lock` for timings, no
  GPU). Run every `test_bin` through `flock /tmp/quadriga-fixes-suite.lock`.
- Install nothing system-wide; Python via `PYTHONPATH=/sjc/quadriga-lib-fixes/lib`, MATLAB/Octave via `addpath`.
- Do not commit. Do not edit `md/bugs.md`, `md/bug_fix_plan.md`, `md/bug_hunt_brief.md`.
- Documentation: when behaviour changes, update the C++ comment block, the MEX and the Python block in the same work
  package. Do not regenerate `html_docu/` (WP-17 does that once).
- End with `md/fix_logs/WP-xx.md` (format in section 6) and a short final message: items fixed / not fixed, test
  counts, anything unexpected.

## 3. Checks by the manager

### 3.1 After every work package

1. Read `md/fix_logs/WP-xx.md` and `git diff --stat`: only files the work package needs are changed.
2. Per item: the new test exists and asserts what `md/bugs.md` asks for; the repro folder of the item (index at the
   end) is re-run with `bash build_linux/scratch/hunt/final/<folder>/run.sh`, and its output now shows the fixed
   behaviour (correct values, a clean exception, or no sanitizer report). Note: the repro scripts print the *old*
   behaviour as their expectation in some messages ("expected ..."); judge by the values.
3. Full Catch2 suite, AVX2 and generic builds, passes; record assertion and case counts.
4. ASan + UBSan suite (`build_asan`) with `ASAN_OPTIONS=halt_on_error=0:detect_leaks=1:suppress_equal_pcs=0`: no
   report except those of still-open items and of excluded code (`ray_progress`).
5. If MEX or Python files changed: `tests/quadriga_lib_mex_tests.m` in Octave (and MATLAB R2024a when MATLAB-specific
   behaviour changed), `pytest tests/python_tests`.
6. Documentation blocks changed together with behaviour (C++, MEX, Python agree).
7. Report to the user: per item fixed / partly / not fixed, with the evidence; suite numbers; anything the worker
   changed beyond the work package.

### 3.1a Commit messages (rule of 2026-10-17, from WP-04 on)

The files in `md/` are deleted once all bugs are fixed, so commit messages must stand on their own. A commit message
(subject and body) does not refer to any document in `md/` or to its content: no bug IDs (B-xxx), no work-package
numbers (WP-xx), no "decision (a)", no "worker log", no file paths under `md/`. It describes what changed in the code,
why (the defect in plain words), behaviour changes for users, and the test and benchmark results. The commit itself
may still contain the `md/` files. Earlier commits (WP-01 to WP-03) are not rewritten.

### 3.2 Extra checks per package

Named in section 5 under "Acceptance" (fuzz re-runs, MATLAB R2026a exit, timings).

### 3.3 Updating `md/bugs.md`

`md/bugs.md` keeps every item and its reproduction; IDs never change. After each committed work package:

- Add a status line as the first bullet of each item's section:
  `- Status: fixed (WP-xx)` / `partly fixed (WP-xx): <what remains>` /
  `won't fix (<reason>, decided <date>)` / `documented (WP-xx)` / `open`.
  The update of `md/bugs.md` and this plan goes into the work package's own commit, which cannot name its own hash;
  the commit hash is therefore recorded only in the Status table of section 7, filled in at the next manager update
  (the user names the hash, or it is the commit right after the previous package's hash in `git log`).
- Add a "Status" column to the summary table (after "Mark") and keep the counts line of the summary table current
  ("open: n, fixed: n, ...").
- If a fix changed the analysis (cause, severity, scope), correct the item text and say so in one sentence.
- Update the Baseline section only for suite numbers (assertions/cases) at the end of each work package, as a new row
  "after WP-xx", not by overwriting the original rows.
- New defects found by workers become new items B-119, B-120, ... at the end of the document (with severity and the
  usual sections), and are assigned to a work package here.

## 4. Decisions (WP-00, before any worker starts)

The user answers each row (column "Decision"); the manager copies the answers into the worker prompts. "Recommended"
is the coordinator's suggestion from the hunt.

| Item | WP | Question | Options | Recommended | Decision |
|---|---|---|---|---|---|
| B-001 | 02 | When `hdf5_write` rewrites a slot, are old `par_*` fields of that slot kept? | (a) slot is replaced completely (b) existing `par_*` fields are kept unless rewritten | (a) | (a) slot replaced completely (2026-10-15) |
| B-002 | 03 | Small negative delays in `quantize_delays` (normal output of `get_channels_spherical` with relative delays) | (a) clamp delays above `-tap_spacing/2` to tap 0, throw below (b) throw for any negative delay | (a), and document in `get_channels_spherical` that relative per-element delays can be slightly negative | (a) clamp above -tap_spacing/2 to tap 0, throw below; document negative relative delays in get_channels_spherical (2026-10-15) |
| B-007 | 09 | NaN / Inf angles in `arrayant::interpolate` | (a) NaN output, no exception (b) throw `invalid_argument` | (a) | (a) NaN output, no exception (2026-10-15) |
| B-012 | 10 | qdant writer magnitude floor (-100 dB absolute) | (a) write all finite dB values, -Inf only for exact zeros (b) floor relative to the element peak | (a) | (a) write all finite dB values, -Inf only for exact zeros (2026-10-15) |
| B-019 | 04 | `baseband_freq_response_multi` with relative delays | (a) document that absolute delays are required (b) add an argument for the LOS delay (c) change the phase convention of `get_channels_*` | (a) | (a) document that use_absolute_delays = true is required (2026-10-15) |
| B-023 | 11 | IEEE indoor uplink: conjugate (current) or reciprocal transpose | (a) transpose only (reciprocity, like `get_channels_spherical`) (b) keep conjugation, document it | (a) | (a) transpose only (reciprocity); fix code, comment at line 710 and the test "IEEE Chan - Reciprocity"; list as a behaviour change (2026-10-15) |
| B-041 | 10 | `qdant_write` without a layout replaces a custom layout | (a) keep the existing layout node (b) keep current behaviour and document it | (b) | (b) keep current behaviour, document it (2026-10-15) |
| B-045 | 10 | `combine_pattern` for acoustic arrays | (a) add an optional `propagation_speed` argument (default 299792458) in C++, MEX, Python (b) document the limitation | (a) | (a) optional propagation_speed argument, default 299792458, in C++, MEX, Python (2026-10-15) |
| B-047 | 10 | `generate_arrayant_multibeam` default M, N | (a) header to 1, 1 (as docs and wrappers) (b) docs and wrappers to 2, 2 | (a) | (a) header defaults to 1, 1 (2026-10-15) |
| B-052 | 02 | `hdf5_write` rewriting a slot grows the file | (a) delete the old group link, document that space is not reclaimed (b) document only | (a) | (a) delete the old group link, document that space is reclaimed only by h5repack (2026-10-15) |
| B-060 | 17 | `calc_effective_path_gain`: comment says "average", code sums | (a) fix the comment (b) divide by `n_rx * n_tx` | (a) | (a) fix the comment to 'sum' (2026-10-15) |
| B-069 | 04 | Naming of the cross-polarisation rows of `M` | (a) one convention written out in words ("rows 2-3: response of the H port to a V input") in all docs, XPR columns checked against it (b) other | (a) | Global convention: M is the 2x2 Jones matrix (rows = RX polarisation V/H, columns = TX polarisation V/H) stored column-major as in MATLAB, i.e. rows 0-1 = (V rx, V tx), rows 2-3 = (H rx, V tx), rows 4-5 = (V rx, H tx), rows 6-7 = (H rx, H tx), Re/Im interleaved. This matches the measured code behaviour; fix every doc and label that says otherwise, and check the XPR columns of calc_cross_polarization_ratio against it (2026-10-15) |
| B-070 | 13 | `calc_diffraction_gain` C++ frequency in scalar mode | (a) document the EM-equivalent frequency (b) convert inside C++ and drop the wrapper conversion | (a) | (b) C++ takes the acoustic frequency and converts internally; drop the conversion in the MEX and Python wrappers; API change for C++ callers (2026-10-15) |
| B-074 | 08 | Invalid `use_kernel` values | (a) throw for values outside the documented set, and for 2 without AVX2 (as `sos_val`) (b) document "falls back to auto/generic" | (a) | (a) throw for values outside the documented set, and for 2 without AVX2 (2026-10-15) |
| B-077 | 12 | `obj_overlap_test` with an empty mesh | (a) return an empty result (b) throw | (a) | (a) return an empty result (2026-10-15) |
| B-082 | 12 | `cube` / `plane` with a negative scale | (a) swap the winding so normals stay outward / +z (b) reject negative scales | (a) | Mirror Blender: when a negative scale is applied to the mesh, Blender swaps the winding so normals stay outward / +z (checked with Blender 5.2.2, Object > Apply > Scale: cube scale (-1,1,1) and (1,1,-2) give 6 of 6 outward faces, plane scale (-1,1,1) gives normal z = +4). So: (a) swap the winding when the scale product is negative (2026-10-15) |
| B-087 | 15 | Multibeam element beamwidth default in MATLAB / Python | (a) wrappers pass 120 (docs and C++ default) (b) docs to 90 | (a) | (a) wrappers pass 120 deg (docs and C++ default) (2026-10-15) |
| B-088 | 15 | `calc_angular_spread` defaults (wrapping, bank angle) | (a) wrappers follow C++ (on, on) (b) C++ follows the wrappers (off, off) | user's call: which one existing scripts rely on | (b) C++ follows the wrappers: wrapping off, bank angle off by default; behaviour change for C++ callers (2026-10-15) |
| B-091 | 15 | MEX `get_channels_irs` `i_irs` base | (a) 1-based per `md/SPEC_MATLAB_API.md` (breaking) (b) 0-based, documented | (a) | (a) 1-based per md/SPEC_MATLAB_API.md: default 1, reject 0; breaking for MATLAB callers (2026-10-15) |
| B-092 | 15 | MEX `arrayant_qdant_read(fn)` without `id` | (a) keep reading all entries, validate that they form a frequency set, document the difference to Python (b) read ID 1 like Python | (a) | (a) keep reading all entries, validate that they form a frequency set, document the difference to Python (2026-10-15) |
| B-095 | 15 | Diffraction level of detail 7 | (a) public: document 1-7 everywhere, wrappers take the `coord` size from C++ (b) not public: reject 7 in C++ | (a) | (a) public: document 1-7 everywhere, wrappers take the coord size from C++ (2026-10-15) |
| B-098 | 05 | Python `view = true` | (a) document as "fast contiguous copy" now (b) real aliasing (audit, GIL notes in the item) | (a) now, (b) as a later separate task | (b) real aliasing now, in WP-05: switch to copy_aux_mem = false with the audit of all wrappers and the GIL / lifetime notes of the item (2026-10-15) |
| B-099 | 05 | Python kwarg names `sub_cloud_index`, `map_to_itu_materials` | (a) docs follow the registered names (b) registration follows the docs (breaking) | (a) | (a) docs follow the registered names (2026-10-15) |
| B-113 | 15 | 0 paths with `add_fake_los_path` | (a) C++ returns the fake LOS path (all APIs agree with today's MEX) (b) MEX calls C++ and throws like C++ | (a) | (a) C++ returns the zero-power fake LOS path (with the correct delay) for 0 paths; all APIs agree (2026-10-15) |
| WP-02 | 02 | Reads are 24-35% slower after the WP-02 rewrite (manager's benchmark; writes 35-38% faster); most of it is the per-dataset storage check | (a) accept and commit (b) round 2: make the check cheap, keep every check | (b) | (b) round 2; accept when both reads are within 10% of the old code (2026-10-16); result: W2 1.03-1.04, W4 1.01-1.03, writes 0.55-0.97 of the old time (manager, load 2.1-2.9) |
| B-050 | 02 | Empty `tx_orientation` / `rx_orientation` read back as 3x1 zeros | (a) keep and document (b) marker attribute, read back empty (changes MATLAB/Python output) | (a) | (a) keep the current behaviour, documented; B-050 counts as fixed (2026-10-16) |
| B-119 | - | HDF5 1.14.6 itself misbehaves on crafted files (`H5Fopen` 4 GB zero-fill, `H5Gopen2` 2^64-byte request) | (a) record as an item, won't fix (b) mitigate | (a) | (a) recorded as B-119, won't fix (upstream) (2026-10-16) |
| B-002 | 03 | Relative delays of acoustic arrays and of radio arrays of about 1.5 m and more go below -tap_spacing/2 and throw (manager's check of WP-03: a 5 cm microphone pair at 48 kHz already throws) | (a) keep: tapped delay line emulators need non-negative delays; document the limit and point to absolute delays (b) clamp all negative delays (c) allow negative taps | (c) | (a) keep the decision; document the limit in quantize_delays and get_channels_*, and the error message points to use_absolute_delays (2026-10-17) |
| B-120 | 13 | Circular XPR columns of `calc_cross_polarization_ratio` inverted for the LOS matrix of `get_channels_*` (found in WP-04) | (a) record as an item, fix in WP-13 (b) leave | (a) | (a) recorded as B-120, assigned to WP-13 (2026-10-17) |
| B-121 | 12 | AVX2-vs-generic ray kernel tests never run (`test_bin` without `BUILD_WITH_AVX2`, found in WP-04) | (a) record as an item, fix first in WP-12 (b) leave | (a) | (a) recorded as B-121, assigned to WP-12 (2026-10-17) |
| WP-08 | 08 | One shared sin/cos core for the baseband DFT kernels and `fastmath_avx2.h` (proposed by the WP-04 worker) | (a) add to WP-08 as a note (b) leave | (a) | (a) added to WP-08 as a note; the WP-08 worker decides after measuring (2026-10-17) |

## 5. Work packages

Order is the execution order. Each package lists its items (IDs of `md/bugs.md`), the main files, notes, and the
acceptance checks beyond section 3.1. Severities in brackets.

### WP-01 Quick wins (one-liners)

- Items: B-016 [S2] (misplaced brace in `qrt_file_read`, plus the `memcpy` guard for 0 paths), B-029 [S2]
  (`-Wl,-z,nodelete` on the Linux MATLAB MEX link line, and the comment at `CMakeLists.txt:196`), B-031 [S2]
  (`copy = true` for `i_cir` / `snap`), B-039 [S3] (validate before removing the file), B-043 [S3]
  (`check_ptr[9] = {}`), B-063 [S3] (13 → 12 in the IEEE test), B-080 [S3] (`delete[]` / `std::vector`), B-085 [S3]
  (`fast_sincos` default `false`, "single" in the doc), B-086 [S3] (`center_freq` default), B-105 [S4] (`memcpy` with
  size 0 guards), B-112 [S4] (`uint8` in the doc).
- Files: `src/qd_channel_qrt.cpp`, `CMakeLists.txt`, `api_mex/qrt_file_read.cpp`, `api_mex/hdf5_read_channel.cpp`,
  `src/qd_arrayant_xio.cpp`, `include/quadriga_arrayant.hpp`, `tests/catch2_tests/test_qd_ieee_chan.cpp`,
  `src/point_cloud_split.cpp`, `api_mex/fast_sincos.cpp`, `api_mex/arrayant_qdant_write.cpp`,
  `src/get_channels_irs.cpp`, `api_python/python_arma_adapter.hpp`.
- Acceptance: R2026a `-batch` with one `hdf5_create_file` call exits 0 and writes no crash dump (B-029); the ASan suite
  no longer reports `test_qd_ieee_chan.cpp:318`.

### WP-02 HDF5 reader and writer

- Items: B-001 [S1], B-013 [S2], B-014 [S2], B-015 [S2], B-050 [S3], B-051 [S3], B-052 [S3], B-053 [S3].
- File: `src/qd_channel_hdf5.cpp` (plus documentation in `include/quadriga_channel.hpp`).
- Notes: B-001 and B-051 interact (error printing switched off hides the failure); introduce a checked read helper
  (rank, extents, return codes) once and use it everywhere (B-013); decisions B-001, B-052.
- Notes after WP-01: decision B-001 (a) contradicts the test proposed in `md/bugs.md` B-001 and the "expected a=1 b=2"
  of its repro: after `hdf5_write` rewrites a slot, only the new channel's `par_*` fields are there. B-001 and B-052
  together mean: rewriting a slot deletes the old group link and writes a fresh group. B-015 is the last ASan report of
  the Catch2 suite besides the excluded `ray_progress`; after WP-02 the ASan suite must show only `ray_progress`.
  B-051 removes `H5close()`: re-check the R2026a exit (B-029, repro `F-1`). The 11 crash and 20 hang inputs of the
  phase-0 HDF5 campaign are in `build_linux/scratch/hunt/fuzz/out/hdf5/{main,cmp}/{crashes,hangs}/`; replay them
  against the fixed harness, and fuzz into a new output directory (do not overwrite `out/hdf5`).
- HDF5 timing for later work: `bash build_linux/scratch/fix/WP-02/manager/bench/run_bench.sh` compares the current
  `src/qd_channel_hdf5.cpp` with 3a7c2cc (7 workloads, single thread, under `flock /tmp/bench.lock`; valid at load < 4).
  After WP-02: reads 1.01-1.04, writes 0.55-0.97 of the 3a7c2cc time.
- Acceptance: re-run the HDF5 AFL harness for 15 minutes (`build_linux/scratch/hunt/fuzz/`, rebuild `build_afl_plain`
  and `build_afl_cmplog` first): no crash. The HDF5 cases of the ASan suite are clean.

### WP-03 `quantize_delays`

- Items: B-002 [S1], B-054 [S3], B-055 [S3], B-103 [S4].
- Files: `src/qd_channel_quantize_delays.cpp`; doc sentence on relative delays in `src/get_channels_spherical.cpp`
  (and MEX/Python blocks).
- Notes: decision B-002. B-054 (dense vectors over the absolute delay range) and the validation of B-002 touch the
  same code; do them together.
- Acceptance: the `repro_normal.cpp` case of B-002 runs; Python call returns; the B-054 timing under
  `flock /tmp/bench.lock` with load below 4 is within 10× of the unshifted case.

### WP-04 Channel generation: delays and phases

- Items: B-003 [S1], B-019 [S2], B-020 [S2], B-068 [S3], B-069 [S3].
- Files: `src/get_channels_spherical.cpp`, `src/get_channels_multifreq.cpp`, `src/get_channels_irs.cpp`,
  `src/baseband_freq_response.cpp`, `src/baseband_freq_response_avx2.cpp`, `src/quadriga_lib_generic_functions.hpp`,
  documentation of `get_channels_*` and `calc_cross_polarization_ratio` (C++, MEX, Python).
- Notes after WP-03: run the full checks (generic, ASan, MATLAB); WP-03 round 2 ran only targeted tests. WP-03
  changed only documentation in the `get_channels_*` files (delay notes in C++, MEX, Python); keep those notes true
  when B-003 changes the delays. Add the "Performance" section to the prompt (section 5, "Performance check in worker
  prompts").
- Notes: B-003 changes results of every NLOS path whose `path_length` exceeds the polyline; check the existing tests
  that pin delays and the MEX/Python tests. B-020 (phase in double) also removes the baseband route into B-026.
  Decisions B-019, B-069.
- Acceptance: AVX2 and generic baseband results agree to 1e-6 for delays up to 10 s. Restated after WP-04: the error
  must not grow with the delay and stay at single-precision level; measured 5-8e-7 (10 paths) and 1.2-2.9e-6 (100
  paths, sum of 100 float terms, |H| about 4) at every delay from 1 us to 10 s, AVX2 vs generic and vs an exact
  reference.

### WP-05 Python bindings

- Items: B-004 [S1], B-005 [S1], B-035 [S2], B-036 [S2], B-096 [S3], B-097 [S3], B-098 [S3], B-099 [S3], B-100 [S3],
  B-116 [S4], B-117 [S4], B-118 [S4].
- Files: `api_python/qpy_geo2cart.cpp`, `api_python/qpy_cart2geo.cpp`, `api_python/python_arma_adapter.hpp`,
  `api_python/python_quadriga_adapter.hpp`, `api_python/qpy_arrayant_interpolate.cpp`,
  `api_python/qpy_baseband_freq_response.cpp`, `api_python/qpy_qrt_file_read.cpp`, `api_python/qpy_hdf5_*.cpp`,
  documentation blocks of the named wrappers.
- Notes: decisions B-098, B-099. B-098 was decided as real aliasing (`copy_aux_mem = false`) now: audit every wrapper
  for writes into `view = true` inputs (area G found none), keep list entries referenced while the GIL is released in
  `get_channels_spherical` batch mode, handle read-only arrays; this is the largest item of the package. Rebuild into
  `lib/` (the build of the Baseline section).
- Acceptance: an ASan build of the Python module (as area G built it: `build_linux/scratch/hunt/G/asan_mod`) runs the
  Python suite without reports. Build it with `bash build_linux/scratch/fix/build_asan_mod.sh [jobs]` (written in
  WP-01, about 50 s; needs an up-to-date `build_asan`). Positive control: before the B-004 fix, repro `G-1` reports a
  heap-use-after-free with this module.
- Notes after WP-02: `tests/python_tests` has no HDF5 tests at all; add round-trip tests for the Python HDF5 writers and
  readers in this package. Since WP-02, a 1-element vector read with `hdf5_read_dset` comes back as a numpy array of
  shape `(1,)`, not a Python scalar (also from old files). Open question from WP-02: a `(1, 1)` numpy array written with
  `hdf5_write_dset` reads back with shape `(1,)`; find out whether the Python writer flattens it.

### WP-06 MEX helpers

- Items: B-030 [S2], B-032 [S2], B-033 [S2], B-034 [S2], B-093 [S3], B-094 [S3].
- Files: `src/mex_helper_functions.hpp`; `api_mex/channel_export_obj_file.cpp`, `api_mex/point_inside_mesh.cpp`,
  `api_mex/obj_file_write.cpp`, `api_mex/arrayant_interpolate.cpp`; the six `qd_mex_sos_seed` copies in
  `api_mex/sos_*.cpp` (B-034).
- Notes: one central change in the helpers fixes almost every wrapper; the full MEX suite must pass in Octave, MATLAB
  R2024a and R2026a.
- Acceptance: the area-F malformed-input sweep (`build_linux/scratch/hunt/F/probes/sweep/`) re-run in Octave: no crash;
  B-033 checked in R2026a.

### WP-07 QRT files

- Items: B-017 [S2], B-048 [S3], B-049 [S3], B-059 [S3].
- Files: `src/qd_channel_qrt.cpp`, `src/qd_channel.cpp` (B-059 `add_paths`).
- Acceptance: re-run the QRT AFL harness for 15 minutes: no hang input that runs longer than 2 s outside AFL.

### WP-08 AVX2 fast math

- Items: B-026 [S2], B-037 [S3], B-074 [S3], B-075 [S3], B-076 [S3].
- Files: `src/fastmath_avx2.h`, `src/fastmath_vectorized_avx2.cpp`, `src/fastmath_armadillo.cpp`,
  `src/ray_triangle_intersect.cpp`, `src/ray_point_intersect.cpp` (B-074).
- Notes after WP-04 (decision 2026-10-17): the baseband DFT kernels have their own sin/cos of 2*pi*r for reduced
  phases (`qd_sincos_2pi_reduced` in `src/quadriga_lib_generic_functions.hpp`, an AVX2 helper in
  `src/baseband_freq_response_avx2.cpp`). Consider one shared core with `fastmath_avx2.h`, measured with
  `build_linux/scratch/fix/WP-04/bench/run_bench.sh`; details in `md/fix_logs/WP-04.md` ("Proposed for WP-08"). Both
  kernels round with the magic-number trick (exact only without `-ffast-math`); add a comment that says so.
- Notes: decision B-074. The sin/cos change touches every AVX2 user (antenna interpolation, `qd_coeff_combine_avx2`,
  baseband); compare AVX2 against generic before and after over the ranges of the hunt (area D probes in
  `build_linux/scratch/hunt/D/`).
- Acceptance: AVX2 and generic `fast_sincos` agree to 1e-6 on [-1e10, 1e10]; antenna interpolation AVX2 vs generic
  unchanged within 1e-5 on the area-A sweep (`build_linux/scratch/hunt/A/p1_kernel_cmp.cpp`); timings of the
  interpolation kernel under `flock /tmp/bench.lock` at low load before and after (report, no hard limit).

### WP-09 Array antennas: interpolation and members

- Items: B-006 [S2], B-007 [S2], B-008 [S2], B-009 [S2], B-010 [S2], B-011 [S2], B-044 [S3], B-046 [S3], B-101 [S4].
- Files: `src/qd_arrayant_interpolate_avx2.cpp`, `src/qd_arrayant_interpolate.cpp`, `src/qd_arrayant.cpp`,
  `src/qd_arrayant_generate.cpp`.
- Notes: decision B-007. B-008 can reuse the coupling assembly of `arrayant_concat_multi`.

### WP-10 qdant files and arrayant generators

- Items: B-012 [S2], B-038 [S3], B-040 [S3], B-041 [S3], B-042 [S3], B-045 [S3], B-047 [S3].
- Files: `src/qd_arrayant_qdant.cpp`, `src/qd_arrayant_xio.cpp`, `src/qd_arrayant.cpp`, `include/quadriga_arrayant.hpp`;
  MEX and Python wrappers if B-045 adds an argument.
- Notes: decisions B-012, B-041, B-045, B-047. B-012 and B-038 change the file content (more digits, no floor); old
  files stay readable.
- Acceptance: re-run the qdant AFL harness for 15 minutes; qdant round trip of the area-A probe
  `build_linux/scratch/hunt/A/p5_qdant_roundtrip.cpp` exact to 1e-12 relative.

### WP-11 IEEE indoor model

- Items: B-021 [S2], B-022 [S2], B-023 [S2], B-064 [S3], B-065 [S3] (the `observation_time` part; the `quantize`
  part of B-065 goes with WP-13 if simpler, note it in the log).
- Files: `src/get_channels_ieee_indoor.cpp`, `src/ieee_channel_model_functions.hpp`,
  `tests/catch2_tests/test_qd_ieee_chan.cpp` (B-023 changes the pinned behaviour).
- Notes: decision B-023. WP-01 (B-063) changed `tests/catch2_tests/test_qd_ieee_chan.cpp` lines 301 and 318: `d` is
  now a copy (`arma::mat(ptr, 16, 13)` / `(ptr, 16, 12)`), no longer a strict alias of `chan[0].delay[0]`; start from
  these lines.

### WP-12 Geometry

- Items: B-027 [S2], B-028 [S2], B-071 [S3], B-072 [S3], B-073 [S3], B-077 [S3], B-078 [S3], B-079 [S3], B-081 [S3],
  B-082 [S3], B-084 [S3], B-121 [S3] (added after WP-04).
- Notes after WP-06: the MEX side of `icosphere(-1)` / `icosphere(NaN)` is fixed, but the C++ `icosphere` still writes
  out of bounds for very large `n_div` (B-078; the C++ part of repro `F-6` still segfaults).
- Notes after WP-04: fix B-121 first (define `BUILD_WITH_AVX2` for `test_bin` when the library has AVX2), so that the
  AVX2-vs-generic kernel tests run before the kernels change; they may expose B-071.
- Files: `src/ray_triangle_intersect*.cpp`, `src/ray_point_intersect*.cpp`, `src/point_inside_mesh.cpp`,
  `src/obj_overlap_test.cpp`, `src/subdivide_triangles.cpp`, `src/icosphere.cpp`, `src/cube.cpp`, `src/plane.cpp`.
- Notes: decisions B-077, B-082. B-027 (shift to a local origin before the float kernel) changes AVX2 results slightly
  for every caller; run the area-D comparison probes (`build_linux/scratch/hunt/D/rti_random.cpp`, `rpi_random.cpp`).
  The CUDA ray kernels have the same slab-test logic (B-072); CUDA is out of scope, note it for later.

### WP-13 Channel tools

- Items: B-024 [S2], B-025 [S2], B-061 [S3], B-062 [S3], B-066 [S3], B-067 [S3], B-070 [S3], B-083 [S3], B-107 [S4],
  B-120 [S3] (added after WP-04: circular XPR columns, `src/qd_tools_calc_xpr.cpp`; decide the handedness convention
  with the user in the prompt if the fix needs it). B-062 also applies to the `freq` array that WP-04 added to
  `baseband_freq_response`.
- Files: `src/update_xpr.cpp`, `src/path_to_tube.cpp`, `src/combine_irs_coord.cpp`, `src/baseband_freq_response.cpp`
  (B-062), `src/qd_tools_calc_delay_spread.cpp`, `src/qd_tools_acdf.cpp`, `src/calc_diffraction_gain.cpp` (doc),
  `src/interp.cpp`.
- Notes: decision B-070 (convert the acoustic frequency inside C++; remove the conversion from
  `api_mex/calc_diffraction_gain.cpp` and `api_python/qpy_calc_diffraction_gain.cpp`; API change for C++ callers).
  B-083 timing under `flock /tmp/bench.lock` at low load.

### WP-14 OBJ, MTL, PNG and Mitsuba writers/readers

- Items: B-018 [S2], B-056 [S3], B-057 [S3], B-058 [S3].
- Files: `src/obj_file_write.cpp`, `src/obj_file_read.cpp`, `src/mitsuba_xml_file_write.cpp`, `src/write_png.cpp`.
- Acceptance: re-run the OBJ, MTL and CSV AFL harnesses for 15 minutes each.

### WP-15 API consistency of the wrappers

- Items: B-087 [S3], B-088 [S3], B-089 [S3], B-090 [S3], B-091 [S3], B-092 [S3], B-095 [S3], B-113 [S4], B-114 [S4].
- Files: `api_mex/arrayant_generate.cpp`, `api_python/qpy_arrayant_generate.cpp`, `api_mex/calc_angular_spread.cpp`,
  `api_python/qpy_calc_angular_spread.cpp` (or the C++ defaults), `api_mex/triangle_mesh_segmentation.cpp`,
  `api_mex/subdivide_triangles.cpp`, `api_mex/baseband_freq_response.cpp`, `api_mex/get_channels_irs.cpp`,
  `api_mex/arrayant_qdant_read.cpp`, `api_mex/calc_diffraction_gain.cpp`, `api_mex/generate_diffraction_paths.cpp`,
  `api_python/qpy_calc_diffraction_gain.cpp`, `src/generate_diffraction_paths.cpp`, `api_mex/get_channels_*.cpp`,
  `src/get_channels_*.cpp` (B-113), `api_mex/arrayant_rotate_pattern.cpp`.
- Notes after WP-06: MEX `subdivide_triangles` and `triangle_mesh_segmentation` subtract 1 from `mtl_ind_in`, which is
  1-based in C++ with 0 = "no material", so an input 0 wraps around (related to B-089).
- Notes after WP-05: Python `hdf5_write_channel` and `hdf5_read_layout` still return uint32, while `hdf5_create_file`
  returns int64 since WP-05 (B-116); align them with the integer-output rule of the Python spec.
- Notes: decisions B-087, B-088, B-091, B-092, B-095, B-113. B-088 changes the C++ defaults of
  `calc_angular_spread` (`include/quadriga_tools.hpp`, `src/qd_tools_calc_angular_spread.cpp`) to wrapping off and
  bank angle off; the wrappers stay. Several items are breaking changes (B-087 for MATLAB/Python, B-088 for C++,
  B-091 for MATLAB); list them in the log for the release notes.

### WP-16 Sum of sinusoids, small items

- Items: B-109 [S4], B-110 [S4], B-111 [S4].
- Files: `src/sos_val.cpp`, `src/sos_remove_bias.cpp`, `src/sos_rand.cpp`, `md/SPEC_PYTHON_API.md`.
- Acceptance: the SOS output hashes of the second SOS review stay unchanged (valid calls only change in their error
  messages).

### WP-17 Documentation sweep and final verification

- Items: B-060 [S3], B-102 [S4], B-104 [S4], B-106 [S4], B-108 [S4], B-115 [S4], and any documentation parts that
  earlier packages left open (the manager lists them in the prompt).
- Left open by earlier packages: `CMakeLists.txt:498` (MATLAB API section) still says MATLAB links quadriga_static "to
  avoid conflicts with MATLAB's own bundled libhdf5.so"; the static link does not separate the HDF5 libraries, the
  different soname of the system HDF5 does (see the comment at `CMakeLists.txt:196`, rewritten in WP-01).
- From WP-06: `include/quadriga_tools.hpp` calls `obj_ind` of `obj_file_write` "1-based"; the C++ code and its
  documentation block use 0-based indices.
- From WP-03: `src/qd_channel_quantize_delays.cpp`: a NaN `power_exponent` passes the `<= 0` check and gives NaN
  coefficients (reject it); the short parameter comments of `quantize_delays` in `include/quadriga_channel.hpp`
  (around line 368) do not mention the valid delay range or the shape rule.
- From WP-02: `include/quadriga_channel.hpp` (`hdf5_write` block) says a new file is created with `nx = 65535`; the
  default is 65536. `src/qd_channel_hdf5.cpp`: the message "Error creating file. It may be already opened already.".
- Files: header comments in `include/*.hpp`, documentation blocks in `src/`, `api_mex/`, `api_python/`; dead code
  (`src/avx_mathfun.h`, `arma::mt19937_64_instance`); `src/runtime_AVX2_Check_avx2.cpp` (FMA / XGETBV check).
- Then: `make documentation` once; full verification of the Baseline section (all suites, both sanitizer builds,
  MEX in Octave, MATLAB R2024a and R2026a, Python, 15 minutes of AFL per reader) recorded as the final row in
  `md/bugs.md`.

### Performance check in worker prompts

From WP-03 on, the manager adds a "Performance" section (as in the WP-03 prompt) to every package that changes
computation code: WP-04, WP-08, WP-09, WP-12, WP-13. The worker builds an old-vs-new benchmark of the changed source
file before the first change (same flags, same library, alternating runs under `flock /tmp/bench.lock`, output
checksums), measures after each design step, and meets the limits (typically 1.05 of old) before it reports. Packages
that change only I/O, wrappers or documentation get no such section.

## 6. Templates

### 6.1 Worker prompt (the manager fills the brackets)

```
You are a worker session for the bug fixes on branch `fixes` in /sjc/quadriga-lib-fixes.
Read md/bug_fix_plan.md sections 2 (rules) and 5 (your work package) in full, and the sections of md/bugs.md for
your items. Then fix work package [WP-xx]: [items].

Decisions that apply (from md/bug_fix_plan.md section 4): [copied rows, with the user's answers].
Notes from the manager: [anything learned in earlier packages that affects this one].

Work test first (failing test, then fix). Rebuild what you touch (Baseline section of md/bugs.md). Run the full
Catch2 suite (AVX2 and generic), the ASan suite with suppress_equal_pcs=0, and the MEX/Python suites if you touched
wrappers; all through flock /tmp/quadriga-fixes-suite.lock. Follow ~/.claude/CLAUDE.md. Do not commit; do not edit
md/bugs.md or md/bug_fix_plan.md. Finish with md/fix_logs/[WP-xx].md in the format of section 6.2 of the plan.
```

### 6.2 Worker log `md/fix_logs/WP-xx.md`

```
# WP-xx log

- Date, commit the work started from, load during the suite runs.

| Item | Result (fixed / partly / not fixed / needs decision) | Test added (file, case name) | Failed before the fix? | Files changed |
|---|---|---|---|---|

## Per item
Short: what was changed and why that closes the item; anything that differs from the proposed fix in md/bugs.md.

## Suites
Catch2 AVX2 / generic (assertions, cases), ASan suite (reports), MEX (Octave, MATLAB), Python, other checks of
section 5 for this package.

## Behaviour changes for users
Changed defaults, shapes, errors, file content (for the release notes).

## Found on the way
Defects outside the package (not fixed here), with file:line.
```

### 6.3 Manager report to the user (after each package)

Per item: fixed / partly / not fixed, with the evidence (test name, repro re-run output line); suite numbers;
unexpected changes in the diff; proposed `md/bugs.md` updates; anything the next package needs to know.

## 7. Status

| WP | Items | Status | Commit |
|---|---|---|---|
| WP-00 | decisions (section 4) | done 2026-10-15 | - |
| WP-01 | 11 | done 2026-10-16 (11 of 11 fixed) | 3a7c2cc |
| WP-02 | 8 | done 2026-10-16 (8 of 8 fixed; round 2: reads back to the old speed) | e8f4edc (round 1), 94a8f33 (round 2) |
| WP-03 | 4 | done 2026-10-17 (4 of 4 fixed; round 2: limit documented, message points to absolute delays) | 9e5af26 |
| WP-04 | 5 | done 2026-10-17 (B-003, B-020 fixed; B-019, B-068, B-069 documented; B-120, B-121 found) | 4e2cf89 |
| WP-05 | 12 | done 2026-10-17 (10 fixed, B-099 and B-117 documented) | bf76f2f |
| WP-06 | 6 | done 2026-10-17 (6 of 6 fixed; integer conversion of large index inputs 14-17% slower, accepted pending the user's review) | commit after bf76f2f (hash at next update) |
| WP-07 | 4 | open | - |
| WP-08 | 5 | open | - |
| WP-09 | 9 | open | - |
| WP-10 | 7 | open | - |
| WP-11 | 5 | open | - |
| WP-12 | 12 | open | - |
| WP-13 | 10 | open | - |
| WP-14 | 4 | open | - |
| WP-15 | 9 | open | - |
| WP-16 | 3 | open | - |
| WP-17 | 6 | open | - |

## 8. Item index

Repro folders are under `build_linux/scratch/hunt/final/`; run `bash build_linux/scratch/hunt/final/<folder>/run.sh`.
"-" means the item has no reproduction ([C] items, or it refers to other items).

| Item | Sev | WP | Repro folder(s) | Title |
|---|---|---|---|---|
| B-001 | S1 | WP-02 | `B-003` | `hdf5_write` silently drops the `par_*` data when the slot already holds a `Data` group |
| B-002 | S1 | WP-03 | `B-006` | `quantize_delays` segfaults on a negative delay, including the slightly negative relative delays that `get_channels_spherical` returns |
| B-003 | S1 | WP-04 | `C-1` | `get_channels_spherical`, `_multifreq` and `_irs` ignore `path_length` for NLOS paths: delays and phases too short |
| B-004 | S1 | WP-05 | `G-1` | Python `tools.geo2cart` / `tools.cart2geo` with separate inputs read freed memory: wrong values, segfault |
| B-005 | S1 | WP-05 | `G-3` | Python HDF5 writers ignore numpy strides for 1-D and (n,1) arrays: wrong data written, heap over-read |
| B-006 | S2 | WP-09 | `A-01` | AVX2 antenna interpolation reads up to 56 bytes past the angle matrices (per-element angles, n_out not a multiple of 8) |
| B-007 | S2 | WP-09 | `A-02` | Generic antenna interpolation reads before the pattern buffer for NaN or infinite angles |
| B-008 | S2 | WP-09 | `A-04` | `arrayant::append` builds a wrong coupling matrix, and segfaults when `coupling_im` is empty |
| B-009 | S2 | WP-09 | `A-06` | `arrayant::rotate_pattern` loses coupling, centre frequency and name |
| B-010 | S2 | WP-09 | `A-07` | `arrayant::calc_beamwidth_deg` is wrong when the main lobe wraps ±180° or points towards a pole |
| B-011 | S2 | WP-09 | `A-08` | `generate_arrayant_multibeam` with a beam towards el = ±90° returns wrong ports (pol 1) or segfaults (pol 2, 3) |
| B-012 | S2 | WP-10 | `A-09` | qdant writer drops every pattern sample below -100 dB absolute power; a whole low-level element reads back as zeros |
| B-013 | S2 | WP-02 | `B-001` | `hdf5_read_channel`: a malformed HDF5 file overflows heap and stack buffers (sizes taken from attributes, not from the datasets) |
| B-014 | S2 | WP-02 | `B-002` | HDF5 storage layout: `nx*ny*nz*nw` wraps in 32 bits; `hdf5_write` then reads and writes past the index buffer |
| B-015 | S2 | WP-02 | `B-004` | `hdf5_write` copies 256 bytes from the channel name: heap memory ends up in the file; names of 256+ characters over-read on reading |
| B-016 | S2 | WP-01 | `B-005` | `qrt_file_read` (C++) fills `aod`, `eod`, `aoa`, `eoa` only when `path_gain` or `path_length` is also requested |
| B-017 | S2 | WP-07 | `B-007` | `qrt_file_append` corrupts the file when a CIR orientation is non-zero in double but rounds to 0 in float |
| B-018 | S2 | WP-14 | `B-008` | OBJ round trip loses "no material": faces with `mtl_ind = 0` inherit the previous material |
| B-019 | S2 | WP-04 | `C-2` | `baseband_freq_response_multi` gives wrong phases for `get_channels_*` output with the default relative delays |
| B-020 | S2 | WP-04 | `C-3` | `baseband_freq_response`: phase computed in single precision, error grows with delay × bandwidth (both builds) |
| B-021 | S2 | WP-11 | `C-5` | IEEE indoor model: `n_floors > 0` shifts all tap delays (3-D path length minus 2-D distance) |
| B-022 | S2 | WP-11 | `C-6` | IEEE indoor model prunes taps with an absolute power threshold after path loss (taps vanish with distance, floors, walls) |
| B-023 | S2 | WP-11 | `C-7` | IEEE indoor model uplink conjugates the channel (Doppler sign reversed), unlike the reciprocity of `get_channels_spherical` (user decision) |
| B-024 | S2 | WP-13 | `C-8` | `xpr_update`: data race when `ray_index` contains duplicates (wrong, non-deterministic result with threads) |
| B-025 | S2 | WP-13 | `C-9` | `path_to_tube`: NaN vertices when interior path points are collinear |
| B-026 | S2 | WP-08 | `D-1`, `C-4` | AVX2 sin/cos returns values far outside [-1, 1], ±inf or NaN for large arguments (fast_sincos, fast_geo2cart, baseband_freq_response) |
| B-027 | S2 | WP-12 | `D-2` | `point_inside_mesh` misclassifies points 5 cm from a face at UTM-like coordinates in AVX2 builds (float ray–triangle kernel) |
| B-028 | S2 | WP-12 | `D-3` | `obj_overlap_test` misses a clear overlap when vertices of the penetrating object lie on extensions of the other object's face planes |
| B-029 | S2 | WP-01 | `F-1` | MATLAB R2026a segfaults at exit after any HDF5 MEX call (the unloaded thread-safe libhdf5 leaves a dangling pthread-key destructor) |
| B-030 | S2 | WP-06 | `F-2` | MEX: sparse inputs are read as dense arrays (wrong values, heap over-read) in every wrapper |
| B-031 | S2 | WP-01 | `F-3` | MEX `qrt_file_read` (`i_cir`) and `hdf5_read_channel` (`snap`) modify the caller's uint64 variable in place |
| B-032 | S2 | WP-06 | `F-4` | MEX `channel_export_obj_file` segfaults when `no_interact` has fewer snapshots than the coefficients |
| B-033 | S2 | WP-06 | `F-5` | MEX `qd_mex_get_strings` dereferences empty cell elements: `cell(1,1)` as a name list crashes MATLAB R2026a |
| B-034 | S2 | WP-06 | `F-6`, `E-2` | MEX helpers convert NaN, negative, fractional and out-of-range numbers to integers without checks: crashes, bad_alloc, silent truncation, wrapped seeds |
| B-035 | S2 | WP-05 | `G-2` | Python `arrayant.interpolate(local_angles=True)` without `dist=True` returns uninitialised memory |
| B-036 | S2 | WP-05 | `G-4` | Python `channel.baseband_freq_response`: an exception inside the OpenMP region aborts the interpreter |
| B-037 | S3 | WP-08 | `A-03` | AVX2 antenna interpolation returns 0 where all four corner amplitudes are below 1.19e-7 (generic interpolates) |
| B-038 | S3 | WP-10 | `A-10` | qdant writer stores element positions, coupling and pattern values with only 6 significant digits |
| B-039 | S3 | WP-01 | `A-11` | `qdant_write_multi` deletes an existing file before it validates its input |
| B-040 | S3 | WP-10 | `A-12` | `qdant_read(fn, 1)` returns another entry without error when the file has no ID 1 |
| B-041 | S3 | WP-10 | `A-13` | `arrayant::qdant_write` without a layout replaces an existing custom layout in the file (user decision) |
| B-042 | S3 | WP-10 | `A-14` | A 138-byte qdant file makes the reader allocate 4.7 GB and fail with `std::bad_alloc` |
| B-043 | S3 | WP-01 | `A-15` | `arrayant` default constructor leaves `check_ptr` uninitialised; the quick `is_valid()` can pass an invalid object |
| B-044 | S3 | WP-09 | `A-16` | ULA and 3GPP generators keep the custom pattern's element positions and coupling, contrary to the docs |
| B-045 | S3 | WP-10 | `A-17` | `combine_pattern` and `arrayant_combine_pattern_multi` hard-code the speed of light (acoustic arrays combined without position phase) |
| B-046 | S3 | WP-09 | `A-18` | `remove_zeros` and `calc_directivity_dBi` use absolute power thresholds (low-level patterns lose data or report 0 dBi) |
| B-047 | S3 | WP-10 | - | `generate_arrayant_multibeam` defaults M = N = 2 in the header, 1 in the docs and both wrappers |
| B-048 | S3 | WP-07 | `B-009` | QRT readers trust header and path counts: a 136-byte file returns 20 million zero paths or allocates GBs |
| B-049 | S3 | WP-07 | `B-010` | `qrt_file_read` truncates `i_cir` / `i_orig` to 32 bits before the range check |
| B-050 | S3 | WP-02 | `B-011` | HDF5 write/read round trip changes data (empty snapshots, missing frequency, 1-element vectors, empty strings, long names) |
| B-051 | S3 | WP-02 | `B-012` | HDF5 functions call `H5close()`, switch off error printing process-wide, and leak handles and memory on error paths |
| B-052 | S3 | WP-02 | `B-013` | `hdf5_write` "overwrites" a slot by adding a new group; the file grows with every rewrite |
| B-053 | S3 | WP-02 | `B-014` | `hdf5_read_dset` returns a variable-length string as 8 bytes of a heap pointer, and leaks it |
| B-054 | S3 | WP-03 | `B-015` | `quantize_delays` builds dense vectors over the absolute delay range: a 20 ms offset makes it 4000 times slower |
| B-055 | S3 | WP-03 | `B-016` | `quantize_delays` gives wrong results when the outputs alias the inputs (undocumented) |
| B-056 | S3 | WP-14 | `B-017` | `mitsuba_xml_file_write` is quadratic in the number of vertices per object |
| B-057 | S3 | WP-14 | `B-018` | OBJ reader: CRLF `.mtl` files lose their materials; face indices unchecked when only `face_ind` is requested; no-output call skips validation |
| B-058 | S3 | WP-14 | `B-019` | `write_png`: constant, NaN or inverted-range data trigger a NaN-to-integer cast and all pixels get the last colour; an empty matrix writes an invalid PNG |
| B-059 | S3 | WP-07 | `B-020` | `channel::add_paths` accepts inconsistent counts and leaves an invalid object; `qrt_file_read` accepts any `normalize_M` |
| B-060 | S3 | WP-17 | `B-020` | Channel and file-I/O header comments contradict the code (index bases, `calc_effective_path_gain` "average", `normalize_M` column power) |
| B-061 | S3 | WP-13 | `C-10` | `combine_irs_coord`: no shape check on `interact_coord_1/2`, heap over-read |
| B-062 | S3 | WP-13 | `C-11` | `baseband_freq_response`: `aligned_alloc` with sizes that are not a multiple of 32 (generic build) and unchecked NULL return |
| B-063 | S3 | WP-01 | `C-12` | Test defect: `test_qd_ieee_chan.cpp:318` reads past the delay cube (hidden in the full ASan suite run) |
| B-064 | S3 | WP-11 | `C-13` | IEEE indoor model: `XPR_NLOS_linear = 0` gives NaN coefficients |
| B-065 | S3 | WP-11 | `C-14` | Unchecked float-to-integer casts (UB) for non-finite `observation_time` and tiny `quantize` |
| B-066 | S3 | WP-13 | `C-15` | `calc_delay_spread` with `granularity`: memory n_cir × global delay span / granularity (800 MB for 100 paths) |
| B-067 | S3 | WP-13 | `C-16` | `acdf`: all-NaN data sets bias the averaged CDF; unsorted user bins are accepted |
| B-068 | S3 | WP-04 | `C-17` | Docs of `add_fake_los_path` do not match the code: always one extra path; a present LOS is moved to slot 0 and a zero path is left behind |
| B-069 | S3 | WP-04 | `C-18` | Naming of the cross-polarisation rows of `M` is inconsistent; `calc_cross_polarization_ratio` V/H columns are ambiguous (user decision) |
| B-070 | S3 | WP-13 | `C-19` | `calc_diffraction_gain` (C++) scalar mode expects an electromagnetic-equivalent frequency, the docs say the speed of sound is used |
| B-071 | S3 | WP-12 | `D-4` | `ray_triangle_intersect`: AVX2 and generic kernels swap `fbs_ind`/`sbs_ind` for hits at equal distance (shared edge or vertex) |
| B-072 | S3 | WP-12 | `D-5` | `ray_triangle_intersect`: the AABB slab test rejects rays with dz = 0 in the plane z = Zmin/Zmax (x and y are accepted) |
| B-073 | S3 | WP-12 | `D-6` | `ray_point_intersect` requires sub-clouds aligned to 8 only in AVX2 builds; the default `point_cloud_segmentation` output makes it throw |
| B-074 | S3 | WP-08 | `D-7`, `G-10` | Invalid `use_kernel` values are accepted silently; `use_kernel = 2` does not throw in builds without AVX2 (confirms item 1.7 of the Python fix plan) |
| B-075 | S3 | WP-08 | `D-8` | AVX2 `fast_atan2` / `fast_cart2geo`: NaN becomes π/4, (inf, inf) gives NaN, tiny or huge double vectors give elevation 0; two doc claims are wrong |
| B-076 | S3 | WP-08 | `D-9` | AVX2 `fast_slerp` fast path: error 3e-6 (about 50 ULP, documented ~5) and each lane's result depends on its neighbours |
| B-077 | S3 | WP-12 | `D-10` | `obj_overlap_test` crashes on an empty mesh |
| B-078 | S3 | WP-12 | `D-11` | `subdivide_triangles` (and `icosphere`, `cube`, `plane`): unchecked overflow of the face count leads to writes through a null pointer |
| B-079 | S3 | WP-12 | `D-12` | An output that aliases an input gives wrong results (`ray_triangle_intersect` with `fbs = &orig`, `subdivide_triangles` in place) |
| B-080 | S3 | WP-01 | `D-13` | `point_cloud_split` leaks its work buffer on every failed split |
| B-081 | S3 | WP-12 | `D-14` | `obj_overlap_test` docs call `obj_ind` 1-based and "the output of `obj_file_read`", which is 0-based; IDs are returned as given |
| B-082 | S3 | WP-12 | `D-15` | `icosphere(radius = 0)` returns NaN; `cube` / `plane` with a negative scale return inward / -z normals, contrary to the docs |
| B-083 | S3 | WP-13 | `D-16` | `interp` 1-D/2-D index search is O(n·m): 4 s for 160 000 samples |
| B-084 | S3 | WP-12 | `D-17` | `obj_overlap_test`: reason strings are non-deterministic (unseeded shuffle, thread count), and a flag array is read outside its critical section |
| B-085 | S3 | WP-01 | `F-7` | MEX `fast_sincos(x, [])` returns cos instead of sin; the single-precision output is undocumented |
| B-086 | S3 | WP-01 | `F-8` | MEX `arrayant_qdant_write` (separate inputs) writes `center_freq = NaN` for `[]` instead of the documented 299792458 |
| B-087 | S3 | WP-15 | `F-9` | `arrayant_generate('multibeam')` uses a 90° element beamwidth by default in MATLAB and Python; docs (and the C++ default) say 120° |
| B-088 | S3 | WP-15 | `F-10` | `calc_angular_spread` defaults differ: MATLAB and Python disable wrapping and the bank angle, C++ enables both |
| B-089 | S3 | WP-15 | `F-11` | `triangle_mesh_segmentation` pads `mtl_ind_out` with material 1 in MATLAB, with 0 ("no material") in C++ and Python |
| B-090 | S3 | WP-15 | `F-12` | MEX `baseband_freq_response` with `carrier_freq`: descending carriers are rejected, duplicate carriers give silent NaN (Python handles both) |
| B-091 | S3 | WP-15 | `F-13` | MEX `get_channels_irs` `i_irs` is 0-based, against the 1-based rule of the MATLAB spec; the doc does not say which |
| B-092 | S3 | WP-15 | `F-14` | `arrayant_qdant_read(fn)` returns every entry as a struct array in MATLAB, Python's default reads ID 1 |
| B-093 | S3 | WP-06 | `F-15` | MEX: arrays with more than 4 dimensions are silently truncated; a 4-D dataset written with `hdf5_write_dset` comes back 3-D |
| B-094 | S3 | WP-06 | `F-16` | MEX: index value 0 is not rejected in several wrappers (silent wrong result or a misleading error) |
| B-095 | S3 | WP-15 | `F-18`, `G-11` | Diffraction level of detail 7: accepted by C++, empty `coord` in MATLAB, internal error in Python, rejected by MATLAB `generate_diffraction_paths`, documented as 0-6 |
| B-096 | S3 | WP-05 | `G-5` | Python `qrt_file_read`: an exception in the main loop while the reader thread runs calls `std::terminate` |
| B-097 | S3 | WP-05 | `G-6` | Python HDF5 writers reject numpy scalars, 0-d arrays and non-native dtypes that the docs list as supported |
| B-098 | S3 | WP-05 | `G-7` | Python adapter `view = true` always copies (confirms item 1.1 of `md/python_fix_plan_v00.md`) |
| B-099 | S3 | WP-05 | `G-8` | Python: documented kwargs `sub_cloud_index` and `map_to_itu_materials` raise `TypeError` |
| B-100 | S3 | WP-05 | `G-9` | Python: invalid input accepted silently (2-D positions, short `interact_coord`, 4-D arrayant without `center_freq`) |
| B-101 | S4 | WP-09 | `A-20` | `arrayant::copy_element` ignores `read_only`; an empty destination gives an Armadillo-internal error |
| B-102 | S4 | WP-17 | - | Arrayant documentation typos and misleading statements; latent int32 offsets and a vertex sentinel (grouped) |
| B-103 | S4 | WP-03 | `B-022` | `quantize_delays` docs: `fix_taps` 1 and 2 give `[1, 1, n_taps]` delays only for shared input |
| B-104 | S4 | WP-17 | - | Channel and file-I/O documentation and message nits (grouped) |
| B-105 | S4 | WP-01 | `C-20`, `G-14` | `memcpy` with a null pointer and size 0 (formal UB) in `get_channels_irs` and in the Python list adapter |
| B-106 | S4 | WP-17 | `C-21`, `C-22` | Docs and code of `path_to_tube`, `qd_path` and `calc_diffraction_gain` debug output disagree (grouped) |
| B-107 | S4 | WP-13 | `D-18` | `interp`: a NaN query returns the first data value; clamped extrapolation is undocumented |
| B-108 | S4 | WP-17 | `D-19` | Geometry documentation and validation nits (index-map directions, 9-column check, precision claim, header typos; dead code; CPU check) |
| B-109 | S4 | WP-16 | `E-1` | `sos_val` / `sos_remove_bias`: an output that partially overlaps an input is accepted and gives wrong values (docs promise an error) |
| B-110 | S4 | WP-16 | `E-3` | `sos_randn` / `sos_rand` / `sos_randi` report an unknown preset as "Input 'name' ...", an argument they do not have |
| B-111 | S4 | WP-16 | - | `md/SPEC_PYTHON_API.md` gives two contradicting rules for the headings of the `sos` entries |
| B-112 | S4 | WP-01 | `F-17` | MEX `qrt_file_read` returns `interact_type` as uint8; the doc says uint32 |
| B-113 | S4 | WP-15 | `F-19` | MEX `get_channels_*` with 0 paths and `add_fake_los_path` return a zero path with delay 0, where C++ throws |
| B-114 | S4 | WP-15 | `F-20` | MEX `arrayant_rotate_pattern`: struct arrays reject usage 3 and 4 (Python maps them); a call without output argument errors |
| B-115 | S4 | WP-17 | `F-21` | MEX documentation, error IDs and latent helper defects (grouped) |
| B-116 | S4 | WP-05 | `G-12` | Python `hdf5_create_file` returns uint32 although the wrapper builds int64 (confirms item 1.3 of the Python fix plan) |
| B-117 | S4 | WP-05 | `G-13` | Python output dtypes and shapes differ from the docs (`qdant_read`, `point_inside_mesh`, `generate_speaker`, `get_ieee_indoor`) |
| B-118 | S4 | WP-05 | `G-15` | Python single-frequency arrayant dict: a 1-element `center_freq` array gives a numpy DeprecationWarning, 2 elements an unclear error |
| B-119 | S4 | - | `B-119` | Crafted HDF5 files make HDF5 1.14.6 itself zero-fill about 4 GB in `H5Fopen` or request a 2^64-byte allocation in `H5Gopen2` (upstream; won't fix) |
| B-120 | S3 | WP-13 | `B-120` | `calc_cross_polarization_ratio`: circular XPR columns inverted for the LOS matrix of `get_channels_*`; per transmit vs per receive polarisation |
| B-121 | S3 | WP-12 | `B-121` | Test defect: `test_bin` without `BUILD_WITH_AVX2`, the AVX2-vs-generic ray kernel tests never run |
