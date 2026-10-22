# Brief: bug hunt on branch `fixes`

This brief is the whole task description. Read all of it before you start, and read it again at the
start of every phase. Everything you find goes into `md/bugs.md` (create it), with one section per
issue. A later fix session uses that document alone: the user triages the items by severity and fixes
them one at a time. Write every section so that someone who has never seen this session can act on it.

## Setting

- Worktree: `/sjc/quadriga-lib-fixes`, branch `fixes`, starting at 4720a92. Work only there.
- Do not touch the main checkout `/sjc/quadriga-lib` in any way. Other sessions work there; reading
  it is allowed but not needed.
- The worktree starts without a build. `build_linux/`, `lib/`, `+quadriga_lib/`, the unpacked
  `external/*` libraries and the `test*.txt` baselines are all gitignored and so are missing. CMake
  unpacks the zips in `external/` itself.

## Hard rules

- **No change to `src/`, `include/`, `tests/`, `api_mex/`, `api_python/`, `data/`, `tools/` or any
  other tracked file.** You may create two files: `md/bugs.md`, and anything under
  `build_linux/scratch/hunt/` (gitignored). When you finish, `git status` must show exactly
  `md/bugs.md` and this brief as untracked, and nothing as modified.
- Do not commit. The user commits manually.
- Install nothing system-wide. No `pip install`, no `make python_install`, no `cp` into
  site-packages, no `apt`. Run Python against the worktree's `lib/` through `PYTHONPATH`. Run
  MATLAB/Octave with `addpath` on the worktree's `+quadriga_lib` parent.
- Do not fix anything, not even in a scratch copy you then present as "the fix". A scratch what-if
  build to *confirm a root cause* is allowed; say that it is a what-if.
- Measure, don't guess. A finding counts as reproduced only when a program or script you ran showed
  the wrong behaviour. Never present arithmetic you did by hand as an observed result. If a command
  does not run, say so and either fix the probe or record the finding as unconfirmed.
- Follow the machine rules in `~/.claude/CLAUDE.md` (load check, thread counts, `flock` for
  benchmarks), and pass them to every sub-agent word for word.
- CUDA is out of scope (the user's decision): no GPU runs and no review of the CUDA code.

## Scope

**In scope: the whole library except the ray-interaction machinery.**

- C++ core: everything in `src/` and `include/` not excluded below.
- AVX2 variants: the AVX2 files (`*_avx2.cpp`, `fastmath_*`, `qd_coeff_combine_avx2.cpp`,
  `sos_val_avx2.cpp`) are compared against their generic counterparts **by measurement**: build with
  `ENABLE_AVX2=ON` and `OFF`, then compare the outputs on the same inputs, including odd sizes, sizes
  not divisible by the vector width, empty input and NaN/Inf.
- MATLAB/Octave wrappers: `api_mex/*.cpp`, `src/mex_helper_functions.hpp`,
  `src/mex_quadriga_lib_functions.hpp`, `src/mex_compat_shim.cpp`. Build the MEX with g++-12, because
  MATLAB's bundled libstdc++ is old (`GNUmakefile` already sets `gcc_version = 12`).
  `/usr/local/bin/matlab` is R2026a and `matlab24` is R2024a; Octave is at `/usr/bin/octave`.
- Python bindings: `api_python/*`, built with `make python` into the worktree's `lib/`.
- File readers and writers: qdant (XML), OBJ, MTL/material import, HDF5, QRT, conf files, PNG,
  Mitsuba XML.

**Excluded (the user's decision):** the ray-interaction machinery, which has its own plan and
redesign:

- `src/ray_mesh_interact.cpp`, `src/ray_progress.cpp`, `src/ray_commit.cpp`, `src/ray_init.cpp`,
  `src/ray_subdivide_flag.cpp`, `src/subdivide_rays.cpp`, `src/quadriga_lib_material_helpers.hpp`
- their wrappers (`api_mex/ray_mesh_interact.cpp` and its Python counterparts, if any)
- the material columns and the physics they drive (`data/materials*.csv`, `md/wave.md`,
  `md/acoustic.md`, `md/sheet_solid_fix_plan_*`)
- the tests `test_wave_*`, `test_acoustic_*`, `test_ray_commit`, `test_ray_init`,
  `test_ray_subdivide_flag`

**Also excluded: CUDA** (to be refactored soon): `src/*.cu`, `src/cuda_common.hpp`,
`src/quadriga_lib_cuda_functions.hpp`, and the CUDA code paths of the functions that dispatch to
the GPU. The CPU paths of those functions stay in scope. Build with `ENABLE_CUDA=OFF`.

The geometric primitives stay in scope: `ray_triangle_intersect*`, `ray_point_intersect*`,
`point_inside_mesh`, `obj_overlap_test`, the mesh and point-cloud tools, `calc_diffraction_gain`.
Some in-scope functions call excluded code. If a defect in such a function traces into the excluded
code, do not analyse it. Give it one line in the appendix "Seen in excluded code" and move on.

**Kinds of findings, all counted:**

1. Wrong results, crashes, undefined behaviour, memory errors, data races, unchecked input that
   leads to UB, resource leaks, exception-safety holes.
2. Documentation that disagrees with the code: the header comment blocks in `src/` (the HTML and the
   MD API docs are generated from them), `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`,
   `md/SPEC_PYTHON_API.md`. A stale `html_docu/` alone is a minor item; one item covers all of it.
3. Inconsistencies between the APIs: C++, MATLAB/Octave and Python give different results,
   defaults, shapes, types or error behaviour for the same call.
4. Clear performance defects, such as accidental quadratic work, copies in hot loops, false sharing,
   or OpenMP regions that serialise. General tuning ideas are not findings.

## Earlier reviews: do not repeat them

Read these first. They record findings and their status. They are no longer in the tree; read them from
the git history, e.g. `git show b7c1009:md/bugs.md`:

- `md/bugs.md` (the bug hunt of 2026-10-15, 123 items; all closed by the fix series merged in 730641b)
- `md/REVIEW_QD_SOS.md`, `md/REVIEW_QD_SOS_V2.md` (sum of sinusoids, reviewed twice)
- `md/REVIEW_CONF_FILE.md`, `md/PLAN_CONF_FILE_FIXES.md`, `md/LOG_CONF_FILE_FIXES.md` (conf files)
- `md/python_fix_plan_v00.md` (Python API, 2026-10-05, from reading the code only)

A finding marked there as resolved is reported again only if you reproduce it at HEAD, and then as a
**regression** with a reference to the old ID. A finding that is still open there is not copied. If
you confirm it by measurement for the first time (for example a "(verify)" item of the Python plan),
give it its own section that cites the old item and adds the reproduction. The SOS and conf-file
code has had two passes; give it a lighter pass focused on regressions, the wrappers and areas the
reviews did not name.

## Phases

### Phase 0: build and baseline (you do this yourself, not a sub-agent)

1. Check the load, then build the library and the Catch2 suite in the worktree (`ENABLE_TESTS=ON`).
   Run the whole suite once and record the pass line (assertions and cases), the wall time and the
   load. Save it as `build_linux/scratch/hunt/base_test.txt`. If the suite does not pass at HEAD,
   that is finding number one; record it and continue.
2. Build the other configurations the scope needs, each in its own build directory in the worktree:
   generic (`ENABLE_AVX2=OFF`), AVX2 (`ENABLE_AVX2=ON`), and ASan+UBSan (`-fsanitize=address,undefined
   -fno-omit-frame-pointer`, AVX2 on). Run the suite under the sanitizers once. Every sanitizer report
   is a candidate finding. Use `halt_on_error=0` so one report does not hide the rest.
3. Build the MEX for Octave and MATLAB and the Python module. Run `tests/quadriga_lib_mex_tests.m`
   (Octave and MATLAB) and `pytest tests/python_tests`, and record the results.
4. Fuzzing with AFL++ 4.33c (installed, checked on a toy target 2026-10-15). Facts and traps:
   - Use `afl-clang-fast++` (clang 21). `afl-g++-fast` fails with a GCC plugin version error, and
     `afl-g++` has been removed.
   - AFL's clang has no OpenMP (no `omp.h`, no libomp). Build the fuzzing library without OpenMP.
     CMake falls back to single-threaded by itself, and the four files that call the OpenMP API guard
     it with `#ifdef _OPENMP`. A separate build directory, e.g. `build_afl`, with
     `CC=afl-clang-fast CXX=afl-clang-fast++`, ASan on, `ENABLE_AVX2=OFF`, no MEX or Python. Whether
     the whole library compiles with clang is untested (one TU did). If it does not, link the harness
     against only the reader sources it needs.
   - The system sends core dumps to apport, which makes AFL report crashes as **hangs** at the
     default 20 ms timeout. Do not change `core_pattern` (a system setting). Instead, call
     `prctl(PR_SET_DUMPABLE, 0)` first in every harness's `main`, and run `afl-fuzz` with
     `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CPUFREQ=1 AFL_NO_UI=1`. With that, the toy
     crash was found as a crash under ASan within 2 s.
   - Build harnesses with ASan, with `AFL_LLVM_LAF_ALL=1` or a CMPLOG binary (`-c`). Without these,
     a three-byte magic compare was not found in 160,000 executions; with them it was found.
   - `tests/afl/Makefile` hard-codes paths into the main checkout and an old armadillo. Do not use
     it; `tests/afl/src/afl_qdant_read.cpp` and `tests/afl/qdant_samples/` can be read as a template
     and seeds.
   Write one harness per reader (qdant, OBJ, MTL/material import, HDF5, QRT, conf, PNG if a reader
   exists) in `build_linux/scratch/hunt/fuzz/`, with seeds from `tests/data` and `data/`. Have the
   harness catch the library's exceptions. Run each reader for a limited time (say 15 minutes at idle
   load, one `afl-fuzz` instance per free core, within the load rules), and record the executions and
   the coverage. A clean exception is correct behaviour. A crash, a sanitizer report, a real hang
   (re-run it outside AFL) or an accepted-but-wrong result is a finding.

Write the build commands and results into the "Baseline" section of `md/bugs.md` before phase 1.

### Phase 1: the area hunt (sub-agents allowed)

Split the scope into areas and give each area to one sub-agent, or do an area yourself. A suggested
split:

| Area | Contents |
|---|---|
| A. Array antennas | `qd_arrayant*`, interpolation (generic + AVX2), generate, combine, rotate, qdant read/write, xio, speaker, beamwidth/directivity |
| B. Channel object and file I/O | `qd_channel*`, HDF5, QRT, `quantize_delays`, OBJ read/write, Mitsuba, PNG, conf files (light pass) |
| C. Channel generation and tools | `get_channels_*` (planar, spherical, multifreq, IRS, IEEE indoor), `baseband_freq_response` (+ AVX2), `qd_tools_*`, acdf, diffraction gain and paths, `qd_path`, `coord2path`, `path_to_tube`, `update_xpr`, `combine_irs_coord`, `qd_scenpar` |
| D. Geometry and math | ray/triangle and ray/point intersect (generic, AVX2), `point_inside_mesh`, point-cloud and triangle-mesh tools, subdivide, icosphere/cube/plane, `obj_overlap_test`, fastmath, slerp, rotation matrix, interp, geo/cart |
| E. SOS | `sos_*` (light pass, regressions) |
| F. Cross-API layer | MEX helpers and every `api_mex` wrapper, the Python adapters and every `qpy_*` wrapper: argument checks, shapes, types, defaults, copies, error paths, docs against the SPEC files |

Rules for sub-agents:

- Give each sub-agent this brief's hard rules, the scope, the finding format below and the machine
  rules. Each one writes **only** to its own `build_linux/scratch/hunt/<area>/` folder, with findings
  in `findings.md` there. Only you write `md/bugs.md`.
- Builds: sub-agents use the libraries you built in phase 0 and link their own small probes against
  them. They do not rebuild the library or the suite. Only one Catch2 suite binary may run at a time:
  two OpenMP suites at once each take about ten times as long. Run full-suite and other heavy runs
  through `flock /tmp/quadriga-fixes-suite.lock`. Small probes with `OMP_NUM_THREADS=1` may run side
  by side.
- Sub-agents may read test files, and a probe may `#include` a test file by path, but nothing under
  `tests/` is edited.
- Methods that work: read each public function against its documentation; probe edge cases (empty,
  size 1, mismatched sizes, NaN/Inf, negative and zero values, degenerate geometry, very large
  input); compare each AVX2 result with generic; compare 1 thread with many threads (non-determinism,
  races); compare C++ with MATLAB and Python on the same input; run probes under ASan/UBSan; check the
  OpenMP shared and private clauses; check integer overflow in size arithmetic (`uword` vs `int`,
  `n_rows * n_cols * n_slices`); check aliasing of output arguments with inputs.

### Phase 2: merge, deduplicate, judge

Merge the area findings into `md/bugs.md`. Merge duplicates. When one root cause shows up in several
places (for example a MEX helper used by 20 wrappers), it is one section listing every place. Give each
item its severity yourself, one by one (scale below). Re-run every reproduction once from the
commands as written in `md/bugs.md`, so the document reproduces as written, not only as remembered.

## Format of `md/bugs.md`

Top of the document:

1. Title, date, branch, commit, and the marks used: **[R]** reproduced by a program or script run in
   this session; **[M]** measured (timings, counts); **[C]** read from the code, not reproduced.
2. **Summary table**: ID, title (one line), area, kind (1-4 above), severity, mark. Sorted by
   severity, then area.
3. **Baseline**: build commands for every configuration, suite results, sanitizer results, MEX and
   Python test results, fuzzing time and iterations per reader, load during each.
4. **Coverage**: one row per in-scope source file: reviewed by reading (yes / partly / no), probed
   (yes / no), sanitizer-run (yes / no), and which IDs refer to it. This tells the fix session what
   was *not* looked at.

Then one section per issue:

```
## B-007: <one-line title>

- Severity: S2, because <one sentence: who is affected and how badly>
- Kind: wrong result | crash | UB | memory | race | leak | doc | cross-API | performance
- Mark: [R] / [C]
- Where: `src/file.cpp:123` (function), plus every other place with the same cause
- Since: commit, if you checked `git log -L` or blame (optional)

**What happens.** Observed against expected, with the numbers.

**Reproduce.** A minimal, self-contained program or script, shown in full inside this section (the
scratch folder is gitignored and may be lost), with the exact build and run commands from the
worktree root, and the output it printed.

**Cause.** The root cause, if known, with line references. Otherwise "unknown" and what is known.

**Proposed fix.** The smallest change that closes the issue fully, and the files it touches. If there
are several options, give a recommendation and its trade-off. If the fix changes documented behaviour
or an API (a default, a shape, an error), say so: the user decides which side to change.

**Test for the fix session.** A Catch2 / MOxUnit / pytest case that fails now and passes after the
fix: what it builds and what it asserts. For input that is invalid but can be built, the test asserts
a clean error rather than a result.

**Further analysis** (only if the cause or the fix is not settled): the next concrete steps.
```

Number the items B-001, B-002, ... in the order of the final summary table. After the items, add two
short appendices: "Seen in excluded code" (one line each) and "Checked and found correct" (one line per
suspicion that was tested and turned out to be fine, so the fix session does not hunt it again).

### Severity scale

- **S1 critical**: memory corruption, a crash, or silently wrong results on valid input in normal
  use. Also data loss in a writer.
- **S2 major**: wrong results or UB on valid but less common input (edge sizes, specific options), a
  data race, a wrapper that returns wrong data, or AVX2 differing from generic beyond rounding.
- **S3 minor**: invalid input accepted without an error, or rejected with an unclear one; API
  inconsistencies that do not corrupt results; documentation that would lead a user to a wrong call;
  clear performance defects.
- **S4 cosmetic**: typos in the documentation, stale generated docs, misleading names, dead code.

Judge each item on its own merits. Give the reason in one sentence, and do not inflate.

## Working with the user

The user is available but triages later. Ask only when you are blocked on a decision that is theirs
(for example whether the documented behaviour or the code is the intended one). Collect such questions and ask them together, not one by one. Do not report system load unless
it affected a result.

## Finish

1. `md/bugs.md` is complete in the format above, and every [R] reproduction has been re-run from the
   document.
2. `git -C /sjc/quadriga-lib-fixes status --short` shows only `?? md/bugs.md` and
   `?? md/bug_hunt_brief.md`.
3. Your final message gives the item count per severity, the three to five most serious items in one
   line each, and what was not covered.
