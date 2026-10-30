# Bug scan

This document is the whole procedure for a full scan of the library for bugs and other defects. A coding session (the "manager") runs it with sub-agents, collects the findings in a temporary findings file, and at the end presents them to the maintainer one by one. The maintainer decides for each one whether it becomes a GitHub issue. Filing follows [CONTRIBUTING.md](../CONTRIBUTING.md), section "Filing an issue".

A full scan is expensive. It is meant to run occasionally and unattended, for example overnight. Problems found during normal work are filed directly as issues and do not need a scan.

Read this document completely before starting, and again at the start of every phase.

## Phase 0: settle the run with the maintainer

This is the only phase that needs the maintainer before the end. Ask all of these in one round, then run unattended:

1. **Exclusions.** Show the area table (below) and ask whether any area, file or kind of finding should be left out of this run. By default everything is in scope.
2. **Working folder.** All files of the scan live in one working folder outside the repository: the worktree, the builds, the logs, the findings file. The maintainer's local agent notes name its location; otherwise ask.
3. **Commit.** By default the scan covers the current `main`. Record the commit hash.
4. **Time budget.** By default about eight hours. Split it over the phases (fuzzing takes the largest share).
5. **GPU.** If the CUDA area is in scope, whether the GPU may be used during the run.

Then create the worktree, detached at the recorded commit: `git worktree add --detach <working folder>/worktree <commit>`.

## Hard rules

- No change to any tracked file in the worktree or in the main checkout. Everything the scan writes goes into the working folder.
- No commits, no pushes, no branches.
- Nothing is filed on GitHub before phase 5, and in phase 5 only after the maintainer's decision for that finding.
- Install nothing system-wide (no `pip install`, no `apt`, no `make python_install`) and change no system setting (for example `core_pattern`). Run Python against the worktree's `lib/` through `PYTHONPATH`, and MATLAB / Octave with `addpath`.
- Do not fix anything. A what-if build in the working folder to confirm a cause is allowed; say that it is a what-if.
- Measure, do not guess. A finding counts as reproduced only when a program or script run in this scan showed the wrong behaviour. Arithmetic done by hand is not an observation. If a command does not run, fix the probe or record the finding as not reproduced.
- The local machine rules (thread limits, lock files, GPU use, load checks) apply to every step. Pass them to every sub-agent word for word.
- Overnight nobody answers questions. A decision that belongs to the maintainer (for example whether the code or its documentation is the intended behaviour) is written into the finding and asked in phase 5.

## Resource limits for an unattended run

A run that exhausts memory or floods the session with output can end the session and lose the night's work. Every heavy command is therefore bounded:

- **Wall-clock limit.** Every build, suite run, test run and fuzzer runs under `timeout`. For a sanitizer suite run, use about five times the plain suite time measured in phase 1.
- **Memory limit, twice.** Sanitizer runs set `ASAN_OPTIONS=hard_rss_limit_mb=<limit>`, which stops the process with a clean report when it exceeds the limit. As a backstop, every sanitizer run and every fuzzer runs inside a memory-capped scope: `systemd-run --user --scope --quiet -p MemoryMax=<limit> -p MemorySwapMax=0 <command>`, which kills it (exit code 137) at the cap. Both were checked on a runaway test program (2026-10-30). Suggested limits: suite runs 16 GB RSS and a 24 GB cap; each fuzzer instance 2 GB RSS and a 4 GB cap.
- **Sanitizer options.** `ASAN_OPTIONS=halt_on_error=0:log_path=<folder>/asan:hard_rss_limit_mb=<limit>` and `UBSAN_OPTIONS=print_stacktrace=1:log_path=<folder>/ubsan`. `halt_on_error=0` needs the build flag `-fsanitize-recover=address`.
- **Output into files, summaries into the session.** All build, test, sanitizer and fuzzer output goes into log files in the working folder. Read summaries only: pass lines, counts, the first report of each distinct stack. Never print a whole log into the session.
- **Disk.** Check free space before phase 1; the builds take several GB.
- **Progress journal.** After each phase, the manager writes a short entry into `progress.md` in the working folder: what is done, where the results are, what comes next. A session that was interrupted or compacted resumes from there.

## Scope

Everything in the library is in scope unless the maintainer excluded it in phase 0.

| Area | Contents |
| :--- | :------- |
| A. Array antennas | `qd_arrayant*`, interpolation (generic and AVX2), generate, combine, rotate, qdant read/write, xio, speaker, beamwidth and directivity |
| B. Channel object and file I/O | `qd_channel*`, HDF5, QRT, `quantize_delays`, OBJ read/write, Mitsuba XML, PNG, conf files |
| C. Channel generation and tools | `get_channels_*` (planar, spherical, multifreq, IRS, IEEE indoor), `baseband_freq_response` (generic and AVX2), `qd_tools_*`, acdf, diffraction gain and paths, `qd_path`, `coord2path`, `path_to_tube`, `update_xpr`, `combine_irs_coord` |
| D. Stochastic channel model | `qd_scenpar`, `lsp_generate`, `ssf_clusters`, `qd_gbsm_phases`, `path_loss`, `los_probability`, `o2i_loss`, `qd_atmospheric_absorption`, `quadriga_gbsm_helpers` |
| E. SOS | `sos_*` (generic and AVX2) |
| F. Geometry and math | ray/triangle and ray/point intersect (generic and AVX2), `point_inside_mesh`, point-cloud and triangle-mesh tools, subdivide, icosphere / cube / plane, `obj_overlap_test`, fastmath, slerp, rotation matrix, interp |
| G. Ray interaction and acoustics | `ray_mesh_interact`, `ray_progress`, `ray_commit`, `ray_init`, `ray_subdivide_flag`, `subdivide_rays`, the material tables in `data/` |
| H. CUDA | `src/*.cu`, `cuda_common.hpp`, `quadriga_lib_cuda_functions.hpp`, and the GPU paths of the functions that dispatch to them |
| I. Cross-API layer | the MEX helpers and every `api_mex` wrapper, the Python adapters and every `qpy_*` wrapper |

Notes on the areas:

- **G.** Comparisons in the acoustic material tests marked `ENGINE LIMIT (<name>):` are known engine limits. They are not findings, unless the measured value moved.
- **H.** CUDA runs follow the local GPU rules: one GPU user at a time, never in parallel sub-agents. The CPU paths of dispatching functions belong to their own area.
- If a defect in one area traces into an excluded area, give it one line in the appendix "Seen in excluded code" and do not analyse it.

**Kinds of findings, all counted:**

1. Wrong results, crashes, undefined behaviour, memory errors, data races, unchecked input that leads to undefined behaviour, resource leaks, exception-safety holes.
2. Documentation that disagrees with the code: the header comment blocks in `src/` (the HTML and MD API docs are generated from them), `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`. A stale `html_docu/` alone is one minor item covering all of it.
3. Inconsistencies between the APIs: C++, MATLAB / Octave and Python give different results, defaults, shapes, types or error behaviour for the same call.
4. Clear performance defects, such as accidental quadratic work, copies in hot loops, false sharing, or OpenMP regions that serialise. General tuning ideas are not findings.

## What is known already

Read this before the hunt, so known items are not reported as new:

- **GitHub issues, open and closed:** `gh issue list --state all --limit 1000 --json number,title,state,stateReason,labels`, then `gh issue view <N>` for the ones near a finding.
  - An open issue that covers a finding is not a new finding. If the scan adds information (a reproduction, a second place, a cause), the finding is presented in phase 5 as a comment for that issue.
  - An issue closed as completed is reported again only if the scan reproduces it at the scanned commit, and then as a **regression** of that issue.
  - An issue closed as not planned is not reported again, unless the scan shows it is worse than the closing comment says.
- **The first full bug hunt** (2026-10-15, 123 items, all fixed afterwards): `git show b7c1009:md/bugs.md`. Use it to look for regressions and to see which methods found what.

## Phase 1: build and baseline (the manager, no sub-agents)

1. Check the load. Build the library and the Catch2 suite in the worktree (`ENABLE_TESTS=ON`), each configuration in its own build directory in the working folder. Run the whole suite once and record the pass line (assertions and cases), the wall time and the load. If the suite does not pass at the scanned commit, that is finding number one; record it and continue.
2. Build the other configurations the scope needs: generic (`ENABLE_AVX2=OFF`), AVX2 (`ENABLE_AVX2=ON`), sanitizers (`-fsanitize=address,undefined -fsanitize-recover=address -fno-omit-frame-pointer`, AVX2 on), and CUDA (`ENABLE_CUDA=ON`) if area H is in scope. Run the suite under the sanitizers once, within the resource limits above. Every sanitizer report is a candidate finding.
3. Build the MEX for Octave and MATLAB and the Python module. The MEX for MATLAB needs a compiler whose libstdc++ is not newer than MATLAB's (`GNUmakefile` sets `gcc_version`). Run `tests/quadriga_lib_mex_tests.m` (Octave and MATLAB) and `pytest tests/python_tests`, and record the results.
4. Write the build commands and results into the "Baseline" section of the findings file.

## Phase 2: fuzzing the file readers (the manager)

Write one harness per reader (qdant, OBJ, MTL / material import, HDF5, QRT, conf files, PNG if a reader exists) in the working folder, with seeds from `tests/data` and `data/`. The harness catches the library's exceptions. A clean exception is correct behaviour. A crash, a sanitizer report, a hang that repeats outside the fuzzer, or input that is accepted but gives a wrong result is a finding.

Facts and traps for AFL++ (checked with AFL++ 4.33c and clang 21, 2026-10-15):

- Use `afl-clang-fast++`. `afl-g++-fast` fails with a GCC plugin version error.
- AFL's clang has no OpenMP. Build the fuzzing library without OpenMP in its own build directory (`CC=afl-clang-fast CXX=afl-clang-fast++`, ASan on, `ENABLE_AVX2=OFF`, no MEX or Python). CMake falls back to single-threaded by itself. If the whole library does not compile with clang, link the harness against only the reader sources it needs.
- If the system sends core dumps to a crash handler (for example apport on Ubuntu), AFL reports crashes as hangs. Do not change `core_pattern`. Instead, call `prctl(PR_SET_DUMPABLE, 0)` first in every harness's `main`, and run `afl-fuzz` with `AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 AFL_SKIP_CPUFREQ=1 AFL_NO_UI=1`.
- Build harnesses with ASan and with `AFL_LLVM_LAF_ALL=1` or a CMPLOG binary (`-c`). Without these, a three-byte magic compare was not found in 160,000 executions.
- `tests/afl/Makefile` hard-codes old paths; do not use it. `tests/afl/src/afl_qdant_read.cpp` and `tests/afl/qdant_samples/` serve as a template and seeds.
- Bound every instance: `afl-fuzz -V <seconds>` for the run time, `-t <ms>` for the per-input timeout, `-m none` together with the memory-capped scope and `hard_rss_limit_mb` from the resource limits. Run one instance per free core within the local thread limits.

Record executions and coverage per reader in the findings file.

## Phase 3: the area hunt (sub-agents)

Give each area in scope to one sub-agent, or do an area yourself. Rules for sub-agents:

- Each sub-agent gets this document's hard rules, the resource limits, the scope of its area, the finding format below and the local machine rules. It writes only into its own folder `<working folder>/areas/<area>/`, with its findings in `findings.md` there. Only the manager writes the findings file.
- Sub-agents use the builds from phase 1 and link their own small probes against them. They do not rebuild the library or the suite. Only one Catch2 suite runs at a time (two OpenMP suites at once each take about ten times as long); heavy runs go through a lock file. Small probes with `OMP_NUM_THREADS=1` may run side by side.
- Sub-agents may read test files, and a probe may `#include` a test file by path, but nothing under `tests/` is edited.
- A sub-agent returns a short summary (finding count, titles, what it did not cover). The details stay in its `findings.md`.

Methods that work: read each public function against its documentation; probe edge cases (empty, size 1, mismatched sizes, NaN / Inf, negative and zero values, degenerate geometry, very large input); compare AVX2 with generic results; compare one thread with many (non-determinism, races); compare C++ with MATLAB and Python on the same input; run probes under the sanitizers; check OpenMP shared and private clauses; check integer overflow in size arithmetic (`uword` against `int`, `n_rows * n_cols * n_slices`); check aliasing of output arguments with inputs.

## Phase 4: merge, compare, judge (the manager)

1. Merge the area findings into the findings file. Merge duplicates: one root cause that shows up in several places (for example a MEX helper used by 20 wrappers) is one finding listing every place.
2. Compare each finding with the GitHub issues as described under "What is known already". Mark it as new, as a comment for an open issue, or as a regression of a closed one.
3. Give each finding its severity, one by one (scale below), with the reason in one sentence. Do not inflate.
4. Re-run every reproduction once from the commands as written in the findings file, so the file reproduces as written, not as remembered.
5. Write the final entry into `progress.md`: findings per severity and what was not covered.

## Phase 5: triage with the maintainer

When the maintainer is back, present the findings one at a time, in order of severity. For each:

1. Say in plain terms what happens, physical meaning first; how large the effect is and who meets it in typical use; whether it was reproduced; the proposed fix in one or two sentences; any decision that belongs to the maintainer (code or documentation, which side to change).
2. Give a recommendation: file as a new issue, add as a comment to issue #N, or drop. Say why.
3. Ask with the question tool: **File as issue**, **Comment on #N** (when an issue matches), **Drop**, **Later**.
4. Act on the answer:
   - **File:** create the issue as in CONTRIBUTING.md, section "Filing an issue". The decision is the go-ahead. Labels: the type and the area labels; no `needs-triage`. The body uses the sections of the finding (below), without the scan's internal ID, severity marks or working-folder paths. The reproduction is shown in full in the body.
   - **Comment:** add the new information with `gh issue comment <N> --body-file <file>`.
   - **Drop:** nothing on GitHub.
   - **Later:** stays in the findings file for the next triage round.
5. Update the finding's status line in the findings file right away (`filed #N`, `commented #N`, `dropped`, `later`), so an interrupted triage resumes where it stopped.

At the end, list the issues filed and commented, and the findings left as "later". Removing the worktree (`git worktree remove`) and the working folder is the maintainer's call.

## Format of the findings file

The findings file is `<working folder>/findings.md`. Top of the file:

1. Title, date, the scanned commit, the areas in scope and the exclusions from phase 0, and the marks used: **[R]** reproduced by a program or script run in this scan; **[M]** measured (timings, counts); **[C]** read from the code, not reproduced.
2. **Summary table:** ID, title (one line), area, kind (1 to 4), severity, mark, relation to GitHub (new / #N / regression of #N), status. Sorted by severity, then area.
3. **Baseline:** build commands for every configuration, suite results, sanitizer results, MEX and Python test results, fuzzing time and executions per reader, and the load during each.
4. **Coverage:** one row per in-scope source file: reviewed by reading (yes / partly / no), probed (yes / no), run under the sanitizers (yes / no), and the IDs that refer to it.

Then one section per finding:

```
## F-007: <one-line title>

- Status: open | filed #N | commented #N | dropped | later
- Severity: S2, because <one sentence: who is affected and how badly>
- Kind: wrong result | crash | UB | memory | race | leak | doc | cross-API | performance
- Mark: [R] / [C]
- Area: <letter and name>
- GitHub: new | covered by #N (adds ...) | regression of #N
- Where: `src/file.cpp:123` (function), plus every other place with the same cause

**What happens.** Observed against expected, with the numbers.

**Reproduce.** A minimal, self-contained program or script, shown in full, with the exact build and run commands from the worktree root, and the output it printed.

**Cause.** The root cause, if known, with line references. Otherwise "unknown" and what is known.

**Proposed fix.** The smallest change that closes the issue fully, and the files it touches. If there are several options, a recommendation and its trade-off. If the fix changes documented behaviour or an API, say so: the maintainer decides which side to change.

**Test for the fix.** A Catch2, MOxUnit or pytest case that fails now and passes after the fix: what it builds and what it asserts. For input that is invalid but can be built, the test asserts a clean error.
```

Number the findings F-001, F-002, ... in the order of the summary table. After the findings, add two short appendices: "Seen in excluded code" (one line each) and "Checked and found correct" (one line per suspicion that was tested and turned out fine).

### Severity scale

- **S1 critical:** memory corruption, a crash, or silently wrong results on valid input in normal use. Also data loss in a writer.
- **S2 major:** wrong results or undefined behaviour on valid but less common input (edge sizes, specific options), a data race, a wrapper that returns wrong data, or AVX2 differing from generic beyond rounding.
- **S3 minor:** invalid input accepted without an error, or rejected with an unclear one; API inconsistencies that do not corrupt results; documentation that would lead a user to a wrong call; clear performance defects.
- **S4 cosmetic:** typos in the documentation, stale generated docs, misleading names, dead code.

## Finish

1. The findings file is complete in the format above, and every [R] reproduction has been re-run from the file.
2. `git -C <working folder>/worktree status --short` shows nothing, and `git -C <main checkout> status --short` shows nothing the scan wrote.
3. The final message of the unattended part gives the finding count per severity, the three to five most serious findings in one line each, and what was not covered. Phase 5 starts when the maintainer asks for it.
