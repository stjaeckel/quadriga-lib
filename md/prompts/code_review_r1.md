# Task: critical code review of the large-scale model port, round 1

You are an independent reviewer; you wrote none of this code. Working directory
`/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes`. Review all code changes since commit `46dd28d` (the last
spec-only commit): `git diff 46dd28d..HEAD -- . ':!md'` plus the uncommitted changes, if any. Find bugs first,
then everything else that is wrong, fragile, bloated or inconsistent.

Reference material:
- The implementation spec `md/gbsm_lsf_spec.md` (what the code must do). The code may be right where the spec is
  wrong: then say so.
- TR 38.901 v19.5.0: `/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md` (clauses 7.4, 7.5 steps 1-4,
  Tables 7.5-6 to 7.5-12, 7.6.3, 7.6.9). Where QuaDRiGa-NG and the TR conflict, the TR wins unless the spec states
  a reason.
- QuaDRiGa-NG MATLAB code (read-only): `/home/stephan/sjc/QuaDRiGa-NG/quadriga_src` (legacy behaviour).
- Coding rules: `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md`, `md/SPEC_PYTHON_API.md`.
- Implementation reports: `md/reports/r1_report.md` ... `r4b_report.md` (decisions the implementers made).

What to look for, in order of priority:
1. **Correctness bugs**: wrong formulas or constants against the TR, wrong units, off-by-one, wrong
   indexing/strides (column-major), wrong frequency handling, sign of the EoD offset, reciprocity and bias
   removal, floor separation, dual mobility, clipping, Cholesky order, sqrtm convergence, probit tails, O2I model
   selection, path-loss model branches and break points, values in the 23 v19 conf files. Recompute numbers; write
   small probe programs in your scratchpad (link against the built library in `build_linux/` or use Python/MATLAB)
   to confirm suspected bugs. Every bug finding states a concrete input and the wrong vs. expected output.
2. **Robustness**: input validation (before any output is written), NaN/Inf, empty inputs, pre-sized outputs,
   aliasing, numerical edge cases (d2D = 0, hUT = hBS, near-singular correlation matrices, p = 0 or 1).
3. **Tests**: are the oracles independent of the implementation, are tolerances justified, do tests actually
   exercise the branches they claim, are there untested branches?
4. **Wrappers**: MEX and Python argument handling (incl. Python flags: the Run 4b report notes that `remove_bias=None` means False because the flags are plain bool, while the existing conf wrappers treat None as the default), 0/1-based indices, types, outputs not requested, error
   messages, documentation blocks consistent with C++.
5. **Code quality** (user's rules): lean code, reuse of existing helpers, no duplication, helper scoping (lambda if
   used once, `static inline` / anonymous namespace within a file, private documented header only across files),
   clear names, readable for humans and AI, MSVC-compatible (no GCC-only constructs, OpenMP 2.0, no VLAs,
   `_USE_MATH_DEFINES`), no BLAS/LAPACK use, no `std::to_chars` for floating point.
6. **No internal references** in produced files: no `md/` paths, no spec section numbers, no planning or review
   IDs; only TR 38.901 or other published material may be cited.
7. **SSF readiness**: the shared helpers in `src/quadriga_gbsm_helpers.hpp` and the field machinery must be
   reusable unchanged by a later small-scale port (spec section 8).

Rules:
- Do not edit repository files except the two named below. Probes go into your scratchpad. You may build and run
  the tests once (machine is shared: `cat /proc/loadavg`, at most 12 threads, never two builds or suites at once).
- Create `md/gbsm_lsf_code_findings.md` with a table (ID `C1-nn`, round, severity blocker / major / minor / nit,
  file:line, finding with evidence and proposed fix, resolution column empty). Mark findings that need the user's
  decision with `[escalate]`.
- Full report to `md/reports/code_r1_review.md`: method, probes run and their results, findings rationale,
  untested areas, overall judgement.
- Final answer: at most 20 lines — counts per severity, blockers and majors one line each, escalations, verdict.

## Additional review criteria from the user (binding)

- **References**: no produced source, header, test, wrapper or conf file may reference temporary documents in `md/`
  (spec, plan, findings, reports, prompts) or carry tracking numbers. They must stand on their own and may cite
  published material. The persistent coding-rule files `md/SPEC_*.md` may be referenced.
- **No code bloat**: reuse existing helpers; clean, lean structure, readable for humans and AI. Given only a source
  file, its purpose and working must be clear.
- **Public API** must fit the existing code base (`html_docu/quadriga_lib_api_cpp.md`): only what a caller needs;
  implementation details out unless needed to use the API correctly.
- **Names and scope**: well-crafted, clear names. A helper used once inside one function is a lambda in that
  function; helpers reused within one cpp are `static inline` or in an anonymous namespace; only helpers used across
  several files go into a private header, which needs its own documentation.
- **File structure**: several public functions may share one cpp file if they belong together, share helpers, or
  if it eases documentation (as the read/write functions of one file type do). Many of the new files have only a
  few hundred lines: evaluate merging them (e.g. `lsf_init_phases` + `lsf_draw` + the helper header; `path_loss`;
  `los_probability` + `o2i_loss`), especially where it improves readability and removes helpers, private-header
  entries or scattered tools. Make a concrete proposal (target file list, what moves where, what the private header
  still needs to hold, effect on the SSF reuse requirement) and judge whether it is worth it.
- **BLAS/LAPACK disabled**: work around it. **MSVC** must compile the code (not available here).
- **Tests**: use the library API (`html_docu/quadriga_lib_api_cpp.md`) for supporting tasks instead of
  re-implementing what one library call does (geometry helpers such as cubes or planes, coordinate conversions,
  interpolation, ...). The functions under test are the exception and must not be used as their own oracle. Flag
  test bloat and re-implemented helpers.

## Sanitizers

Build the Catch2 tests with AddressSanitizer and UndefinedBehaviorSanitizer in a separate gitignored build directory
(e.g. `build_asan`, `-fsanitize=address,undefined -fno-omit-frame-pointer -g`, AVX2 on) and run the test cases of
the new and changed functions (`lsf_*`, `lsp_generate`, `path_loss`, `los_probability`, `o2i_loss`, `conf_file_*`,
`scenpar_validate`). Report every sanitizer diagnostic as a finding. Do not run unbounded allocation probes under
ASan (a previous one crashed the machine); reproduce huge-allocation cases only in a plain build with `ulimit -v`.
Resource rules apply (at most 12 threads, one build at a time, check the load).
