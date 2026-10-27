# Task: fix the findings of code review round 1

Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes` (clean tree at HEAD). Resolve all findings
C1-01 to C1-20 in `md/gbsm_lsf_code_findings.md`. Read first: the findings table, the review report
`md/reports/code_r1_review.md` (evidence, probes, the file-merge proposal), then the affected code. The spec
`md/gbsm_lsf_spec.md` and `md/SPEC_*.md` are the reference; update them where a decision below changes them.

## User decisions on the escalated findings (binding)

- **C1-01: option A.** Python flags: `None` = default, non-bool = `TypeError`, read with the existing
  `qd_python_flag` in all six new wrappers (as the conf functions do). Add the rule to `md/SPEC_PYTHON_API.md` and
  to spec 6.2; tests for `None` and for a non-bool value.
- **C1-10: option A.** Document in the C++, MEX and Python blocks of `lsf_draw` (and `lsf_init_phases` if it takes
  floors) that with dual mobility both devices are on the link's floor. No API change. Update the spec text.
- **C1-11: option A.** Keep the geometric EoD offset with actual heights; add a comment to the RMa NLOS and both
  RMa O2I conf files that it generalizes TR Table 7.5-9 (equal at hBS 35 m, hUT 1.5 m; 0 for UTs above the 3.5 m
  clutter). Update the spec text if it claims literal TR equality.
- **C1-14: merge.** Merge `src/lsf_init_phases.cpp` and `src/lsf_draw.cpp` into one file (choose a clear name,
  e.g. `src/lsf_phases.cpp`, and justify it) with the phase-set machinery file-local (anonymous namespace), as in
  the review proposal; shrink `src/quadriga_gbsm_helpers.hpp` to what several files need and keep its own
  documentation current. Merge the two test files likewise and remove the duplicated fixtures. Keep the
  signatures of the shared machinery so the later small-scale port can add its generator sets to this file.
  Update the spec (file names, the helper-header section, the SSF section).
- **C1-15: option A.** Tests use `quadriga_lib::sos_acfi` / `interp_1D` for table interpolation; the tables stay
  typed in the tests as oracle data.

All other findings: apply the reviewer's proposed fix, or a better one with a reason. Fill the resolution cell of
every row (`fixed: <what/where>` or `rejected: <reason>`).

## Rules (binding)

- Produced source, header, test, wrapper and conf files never reference temporary `md/` documents (spec, plan,
  findings, reports, prompts) or tracking IDs; they may cite published material; `md/SPEC_*.md` may be referenced.
- Lean code, reuse helpers, clear names; a helper used once is a lambda; reused within one cpp `static inline` or
  anonymous namespace; across files a documented private header. Public API fits `html_docu/quadriga_lib_api_cpp.md`.
- Tests use library calls for supporting tasks (not for the function under test); no test bloat.
- No BLAS/LAPACK; MSVC-compatible; no `std::to_chars` for floating point; objdump gate on the MEX files (nothing
  above GLIBCXX_3.4.28).
- Machine is shared: `cat /proc/loadavg`, at most 12 threads, never two builds or suites at once.
- Do not commit.

## Verification

- Full Catch2 suite (AVX2 on, and `make test`), MEX suite in Octave, R2024a and R2026a, full Python suite,
  objdump gate, no compiler warnings.
- Rebuild the ASan/UBSan tree (`build_asan`) and rerun the test cases of the new and changed functions: zero
  sanitizer diagnostics. No unbounded allocation probes under ASan.
- Rerun the reviewer's probes from the report that demonstrated bugs (NaN cases, aliasing, path_loss memory) and
  state the new results.

## Report

`md/reports/code_r1_fix.md`: per finding what changed, rejected findings with reasons, test and sanitizer results
(actual output summaries), probe results, and a proposed commit split with messages (no internal IDs or `md/`
references in the messages; the `md/` working files go into a separate "planning notes" commit).

Final answer: at most 15 lines — fixed / rejected counts, rejected ones in one line each, suite and sanitizer
results, new file layout.
