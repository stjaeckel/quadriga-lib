# Task: fix the findings of code review round 2

Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes` (clean tree at HEAD). Resolve all findings
C2-01 to C2-08 in `md/gbsm_lsf_code_findings.md`. Read first: the findings table, the review report
`md/reports/code_r2_review.md` (evidence, probes, the file-merge proposal), then the affected code. The spec
`md/gbsm_lsf_spec.md` and `md/SPEC_*.md` are the reference; update them where a decision below changes them.

## User decisions on the escalated findings (binding)

- **C2-05: option B.** Keep `qd_sqrtm8` and `qd_los_probit` as named file-local functions directly above their single caller. Add the refined rule to `md/SPEC_CPP_API.md` (helper rules): a helper used once is a lambda, except a self-contained algorithm of more than about 20 lines, which may be a named file-local function placed directly above its single caller.
- **C2-08: option B.** Keep the file layout; no further merging.
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

`md/reports/code_r2_fix.md`: per finding what changed, rejected findings with reasons, test and sanitizer results
(actual output summaries), probe results, and a proposed commit split with messages (no internal IDs or `md/`
references in the messages; the `md/` working files go into a separate "planning notes" commit).

Final answer: at most 15 lines — fixed / rejected counts, rejected ones in one line each, suite and sanitizer
results, new file layout.
