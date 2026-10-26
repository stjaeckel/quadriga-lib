# Task: revise the implementation spec after review round 3

Working directory `/home/stephan/sjc/quadriga-lib-fixes`. Revise `md/gbsm_lsf_spec.md` (r2.1 → r3) so that it
resolves the round-3 findings `R3-nn` in `md/gbsm_lsf_findings.md`. Read first: the findings table (all rounds),
the review report `md/reports/spec_r3_review.md`, then the spec.

Guidance (from the user, binding):
- The spec stands on its own: no references to other internal documents or earlier revisions in normative text;
  no internal IDs in anything copied into code, headers, tests, wrappers or conf files. The non-normative change
  log at the end may keep IDs.
- Follow TR 38.901 where QuaDRiGa-NG conflicts, unless the QuaDRiGa choice gives correct statistics for a good
  reason. 36.873 and mmMAGIC are deprecated.
- Double functions; SOS evaluation may run in single precision / AVX2 with matching tolerances.
- Lean code and API; reuse helpers; helper scoping (lambda if used once, `static inline` / anonymous namespace
  within a file, private documented header only across files); MSVC; no BLAS/LAPACK.
- New conf keys fully integrated into reader and writer with defaults and tests.
- User decisions (resolutions starting with `user:`) are final.

Rules:
- Resolve every R3 finding: fix the spec and fill its resolution cell (`fixed: <where>` or `rejected: <reason>`).
- Findings marked `[escalate]` are the user's decision: do not apply; add a `[pending user decision R3-nn]` note
  at the place, prepare both options as local edits, resolution `pending user`.
- Bump the status line to r3 and extend the change log.
- Write only the spec, the findings table and `md/reports/spec_r3_revise.md`. No code, no build.
- Final answer: at most 15 lines — counts fixed / rejected / pending, rejected and pending one line each with your
  recommendation.
