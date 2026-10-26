# Task: revise the implementation spec after review round 1

Working directory `/home/stephan/sjc/quadriga-lib-fixes`. Revise `md/gbsm_lsf_spec.md` (draft r0 → r1) so that it
resolves the findings R1-01..R1-30 in `md/gbsm_lsf_findings.md`. Read first: the findings, the review report
`md/reports/spec_r1_review.md` (per-W verdicts, I-1..I-8 resolutions, BLAS assessment, SSF gaps), the writer
report `md/reports/spec_r0_writer.md`, then the spec.

User guidance that applies to all revisions (from the user's review of r0):
- D-decisions are not binding where a better solution exists; writer choices must be justified.
- Where QuaDRiGa-NG conflicts with TR 38.901, follow the TR, unless the QuaDRiGa choice gives correct statistics
  for a good reason. 36.873 is deprecated, low priority.
- Functions are double; the SOS evaluation may run in single precision / the AVX2 kernel, with matching
  tolerances.
- No references to `md/` documents or internal IDs (D-, W-, I-, R1-) in any produced source, header, test,
  wrapper or conf file; only 38.901 or other published material may be cited there.
- Lean code and lean public API; reuse existing helpers; helpers used once are lambdas, reused within one file
  are `static inline` / anonymous namespace, a private header only for helpers used across files (documented).
- MSVC compatibility. New conf keys fully integrated into reader and writer with defaults (not written when equal
  to the default) and tests.
- BLAS/LAPACK stays disabled.

Rules:
- Resolve every finding: fix the spec and fill the resolution column of its row in the findings table
  (`fixed: <where>`, or `rejected: <reason>`). Do not delete rows.
- Findings marked `[escalate]` (R1-09 sqrtm option, R1-10 o2i_loss scope, and any other) are the user's decision:
  do NOT apply them. Keep the current spec text, add a short `[pending user decision R1-nn]` note at the place,
  describe both options with their trade-offs in your report, and set the resolution to `pending user`.
  Write the spec so that each option can be applied later with a local edit.
- Bump the spec status line to r1 and add a short change log section (internal, IDs allowed there).
- Write only the spec, the findings table and your report `md/reports/spec_r1_revise.md` (what changed per
  finding, anything rejected and why, the escalations with options). No code, no build.
- Final answer to the coordinator: at most 20 lines — counts fixed / rejected / pending, the rejected ones in one
  line each, the pending decisions in one line each with your recommendation.
