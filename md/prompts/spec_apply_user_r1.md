# Task: apply the user's decisions on spec r1

Working directory `/home/stephan/sjc/quadriga-lib-fixes`. Edit `md/gbsm_lsf_spec.md` (r1 → r1.1). Read
`md/gbsm_lsf_findings.md` and `md/reports/spec_r1_revise.md` first (they describe the prepared local edits).

The user's decisions (binding):

1. **Drop `PL_BP2D`.** The 0.034 dB difference between the 3D-breakpoint form (QuaDRiGa `dual_slope`) and the
   TR's 2D form for RMa/SMa is accepted; it does not justify a key. Remove the key everywhere (key lists, v19
   file values, path_loss algorithm, tests). The TR-oracle test for RMa/SMa LOS beyond the breakpoint uses a
   tolerance derived from the worst case over the TR height/frequency ranges (about 0.034 dB) with a comment that
   explains the difference. Update the R1-02 resolution in the findings table.
2. **Keep `sqrtm` as an option (R1-09 decided: keep).** The user prefers the symmetric square root because it
   removes the dependence on the parameter order; Cholesky in the TR order stays the default (TR behaviour). Use
   the compact algorithm prepared in spec 3.4.3 (Newton-Schulz or another short, robust method for a symmetric
   positive-definite 8x8 matrix; state convergence criterion, iteration limit, scaling, and the tolerance of the
   test `R^(1/2) · R^(1/2) = R`). No LAPACK. Resolution of R1-09: `user: keep sqrtm option`.
3. **Reduce `o2i_loss` to the TR models (R1-10 decided: reduce).** Apply the edits of spec 3.6.0: models low,
   high, low_A, legacy_6GHz, car, car_metal; 4-column `rnd`; drop `high_v14`, mmMAGIC, `loss_el`, the legacy
   d2D-in table and the geometry inputs/outputs; `path_loss` takes the per-link `d2d_in` as prepared. mmMAGIC is
   deprecated and stays in QuaDRiGa-NG. Update the integration hand-over accordingly. Resolution of R1-10:
   `user: reduce`.
4. **The spec must stand on its own.** An implementer must not need any other document except the source code
   and TR 38.901 (or other published material). Remove all references to the plan (`md/gbsm_lsf_plan.md`), to
   earlier revisions (r0, writer reports, review reports), and to D-/W-/I-/R1- numbers in the normative text.
   Where a decision matters, state it as a requirement with its one-line reason directly in the spec. A short
   internal change log at the end may keep IDs, marked as non-normative. Also remove such references from all
   text that implementers copy into code (header text, documentation blocks, test comments, conf comments).
   Also drop the 36.873 remnants if any remain (deprecated).

Rules: write only the spec, the resolution column of the findings table, and a short report
`md/reports/spec_r1_user_apply.md`. No code, no build. Final answer: at most 10 lines — what changed, and anything
you could not apply cleanly.
