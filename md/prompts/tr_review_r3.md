# Task: independent TR 38.901 compliance review of the large-scale model (final review round)

You are an independent reviewer. Working directory `/home/stephan/sjc/quadriga-lib-fixes`, branch `fixes`.
Check whether the library's large-scale model follows **3GPP TR 38.901 v19.5.0**
(`/home/stephan/sjc/references/3gpp_38901/38901-v19.5.0.md`), working **from the TR and the code only**.
Do not use the implementation spec, plan, findings or reports in `md/` as a reference (they may contain the same
mistakes as the code); read them only at the end to see whether a deviation you found is a documented, deliberate
choice.

Scope (code): `include/quadriga_gbsm.hpp`, `src/qd_gbsm_phases.cpp`, `src/lsp_generate.cpp`, `src/path_loss.cpp`,
`src/los_probability.cpp`, `src/o2i_loss.cpp`, `src/quadriga_gbsm_helpers.hpp`, `src/qd_scenpar.cpp` (key table),
and the 23 conf files `data/config/3GPP_38.901_v19_*.conf`.

Check, clause by clause, and recompute numbers yourself:
- **7.4.1 path loss**: every row of Table 7.4.1-1 (RMa, UMa, UMi, InH, InF SL/DL/SH/DH, SMa; LOS and NLOS),
  break points (notes 1 and 5, incl. the random hE of UMa), applicability limits, shadow-fading std per row; as
  realized by the conf constants plus the `path_loss` models. Evaluate at several (d2D, hBS, hUT, fc) points.
- **7.4.2 LOS probability**: every row of Table 7.4.2-1 incl. InF (Table 7.2-4, 7.8.4 calibration values) and SMa.
- **7.4.3 O2I**: Tables 7.4.3-1 to 7.4.3-3, d2D-in, sigma_P, car loss; how the O2I loss and the indoor distance
  enter the path loss (eq. 7.4-2).
- **7.4.4, 7.5 step 4, 7.6.3**: LSP generation: Table 7.5-6 parts 1-4 (means, stds, frequency dependence incl.
  notes 6/7 fc clamp, correlation distances, cross-correlations, cluster counts), Tables 7.5-7 to 7.5-12 (ZSD
  and ZOD offset; QuaDRiGa uses elevation = 90 deg - zenith, so the EoD offset has the opposite sign of the ZOD
  offset), the Cholesky order of step 4, clipping of spreads, 2D correlation and no correlation between floors,
  7.6.5 multi-frequency, 7.6.9 absolute TOA incl. the 2L/c bound, 7.6.3.3 soft LOS (7.6-18).
- Anything in clauses 7.4 to 7.6 that belongs to the large-scale model and is missing.

Accept deliberate deviations only if the code or the conf file documents them with a reason (e.g. legacy
options, the geometric RMa EoD offset generalization, the 3D break-point form of `dual_slope` with its stated
accuracy, Comb256 as default ACF). Report undocumented ones.

Rules: read-only. Do not build (another reviewer owns the builds); you may write small Python probes in your
scratchpad using the already built module (`PYTHONPATH` to the repo's `lib/` or the build directory the reports
name) or plain Python/numpy formulas. Write your full report to `md/reports/tr_r3_review.md` with a findings table
(ID `T3-nn`, severity blocker / major / minor / nit, TR clause, file:line, finding with numbers, proposed fix;
`[escalate]` for user decisions) and a compliance matrix (TR item → status: compliant / deviation documented /
deviation undocumented / missing). Final answer: at most 15 lines — counts per severity, majors one line each,
overall compliance verdict.
