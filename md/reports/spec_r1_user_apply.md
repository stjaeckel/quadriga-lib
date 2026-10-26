# Report: user decisions applied to spec r1 (r1 -> r1.1)

Task: `md/prompts/spec_apply_user_r1.md`. Changed `md/gbsm_lsf_spec.md` (1953 -> about 1900 lines) and the
resolution cells of R1-02, R1-09, R1-10 in `md/gbsm_lsf_findings.md`. No code, no build. Two scratch checks:
the RMa/SMa breakpoint difference over the TR ranges, and the coupled Newton-Schulz root on test matrices.

## 1. Decisions applied

- **`PL_BP2D` dropped.** Removed from 1.2, 3.3.4 (defaults, `dual_slope`, `nlos`), 3.3.5, 3.3.7, 4.3 (key row,
  writer comment row, now 5 keys), 4.4.2 (file lines), 5.3, 5.8, R3a and section 10. `dual_slope` keeps the 3D
  breakpoint form for every scenario; this is stated as a requirement in 1.2, with the reason. TR oracle (5.3):
  RMa/SMa LOS terms beyond dBP at 0.035 dB, the recomputed worst case (RMa 0.0341 dB at 0.5 GHz and hUT 1 m; SMa
  0.0313 dB), with the reason in a test comment, plus a 1e-9 dB check against the 3D-breakpoint form. New comment
  line in the 6 RMa/SMa LOS-term files.
- **`sqrtm` kept** (`cholesky = false`; Cholesky in TR order stays the default). 3.4.3 rewritten: coupled
  Newton-Schulz on `A/tr(A)`, stop at `||I - Z·Y||_F <= 1e-10` (the last update is kept), at most 100 iterations,
  symmetrized. No exception: for nearly singular input it stops at the limit, and the stable iteration returns the
  best root that double precision allows. Checked: 15/25/37 iterations for smallest eigenvalues 1e-2/1e-6/1e-10;
  residual 2e-14 at 1e-6. Test tolerance 1e-12 for `X·X = R` and `X = X^T` (5.4).
- **`o2i_loss` reduced** to low, high, low_A, legacy_6GHz, car, car_metal with a 4-column `rnd`. Header, 2.3,
  3.6 (rewritten), 5.6, 6.1, 6.2, 8 and 10 updated; Appendix 11.3 deleted. `path_loss` takes `d2d_in`, and the
  legacy distance geometry (d3D-in from d2D-in, 10 m minimum outdoor distance) moved into 3.3.2. Tests moved to 5.3.
- **Self-contained spec.** The decision table, writer-choice list and issue list are replaced by 1.2
  "Requirements (binding)", each with its reason. The content of the old lists now stands where it applies; new
  reasons were added for the symmetric coupling, the default `sf_sigma`, `DS_VS = 47/11`, full-precision constants
  and the release-note location. All D-/W-/I-/R1- markers and all references to the plan, r0 and review reports
  are gone from sections 1 to 11, including test-row and conf-comment text. The 36.873 remnants are removed: the
  equivalence paragraph, the documentation item and the test point. Section 10 is split into 10.1 to 10.5 (the
  old 15a to 15e labels are gone). Section 12 is a short non-normative change log that keeps the IDs.

## 2. Not applied cleanly / for review

- References to `md/SPEC_CPP_API.md`, `md/SPEC_MATLAB_API.md` and `md/SPEC_PYTHON_API.md` stay. They are the
  repository's API conventions, not documents of this series. If they must go too, their relevant rules have to
  be copied into the spec.
- Section 10.3 keeps one line on NG's deprecated 3GPP_3D groups: "no model name in the library; NG drops or maps
  them". This is integration guidance, not library behaviour.
- `path_loss` legacy distance: only links with `d2d_in(l) > 0` are changed. Reason: an outdoor link at d2D = 0
  would otherwise get d3in = d3 - 10 by the MATLAB NaN rule. MATLAB never applies this geometry to outdoor UTs.
- `md/reports/r<N>_report.md` (run reports, an output of each run) and the change log's pointer to the findings
  file stay. Neither is needed as input by an implementer.
