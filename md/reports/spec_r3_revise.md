# Report: spec revision r2.1 -> r3

Task: `md/prompts/spec_revise_r3.md`. Revised `md/gbsm_lsf_spec.md` (r2.1, 1990 lines -> r3, 2074 lines) for the
findings R3-01 to R3-15 of `md/gbsm_lsf_findings.md` and filled all 15 resolution cells. Nothing else in the
repository was changed, and nothing was built. Checked against the sources: the v19 conf files (section of
`absTOA_lambda`, the "Break point of note ..." lines, "13.34 (rounded)" only in the 4 UMa files, the SMa NLOS
comment block), the planning labels in the conf tests, `get_channels.m` (`scale_sf` only in the baseline branch),
`gen_ssf_parameters.m` (`xpr_sigma = 0.1 * lsp_vals(8,2,:)`), `gen_lsf_parameters.m` (2x2 matrix
`[1 A_ASD; A_ASA 1]`), `sos_init_phases` (column-by-column stream, seed message), the key table (defaults and
correlation key names) and TR Table 7.2-5 (SMa hBS 35 m). One scratch computation: the variance excess of
QuaDRiGa's extra root for unequal ACF values (1.73 and 1.29 at A_g 0.9, A_m 0.5).

Counts: **15 fixed, 0 rejected, 0 pending user** (no R3 finding was marked `[escalate]`).

## 1. Changes per finding

| Finding | Change in the spec |
|---|---|
| R3-01 | 5.3 random-hE case: the `nlos` LOS term is observed through a plpar copy of the UMa NLOS file with `Bn = -1000` (the `nlos` path with `hE_rand` is kept). The unmodified NLOS file is asserted as `max(PLn_TR, PL_LOS_TR(hE))`. This applies to both the 0.05 GHz identification part and the floor-7/8 points. |
| R3-02 | 3.0.4: the outer limit `sigma_i = max(0, s_i + kappa/tau/beta terms)`, with the reason. Tests: 5.3 (SF_sigma 1.4, SF_beta 1 at 1 and 10 deg elevation) and 5.4 (a link with a negative ASD std term). Also in 1.2 (corrected bugs), the 3.4.4 doc item, 10.2 (non-switchable), 10.4 (such links left out) and the R4b release note. |
| R3-03 | Made **switchable** instead of listing it as non-switchable. The library already has `sf_sigma` as an optional input, so NG's legacy preset can omit it and reproduce today's default mode. 10.2 gets a paragraph and a switch-table row; 10.4 gets the SF comparison rule as proposed. |
| R3-04 | 4.4.1: section "Absolute time of arrival offset". |
| R3-05 | 4.4.2: (a) In 9 files only the value changes, and no line is added. The "40/3" and "20 pi / 3" edits name the line they change. (b) SMa NLOS (2 files): the break-point line and the dual-slope formula block are removed. |
| R3-06 | 5.4: the covariance case uses the inputs proposed in the finding. Spot-check geometry is given. 5.5: SMa rows get hB 35 m. |
| R3-07 | 5.2: tolerance 0.1 (5 SE 0.05 + directional SOS error up to 0.045), with the computed directional ACF values in the comment. I chose 0.1 rather than the proposed 0.09, because 0.045 + 0.05 = 0.095 > 0.09. |
| R3-08 | 3.0.5 now fixes the signatures of `qd_gbsm_floor_offset`, `qd_gbsm_eval` and `qd_gbsm_remove_bias` (anonymous namespace, explicit parameters, no lambdas, comments without internal references). 3.1, 3.2, 3.2.1, 5.7 and the R2a steps refer to them. 9.1 has an explicit exception to the lambda rule. 8: the seed rule needs the LSF `n_coeff`; otherwise a derived seed is used. |
| R3-09 | 3.1: validation item 1 checks the seed; 5.1 adds a validation section. |
| R3-10 | 9.1: existing labels in edited files are reworded. 5.8 lists the labels found (O10; Q1, Q2, Q3; B1; "review point A") and their files. R1 step 5. |
| R3-11 | 5.8: the writer case compares the complete written lines at non-default values, comment included. |
| R3-12 | 8 XPR row: the factor 0.1 of QuaDRiGa's default mode. |
| R3-13 | 6.1: outputs beyond nlhs are passed as `nullptr` in every wrapper; the `lsp_generate` row and a new MEX test point. 6.2: Python note for column 9. |
| R3-14 | (a) to (d) as proposed. For (c) the spec now shows `sqrtm([[1, A_g], [A_m, 1]])` with the equal-distance covariance and the computed unequal example. |
| R3-15 | Each reason is kept once: the breakpoint bound in 3.3.4, the coupling in 3.2, `lsp_vals` in 8. 1.2, 5.3 and 10.2 refer to them. The 3.3.7 doc item and the 4.4.2 conf comment keep their own text, because those lines go into produced files. |

Other edits: status line r3, change-log row r3. "No user decision is open" stays true.

## 2. Rejected

None.

## 3. Pending user decisions

None.

## 4. Deviations from the proposed resolutions

- R3-03: switchable through the existing optional `sf_sigma` (switch-table row) instead of non-switchable.
- R3-07: 0.1 instead of 0.09 (the stated sum is 0.095).
