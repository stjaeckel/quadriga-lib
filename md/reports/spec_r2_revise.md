# Report: spec revision r1.1 -> r2

Task: `md/prompts/spec_revise_r2.md`. Revised `md/gbsm_lsf_spec.md` (r1.1, 1896 lines -> r2, about 2040 lines) for
the findings R2-01 to R2-22 of `md/gbsm_lsf_findings.md`, and filled all 22 resolution cells. Nothing else in the
repository was changed, and nothing was built. Scratch checks (Python, scratchpad): the UMa p_LOS values of R2-06,
the 3.7 dB hE step of R2-10, d'BP for floors 7 and 8, and the symmetric probit against mpmath (worst relative
error 3.1e-16 over [1e-12, 1 - 1e-12]). TR 38.901 was read for 7.6.9 (L = max(length, width, height) for InF and
InH), the hall sizes of 7.8.4, the office of Table 7.2-2 and the UT heights of Table 7.2-1. In the library, I checked
`sos_init_phases` (column g does not depend on n_gen) and the `sos_val` interface (several phase columns per call).

Counts: **19 fixed, 0 rejected, 3 pending user** (R2-11, R2-12, R2-16).

## 1. Changes per finding

| Finding | Change in the spec |
|---|---|
| R2-01 | 3.2 step 5: the pairs are `(a_g + b_g)/sqrt(2)` with no further step. The reason is written out: the pair already has `Corr = (A_g + A_m)/2`, while QuaDRiGa's extra root gives `[[1 + c², 2c], [2c, 1 + c²]]`. The clamp and the 2x2 root are gone. Also updated: 3.0.5 (evaluation without coupling), the header comment of `lsf_draw`, the 3.2 properties (pair correlation; `x_g = x_m` exactly at D = 0), the 3.1 property wording and 5.0 (3e-5 class removed). 5.2 has a new statistical case: variance of columns 3..6 within 0.07, pair correlation within 0.08 of `(A_g + A_m)/2` at D = 5, 10, 20 m, and D = 0 exact. It is listed in 1.2 among the corrected QuaDRiGa bugs, in 10.2 among the non-switchable changes, and in the R4b release note. |
| R2-02 | 3.3.4 `dual_slope`: for the random hE, `hm = max(hUT, 0.1)` with no 1.1·hE clamp. The `hb` clamp stays: it never acts at hBS 25 m and keeps `hb - hE > 0` for a lower BS (a small deviation from the proposal, which dropped both). 3.3.6 and 3.3.7 updated. 5.3 random-hE case: hUT 19.5/22.5 with hE 18/21 forced through `rnd_he`, d2D 200/1000 m (d'BP 490/280 m), checked against the TR LOS formula for the LOS file and the LOS term of the NLOS file at 1e-9 dB. |
| R2-03 | 3.5.4: `F(p) = -F(1 - p)` for `p > 0.5`. Acklam (lower-tail and central branches) and Halley run only for `p <= 0.5`; the upper-tail branch is removed. The reason is stated. 5.5: the test-side probit is a bisection on `min(p, 1 - p)` with the sign rule, and the accuracy case covers both tails. |
| R2-04 | 10.2 non-switchable changes: per-link indoor 3D distance, per-link `sf_sigma`, `max(PLn, PL_LOS + o2i) + o2i` for O2I links of `nlos` files. 10.4: compare at segment starts or single-position tracks; for O2I links of `nlos` files, compare with the O2I loss set to 0 in the reference script. The 1.2 parity statement is qualified to match. |
| R2-05 | 2.1: the moved block is anchored by content, from the line after the group line to the end of the `conf_file_write` declaration, `write_defaults` included. |
| R2-06 | 5.5: d 18.1 m with hU 20 m and 22.5 m gives 1; d 20 m gives 0.97779. 3.5.2 states the excess range. |
| R2-07 | Section 8 `lsp_vals` row and 10.2: per-link `lsp_mu` / `lsp_sigma`, or their builder mean. The reference-geometry claim is gone. Added to the non-switchable list. |
| R2-08 | 10.3: the layout derives `round((hUT - 1.5)/3)` (outdoor 0) and stores it. 10.2: the builder passes it on drop and track positions. Section 8: the SSF sets use the same IDs. Also in 1.1, 1.2 and the 3.2 documented property. |
| R2-09 | New row in section 8: the SSF sets take columns 11.. of one `sos_init_phases(n_coeff, 11 + n_ssf, seed)` call, or a derived seed. The builder stores an explicit seed or the cubes (10.2). |
| R2-13 | 2.1 and 1.1: `o2i_loss(fc, rnd, models, weights, loss, d2d_in_max, d2d_in, model_index)`. |
| R2-14 | Resolved by setting the values, which the TR requires (7.6.9 bounds InF and InH). 4.4.1: `absTOA_Lmax` 120 m (InF SL, DH; Indoor) and 300 m (InF DL, SH), replacing the "not set here" comment; these are the halls already used for `DS_VS`. Also updated: 1.2, 3.4.4, 5.8 (the existing `absTOA_Lmax == 0` check becomes per-file values), the R4b release note and 10.2. The bound rarely acts (InF small hall above 3.5 sigma; InH practically never). |
| R2-10, R2-15, R2-17 to R2-22 | As stated in the findings table. 3.2.4 is renumbered 3.2.1 (all references updated). `qd_gbsm_freq` returns tables per distinct lambda plus `tab`. The InF fallback reason now names the literal results (1 for hB < hU, 0 for hB = hU). |

Other edits: status r2; a header paragraph on the pending-decision markers; R4b release-note items; a section 8
note on the TR XPR variant (`lsp_mu`/`lsp_sigma` slice 7, from report section 3); change-log row r2.

## 2. Rejected

None.

## 3. Pending user decisions (both options prepared in place)

- **R2-11, default preset** (note in 1.2, pointer under 2.1). A: keep `Comb256` and add its reason (better SOS fit,
  smooth tracks). B: `Exp256` (TR eq. 7.4-5), with `Comb256` as a legacy row in 10.2; the edits are listed.
  **Recommend B.** The TR fixes the exponential ACF, and Comb gives different statistics (0.78 vs 0.61 at half the
  distance), so guidance 2 favours the TR. Note: the library's `sos_rand*` default to `Comb256`, and neither
  option changes them.
- **R2-12, SPEC files** (note in the header). A: one sentence declaring them part of the source tree. B: copy their
  rules into a new 9.3 (about 900 lines, duplicated). **Recommend A.** They are versioned repository conventions
  that R1 edits, and a copy would drift.
- **R2-16, `vs_ratio`** (note in 1.2, pointer under 2.1). A: drop it, with the full edit list. B: keep it.
  **Recommend A.** The caller sets `DS_VS` in its scenpar copy, as it does for `absTOA_Lmax`. This removes one
  argument in three APIs, one rule and its tests.
