# Report: spec writer, round 0

Task: `md/prompts/spec_writer.md`. Output: `md/gbsm_lsf_spec.md` (draft r0, about 1660 lines) and
`md/gbsm_lsf_findings.md` (header and empty findings table). No source, test, conf or plan file was changed; nothing
was built.

## Sections written

1. Scope and decisions: goal and data flow, D1 to D32 carried over unchanged, out-of-scope list, 40 writer choices
   (W-1 to W-40), 8 issues (I-1 to I-8), approved differences to the SPEC files.
2. Public API: complete text of `include/quadriga_gbsm.hpp` (four moved declarations, six new ones), LSF generator
   index with mirror and bias-removal columns, resolution of the plan's review-point-A items.
3. Normative behaviour: notation, common validation with messages, scenpar handling, the shared LSP mean/std
   formula, the internal helper header; per function validation (ordered, with messages), algorithm, edge rules and
   the items its documentation block must state. Path loss: all 10 models with required keys, defaults, clamps and
   `get_pl.m` line numbers; UMa random hE; O2I distances. LSPs: both roots (Cholesky in TR order, Jacobi sqrtm),
   clipping, abs TOA, ground permittivity, EoD offset. LOS probability: 20 models, edge rules, soft LOS, probit
   algorithm with coefficients. O2I: 9 models, rnd columns, model choice, d2D-in rules, d3D-in geometry, mmMAGIC
   elevation output.
4. Conf keys: the 6 table entries as C++ lines (class, section, paragraph, comment), one cross-key check, the 5 new
   `PL_` keys with writer comments, exact values and comment changes for the 23 v19 files (R1 part and R3 part).
5. Catch2 tests: conventions, per function a table of test cases (assertion, oracle, tolerance and reason), the
   internal helper tests, the changes to existing tests.
6. Wrappers: MEX usage lines, defaults, argument counts, output handling and test points; Python usage lines of
   submodule `gbsm`, types, flags, outputs and test points.
7. Runs R1 to R4: files, steps, acceptance, report format.
8. SSF readiness: table of 19 needed quantities with source and provider.
9. Constraints. 10. Integration hand-over (plan section 10 plus the legacy-switch table and the parity phase
   mapping). 11. Appendix tables (satellite zenith attenuation, 38.811 tables, legacy d2D-in CDF).

## Writer choices

| # | Choice |
|---|---|
| W-1 | Floor separation by per-floor phase offsets: Fourier offsets `2·pi·k·n/N` within blocks of `n_coeff` IDs (exactly zero correlation), `std::mt19937_64(block)` offsets between blocks. |
| W-2 | Dual mobility: Tx side of ASD/ASA/ESD/ESA uses the mirror generator's frequencies (`freq_scale_b = lambda_g/lambda_m`) -> exact swap reciprocity also for unequal lambdas. |
| W-3 | Symmetric dual-mobility coupling `c = (A_g(d) + A_m(d))/2` (MATLAB: non-symmetric). |
| W-4 | Bias removal only for generators 0..8, not for the Uniform generators 9, 10 (MATLAB: no bias removal for GR either). |
| W-5 | Bias removal with dual mobility: same-map correction for self-mirrored, two-map for 3..6; single mobility refreshes the Tx slice by mirroring. |
| W-6 | Dual mobility + bias removal + positions without `tx_pos` throws. |
| W-7 | 2D mode sets z = 0 for evaluation, bias removal and coupling distance. |
| W-8 | Default `sf_sigma` of `path_loss` = SF std of the LSP formula, not `SF_sigma + SF_delta·lg f`. |
| W-9 | Missing required PL parameters throw; first value of a `plpar` entry used; breakpoint factors > 0; `winner_los` needs `A1 != A2`. |
| W-10 | Satellite at or below the horizon: `PL = +Inf`; asin argument limited to 1. |
| W-11 | UMa random hE: `C' = 1` above 23 m, empty set (hUT < 13.5 m) gives 1 m, fixed mapping of one uniform value. |
| W-12 | Full-precision path-loss constants in the v19 files (`PL_E = 40/3`, `20·pi/3`, RMa/SMa derived values). |
| W-13 | `DS_VS` InF LOS = 47/11, DL/SH = 50/11, full precision. |
| W-14 | V/S term only if V/S > 0; validation of `DS_VS` without a, b; `vs_ratio > 0` without a, b throws. |
| W-15 | LOS probabilities limited to [0, 1]. |
| W-16 | Edge rules for InF-SH/DH and SMa heights. |
| W-17 | `36.873_UMa` uses `d^3` of TR 36.873, not the `d^2` of `set_scenario.m`. |
| W-18 | 38.811 tables fixed, elevation `atan2(h_bs, d2d_out)` from the absolute BS height. |
| W-19 | `o2i_loss`: `rnd` 4 or 6 columns; models `low, high, low_A, high_v14, legacy_6GHz, car, car_metal, mmMAGIC_low, mmMAGIC_high`; new output `loss_el [n_ut, n_tx]`; cumulative-weight rule. |
| W-20 | `c = 299792458` in the library (2L/c, soft-LOS wavelength). |
| W-21 | Probit: Acklam + one Halley step on `std::erfc`. |
| W-22 | `n_link = 0` valid, empty outputs. |
| W-23 | Positions and numeric arrays validated to be finite. |
| W-24 | Catch2 diagnostics via `qd_test_diag.hpp` area `QD_DIAG_GBSM` (the repository's current mechanism) instead of `QD_TEST_VERBOSE`. |
| W-25 | MEX/Python take the path-loss model from the `model` entry of `plpar`; missing optional outputs `[]`/`None`; complex `gr_eps` in Python with the `complex` flag. |
| W-26 | `SECTION_DESC` in `src/qd_scenpar.cpp` and the two `conf_file_read` wrappers, written in R4. |
| W-27 | Ground permittivity keeps QuaDRiGa's positive imaginary part. |
| W-28 | Clamped `hBS` in all LSP terms incl. `ES_D_mu_C/D`. |
| W-29 | Functions check only the scenpar keys they use. |
| W-30 | Slice 0 of `lsf_init_phases` = `sos_init_phases(n_coeff, 11, seed)` (documented contract). |
| W-31 | MEX/Python read `floor_id` as double and check for non-negative integers. |
| W-32 | `winner_los` keeps MATLAB's mean-height breakpoint over all links of the call. |
| W-33 | `LOS_state_lambda` in the 9 NLOS files, `IO_state_lambda` and `O2I_d2D_in_max` in the 8 O2I files. |
| W-34 | Writer sections/paragraphs of the 6 new keys. |
| W-35 | Legacy O2I distances with `d3D = 0`: indoor fraction 0. |
| W-36 | Random hE uses the model's d2D (after legacy O2I scaling). |
| W-37 | Column 9 of `rnd` checked only when `gr_eps` is requested. |
| W-38 | Release-note text into the R4 report. |
| W-39 | Internal helper header with generic generator-set evaluation (SSF reuse). |
| W-40 | Missing new `PL_` keys keep today's behaviour. |

## Issues with the plan

| # | Issue | Handling in the spec |
|---|---|---|
| I-1 | D21 bias removal in dual mobility breaks the exact ASD/ASA mirror relation (per-device corrections), so the planned swap-reciprocity test holds only without bias removal. | D21 kept; test without bias removal; self-mirrored generators tested with it. |
| I-2 | `set_scenario.m` line 367 uses `1.25e-6·d^2` for 36.873 UMa; TR 36.873 has `d^3` (from memory, TR 36.873 not in the references folder). | Library follows TR 36.873; Stephan to confirm or request a legacy flag. |
| I-3 | Plan: `DS_VS = 4.27` "reproduces today's value" of InF LOS. It does not (-7.2530 vs -7.2535; the file value is the mean of two lgDS values). | 47/11 used; InF LOS DS median changes by 7e-4 in lg. |
| I-4 | README.md has no release-notes section (WP-7). | Text goes into the R4 report. |
| I-5 | v19 path-loss constants are rounded (13.34, 20.944, RMa 0.0014, ...); TR oracles would need 0.03 dB tolerance. | Full precision in R3; v19 path loss changes by up to 0.02 dB. |
| I-6 | `gen_o2i_loss.m` draws the O2I deviation from a uniform variable. | Library uses N(0,1); not part of D23 parity. |
| I-7 | `get_pl.m` `nlos` adds the O2I loss twice when the LOS term wins. | Library never adds O2I; note for NG. |
| I-8 | One model name `mmMAGIC` cannot express MATLAB's low/high wall mix. | `mmMAGIC_low`, `mmMAGIC_high`, `high_v14`. |

## SSF requirements found

All provided by the six functions as specified: per-link, per-frequency DS/ASD/ASA/ESD/ESA, KF (dB by `10·lg`), SF
with `sf_sigma`, XPR mean and std (`lsp`, `lsp_sigma` slice 7), `lsp_vals` replacement (`lsp_mu`/`lsp_sigma` with
`rnd = zeros`), per-link ZSD mean for `c_ZSD = 3/8·10^mu` (`lsp_mu` slice 5), EoD offset, absolute TOA, ground
permittivity, soft LOS, cluster-number range keys, path loss. What changed the design: the SSF needs its own
spatially consistent cluster/ray generators with the same mirroring, floor, 2D/3D and dual-mobility rules, so the
LSF phase set stays fixed at 11 generators and R2 must write the evaluation and bias-removal helpers generic over a
generator table (`qd_gbsm_eval`, `qd_gbsm_remove_bias` in `src/quadriga_gbsm_helpers.hpp`). No public API addition
was needed for SSF; the only API changes against the plan are `loss_el` and the O2I model names (both delegated to
review point A by the plan).

## Not verified

- Acklam's probit coefficients are typed from memory; the test of 5.7 (`Phi(probit(p)) = p`) catches an error.
- TR 36.873 `g(d) = 1.25e-6·d^3·exp(-d/150)` (I-2) from memory; the TR is not in the references folder.
- The mmMAGIC O2I model is ported from `gen_o2i_loss.m`; the mmMAGIC deliverable D2.2 was not consulted.
- That the Jacobi root matches MATLAB `sqrtm` to 1e-12 relative (D23) is argued (unique SPD root), not measured.
- The exact-zero floor correlation assumes distinct horizontal frequencies of the presets; checked from the lattice
  construction (`qd_sos_directions`: upper hemisphere, golden-angle azimuths), not numerically.
- The full-precision constants were computed with Python double arithmetic (values in 4.4.2).
- Line numbers of the NG files were checked for `get_pl.m`, `gen_lsf_parameters.m`, `set_scenario.m`,
  `gen_o2i_loss.m`, `init_sos.m`, `qd_builder.m`; `qd_scenpar.cpp` line numbers refer to cebd0c0.
- Nothing was built or run.
