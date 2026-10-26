# Report: spec revision r0 -> r1

Task: `md/prompts/spec_revise_r1.md`. Revised `md/gbsm_lsf_spec.md` (r0, 1656 lines -> r1, about 1950 lines) for the
findings R1-01 to R1-30 of `md/gbsm_lsf_findings.md`; filled the resolution column of all 30 rows. Nothing else in
the repository was changed; nothing was built. Two checks were run in the scratchpad: the RMa/SMa breakpoint
difference over the TR ranges (Python), and the conf files of `data/config` and QuaDRiGa-NG for multi-value `PL_`
keys (none) and for the WINNER LOS `C1`/`C2`/`F1`/`F2` values.

Counts: **28 fixed, 0 rejected, 2 pending user** (R1-09, R1-10). Two fixes take only the required part of a
finding and leave its optional part (R1-12, R1-19); one fix resolves the finding differently from the proposal
(R1-02). Details below.

## 1. Changes per finding

| Finding | Change in the spec |
|---|---|
| R1-01 | 3.0.1 precision rule: `lsf_draw` calls `sos_val<double>` with `use_kernel = 0` (AVX2, single-precision sin/cos); everything else pure double. 5.0 tolerance rules: Normal 2e-5 at 256 sinusoids scaled with `sqrt(n_coeff/256)` (1.5e-5 / 4e-5 for 128 / 1024), 3e-5 for the coupled dual-mobility columns 3..6 (two kernel errors, coupling gain up to sqrt 2), Uniform 1e-5, zero means through `lsf_draw` 2e-5; `ref_sos` phase checks 1e-12; same-evaluation checks exact. 5.1/5.2 rows use them; 9.1 rule; W-44. |
| R1-02 | Different resolution: the library follows the TR. New key `PL_BP2D` (dual_slope and the LOS term of nlos): non-zero gives `PL1(dBP) + A2·lg(d3D/dBP)` with the 2D breakpoint (TR RMa/SMa LOS). Set in RMa_LOS, _LOS_O2I, _NLOS, _NLOS_O2I, SMa_LOS, _LOS_O2I (R3a). TR oracle stays 1e-9 dB; the legacy-model test checks both forms. W-41, 3.3.4, 4.3 (writer comment row), 4.4.2 (file lines and comment), 5.3, 5.8. Reason: over the TR ranges (RMa hBS 10-150 m, hUT 1-10 m, 0.5-30 GHz; SMa hBS 25-35 m, hUT 1-14 m, 0.5-37 GHz) the 3D form differs from the TR by up to 0.034 dB (RMa) / 0.031 dB (SMa), not < 5e-3 dB (that holds only at the test points). The same order of error was the reason for W-12 (full-precision constants), and the user's rule is to follow the TR unless QuaDRiGa has a statistical reason; it has none here. Cost: one key, one branch, 6 file lines. |
| R1-03 | Header comments without D-IDs (2.1). 5.0: no local path; oracles described by content ("TR formula typed", "typed offset rule"); test files carry no internal IDs, spec section numbers, `md/` or local paths, NG line numbers. 9.1: the same rule for every produced source, header, test, wrapper, conf file and documentation block. Oracle columns of 5.x cleaned (W-15, 3.2.4, 3.3.2, 3.0.4, 3.6.2 references replaced). |
| R1-04 | 4.1: writer DS overview line `log10( a * V/S + b ) - 9.35 + Xds * ( ... )` when `DS_VS > 0`, doc item. 5.8 (R1): the 6 keys in the `QD_NONNEG` lists of both value-check cases; writer case for section/paragraph and not-at-default; DS overview case next to the ESD case; cross-key rule through `scenpar_validate`, `conf_file_read` (file and line of `DS_VS`) and `conf_file_write`; MEX/Python error tests. 4.3 + 5.8 (R3a): comment check per new `PL_` key (6 keys incl. `BP2D`). |
| R1-05 | Lean option: bias removal treats `rx_pos` as one field (floor 0); multi-floor drops pass one floor or switch it off (W-43, 3.1 properties, header argument comment). Reason: one phase offset per generator cannot zero the mean of several floor fields at once; a union mean over floors is a weaker property not worth 25 lines. |
| R1-06 | Section 7: R1, R2a (header, `lsf_init_phases`, `lsf_draw`), R2b (`lsp_generate`, LSP mean/std, Cholesky + `qd_scenpar_xcorr` refactor), R3a (`path_loss`, PL keys, 4.4.2), R3b (`los_probability`, `o2i_loss`), R4a (MEX, SECTION_DESC, See also), R4b (Python, release notes), each with read list, files, steps, acceptance. Run names updated throughout (4.3, 4.4.2, 5.0, 5.8, 6.2, 9.2, 10). |
| R1-07 | `winner_los` uses `C2` beyond the breakpoint (W-42, 3.3.4; continuity at the breakpoint for `F1 = F2`, which holds in all 5 WINNER LOS files); parity of the WINNER LOS files only before the breakpoint (10, 15d); listed under the D2 departures (1.2). |
| R1-08 | 3.0.5: header with documentation block and only cross-file helpers (`qd_gbsm_gen`, `qd_gbsm_lsf_generators`, `qd_gbsm_freq`, `qd_gbsm_positions`, `qd_gbsm_lsp_mu_sigma`, `qd_gbsm_chol8`); `qd_scenpar_xcorr` calls `qd_gbsm_chol8` (R2b). Evaluation, bias removal, floor offsets, legacy root, probit file-local, the first two generic over the generator table for the SSF port. `qd_gbsm_scenpar` dropped (direct `scenpar_validate`). 5.7: no internal test file, coverage table instead. Section 8 updated. |
| R1-11 | Adopted. 3.1: one `sos_remove_bias` per Normal field over `U = R` (single) or `[R; T]` (dual), then mirroring; one code path. 3.2: each field evaluated once per device in one-device mode, the Tx side of g with `F_m(g)` and slice-1 phases (identical bits to the partner's field), combined in the library; this makes the swap reciprocity exact bit for bit (also for unequal distances and with bias removal) and avoids comparing scaled frequency tables, which are not bitwise equal. W-2, W-5 rewritten; I-1 resolved; 5.1 and 5.2 test with bias-removed phases. Trade-off stated: in dual mobility the ASD/ASA, ESD/ESA link means are only approximately zero. |
| R1-12 | `36.873_UMi`, `36.873_UMa`, `indoor_ut` dropped (2.1, 3.5.1/3.5.2 with the equivalence shown, 3.5.5, 5.5, 6, 10); I-2 resolved without a legacy flag (bug of `set_scenario.m`). Optional part not taken: `LOS`/`NLOS` and the 38.811 tables stay (W-47): uniform group table in NG, 2 and about 20 lines, D7 named models. |
| R1-13 | `o2i_total_distance` removed; a given `d3d_in` selects the legacy outdoor distance (W-46, 2.1, 3.3.2, 3.3.7, 6.1, 6.2, 10). Note for the R1-10 option (input becomes `d2d_in`). |
| R1-14 | 3.0.3: `scenpar_validate(copy, "legacy", ..., true)`; key checks removed from 3.1, 3.2, 3.3.1, 3.4.1 (the `DS_VS` cross-key rule now comes from the key table); tests check pass-through (W-29). |
| R1-15 | Per-link heights for the `winner_los` breakpoint (W-32, 3.3.4, 5.3, 10). |
| R1-16 | 2.3 and 3.2.4: exact zero only at zero horizontal lag, residuals at other lags stated; `psi_b` from `qd_sos_rng(b)` / `qd_sos_uniform`; header comment "uncorrelated at equal horizontal positions"; W-1. Method kept. |
| R1-17 | 3.1 "Difference to QuaDRiGa" (single mobility affected when the pair distances differ, mechanism from `init_sos.m`); 10 list of non-switchable changes; release-note item (R4b). |
| R1-18 | 10 (15d) split: `lsp_generate`/`path_loss` with MATLAB's SOS values as `rnd` at 1e-12 relative; `lsf_draw` vs `val` at 1e-5 absolute; phases of column 11 (hE) any, unused. |
| R1-19 | 5.4: formula mechanics with custom keys (all terms, clamps, ESD terms and floor, V/S, EoD) plus TR spot checks for UMa NLOS, UMi LOS, SMa NLOS, InF NLOS DL. Optional part not taken: `test_conf_files_v19.cpp` keeps its key-level check independent of `lsp_generate` (an end-to-end copy would be a second oracle of the same tables). |
| R1-20 | 1.1 layout order and d2D-out clamp; 3.5.5 doc items incl. the InF `LOS_state_lambda` note; R4a SECTION_DESC; 10 (15c). |
| R1-21 | 5.3 oracle text: O2I SF of Table 7.5-6 (7/8 dB = `SF_sigma`). |
| R1-22 | Section 8: rows for 7.6.5.1 (anchor added to `fc`; `clip_spreads` applies because 7.6.5.1 takes the spreads "according to Step 4"), soft LOS (LOS and NLOS builder, two `lsf_init_phases` with different seeds), `lsp_vals` not 1:1 (reference geometry). |
| R1-23 | `qd_sos_acf_interp` for A_g(D), satellite, 38.811 and legacy d2D-in tables; `qd_sos_rng`/`qd_sos_uniform` for floor offsets; additionally `qd_sos_preset_lookup` for the preset check (W-45); 9.1 reuse rule. |
| R1-24 | Probit file-local in `los_probability.cpp`, target `1e-10·max(1, abs(F))`; 5.5 test through `los_soft` (RMa model as p generator, `rnd_g = 0`, fc 1 kHz, bisection probit of the returned `p_los`). |
| R1-25 | `sf_sigma` finite and >= 0 (3.4.1, 3.4.4, 5.4). |
| R1-26 | 4.1 key-table comment describes the content. |
| R1-27 | R1 creates the header skeleton; each run adds the declarations it defines (1.6, 2.1, 7). |
| R1-28 | 3.1 output wording; 5.1 each `ref_sos` with the generator's own lambda; `n_link = 0` skips all SOS calls (3.1, 3.2 step 0, tests); preset message names `preset` (W-45). |
| R1-29 | 3.4.2 step 7 and 3.4.4: TR eq. (7.6-40) has the opposite sign; take the conjugate. |
| R1-30 | One value per `plpar` entry a model reads (`Path-loss parameter '<key>' must have one value.`; no conf file in `data/config` or NG has multi-value `PL_` keys); D27 comment shortened. |

Other edits: spec header (status r1, marker for pending decisions, no-internal-reference rule), 1.2 list of the
departures from decision wording with reasons (D2, D4, D7, D14, D21), W-list rows W-1, W-2, W-5, W-9, W-12, W-17,
W-21, W-29, W-32, W-39, W-40 updated and W-41 to W-47 added, I-1, I-2, I-5, I-8 updated, section 12 change log.

## 2. Rejected

None. Partial: the optional parts of R1-12 (drop `LOS`/`NLOS`, move 38.811 to NG) and R1-19 (let
`test_conf_files_v19.cpp` call `lsp_generate`) were not taken, reasons above. R1-02 is fixed by following the TR
instead of widening the oracle.

## 3. Escalations (pending user decision)

### R1-09 — keep or drop the legacy symmetric root (`lsp_generate` argument `cholesky`)

- **A, drop (recommended):** the library implements only the TR rule (Cholesky in TR order). NG's legacy mode
  computes `x = sqrtm(R)·s` with MATLAB's own `sqrtm` and calls `lsp_generate` with a scenpar whose 28 correlation
  keys are 0 (identity root) and `rnd(:, 0..7) = x`. Gains: one API argument less (C++, MEX, Python), about 45
  lines of Jacobi plus about 40 test lines less, and better D23 parity (MATLAB's own algorithm instead of a second
  implementation). Costs: about 3 lines of legacy code in NG; a Python/C++ user who wants QuaDRiGa's legacy
  correlation must compute the root himself (e.g. `scipy.linalg.sqrtm`).
- **B, keep:** one call gives the legacy LSPs in every API. If kept, replace Jacobi by the coupled Newton-Schulz
  iteration on `A / tr(A)` (about 15 lines, matrix products only; converges for SPD input since all eigenvalues of
  `A / tr(A)` lie in (0, 1)); the spec text for this is ready in 3.4.3.
- Local edits for A are listed in 3.4.3 (delete 3.4.3 and the `cholesky = false` bullet, the argument in 2.1, 6.1,
  6.2, two test rows of 5.4, one MEX test point, one row of the switch table, a sentence in 10 (15d)).

### R1-10 — scope of `o2i_loss`

- **A, keep the r0 scope (D25):** 9 models incl. `high_v14`, `mmMAGIC_low/high` with `loss_el`, 4 or 6 `rnd`
  columns, the legacy d2D-in table (Appendix 11.3) and the geometry inputs/outputs `tx_pos`, `rx_pos`, `d3d_in`.
  `gen_o2i_loss.m` can be deleted entirely after the integration, as D25 asked. Costs: 6 arguments/outputs, 3
  non-TR models, one appendix, about 120 lines plus tests, none of which gives parity with today's values (the O2I
  deviation of MATLAB is uniform, I-6, so O2I losses are not reproducible anyway).
- **B, reduce to the TR (recommended):** models `low`, `high`, `low_A`, `legacy_6GHz` (Table 7.4.3-3, part of
  38.901), `car`, `car_metal`; `rnd` always `[n_ut, 4]`; no `tx_pos`, `rx_pos`, `d3d_in`, `loss_el`,
  `d2d_in_min_of_two`. The d3D-in geometry is needed only by the legacy path-loss distance and moves to `path_loss`
  (optional per-link `d2d_in` instead of `d3d_in`, with the 10 m minimum outdoor distance) or to NG. Gains: lean
  API, every model of the library is a 38.901 model. Costs: D25's mmMAGIC and geometry parts move to NG (or are
  dropped there), so `gen_o2i_loss.m` is not fully retired if mmMAGIC is still wanted.
- Local edits for B are listed in 3.6.0 (and marked at 2.1, 2.3, 3.3.2, 5.6, 6, 8, 10, 11.3).

## 4. Notes for the next review round

- The swap reciprocity is now claimed exact (bit for bit) because swapped calls repeat the same one-device
  `sos_val` evaluations in swapped roles and the combination uses commutative two-term sums; this assumes that the
  kernel result for a position does not depend on the other positions of the call (the "a link does not depend on
  the other links" test checks the same property).
- The new key `PL_BP2D` is a plpar key (open map), so it needs no key-table entry; it is written by the writer like
  every plpar entry, with the new comment row.
- The 0.034 dB maximum of the RMa/SMa breakpoint difference is at low frequency and small hUT (0.5 GHz, hUT 1 m,
  dBP about 300 m); at the test points of 5.3 it is 4.1e-3 dB (RMa) and 9e-5 dB (SMa).
