# Report: critical review of the implementation spec, round 3

Reviewed: `md/gbsm_lsf_spec.md` r2.1 (1990 lines), against TR 38.901 v19.5.0, QuaDRiGa-NG `quadriga_src`
(`@qd_builder`, `@qd_layout`, `@qd_sos`, `config`), the quadriga-lib sources (`include/`, `src/`, `api_mex/`,
`api_python/`, `tests/`, `data/config/`) and the three SPEC files. Findings R1-01 to R2-22 and their resolutions were
read first; none is re-raised. Numbers were recomputed with short Python scripts (numpy, mpmath) in the session
scratchpad: path-loss constants and breakpoint differences, LOS-probability values, the probit, the Newton-Schulz
iteration, and SOS statistics computed from the preset tables of `src/sos_preset.cpp` with the direction rule of
`qd_sos_directions`. Nothing was built.

Result: 15 findings, R3-01 to R3-15: 0 blocker, 1 major, 7 minor, 7 nit, no escalation.

## 1. Rationale of the findings

### Major

**R3-01 (5.3, random hE case).** Both parts of the case that involve the UMa NLOS file assume that its LOS term can be
seen in `pl`. At the stated points the NLOS term of Table 7.4.1-1 is larger:

| Point | PLn [dB] | LOS term [dB] |
|---|---|---|
| hBS 25, hUT 20, d2D 400 m, 0.05 GHz | 78.11 | hE 1: 61.37, hE 12: 72.91, hE 15: 78.56, hE 18: 87.62 |
| hUT 19.5, d2D 200 / 1000 m, 3.5 GHz, hE 18 | 103.55 / 130.86 | 89.51 / 110.46 |
| hUT 22.5, d2D 200 / 1000 m, 3.5 GHz, hE 21 | 101.75 / 129.06 | 89.50 / 114.83 |

"`pl` equals the TR LOS formula with that hE ... for the LOS term of the UMa NLOS file" fails for a correct
implementation at the floor-7/8 points. The hE identification through the NLOS file works only for hE 15 and 18 m in
the 0.05 GHz part. The case therefore does not test what the 1.1·hE fix (R2-02) changed for the `nlos` model. A
plpar copy with `Bn = -1000` makes the LOS term win and keeps the `nlos` code path with `hE_rand`. Calling
`dual_slope` with the NLOS plpar would test the other model.

### Minor

**R3-02 (negative default `sf_sigma`).** In 3.0.4 only `P_sigma + P_delta·lg(P_omega + f')` is limited at 0. This is
MATLAB's `sigma(sigma<0) = 0` in `get.lsp_vals`. The distance, height and elevation terms are added afterwards. The
v19 files have none of these terms. QuaDRiGa-NG has 11 files with `SF_beta`, all satellite or 5G-ALLSTAR files, and
more files with beta terms of other LSPs. Example: QuaDRiGa_NTN_Rural_LOS, `1.4 + 1·lg(alpha)`, is negative below
alpha = 0.0398 rad (2.3 deg). In that case the default `sf_sigma` of `path_loss` is negative, and the documented chain
throws at R1-25's check. The defect is in the formula, not in the check: a negative standard deviation only flips the
sign of the realization. That breaks the cross-correlation with the other LSPs. The cleaner fix is to limit the whole
std at 0.

**R3-03 (SF std in QuaDRiGa-NG's default mode).** In `get_channels.m`, the branch
`use_3GPP_baseline || isempty(h_builder.sos)` uses `[~,~,path_loss,scale_sf] = get_pl(...)`. The default branch uses
`path_gain = -get_pl(...)` and `[sf,kf] = get_sf_profile(...)`. `get_sf_profile` calls `gen_lsf_parameters` with
usage 3, which does not scale. `get_los_channels` is the only other user of `scale_sf`. So today the `sig`/`SF` keys of
a path-loss model act only in baseline mode. With user decision D16 (`sf_sigma` from `path_loss` into `lsp_generate`)
they act in every mode. That is the TR behaviour, but it is a change that 10.2 does not list. It also matters for the
parity reference of 10.4.

**R3-04, R3-05 (conf-file anchors).** Checked against the files: `absTOA_lambda` and the line "% L depends on the
deployment ..." are in section "Absolute time of arrival offset" in all five files. In SMa_NLOS(_O2I),
`% Break point of note 5 ... E = 2 pi / 0.3 = 20.944 and hE = 0.` remains after `PL_E`/`PL_hE` are removed. "13.34
(rounded)" appears only in the UMa files.

**R3-06 (5.4/5.5 inputs).** The covariance case cannot recover `x_KF` from the UMa NLOS file (`KF_sigma = 0`).
Clipping falsifies `x` for 16 % of the ASA values with the file's own mean and std at 6 GHz. Both are easy to fix,
but an agent would stop on a failing statistical test.

**R3-07 (ACF tolerance).** These are directional SOS ACF values in the x-y plane, computed from the tables at
lambda 10 m:

| Preset | 5 m x / y | 10 m x / y | 20 m x / y | Target 5 / 10 / 20 m |
|---|---|---|---|---|
| Comb256 | 0.778 / 0.774 | 0.408 / 0.393 | 0.134 / 0.128 | 0.779 / 0.368 / 0.135 |
| Exp256 | 0.582 / 0.572 | 0.377 / 0.388 | 0.138 / 0.158 | 0.607 / 0.368 / 0.135 |

A simulation of the test (100 x 100 grid, 50 m spacing, ±x and ±y offsets, 20 random phase sets) gave worst
deviations of 0.064 (Comb256) and 0.045 (Exp256). The check passes, but the stated 0.03 budget is wrong.

**R3-08 (SSF interface of the generic evaluation).** See section 3.

### Nits

R3-09 to R3-15 are listed in the table. R3-14(b) was recomputed: coupled Newton-Schulz on random 8x8 correlation
matrices needs 13, 24, 35 and 41 iterations for smallest eigenvalues 2.8e-2, 2.9e-6, 2.9e-10 and 2.9e-12. At about
5e-17 it stops at the limit of 100, with a residual of 1.5e-8. The 0.999999 matrix takes 25 iterations, residual
2.4e-14. The spec's statements match; only the "factor 2.25" sentence is wrong.

## 2. Checked and confirmed (no finding)

- **Path-loss constants (4.4.2):** RMa `A1 = 20.477913900145722`, `B = 31.74083179916828`,
  `D = 0.0013979400086720378`, `Bn = 31.754962063305896`; SMa `A1 = 21.57442238074932`, `B = 30.132619360949676`,
  `D = 0.002`; SMa NLOS `Bn = 36.15`, `D2n = 370`. `E = 40/3 = 13.333333333333334` and
  `20·pi/3 = 20.943951023931955`. The "up to 0.02 dB" change of the files comes mostly from RMa `PL_D` at 10 km.
- **3D-breakpoint difference (R1-02):** RMa maximum 0.0341 dB (hBS 33.4 m, hUT 1 m, 0.5 GHz), SMa 0.0313 dB, at the
  5.3 test points 4.1e-3 dB. All three confirmed. `dual_slope` equals Table 7.4.1-1 for UMa and UMi; the `nlos`
  parameters reproduce the RMa/SMa NLOS formula term by term.
- **Random hE (R2-02, R2-10):** the set size `n`, `p1`, the `rnd_he` choice that gives hE 18 and 21 m, d'BP 490 and
  280 m, and the 3.7 dB step (3.707 dB) are confirmed. The `get_pl.m` behaviour is confirmed: C1 in both branches
  (lines 233, 241), mean heights per call, the `dual_slope` clamps, the `nlos` recursion that adds O2I twice, and
  `sf_sigma` averaged per track.
- **LOS probability (3.5):** all formulas match Table 7.4.2-1. UMa gives 1.00246 and 1.00470 at 18.1 m and 0.97779 at
  20 m; the excess ends at 18.29 m (maximum 1.0038) and 18.47 m (1.0060). The InF defaults match Table 7.8-7 by
  sub-scenario. The 38.811 and QuaDRiGa_Industrial formulas match `set_scenario.m`.
- **Soft LOS and probit (R2-03):** symmetric Acklam with Halley, recomputed with mpmath: worst error
  3.1e-16·max(1, abs(F)) over [1e-12, 1 - 1e-12]. The tolerance chain of the 5.5 cases is consistent; at fc 1 kHz
  `k` is 8.2e-3.
- **Dual-mobility coupling (R2-01):** Cov of the two-map pair is (A_g + A_m)/2, the variance is 1, and x_g = x_m at
  D = 0. Swap reciprocity is bit-exact, because `sos_val` documents that a value does not depend on the other
  positions or generators of a call. The test tolerances hold (SOS deviation of the pair correlation at most 0.023).
- **Floor offsets:** zero cross-correlation at zero lag within a block. Recomputed residuals at 0 to 60 m: 0.12 to
  0.16 for 256 sinusoids, 0.04 to 0.10 for 1024. The 5.2 floor and marginal cases pass in 20 random realizations
  (floors < 0.03 against 0.05; (0, 256) 0.052 against 0.25; mean 0.010; std 0.017; uniform 0.003 and 0.006).
- **Bias removal over the union:** consistent with `sos_remove_bias` (one-device mode, offset per column, phases in
  [-pi, pi)) and with the mirror relation. The documented properties hold. The QuaDRiGa difference was confirmed in
  `init_sos.m`: `remove_bias` runs before the mirroring, and MATLAB's ASD generator holds [ASA_orig, ASD_orig]
  phases.
- **InF V/S and absTOA_Lmax:** halls SL/DH 120x60x10 (V/S 4) and DL/SH 300x150x10 (50/11). `DS_VS` 4.2727272727272725
  is exactly 47/11. Median -7.25277 against today's -7.25347. The 2L/c values are 0.8006 µs and 2.0014 µs, and the
  InF exceedance starts at x > 3.508. The Indoor office is 120 x 50 x 3 m (Table 7.2-2). TR 7.6.9 restricts the
  bound to InF/InH NLOS.
- **o2i_loss:** Tables 7.4.3-1 to -3 (IRR glass v19 25.4 + 0.11 f), the car model, the correlation distances of
  7.6.3.3, and the model-choice rule including zero weights.
- **TR order and roots:** the Cholesky permutation [SF, K, DS, ASD, ASA, ZSD, ZSA] with x_SF = s_SF and Q·Q^T = R,
  and the clipping at 104/52 deg (step 4).
- **Ground permittivity and EoD offset:** MATLAB `gen_lsf_parameters.m` lines 319-344 and Tables 7.5-7/7.5-12
  (including the SMa geometric term); the sign note on eq. 7.6-40 is correct.
- **Legacy O2I distance (3.3.2):** matches `gen_o2i_loss.m` (min outdoor distance 10 m; NaN -> Inf -> d3).
- **Code anchors:** the `quadriga_tools.hpp` block (lines 517-567), the key table (165 keys, comment at line 53), the
  writer sorting (paragraph, matrix position, table order), the section/paragraph numbers of the 6 keys, the
  `NF_alpha`/`NF_beta` check, `qd_conf_pl_comments`, the SOS helpers (`qd_sos_rng`, `qd_sos_uniform`,
  `qd_sos_acf_interp` with constant ends, `qd_sos_preset_lookup`), the MEX and Python adapter helpers, the
  `python_main.cpp`/`CMakeLists.txt` lines, the source globs (new `src/*.cpp`, `api_mex/*.cpp` and test files are
  picked up automatically), `qd_test_diag.hpp`, and the existing `NAN` defaults in public headers (MSVC precedent).
  Every user of the conf functions is listed in R1, step 1.
- **Standalone (guidance 1):** no D/W/I/R IDs, plan references or local paths in normative text; the SPEC files are
  allowed by R2-12. QuaDRiGa-NG line numbers occur only as source pointers and are excluded from produced files by
  9.1.
- **Not raised (user decisions):** `Comb256` default (D13/R2-11); bias removal on by default (D14). D14 departs from
  the TR's unconditioned field statistics; the variance loss is documented in `sos_remove_bias` and in 3.1.

## 3. SSF readiness (independent verification)

Sources: `gen_ssf_parameters.m`, `gen_fbs_lbs.m`, `get_channels.m`, `add_sdc.m`, `get_los_channels.m`, `init_sos.m`;
TR 7.5 steps 5-12, 7.6.2, 7.6.3.1/7.6.3.3, 7.6.5/7.6.5.1, 7.6.8, 7.6.9, 7.6.13-7.6.16.

**What the SSF code reads from the LSF part, and where the library provides it:**
- DS, ASD, ASA, ESD, ESA and KF per link and frequency (`gen_ssf_parameters` lines 243-248): `lsp` slices.
- SF for step 12: `lsp` slice 2 with `sf_sigma`.
- XPR mean and std: `lsp` slice 7 with `lsp_sigma`, or `lsp_mu`/`lsp_sigma` slice 7 for the TR variant. R3-12 adds
  the 0.1 factor of QuaDRiGa's default mode.
- Ground permittivity (lines 112, 262-265): `gr_eps_re`/`gr_eps_im`; the sign note is correct.
- Absolute TOA (lines 813-818): `abs_toa`.
- `add_sdc` uses `mean(lsp_vals(4..7,1,:))` and `get_pl` at custom positions: covered by `lsp_mu` (R2-07) and
  `path_loss`.
- `gen_fbs_lbs` reads only SSF outputs and `scenpar`.
- TR steps 5-7 need K in dB (`10·lg` of slice 1), the ZOD offset (`eod_offset_deg`) and the per-link ZSD mean for
  c_ZSD (`lsp_mu` slice 5, after the ESD floor).
- Step 9 needs XPR mu and sigma per frequency, and step 11 the K-factor and Δτ.
- 7.6.13 to 7.6.16 need only keys and SSF draws.

**Result:** the LSF API has no gap. This confirms the claim of rounds 1 and 2.

**Per-frequency handling:** consistent with TR 7.6.5.
- One `x` for all frequencies.
- The propagation condition is frequency-independent (`p_los`); `los_soft` is per frequency, as the TR allows.
- PL, SF std, LSPs, XPR, gr_eps and the EoD offset are per frequency (`f'` clamp); O2I loss is per frequency with
  one deviation per UT.
- Δτ is per link.
- The anchor frequency of 7.6.5.1 is handled by appending it to `fc`, with clipping.
- Per-cluster shadowing Z_n and steps 8-11 need independent generators per frequency (`xpr_sos`/`pin_sos` are
  `[clusters, M(+1), n_freq]`). These are SSF generator sets; the generic machinery takes any number of generators,
  grouped by frequency table (R2-21).

**Extensibility of the generator machinery:** the `qd_gbsm_gen` table covers everything the MATLAB SSF sets use:
- per generator: decorrelation distance, Normal or Uniform, mirror partner, bias flag;
- mirror pairs inside a set (`path_sos` columns 2/3 and 4/5);
- self-mirrored generators in same-map mode;
- two-map pairs, floors, 2D/3D, and the union bias removal.

Two gaps in how it is specified (R3-08):
- The functions must be movable unchanged, but 3.0.5/9.1 allow capturing lambdas. Their signatures should be fixed
  in the spec.
- The seed rule of section 8 assumes that the SSF sets use the LSF preset.

**What the builder must store** (10.2, 10.3 and the above):
- `lsf_phase` `[n_coeff, 11, 2]`;
- an explicit seed (or the SSF phase cubes);
- per UT, from the layout: floor index, `d2d_in` (legacy mode), O2I loss per frequency, LOS state (and the soft-LOS
  inputs);
- `rnd_he` per segment, if steps along a segment are unwanted;
- the simulation settings used at init (`preset`, `horizontal`, `dual_mobility`): `lsf_draw` cannot detect a different
  preset with the same `n_coeff`. They live in `simpar` today, so no finding.

The SF profile normalization `sf · sf_init_builder / sf_init_sos` in `get_channels` remains NG logic.

## 4. Implementability per run

| Run | Verdict | What would make the agent stop or guess |
|---|---|---|
| R1 | ready after R3-04 | Section name of `absTOA_Lmax` (R3-04); what to do with the decision labels in the conf tests it edits (R3-10); writer comment check (R3-11). Large but fits: the header move and the Python move are mechanical, and every include site is named. |
| R2a | ready after R3-08 | Form and signature of the generic evaluation and bias removal (R3-08). Tolerance comment of the ACF case (R3-07: the case passes, but the stated budget does not match what the agent will measure). Seed check (R3-09). |
| R2b | ready after R3-06 (a, b) | Covariance case inputs (KF sigma 0, clipping) and spot-check geometry (R3-06). Optional: the std limit (R3-02) would add one formula change and a test here. |
| R3a | ready after R3-01, R3-05 | The NLOS part of the random-hE case fails as written (R3-01); comment anchors (R3-05). The largest run (21 conf files, TR oracle for 23 files), but it fits one context. |
| R3b | ready (R3-06 c) | SMa test row needs hB. |
| R4a | ready (R3-13) | Allocation of outputs beyond nlhs for `lsp_generate`. |
| R4b | ready (R3-13) | `gr_eps` is always computed in Python. |

Sizes, validation messages and orders, edge cases (`n_link = 0`, pre-sized outputs, nullptr outputs), oracles and
acceptance criteria are otherwise unambiguous. The tolerances are justified and hold where I recomputed them. The
read lists cover the sections each run needs. R4a also needs 4.1 for the key names in `SECTION_DESC`, which it can
read from the code.

## 5. Overall judgement

**Ready after fixes.** There is no blocker and no contradiction with a user decision. All findings are local text
edits:
- one test specification that fails for a correct implementation (R3-01);
- one formula edge in NG files (R3-02);
- one hand-over gap (R3-03);
- conf-file anchors (R3-04, R3-05);
- under-specified test inputs (R3-06);
- one wrong tolerance rationale (R3-07);
- the SSF movability interface (R3-08);
- nits.

After R3-01, R3-04 to R3-06 and R3-08 are resolved, each run can be handed to its agent without further questions.
R3-02 and R3-03 can be fixed in the same pass, since they only add text.
