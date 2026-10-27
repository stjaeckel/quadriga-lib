# Hand-off: integrating the large-scale model of quadriga-lib into QuaDRiGa-NG

For the agent that integrates the library into QuaDRiGa-NG (NG, `/home/stephan/sjc/QuaDRiGa-NG`, MATLAB/Octave).
NG line numbers below refer to NG commit 6253027; check them before editing.

## 1. Goal, scope, library

Goal: NG replaces its MATLAB implementation of the large-scale model (path loss, shadow fading, the 8 LSPs, the
absolute TOA, ground permittivity, EoD offset, LOS probability, O2I loss) by calls of quadriga-lib, with TR 38.901
v19.5.0 behaviour by default and today's QuaDRiGa behaviour reachable through legacy switches. The small-scale
fading (SSF) stays in MATLAB for now (section 7).

Library functions (C++ header `include/quadriga_gbsm.hpp`, MEX `quadriga_lib.<name>`, Python `quadriga_lib.gbsm`):

| Function | Purpose |
|---|---|
| `conf_file_read` | read a conf file: `scenpar` (171 numeric keys + `SubpathMethod`), `plpar` (`PL_` keys, field `model`), LSP correlation matrix |
| `scenpar_validate` | complete a parameter set with defaults and validate it |
| `conf_file_list`, `conf_file_write` | find and write conf files |
| `lsf_init_phases` | random phases of the 11 LSF generators of one BS site and scenario, `[n_coeff, 11, 2]`, optional bias removal |
| `lsf_draw` | spatially correlated normalized values per link, `[n_link, 11]`: 0 DS, 1 KF, 2 SF, 3 ASD, 4 ASA, 5 ESD, 6 ESA, 7 XPR, 8 abs TOA (Normal), 9 ground permittivity, 10 UMa hE (Uniform); MATLAB columns are +1 |
| `path_loss` | path loss and SF std per link and frequency, 10 parametric models of the conf files |
| `lsp_generate` | final LSPs `[n_link, n_freq, 8]` (DS [s], KF, SF linear, AS/ES [deg], XPR linear), abs TOA, ground permittivity, EoD offset, `lsp_mu`, `lsp_sigma` |
| `los_probability` | LOS probability of 18 named models, optional soft LOS state (eq. 7.6-18) |
| `o2i_loss` | O2I penetration loss and indoor distance d2D-in of the 6 models of TR clause 7.4.3 |

MEX usage lines:
```
phase = quadriga_lib.lsf_init_phases( scenpar, seed, tx_pos, rx_pos, dual_mobility, remove_bias, preset, horizontal );
rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, floor_id, dual_mobility, preset, horizontal );
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, d2d_in, rnd_he );
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( scenpar, fc, tx_pos, rx_pos, rnd, sf_sigma, cholesky, clip_spreads );
[ p_los, los_soft ] = quadriga_lib.los_probability( model, d2d_out, h_bs, h_ut, d_clutter, r_clutter, h_clutter, rnd_g, fc );
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, d2d_in_max );
```

Data flow per drop (layout): `o2i_loss` for the indoor UTs (loss, d2D-in; random inputs from `sos_rand` /
`sos_randn` at the TR clause 7.6.3.3 distances: building type 50 m, d2D-in 25 m, sigma_P 10 m) → `d2d_out =
max(d2D - d2d_in, 0)` (outdoor UTs: d2d_in = 0) → `los_probability` → LOS-state draw by the caller at
`LOS_state_lambda`, indoor state at `IO_state_lambda`.

Data flow per BS site and scenario (builder): `conf_file_read` once per file → `lsf_init_phases` once per builder
(store `phase`) → `lsf_draw` at the links (drop positions and every track position) → `path_loss` (with
`rnd_he = rnd(:, 11)`, and `d2d_in` only in legacy mode) → `lsp_generate` (with `sf_sigma` in TR mode).
Every model function is deterministic; randomness enters only through the seed and the normalized values.

Documentation: the documentation blocks in `src/qd_gbsm_phases.cpp`, `src/path_loss.cpp`, `src/lsp_generate.cpp`,
`src/los_probability.cpp`, `src/o2i_loss.cpp`, `src/qd_scenpar.cpp` (section "Geometry-based stochastic model"),
the MEX blocks in `api_mex/<name>.cpp`, `html_docu/` after `make documentation`. The 23 v19 conf files are in
`data/config/3GPP_38.901_v19_*.conf`; their comments state the TR source and the encoding of every value.

## 2. Library update in NG

- Copy the new library as in earlier ports (`external/`, the MEX package `+quadriga_lib`); copy the 23 v19 conf
  files to `quadriga_src/config`.
- Same commit: NG calls `quadriga_lib.interp( x, 0, data, xi )`, which the library now rejects (`Input 'yI' is
  required when 'y' is given (2D case).`, a fix in `main`, not part of this port). Pass `[]` instead of `0` in
  `@qd_builder/get_pl.m:469` (satellite branch; moot once `get_pl` wraps `path_loss`) and
  `@qd_track/interpolate.m:241`, `:266`. Without this 29 of 129 NG unit tests fail in Octave; with it 28 pass. The
  29th (`testLayout_Closed_Chan_Generation`) looks up its conf file relative to the current folder (pre-existing);
  `testBuilder_parameter_generation` depends on the test order with the old and the new library alike.

## 3. Builder

Store in the builder (replacing the 8 + 2 `qd_sos` objects in `sos`, `absTOA_sos`, `gr_sos`): `lsf_phase`
`[n_coeff, 11, 2]`; an explicit seed (drawn once, e.g. `floor(rand·2^52)`, if the user gives none; needed for the
SSF sets, section 7); `preset` and `horizontal` used at init (a preset change with the same `n_coeff` is not
detected); the hE value `rnd(:, 11)` (or all of `rnd`) per rx position, so `get_pl()` and `get.pow` need no new
draw; `d2d_in` per rx position in legacy mode; `dual_mobility` (never -1 at call time); later `floor_id` per rx
position. `lsp_mu` / `lsp_sigma` are stored or recomputed with `rnd = zeros(n, 11)`. `lsf_draw` reads the
decorrelation distances from `scenpar` at each call: changing them after `lsf_init_phases` changes the fields and
leaves the bias removal stale (document, or store the lambdas).

Call-site mapping (NG function → library calls; positions transposed to rows):

| NG function | Library calls |
|---|---|
| `init_sos` (LSF part), `absTOA_sos`, `gr_sos` | `lsf_init_phases( scenpar, seed, tx.', rx_bias.', dual, remove_bias, preset, horizontal )` |
| `gen_lsf_parameters` (drop), `get_sf_profile` (tracks) | `rnd = lsf_draw(...)`; TR mode `[pl, sfs] = path_loss( plpar, scenpar, fc, tx.', rx.', [], rnd(:,11) )`; `lsp_generate( ..., rnd, sfs_or_[], cholesky, clip )`; `ds = lsp(:,:,1).'` ..., `gr_epsilon_r = complex(eps_re, eps_im).'`, `absTOA_offset = abs_toa.'` (only if `absTOA_mu > -30`, as today); fill only empty properties, as today |
| `get_lsp_map` | `lsf_draw` (no `floor_id`) + `lsp_generate` on the grid; pass `tx_position(:, 1)` (after `check_dual_mobility` it is `[3, N]` and would take the dual-mobility branch, pre-existing) |
| `lsp_vals` (property) | per-link `lsp_mu` / `lsp_sigma` of `lsp_generate( ..., zeros(n,11) )`, or their mean over the builder's links where one value is needed (not 1:1: no geometry reproduces `lsp_vals` for files with distance, height or elevation terms) |
| `get_pl` | wrapper of `path_loss( plpar, scenpar, fc, tx.', rx.', d2d_in, rnd_he )`; `[pl; pl_init]` by appending the initial position as an extra row; adds the O2I loss of the track once (today's `nlos` branch adds it twice when the LOS term wins; `path_loss` never adds O2I loss) |
| `get_pl` `scale_sf` (`get_channels` baseline, `get_los_channels`) | dropped; SF from `lsp_generate` (with `sf_sigma` in TR mode) |
| `get_channels`, `get_channels_seg_unmerged`, `get_los_channels`, `check_los`, `gen_lsf_from_ssf` | the wrappers above |
| `add_sdc` | `path_loss` on its synthetic positions; `lsp_mu`/`lsp_sigma` means for the four `lsp_vals` means and the XPR mu/sigma (`add_sdc.m:424`, `:431-447`, `:580-590`; `h_bld.sos = []` becomes the phase cube) |
| `gen_ssf_parameters` (LSF inputs) | builder `ds ... xpr`, `gr_epsilon_r`, `absTOA_offset`; XPR mu/sigma from `lsp_mu`/`lsp_sigma` slice 8 |
| `copy`, `split_rx`, `split_multi_freq`, `add_paths` | copy `lsf_phase`, seed, `preset`, `horizontal`, hE, `d2d_in`; per-frequency builders call `lsp_generate` with their one `fc` (same `x`); `add_paths` re-inits on the assembled positions |
| other sites that store, copy or clear the LSF state | `@qd_layout/power_map.m:185`, `:191` (copies `.sos` of a template builder), `get_channels_seg_unmerged.m:38`, `qd_plan_mimo.m:114`, `:248`, `gen_lsf_from_ssf.m:63`, `qd_builder.m:83`, `:415-424` (`loadobj`: migrate saved `qd_sos` objects with `phase(:, g, 1) = sos(g).sos_phase(:, 1)`, slice 2 by mirroring) |
| `qd_layout.set_scenario` | `los_probability( model, max(d2D - d2d_in, 0), h_bs, h_ut, d_clutter, r_clutter, h_clutter )`; LOS draw, the 38.811 "below horizon" handling and the renormalisation of the LOS-state draw stay in NG (legacy option) |
| `qd_layout.gen_o2i_loss` | `[loss, d2d_in] = o2i_loss( fc, rnd, models, weights, O2I_d2D_in_max )`; layout stores `d2d_in` per UT |
| `qd_layout.power_map` | `lsf_init_phases`/`lsf_draw`/`path_loss`/`lsp_generate` on the grid; fixes the pre-existing bug that `power_map(..., 'sf')` has no shadow fading (`power_map.m:244-247`, `get_los_channels.m:265` run only `init_sos`) |
| `qd_simulation_parameters` | the legacy switches of section 4, a "legacy" preset; preset mapping of `autocorrelation_function` |

Integration notes:
- Positions: NG stores columns `[3, N]`, the library takes rows. Use one conversion helper; a missing transpose is
  rejected for N ≠ 3 but silently accepted for N = 3, so NG tests use exactly 3 positions at least once.
- Zero links: MEX `lsf_draw`, `path_loss`, `lsp_generate` reject `zeros(0, 3)` and `o2i_loss` rejects
  `zeros(0, 4)`; `init_builder` creates builders without rx positions. The wrappers return empty outputs for
  n == 0 before calling the MEX.
- `dual_mobility = -1` (NG's "unchecked" sentinel, `qd_builder.m:40`) is read as true: call
  `check_dual_mobility` before every library call.
- `simpar.autocorrelation_function`: map legacy names (`Comb300` ...) with the `qd_sos` regexp; `'Disable'` maps to
  `Comb256` with a scenpar copy that sets `SC_lambda = absTOA_lambda = 0` (used as 0.1 m).
- 3GPP-baseline mode (and `SC_lambda = 0`, `'Disable'`): NG draws ground permittivity and abs TOA with `rand` /
  `randn`, the library columns 10 and 11 (MATLAB) are spatially consistent. Decide per mode; overwriting these
  `rnd` columns before `lsp_generate` reproduces the old draws.
- `get_pl` rule stays: without a track (`pow`/`gain`, `check_los`, `gen_lsf_from_ssf`, `get_los_channels`) no O2I
  loss and no indoor distance; with a track (`get_channels`) the O2I loss of the track `par` and `d2d_in` in legacy
  mode. Changing it would change the `pow`/`gain` relation. In TR mode `gen_lsf_parameters` also calls `path_loss`
  at the builder positions for `sf_sigma`, with the stored `rnd_he`.
- Random hE (`PL_hE_rand = 1`, v19 UMa files): `rnd_he` is spatially consistent, so along a track hE and the loss
  beyond the breakpoint step where it crosses a threshold. To avoid steps within a segment, evaluate `rnd_he` once
  at the segment start and use it for all snapshots.
- Floor IDs: NG has no multi-floor model. Pass none; the bias removal covers all positions of the builder. Without
  floor IDs the 2D correlation gives UTs stacked above each other identical LSPs (aerial UTs, D2D at different
  heights, `get_lsp_map` at several heights): document it; the legacy 3D mode is the alternative. Rule for a
  future multi-floor extension: the layout stores per UT `max(0, round((hUT - 1.5)/3))` for O2I UTs (TR Table
  7.2-1 heights `3(n_fl - 1) + 1.5`), 0 otherwise; the builder passes it as `floor_id` to `lsf_draw` on drop and
  track positions, in 2D mode only (`floor_id` with `horizontal = false` is an error); bias removal over the
  floor-0 positions in 2D mode (when more than 2, else `remove_bias = false`), over all positions in 3D mode; the
  SSF sets use the same IDs. Data path: track `par` fields `floor_id` `[1, n_seg]` and `o2i_d2din`
  `[n_tx or 1, n_seg]` (replacing `o2i_d3din`) in the fixed field list of `@qd_track/get_subtrack.m:72`, `:151`
  and in `copy.m:151`, `:181`; builder property `floor_id` `[1, n_rx]` filled in `init_builder`, carried by `copy`,
  `split_rx`, `split_multi_freq`, `add_paths` and the per-segment builders of `get_channels_seg_unmerged`. The
  `o2i_d2din` field is needed now for the legacy `d2d_in`.
- Performance (R2026a, light probe at load 1.3 to 1.5, provisional): 0.13 to 0.16 ms MEX overhead per call with a
  full scenpar struct; per link `lsf_draw` 1.2 µs, `path_loss` 0.14 µs, `lsp_generate` 0.2 to 0.27 µs;
  `lsf_init_phases` 0.14 ms (1.35 ms with bias removal over 1000 positions); 3 to 4 calls (about 0.5 ms) per
  segment. `get_channels` (`:231`, `:338-339`, `:415`, `:429-447`) re-evaluates `h_builder.pow(i_mobile, ...)`
  for all links inside the per-mobile loop (O(n²), as today): cache `pow` and the builder PL before the loop;
  optionally batch `get_pl` / `get_sf_profile` of all mobiles of a builder.

## 4. Legacy switches

`simpar` gets these switches with a "legacy" preset:

| Switch | TR default | Legacy (today's QuaDRiGa) |
|---|---|---|
| `lsp_generate` `cholesky` | true (TR order) | false (symmetric square root, as `sqrtm`) |
| `lsp_generate` `clip_spreads` | true | false |
| `lsf_init_phases` / `lsf_draw` `horizontal` | true (2D) | false (3D) |
| `path_loss` `d2d_in` | omitted (geometric distance) | indoor 2D distance per link (outdoor distance with the 10 m rule) |
| `lsp_generate` `sf_sigma` | `sf_sigma` of `path_loss` | omitted (SF std from the scenpar keys, today's default mode; today's baseline mode corresponds to the TR default) |

Today the `sig`, `sig1` to `sig3` and `SF` keys act only in the baseline branch of `get_channels` (and in
`get_los_channels`) via `scale_sf`; the library replaces this by the `sf_sigma` input.

## 5. Behaviour changes that the integration makes visible

Switchable (flag of section 4 or argument): 2D correlation, Cholesky, spread clipping, geometric O2I distance, SF
std of the path-loss model (e.g. v19 RMa_LOS and SMa_LOS: 4 or 6 dB by the side of the breakpoint instead of
`SF_sigma = 5`); InF-DH clutter density r = 0.6 (TR Table 7.8-7; NG `set_scenario.m:660-665` uses 0.4, reachable
with `r_clutter = 0.4`); spatially consistent ground permittivity and abs TOA in baseline mode (overwrite the
`rnd` columns).

Not switchable (corrections or new TR content):
- `winner_los`: `C2` beyond the breakpoint and per-link breakpoint heights (the 5 WINNER LOS files change beyond the
  breakpoint and where heights vary within one MATLAB call).
- Bias removal: each Normal field is corrected once over all positions where it is evaluated, the Tx side mirrored
  afterwards; bias-removed ASD/ASA and ESD/ESA change in dual mobility and, for unequal decorrelation distances of a
  pair (e.g. UMa LOS ASD 18 m, ASA 15 m), in single mobility too.
- Dual mobility: ASD/ASA and ESD/ESA are the two-map sums without QuaDRiGa's extra 2x2 square root, so they stay
  Normal(0,1); the values change in dual mobility.
- A std made negative by distance, height or elevation terms is limited at 0 (MATLAB flips the sign; NG satellite
  and 5G-ALLSTAR files at low elevation).
- O2I: only the TR models; deviation N(0,1) (QuaDRiGa: uniform); mmMAGIC, the TR v14 wall and the legacy d2D-in
  distribution stay in MATLAB only if NG still needs them.
- Legacy O2I distance per link (MATLAB: once per segment start); `sf_sigma` per link (MATLAB: averaged over the
  segment). Reachable by the caller, not needed for the parity test: `d2d_in(l) = d2D(l)·d3in_seg/d3D(l)`, and
  average `sf_sigma` over the segment snapshots.
- O2I links of `nlos` files: MATLAB returns `max(PLn, PL_LOS + o2i) + o2i`, the library `max(PLn, PL_LOS)` plus
  the O2I loss once by the caller.
- `lsp_vals` has no counterpart; `add_sdc` and the XPR sigma of `gen_ssf_parameters` change for files with
  distance, height or elevation terms.
- v19 conf files: path-loss constants recomputed from the TR and rounded to 4 significant digits (up to 0.014 dB
  against the old files); UMa random hE (`PL_hE_rand = 1`); SMa NLOS without the LOS maximum (`PL_LOSmax = 0`);
  InF-DL with the InF-SL term (`PL_Ax/Bx/Cx`); InF delay spread with the V/S term (`DS_VS`); abs TOA bounded by
  `2L/c` in the InF and InH NLOS files (`absTOA_Lmax`); abs TOA also in the LOS_O2I files of UMa, UMi, RMa, SMa
  (an O2I link has no LOS ray); new keys `LOS_state_lambda` (NLOS files), `IO_state_lambda` (O2I files),
  `O2I_d2D_in_max`; dual_slope with the 3D breakpoint distance also for RMa/SMa LOS (< 0.035 dB).

## 6. Layout, groups, parity test, retired code

`set_scenario` keeps a table group → (LOS model name, member files, clutter parameters, O2I models and weights) and
reads `LOS_state_lambda` from the group's NLOS file, `IO_state_lambda` from an O2I file (0 elsewhere = "not stated
here"), `O2I_d2D_in_max` from the O2I files. Groups for the v19 set (file prefix `3GPP_38.901_v19_`):

| Group | Model | Files | O2I |
|---|---|---|---|
| RMa | `38.901_RMa` | RMa_LOS, RMa_NLOS, RMa_LOS_O2I, RMa_NLOS_O2I | `low` only, d2D-in max 10 m |
| UMi | `38.901_UMi` | UMi_LOS, _NLOS, _LOS_O2I, _NLOS_O2I | `low`/`high` (today 50/50), 25 m |
| UMa | `38.901_UMa` | UMa_LOS, _NLOS, _LOS_O2I, _NLOS_O2I | `low`/`high`, 25 m |
| SMa | `38.901_SMa` | SMa_LOS, _NLOS, _LOS_O2I, _NLOS_O2I | TR Table 7.2-5: 90 % residential (d2D-in max 10 m) / 10 % commercial (25 m), one `o2i_loss` call per building type (the Rel-19 calibration used low-loss A, `low_A`); in-car option `car`, `car_metal` |
| Indoor mixed / open office | `38.901_InH_Mixed` / `38.901_InH_Open` | Indoor_LOS, Indoor_NLOS | none |
| InF-SL, -DL, -SH, -DH | `38.901_InF_SL` ... `_DH` (r, d_clutter, h_c defaults of Table 7.8-7) | InF_LOS + InF_NLOS_SL / _DL / _SH / _DH | none (InF uses `high` if O2I is ever needed) |
| InF-HH | `38.901_InF_HH` (p = 1) | InF_LOS | none |

Deprecated 3GPP_3D groups: `3GPP_3D_UMi` maps to `38.901_UMi` (identical); `3GPP_3D_UMa` keeps its TR 36.873
formula in `set_scenario.m:352-381` (`g(d) = 1.25e-6·d²·exp(-d/150)`, C for indoor UTs only), the only p_LOS
formula left in NG; `calib_3GPP_3D_phase1.m`, `calib_3GPP_3D_phase2.m` stay unchanged. The other NG groups map to
the library model of their formula (5G-ALLSTAR: `38.811_*`; mmMAGIC_UMi, QuaDRiGa_UD2D: `38.901_UMi`;
mmMAGIC_Indoor: `38.901_InH_Mixed`; `QuaDRiGa_Industrial`); check each against `set_scenario.m` before mapping. NG draws all O2I variables at the LOS-state distance (50/60/10 m,
`set_scenario.m:217-221`); the TR distances (50/25/10 m) are decided in the `gen_o2i_loss` integration. Order:
`o2i_loss` before `los_probability`.

Parity test (MATLAB script with the pre-integration code; reference values in NG's test data; single mobility,
fixed phases, legacy files):
1. `path_loss` and `lsp_generate` with the legacy switches, fed with MATLAB's SOS values as `rnd` (columns 1..8
   from `val`), reproduce PL, SF and the LSPs to 1e-12 relative. Exceptions: the 5 `winner_los` files only before
   the breakpoint; legacy O2I via `d2d_in = d2D·d3din/d3D` (NG stores only `o2i_d3din`; equal at segment starts incl.
   the 10 m rule and the clamp at 0), compared at segment starts or single-position tracks; O2I links of `nlos`
   files with the O2I loss set to 0 in the reference script; SF without `sf_sigma` against `gen_lsf_parameters`,
   with `sf_sigma` against `10·lg(sf)·scale_sf` only for files without `SF_omega`, `SF_kappa`, `SF_tau`, `SF_beta`;
   links where the std limit at 0 acts are left out.
2. `lsf_draw` against MATLAB's `val` to 1e-5 absolute (AVX2 kernel with single-precision sin/cos; `qd_sos`
   frequencies are not bitwise equal to `sos_preset`).
Phase cube from the old generators: `phase(:, g, 1) = h_builder.sos(g).sos_phase(:, 1)` for g = 1..8,
`absTOA_sos` for column 9 (any phases if absent), `gr_sos` for 10, any phases for 11; slice 2 by mirroring; no bias
removal in the library (MATLAB's phases are already corrected). A read-only probe over all 91 NG conf files gave
2.2e-16 relative for `path_loss` (non-satellite) and 6.6e-14 for the LSPs; satellite files need the `interp` fix.

Retired MATLAB code (target): `gen_o2i_loss.m` except deprecated parts NG keeps; the LSF part of `init_sos.m`; most
of `gen_lsf_parameters.m`, `get_pl.m`, `get_sf_profile.m`, `get_lsp_map.m`; the p_LOS formulas of `set_scenario.m`
except `3GPP_3D_UMa`; the `lsp_vals` property. Unit tests that inspect internals are rewritten:
`testBuilder_parameter_generation.m:36-54`, `testBuilder_add_clusters.m:78`, `:163`,
`testBuilder_gen_lsf_ssf.m:15-53`, `testBuilder_absTOA_model.m:31`. All `xunit_new` tests pass in MATLAB R2024a and
Octave; `calib_3GPP_38901_large_scale.m` and `calib_3GPP_38901_full.m` run with the v19 groups.

## 7. SSF readiness

The library provides everything the later SSF port needs from the LSF step: DS, ASD, ASA, ESD, ESA, KF, SF per
link and frequency (`lsp`); XPR mean and std (`lsp`, `lsp_mu`, `lsp_sigma` slice 8); per-link ZSD mean for the
cluster ESD (`lsp_mu` slice 6); EoD offset; abs TOA; ground permittivity (take the conjugate for the TR reflection
formulas); soft LOS state (the H_LOS/H_NLOS combination needs a LOS and an NLOS builder with independent phase sets,
i.e. two `lsf_init_phases` calls with different seeds); anchor-frequency LSPs of clause 7.6.5.1 (add the anchor to
`fc`). The SSF phase sets (`path_sos`, `xpr_sos`, `pin_sos`, `clst_dl_sos` of `init_sos.m:168-237`) are SSF work:
the generic machinery (generator table, evaluation with floor offsets, bias removal, one call per decorrelation
distance) is file-local in `src/qd_gbsm_phases.cpp`, whose header states the design and the seed rule (SSF sets
take columns 12.. of one `sos_init_phases` call with the builder seed, or a derived seed), hence the stored seed.

## 8. Calibration step

After the parity test, calibrate the TR mode against the 3GPP campaigns; details, download paths, reference numbers
and the test set are in `md/3gpp_calibration_references.md`. In short: no campaign is open; the last ones (Rel-19
7–24 GHz, R1-2506406, RAN1#122; ISAC, R1-2509126, RAN1#123) used v19.0.0, the corrections up to v19.5.0 are
uncalibrated. NG's `tutorials/calib_3GPP_ref_data.m` holds only 2016 data (38.900 v14.0.0, outdated for UMa and
InH) and no InF, Rel-19 or SMa data. Test against the company min–max band (and within 1 dB / 10 % of the median at
p50) with at least 20,000 UEs per case: coupling loss and geometry test path loss, LOS probability, O2I and SF; the
full-calibration spread CDFs test the LSPs approximately; the ZOD offset and CR 0043 only by unit tests against the
TR tables.

## 9. Known issues and constraints

- The MEX test `test_ray_mesh_interact` fails since before this port (unrelated).
- MATLAB R2026a aborted at exit with a glibc heap error in 3 of 5 runs under heavy load (load 18–24); not
  reproduced at low load and not tied to this port.
- MATLAB R2024a is required (libstdc++ up to `GLIBCXX_3.4.28`: check new MEX files with
  `objdump -T +quadriga_lib/*.mexa64 | grep -o 'GLIBCXX_3\.4\.[0-9]*' | sort -u`); Octave as well.
- NG unit tests are in `xunit_new/` (`run_tests.m`).
- The machine is shared: check `cat /proc/loadavg`, at most 12 threads, one test suite at a time, heavy tools once
  on the final code.
