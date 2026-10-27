# QuaDRiGa-NG integration readiness review, final round (r3)

Reviewer: independent, read-only. Library: branch `fixes` (b10f332), `include/quadriga_gbsm.hpp`, the MEX wrappers
`lsf_init_phases`, `lsf_draw`, `path_loss`, `lsp_generate`, `los_probability`, `o2i_loss`, `conf_file_*`,
`scenpar_validate`, their docs and tests, spec sections 8 and 10. NG: `/home/stephan/sjc/QuaDRiGa-NG/quadriga_src`
(6253027), all callers named in the brief, the tutorials and `xunit_new`. Nothing was built and nothing was written
into QuaDRiGa-NG; the probes ran from the scratchpad with the MEX files already in `+quadriga_lib`.

## Verdict

No blocker in the library. The six functions cover every NG LSF call site, with the data available where NG needs
it. Every legacy behaviour that the parity test needs can be reached by an argument. The SSF port needs nothing more
from the LSF step. One integration issue is major: the library update itself (10.1) breaks 29 NG unit tests
through a changed `quadriga_lib.interp` call form. The cause is a bug fix already in `main` (33fc745), not this port,
and the fix is three one-line edits in NG that the hand-over does not mention yet. The minor findings are NG design
points that spec 10 leaves open (where floor IDs are stored, the bias-removal rule with floors, the list of call
sites, the `get_pl` O2I rule, the 36.873 groups). None of them needs a library change.

Counts: blocker 0, major 1, minor 7, nit 12.

## Findings

| ID | Severity | Location (library / NG) | Finding | Proposed fix / integration note |
|---|---|---|---|---|
| N3-01 | major | NG `@qd_builder/get_pl.m:469`, `@qd_track/interpolate.m:241`, `:266`; library `api_mex/interp.cpp` (commit 33fc745, in `main`) | NG calls `quadriga_lib.interp( x, 0, data, xi )` (1-D data with `y = 0`). The new library rejects this: `Input 'yI' is required when 'y' is given (2D case).` In an Octave run of 129 NG unit tests with the new library first on the path, 30 failed. 29 of them failed only on this error (every test that interpolates a track: GetChannels*, InitBuilder, SetScenario, Track_*, …). In a scratchpad copy of `quadriga_src` with the three calls changed to `[]`, 28 of those 29 pass. The 29th (`testLayout_Closed_Chan_Generation`) only fails because its conf file is looked up relative to the current folder. The remaining failure of the 30 (`testBuilder_parameter_generation`) depends on test order and passes alone with both the old and the new library. The satellite branch of `get_pl` fails the same way (all 16 satellite files in the path-loss probe) | No library change: the rejection is a deliberate fix. Add to hand-over 10.1: "replace `interp( x, 0, …)` by `interp( x, [], …)` in `get_pl.m` (moot once `get_pl` wraps `path_loss`) and `qd_track/interpolate.m` (2 calls)" |
| N3-02 | minor | `src/mex_helper_functions.hpp:264-298` (an empty array becomes 0x0); spec `md/gbsm_lsf_spec.md:1719-1722` (accepted); NG `@qd_layout/init_builder.m:137-166` | MEX `lsf_draw`, `path_loss`, `lsp_generate` reject `zeros(0,3)` positions and `o2i_loss` rejects `zeros(0,4)` (probe). `lsf_init_phases` and `los_probability` accept empty input. NG creates builders without rx positions (one per scenario × Tx in `init_builder`). Today `gen_parameters`/`gen_lsf_parameters` skip them (`no_rx_positions == 0`), and `get.pow` returns `[]` | Accepted restriction. The new `get_pl`, `get_sf_profile`, `get_lsp_map`, `gen_lsf_parameters` wrappers and the layout O2I call return empty outputs for `n == 0` before calling the MEX |
| N3-03 | minor | all position inputs; NG `rx_positions` `[3, N]` (`qd_builder.m:58-61`) | NG stores positions as columns, the library takes rows. A missing transpose is rejected for N ≠ 3 but silently accepted for N = 3 (probe: `lsf_draw` values differ by up to 2.2). A Tx given as `[3,1]` is rejected (good) | One conversion helper in NG for all calls. NG tests use exactly 3 positions at least once |
| N3-04 | minor | NG `@qd_track/get_subtrack.m:72`, `:151` (fixed field list `o2i_loss`, `o2i_d3din`); `copy.m:151`, `:181`; `split_rx.m:86-117`; `split_multi_freq.m:108-148`; `add_paths.m:166-185`; `@qd_layout/private/get_channels_seg_unmerged.m:38-50`; spec 10.2/10.3 | Floor IDs (and the legacy `d2d_in`) have no data path in NG. Spec 10.3 says "the layout stores it" but not where. Track `par` copies only the two hard-coded fields into subtracks. The builder needs one value per rx position that survives `copy`, `split_rx`, `split_multi_freq`, `add_paths` (which builds positions from several builders) and the per-segment builders of `get_channels_seg_unmerged`. `round((hUT - 1.5)/3)` is negative for an indoor UT below z = 0, which the MEX rejects (`Entries in 'floor_id' must be non-negative integers…`, probe) | Proposal for 10.3: track `par` fields `floor_id` `[1, n_seg]` and `o2i_d2din` `[n_tx or 1, n_seg]` (replacing `o2i_d3din`), added to `get_subtrack`; a builder property `floor_id` `[1, n_rx]` filled in `init_builder` and carried by the functions listed; `max(0, round((hUT - 1.5)/3))` for O2I UTs, 0 otherwise |
| N3-05 | minor [escalate] | NG `init_sos.m:126-130`; library `lsf_init_phases` documentation (all positions are treated as floor 0) | NG removes the bias over all `rx_positions` of a builder. An O2I builder of UMa/UMi holds UTs on up to 8 floors (calibration tutorials: `calib_3GPP_38901_large_scale.m:81-85`). The library treats all positions as one floor-0 field, so the floor-0 field is not mean-free over its own UTs and the other floors keep a random mean. Spec 3.1 offers the options, but 10.2 does not fix the NG rule | Decision for the integration: pass the positions with `floor_id == 0` (when more than 2) or `remove_bias = false` for builders with several floors. Recommendation: floor-0 positions (outdoor and first-floor UTs, the majority) |
| N3-06 | minor | spec 10.2 list (`md/gbsm_lsf_spec.md:2046-2048`); NG `@qd_layout/power_map.m:185`, `:191`; `get_channels_seg_unmerged.m:38`; `add_sdc.m:424`, `:431-447`, `:580-590`; `qd_plan_mimo.m:114`, `:248`; `gen_lsf_from_ssf.m:63`; `qd_builder.m:83`, `:415-424`; tests | The hand-over names only `copy`, `split_rx`, `split_multi_freq`, `add_paths`. Further sites store, copy or clear the LSF state or read `lsp_vals`: `power_map` (copies `.sos` of a template builder), `get_channels_seg_unmerged`, `add_sdc` (`h_bld.sos = []`, four `lsp_vals` means, XPR mu/sigma), `qd_plan_mimo`, `gen_lsf_from_ssf`, `loadobj` (saved builders hold `qd_sos` objects and need a migration to `lsf_phase`, as in 10.4), the public `lsp_vals` property. Unit tests that inspect internals need rewriting: `testBuilder_parameter_generation.m:36-54`, `testBuilder_add_clusters.m:78`, `:163`, `testBuilder_gen_lsf_ssf.m:15-53`, `testBuilder_absTOA_model.m:31` | Complete the list in 10.2/10.5. The `loadobj` migration builds the phase cube from `sos(g).sos_phase(:,1)` as described in 10.4 |
| N3-07 | minor | NG `get_pl.m:73-84` vs `:85-122`; `qd_builder.m:224`, `:361`; `check_los.m:149`; `gen_lsf_from_ssf.m:116`; `get_los_channels.m:264` | `get_pl()` without a track ignores the O2I loss and the indoor distance. With a track it applies both. `pow`↔`gain`, `check_los`, `gen_lsf_from_ssf` and `get_los_channels` use the first form, `get_channels` the second. The wrapper must keep this rule or change it deliberately, because that changes the `pow`/`gain` relation. In TR mode `gen_lsf_parameters` also needs `path_loss` at the builder positions (`sf_sigma`), with `rnd_he`, which `get.pow` needs on every access too | Keep the rule: the O2I loss is added for track evaluations only; `d2d_in` (legacy) as today. The builder stores column 11 of `lsf_draw` (or all of `rnd`) per rx position, so that `get_pl()` needs no new draw |
| N3-08 | minor [escalate] | NG `@qd_layout/set_scenario.m:352-381`; tutorials `calib_3GPP_3D_phase1.m:97-99`, `calib_3GPP_3D_phase2.m:103-106`; spec 10.3, 10.5 | The 36.873 group `3GPP_3D_UMa` has no library model. Its g(d) is `1.25e-6·d²·exp(-d/150)` (38.901 uses d³), and NG applies C to indoor UTs only. `3GPP_3D_UMi` equals `38.901_UMi` exactly. Two tutorials use the groups, while 10.5 retires "the p_LOS formulas in set_scenario.m" | Decide: keep the 36.873 UMa formula in NG (the only p_LOS formula left there), map it to `38.901_UMa` (tutorial values change), or drop the 3GPP_3D groups and tutorials |
| N3-09 | nit | MEX `qd_mex_get_scalar<bool>` (`api_mex/lsf_draw.cpp:252` and others); NG `qd_builder.m:40` | NG's "unchecked" sentinel `dual_mobility = -1` is read as `true` by the MEX (probe) | Call `check_dual_mobility` before every library call (as today in most paths) |
| N3-10 | nit | NG `@qd_sos/qd_sos.m:114-118`, `init_sos.m:95-99` | `simpar.autocorrelation_function` accepts legacy names (`Comb300` …) and `'Disable'`; `lsf_init_phases` rejects both (probe) | Wrapper maps them with the `qd_sos` regexp. `'Disable'` → `Comb256` plus a scenpar copy with `SC_lambda = absTOA_lambda = 0` (lambda 0.1 m) for the generators that NG drew without spatial consistency |
| N3-11 | nit | library `lsf_draw` (decorrelation distances from `scenpar` at each call); NG `init_sos.m:113-124` (lambdas frozen in `qd_sos`) | Changing scenpar lambdas after `init_sos` now changes the fields and leaves the bias removal stale. A preset change with the same `n_coeff` is not detected | Store `preset` and `horizontal` with `lsf_phase`. Document that lambdas are read at draw time (or store them too) |
| N3-12 | nit | spec `md/gbsm_lsf_spec.md:2088-2092`; NG `get_pl.m:487-489` | Two items listed as "not switchable" can be reproduced by the caller. (a) Per-segment SF std: average the `sf_sigma` of `path_loss` over the segment snapshots before passing it to `lsp_generate`. (b) Per-segment indoor 3D distance: pass `d2d_in(l) = d2D(l)·d3in_seg/d3D(l)` | Reword 10.2: "reachable by the caller, not needed for the parity test" |
| N3-13 | nit | NG `gen_o2i_loss.m:317-318`; spec 10.4 | The pre-integration code stores only `o2i_d3din`, not `d_2d_in`. The reference script can use `d2d_in = d2D·d3din/d3D`. With it the library's legacy distance equals MATLAB's at segment starts, including the 10 m rule and the clamp at 0 (then `d2d_in = 0`, an outdoor link) | Add the rule to 10.4 |
| N3-14 | nit | NG `set_scenario.m:660-665` | NG's InF-DH uses `r = 0.4`; the library default is 0.6 (TR Table 7.8-7) | List it in the 10.2 behaviour changes; the legacy value is reachable with `r_clutter = 0.4` |
| N3-15 | nit | NG `gen_lsf_parameters.m:314`, `:355`; `init_sos.m:147`, `:163` | In 3GPP-baseline mode (and with `SC_lambda = 0` or `'Disable'`), NG draws the ground permittivity and the abs TOA with `rand`/`randn`. Columns 9/10 of `lsf_draw` are spatially consistent | Decide per mode. Overwriting the `rnd` columns before `lsp_generate` reproduces the old draws |
| N3-16 | nit (perf) | NG `get_channels.m:231`, `:338-339`, `:415`, `:429-447` | Measured (R2026a, load 1.3-1.5, light probe, provisional): about 0.13-0.16 ms per MEX call with a full 172-field scenpar. Per link: `lsf_draw` 1.2 µs, `path_loss` 0.14 µs, `lsp_generate` 0.2-0.27 µs. `lsf_init_phases` 0.14 ms (1.35 ms with bias removal over 1000 positions). Per segment this is 3-4 calls (≈0.5 ms), acceptable. `h_builder.pow(i_mobile, …)` re-evaluates `get_pl` for all links on every access inside the per-mobile loop: O(n²), as today | Cache `pow` (and the builder-level PL) before the mobile loop. Optionally batch `get_pl`/`get_sf_profile` of all mobiles of a builder into one call |
| N3-17 | nit | NG `power_map.m:244-247`, `get_los_channels.m:265` | Pre-existing NG bug: `power_map(…,'sf')` only runs `init_sos`, `h_builder.sf` stays empty, and the map has no shadow fading | Fix in the integration: `lsf_draw` + `lsp_generate` (+ `path_loss` for `sf_sigma`) on the grid |
| N3-18 | nit | NG `get_lsp_map.m:94` | `get_lsp_map` passes `h_builder.tx_position`, which is `[3, N]` after `check_dual_mobility` for N > 1 and then takes the dual-mobility branch (pre-existing) | Wrapper passes `tx_position(:,1)` |
| N3-19 | nit | library default `horizontal = true` | Without floor IDs, the 2D correlation gives stacked UTs identical LSPs: aerial UTs, D2D at different heights, `get_lsp_map` at several `zc` | NG documentation; the legacy 3D flag exists |
| N3-20 | nit | NG `set_scenario.m:217-221`, `gen_o2i_loss.m` (`SC_lambda`) | NG draws all O2I variables at the LOS-state distance (50/60/10 m). The library documents 50 m (building type), 25 m (d2D-in) and 10 m (sigma_P), TR 7.6.3.3 | Decide in the integration of `gen_o2i_loss` |

## Mapping: NG function → library calls → status

| NG function | Library calls (MATLAB form, positions transposed to rows) | Status |
|---|---|---|
| NG with the new library (10.1) | — | blocked until N3-01 (3 one-line NG edits) |
| `init_sos` LSF part, `absTOA_sos`, `gr_sos` | `lsf_init_phases( scenpar, seed, tx_position.', rx_bias.', dual_mobility, remove_bias, preset, horizontal )`; seed drawn once from the MATLAB RNG (`floor(rand·2^52)`, as `qd_sos.init`) | ready with note (N3-05, N3-10, N3-11) |
| `gen_lsf_parameters` (usage 1/2) | `rnd = lsf_draw( scenpar, phase, tx.', rx.', floor_id, dual, preset, horizontal )`; TR mode: `[pl, sfs] = path_loss( plpar, scenpar, fc, tx.', rx.', d2d_in_or_[], rnd(:,11) )`; `[lsp, abs_toa, eps_re, eps_im, eod, mu, sg] = lsp_generate( scenpar, fc, tx.', rx.', rnd, sfs_or_[], cholesky, clip )`; `ds = lsp(:,:,1).'` …, `gr_epsilon_r = complex(eps_re, eps_im).'`, `absTOA_offset = abs_toa.'` (only if `absTOA_mu > -30`, as today); fill only the empty properties, as today | ready |
| `gen_lsf_parameters` usage 3 / `get_sf_profile` | the same on the track positions (one call per track; `floor_id` of the UT repeated); TR mode `sf_sigma` from `path_loss` per snapshot (legacy: averaged by NG, N3-12) | ready with note (N3-16) |
| `get_lsp_map` | `lsf_draw` (no `floor_id`) + `lsp_generate` on the grid; single precision accepted | ready with note (N3-18, N3-19) |
| `qd_builder.lsp_vals` | `lsp_generate( …, zeros(n,11) )` outputs 6/7 (`lsp_mu`, `lsp_sigma`), per link; mean over links or frequencies where NG needs one value | ready with note (spec 8: not 1:1) |
| `get_pl` | `path_loss( plpar_or_alt, scenpar, fc, tx.', rx.', d2d_in, rnd_he )` + O2I loss from track `par` added once by NG; `[pl; pl_init]` by appending the initial position as an extra row | ready with note (N3-07) |
| `get_pl` `scale_sf` (`get_channels` baseline, `get_los_channels`) | dropped; SF from `lsp_generate` with `sf_sigma` | ready |
| `get_channels` (SF/KF profile, path gain) | wrappers above per mobile (or batched) | ready with note (N3-16) |
| `get_los_channels`, `check_los`, `gen_lsf_from_ssf` | `path_loss` | ready |
| `add_sdc` | `path_loss` on synthetic positions (`:524`); `lsp_mu`/`lsp_sigma` means for `:431-447`, `:580-590` | ready with note (N3-06) |
| `gen_ssf_parameters`, `gen_fbs_lbs` (LSF inputs) | builder `ds … xpr`, `gr_epsilon_r`, `absTOA_offset`; XPR mu/sigma from `lsp_mu`/`lsp_sigma` slice 8 | ready |
| `add_paths` | `lsf_init_phases` on the assembled positions (dual mobility when the Tx differ), floor IDs concatenated | ready with note (N3-04) |
| `copy`, `split_rx`, `split_multi_freq` | copy `lsf_phase`, seed, `preset`/`horizontal`, `floor_id`, `rnd_he`; per-frequency builders call `lsp_generate` with their one `fc` (same `x`, equal to the slice of a multi-frequency call) | ready with note (N3-04) |
| `qd_layout.set_scenario` | `los_probability( model, max(d2D - d2d_in, 0), h_bs, h_ut, d_clutter, r_clutter, h_clutter )`; 38.811 "Null" below the horizon stays in NG; LOS draw and renormalisation stay in NG | ready with note; `3GPP_3D_UMa` blocked by decision (N3-08) |
| `qd_layout.gen_o2i_loss` | `[loss, d2d_in] = o2i_loss( fc, rnd, {'low','high'}, [f, 1-f], O2I_d2D_in_max )` (RMa `{'low'}`, 1); mmMAGIC stays in MATLAB | ready with note (N3-04, N3-20) |
| `qd_layout.power_map` | `lsf_init_phases`/`lsf_draw`/`path_loss`/`lsp_generate` on the grid; template builder copies `lsf_phase` | ready with note (N3-06, N3-17) |
| `qd_layout.get_channels`, `get_channels_seg_unmerged` | as the builder | ready with note (N3-06) |
| `qd_track` `par.o2i_loss`, `o2i_d3din`, segments | new fields `o2i_d2din`, `floor_id` | ready with note (N3-04) |
| `qd_simulation_parameters` | legacy switches (`horizontal`, `cholesky`, `clip_spreads`, `d2d_in` mode, `sf_sigma` mode), preset mapping | ready with note (N3-10) |
| tutorials `calib_3GPP_38901_*`, unit tests | v19 groups; tests of internals rewritten | ready with note (N3-06); `calib_3GPP_3D_*` see N3-08 |

## State the builder must store

`lsf_phase` `[n_coeff, 11, 2]` (replaces `sos`, `absTOA_sos`, `gr_sos`); the explicit seed, for the SSF sets of spec 8; the
`preset` and `horizontal` flag used at init; `floor_id` per rx position; column 11 of `rnd` (or all of `rnd`) per rx
position, for `get_pl()` in TR mode; `d2d_in` per rx position in legacy mode (from the track `par`);
`dual_mobility` (already there, never -1 at call time). `lsp_mu`/`lsp_sigma` are either stored or recomputed with
`rnd = zeros`.

## Legacy reachability (parity test 10.4)

All parity-relevant legacy behaviours can be reached: 3D correlation (`horizontal = false`), `sqrtm` (`cholesky =
false`), no clipping, outdoor distance (`d2d_in`), scenpar SF std (omit `sf_sigma`), and no bias removal (MATLAB phases
supplied). Probe against NG's own MATLAB code over all 91 NG conf files (single mobility, constant heights, 2 frequencies,
Octave): the `path_loss` of every non-satellite file equals `get_pl` to 2.2e-16 relative. The 8 LSPs of `lsp_generate`
(legacy switches, same `x`) equal NG's formula to 6.6e-14 relative, excluding links with a negative std. The satellite files
could not be compared because NG's `get_pl` fails with the new `interp` (N3-01). Not reachable by design: QuaDRiGa's
dual-mobility 2x2 coupling, `winner_los` with `C1` beyond the breakpoint, the uniform O2I deviation, negative std,
MATLAB's bias-removal order (all listed in 10.2). Smoke test: lsf_init_phases → lsf_draw → path_loss (both modes, with
`d2d_in` and `rnd_he`) → lsp_generate (both roots) ran on all 91 NG files at terrestrial and 600 km Tx heights without
an error or a non-finite LSP.

## SSF readiness (spec 8)

What the SSF code takes from the LSF step, and whether the library provides it:
- `gen_ssf_parameters.m:243-248`: DS, ASD, ASA, ESD, ESA, KF per link and frequency. Provided by `lsp` slices 1, 4-7, 2.
- `gen_ssf_parameters.m:112`: `gr_epsilon_r`. Provided by `gr_eps_re/im` (conjugate for the TR formula, documented).
- `gen_ssf_parameters.m:813-818`: `absTOA_offset`. Provided by `abs_toa`.
- `gen_ssf_parameters.m:1048-1049`, `:1065-1066`: XPR mu and sigma. Provided by `lsp`/`lsp_mu`/`lsp_sigma` slice 8.
- `gen_ssf_parameters.m:214`: dual mobility. Provided by the argument and the slice convention.
- `get_channels.m:429-480`: KF and SF along tracks and the path gain. Provided by the `get_sf_profile`/`get_pl` wrappers.
- `get_channels.m:332`: `gr_epsilon_r(i_mobile)`, single frequency. Provided.
- `gen_fbs_lbs.m`: uses only scenpar and SSF data.

The SSF phase sets (`path_sos`, `xpr_sos`, `pin_sos`, `clst_dl_sos`, `init_sos.m:168-237`) are SSF work. The generic
machinery exists in `src/qd_gbsm_phases.cpp:47-300` (`qd_gbsm_gen`, `qd_gbsm_freq`, `qd_gbsm_eval` with floor
offsets, `qd_gbsm_remove_bias`). The seed rule of spec 8 needs the stored builder seed (see above). Spec 8's claim of
no API gap holds.

## Probes (scratchpad, Octave 8 / MATLAB R2026a, read-only for NG)

- MEX shapes and types: zero links, 3x3 orientation, `dual_mobility = -1`, presets, `floor_id` -1, a seed of 2^52, outputs
  `[n,1,8]` for one frequency.
- All 91 NG conf files through the six library functions; legacy parity of `path_loss` and LSPs against NG code.
- Timing (R2026a, load 1.35 → 1.51 before/after; light probe, not a benchmark, numbers provisional).
- 129 NG unit tests in Octave with the new library: 99 pass, 30 fail. 29 of the failures are N3-01 (28 pass after the
  `interp` patch in a copy, the 29th fails only on a conf file it looks up in the current folder). The 30th is order-dependent and passes alone with both
  libraries. 12 file-writing tests were excluded. Load during the run: 1.4 → about 13-24 (other sessions); timing was not
  measured there.
